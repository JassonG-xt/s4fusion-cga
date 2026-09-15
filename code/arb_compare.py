"""Arbitration pre-check step 3: does committing (vs blending) in conflict
regions recover detection on high-conflict objects? Compare baseline-fused vs the
three arbitrated variants (and the best-single oracle) on high-conflict GT."""
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from scipy import stats
import diag_conflict_detection as D

ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
VARIANTS = sys.argv[1:] or ["A1_ircommit", "A2_contrast", "A3_max"]


def preds_for(variant, sid):
    return D.load_yolo(ARB / variant / "runs/det/labels" / f"{sid}.txt", True)


def main():
    ids = sorted(f.stem for f in (D.OLD / "datasets/m3fd_baseline/test/labels").glob("*.txt"))
    rec = {k: [] for k in ["conflict", "base", "best_single", *VARIANTS]}
    for sid in ids:
        ir_p, vi_p = D.SRC / "Ir" / f"{sid}.png", D.SRC / "Vis" / f"{sid}.png"
        if not (ir_p.is_file() and vi_p.is_file()):
            continue
        ir = np.asarray(Image.open(ir_p).convert("L"), float) / 255
        vi = np.asarray(Image.open(vi_p).convert("RGB").convert("YCbCr").getchannel("Y"), float) / 255
        H, W = ir.shape
        gxi, gyi, gi = D.grad(ir); gxv, gyv, gv = D.grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2
        gt = D.load_yolo(D.OLD / "datasets/m3fd_baseline/test/labels" / f"{sid}.txt", False)
        pb = D.load_yolo(D.OLD / "runs/baseline/labels" / f"{sid}.txt", True)
        pi = D.load_yolo(D.OLD / "runs/ir/labels" / f"{sid}.txt", True)
        pv = D.load_yolo(D.OLD / "runs/vi/labels" / f"{sid}.txt", True)
        pv_arb = {v: preds_for(v, sid) for v in VARIANTS}
        for g in gt:
            x0, y0, x1, y1 = D.to_xyxy(g, W, H)
            xi0, yi0, xi1, yi1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if xi1 - xi0 < 2 or yi1 - yi0 < 2:
                continue
            gx = (x0, y0, x1, y1)
            rec["conflict"].append(float(conflict[yi0:yi1, xi0:xi1].mean()))
            rec["base"].append(D.detected(gx, g[0], pb, W, H))
            rec["best_single"].append(max(D.detected(gx, g[0], pi, W, H), D.detected(gx, g[0], pv, W, H)))
            for v in VARIANTS:
                rec[v].append(D.detected(gx, g[0], pv_arb[v], W, H))

    A = {k: np.array(v) for k, v in rec.items()}
    cf = A["conflict"]
    q2 = np.percentile(cf, 66)
    hi = cf > q2
    print(f"GT objects: {len(cf)}  high-conflict (top tertile): {hi.sum()}")
    print(f"\nrecall on ALL objects:   base={A['base'].mean():.3f}  best_single={A['best_single'].mean():.3f}  " +
          "  ".join(f"{v}={A[v].mean():.3f}" for v in VARIANTS))
    print(f"recall on HIGH-conflict:  base={A['base'][hi].mean():.3f}  best_single={A['best_single'][hi].mean():.3f}  " +
          "  ".join(f"{v}={A[v][hi].mean():.3f}" for v in VARIANTS))
    print("\n[DECISIVE] arbitration vs baseline on high-conflict objects (McNemar exact):")
    best = None
    for v in VARIANTS:
        b, a = A["base"][hi], A[v][hi]
        n01 = int(((b == 0) & (a == 1)).sum())  # arb recovers what base missed
        n10 = int(((b == 1) & (a == 0)).sum())  # arb loses what base had
        p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
        delta = a.mean() - b.mean()
        print(f"  {v:>12}: recall {b.mean():.3f}->{a.mean():.3f} ({delta:+.3f})  recovered={n01} lost={n10}  p={p:.2e}")
        if best is None or delta > best[1]:
            best = (v, delta, n01, n10, p)
    v, delta, n01, n10, p = best
    print(f"\n=> best arbitration: {v} recovers {delta*100:+.1f}pp on high-conflict "
          f"(recovered {n01} vs lost {n10}, p={p:.2e})")
    if delta > 0.02 and n01 > n10 and p < 0.1:
        print("=> CONCEPT VALIDATED: committing in conflict recovers detection a blend loses. "
              "Worth building the learned arbitration mechanism.")
    else:
        print("=> concept NOT clearly validated by heuristic arbitration; the fix may need a "
              "smarter (learned/task-aware) commit criterion, or reconsider.")


if __name__ == "__main__":
    main()
