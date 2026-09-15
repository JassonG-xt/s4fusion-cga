"""Compare trained CGA vs its same-regime baseline on M3FD detection, focusing on
high-conflict objects. Both variants freshly detected under results/arb/{v}/runs."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import stats
import diag_conflict_detection as D

ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
TEST, BASE = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("CGA", "B0cmp")


def preds(v, sid):
    return D.load_yolo(ARB / v / "runs/det/labels" / f"{sid}.txt", True)


ids = sorted(f.stem for f in (D.OLD / "datasets/m3fd_baseline/test/labels").glob("*.txt"))
cf, base, test = [], [], []
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
    pb, pt = preds(BASE, sid), preds(TEST, sid)
    for g in gt:
        x0, y0, x1, y1 = D.to_xyxy(g, W, H)
        xi0, yi0, xi1, yi1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        if xi1 - xi0 < 2 or yi1 - yi0 < 2:
            continue
        gx = (x0, y0, x1, y1)
        cf.append(float(conflict[yi0:yi1, xi0:xi1].mean()))
        base.append(D.detected(gx, g[0], pb, W, H))
        test.append(D.detected(gx, g[0], pt, W, H))

cf, base, test = np.array(cf), np.array(base), np.array(test)
hi = cf > np.percentile(cf, 66)
print(f"objects {len(cf)}  high-conflict {hi.sum()}   ({TEST} vs {BASE})")
print(f"recall ALL:            base={base.mean():.3f}  {TEST}={test.mean():.3f}  ({test.mean()-base.mean():+.3f})")
print(f"recall HIGH-conflict:  base={base[hi].mean():.3f}  {TEST}={test[hi].mean():.3f}  ({test[hi].mean()-base[hi].mean():+.3f})")
b, a = base[hi], test[hi]
n01 = int(((b == 0) & (a == 1)).sum()); n10 = int(((b == 1) & (a == 0)).sum())
p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
print(f"[DECISIVE] high-conflict McNemar: recovered={n01} lost={n10} p={p:.2e}")
if test[hi].mean() > base[hi].mean() and n01 > n10 and p < 0.1 and (test.mean() >= base.mean() - 0.005):
    print("=> LEARNED CGA WORKS: high-conflict detection up without aggregate loss.")
else:
    print("=> CGA gain weak/absent or has aggregate tax; inspect gate strength / add conflict-weighted loss.")
