"""Part C: does modal conflict actually hurt DETECTION? Detector-free — reuses
the old M3FD detection predictions (IR/VI/baseline-fused) + GT boxes. For each GT
object we compute its conflict level and whether each input detects it (IoU>=0.5,
same class, conf>=0.25). Decisive test: fusion should beat the best single
modality; if that advantage SHRINKS or REVERSES on high-conflict objects, then
conflict is where fusion fails the task — justifying a conflict-arbitration method.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy import stats

OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/results_metrics/downstream_detection")
SRC = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/M3FD/full")
KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4
CONF = 0.25
IOU = 0.5


def grad(a):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def load_yolo(p, with_conf):
    out = []
    if not p.is_file():
        return out
    for ln in p.read_text().split("\n"):
        s = ln.split()
        if len(s) >= (6 if with_conf else 5):
            c = int(float(s[0])); cx, cy, w, h = map(float, s[1:5])
            conf = float(s[5]) if with_conf else 1.0
            out.append((c, cx, cy, w, h, conf))
    return out


def to_xyxy(b, W, H):
    _, cx, cy, w, h, _ = b
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detected(gt_xyxy, gt_cls, preds, W, H):
    for p in preds:
        if p[0] == gt_cls and p[5] >= CONF and iou(gt_xyxy, to_xyxy(p, W, H)) >= IOU:
            return 1
    return 0


def main():
    ids = sorted(f.stem for f in (OLD / "datasets/m3fd_baseline/test/labels").glob("*.txt"))
    rows = []
    for sid in ids:
        ir_p, vi_p = SRC / "Ir" / f"{sid}.png", SRC / "Vis" / f"{sid}.png"
        if not (ir_p.is_file() and vi_p.is_file()):
            continue
        ir = np.asarray(Image.open(ir_p).convert("L"), float) / 255.0
        vi = np.asarray(Image.open(vi_p).convert("RGB").convert("YCbCr").getchannel("Y"), float) / 255.0
        H, W = ir.shape
        gxi, gyi, gi = grad(ir); gxv, gyv, gv = grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2.0
        gt = load_yolo(OLD / "datasets/m3fd_baseline/test/labels" / f"{sid}.txt", False)
        pb = load_yolo(OLD / "runs/baseline/labels" / f"{sid}.txt", True)
        pi = load_yolo(OLD / "runs/ir/labels" / f"{sid}.txt", True)
        pv = load_yolo(OLD / "runs/vi/labels" / f"{sid}.txt", True)
        for g in gt:
            x0, y0, x1, y1 = to_xyxy(g, W, H)
            xi0, yi0, xi1, yi1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if xi1 - xi0 < 2 or yi1 - yi0 < 2:
                continue
            cf = float(conflict[yi0:yi1, xi0:xi1].mean())
            area = ((x1 - x0) * (y1 - y0)) / (W * H)
            gx = (x0, y0, x1, y1)
            rows.append(dict(cls=g[0], conflict=cf, area=area,
                             det_fused=detected(gx, g[0], pb, W, H),
                             det_ir=detected(gx, g[0], pi, W, H),
                             det_vi=detected(gx, g[0], pv, W, H)))

    n = len(rows)
    conflict = np.array([r["conflict"] for r in rows])
    area = np.array([r["area"] for r in rows])
    df = np.array([r["det_fused"] for r in rows])
    di = np.array([r["det_ir"] for r in rows])
    dv = np.array([r["det_vi"] for r in rows])
    best_single = np.maximum(di, dv)
    fusion_gain = df - best_single  # +1 fusion saves it, -1 fusion loses what a modality had

    print(f"GT objects: {n} over {len(set(r['cls'] for r in rows))} classes; images: {len(ids)}")
    print(f"overall recall  fused={df.mean():.3f}  ir={di.mean():.3f}  vi={dv.mean():.3f}  best-single={best_single.mean():.3f}")
    print(f"fusion vs best-single (overall): {fusion_gain.mean():+.3f}\n")

    # tertiles by conflict
    q1, q2 = np.percentile(conflict, [33, 66])
    bins = [("low", conflict <= q1), ("mid", (conflict > q1) & (conflict <= q2)), ("high", conflict > q2)]
    print(f"{'conflict':>8} {'n':>4} {'area':>6} {'rec_fused':>9} {'rec_best1':>9} {'fus_gain':>8}")
    gains = {}
    for name, m in bins:
        gains[name] = fusion_gain[m]
        print(f"{name:>8} {m.sum():>4} {area[m].mean():>6.4f} {df[m].mean():>9.3f} {best_single[m].mean():>9.3f} {fusion_gain[m].mean():>+8.3f}")

    # decisive: is fusion_gain lower in high-conflict than low-conflict?
    p = stats.mannwhitneyu(gains["high"], gains["low"], alternative="less").pvalue
    print(f"\n[DECISIVE] fusion advantage high-conflict ({gains['high'].mean():+.3f}) vs low-conflict "
          f"({gains['low'].mean():+.3f}), Mann-Whitney p(high<low)={p:.2e}")
    # recall drop with conflict, size-controlled: compare small-object recall across conflict
    small = area < np.median(area)
    print(f"\n[size control] among SMALL objects: rec_fused low-conflict={df[small & (conflict<=q1)].mean():.3f} "
          f"vs high-conflict={df[small & (conflict>q2)].mean():.3f}")
    print(f"[correlation] point-biserial det_fused vs conflict: r={np.corrcoef(df, conflict)[0,1]:+.3f}  "
          f"(area vs conflict r={np.corrcoef(area, conflict)[0,1]:+.3f})")

    if gains["high"].mean() < gains["low"].mean() and p < 0.05:
        print("\n=> CONFLICT HURTS THE TASK: fusion's benefit over the best single modality "
              "significantly erodes on high-conflict objects. Problem is real AND task-relevant.")
    else:
        print("\n=> NO task-level conflict penalty detected: fusion advantage does not erode with conflict. "
              "Structure-retention drop is likely cosmetic; reconsider before building.")


if __name__ == "__main__":
    main()
