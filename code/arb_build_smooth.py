"""CGA minimal prototype (training-free, smooth): instead of hard pixel-replace,
softly blend toward the max-response modality with a SMOOTH, conflict-gated weight
w = alpha * smooth(conflict). Tests whether smoothness removes the artifact tax
(aggregate-mAP drop) while keeping the high-conflict recall gain that A3 showed.
If yes, the learned CGA is very likely to work."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/results_metrics/downstream_detection")
SRC = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/M3FD/full")
OUT = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
BASE = OLD / "datasets/m3fd_baseline/test/images"
KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4
ALPHAS = {"S_a06": 0.6, "S_a10": 1.0}


def grad(a):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def main():
    ids = sorted(f.stem for f in BASE.glob("*.png"))
    for v in ALPHAS:
        (OUT / v / "images").mkdir(parents=True, exist_ok=True)
    cover = []
    for sid in ids:
        ir = np.asarray(Image.open(SRC / "Ir" / f"{sid}.png").convert("L"), float) / 255
        vi = np.asarray(Image.open(SRC / "Vis" / f"{sid}.png").convert("RGB").convert("YCbCr").getchannel("Y"), float) / 255
        fused = np.asarray(Image.open(BASE / f"{sid}.png").convert("L"), float) / 255
        h = min(ir.shape[0], vi.shape[0], fused.shape[0]); w = min(ir.shape[1], vi.shape[1], fused.shape[1])
        ir, vi, fused = ir[:h, :w], vi[:h, :w], fused[:h, :w]
        gxi, gyi, gi = grad(ir); gxv, gyv, gv = grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2
        edge = np.maximum(gi, gv) > np.percentile(np.maximum(gi, gv), 70)
        c95 = np.percentile(conflict[edge], 95) + 1e-6
        soft = np.clip(conflict / c95, 0, 1)          # soft conflict, 0..1
        soft = gaussian_filter(soft, sigma=2.0)        # spatial smoothness -> no hard edges
        commit = np.maximum(ir, vi)                    # A3 winner: max-response commit
        cover.append(float(soft.mean()))
        for v, a in ALPHAS.items():
            wgt = a * soft
            out = fused * (1 - wgt) + commit * wgt
            rgb = np.stack([out] * 3, -1)
            Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(OUT / v / "images" / f"{sid}.png")
    print(f"built {len(ids)} imgs x {len(ALPHAS)} smooth variants; mean soft-weight {np.mean(cover)*100:.1f}%")


if __name__ == "__main__":
    main()
