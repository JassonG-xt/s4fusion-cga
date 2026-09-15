"""Arbitration-concept pre-check (training-free), step 1: build arbitrated images.
In conflict regions, replace the baseline blend with a COMMITTED single-modality
choice (three heuristics). If detection on these recovers high-conflict objects,
the 'commit not average' concept is validated before writing any training code.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import convolve, binary_dilation

OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/results_metrics/downstream_detection")
SRC = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/M3FD/full")
OUT = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
BASE = OLD / "datasets/m3fd_baseline/test/images"
KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4
VARIANTS = ["A1_ircommit", "A2_contrast", "A3_max"]


def grad(a):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def main():
    ids = sorted(f.stem for f in BASE.glob("*.png"))
    for v in VARIANTS:
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
        thr = np.percentile(conflict[edge], 85)
        mask = binary_dilation(conflict > thr, iterations=2)
        cover.append(mask.mean())
        committed = {
            "A1_ircommit": ir,
            "A2_contrast": np.where(gi >= gv, ir, vi),
            "A3_max": np.maximum(ir, vi),
        }
        for v, com in committed.items():
            out = fused.copy(); out[mask] = com[mask]
            rgb = np.stack([out] * 3, -1)
            Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(OUT / v / "images" / f"{sid}.png")
    print(f"built {len(ids)} images x {len(VARIANTS)} variants; mean arbitrated coverage {np.mean(cover)*100:.1f}% of pixels")


if __name__ == "__main__":
    main()
