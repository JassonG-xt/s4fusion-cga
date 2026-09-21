"""R1 (review round 2026-09-18): parameter-free hard-commitment baselines on the
full 300-image M3FD test set.

Why: the content controls showed that the *commitment* moves the numbers while the
conflict trigger does not. The surviving claim is therefore "image-space per-pixel
hard commitment works". That claim is untestable without a PARAMETER-FREE commitment
rule to compare against: if `max(I_ir, I_vi)` matched the detector endpoint, the
learned selection would add nothing.

Design (2 x 2, all parameter-free):
    applied-to   : {whole image, conflict mask only}
    selection    : {max intensity, larger local gradient}

    P_all       = max(ir, vi)                      (whole image)
    P_conf      = B0 with mask regions -> max(ir, vi)
    P_grad      = where(gi >= gv, ir, vi)          (whole image)
    P_gradconf  = B0 with mask regions -> gradient select

P_gradconf is the sharpest of the four: the gradient comparison is exactly CGA's own
zero-initialization rule (s = sigmoid(2p), p = 2*1[gi>gv]-1), made hard. If learning
added nothing beyond the initialization prior, P_gradconf and CGA coincide.

Mask definition: byte-identical to code/arb_build.py (the existing 90-image A3_max
arm), so no new convention is introduced -- edge-p85 + binary_dilation(2).

Run with Windows python (paths are E:/...). Pure CPU, no model forward.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, binary_dilation

ROOT = Path("E:/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026")
SRC = ROOT / "dataset/M3FD/full"
MANIFEST = ROOT / "dataset/manifests/m3fd_test.csv"
BASE_ARM = ROOT / "code/results/arb/B0cmp_full/images"   # 20-epoch same-budget baseline
OUT = ROOT / "code/results/arb"

KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4

ARMS = ["P_all", "P_conf", "P_grad", "P_gradconf"]


def grad(a: np.ndarray):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def load_ids() -> list[str]:
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        return sorted({r["sample_id"] for r in csv.DictReader(fh)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N ids")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    args = ap.parse_args()

    ids = load_ids()
    if args.limit:
        ids = ids[: args.limit]
    for v in args.arms:
        (OUT / v / "images").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n, sid in enumerate(ids, 1):
        ir = np.asarray(Image.open(SRC / "Ir" / f"{sid}.png").convert("L"), float) / 255
        vi = (np.asarray(Image.open(SRC / "Vis" / f"{sid}.png").convert("RGB")
                         .convert("YCbCr"), float)[..., 0]) / 255
        fused = np.asarray(Image.open(BASE_ARM / f"{sid}.png").convert("L"), float) / 255
        h = min(ir.shape[0], vi.shape[0], fused.shape[0])
        w = min(ir.shape[1], vi.shape[1], fused.shape[1])
        ir, vi, fused = ir[:h, :w], vi[:h, :w], fused[:h, :w]

        gxi, gyi, gi = grad(ir)
        gxv, gyv, gv = grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2
        edge = np.maximum(gi, gv) > np.percentile(np.maximum(gi, gv), 70)
        thr = np.percentile(conflict[edge], 85)
        mask = binary_dilation(conflict > thr, iterations=2)

        hard_max = np.maximum(ir, vi)
        hard_grad = np.where(gi >= gv, ir, vi)

        committed = {
            "P_all": hard_max,
            "P_conf": np.where(mask, hard_max, fused),
            "P_grad": hard_grad,
            "P_gradconf": np.where(mask, hard_grad, fused),
        }
        for v, com in committed.items():
            if v not in args.arms:
                continue
            rgb = np.stack([com] * 3, -1)
            Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(
                OUT / v / "images" / f"{sid}.png")

        rows.append({"sample_id": sid, "mask_coverage": round(float(mask.mean()), 6),
                     "conflict_max": float(conflict.max())})
        if n % 50 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} images", flush=True)

    cov = np.array([r["mask_coverage"] for r in rows])
    report = OUT / "P_baselines_coverage.csv"
    with report.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["sample_id", "mask_coverage", "conflict_max"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"built {len(ids)} images x {len(args.arms)} arms: {' '.join(args.arms)}")
    print(f"conflict-mask coverage: mean {cov.mean()*100:.2f}%  min {cov.min()*100:.2f}%  "
          f"max {cov.max()*100:.2f}%")
    print(f"coverage record -> {report}")


if __name__ == "__main__":
    main()
