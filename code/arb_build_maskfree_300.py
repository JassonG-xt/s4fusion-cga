"""GATE-A follow-up (2026-09-18): free commitment rules on CGA's OWN support.

Why: R1's four parameter-free arms were applied through the heuristic mask
(edge-p85 + dilate2), while CGA acts through its own learned map M. One alternative
explanation therefore survived: perhaps a free rule, given CGA's support, matches CGA,
leaving the learned selector with no incremental value. These three arms close it.

Base image: B0cmp_full -- the same 20-epoch same-budget baseline CGA starts from.
Support:   M > 0.5, read from code/results/cga_mask/ (dumped by export_cga_mask.py,
           pixel-aligned with the fused images).

Arms
  P_M_init  CGA with the selector FROZEN AT ITS INITIALIZATION
            F = B0 + gate * M * (K0 - B0),  K0 = s0*ir + (1-s0)*vi,  s0 = sigmoid(2p)
            i.e. the learned selection head reset to its zero-init value. This is the
            sharpest available control: it isolates "what training the selector bought"
            while keeping everything else about CGA identical.
  P_M_grad  hard gradient selection inside M   (F[mask] = where(gi>=gv, ir, vi))
  P_M_max   hard max selection inside M        (F[mask] = max(ir, vi))

Run with Windows python (paths are E:/...). Pure CPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

ROOT = Path("E:/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026")
SRC = ROOT / "dataset/M3FD/full"
MANIFEST = ROOT / "dataset/manifests/m3fd_test.csv"
BASE_ARM = ROOT / "code/results/arb/B0cmp_full/images"
MASK_DIR = ROOT / "code/results/cga_mask"
OUT = ROOT / "code/results/arb"

KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4
ARMS = ["P_M_init", "P_M_grad", "P_M_max"]


def grad(a: np.ndarray):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, required=True,
                    help="trained tanh(gate_scale); printed by export_cga_mask.py")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    args = ap.parse_args()

    with MANIFEST.open(encoding="utf-8-sig") as fh:
        ids = sorted({r["sample_id"] for r in csv.DictReader(fh)})
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

        m = np.asarray(Image.open(MASK_DIR / f"{sid}.png").convert("L"), float) / 255
        if m.shape != fused.shape:
            raise SystemExit(f"mask shape {m.shape} != fused {fused.shape} for {sid}")
        mask = m > args.mask_threshold

        _gxi, _gyi, gi = grad(ir)
        _gxv, _gyv, gv = grad(vi)

        # CGA with the selector at its initialization: s0 = sigmoid(2p)
        p = np.where(gi > gv, 1.0, -1.0)
        s0 = 1.0 / (1.0 + np.exp(-2.0 * p))
        k0 = s0 * ir + (1.0 - s0) * vi
        f_init = fused + args.gate * m * (k0 - fused)

        hard_grad = np.where(gi >= gv, ir, vi)
        hard_max = np.maximum(ir, vi)

        committed = {
            "P_M_init": f_init,
            "P_M_grad": np.where(mask, hard_grad, fused),
            "P_M_max": np.where(mask, hard_max, fused),
        }
        for v, com in committed.items():
            if v not in args.arms:
                continue
            rgb = np.stack([np.clip(com, 0, 1)] * 3, -1)
            Image.fromarray((rgb * 255).clip(0, 255).round().astype(np.uint8)).save(
                OUT / v / "images" / f"{sid}.png")

        rows.append({
            "sample_id": sid,
            "mask_coverage": round(float(mask.mean()), 6),
            "M_mean": round(float(m.mean()), 6),
            "mean_abs_change_init": round(float(np.abs(f_init - fused).mean()), 6),
        })
        if n % 50 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} images", flush=True)

    cov = np.array([r["mask_coverage"] for r in rows])
    chg = np.array([r["mean_abs_change_init"] for r in rows])
    report = OUT / "P_M_baselines_coverage.csv"
    with report.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["sample_id", "mask_coverage", "M_mean",
                                            "mean_abs_change_init"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"built {len(ids)} images x {len(args.arms)} arms: {' '.join(args.arms)}")
    print(f"M>0.5 coverage: mean {cov.mean()*100:.4f}%  min {cov.min()*100:.4f}%  "
          f"max {cov.max()*100:.4f}%")
    print(f"P_M_init mean|F-B0| = {chg.mean():.6f}  (paper reports 0.0125-0.0170 for CGA-B0)")
    print(f"coverage record -> {report}")


if __name__ == "__main__":
    main()
