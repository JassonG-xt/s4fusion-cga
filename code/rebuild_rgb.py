"""A3: rebuild RGB fusion images from grayscale fused Y + original VI chroma.

Reproduces evaluate_brss.py `_save_outputs` semantics (fused Y + original VI
Cb/Cr merged as YCbCr -> RGB, PIL full-range BT.601) but writes files under
the SAME stem name (no `_rgb` suffix) so eval_metrics_extended.py can pair
them without the suffix-mismatch pitfall. One cell per invocation.

Usage (per cell, from code/):
  .venv-brss/bin/python rebuild_rgb.py \
      --vi-dir baselines/inputs/m3fd/vi --fused-dir results/arb/<cell>/images \
      --out-dir results/arb/<cell>/images_rgb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vi-dir", required=True)
    ap.add_argument("--fused-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    vi_dir, fu_dir, out_dir = Path(args.vi_dir), Path(args.fused_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for fu_path in sorted(fu_dir.glob("*.png")):
        vi_path = vi_dir / fu_path.name
        if not vi_path.exists():
            continue
        vi_rgb = Image.open(vi_path).convert("RGB")
        vi_ycbcr = vi_rgb.convert("YCbCr")
        y, cb, cr = vi_ycbcr.split()
        fu_y = Image.open(fu_path).convert("L")
        if fu_y.size != vi_rgb.size:
            fu_y = fu_y.resize(vi_rgb.size, Image.BILINEAR)
        fused_rgb = Image.merge("YCbCr", (fu_y, cb, cr)).convert("RGB")
        fused_rgb.save(out_dir / fu_path.name)
        n += 1
    print(f"rebuilt {n} RGB images -> {out_dir}")


if __name__ == "__main__":
    main()
