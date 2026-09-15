#!/usr/bin/env python3
"""Collect baseline fused outputs into a unified layout, align resolution to IR,
then run the unified evaluator (eval_metrics.py) per method/dataset.

Layout expected from baseline inference (each method may differ slightly):
  <out-root>/<method>/<dataset>/*.png|.jpg|.bmp   (named by sample_id)

Resolution alignment: fused images are bilinear-resized to the IR image size
(same sample_id) so every method is compared at identical resolution.

Usage:
  python collect_outputs.py --methods meta_fusion,dcevo --datasets m3fd \
      --inputs ./inputs --baseline-out <out-root> --metrics-out ../results/baselines
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_image(d: Path, sid: str):
    for ext in IMG_EXT:
        p = d / f"{sid}{ext}"
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True, help="comma-separated method dirs under baseline-out")
    ap.add_argument("--datasets", default="m3fd", help="comma-separated dataset keys")
    ap.add_argument("--inputs", default="./inputs")
    ap.add_argument("--baseline-out", default="./outputs", help="dir containing <method>/<dataset>/ fused images")
    ap.add_argument("--metrics-out", default="../results/baselines")
    ap.add_argument("--manifest-dir", default="../dataset/manifests")
    ap.add_argument("--eval-script", default="eval_metrics.py")
    ap.add_argument("--no-align", action="store_true", help="skip resolution alignment (only collect)")
    ap.add_argument("--skip-metrics", action="store_true", help="only align/collect, no eval run")
    ap.add_argument("--skip-perceptual", action="store_true", help="pass --skip-perceptual to eval_metrics (no TOPIQ/MUSIQ)")
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    inputs = Path(args.inputs)
    baseline_out = Path(args.baseline_out)
    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for key in datasets:
        # IR sizes come from the inputs prepared by prepare_baseline_inputs.py
        ir_dir = inputs / key / "ir"
        if not ir_dir.is_dir():
            print(f"skip {key}: no {ir_dir}", file=sys.stderr)
            continue
        for method in methods:
            src = baseline_out / method / key
            if not src.is_dir():
                print(f"skip {method}/{key}: no {src}", file=sys.stderr)
                continue
            aligned = baseline_out / f"_aligned" / method / key
            aligned.mkdir(parents=True, exist_ok=True)
            n = 0
            for img in sorted(src.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                sid = img.stem
                ir_ref = _find_image(ir_dir, sid)
                if ir_ref is None:
                    continue
                with Image.open(img) as im:
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    out = im
                    if not args.no_align:
                        ref = Image.open(ir_ref)
                        if out.size != ref.size:
                            out = im.resize(ref.size, Image.BILINEAR)
                    out.save(aligned / f"{sid}.png")
                n += 1
            print(f"aligned {method}/{key}: {n} images -> {aligned}")
            if args.skip_metrics:
                continue
            csv_path = metrics_out / f"{method}_{key}.csv"
            cmd = [
                sys.executable, args.eval_script,
                "--ir-path", str(ir_dir),
                "--vi-path", str(inputs / key / "vi"),
                "--fused-path", str(aligned),
                "--use-y",
                "--out-csv", str(csv_path),
            ]
            if args.skip_perceptual:
                cmd.append("--skip-perceptual")
            print("RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
