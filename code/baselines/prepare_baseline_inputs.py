#!/usr/bin/env python3
"""Prepare baseline inputs from manifests.

Reads <manifest-dir>/<dataset>_test.csv, writes to <out-root>/<dataset>/{ir,vi}/
with sample_id naming, so every baseline official script can consume them
unmodified. Copies (or symlinks) original images; does not resize (resize
alignment happens at output-collection time).

Usage:
  python prepare_baseline_inputs.py --datasets m3fd --manifest-dir ../dataset/manifests --out-root ./inputs
"""
import argparse
import csv
import shutil
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="m3fd", help="comma-separated dataset keys")
    ap.add_argument("--manifest-dir", default="../dataset/manifests")
    ap.add_argument("--out-root", default="./inputs")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_root = Path(args.out_root)
    for key in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        mf = manifest_dir / f"{key}_test.csv"
        if not mf.is_file():
            raise FileNotFoundError(f"manifest not found: {mf}")
        ir_out = out_root / key / "ir"
        vi_out = out_root / key / "vi"
        ir_out.mkdir(parents=True, exist_ok=True)
        vi_out.mkdir(parents=True, exist_ok=True)

        n = 0
        with mf.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["sample_id"]
                ir_src = manifest_dir.parent / row["ir_path"]
                vi_src = manifest_dir.parent / row["vi_path"]
                for src, dst_dir in ((ir_src, ir_out), (vi_src, vi_out)):
                    if not src.is_file():
                        raise FileNotFoundError(f"{src} missing")
                    ext = src.suffix if src.suffix.lower() in IMG_EXT else ".png"
                    dst = dst_dir / f"{sid}{ext}"
                    if args.copy:
                        if not dst.exists():
                            shutil.copy2(src, dst)
                    else:
                        if dst.exists() and dst.is_symlink():
                            continue
                        if dst.exists():
                            dst.unlink()
                        dst.symlink_to(src.resolve())
                n += 1
        print(f"{key}: {n} pairs -> {ir_out.parent}")


if __name__ == "__main__":
    main()
