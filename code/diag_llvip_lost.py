#!/usr/bin/env python3
"""Diagnose why CGA loses on LLVIP: compare lost vs recovered object stats."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026")
ARB = ROOT / "code" / "results" / "arb"
MANIFEST = ROOT / "dataset" / "manifests" / "llvip_test_sub1000.csv"
DATA = ROOT / "dataset"
CONF = 0.25

sys.path.insert(0, str(ROOT / "code"))
import importlib.util
spec = importlib.util.spec_from_file_location("arbd", ROOT / "code" / "arb_detect_llvip.py")
arbd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arbd)


def load_boxes(txt: Path, conf: float = CONF):
    out = []
    if txt.is_file():
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) >= 6 and float(p[5]) >= conf:
                out.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
    return out


def crop_stats(img: np.ndarray, box_xyxy, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img[y1:y2, x1:x2]
    bg = img[max(0, y1 - 20):min(h, y2 + 20), max(0, x1 - 20):min(w, x2 + 20)]
    return float(patch.std()), float(abs(patch.mean() - bg.mean())), float(patch.mean())


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    gt_dir = ARB / "B0cmp_llvip" / "ds_llvip" / "test" / "labels"
    b_lbl = ARB / "B0cmp_llvip" / "runs_llvip" / "det" / "labels"
    o_lbl = ARB / "CGA_llvip" / "runs_llvip" / "det" / "labels"
    cga_dir = ARB / "CGA_llvip" / "images"
    b0_dir = ARB / "B0cmp_llvip" / "images"

    lost_stats, rec_stats = [], []
    for sid in ids:
        gf = gt_dir / f"{sid}.txt"
        if not gf.is_file():
            continue
        gt = [([0] + [float(x) for x in l.split()[1:5]]) for l in gf.read_text().splitlines() if l.split()]
        if not gt:
            continue
        b_pred, o_pred = load_boxes(b_lbl / f"{sid}.txt"), load_boxes(o_lbl / f"{sid}.txt")
        b_missed = arbd.match_stats(gt, b_pred, 1280, 1024)[3]
        o_missed = arbd.match_stats(gt, o_pred, 1280, 1024)[3]
        rec = [gi for gi in b_missed if gi not in o_missed]
        lost = [gi for gi in o_missed if gi not in b_missed]
        if not (rec or lost):
            continue
        try:
            ir = np.array(Image.open(DATA / f"LLVIP/LLVIP/infrared/test/{sid}.jpg").convert("L"))
            vi = np.array(Image.open(DATA / f"LLVIP/LLVIP/visible/test/{sid}.jpg").convert("L"))
            cb = np.array(Image.open(b0_dir / f"{sid}.png").convert("L"))
            cc = np.array(Image.open(cga_dir / f"{sid}.png").convert("L"))
        except FileNotFoundError:
            continue
        for gi in rec:
            box = arbd.yolo_to_xyxy(gt[gi], 1280, 1024)
            rec_stats.append((sid, "rec", *crop_stats(ir, box, 1280, 1024),
                              *crop_stats(vi, box, 1280, 1024),
                              *crop_stats(cb, box, 1280, 1024),
                              *crop_stats(cc, box, 1280, 1024), box[2] - box[0]))
        for gi in lost:
            box = arbd.yolo_to_xyxy(gt[gi], 1280, 1024)
            lost_stats.append((sid, "lost", *crop_stats(ir, box, 1280, 1024),
                               *crop_stats(vi, box, 1280, 1024),
                               *crop_stats(cb, box, 1280, 1024),
                               *crop_stats(cc, box, 1280, 1024), box[2] - box[0]))

    def report(name, rows):
        if not rows:
            print(f"{name}: n=0")
            return
        a = np.array([r[2:] for r in rows], dtype=float)
        means = a.mean(axis=0)
        print(f"{name}: n={len(rows)}")
        print(f"  IR  std={means[0]:.1f} mean={means[2]:.1f}  (patch std, mean)")
        print(f"  VI  std={means[3]:.1f} mean={means[5]:.1f}")
        print(f"  B0  std={means[6]:.1f} mean={means[8]:.1f}")
        print(f"  CGA std={means[9]:.1f} mean={means[11]:.1f}")
        print(f"  box width px: {means[12]:.0f}")

    report("RECOVERED (B0 missed, CGA hit)", rec_stats)
    report("LOST (B0 hit, CGA missed)", lost_stats)


if __name__ == "__main__":
    main()