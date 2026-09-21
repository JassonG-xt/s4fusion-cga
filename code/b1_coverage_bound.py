#!/usr/bin/env python3
"""TEMP DIAGNOSTIC — upper bound on the [integrity] coverage asymmetry.

arb_full300_stats.py prints [integrity] FAIL when the two arms have different
numbers of prediction files, on the premise that a missing file is lost data.
Measured facts say otherwise for this pipeline:
  * no arm has a single empty label file (290-298 non-empty files per arm), so
    ultralytics simply does not write a file for an image with zero detections;
  * every id missing from the CGA_str arm still has its CGA_str fused image on
    disk, so the detector ran on it and found nothing above conf 0.25.
Scoring those objects as "not detected" is therefore correct, not a bias.

This script quantifies how much the asymmetry COULD matter under the most
adversarial reading, i.e. if one insisted that the extra missing images in the
CGA_str arm are lost data. Upper bound = (their GT objects) / (high-conflict
objects in the pooled subset): assume every object in those images is
high-conflict AND was wrongly scored as missed.

Run from the code directory with the training interpreter.
"""
import csv
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

CODE = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code"
ARB = Path(CODE) / "results" / "arb"
MANIFEST = Path(CODE).parent / "dataset" / "manifests" / "m3fd_test.csv"
ARCHIVE = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/archive.zip")

# From the B-1 run (2026-09-15): pooled objects and the high-conflict subset.
N_OBJECTS = 2583
N_HIGH = 878
DELTA_HIGH = 0.0524      # observed uniform - learned on high-conflict recall
P_HIGH = 1.115e-05


def gt_count(zf, sid):
    try:
        root = ET.fromstring(zf.read(f"Annotation/{sid}.xml"))
    except (KeyError, ET.ParseError):
        return None
    size = root.find("size")
    w = float(size.findtext("width", "0")) if size is not None else 0.0
    h = float(size.findtext("height", "0")) if size is not None else 0.0
    if w <= 0 or h <= 0:
        return None
    return sum(1 for _ in root.findall("object"))


def label_ids(arm):
    d = ARB / arm / "runs_full" / "det" / "labels"
    return {f.stem for f in d.glob("*.txt")}


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})

    learned = label_ids("CGA_str")
    uniform = label_ids("cga_uniform")

    only_learned_missing = sorted((set(ids) - learned) - (set(ids) - uniform))
    print(f"[bound] ids in manifest            : {len(ids)}")
    print(f"[bound] learned label files        : {len(learned)}")
    print(f"[bound] uniform label files        : {len(uniform)}")
    print(f"[bound] ids missing ONLY in learned: {len(only_learned_missing)} -> {only_learned_missing}")

    with zipfile.ZipFile(ARCHIVE) as zf:
        counts = {sid: gt_count(zf, sid) for sid in only_learned_missing}

    total = 0
    for sid in only_learned_missing:
        c = counts[sid]
        print(f"[bound]   {sid}: GT objects={c if c is not None else 'NO_XML'}")
        if c:
            total += c

    share = total / N_HIGH if N_HIGH else float("nan")
    print(f"[bound] worst-case extra high-conflict objects: {total}")
    print(f"[bound] as a share of the high-conflict subset ({N_HIGH}): {share:.4f}")
    print(f"[bound] observed uniform - learned delta: {DELTA_HIGH:+.4f} (p={P_HIGH:.3e})")
    corrected = DELTA_HIGH - share
    print(f"[bound] delta after the most adversarial correction: {corrected:+.4f}")
    print("[bound] VERDICT: "
          + ("the asymmetry CANNOT explain the observed gap (direction survives)"
             if corrected > 0 else
             "the asymmetry COULD in principle explain the gap -- needs a subset re-run"))
    print(f"[bound] sanity: extra objects / all pooled objects = {total / N_OBJECTS:.4f}")


if __name__ == "__main__":
    main()
