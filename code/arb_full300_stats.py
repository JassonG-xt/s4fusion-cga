#!/usr/bin/env python3
"""Full-300 per-object arbitration stats: trained CGA variant vs same-regime B0.

Extends cga_compare.py (legacy 90-image split) to the official 300-image M3FD
test manifest, reading GT from archive.zip annotations (same parser as
arb_detect_full300.py) and predictions from runs_full/det labels.

Reports THREE verdicts, deliberately kept separate:

  [primary-H5]   the FROZEN primary criterion: p < 0.05 AND recovered-lost >= 3.
                 This is the pre-registered criterion that governs the route
                 decision (GATE1_FREEZE.md §3). It does NOT include a magnitude
                 threshold.
  [gate-C1]      the STRICTER gate: high-conflict recall delta >= +0.02 AND
                 n01 > n10 AND p < 0.05. GATE1_FREEZE.md L58-64 registers this as
                 a stricter gate that the frozen evidence FAILED. Reporting it is
                 required, but it must never be substituted for the primary
                 criterion — under the frozen data the primary criterion PASSES
                 while gate-C1 FAILS, so reading the wrong line reverses the
                 route choice.
  [integrity]    prediction-coverage check. A missing prediction file used to be
                 silently scored as "not detected", which biases the delta
                 towards whichever arm has fewer labels. Asymmetric coverage now
                 exits non-zero (override with --allow-missing when deliberately
                 reproducing a frozen number).
"""
import argparse
import csv
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

import diag_conflict_detection as D

TOOLS = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/tools/downstream_detection"
sys.path.insert(0, TOOLS)
import run_m3fd_detection_eval as R  # noqa: E402

ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
MANIFEST = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/manifests/m3fd_test.csv")
ROOT_OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation")

def parse_args():
    """Positional TEST BASE are kept positional so existing callers keep working
    (`arb_full300_stats.py <TEST> <BASE>`, or no args for the frozen default)."""
    ap = argparse.ArgumentParser(
        description="Full-300 arbitration stats: [primary-H5] + [gate-C1] + [integrity]")
    ap.add_argument("test", nargs="?", default="CGA_str")
    ap.add_argument("base", nargs="?", default="B0cmp_full")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not exit non-zero on asymmetric prediction coverage "
                         "(use only to reproduce a frozen number deliberately)")
    return ap.parse_args()


def read_annotations(zf: zipfile.ZipFile) -> dict:
    anns = {}
    for n in zf.namelist():
        if not (n.startswith("Annotation/") and n.endswith(".xml")):
            continue
        stem = Path(n).stem
        try:
            anns[stem] = R.parse_xml_labels(zf, f"Annotation/{stem}.xml")
        except (KeyError, ET.ParseError):
            continue
    return anns


def main():
    args = parse_args()
    TEST, BASE = args.test, args.base
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    with zipfile.ZipFile(ROOT_OLD / "archive.zip") as zf:
        anns = read_annotations(zf)

    cf, base, test = [], [], []
    miss_b = miss_t = 0
    n_has_gt = n_has_src = 0
    for sid in ids:
        gt_lines = anns.get(sid)
        if not gt_lines:
            continue
        n_has_gt += 1
        ir_p, vi_p = D.SRC / "Ir" / f"{sid}.png", D.SRC / "Vis" / f"{sid}.png"
        if not (ir_p.is_file() and vi_p.is_file()):
            continue
        n_has_src += 1
        ir = np.asarray(Image.open(ir_p).convert("L"), float) / 255
        vi = np.asarray(Image.open(vi_p).convert("RGB").convert("YCbCr").getchannel("Y"), float) / 255
        H, W = ir.shape
        gxi, gyi, gi = D.grad(ir)
        gxv, gyv, gv = D.grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2

        pb_path = ARB / BASE / "runs_full" / "det" / "labels" / f"{sid}.txt"
        pt_path = ARB / TEST / "runs_full" / "det" / "labels" / f"{sid}.txt"
        pb = D.load_yolo(pb_path, True) if pb_path.is_file() else None
        pt = D.load_yolo(pt_path, True) if pt_path.is_file() else None
        if pb is None:
            miss_b += 1
        if pt is None:
            miss_t += 1

        for line in gt_lines:
            s = line.split()
            if len(s) < 5:
                continue
            g = (int(float(s[0])), float(s[1]), float(s[2]), float(s[3]), float(s[4]), 1.0)
            x0, y0, x1, y1 = D.to_xyxy(g, W, H)
            xi0, yi0 = int(max(0, x0)), int(max(0, y0))
            xi1, yi1 = int(min(W, x1)), int(min(H, y1))
            if xi1 - xi0 < 2 or yi1 - yi0 < 2:
                continue
            gx = (x0, y0, x1, y1)
            cf.append(float(conflict[yi0:yi1, xi0:xi1].mean()))
            base.append(bool(pb is not None and D.detected(gx, g[0], pb, W, H)))
            test.append(bool(pt is not None and D.detected(gx, g[0], pt, W, H)))

    A = {k: np.array(v) for k, v in dict(cf=cf, base=base, test=test).items()}
    hi = A["cf"] > np.percentile(A["cf"], 66)
    b_hi, t_hi = A["base"][hi], A["test"][hi]
    n01 = int(((b_hi == 0) & (t_hi == 1)).sum())
    n10 = int(((b_hi == 1) & (t_hi == 0)).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    delta_hi = float(t_hi.mean() - b_hi.mean())
    delta_all = float(A["test"].mean() - A["base"].mean())

    # --- integrity (M3) ------------------------------------------------------
    # A missing prediction file is scored as "not detected" (see pb/pt = None
    # above), which biases the delta towards whichever arm has fewer labels.
    print(f"[integrity] ids_total={len(ids)} ids_with_gt={n_has_gt} "
          f"ids_with_sources={n_has_src} objects={len(A['cf'])}")
    print(f"[integrity] the high-conflict subset is variant-independent by "
          f"construction (conflict is computed from the IR/VI inputs, not the predictions)")
    print(f"[integrity] prediction coverage: {BASE}={n_has_src - miss_b}/{n_has_src} "
          f"{TEST}={n_has_src - miss_t}/{n_has_src}")
    asym = miss_b != miss_t
    if asym:
        favoured = TEST if miss_b > miss_t else BASE
        print(f"[integrity] FAIL: prediction coverage is ASYMMETRIC "
              f"(missing {BASE}={miss_b}, {TEST}={miss_t}). The arm with fewer labels is "
              f"scored as 'not detected' more often, which biases the delta towards "
              f"{favoured}. Investigate before reading any verdict below.")
    elif miss_b:
        print(f"[integrity] WARN: {miss_b} prediction file(s) missing in BOTH arms; "
              f"coverage is symmetric so the delta is not directionally biased.")

    print(f"[stats] objects={len(A['cf'])} high-conflict={int(hi.sum())} "
          f"(missing preds: {BASE}={miss_b}, {TEST}={miss_t})")
    print(f"[stats] recall ALL:   {BASE}={A['base'].mean():.4f} -> {TEST}={A['test'].mean():.4f} ({delta_all:+.4f})")
    print(f"[stats] recall HIGH:  {BASE}={b_hi.mean():.4f} -> {TEST}={t_hi.mean():.4f} ({delta_hi:+.4f})  "
          f"recovered={n01} lost={n10}  p={p:.3e}")

    # --- the FROZEN primary criterion: this is what governs the route decision --
    ok_primary = (p < 0.05) and ((n01 - n10) >= 3)
    print(f"[primary-H5] p<0.05 AND recovered-lost>=3 "
          f"(FROZEN primary criterion, GATE1_FREEZE.md \u00a73): "
          f"{'PASS' if ok_primary else 'FAIL'} "
          f"(p={p:.3e}, recovered-lost={n01 - n10:+d})   <== GOVERNS THE ROUTE DECISION")

    # --- the STRICTER gate: must be reported, must NEVER replace the primary ---
    ok_c1 = delta_hi >= 0.02 and n01 > n10 and p < 0.05
    print(f"[gate-C1] high-conflict recall >=+0.02 & p<0.05 "
          f"(STRICTER pre-registered gate, GATE1_FREEZE.md L58-64): "
          f"{'PASS' if ok_c1 else 'FAIL'} "
          f"(delta={delta_hi:+.4f}, p={p:.2e})   <== secondary, not the primary criterion")

    if asym and not args.allow_missing:
        print("[integrity] exiting non-zero: coverage is asymmetric. "
              "Use --allow-missing ONLY to reproduce a frozen number deliberately.")
        sys.exit(3)


if __name__ == "__main__":
    main()
