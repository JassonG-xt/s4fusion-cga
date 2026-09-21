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
  [integrity]    prediction-coverage check. REVISED 2026-09-15 (M10). The original
                 guard failed on any inter-arm difference in prediction-file count,
                 which measurement showed to be wrong for this pipeline: ultralytics
                 writes a label file only for images that have detections (290-298
                 files per arm, zero empty files across all seven arms), so a missing
                 file is a zero-detection result and scoring it as "not detected" is
                 correct. The guard now fails only on evidence of real data loss --
                 empty label files, missing fused inputs, or a count below
                 --min-labels -- and reports a bare inter-arm difference as INFO.
                 --strict-coverage restores the old behaviour so the original
                 verdict stays reproducible, and --allow-missing suppresses the exit
                 code entirely.

ADDED 2026-09-18 (review round, R1/R5) -- additive only, no computed value changes:
  * `--tests A B C`  evaluate several candidate arms against ONE base in a single
    pass over the manifest, reusing the annotation load and the conflict field.
    Without the flag the script behaves exactly as before (default TEST=BASE pair).
  * `--csv PATH`     write the headline numbers of every evaluated pair, including
    the McNemar odds ratio n01/n10 with a 95% interval derived from the exact
    Clopper-Pearson interval on the discordant-pair proportion. This is the
    machine-readable source for the effect-size / CI reporting the review asked
    for, so no such number has to be re-derived by hand.

The refactor that enables `--tests` splits the old single loop into (1) an
object/conflict collection pass that does not touch predictions and (2) a
per-arm scoring pass. The object order, the filtering and every formula are
unchanged: the frozen default pair still reproduces 0.4385 -> 0.4579, 20/3,
p=4.883e-04 digit for digit (regression test run on 2026-09-18).
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
    ap.add_argument("--tests", nargs="*", default=None,
                    help="evaluate several arms against the same BASE in one pass; "
                         "overrides the positional TEST")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="explicit test:base pairs evaluated in one pass (each pair may "
                         "have a different base); overrides --tests and the positionals")
    ap.add_argument("--csv", default=None,
                    help="write headline numbers (+ McNemar OR with 95%% CI) for every "
                         "evaluated pair to this path")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not exit non-zero on any integrity finding "
                         "(use only to reproduce a frozen number deliberately)")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="M10: restore the pre-2026-09-15 guard, which treated any "
                         "inter-arm difference in prediction-file count as lost data. "
                         "Kept so the original verdict stays reproducible; the premise "
                         "was measured to be wrong for this pipeline (see the integrity "
                         "block in main()).")
    ap.add_argument("--min-labels", type=int, default=None,
                    help="M10: fail if either arm has fewer than this many prediction "
                         "files (pass that arm's own historical count).")
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


def _label_dir(arm: str) -> Path:
    return ARB / arm / "runs_full" / "det" / "labels"


def _label_stats(arm: str) -> tuple[int, int]:
    """(file count, empty-file count) for one arm's prediction directory.

    The empty-file count is the discriminator that matters: ultralytics writes a
    label file only when an image has at least one detection, so a MISSING file is
    a zero-detection result, whereas an EMPTY file means the image was reached but
    the write was truncated. Only the latter (plus missing fused inputs) is
    evidence of lost data.
    """
    d = _label_dir(arm)
    files = sorted(d.glob("*.txt")) if d.is_dir() else []
    return len(files), sum(1 for f in files if f.stat().st_size == 0)


def collect_objects(ids, anns):
    """Pass 1 -- image/annotation/conflict only; predictions are NOT touched here.

    The conflict field and the high-conflict subset are functions of the IR/VI
    inputs alone, so they are computed once and shared by every arm. Object order
    is exactly the frozen loop's order (ids ascending, then GT lines ascending).
    """
    objects = []          # (sid, box_xyxy, cls, conflict_mean)
    meta = {}             # sid -> (H, W)
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
        meta[sid] = (H, W)
        gxi, gyi, gi = D.grad(ir)
        gxv, gyv, gv = D.grad(vi)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2
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
            objects.append((sid, (x0, y0, x1, y1), g[0],
                            float(conflict[yi0:yi1, xi0:xi1].mean())))
    return objects, meta, n_has_gt, n_has_src


def arm_vector(arm: str, objects, meta):
    """Pass 2 -- detection vector for one arm, in the frozen object order.

    `miss` counts IDS (images) whose prediction file is absent, not objects: that
    is the frozen semantics and it is what makes the coverage line
    `n_has_src - miss` agree with the label-file count reported beside it.
    """
    cache: dict = {}
    missing_ids: set = set()
    det = np.zeros(len(objects), dtype=bool)
    for i, (sid, gx, cls, _cf) in enumerate(objects):
        if sid not in cache:
            path = _label_dir(arm) / f"{sid}.txt"
            cache[sid] = D.load_yolo(path, True) if path.is_file() else None
        preds = cache[sid]
        if preds is None:
            missing_ids.add(sid)
        H, W = meta[sid]
        det[i] = bool(preds is not None and D.detected(gx, cls, preds, W, H))
    return det, len(missing_ids)


def or_ci(n01: int, n10: int) -> tuple[float, float, float]:
    """McNemar odds ratio n01/n10 with a 95% interval from the exact Clopper-Pearson
    interval on the discordant-pair proportion pi = n01/(n01+n10).

    pi_lo = CP lower bound -> OR_lo = pi_lo/(1-pi_lo); pi_hi -> OR_hi. The point
    estimate is the sample OR. Guarded against an empty discordant set (no test is
    possible) and against a degenerate bound at 0 or 1 (open interval -> inf).
    """
    n = n01 + n10
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    or_hat = float("inf") if n10 == 0 else n01 / n10
    ci = stats.binomtest(n01, n).proportion_ci(confidence_level=0.95, method="exact")
    lo, hi = float(ci.low), float(ci.high)

    def _t(p: float) -> float:
        if p <= 0.0:
            return 0.0
        return float("inf") if p >= 1.0 else p / (1.0 - p)

    return or_hat, _t(lo), _t(hi)


def score(test: str, base: str, objects, meta, args, anns, ids, n_has_gt, n_has_src,
          base_vec=None, base_miss=None):
    det_t, miss_t = arm_vector(test, objects, meta)
    if base_vec is None:
        det_b, miss_b = arm_vector(base, objects, meta)
    else:
        det_b, miss_b = base_vec, base_miss

    cf = np.array([o[3] for o in objects])
    hi = cf > np.percentile(cf, 66)
    b_hi, t_hi = det_b[hi], det_t[hi]
    n01 = int(((b_hi == 0) & (t_hi == 1)).sum())
    n10 = int(((b_hi == 1) & (t_hi == 0)).sum())
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    delta_hi = float(t_hi.mean() - b_hi.mean())
    delta_all = float(det_t.mean() - det_b.mean())

    # --- integrity (M3, revised 2026-09-15 = M10) ---------------------------
    # The original guard treated ANY inter-arm difference in the number of
    # prediction files as lost data. Measured on 2026-09-15 (B-1 content control)
    # that premise is wrong for this pipeline: ultralytics writes a label file
    # only when an image has >=1 detection -- all seven arms hold 290-298 files
    # and NOT ONE empty file -- so a missing file IS a zero-detection result and
    # scoring it as "not detected" is correct, not a bias. Meanwhile the frozen
    # H5 control itself compared 293 against 290 files, so the old guard would
    # have invalidated the frozen verdict too.
    #
    # Revised rule. FAIL (exit non-zero) only on evidence of real data loss:
    #   (a) an arm contains EMPTY label files      -> write was truncated
    #   (b) a fused image is missing for an id with GT -> the arm never ran
    #   (c) a label count below --min-labels       -> below that arm's own history
    # A bare inter-arm count difference is reported as INFO. --strict-coverage
    # restores the old FAIL behaviour, for deliberately reproducing the original
    # verdict. No number computed above is affected by any of this.
    print(f"[integrity] pair {test} vs {base}")
    print(f"[integrity] ids_total={len(ids)} ids_with_gt={n_has_gt} "
          f"ids_with_sources={n_has_src} objects={len(objects)}")
    print(f"[integrity] the high-conflict subset is variant-independent by "
          f"construction (conflict is computed from the IR/VI inputs, not the predictions)")
    n_b_files, n_b_empty = _label_stats(base)
    n_t_files, n_t_empty = _label_stats(test)
    print(f"[integrity] prediction coverage: {base}={n_has_src - miss_b}/{n_has_src} "
          f"({n_b_files} files, {n_b_empty} empty) "
          f"{test}={n_has_src - miss_t}/{n_has_src} ({n_t_files} files, {n_t_empty} empty)")

    empty_present = (n_b_empty + n_t_empty) > 0
    missing_fused = [sid for sid in ids
                     if not ((ARB / base / "images" / f"{sid}.png").is_file()
                             and (ARB / test / "images" / f"{sid}.png").is_file())]
    low_labels = (args.min_labels is not None
                  and min(n_b_files, n_t_files) < args.min_labels)

    if empty_present:
        print(f"[integrity] FAIL (a): EMPTY label files present "
              f"({base}={n_b_empty}, {test}={n_t_empty}) -- that arm was reached but "
              f"its write was truncated. This IS lost data; investigate.")
    if missing_fused:
        print(f"[integrity] FAIL (b): {len(missing_fused)} id(s) with GT have no fused "
              f"image in one of the arms (e.g. {missing_fused[:5]}) -- that arm never "
              f"ran on them. This IS lost data; investigate.")
    if low_labels:
        print(f"[integrity] FAIL (c): label count below --min-labels={args.min_labels} "
              f"({base}={n_b_files}, {test}={n_t_files}) -- below this arm's own history.")

    asym = miss_b != miss_t
    if asym:
        print(f"[integrity] INFO: the two arms differ in prediction-file count "
              f"(missing {base}={miss_b}, {test}={miss_t}). Under this pipeline a missing "
              f"file is a zero-detection result (ultralytics omits files for images with "
              f"no detection; zero empty files across every arm), so the difference is "
              f"detector behaviour, not lost data. Scoring those objects as 'not detected' "
              f"is correct.")
        if args.strict_coverage:
            favoured = test if miss_b > miss_t else base
            print(f"[integrity] FAIL (strict-coverage): treating the asymmetry as lost data "
                  f"biases the delta towards {favoured}. --strict-coverage was requested.")

    hard_fail = empty_present or bool(missing_fused) or low_labels
    soft_fail = asym and args.strict_coverage
    if (hard_fail or soft_fail) and not args.allow_missing:
        print("[integrity] exiting non-zero. Use --allow-missing ONLY to reproduce a "
              "frozen number deliberately.")
        sys.exit(3)
    if not hard_fail and not soft_fail:
        print("[integrity] PASS: no empty files, no missing fused inputs, no count below "
              "the arm's own history.")

    print(f"[stats] objects={len(objects)} high-conflict={int(hi.sum())} "
          f"(missing preds: {base}={miss_b}, {test}={miss_t})")
    print(f"[stats] recall ALL:   {base}={det_b.mean():.4f} -> {test}={det_t.mean():.4f} ({delta_all:+.4f})")
    print(f"[stats] recall HIGH:  {base}={b_hi.mean():.4f} -> {test}={t_hi.mean():.4f} ({delta_hi:+.4f})  "
          f"recovered={n01} lost={n10}  p={p:.3e}")

    or_hat, or_lo, or_hi = or_ci(n01, n10)
    print(f"[stats] McNemar OR (recovered/lost) = {or_hat:.4f} "
          f"[95% CI {or_lo:.4f}, {or_hi:.4f}]  (exact CP interval on {n01}/{n01+n10} discordant)")

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

    # The integrity verdict (M10) is emitted above, next to the coverage numbers it
    # is derived from, so the three checks and their evidence stay together.

    return {
        "test": test, "base": base, "objects": len(objects),
        "high_conflict_n": int(hi.sum()),
        "base_recall_high": round(float(b_hi.mean()), 6),
        "test_recall_high": round(float(t_hi.mean()), 6),
        "delta_high": round(delta_hi, 6),
        "recovered": n01, "lost": n10,
        "or_mcnemar": ("inf" if or_hat == float("inf") else round(or_hat, 6)),
        "or_ci_lo": ("inf" if or_lo == float("inf") else round(or_lo, 6)),
        "or_ci_hi": ("inf" if or_hi == float("inf") else round(or_hi, 6)),
        "p_exact_binom": f"{p:.6e}",
        "base_recall_all": round(float(det_b.mean()), 6),
        "test_recall_all": round(float(det_t.mean()), 6),
        "delta_all": round(delta_all, 6),
        "primary_H5": "PASS" if ok_primary else "FAIL",
        "gate_C1": "PASS" if ok_c1 else "FAIL",
        "missing_base": miss_b, "missing_test": miss_t,
    }


def main():
    args = parse_args()
    if args.pairs:
        pairs = []
        for spec in args.pairs:
            if ":" not in spec:
                raise SystemExit(f"--pairs entry must be test:base, got {spec!r}")
            t, b = spec.split(":", 1)
            pairs.append((t.strip(), b.strip()))
    else:
        tests = args.tests if args.tests else [args.test]
        pairs = [(t, args.base) for t in tests]

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    with zipfile.ZipFile(ROOT_OLD / "archive.zip") as zf:
        anns = read_annotations(zf)

    objects, meta, n_has_gt, n_has_src = collect_objects(ids, anns)

    base_cache: dict = {}
    out = []
    for test, base in pairs:
        if base not in base_cache:
            base_cache[base] = arm_vector(base, objects, meta)
        det_b, miss_b = base_cache[base]
        out.append(score(test, base, objects, meta, args, anns, ids,
                         n_has_gt, n_has_src, base_vec=det_b, base_miss=miss_b))
        print()

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            wr.writeheader()
            wr.writerows(out)
        print(f"[csv] wrote {len(out)} row(s) -> {path}")


if __name__ == "__main__":
    main()
