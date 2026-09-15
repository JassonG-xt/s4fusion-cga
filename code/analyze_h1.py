"""H1 gate analysis: paired significance of cell S (state modulation) vs B0
(baseline) on per-image fusion metrics. Reads two evaluate_brss.py metric CSVs,
aligns by (dataset, sample_id), and reports for each metric the mean/median
delta, paired Wilcoxon p (two-sided + one-sided S>B0), a 95% bootstrap CI on the
median delta, and Holm-corrected significance. All metrics here are
higher-is-better, so delta = S - B0 (positive favors the mechanism).

Usage:
  analyze_h1.py --b0-csv results/abl/B0_m3fd.csv --s-csv results/abl/S_m3fd.csv \
                [--b0-ckpt ...pt --s-ckpt ...pt]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import stats

METRICS = ("EN", "SF", "AG", "MI", "QABF")          # higher = better
STRUCTURAL = ("SF", "AG", "QABF")                    # H1 decision metrics
BOOT_SEED = 42
BOOT_N = 2000


def _load(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"].startswith("__"):        # skip __mean__/__std__ footer
                continue
            rows[(row["dataset"], row["sample_id"])] = row
    return rows


def _param_count(path):
    import torch
    state = torch.load(path, map_location="cpu")["model"]
    return sum(v.numel() for v in state.values())


def _holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-csv", required=True)
    ap.add_argument("--s-csv", required=True)
    ap.add_argument("--b0-ckpt", default="")
    ap.add_argument("--s-ckpt", default="")
    args = ap.parse_args()

    b0, s = _load(args.b0_csv), _load(args.s_csv)
    keys = sorted(set(b0) & set(s))
    if not keys:
        raise SystemExit("no overlapping (dataset, sample_id) between the two CSVs")
    print(f"paired images: {len(keys)}  (B0 rows={len(b0)}, S rows={len(s)})")
    if len(keys) != len(b0) or len(keys) != len(s):
        print(f"  WARNING: unmatched rows dropped (B0-only={len(b0)-len(keys)}, S-only={len(s)-len(keys)})")

    rng = np.random.default_rng(BOOT_SEED)
    raw_p, labels = [], []
    print(f"\n{'metric':<6} {'B0 mean':>9} {'S mean':>9} {'meanD':>8} {'medD':>8} "
          f"{'medD 95% CI':>20} {'Wilcox p':>10} {'p(S>B0)':>9}")
    results = {}
    for m in METRICS:
        b0v = np.array([float(b0[k][m]) for k in keys])
        sv = np.array([float(s[k][m]) for k in keys])
        d = sv - b0v
        med = float(np.median(d))
        boot = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(BOOT_N)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if np.allclose(d, 0):
            p_two = p_gt = 1.0
        else:
            p_two = stats.wilcoxon(sv, b0v, zero_method="wilcox", alternative="two-sided").pvalue
            p_gt = stats.wilcoxon(sv, b0v, zero_method="wilcox", alternative="greater").pvalue
        results[m] = dict(b0=b0v.mean(), s=sv.mean(), meanD=d.mean(), medD=med, ci=(lo, hi), p=p_two, p_gt=p_gt)
        raw_p.append(p_two)
        labels.append(m)
        print(f"{m:<6} {b0v.mean():>9.4f} {sv.mean():>9.4f} {d.mean():>+8.4f} {med:>+8.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}] {p_two:>10.2e} {p_gt:>9.2e}")

    adj = _holm(np.array(raw_p))
    print("\nHolm-corrected two-sided p:")
    for m, a in zip(labels, adj):
        print(f"  {m:<6} {a:.2e}  {'*' if a < 0.05 else ''}")

    if args.b0_ckpt and args.s_ckpt:
        pb, ps = _param_count(args.b0_ckpt), _param_count(args.s_ckpt)
        print(f"\nparams: B0={pb}  S={ps}  delta=+{ps - pb} ({100*(ps-pb)/pb:.4f}%)")

    # H1 verdict
    holm = dict(zip(labels, adj))
    wins = [m for m in STRUCTURAL if results[m]["medD"] > 0 and holm[m] < 0.05]
    print("\n=== H1 VERDICT ===")
    print(f"structural metrics with S>B0 AND Holm p<0.05: {wins or 'NONE'}")
    print("H1 SUPPORTED (proceed to G/R/F/F+G/S-)" if len(wins) >= 2
          else "H1 NOT met on >=2 structural metrics — revisit mechanism framing per protocol")


if __name__ == "__main__":
    main()
