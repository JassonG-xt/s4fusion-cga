"""F3 verdict: compare cross-scale drift of S vs B0. Lower drift = more
scale-consistent. D1 is supported if S drifts significantly LESS than B0 at high
resolution (delta = S - B0 negative). Paired Wilcoxon per resolution + drift-AUC."""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import stats


def _load(path):
    with open(path, encoding="utf-8-sig") as h:
        return {(r["dataset"], r["sample_id"]): r for r in csv.DictReader(h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-csv", required=True)
    ap.add_argument("--s-csv", required=True)
    args = ap.parse_args()
    b0, s = _load(args.b0_csv), _load(args.s_csv)
    keys = sorted(set(b0) & set(s))
    cols = [c for c in next(iter(b0.values())) if c.startswith("drift_")]
    print(f"paired images: {len(keys)}   resolutions: {[c.split('_')[1] for c in cols]}")
    print(f"\n{'res':>6} {'B0 drift':>10} {'S drift':>10} {'delta(S-B0)':>12} {'p(S<B0)':>10}  D1?")
    s_wins = 0
    b0_auc = np.zeros(len(keys))
    s_auc = np.zeros(len(keys))
    for j, c in enumerate(cols):
        b0v = np.array([float(b0[k][c]) for k in keys])
        sv = np.array([float(s[k][c]) for k in keys])
        b0_auc += b0v
        s_auc += sv
        d = sv - b0v
        p_less = stats.wilcoxon(sv, b0v, alternative="less").pvalue if not np.allclose(d, 0) else 1.0
        favors = d.mean() < 0 and p_less < 0.05
        s_wins += favors
        print(f"{c.split('_')[1]:>6} {b0v.mean():>10.4f} {sv.mean():>10.4f} {d.mean():>+12.4f} "
              f"{p_less:>10.2e}  {'YES' if favors else 'no'}")
    # drift area under curve (lower = better across the whole resolution sweep)
    dauc = s_auc - b0_auc
    p_auc = stats.wilcoxon(s_auc, b0_auc, alternative="less").pvalue
    print(f"\ndrift-AUC  B0={b0_auc.mean():.4f}  S={s_auc.mean():.4f}  "
          f"delta={dauc.mean():+.4f}  p(S<B0)={p_auc:.2e}")
    print("\n=== F3 VERDICT (D1 real claim) ===")
    hi = cols[-1].split("_")[1]
    print(f"resolutions where S drifts significantly LESS than B0: {s_wins}/{len(cols)}")
    if dauc.mean() < 0 and p_auc < 0.05:
        print(f"D1 SUPPORTED: mechanism is more scale-consistent (lower drift-AUC, incl high res {hi}).")
    else:
        print("D1 NOT SUPPORTED on cross-scale drift either: mechanism does not improve "
              "global scale-consistency. Strong signal to pivot the proposition.")


if __name__ == "__main__":
    main()
