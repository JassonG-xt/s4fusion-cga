#!/usr/bin/env python3
"""TEMP DIAGNOSTIC — the pooled cross-seed test required by GATE1_PREREG_B3 §4.2.

§3.1 of the unfreeze ledger lists the pooled test as a mandatory field, and §4.2
says the per-seed exact binomials must be accompanied by a "combined test
stratified by image (Mantel-Haenszel or Fisher)". Neither b3eval nor the stats
script emits it, so it is computed here from the per-seed contingency counts.

For paired binary data (each high-conflict object is either
recovered by CGA but not B0, n01, or lost, n10) the stratum-wise
Mantel-Haenszel statistic reduces to summing n01 and n10 across strata and running
the same exact binomial the per-seed criterion uses. Fisher's method over the three
per-seed p-values is reported as a second, independent combination.

Inputs are transcribed from the b3eval log (_temp/p1_20260916_122458.log, lines
51/93/135) and are re-checked against it below.
"""
import numpy as np
from scipy import stats

# seed: (recovered n01, lost n10, per-seed two-sided binomial p, high-conflict n)
PER_SEED = {
    42:   (18, 4, 4.344e-03, 878),
    123:  (13, 7, 2.632e-01, 878),
    3407: (11, 7, 4.807e-01, 878),
}


def main():
    print("=== per-seed (from the b3eval log) ===")
    for s, (n01, n10, p, n) in PER_SEED.items():
        rec = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
        flag = "PASS" if (rec < 0.05 and (n01 - n10) >= 3) else "FAIL"
        print(f"  s{s:<5} recovered={n01:<3} lost={n10:<3} delta={n01 - n10:+d}  "
              f"p={p:.3e} (recomputed {rec:.3e})  [primary-H5] {flag}  n_high={n}")

    n01_tot = sum(v[0] for v in PER_SEED.values())
    n10_tot = sum(v[1] for v in PER_SEED.values())
    n_tot = sum(v[3] for v in PER_SEED.values())

    print("\n=== pooled, stratified by seed (Mantel-Haenszel reduction for paired data) ===")
    print(f"  pooled discordant pairs: recovered={n01_tot} lost={n10_tot} "
          f"(total {n01_tot + n10_tot}); pooled high-conflict n = {n_tot}")
    p_pool = stats.binomtest(min(n01_tot, n10_tot), n01_tot + n10_tot, 0.5).pvalue
    print(f"  exact binomial p = {p_pool:.4e}   delta = {n01_tot - n10_tot:+d}")
    print(f"  => pooled test {'PASSES' if p_pool < 0.05 else 'FAILS'} p<0.05")

    # stratification check: is the direction stable across strata?
    deltas = {s: v[0] - v[1] for s, v in PER_SEED.items()}
    print(f"  per-seed deltas: {deltas}  -> negative seeds: "
          f"{[s for s, d in deltas.items() if d < 0] or 'none'}")

    print("\n=== Fisher's method over the three per-seed p-values (sensitivity) ===")
    ps = [v[2] for v in PER_SEED.values()]
    chi2 = -2 * np.sum(np.log(ps))
    p_fisher = stats.chi2.sf(chi2, 2 * len(ps))
    print(f"  chi2 = {chi2:.4f} (df={2 * len(ps)}), combined p = {p_fisher:.4e}")

    print("\n=== aggregation verdict under the pre-registered rule ===")
    k = sum(1 for v in PER_SEED.values()
            if v[2] < 0.05 and (v[0] - v[1]) >= 3)
    # s123 and s3407 share a deterministic high-conflict subset; their statistics
    # here differ (13/7 vs 11/7), so they are NOT duplicates in this 口径 -- the
    # k_indep rule is applied as pre-registered regardless.
    print(f"  k = {k}/3 (seeds passing [primary-H5])")
    print(f"  pre-registered threshold: k >= 2 => "
          f"{'replicated' if k >= 2 else 'NOT replicated'}")
    print(f"  note: the pooled test is significant ({p_pool:.2e}) while k = {k};")
    print(f"        §4.2 requires BOTH to be reported, and §4.1 makes k the criterion.")


if __name__ == "__main__":
    main()
