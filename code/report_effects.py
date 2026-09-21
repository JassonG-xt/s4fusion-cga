#!/usr/bin/env python3
"""R5 (review round 2026-09-18): effect sizes and confidence intervals.

Why: the cross-seed table reports only pooled per-image Wilcoxon p-values, some as
extreme as 1e-146. With 900 paired images such p-values are driven by N, not by the
size or the reproducibility of the effect, so the review asked for the two things
that are actually interpretable: (1) an effect size for every test, and (2) an
interval on every reported delta.

This script adds, without touching any frozen number:
  * rank-biserial correlation  r_rb = (W+ - W-) / (W+ + W-)
    (0 = no effect, +/-1 = every pair moves one way; computed on the same non-zero
    differences that the frozen Wilcoxon uses, zero_method='wilcox')
  * a 95% bootstrap interval for the mean pooled delta (resampling images, fixed seed)
  * per-seed rows as well, so the evidence body can be moved from the pooled p-value
    to per-seed effect sizes, which is what the review recommended

The pairing, the per-seed grouping and the Wilcoxon call mirror
code/analyze_seeds.py exactly, so running it on the frozen inputs reproduces the
frozen p-values (self-check: primary tag must give EN 1.31e-16 ... AG 4.35e-147).

Usage (from code/):
  # frozen cross-seed table: CGA vs the 20-epoch baseline
  python report_effects.py --glob-test 'results/seeds/CGA_str_s*.csv' \
      --glob-base 'results/seeds/B0_s*.csv' --tag primary \
      --out results/seeds/effects_primary.csv
  # budget-matched: CGA vs each seed's own 32-epoch baseline
  python report_effects.py --glob-test 'results/seeds/CGA_str_s*.csv' \
      --glob-base 'results/seeds/B0_e32_s*.csv' --tag budget \
      --out results/seeds/effects_budget_matched.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

DEFAULT_METRICS = ["EN", "SF", "AG", "MI", "QABF", "VIF"]


def _load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8-sig") as h:
        rows = [r for r in csv.DictReader(h) if not str(r.get("name", "")).startswith("__")]
    return {r["name"]: {m: float(r[m]) for m in r if m != "name"} for r in rows}


def _glob_seed_files(pattern: str) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(Path(".").glob(pattern)):
        m = re.search(r"_s(\d+)\.csv$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise SystemExit(f"no files match {pattern} with seed in name (_s<seed>.csv)")
    return out


def _holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def rank_biserial(diffs: list[float]) -> tuple[float, float, float]:
    """(r_rb, W+, W-) on the non-zero differences, ties averaged."""
    d = np.asarray([x for x in diffs if x != 0.0], dtype=float)
    if d.size == 0:
        return float("nan"), 0.0, 0.0
    r = rankdata(np.abs(d))
    w_plus = float(r[d > 0].sum())
    w_minus = float(r[d < 0].sum())
    den = w_plus + w_minus
    return (float("nan") if den == 0 else (w_plus - w_minus) / den), w_plus, w_minus


def boot_ci(diffs: list[float], boot: int, seed: int) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob-test", required=True)
    ap.add_argument("--glob-base", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    test_files = _glob_seed_files(args.glob_test)
    base_files = _glob_seed_files(args.glob_base)
    seeds = sorted({s for s, _ in test_files} & {s for s, _ in base_files})
    if not seeds:
        raise SystemExit("no common seeds between the two file sets")

    cache = {}

    def load(kind, path):
        key = (kind, str(path))
        if key not in cache:
            cache[key] = _load(path)
        return cache[key]

    # per-seed diff vectors, in the frozen order (seed ascending, names sorted)
    per_seed: dict[int, dict[str, list[float]]] = {}
    for seed in seeds:
        t = load("test", next(p for s, p in test_files if s == seed))
        b = load("base", next(p for s, p in base_files if s == seed))
        per_seed[seed] = {
            m: [t[n][m] - b[n][m] for n in sorted(set(t) & set(b))
                if m in t[n] and m in b[n]]
            for m in metrics
        }
    pooled = {m: [x for seed in seeds for x in per_seed[seed][m]] for m in metrics}

    rows = []

    # --- pooled -------------------------------------------------------------
    p_raw = []
    for m in metrics:
        d = pooled[m]
        _s, p = wilcoxon(d) if len(d) >= 10 else (float("nan"), float("nan"))
        p_raw.append(p)
    p_holm = _holm(p_raw)
    for i, m in enumerate(metrics):
        d = pooled[m]
        rrb, wp, wm = rank_biserial(d)
        lo, hi = boot_ci(d, args.boot, args.boot_seed)
        rows.append({
            "scope": "pooled", "seeds": len(seeds), "metric": m, "n_pairs": len(d),
            "mean_delta": round(statistics.mean(d), 6),
            "median_delta": round(statistics.median(d), 6),
            "boot_ci_lo": round(lo, 6), "boot_ci_hi": round(hi, 6),
            "rank_biserial": round(rrb, 6), "w_plus": wp, "w_minus": wm,
            "p_wilcoxon": f"{p_raw[i]:.3e}",
            "p_holm_pooled6": f"{p_holm[i]:.3e}",
        })

    # --- per seed -----------------------------------------------------------
    all_p = [p for seed in seeds for p in
             [wilcoxon(per_seed[seed][m])[1] if len(per_seed[seed][m]) >= 10 else float("nan")
              for m in metrics]]
    valid_idx = [i for i, p in enumerate(all_p) if p == p]
    adj18 = {}
    if valid_idx:
        h = _holm([all_p[i] for i in valid_idx])
        adj18 = dict(zip(valid_idx, h))
    k = 0
    for seed in seeds:
        for m in metrics:
            d = per_seed[seed][m]
            _s, p = wilcoxon(d) if len(d) >= 10 else (float("nan"), float("nan"))
            rrb, wp, wm = rank_biserial(d)
            lo, hi = boot_ci(d, args.boot, args.boot_seed)
            rows.append({
                "scope": f"s{seed}", "seeds": 1, "metric": m, "n_pairs": len(d),
                "mean_delta": round(statistics.mean(d), 6),
                "median_delta": round(statistics.median(d), 6),
                "boot_ci_lo": round(lo, 6), "boot_ci_hi": round(hi, 6),
                "rank_biserial": round(rrb, 6), "w_plus": wp, "w_minus": wm,
                "p_wilcoxon": f"{p:.3e}",
                "p_holm_pooled6": "",
                "p_holm_18": f"{adj18.get(k, float('nan')):.3e}" if k in adj18 else "nan",
            })
            k += 1

    print(f"[effects] tag={args.tag} seeds={seeds} metrics={metrics} "
          f"bootstrap B={args.boot} seed={args.boot_seed}")
    print()
    print(f"{'scope':<9}{'metric':<7}{'n':>5}{'mean D':>11}{'95% CI':>23}"
          f"{'r_rb':>9}{'p':>12}{'p_holm':>12}")
    for r in rows:
        ph = r.get("p_holm_pooled6") or r.get("p_holm_18", "")
        print(f"{r['scope']:<9}{r['metric']:<7}{r['n_pairs']:>5}{r['mean_delta']:>+11.4f}"
              f"{('[' + format(r['boot_ci_lo'], '+.4f') + ', ' + format(r['boot_ci_hi'], '+.4f') + ']'):>23}"
              f"{r['rank_biserial']:>+9.4f}{r['p_wilcoxon']:>12}{ph:>12}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["scope", "seeds", "metric", "n_pairs", "mean_delta", "median_delta",
                "boot_ci_lo", "boot_ci_hi", "rank_biserial", "w_plus", "w_minus",
                "p_wilcoxon", "p_holm_pooled6", "p_holm_18"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n[effects] wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
