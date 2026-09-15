"""A5: cross-seed aggregation for the K5 robustness table.

Reads per-image metric CSVs produced by evaluate_brss.py / eval_metrics.py
(one CSV per cell per seed) and produces:
  1. mean +/- std per seed (the per-seed robustness table)
  2. pooled per-image paired Wilcoxon (CGA vs B0, all seeds' image pairs
     concatenated, one sample = one image) + Holm correction across metrics
  3. seed-direction consistency: how many seeds reproduce the sign of the
     pooled delta (reported, NOT tested — n=3 seed-level Wilcoxon is
     meaningless, min p = 0.25)

Deliberately does NOT do seed-level statistics (n=3). Pooled-image Wilcoxon
treats images as the experimental unit, matching the protocol's per-image
pairing rule (机制消融实验协议.md §5).

Usage:
  .venv-brss/bin/python analyze_seeds.py \
      --cga "results/arb/CGA_str_s42/B0cmp_vs/metrics.csv=42,..." \
      --b0  "results/.../B0cmp_s42.csv=42,..."
Simpler mode when files follow results/seeds/<cell>_s<seed>.csv:
  .venv-brss/bin/python analyze_seeds.py --glob-cga 'results/seeds/CGA_str_s*.csv' \
      --glob-b0 'results/seeds/B0_s*.csv' \
      [--metrics EN,SF,AG,MI,QABF,VIF] [--out results/seeds/seeds_table.md]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
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
        prev = max(prev, val)  # step-down monotonicity
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob-cga", required=True)
    ap.add_argument("--glob-b0", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--out", default="results/seeds/seeds_table.md")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    cga_files = _glob_seed_files(args.glob_cga)
    b0_files = _glob_seed_files(args.glob_b0)
    seeds = sorted({s for s, _ in cga_files} & {s for s, _ in b0_files})
    if not seeds:
        raise SystemExit("no common seeds between CGA and B0 file sets")

    # 1. per-seed mean/std
    table = {}
    for seed in seeds:
        cga = _load(next(p for s, p in cga_files if s == seed))
        b0 = _load(next(p for s, p in b0_files if s == seed))
        names = sorted(set(cga) & set(b0))
        row = {"n": len(names)}
        for m in metrics:
            d = [cga[n][m] - b0[n][m] for n in names if m in cga[n] and m in b0[n]]
            row[m] = (statistics.mean(d), statistics.pstdev(d) if len(d) > 1 else 0.0)
        table[seed] = row

    # 2. pooled per-image Wilcoxon + Holm
    pooled_rows: list[tuple[str, float, float, float, float]] = []
    pvals = []
    for m in metrics:
        diffs: list[float] = []
        for seed in seeds:
            cga = _load(next(p for s, p in cga_files if s == seed))
            b0 = _load(next(p for s, p in b0_files if s == seed))
            for n in sorted(set(cga) & set(b0)):
                if m in cga[n] and m in b0[n]:
                    diffs.append(cga[n][m] - b0[n][m])
        if len(diffs) < 10:
            pooled_rows.append((m, float("nan"), 0.0, float("nan"), float("nan")))
            pvals.append(float("nan"))
            continue
        stat, p = wilcoxon(diffs)
        pooled_rows.append((m, statistics.mean(diffs), statistics.median(diffs), len(diffs), p))
        pvals.append(p)
    valid = [p for p in pvals if p == p]
    adj_map = dict(zip([i for i, p in enumerate(pvals) if p == p], _holm(valid))) if valid else {}
    adj = [adj_map.get(i, float("nan")) for i in range(len(pvals))]

    # 3. seed-direction consistency per metric
    lines = ["# Cross-seed robustness table (pooled per-image pairing)", "",
             "## Per-seed mean delta (CGA - B0), mean +/- std of per-image deltas", "",
             "| seed | n | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 2) + "|"]
    for seed in seeds:
        cells = [f"{table[seed][m][0]:+.4f}±{table[seed][m][1]:.4f}" for m in metrics]
        lines.append(f"| {seed} | {table[seed]['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Pooled paired Wilcoxon (image = unit) + Holm", "",
              "| metric | mean_delta | median_delta | n | p | p_holm | sign-consistent seeds |", "|---" * 7 + "|"]
    for (m, mean_d, med_d, n, p), p_h in zip(pooled_rows, adj):
        signs = []
        for seed in seeds:
            if m in table[seed]:
                signs.append(1 if table[seed][m][0] > 0 else (-1 if table[seed][m][0] < 0 else 0))
        pos = signs.count(1)
        neg = signs.count(-1)
        dirn = f"{pos}+/{neg}-" if (pos or neg) else "n/a"
        lines.append(f"| {m} | {mean_d:+.4f} | {med_d:+.4f} | {int(n)} | {p:.2e} | {p_h:.2e} | {dirn} |")
    lines += ["",
              "*Seed-level tests intentionally omitted (n=3, min Wilcoxon p = 0.25);*",
              "*direction consistency is reported descriptively.*"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
