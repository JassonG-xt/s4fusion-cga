#!/usr/bin/env python3
"""C3 reported on the SAME metric implementation as the main table.

The frozen gate `struct_noninferior.py` implements its own AG (mean of
sqrt((gx^2+gy^2)/2)), which is smaller than `eval_metrics.py`'s AG by roughly a
constant factor. The two implementations agree on every C3 verdict in this
paper, but they do not agree on the boundary magnitudes, and the review asked
for the unrounded value and an interval. This script therefore reports C3 on the
`eval_metrics.py` per-image CSVs --- the implementation behind Table 3 --- with:
  * the relative difference at full precision,
  * a one-sided 95% lower bound from a paired bootstrap over images,
  * the verdict against the inclusive -1% margin.

Run with any Python that has numpy (the seed CSVs are already computed).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEEDS = Path(__file__).resolve().parent / 'results/seeds'
PAIRS = [
    ('matched  s42  ', 'B0_e32_s42.csv', 'CGA_str_s42.csv'),
    ('matched  s123 ', 'B0_e32_s123.csv', 'CGA_str_s123.csv'),
    ('matched  s3407', 'B0_e32_s3407.csv', 'CGA_str_s3407.csv'),
    ('single   s42  ', 'B0_s42.csv', 'CGA_str_s42.csv'),
    ('single   s123 ', 'B0_s123.csv', 'CGA_str_s123.csv'),
    ('single   s3407', 'B0_s3407.csv', 'CGA_str_s3407.csv'),
]
B, SEED = 2000, 20260920


def load(name):
    rows = [r for r in csv.DictReader((SEEDS / name).open(encoding='utf-8'))
            if r['name'] not in ('__mean__', '__std__')]
    return {r['name']: r for r in rows}


def main():
    rng = np.random.default_rng(SEED)
    print('C3 measured on eval_metrics.py per-image values (same implementation as Table 3)')
    print('margin: relative SF/AG change >= -1%, threshold inclusive\n')
    for label, bf, tf in PAIRS:
        b_rows, t_rows = load(bf), load(tf)
        ids = sorted(set(b_rows) & set(t_rows))
        n = len(ids)
        for m in ('SF', 'AG'):
            b = np.array([float(b_rows[i][m]) for i in ids])
            t = np.array([float(t_rows[i][m]) for i in ids])
            rel = (t.mean() - b.mean()) / b.mean()
            idx = rng.integers(0, n, size=(B, n))
            relb = (t[idx].mean(axis=1) - b[idx].mean(axis=1)) / b[idx].mean(axis=1)
            lo = float(np.percentile(relb, 5))
            print(f'{label} {m}: base={b.mean():.6f} test={t.mean():.6f} '
                  f'rel={rel*100:+.6f}%  one-sided-95%-lower={lo*100:+.6f}%  '
                  f'{"PASS" if rel >= -0.01 else "FAIL"} '
                  f'(margin {abs(rel + 0.01) * 100:.6f} pp)  n={n}')


if __name__ == '__main__':
    main()
