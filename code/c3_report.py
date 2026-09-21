#!/usr/bin/env python3
"""C3 boundary report: unrounded relative values and one-sided intervals.

The pre-registered structural gate C3 is "SF/AG non-inferior within -1%", and
the review's objection is that the frozen script prints the relative value to
two decimal places, so a row displayed as exactly -1.00% cannot be resolved
against an inclusive threshold. This script re-uses `struct_noninferior.py`'s
own metric implementation and per-image loader (imported, not copied) and adds:
  * the relative difference at full float precision,
  * a one-sided 95% lower confidence bound on the relative difference, from a
    paired bootstrap over images,
  * the explicit verdict against the inclusive -1% margin at that precision.

Nothing here changes the frozen gate: the gate's own printed verdict remains
what struct_noninferior.py reports, and this script only makes the boundary
legible.

Usage (inside WSL, .venv-brss/bin/python), pairs are TEST BASE:
    python c3_report.py CGA_str B0cmp_full
"""
from __future__ import annotations

import sys

import numpy as np

import struct_noninferior as S

B = 2000


def one_sided(rel_samples):
    """95% one-sided lower bound = 5th percentile of the bootstrap distribution."""
    return float(np.percentile(rel_samples, 5))


def report(test: str, base: str) -> None:
    t_all, b_all = S.load_dir(test), S.load_dir(base)
    ids = sorted(set(t_all) & set(b_all))
    rng = np.random.default_rng(20260920)
    print(f"\n=== C3 boundary: {test} vs {base}  ({len(ids)} paired images) ===")
    for k, idx in (("SF", 1), ("AG", 2)):
        b = np.array([b_all[i][idx] for i in ids], float)
        t = np.array([t_all[i][idx] for i in ids], float)
        rel = (t.mean() - b.mean()) / b.mean()
        n = len(ids)
        boot = rng.integers(0, n, size=(B, n))
        rel_boot = (t[boot].mean(axis=1) - b[boot].mean(axis=1)) / b[boot].mean(axis=1)
        lo = one_sided(rel_boot)
        tight = float((t / b).mean() - 1.0)
        print(f"  {k}: base={b.mean():.8f}  test={t.mean():.8f}  "
              f"delta={t.mean() - b.mean():+.8f}")
        print(f"      relative (mean-of-ratios) = {tight * 100:+.6f}%")
        print(f"      relative (ratio-of-means) = {rel * 100:+.6f}%")
        print(f"      one-sided 95% lower bound on ratio-of-means = {lo * 100:+.6f}%")
        print(f"      gate (-1.00% inclusive) = "
              f"{'PASS' if rel >= -0.01 else 'FAIL'}   "
              f"(margin {abs(-0.01 - rel) * 100:.6f} pp)")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        report(sys.argv[1], sys.argv[2])
    else:
        for pair in (("CGA_str", "B0cmp_full"),
                     ("CGA_str", "B0_e32_s42"),
                     ("CGA_str_s123", "B0_e32_s123"),
                     ("CGA_str_s3407", "B0_e32_s3407")):
            report(*pair)
