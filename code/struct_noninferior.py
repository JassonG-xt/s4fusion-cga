#!/usr/bin/env python3
"""Structural non-inferiority check (pre-registered gate C3): fused images of a
CGA variant vs its same-regime baseline, paired per image on common ids.
Reports EN/SF/AG means, paired Wilcoxon p, and gates at a 1% relative
non-inferiority margin on SF and AG.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
TEST, BASE = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("CGA_str", "B0cmp_full")


def metrics(img: np.ndarray) -> tuple:
    f = img.astype(np.float64)
    h, w = f.shape
    dx = np.diff(f, axis=1)
    dy = np.diff(f, axis=0)
    rf = np.sqrt((dx ** 2).sum() / (h * (w - 1)))
    cf = np.sqrt((dy ** 2).sum() / ((h - 1) * w))
    sf = float(np.sqrt(rf ** 2 + cf ** 2))
    gy, gx = np.gradient(f, axis=0), np.gradient(f, axis=1)
    ag = float(np.sqrt((gx[1:, 1:] ** 2 + gy[1:, 1:] ** 2) / 2).mean())
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p[p > 0]
    en = float(-(nz * np.log2(nz)).sum())
    return en, sf, ag


def load_dir(variant: str) -> dict:
    out = {}
    for f in sorted((ARB / variant / "images").glob("*.png")):
        out[f.stem] = metrics(np.asarray(Image.open(f).convert("L")))
    return out


def main():
    T, B = load_dir(TEST), load_dir(BASE)
    ids = sorted(set(T) & set(B))
    if len(ids) < 30:
        print(f"[gate-C3] FAIL: only {len(ids)} common images")
        return
    arr = {k: np.array([[T[i][j], B[i][j]] for i in ids]) for j, k in enumerate(["EN", "SF", "AG"])}
    print(f"[gate-C3] paired images: {len(ids)}")
    ok = True
    for k, v in arr.items():
        t, b = v[:, 0], v[:, 1]
        delta = float(t.mean() - b.mean())
        rel = delta / b.mean()
        try:
            p = stats.wilcoxon(t, b).pvalue
        except ValueError:
            p = 1.0
        ni = rel >= -0.01
        ok &= ni or k == "EN"
        print(f"  {k}: {b.mean():.4f} -> {t.mean():.4f} ({delta:+.4f}, {rel * 100:+.2f}%)  wilcoxon p={p:.3e}  "
              f"non-inferior(>=-1%): {'PASS' if ni else 'FAIL'}")
    print(f"[gate-C3] SF/AG non-inferiority: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
