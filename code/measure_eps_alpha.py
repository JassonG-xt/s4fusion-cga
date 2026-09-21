#!/usr/bin/env python3
"""Two measurements the revision needs, both read-only.

1. epsilon floor: how far the conflict prior kappa sits above zero at pixels
   where only one modality carries structure, relative to the per-image maximum
   of kappa. Quantifies the statement that kappa is *approximately*, not
   exactly, zero when one modality is edgeless.

2. alpha: the trained scalar gate of every CGA checkpoint, so its sign and range
   can be reported instead of assuming alpha >= 0.

Run inside WSL with .venv-brss/bin/python.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import diag_conflict_detection as D

ROOT = Path('/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026')
CK = ROOT / 'code/checkpoints'


def kappa_stats(n_images: int = 40):
    rows = list(csv.DictReader(
        (ROOT / 'dataset/manifests/m3fd_test.csv').open(encoding='utf-8-sig')))
    ids = sorted({r['sample_id'] for r in rows})[:n_images]
    floors, maxima, meds = [], [], []
    for sid in ids:
        ir_p = D.SRC / 'Ir' / f'{sid}.png'
        vi_p = D.SRC / 'Vis' / f'{sid}.png'
        if not (ir_p.is_file() and vi_p.is_file()):
            continue
        ir = np.asarray(Image.open(ir_p).convert('L'), float) / 255
        vi = np.asarray(Image.open(vi_p).convert('RGB')
                        .convert('YCbCr').getchannel('Y'), float) / 255
        gxi, gyi, _ = D.grad(ir)
        gxv, gyv, _ = D.grad(vi)
        # module convention: eps = 1e-6 (modules/cga.py conflict_signal),
        # NOT the diagnostic helper's 1e-12.
        eps = 1e-6
        gi = np.sqrt(gxi * gxi + gyi * gyi + eps)
        gv = np.sqrt(gxv * gxv + gyv * gyv + eps)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + eps)
        kappa = np.minimum(gi, gv) * (1 - cos) / 2
        # VI carries no structure: its magnitude sits at the epsilon floor.
        vi_floor = gv <= np.sqrt(eps) * 1.001
        one_sided = vi_floor & (gi >= 0.02)
        floor = float(kappa[one_sided].mean()) if one_sided.any() else float('nan')
        mx = float(kappa.max())
        floors.append(floor / mx if mx else np.nan)
        maxima.append(mx)
        meds.append(float(np.median(kappa)))
    return (np.nanmean(floors), np.nanmedian(maxima), np.nanmedian(meds),
            len(floors))


def alpha_of(path: Path):
    sd = torch.load(str(path), map_location='cpu')
    sd = sd.get('model', sd.get('state_dict', sd))
    out = {}
    for k, v in sd.items():
        if 'gate_scale' in k:
            val = float(v.reshape(-1)[0])
            out[k] = (val, float(np.tanh(val)))
    return out


if __name__ == '__main__':
    frac, mx, med, n = kappa_stats()
    print(f'images sampled                     = {n}')
    print(f'mean of (one-sided floor / max kappa) = {frac:.3e}')
    print(f'median per-image max kappa          = {mx:.4f}')
    print(f'median per-image median kappa       = {med:.6f}')
    print(f'analytic one-sided floor 0.5*sqrt(1e-6) = {0.5*np.sqrt(1e-6):.3e}')
    print()
    print('trained gate scalars (tanh = effective gate):')
    for p in sorted(glob.glob(str(CK / 'abl_CGA_str*.pt'))):
        name = Path(p).name
        for k, (raw, t) in alpha_of(Path(p)).items():
            print(f'  {name:28s} {k:24s} alpha={raw:+.6f}  tanh(alpha)={t:+.6f}')
