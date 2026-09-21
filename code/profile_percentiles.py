#!/usr/bin/env python3
"""Per-iteration latency percentiles for the CGA cost table (review item: S1).

The frozen profiler (`profile_brss_complexity.py`) times a block of `iters`
passes and divides, so it can report a mean and a spread across repeats but not
a per-iteration distribution. This script measures the same pipeline with the
same protocol --- 10 warm-up passes, then 3 repeats x 30 timed passes, device
synchronised --- but times each pass individually, which additionally yields
P50 and P95. It writes a separate CSV and does not touch the frozen one.

Run inside WSL with .venv-brss/bin/python.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device

ROOT = Path('/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026')
CK = ROOT / 'code/checkpoints'


def measure(model, device, h, w, warmup, iters, repeats):
    x = torch.randn(1, 1, h, w, device=device)
    y = torch.randn(1, 1, h, w, device=device)
    model.eval()
    per_iter = []
    peaks = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x, y)
        for _ in range(repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(iters):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x, y)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                per_iter.append((time.perf_counter() - t0) * 1000.0)
            if device.type == 'cuda':
                peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    a = np.array(per_iter)
    return {
        'ms_mean': float(a.mean()),
        'ms_p50': float(np.percentile(a, 50)),
        'ms_p95': float(np.percentile(a, 95)),
        'ms_min': float(a.min()),
        'ms_max': float(a.max()),
        'ms_std_pooled': float(a.std(ddof=1)),
        'peak_mem_mb': float(np.mean(peaks)) if peaks else float('nan'),
        'n_samples': int(a.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-csv', default=str(ROOT / 'code/results/complexity_percentiles.csv'))
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    device = _resolve_device('auto')
    print(f'device = {device}')
    if device.type == 'cuda':
        print(f'gpu    = {torch.cuda.get_device_name(0)}')
        print(f'torch  = {torch.__version__}  cuda = {torch.version.cuda}  '
              f'cudnn = {torch.backends.cudnn.version()}')

    cells = [
        ('official', str(ROOT / '../../S4Fusion-main/model/model.pkl'), False),
        ('B0', str(CK / 'abl_B0_s42.pt'), False),
        ('CGA', str(CK / 'abl_CGA_str_s42.pt'), True),
    ]
    sizes = [(507, 507), (1034, 769)]

    rows = []
    for tag, ckpt, use_cga in cells:
        model = build_model(SimpleNamespace(
            depths='1,2,1', teacher_depths='1,2,1', fusion_scales='none',
            gate_type='scalar', dynamic_gate_hidden=16, global_op='scan',
            boundary_mode='none', boundary_hidden=16, use_cga=use_cga,
            cga_hidden=16)).to(device).eval()
        if ckpt and Path(ckpt).is_file():
            load_checkpoint(model, ckpt, device)
        for (h, w) in sizes:
            r = measure(model, device, h, w, args.warmup, args.iters, args.repeats)
            row = {'tag': tag, 'checkpoint': Path(ckpt).name, 'use_cga': use_cga,
                   'height': h, 'width': w, 'device': str(device), **r}
            rows.append(row)
            print(f"[{tag}] {h}x{w}: mean {r['ms_mean']:.2f}  P50 {r['ms_p50']:.2f}  "
                  f"P95 {r['ms_p95']:.2f} ms  (n={r['n_samples']})  "
                  f"peak {r['peak_mem_mb']:.2f} MB", flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print('wrote', out)

    print('\nCGA incremental latency (P50 / P95):')
    for (h, w) in sizes:
        d = {r['tag']: r for r in rows if r['height'] == h}
        for k in ('p50', 'p95'):
            print(f'  {h}x{w} {k}: CGA {d["CGA"]["ms_" + k]:.2f} vs '
                  f'B0 {d["B0"]["ms_" + k]:.2f} ms  ->  '
                  f'{(d["CGA"]["ms_" + k] / d["B0"]["ms_" + k] - 1) * 100:+.2f}%')


if __name__ == '__main__':
    main()
