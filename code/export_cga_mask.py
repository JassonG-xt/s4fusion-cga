#!/usr/bin/env python3
"""GATE-A follow-up (2026-09-18): dump CGA's own learned conflict map M.

Why: R1 compared CGA against four parameter-free commitment rules, but all four were
applied through the *heuristic* mask (edge-p85 + dilate2), whereas CGA acts through
its own learned map M. That leaves one alternative explanation open: maybe a free
rule, applied on CGA's own support, would match CGA -- in which case the learned
selector would add nothing over a free rule and only the mask would matter.

This dumps M once so the CPU builder can apply free rules on exactly the support CGA
uses, which closes that explanation.

Alignment: the forward pass pads to a valid size and crops the output to the original
(top-left), exactly as gen_cga_fused.py does when it produces the fused images the
detector sees. M is cropped the same way, so it is pixel-aligned with the B0 and CGA
fused images.

Output: code/results/cga_mask/<sample_id>.png, 8-bit grey with M scaled by 255.

Run inside WSL:
  /mnt/e/.../code/.venv-brss/bin/python export_cga_mask.py \
      --checkpoint /mnt/e/.../code/checkpoints/abl_CGA_str_s42.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device
from evaluate_brss import _load_pair, _pad_to_valid

CODE = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code")
MANIFEST = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/manifests/m3fd_test.csv")
OUT = CODE / "results" / "cga_mask"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"ids: {len(rows)}", flush=True)

    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none", gate_type="scalar",
        dynamic_gate_hidden=16, global_op="scan", boundary_mode="none", boundary_hidden=16,
        use_cga=True, cga_hidden=16)).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    gate = float(torch.tanh(model.vmunet.cga.gate_scale).item())
    print(f"trained gate tanh(gate_scale) = {gate:.6f}", flush=True)

    droot = Path(args.data_root)
    frac = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            ir, vi, *_ = _load_pair(droot, row)
            ir, vi = ir.to(device), vi.to(device)
            ip, vp, h0, w0 = _pad_to_valid(ir, vi)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _f, aux = model(ip, vp, return_aux=True)
            m = aux["cga"]["conflict_pred"][..., :h0, :w0].squeeze().float().cpu().numpy()
            frac.append(float((m > 0.5).mean()))
            Image.fromarray((np.clip(m, 0, 1) * 255).round().astype(np.uint8)).save(
                OUT / f"{row['sample_id']}.png")
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)

    print(f"[mask] wrote {len(rows)} masks -> {OUT}")
    print(f"[mask] P(M>0.5) mean = {np.mean(frac)*100:.4f}%  "
          f"min {np.min(frac)*100:.4f}%  max {np.max(frac)*100:.4f}%")
    print(f"[mask] gate for the builder: --gate {gate:.6f}")


if __name__ == "__main__":
    main()
