#!/usr/bin/env python3
"""Gradient probe for H5: measure dL_commit/d(gate_scale) at g=0 under the real
pipeline (AMP, real batch), to verify the loss actually pushes the gate open
and to gauge the gradient magnitude vs the main loss. Run on GPU."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device, _adjust_crop_size
from dataset_manifest import DirectoryPairDataset
from modules.cga import cga_commit_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../../S4Fusion-main/model/model.pkl")
    ap.add_argument("--manifest", default="../dataset/manifests/train_all.csv")
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none", gate_type="scalar",
        dynamic_gate_hidden=16, global_op="scan", boundary_mode="none", boundary_hidden=16,
        use_cga=True, cga_hidden=16)).to(device).train()
    load_checkpoint(model, args.checkpoint, device)

    crop = _adjust_crop_size((128, 128))
    ds = DirectoryPairDataset(Path(args.data_root), Path(args.manifest),
                              crop_size=crop, use_y=True, augment=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"{'step':>4} {'dL/dg':>12} {'gate':>10} {'L_commit':>10}")
    gs = []
    with torch.autocast(device_type="cuda", enabled=True):
        for i, (vi, ir) in enumerate(dl):
            if i >= args.steps:
                break
            ir, vi = ir.to(device), vi.to(device)
            fused, aux = model(ir, vi, return_aux=True)
            cga_aux = aux["cga"]
            if cga_aux is None:
                raise RuntimeError("no cga aux")
            g = model.vmunet.cga.gate_scale
            g.requires_grad_(True)
            model.zero_grad()
            commit_l = cga_commit_loss(cga_aux)
            commit_l.backward()
            dg_commit = g.grad.item() if g.grad is not None else float("nan")
            gs.append(dg_commit)
            g.grad = None
            print(f"{i:>4} {dg_commit:>12.3e} {float(torch.tanh(g)):>10.4f} {commit_l.item():>10.4f}")
    print("mean dL_commit/dg:", sum(gs) / len(gs))


if __name__ == "__main__":
    main()
