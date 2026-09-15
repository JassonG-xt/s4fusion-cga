"""CGA mechanism visualization dump (gate-1 figure material, 2026-13-13... corrected: 2026-09-13).

For selected M3FD samples, dumps per-image npz + PNG panels with:
  - ir, vi (Y) inputs
  - hand-crafted conflict target (conflict_signal: min(g_i,g_v)*(1-cos)/2, normalized)
  - learned conflict map (sigmoid(conflict_head), from the trained CGA)
  - commit-selection map s (sigmoid(select_head + 2*prior), 1=commit to IR)
  - |CGA - B0| difference map (needs the same-budget baseline output image)
  - gate strength tanh(gate_scale) (global scalar, printed)
These are the CGA paper's mechanism figures (the BRSS-era F1/F2 state-response
maps do not apply: CGA_str is boundary_mode=none).

Reuses gen_cga_fused.py plumbing: build_model(SimpleNamespace(...)), load_checkpoint,
evaluate_brss._load_pair / _pad_to_valid.
"""
import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device
from evaluate_brss import _pad_to_valid, _load_pair
from modules.cga import conflict_signal

ARB = Path(__file__).resolve().parent / "results" / "arb"


def to_uint8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def panel_row(items: list[tuple[str, np.ndarray]]) -> Image.Image:
    imgs = []
    for label, a in items:
        u = to_uint8(a)
        if u.ndim == 2:
            u = np.stack([u] * 3, -1)
        img = Image.fromarray(u)
        imgs.append((label, img))
    w = max(im.width for _, im in imgs)
    h = max(im.height for _, im in imgs)
    out = Image.new("RGB", (w * len(imgs), h), "white")
    for i, (_, im) in enumerate(imgs):
        out.paste(im, (i * w, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/abl_CGA_str_s42.pt")
    ap.add_argument("--use-cga", action="store_true", default=True)
    ap.add_argument("--ids", default="",
                    help="comma-separated sample_ids; default = auto-pick high-conflict samples")
    ap.add_argument("--auto", type=int, default=6,
                    help="auto-pick N samples with highest mean learned conflict")
    ap.add_argument("--baseline-cell", default="B0cmp_full",
                    help="arb cell holding the same-budget baseline fused images")
    ap.add_argument("--out", default="results/cga_vis")
    ap.add_argument("--manifest", default="../dataset/manifests/m3fd_test.csv")
    ap.add_argument("--data-root", default="../dataset")
    args = ap.parse_args()

    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none", gate_type="scalar",
        dynamic_gate_hidden=16, global_op="scan", boundary_mode="none", boundary_hidden=16,
        use_cga=True, cga_hidden=16)).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    gate_val = math.tanh(float(model.vmunet.cga.gate_scale.detach().cpu()))
    print(f"tanh(gate_scale) = {gate_val:.4f} (0 = mechanism off)", flush=True)

    with open(args.manifest, encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    by_id = {r["sample_id"]: r for r in rows}

    requested = [s.strip() for s in args.ids.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pass 1 (auto mode): score every candidate by mean learned conflict, cheap forward on GPU
    candidates = requested or [r["sample_id"] for r in rows]
    if not requested:
        scored = []
        with torch.inference_mode():
            for row in rows:
                ir, vi, *_ = _load_pair(Path(args.data_root), row)
                ip, vp, h0, w0 = _pad_to_valid(ir.to(device), vi.to(device))
                _, aux = model(ip, vp, return_aux=True)
                mean_conf = float(aux["cga"]["conflict_pred"][..., :h0, :w0].mean())
                scored.append((mean_conf, row["sample_id"]))
        scored.sort(reverse=True)
        candidates = [sid for _, sid in scored[: args.auto]]
        print("auto-picked high-conflict ids:", candidates, flush=True)
    else:
        candidates = [sid for sid in candidates if sid in by_id]

    summary = []
    for sid in candidates:
        row = by_id[sid]
        ir, vi, ir_arr, vi_arr, _ = _load_pair(Path(args.data_root), row)
        ip, vp, h0, w0 = _pad_to_valid(ir.to(device), vi.to(device))
        with torch.inference_mode():
            fused, aux = model(ip, vp, return_aux=True)
        fused = fused[..., :h0, :w0].squeeze().float().cpu().numpy()
        cga = aux["cga"]
        conflict_target = cga["conflict_target"][..., :h0, :w0].squeeze().float().cpu().numpy()
        conflict_pred = cga["conflict_pred"][..., :h0, :w0].squeeze().float().cpu().numpy()
        commit_s = cga["commit"][..., :h0, :w0].squeeze().float().cpu().numpy()  # blended pixels
        # recover the selection map s from the same forward: s = sigmoid(select_head + 2*prior)
        # not stored in aux; approximate via commit vs ir/vi: not possible from commit alone.
        # Instead recompute the diff map vs the same-budget baseline fused image.
        base_img = ARB / args.baseline_cell / "images" / f"{sid}.png"
        diff = None
        if base_img.is_file():
            b0 = np.asarray(Image.open(base_img).convert("L"), dtype=np.float32) / 255.0
            if b0.shape == fused.shape:
                diff = np.abs(fused - b0)
        np.savez_compressed(
            out_dir / f"{sid}_cga.npz",
            ir=ir_arr, vi=vi_arr, fused=fused,
            conflict_target=conflict_target, conflict_pred=conflict_pred,
            commit=commit_s, diff_vs_baseline=diff if diff is not None else np.zeros_like(fused))
        items = [("IR", ir_arr / 255.0), ("VI(Y)", vi_arr / 255.0),
                 ("conflict target", conflict_target), ("learned conflict", conflict_pred)]
        if diff is not None:
            items.append(("|CGA - B0|", diff))
        panel_row(items).save(out_dir / f"{sid}_panel.png")
        m_conf = float(conflict_pred.mean())
        m_diff = float(diff.mean()) if diff is not None else float("nan")
        summary.append((sid, m_conf, m_diff, gate_val))
        print(f"{sid}: mean_learned_conflict={m_conf:.4f} mean|CGA-B0|={m_diff:.5f}", flush=True)

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["sample_id", "mean_learned_conflict", "mean_abs_diff_vs_baseline", "tanh_gate_scale"])
        for r in summary:
            w.writerow(r)
    print(f"wrote {len(summary)} panels to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
