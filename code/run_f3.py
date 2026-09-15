"""F3 cross-scale structural-drift test (D1's real claim).

Fuses each image at several native (untiled) resolutions, downsamples every
result back to the lowest-resolution reference, and measures 1 - SSIM against the
reference fusion. Downsampling strips the extra HR detail, so what remains is the
GLOBAL coarse structure: a scale-consistent model keeps low drift as resolution
(sequence length) grows; if selective-scan state dilutes, drift climbs. D1
predicts cell S drifts less than B0 at high resolution.

Aspect ratio is preserved (longer side = scale). One cell per invocation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device
from evaluate_brss import _load_pair, _valid_model_dim_at_least


def _resize_valid(x, long_side):
    h, w = x.shape[-2:]
    f = long_side / max(h, w)
    th = _valid_model_dim_at_least(round(h * f))
    tw = _valid_model_dim_at_least(round(w * f))
    return F.interpolate(x, (th, tw), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--boundary-mode", required=True, choices=["none", "residual", "state", "full"])
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--scales", default="193,289,385,517,649,769")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--use-cga", action="store_true",
                    help="checkpoint was trained with image-space CGA (train_brss.py --use-cga)")
    ap.add_argument("--cga-hidden", type=int, default=16)
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    ref_scale = scales[0]
    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none",
        gate_type="scalar", dynamic_gate_hidden=16, global_op="scan",
        boundary_mode=args.boundary_mode, boundary_hidden=16,
        use_cga=args.use_cga, cga_hidden=args.cga_hidden,
    )).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)

    with Path(args.manifest).open(encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    if args.limit > 0:
        rows = rows[:args.limit]

    out = []
    droot = Path(args.data_root)
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            ir, vi, *_ = _load_pair(droot, row)
            ir, vi = ir.to(device), vi.to(device)
            fused = {}
            for s in scales:
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    fs = model(_resize_valid(ir, s), _resize_valid(vi, s))
                fused[s] = fs.float().clamp(0, 1)
            ref = fused[ref_scale]
            rh, rw = ref.shape[-2:]
            ref_np = ref.squeeze().cpu().numpy()
            rec = {"dataset": row["dataset"], "sample_id": row["sample_id"]}
            for s in scales[1:]:
                ds = F.interpolate(fused[s], (rh, rw), mode="area").squeeze().cpu().numpy()
                rec[f"drift_{s}"] = float(1.0 - ssim(ds, ref_np, data_range=1.0))
            out.append(rec)
            if i % 20 == 0:
                print(f"[{i}/{len(rows)}] {row['sample_id']}", flush=True)

    fields = ["dataset", "sample_id"] + [f"drift_{s}" for s in scales[1:]]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {args.out_csv} ({len(out)} images)", flush=True)


if __name__ == "__main__":
    main()
