"""Generate fused images (90 M3FD detection ids) for a checkpoint, as RGB for the
detector. Used to A/B the trained CGA vs its same-regime baseline on detection."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device
from evaluate_brss import _pad_to_valid, _load_pair, _apply_cga_conflict_override

OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/results_metrics/downstream_detection")
ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--use-cga", action="store_true")
    ap.add_argument("--gate-override", type=float, default=-1.0,
                    help="if >=0, set tanh(gate_scale) to this value on the trained heads (strength sweep)")
    ap.add_argument("--cga-conflict-override", choices=["none", "uniform", "shuffle"], default="none",
                    help="H5 falsification: override the CGA conflict map at inference "
                         "(uniform=all-ones, shuffle=fixed spatial permutation)")
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--manifest", default="../dataset/manifests/m3fd_test.csv")
    ap.add_argument("--full", action="store_true",
                    help="use ALL manifest ids (300) instead of legacy 90-image detection split")
    args = ap.parse_args()

    ids = set(f.stem for f in (OLD / "datasets/m3fd_baseline/test/labels").glob("*.txt"))
    if args.full:
        import csv as _csv
        with open(args.manifest, encoding="utf-8-sig") as h:
            ids = {r["sample_id"] for r in _csv.DictReader(h)}
        print(f"FULL mode: {len(ids)} ids from manifest", flush=True)
    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none", gate_type="scalar",
        dynamic_gate_hidden=16, global_op="scan", boundary_mode="none", boundary_hidden=16,
        use_cga=args.use_cga, cga_hidden=16)).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    _apply_cga_conflict_override(model, args.cga_conflict_override)

    if args.use_cga and args.gate_override >= 0:
        import math
        g = min(0.999, max(0.0, args.gate_override))
        with torch.no_grad():
            model.vmunet.cga.gate_scale.fill_(math.atanh(g))
        print(f"gate override: tanh(gate)={g}", flush=True)

    out = ARB / args.variant / "images"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.manifest, encoding="utf-8-sig") as h:
        rows = [r for r in csv.DictReader(h) if r["sample_id"] in ids]
    droot = Path(args.data_root)
    n = 0
    with torch.inference_mode():
        for row in rows:
            ir, vi, *_ = _load_pair(droot, row)
            ir, vi = ir.to(device), vi.to(device)
            ip, vp, h0, w0 = _pad_to_valid(ir, vi)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                fused = model(ip, vp)[..., :h0, :w0]
            g = (fused.squeeze().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(np.stack([g] * 3, -1)).save(out / f"{row['sample_id']}.png")
            n += 1
    print(f"{args.variant}: wrote {n} fused images to {out}", flush=True)


if __name__ == "__main__":
    main()
