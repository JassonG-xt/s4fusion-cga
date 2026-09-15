"""Manifest-based inference and unified core metrics for S4Fusion/BRSS models."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from eval_metrics import evaluate_pair
from modules.boundary import BoundaryReliabilityEstimator
from train_brss import build_model, load_checkpoint
from train_gates import _parse_scales, _resolve_device


CORE_METRICS = ("EN", "SF", "AG", "MI_IR", "MI_VI", "MI", "QABF")


def _valid_model_dim_at_least(value: int) -> int:
    value = max(4, int(value))
    while ((value - 4) // 3 + 1) % 4 != 0:
        value += 1
    return value


def _pad_to_valid(ir: torch.Tensor, vi: torch.Tensor):
    height, width = ir.shape[-2:]
    target_h = _valid_model_dim_at_least(height)
    target_w = _valid_model_dim_at_least(width)
    pad_h, pad_w = target_h - height, target_w - width
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        ir = F.pad(ir, (0, pad_w, 0, pad_h), mode=mode)
        vi = F.pad(vi, (0, pad_w, 0, pad_h), mode=mode)
    return ir, vi, height, width


def _positions(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _blend_window(size: int, device, dtype):
    if size <= 2:
        return torch.ones(1, 1, size, size, device=device, dtype=dtype)
    axis = torch.hann_window(size, periodic=False, device=device, dtype=dtype).clamp_min(0.05)
    return (axis[:, None] * axis[None, :]).view(1, 1, size, size)


def _model_forward(model, ir, vi, amp: bool):
    with torch.cuda.amp.autocast(enabled=amp and ir.device.type == "cuda"):
        return model(ir, vi)


def infer_native(model, ir, vi, amp: bool):
    ir_pad, vi_pad, height, width = _pad_to_valid(ir, vi)
    return _model_forward(model, ir_pad, vi_pad, amp)[..., :height, :width]


def infer_tiled(model, ir, vi, tile_size: int, overlap: int, amp: bool):
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size")
    tile_size = _valid_model_dim_at_least(tile_size)
    stride = tile_size - overlap
    ir_pad, vi_pad, height, width = _pad_to_valid(ir, vi)
    pad_h = max(0, tile_size - ir_pad.shape[-2])
    pad_w = max(0, tile_size - ir_pad.shape[-1])
    if pad_h or pad_w:
        mode = "reflect" if ir_pad.shape[-2] > pad_h and ir_pad.shape[-1] > pad_w else "replicate"
        ir_pad = F.pad(ir_pad, (0, pad_w, 0, pad_h), mode=mode)
        vi_pad = F.pad(vi_pad, (0, pad_w, 0, pad_h), mode=mode)

    full_h, full_w = ir_pad.shape[-2:]
    rows = _positions(full_h, tile_size, stride)
    cols = _positions(full_w, tile_size, stride)
    output = torch.zeros_like(ir_pad)
    weight_sum = torch.zeros_like(ir_pad)
    window = _blend_window(tile_size, ir.device, ir.dtype)
    for top in rows:
        for left in cols:
            ir_tile = ir_pad[..., top:top + tile_size, left:left + tile_size]
            vi_tile = vi_pad[..., top:top + tile_size, left:left + tile_size]
            fused = _model_forward(model, ir_tile, vi_tile, amp)
            output[..., top:top + tile_size, left:left + tile_size] += fused * window
            weight_sum[..., top:top + tile_size, left:left + tile_size] += window
    return (output / weight_sum.clamp_min(1e-6))[..., :height, :width]


def _load_pair(data_root: Path, row):
    with Image.open(data_root / row["ir_path"]) as image:
        ir_image = image.convert("L")
        ir_array = np.asarray(ir_image, dtype=np.float32)
        ir = to_tensor(ir_image).unsqueeze(0)
    with Image.open(data_root / row["vi_path"]) as image:
        vi_rgb = image.convert("RGB")
        vi_ycbcr = vi_rgb.convert("YCbCr")
        vi_array = np.asarray(vi_ycbcr.getchannel("Y"), dtype=np.float32)
        vi = to_tensor(vi_ycbcr.getchannel("Y")).unsqueeze(0)
    return ir, vi, ir_array, vi_array, vi_rgb


def _save_outputs(output_dir: Path, row, fused_uint8: np.ndarray, visible_rgb: Image.Image, save_rgb: bool):
    dataset_dir = output_dir / row["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    gray_path = dataset_dir / f"{row['sample_id']}.png"
    Image.fromarray(fused_uint8, mode="L").save(gray_path)
    rgb_path = ""
    if save_rgb:
        ycbcr = visible_rgb.convert("YCbCr")
        fused_y = Image.fromarray(fused_uint8, mode="L")
        fused_rgb = Image.merge("YCbCr", (fused_y, ycbcr.getchannel("Cb"), ycbcr.getchannel("Cr"))).convert("RGB")
        rgb_path_obj = dataset_dir / f"{row['sample_id']}_rgb.png"
        fused_rgb.save(rgb_path_obj)
        rgb_path = str(rgb_path_obj)
    return str(gray_path), rgb_path


def _write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "sample_id", "width", "height", "seconds", "inference_mode", "fused_path", "rgb_path", *CORE_METRICS]
    numeric = ["seconds", *CORE_METRICS]
    mean_row = {"dataset": "__mean__", "sample_id": "__mean__"}
    std_row = {"dataset": "__std__", "sample_id": "__std__"}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        mean_row[key] = float(values.mean())
        std_row[key] = float(values.std())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)


def _apply_reliability_override(model, mode: str) -> int:
    """H3 hook: force every boundary estimator to a content-free reliability."""
    if mode == "none":
        return 0
    count = 0
    for module in model.modules():
        if isinstance(module, BoundaryReliabilityEstimator):
            module.reliability_override = mode
            count += 1
    if count == 0:
        print("warning: --reliability-override set but the model has no boundary "
              "estimator (boundary-mode=none); ignoring", flush=True)
    else:
        print(f"reliability-override={mode} applied to {count} estimator(s)", flush=True)
    return count


def _apply_cga_conflict_override(model, mode: str) -> int:
    """H5 hook: force CGA conflict map to uniform or spatially shuffled."""
    if mode == "none":
        return 0
    from modules.cga import ConflictGatedArbitration
    count = 0
    for module in model.modules():
        if isinstance(module, ConflictGatedArbitration):
            module.conflict_override = mode
            count += 1
    if count == 0:
        print("warning: --cga-conflict-override set but model has no CGA module "
              "(use-cga=false); ignoring", flush=True)
    else:
        print(f"cga-conflict-override={mode} applied to {count} CGA module(s)", flush=True)
    return count


def _dump_learned_scales(path: Path, model) -> int:
    """Dump the trained tanh(gate) magnitudes (protocol figure F5).

    A near-zero scale means the mechanism stayed off after training, which would
    self-falsify the "mechanism drives the gain" claim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["block", "state_scale_ir_vi", "state_scale_shared", "state_scale_delta",
              "residual_scale_ir", "residual_scale_vi"]
    rows = []
    index = 0
    for module in model.modules():
        state = getattr(module, "boundary_state_scale", None)
        if state is None:
            continue
        residual = getattr(module, "boundary_residual_scale", None)
        state_v = torch.tanh(state.detach()).flatten().cpu().tolist()
        residual_v = (torch.tanh(residual.detach()).flatten().cpu().tolist()
                      if residual is not None else [float("nan"), float("nan")])
        rows.append({
            "block": index,
            "state_scale_ir_vi": state_v[0],
            "state_scale_shared": state_v[1],
            "state_scale_delta": state_v[2],
            "residual_scale_ir": residual_v[0],
            "residual_scale_vi": residual_v[1],
        })
        index += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} boundary blocks)", flush=True)
    return len(rows)


def _infer_boundary_aux(model, ir, vi, amp: bool):
    """Extra native forward that returns the per-block reliability maps.

    Runs at native padded resolution regardless of the metric tiling; keep it
    for a bounded set of images (use --limit) since it duplicates the forward.
    """
    ir_pad, vi_pad, _, _ = _pad_to_valid(ir, vi)
    with torch.cuda.amp.autocast(enabled=amp and ir.device.type == "cuda"):
        _, aux = model(ir_pad, vi_pad, return_aux=True)
    return aux["boundary"]


def _dump_boundary_maps(dump_dir: str, row, boundary_list) -> None:
    """Save per-block reliability maps for the state-response figures F1-F4."""
    if not boundary_list:
        return
    dataset_dir = Path(dump_dir) / row["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for block_index, aux in enumerate(boundary_list):
        ir_edge = aux.ir_edge[0].detach().float().cpu().numpy()
        vi_edge = aux.vi_edge[0].detach().float().cpu().numpy()
        payload[f"block{block_index}_weights"] = aux.weights[0].detach().float().cpu().numpy()
        payload[f"block{block_index}_boundary"] = aux.boundary[0].detach().float().cpu().numpy()
        payload[f"block{block_index}_confidence"] = aux.confidence[0].detach().float().cpu().numpy()
        payload[f"block{block_index}_ir_edge"] = ir_edge
        payload[f"block{block_index}_vi_edge"] = vi_edge
        payload[f"block{block_index}_conflict"] = np.abs(ir_edge - vi_edge)
    np.savez_compressed(dataset_dir / f"{row['sample_id']}_boundary.npz", **payload)


def main():
    parser = argparse.ArgumentParser(description="Evaluate S4Fusion/BRSS checkpoints from a CSV manifest")
    parser.add_argument("--data-root", default="../dataset")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--depths", default="1,2,1")
    parser.add_argument("--teacher-depths", default="1,2,1")
    parser.add_argument("--fusion-scales", default="4,8")
    parser.add_argument("--gate-type", choices=["scalar", "dynamic"], default="dynamic")
    parser.add_argument("--dynamic-gate-hidden", type=int, default=16)
    parser.add_argument("--boundary-mode", choices=["none", "residual", "state", "full"], default="full")
    parser.add_argument("--boundary-hidden", type=int, default=16)
    parser.add_argument("--use-cga", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cga-hidden", type=int, default=16)
    parser.add_argument("--global-op", choices=["scan", "pool1x1", "dw3x3"], default="scan",
                        help="fusion global-branch operator; pool1x1/dw3x3 are the H2 external-op controls")
    parser.add_argument("--reliability-override", choices=["none", "uniform", "shuffle"], default="none",
                        help="H3 falsification: replace learned reliability with a content-free map")
    parser.add_argument("--cga-conflict-override", choices=["none", "uniform", "shuffle"], default="none",
                        help="H5 falsification: override CGA conflict map (uniform=all-ones, shuffle=spatial perm)")
    parser.add_argument("--dump-boundary", default="",
                        help="directory for per-image reliability maps (.npz) + learned_scales.csv; "
                             "runs an extra native forward per image, so pair with --limit")
    parser.add_argument("--tile-size", type=int, default=0, help="0 uses native padded inference")
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rgb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("manifest contains no evaluation rows")

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_model(args).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    _apply_reliability_override(model, args.reliability_override)
    _apply_cga_conflict_override(model, args.cga_conflict_override)
    if args.dump_boundary:
        _dump_learned_scales(Path(args.dump_boundary) / "learned_scales.csv", model)

    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            ir, vi, ir_array, vi_array, vi_rgb = _load_pair(data_root, row)
            ir, vi = ir.to(device), vi.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if args.tile_size > 0:
                fused = infer_tiled(model, ir, vi, args.tile_size, args.tile_overlap, args.amp)
                inference_mode = f"tile{_valid_model_dim_at_least(args.tile_size)}_overlap{args.tile_overlap}"
            else:
                fused = infer_native(model, ir, vi, args.amp)
                inference_mode = "native_padded"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            fused_uint8 = (fused.squeeze().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            metrics = evaluate_pair(ir_array, vi_array, fused_uint8.astype(np.float32))
            fused_path, rgb_path = _save_outputs(output_dir, row, fused_uint8, vi_rgb, args.save_rgb)
            if args.dump_boundary and args.boundary_mode != "none":
                _dump_boundary_maps(args.dump_boundary, row, _infer_boundary_aux(model, ir, vi, args.amp))
            result = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "width": fused_uint8.shape[1],
                "height": fused_uint8.shape[0],
                "seconds": seconds,
                "inference_mode": inference_mode,
                "fused_path": fused_path,
                "rgb_path": rgb_path,
                **{key: metrics[key] for key in CORE_METRICS},
            }
            results.append(result)
            print(f"[{index}/{len(rows)}] {row['dataset']}/{row['sample_id']} {seconds:.3f}s", flush=True)

    _write_metrics(Path(args.metrics_csv), results)
    print(f"wrote {args.metrics_csv} ({len(results)} pairs)", flush=True)


if __name__ == "__main__":
    main()
