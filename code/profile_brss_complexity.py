#!/usr/bin/env python3
"""B-2: incremental cost of CGA — parameters, FPS and peak memory.

Ported from S4Fusion-main-Innovation/profile_complexity.py, which profiled a bare
MambaNet and therefore could not express CGA, boundary modes or gate types. The
model is built through train_brss.build_model instead, so every arm the paper
reports (official weights / B0 / CGA) is profileable with the same code path.

Deliberate scope: this measures parameters, single-forward latency, FPS and peak
CUDA memory. It does NOT compute FLOPs/MACs — that would need thop or fvcore as an
extra dependency. Parameters + FPS + peak memory are enough to replace the
PENDING-NOT-FROZEN placeholder in Sec. 3.5; FLOPs can be added later if a venue
insists.

Cross-check: analyze_h1._param_count() counts parameters straight from a
checkpoint. Run it on the same checkpoints and confirm params_m agrees before the
numbers are entered into the unfreeze record.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time
from types import SimpleNamespace

import torch

from train_brss import build_model, load_checkpoint

MISSING = object()


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def parse_sizes(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk:
            a, b = chunk.lower().split("x")
        elif "," in chunk:
            a, b = chunk.split(",")
        else:
            a = b = chunk
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("--sizes must look like 512,512;1024,768")
    return out


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark(model, device, h, w, warmup, iters):
    x = torch.randn(1, 1, h, w, device=device)
    y = torch.randn(1, 1, h, w, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            model(x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    ms = elapsed / max(1, iters) * 1000.0
    fps = 1.0 / (elapsed / max(1, iters))
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else float("nan")
    return ms, fps, peak_mb


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile CGA incremental cost (B-2)")
    ap.add_argument("--tag", required=True, help="row label, e.g. official / B0 / CGA")
    ap.add_argument("--checkpoint", default="", help="checkpoint path; empty profiles random init")
    ap.add_argument("--use-cga", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cga-hidden", type=int, default=16)
    ap.add_argument("--boundary-mode", choices=["none", "residual", "state", "full"], default="none")
    ap.add_argument("--boundary-hidden", type=int, default=16)
    ap.add_argument("--gate-type", choices=["scalar", "dynamic"], default="scalar")
    ap.add_argument("--global-op", choices=["scan", "pool1x1", "dw3x3"], default="scan")
    ap.add_argument("--dynamic-gate-hidden", type=int, default=16)
    ap.add_argument("--depths", default="1,2,1")
    ap.add_argument("--teacher-depths", default="1,2,1")
    ap.add_argument("--fusion-scales", default="none")
    ap.add_argument("--sizes", default="512,512;1024,768",
                    help="HxW sizes to sweep. 1024x768 is the M3FD native size and is "
                         "the one the manuscript needs (reporting 512 alone overstates FPS "
                         "and does not answer the reviewer). If 1024x768 OOMs on a 4 GB "
                         "card, lower it and RECORD the size actually used.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-csv", default="/mnt/e/lunwen/S4Fusion-main/_temp/complexity_b2.csv")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = build_model(SimpleNamespace(
        depths=args.depths,
        teacher_depths=args.teacher_depths,
        fusion_scales=args.fusion_scales,
        gate_type=args.gate_type,
        dynamic_gate_hidden=args.dynamic_gate_hidden,
        global_op=args.global_op,
        boundary_mode=args.boundary_mode,
        boundary_hidden=args.boundary_hidden,
        use_cga=args.use_cga,
        cga_hidden=args.cga_hidden,
    )).to(device).eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)

    total, trainable = count_params(model)
    print(f"[{args.tag}] params={total} ({total/1e6:.6f} M), weights={args.checkpoint or 'random'}", flush=True)

    out = pathlib.Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["tag", "checkpoint", "use_cga", "boundary_mode", "gate_type",
              "height", "width", "params_m", "trainable_params_m",
              "ms_per_iter", "fps", "peak_mem_mb", "device"]
    write_header = not out.exists()

    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        for (h, w) in parse_sizes(args.sizes):
            ms, fps, peak = benchmark(model, device, h, w, args.warmup, args.iters)
            row = {
                "tag": args.tag,
                "checkpoint": args.checkpoint or "random_init",
                "use_cga": args.use_cga,
                "boundary_mode": args.boundary_mode,
                "gate_type": args.gate_type,
                "height": h, "width": w,
                "params_m": round(total / 1e6, 8),
                "trainable_params_m": round(trainable / 1e6, 8),
                "ms_per_iter": round(ms, 4),
                "fps": round(fps, 4),
                "peak_mem_mb": round(peak, 2),
                "device": str(device),
            }
            writer.writerow(row)
            print(f"[{args.tag}] {h}x{w}: {ms:.2f} ms  {fps:.2f} FPS  peak {peak:.1f} MB", flush=True)

    print(f"[{args.tag}] appended to {out}", flush=True)


if __name__ == "__main__":
    main()
