#!/usr/bin/env python3
"""B-2 / S1: incremental cost of CGA — parameters, latency, peak memory, module MACs.

Ported from S4Fusion-main-Innovation/profile_complexity.py, which profiled a bare
MambaNet and therefore could not express CGA, boundary modes or gate types. The
model is built through train_brss.build_model instead, so every arm the paper
reports (official weights / B0 / CGA) is profileable with the same code path.

S1 revision (review round 2026-09-18). The review objected that the frozen table is
a SINGLE measurement per cell with no variance, that the official-vs-B0 difference
(225.7 vs 219.5 ms at 507x507, identical parameter count) is unexplained, and that
FLOPs are absent. Changes made here, all additive:

  * `--repeats` (default 3): the whole timed loop is repeated and the table now
    carries mean +/- std across repeats, so a difference can be compared with the
    run-to-run spread instead of asserted.
  * each forward is timed with `torch.cuda.synchronize()` around the loop and the
    peak is read from `max_memory_allocated()` after `reset_peak_memory_stats()`,
    so the sync discipline is explicit rather than implicit.
  * `cga_macs_m`: exact multiply-accumulates of the CGA module, counted from real
    Conv2d output shapes with forward hooks (not estimated from a formula). The
    backbone is NOT counted — this release has no FLOP-counting dependency and a
    partial total would be worse than a labelled partial number; the column says
    which side of the comparison it measures.

Cross-check: analyze_h1._param_count() counts parameters straight from a
checkpoint. Run it on the same checkpoints and confirm params_m agrees before the
numbers are entered into the unfreeze record.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
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


def cga_macs(model, device, h, w) -> float:
    """Exact MACs of the CGA module on one forward, from real Conv2d output shapes.

    Returns NaN when the model has no CGA module. Conv2d only: the module is a stack
    of convolutions plus a sigmoid, so this is the complete operator cost.
    """
    cga = getattr(getattr(model, "vmunet", None), "cga", None)
    if cga is None:
        return float("nan")
    total = [0]

    def hook(mod, _inp, out):
        if isinstance(mod, torch.nn.Conv2d):
            kh, kw = mod.kernel_size
            total[0] += (out.shape[1] * out.shape[2] * out.shape[3]
                         * kh * kw * (mod.in_channels // mod.groups))

    handles = [m.register_forward_hook(hook) for m in cga.modules()]
    try:
        x = torch.randn(1, 1, h, w, device=device)
        with torch.no_grad():
            model(x, x)
    finally:
        for hd in handles:
            hd.remove()
    return total[0] / 1e6


def benchmark(model, device, h, w, warmup, iters, repeats):
    """Return (ms_mean, ms_std, fps_mean, peak_mean, peak_std) over `repeats` runs."""
    x = torch.randn(1, 1, h, w, device=device)
    y = torch.randn(1, 1, h, w, device=device)
    model.eval()
    ms_list, peak_list = [], []
    with torch.no_grad():
        for _ in range(warmup):
            model(x, y)
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(iters):
                model(x, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ms_list.append(elapsed / max(1, iters) * 1000.0)
            peak_list.append(torch.cuda.max_memory_allocated() / (1024 ** 2)
                             if device.type == "cuda" else float("nan"))
    ms = statistics.mean(ms_list)
    ms_std = statistics.pstdev(ms_list) if len(ms_list) > 1 else 0.0
    peak = statistics.mean(peak_list)
    peak_std = statistics.pstdev(peak_list) if len(peak_list) > 1 else 0.0
    return ms, ms_std, 1000.0 / ms, peak, peak_std


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile CGA incremental cost (B-2/S1)")
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
    ap.add_argument("--sizes", default="507,507;1034,769",
                    help="HxW sizes to sweep. NOT arbitrary: the architecture only "
                         "accepts a dimension d with ((d-4)//3 + 1) %% 4 == 0 "
                         "(evaluate_brss.py L25-29), i.e. d in {12m+1, 12m+2, 12m+3}, "
                         "because PatchMerging2D needs an even feature map "
                         "(modules/utils.py L45-66) and the patch-embed stage divides "
                         "by 6 with floor semantics. The previous default "
                         "(512,512;1024,768) is NOT runnable and produced zero rows; "
                         "measured by code/probe_valid_sizes.py, where 507x507 was the "
                         "only passing size of ten candidates. The defaults here are "
                         "the two sizes the pipeline actually uses: 507x507 is the "
                         "tiled evaluation size behind every frozen fusion metric "
                         "(inference_mode=tile507_overlap64), and 1034x769 is the "
                         "padded native size gen_cga_fused.py feeds to the model. "
                         "Report whichever you use, and say which.")
    ap.add_argument("--warmup", type=int, default=10,
                    help="S1: raised from 5; the review asked for >=10")
    ap.add_argument("--iters", type=int, default=30,
                    help="S1: raised from 20; the review asked for >=30 forwards")
    ap.add_argument("--repeats", type=int, default=3,
                    help="S1: independent timed runs; mean/std are reported over these")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-csv", default="/mnt/e/lunwen/S4Fusion-main/_temp/complexity_s1_20260918.csv")
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
              "ms_per_iter", "ms_std", "fps", "peak_mem_mb", "peak_mem_std_mb",
              "repeats", "iters", "cga_macs_m", "device"]
    write_header = not out.exists()

    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        for (h, w) in parse_sizes(args.sizes):
            macs = cga_macs(model, device, h, w)
            ms, ms_std, fps, peak, peak_std = benchmark(
                model, device, h, w, args.warmup, args.iters, args.repeats)
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
                "ms_std": round(ms_std, 4),
                "fps": round(fps, 4),
                "peak_mem_mb": round(peak, 2),
                "peak_mem_std_mb": round(peak_std, 2),
                "repeats": args.repeats,
                "iters": args.iters,
                "cga_macs_m": round(macs, 4) if macs == macs else "",
                "device": str(device),
            }
            writer.writerow(row)
            print(f"[{args.tag}] {h}x{w}: {ms:.2f} +/- {ms_std:.2f} ms  {fps:.2f} FPS  "
                  f"peak {peak:.1f} +/- {peak_std:.1f} MB  CGA MACs {macs:.2f} M", flush=True)

    print(f"[{args.tag}] appended to {out}", flush=True)


if __name__ == "__main__":
    main()
