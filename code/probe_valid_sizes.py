#!/usr/bin/env python3
"""TEMP DIAGNOSTIC (not a project artifact) — which input sizes can S4Fusion run?

B-2 (run_p1_experiments.sh b2) died on the FIRST size, 512x512, in all three arms:

    Warning, x.shape torch.Size([1, 85, 85, 96]) is not match even
    RuntimeError: The size of tensor a (84) must match the size of tensor b (85)

Cause (code/modules/utils.py L45-66): PatchMerging2D halves H and W by strided
slicing and then does x.view(B, H//2, W//2, 4C). It requires an EVEN feature map.
The patch-embed stage divides the input by 6 with floor semantics (512/6 = 85.33
-> 85, odd), so 512 is not a legal size. The evaluation pipeline already avoids
this by tiling at 507 (507/6 = 84.5 -> 84, even).

This probe finds which sizes are actually runnable, so B-2 can report numbers at
a size that is both legal and defensible in the manuscript.

Run from the code directory with the training interpreter.
"""
import sys
import time
from types import SimpleNamespace

CODE = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code"
sys.path.insert(0, CODE)

import torch  # noqa: E402
from train_brss import build_model, load_checkpoint  # noqa: E402

OFFICIAL = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main/model/model.pkl"

# Multiples of 6 whose quotient is even are expected to work; a few odd quotients
# and non-multiples are included so the boundary is measured rather than assumed.
CANDIDATES = [
    (512, 512),    # expected FAIL (85 odd) - the size the old default used
    (507, 507),    # expected OK   (84 even) - the pipeline's tile size
    (504, 504),    # expected OK   (84 even)
    (510, 510),    # expected FAIL (85 odd)
    (516, 516),    # expected OK   (86 even)
    (600, 600),    # expected OK   (100 even)
    (768, 768),    # expected OK   (128 even)
    (1024, 768),   # expected OK   (170 / 128 even) - M3FD native, what the paper needs
    (1024, 1024),  # expected OK   (170 even)
    (1026, 768),   # expected FAIL (171 odd)
]


def build(device):
    model = build_model(SimpleNamespace(
        depths="1,2,1",
        teacher_depths="1,2,1",
        fusion_scales="none",
        gate_type="scalar",
        dynamic_gate_hidden=16,
        global_op="scan",
        boundary_mode="none",
        boundary_hidden=16,
        use_cga=False,
        cga_hidden=16,
    )).to(device).eval()
    load_checkpoint(model, OFFICIAL, device)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}", flush=True)
    model = build(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[probe] official params={total} ({total / 1e6:.6f} M)", flush=True)

    ok, fail = [], []
    for (h, w) in CANDIDATES:
        x = torch.randn(1, 1, h, w, device=device)
        y = torch.randn(1, 1, h, w, device=device)
        try:
            with torch.no_grad():
                model(x, y)  # warmup
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(5):
                    model(x, y)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 5
                peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                        if device.type == "cuda" else float("nan"))
            fps = 1.0 / dt
            print(f"[probe] OK   {h}x{w}: {dt * 1000:.1f} ms/iter  {fps:.2f} FPS  peak {peak:.1f} MiB",
                  flush=True)
            ok.append((h, w))
        except Exception as e:  # noqa: BLE001
            print(f"[probe] FAIL {h}x{w}: {type(e).__name__}: {str(e)[:130]}", flush=True)
            fail.append((h, w))
        finally:
            del x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[probe] usable={ok}")
    print(f"[probe] rejected={fail}")


if __name__ == "__main__":
    main()
