"""Selective-scan import wrapper.

The real CUDA selective scan from mamba_ssm is required for training and paper
experiments. A tiny PyTorch fallback is exposed only when
S4FUSION_ALLOW_SCAN_FALLBACK=1, so lightweight wiring tests can still run on
machines without nvcc/mamba_ssm.
"""

from __future__ import annotations

import os
import warnings

import torch


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn
    _IMPORT_ERROR = None
    HAS_MAMBA_SELECTIVE_SCAN = True
except Exception as exc:  # pragma: no cover - exercised in subprocess smoke tests
    _mamba_selective_scan_fn = None
    _IMPORT_ERROR = exc
    HAS_MAMBA_SELECTIVE_SCAN = False


def _fallback_selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
    del dts, As, Bs, Cs, z, delta_bias, delta_softplus, return_last_state
    out = xs.float()
    if Ds is not None and torch.numel(Ds) == out.shape[1]:
        out = out * Ds.float().view(1, -1, 1)
    return out


def selective_scan_fn(*args, **kwargs):
    # Explicitly honor the smoke-test switch even when the CUDA extension is
    # installed. This keeps CPU tests from dispatching into a CUDA-only op.
    if os.environ.get("S4FUSION_ALLOW_SCAN_FALLBACK") == "1":
        if not getattr(selective_scan_fn, "_warned", False):
            warnings.warn(
                "Using PyTorch selective_scan fallback. This is only for smoke tests; "
                "do not use it for manuscript experiments.",
                RuntimeWarning,
                stacklevel=2,
            )
            selective_scan_fn._warned = True
        return _fallback_selective_scan(*args, **kwargs)
    if _mamba_selective_scan_fn is not None:
        return _mamba_selective_scan_fn(*args, **kwargs)
    raise ImportError(
        "mamba_ssm selective_scan_fn is required for S4Fusion training/inference. "
        "Install a CUDA/NVCC-compatible mamba-ssm build, or set "
        "S4FUSION_ALLOW_SCAN_FALLBACK=1 only for smoke tests."
    ) from _IMPORT_ERROR


__all__ = ["HAS_MAMBA_SELECTIVE_SCAN", "selective_scan_fn"]
