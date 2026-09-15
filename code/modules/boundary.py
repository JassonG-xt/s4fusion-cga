"""Boundary-reliability components for BRSS-Fusion.

The estimator deliberately combines fixed image-structure operators with a
learned residual.  This prevents the reliability map from degenerating into a
plain maximum-gradient selector while keeping the initial prediction neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _depthwise_kernel(channels: int, kernel: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    weight = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, weight.expand(channels, 1, 3, 3), padding=1, groups=channels)


def feature_gradient(x: torch.Tensor) -> torch.Tensor:
    """Channel-averaged Sobel magnitude for BCHW feature maps."""
    if x.ndim != 4:
        raise ValueError("feature_gradient expects BCHW tensors")
    c = x.shape[1]
    kx = x.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 4.0
    ky = x.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 4.0
    gx = _depthwise_kernel(c, kx, x)
    gy = _depthwise_kernel(c, ky, x)
    # fp32 magnitude: under AMP the 1e-8 epsilon underflows in fp16 and flat
    # regions produce a 0/0 NaN gradient through the sqrt (see sobel_grad).
    gx, gy = gx.float(), gy.float()
    return torch.sqrt(gx.square() + gy.square() + 1e-8).mean(dim=1, keepdim=True)


def feature_laplacian(x: torch.Tensor) -> torch.Tensor:
    """Channel-averaged absolute Laplacian response."""
    c = x.shape[1]
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return _depthwise_kernel(c, kernel, x).abs().mean(dim=1, keepdim=True)


@dataclass
class BoundaryReliabilityOutput:
    weights: torch.Tensor
    boundary: torch.Tensor
    confidence: torch.Tensor
    ir_edge: torch.Tensor
    vi_edge: torch.Tensor

    @property
    def ir(self) -> torch.Tensor:
        return self.weights[:, 0:1]

    @property
    def vi(self) -> torch.Tensor:
        return self.weights[:, 1:2]

    @property
    def shared(self) -> torch.Tensor:
        return self.weights[:, 2:3]


class BoundaryReliabilityEstimator(nn.Module):
    """Estimate IR/VI/shared boundary reliability at every spatial position."""

    def __init__(self, channels: int, hidden: int = 24, use_global_context: bool = True):
        super().__init__()
        hidden = max(8, int(hidden))
        self.channels = int(channels)
        self.use_global_context = bool(use_global_context)

        # Inference-time falsification hook (protocol H3): replace the learned
        # reliability with a content-free map to test whether the *content* of
        # the reliability, not merely any spatial mask, drives the gain.
        # "none" leaves the learned map untouched (training/normal inference).
        self.reliability_override = "none"
        self._override_seed = 1234

        # 8 structure channels + optional two-channel global context.
        in_channels = 10 if self.use_global_context else 8
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        # Neutral initial reliability: softmax([0, 0, 0]).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        denom = x.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (x / denom).clamp(0.0, 8.0)

    def forward(
        self,
        ir: torch.Tensor,
        vi: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> BoundaryReliabilityOutput:
        if ir.shape != vi.shape or ir.ndim != 4:
            raise ValueError("BRE expects equally shaped BCHW IR/VI features")

        ir_edge = self._normalize(feature_gradient(ir))
        vi_edge = self._normalize(feature_gradient(vi))
        ir_lap = self._normalize(feature_laplacian(ir))
        vi_lap = self._normalize(feature_laplacian(vi))
        agreement = torch.exp(-(ir_edge - vi_edge).abs())
        contrast_ir = self._normalize((ir - F.avg_pool2d(ir, 3, 1, 1)).abs().mean(1, keepdim=True))
        contrast_vi = self._normalize((vi - F.avg_pool2d(vi, 3, 1, 1)).abs().mean(1, keepdim=True))
        conflict = (ir_edge - vi_edge).abs()
        descriptors = [ir_edge, vi_edge, ir_lap, vi_lap, agreement, contrast_ir, contrast_vi, conflict]

        if self.use_global_context:
            if global_context is None:
                pooled_ir = F.adaptive_avg_pool2d(ir_edge, 1).expand_as(ir_edge)
                pooled_vi = F.adaptive_avg_pool2d(vi_edge, 1).expand_as(vi_edge)
                global_context = torch.cat([pooled_ir, pooled_vi], dim=1)
            if global_context.shape[1] != 2:
                raise ValueError("global_context must have two channels")
            if global_context.shape[-2:] != ir.shape[-2:]:
                global_context = F.interpolate(global_context, ir.shape[-2:], mode="bilinear", align_corners=False)
            descriptors.append(global_context)

        logits = self.net(torch.cat(descriptors, dim=1))
        # Fixed structural priors are deliberately weak; the learned residual
        # can override them after training.
        prior = torch.cat([ir_edge - vi_edge, vi_edge - ir_edge, agreement], dim=1)
        weights = torch.softmax(logits + 0.1 * prior, dim=1)
        weights = self._apply_override(weights)
        boundary = weights[:, 0:1] * ir_edge + weights[:, 1:2] * vi_edge
        boundary = boundary + weights[:, 2:3] * 0.5 * (ir_edge + vi_edge)
        confidence = weights.max(dim=1, keepdim=True).values
        return BoundaryReliabilityOutput(weights, boundary, confidence, ir_edge, vi_edge)

    def _apply_override(self, weights: torch.Tensor) -> torch.Tensor:
        """Content-free reliability substitutes for the H3 falsification test.

        ``none``    -> untouched learned reliability (default).
        ``uniform`` -> every position gets an equal 1/C weight.
        ``shuffle`` -> the learned per-position triples are scrambled by a fixed
                       spatial permutation, destroying boundary correspondence
                       while preserving the marginal weight statistics.
        The whole three-channel column moves together, so each map stays a valid
        distribution (channels still sum to one).
        """
        mode = getattr(self, "reliability_override", "none")
        if mode == "none":
            return weights
        if mode == "uniform":
            return torch.full_like(weights, 1.0 / weights.shape[1])
        if mode == "shuffle":
            b, c, h, w = weights.shape
            generator = torch.Generator(device=weights.device)
            generator.manual_seed(int(self._override_seed))
            perm = torch.randperm(h * w, generator=generator, device=weights.device)
            return weights.reshape(b, c, h * w)[:, :, perm].reshape(b, c, h, w)
        raise ValueError(f"unknown reliability_override={mode!r}")


def reliability_consistency_loss(
    high_resolution: torch.Tensor,
    low_resolution: torch.Tensor,
) -> torch.Tensor:
    """Cross-resolution consistency for three-channel reliability maps."""
    down = F.interpolate(high_resolution, low_resolution.shape[-2:], mode="bilinear", align_corners=False)
    return F.l1_loss(down, low_resolution.detach())
