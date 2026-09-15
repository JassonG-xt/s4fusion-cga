import math

import torch
from torch import nn
import torch.nn.functional as F


class DynamicScaleGate(nn.Module):
    """Input-adaptive CSSC gate for a fixed set of coarse scales."""

    def __init__(self, channels: int, scales=(), hidden: int = 32):
        super().__init__()
        self.scales = tuple(int(s) for s in (scales or ()))
        self.channels = int(channels)
        hidden = max(4, int(hidden))
        self.net = nn.Sequential(
            nn.Linear(self.channels + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, height: int | None = None, width: int | None = None) -> torch.Tensor:
        if not self.scales:
            return x.new_zeros((x.shape[0], 0))
        if x.ndim != 4:
            raise ValueError("DynamicScaleGate expects BCHW features")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        height = int(height or h)
        width = int(width or w)

        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        log_h = math.log(max(1, height))
        log_w = math.log(max(1, width))
        per_scale = []
        for scale in self.scales:
            hw_cols = pooled.new_tensor([log_h, log_w]).view(1, 2).expand(b, 2)
            scale_col = pooled.new_full((b, 1), math.log(max(1, scale)))
            per_scale.append(torch.cat([pooled, hw_cols, scale_col], dim=1))
        stacked = torch.stack(per_scale, dim=1)
        return torch.sigmoid(self.net(stacked).squeeze(-1))


def cssc_gate_values(
    features: torch.Tensor,
    global_gates: torch.Tensor | None,
    dynamic_gate: DynamicScaleGate | None,
    height: int,
    width: int,
) -> torch.Tensor | None:
    if dynamic_gate is not None:
        return dynamic_gate(features, height=height, width=width)
    if global_gates is None:
        return None
    return global_gates.view(1, -1).expand(features.shape[0], -1)
