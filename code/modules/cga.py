"""Conflict-Gated Arbitration (CGA).

Image-space module that (1) DETECTS modal conflict, (2) learns a per-pixel
COMMIT toward the salient modality, and (3) applies it ONLY in conflict regions
via a zero-initialized gate, so loading an S4Fusion checkpoint reproduces the
baseline exactly before training. This is a DECISION (arbitrate/commit), not a
feature-space SSM modulation (which failed as D1) nor a global blend.

Validated design (training-free pre-check on M3FD detection):
  - committing toward max-response in conflict recovers detection a blend loses;
  - smoothing removes the aggregate-mAP artifact tax.
The learned version replaces the fixed heuristic with a predicted conflict map,
a learned commit selection, and a learned strength.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _sobel(x: torch.Tensor):
    kx = x.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3) / 4.0
    ky = x.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3) / 4.0
    return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)


def conflict_signal(ir: torch.Tensor, vi: torch.Tensor, eps: float = 1e-6):
    """Self-supervised conflict target: both modalities have edges AND they
    disagree (orientation / polarity). Returns (conflict, grad_ir, grad_vi),
    all fp32 to keep the sqrt epsilon effective under AMP (see sobel_grad fix)."""
    gxi, gyi = _sobel(ir.float()); gxv, gyv = _sobel(vi.float())
    gi = torch.sqrt(gxi * gxi + gyi * gyi + eps)
    gv = torch.sqrt(gxv * gxv + gyv * gyv + eps)
    cos = (gxi * gxv + gyi * gyv) / (gi * gv + eps)
    conflict = torch.minimum(gi, gv) * (1.0 - cos) / 2.0
    return conflict, gi, gv


class ConflictGatedArbitration(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        # conflict detection head: [ir, vi, |grad_ir|, |grad_vi|, conflict_prior] -> conflict logit
        self.conflict_head = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # commit selection: which modality's structure to commit to (ir vs vi)
        self.select_head = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # global commit strength; zero-init => tanh(0)=0 => output == baseline fused
        self.gate_scale = nn.Parameter(torch.zeros(1))
        # bias the select head toward the max-response modality at init (the winning heuristic)
        nn.init.zeros_(self.select_head[-1].weight)
        nn.init.zeros_(self.select_head[-1].bias)
        # H5 falsification hook: override the learned conflict map at inference.
        self.conflict_override = "none"

    def _apply_conflict_override(self, conflict: torch.Tensor) -> torch.Tensor:
        """H5 falsification: replace the learned conflict map with a controlled variant.

        uniform: all-ones map -> CGA applies everywhere (tests if spatial selectivity matters)
        shuffle: fixed random spatial permutation -> preserves density but destroys correspondence
                 (tests if correct placement matters, not just any intervention)
        none: no-op (default, learned conflict map used as-is)
        """
        mode = getattr(self, "conflict_override", "none")
        if mode == "none":
            return conflict
        B, C, H, W = conflict.shape
        if mode == "uniform":
            return torch.ones_like(conflict)
        # deterministic shuffle: same permutation for every call (seeded once)
        if not hasattr(self, "_shuffle_perm") or self._shuffle_perm.shape[0] != H * W:
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(H * W, generator=g)
            self._shuffle_perm = perm.to(conflict.device)
        flat = conflict.view(B, C, -1)
        shuffled = flat[:, :, self._shuffle_perm]
        return shuffled.view(B, C, H, W)

    def forward(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor):
        cf, gi, gv = conflict_signal(ir, vi)
        cf_n = (cf / (cf.amax(dim=(-2, -1), keepdim=True) + 1e-6)).clamp(0, 1)
        conflict_logit = self.conflict_head(torch.cat([ir, vi, gi, gv, cf_n], dim=1))
        conflict = torch.sigmoid(conflict_logit)
        conflict = self._apply_conflict_override(conflict)
        # learned selection, initialized to favor the higher-gradient (salient) modality
        prior = (gi > gv).to(fused.dtype) * 2.0 - 1.0
        s = torch.sigmoid(self.select_head(torch.cat([ir, vi, gi, gv], dim=1)) + 2.0 * prior)
        commit = s * ir + (1.0 - s) * vi
        weight = torch.tanh(self.gate_scale) * conflict          # conflict-gated, zero at init
        fused_cga = fused + weight * (commit - fused)
        return fused_cga, {"conflict_logit": conflict_logit, "conflict_pred": conflict,
                           "conflict_target": cf_n.detach(),
                           "commit": commit, "fused_cga": fused_cga}


def cga_conflict_loss(aux) -> torch.Tensor:
    """Ground the conflict head on the self-derived conflict signal (soft target).
    Uses BCE-with-logits so it is safe under AMP autocast."""
    if not aux or "conflict_logit" not in aux:
        return torch.zeros((), device="cpu")
    return F.binary_cross_entropy_with_logits(aux["conflict_logit"], aux["conflict_target"])


def cga_commit_loss(aux) -> torch.Tensor:
    """Conflict-gated commit loss (H5).

    L = mean( conflict_pred * (fused_cga - commit)^2 )

    In conflict regions this pushes the fused output toward the per-pixel
    commitment to the salient modality, which in turn drives the zero-init
    gate_scale open (fused_cga = fused + tanh(gate)*conflict*(commit-fused)).
    conflict_pred is detached so the conflict head is supervised only by
    cga_conflict_loss; this term supervises commit selection + gate strength.
    """
    if not aux or "fused_cga" not in aux or "commit" not in aux:
        return torch.zeros((), device="cpu")
    w = aux["conflict_pred"].detach().clamp(0, 1)
    diff = aux["fused_cga"] - aux["commit"]
    return (w * diff * diff).mean()
