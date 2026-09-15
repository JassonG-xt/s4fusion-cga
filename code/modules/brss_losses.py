"""Training objectives for boundary-reliability S4Fusion variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .boundary import feature_gradient, reliability_consistency_loss


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    peak = x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (x / peak).clamp(0.0, 1.0)


def boundary_fusion_loss(
    fused: torch.Tensor,
    ir: torch.Tensor,
    vi: torch.Tensor,
    boundary_aux,
) -> torch.Tensor:
    """Match fused edges to the reliability-weighted IR/VI soft boundary."""
    fused_edge = _normalize_map(feature_gradient(fused))
    losses = []
    for aux in boundary_aux:
        weights = F.interpolate(aux.weights, fused.shape[-2:], mode="bilinear", align_corners=False)
        ir_edge = _normalize_map(feature_gradient(ir))
        vi_edge = _normalize_map(feature_gradient(vi))
        target = weights[:, 0:1] * ir_edge + weights[:, 1:2] * vi_edge
        target = target + weights[:, 2:3] * 0.5 * (ir_edge + vi_edge)
        losses.append(F.smooth_l1_loss(fused_edge, target.detach()))
    return torch.stack(losses).mean() if losses else fused.new_zeros(())


def reliability_regularization(boundary_aux, total_variation_weight: float = 0.1) -> torch.Tensor:
    """Discourage fragmented reliability maps without forcing a single modality."""
    losses = []
    for aux in boundary_aux:
        w = aux.weights
        entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(1).mean()
        dx = (w[:, :, :, 1:] - w[:, :, :, :-1]).abs().mean()
        dy = (w[:, :, 1:, :] - w[:, :, :-1, :]).abs().mean()
        # A small negative entropy term avoids permanent uniform predictions;
        # TV prevents the resulting selection from becoming noisy.
        losses.append(-0.01 * entropy + total_variation_weight * (dx + dy))
    if not losses:
        raise ValueError("reliability_regularization requires boundary outputs")
    return torch.stack(losses).mean()


def multiscale_reliability_loss(hr_aux, lr_aux) -> torch.Tensor:
    count = min(len(hr_aux), len(lr_aux))
    if count == 0:
        raise ValueError("multiscale reliability loss requires both HR and LR outputs")
    return torch.stack([
        reliability_consistency_loss(hr_aux[i].weights, lr_aux[i].weights)
        for i in range(count)
    ]).mean()


def structural_distillation_loss(student_output, teacher_output, student_aux, teacher_aux):
    """Output, feature, boundary-map and confidence distillation."""
    output_loss = F.l1_loss(student_output, teacher_output.detach())

    feature_losses = []
    for student_feature, teacher_feature in zip(
        student_aux.get("fusion_features", []), teacher_aux.get("fusion_features", [])
    ):
        if student_feature.shape != teacher_feature.shape:
            teacher_feature = F.interpolate(
                teacher_feature.permute(0, 3, 1, 2),
                student_feature.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            channels = min(student_feature.shape[-1], teacher_feature.shape[-1])
            student_feature = student_feature[..., :channels]
            teacher_feature = teacher_feature[..., :channels]
        feature_losses.append(F.smooth_l1_loss(student_feature, teacher_feature.detach()))
    feature_loss = torch.stack(feature_losses).mean() if feature_losses else output_loss.new_zeros(())

    boundary_losses = []
    for student_boundary, teacher_boundary in zip(
        student_aux.get("boundary", []), teacher_aux.get("boundary", [])
    ):
        teacher_weights = F.interpolate(
            teacher_boundary.weights,
            student_boundary.weights.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_losses.append(F.kl_div(
            student_boundary.weights.clamp_min(1e-8).log(),
            teacher_weights.detach().clamp_min(1e-8),
            reduction="batchmean",
        ) / student_boundary.weights[0].numel())
    boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else output_loss.new_zeros(())
    return output_loss, feature_loss, boundary_loss
