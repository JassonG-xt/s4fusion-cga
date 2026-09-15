import torch
import pytest


def test_sobel_grad_finite_backward_under_fp16_amp():
    # Regression: under CUDA AMP the 1e-12 epsilon underflowed in fp16, so flat
    # regions produced a 0/0 NaN gradient and collapsed the GradScaler to 0.
    if not torch.cuda.is_available():
        pytest.skip("fp16 autocast sqrt-underflow is CUDA-specific")
    from train_gates import sobel_grad

    x = torch.zeros(1, 1, 16, 16, device="cuda", requires_grad=True)  # flat: gx=gy=0
    with torch.cuda.amp.autocast():
        grad_map = sobel_grad(x)
    grad_map.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_feature_gradient_finite_backward_under_fp16_amp():
    if not torch.cuda.is_available():
        pytest.skip("fp16 autocast sqrt-underflow is CUDA-specific")
    from modules.boundary import feature_gradient

    x = torch.zeros(1, 4, 12, 12, device="cuda", requires_grad=True)
    with torch.cuda.amp.autocast():
        grad_map = feature_gradient(x)
    grad_map.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_boundary_reliability_is_normalized_and_differentiable():
    from modules.boundary import BoundaryReliabilityEstimator

    estimator = BoundaryReliabilityEstimator(channels=8, hidden=8)
    ir = torch.rand(2, 8, 15, 17, requires_grad=True)
    vi = torch.rand(2, 8, 15, 17, requires_grad=True)
    output = estimator(ir, vi)

    assert output.weights.shape == (2, 3, 15, 17)
    assert torch.allclose(output.weights.sum(1), torch.ones(2, 15, 17), atol=1e-5)
    output.boundary.mean().backward()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    assert vi.grad is not None and torch.isfinite(vi.grad).all()


def test_boundary_loss_prefers_reliability_weighted_edges():
    from types import SimpleNamespace

    from modules.brss_losses import boundary_fusion_loss

    ir = torch.zeros(1, 1, 16, 16)
    vi = torch.zeros(1, 1, 16, 16)
    ir[:, :, :, 8:] = 1.0
    weights = torch.zeros(1, 3, 16, 16)
    weights[:, 0] = 1.0
    aux = SimpleNamespace(weights=weights)
    matching = boundary_fusion_loss(ir.clone(), ir, vi, [aux])
    flat = boundary_fusion_loss(torch.zeros_like(ir), ir, vi, [aux])
    assert matching < flat


def test_brss_fusion_block_zero_init_is_baseline_compatible(monkeypatch):
    monkeypatch.setenv("S4FUSION_ALLOW_SCAN_FALLBACK", "1")
    from modules.fusion import FusionBlock

    torch.manual_seed(0)
    baseline = FusionBlock(d_model=8, d_state=4, expand=1.0, boundary_mode="none")
    torch.manual_seed(0)
    brss = FusionBlock(d_model=8, d_state=4, expand=1.0, boundary_mode="full", boundary_hidden=8)
    brss.load_state_dict(baseline.state_dict(), strict=False)

    ir = torch.rand(1, 9, 9, 8)
    vi = torch.rand(1, 9, 9, 8)
    base_ir, base_vi = baseline(ir, vi)
    brss_ir, brss_vi = brss(ir, vi)

    assert torch.allclose(base_ir, brss_ir, atol=1e-5, rtol=1e-4)
    assert torch.allclose(base_vi, brss_vi, atol=1e-5, rtol=1e-4)
    assert brss.last_boundary_aux is not None


def test_structural_distillation_returns_finite_components():
    from modules.boundary import BoundaryReliabilityEstimator
    from modules.brss_losses import structural_distillation_loss

    estimator = BoundaryReliabilityEstimator(4, hidden=8)
    student_boundary = estimator(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8))
    teacher_boundary = estimator(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8))
    student = torch.rand(1, 1, 16, 16, requires_grad=True)
    teacher = torch.rand(1, 1, 16, 16)
    student_aux = {"fusion_features": [torch.rand(1, 8, 8, 4)], "boundary": [student_boundary]}
    teacher_aux = {"fusion_features": [torch.rand(1, 8, 8, 4)], "boundary": [teacher_boundary]}

    losses = structural_distillation_loss(student, teacher, student_aux, teacher_aux)
    assert len(losses) == 3
    assert all(torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert student.grad is not None


def test_compute_loss_supports_boundary_mode_none():
    from types import SimpleNamespace

    from train_brss import compute_loss

    class Baseline(torch.nn.Module):
        def forward(self, ir, vi, return_aux=False):
            fused = 0.5 * (ir + vi)
            if return_aux:
                return fused, {"boundary": [], "fusion_features": []}
            return fused

    args = SimpleNamespace(
        w_intensity=1.0,
        w_grad=10.0,
        w_smooth=0.1,
        smooth_k=10.0,
        w_boundary=2.0,
        w_reliability=0.1,
        w_scale=0.0,
        w_reliability_scale=0.0,
    )
    ir = torch.rand(1, 1, 16, 16, requires_grad=True)
    vi = torch.rand(1, 1, 16, 16)
    loss, pieces = compute_loss(Baseline(), ir, vi, args)
    assert torch.isfinite(loss)
    assert pieces["boundary"] == 0.0
    assert pieces["reliability"] == 0.0
    loss.backward()
    assert ir.grad is not None


def test_reliability_override_modes(monkeypatch):
    # Hook (b): content-free reliability substitutes for the H3 falsification.
    from modules.boundary import BoundaryReliabilityEstimator

    torch.manual_seed(0)
    estimator = BoundaryReliabilityEstimator(channels=4, hidden=8)
    # Break the neutral zero-init so the learned map is clearly non-uniform.
    torch.nn.init.normal_(estimator.net[-1].weight, std=1.0)
    torch.nn.init.normal_(estimator.net[-1].bias, std=1.0)
    ir = torch.rand(2, 4, 12, 10)
    vi = torch.rand(2, 4, 12, 10)

    base = estimator(ir, vi).weights
    assert not torch.allclose(base, torch.full_like(base, 1.0 / 3.0), atol=1e-3)

    estimator.reliability_override = "uniform"
    uniform = estimator(ir, vi).weights
    estimator.reliability_override = "shuffle"
    shuffled = estimator(ir, vi).weights
    estimator.reliability_override = "none"
    restored = estimator(ir, vi).weights

    # uniform: exactly 1/C everywhere and still a valid distribution.
    assert torch.allclose(uniform, torch.full_like(uniform, 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(uniform.sum(1), torch.ones(2, 12, 10), atol=1e-5)
    # shuffle: relocates weights but preserves the per-channel multiset and sum.
    assert torch.allclose(shuffled.sum(1), torch.ones(2, 12, 10), atol=1e-5)
    assert not torch.allclose(shuffled, base, atol=1e-4)
    assert torch.allclose(
        shuffled.reshape(2, 3, -1).sort(dim=-1).values,
        base.reshape(2, 3, -1).sort(dim=-1).values,
        atol=1e-5,
    )
    # override is non-destructive: resetting to none reproduces the baseline.
    assert torch.allclose(restored, base, atol=1e-6)


def test_global_op_threads_through_build_model_and_runs(monkeypatch):
    # Hook (a): --global-op reaches every FusionBlock and each op runs.
    monkeypatch.setenv("S4FUSION_ALLOW_SCAN_FALLBACK", "1")
    from types import SimpleNamespace

    from train_brss import build_model
    from modules.fusion import FusionBlock

    for op in ("scan", "pool1x1", "dw3x3"):
        args = SimpleNamespace(
            depths="1,2,1", teacher_depths="1,2,1", fusion_scales="4,8",
            gate_type="scalar", dynamic_gate_hidden=8,
            boundary_mode="none", boundary_hidden=8, global_op=op,
        )
        model = build_model(args)
        ops = {m.global_op for m in model.modules() if isinstance(m, FusionBlock)}
        assert ops == {op}

        block = FusionBlock(d_model=8, d_state=4, expand=1.0,
                            global_scales=(4, 8), global_op=op, boundary_mode="none")
        out_x, out_y = block(torch.rand(1, 12, 12, 8), torch.rand(1, 12, 12, 8))
        assert out_x.shape == (1, 12, 12, 8) and torch.isfinite(out_x).all()

    # build_model default keeps the original scan operator.
    default_args = SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="4,8",
        gate_type="scalar", dynamic_gate_hidden=8, boundary_mode="none", boundary_hidden=8,
    )
    default_model = build_model(default_args)
    assert {m.global_op for m in default_model.modules() if isinstance(m, FusionBlock)} == {"scan"}


def test_evaluate_boundary_hooks(tmp_path, monkeypatch):
    # Hook (b) apply + hook (c) dumps, exercised through evaluate_brss helpers.
    monkeypatch.setenv("S4FUSION_ALLOW_SCAN_FALLBACK", "1")
    import csv as _csv
    from types import SimpleNamespace

    import numpy as np

    import evaluate_brss
    from train_brss import build_model
    from modules.boundary import BoundaryReliabilityEstimator

    args = SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none",
        gate_type="scalar", dynamic_gate_hidden=8,
        boundary_mode="full", boundary_hidden=8, global_op="scan",
    )
    model = build_model(args).eval()

    # (b) override flips every estimator; count matches the estimator population.
    count = evaluate_brss._apply_reliability_override(model, "shuffle")
    modes = {m.reliability_override for m in model.modules()
             if isinstance(m, BoundaryReliabilityEstimator)}
    assert count >= 1 and modes == {"shuffle"}

    # (c1) learned-scale dump: one row per boundary block with tanh-bounded values.
    scales_csv = tmp_path / "learned_scales.csv"
    n_blocks = evaluate_brss._dump_learned_scales(scales_csv, model)
    with scales_csv.open() as handle:
        rows = list(_csv.DictReader(handle))
    assert n_blocks == count and len(rows) == count
    assert {"state_scale_ir_vi", "residual_scale_ir"} <= set(rows[0])
    assert abs(float(rows[0]["state_scale_ir_vi"])) <= 1.0

    # (c2) per-image reliability maps via a real aux forward at a valid size.
    size = evaluate_brss._valid_model_dim_at_least(36)
    boundary_list = evaluate_brss._infer_boundary_aux(
        model, torch.rand(1, 1, size, size), torch.rand(1, 1, size, size), amp=False)
    assert len(boundary_list) == count
    evaluate_brss._dump_boundary_maps(str(tmp_path), {"dataset": "unit", "sample_id": "s0"}, boundary_list)
    saved = np.load(tmp_path / "unit" / "s0_boundary.npz")
    assert saved["block0_weights"].shape[0] == 3
    assert "block0_conflict" in saved and np.isfinite(saved["block0_conflict"]).all()

    # override on a boundary-mode=none model is a no-op warning, not a crash.
    none_args = SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none",
        gate_type="scalar", dynamic_gate_hidden=8,
        boundary_mode="none", boundary_hidden=8, global_op="scan",
    )
    assert evaluate_brss._apply_reliability_override(build_model(none_args), "uniform") == 0


def test_cga_zero_init_is_identity():
    # CGA must reproduce the baseline fused output at init (zero-init gate).
    from modules.cga import ConflictGatedArbitration

    torch.manual_seed(0)
    cga = ConflictGatedArbitration(hidden=8)
    ir = torch.rand(2, 1, 16, 16)
    vi = torch.rand(2, 1, 16, 16)
    fused = torch.rand(2, 1, 16, 16)
    out, aux = cga(ir, vi, fused)
    assert torch.allclose(out, fused, atol=1e-6)
    assert aux["conflict_pred"].shape == (2, 1, 16, 16)
    assert torch.isfinite(aux["conflict_target"]).all()


def test_cga_opens_and_grads_finite():
    from modules.cga import ConflictGatedArbitration, cga_conflict_loss

    cga = ConflictGatedArbitration(hidden=8)
    torch.nn.init.constant_(cga.gate_scale, 0.5)  # open the gate
    ir = torch.rand(1, 1, 12, 12, requires_grad=True)
    vi = torch.rand(1, 1, 12, 12)
    fused = torch.rand(1, 1, 12, 12)
    out, aux = cga(ir, vi, fused)
    assert not torch.allclose(out, fused, atol=1e-4)  # open gate changes the output
    (out.mean() + cga_conflict_loss(aux)).backward()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    assert cga.gate_scale.grad is not None and torch.isfinite(cga.gate_scale.grad).all()


def test_model_use_cga_is_baseline_at_init(monkeypatch):
    monkeypatch.setenv("S4FUSION_ALLOW_SCAN_FALLBACK", "1")
    from S4Fusion import MambaNet
    from evaluate_brss import _valid_model_dim_at_least

    torch.manual_seed(0)
    base = MambaNet(depths=[1, 2, 1], depths_decoder=[1, 2, 1], use_cga=False)
    torch.manual_seed(0)
    cga = MambaNet(depths=[1, 2, 1], depths_decoder=[1, 2, 1], use_cga=True)
    cga.load_state_dict(base.state_dict(), strict=False)  # share backbone; CGA stays zero-init
    base.eval(); cga.eval()
    s = _valid_model_dim_at_least(36)
    ir = torch.rand(1, 1, s, s)
    vi = torch.rand(1, 1, s, s)
    with torch.no_grad():
        o_base = base(ir, vi)
        o_cga = cga(ir, vi)
    assert torch.allclose(o_base, o_cga, atol=1e-5)
