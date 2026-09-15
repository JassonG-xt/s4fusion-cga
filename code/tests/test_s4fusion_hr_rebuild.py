import numpy as np
import torch
import os
import subprocess
import sys
import pytest


def _legacy_rebuild():
    """Load optional 2025 audit tooling only when that tree is copied locally."""
    return pytest.importorskip("tools.s4fusion_hr_rebuild")


def test_dynamic_gate_produces_bounded_input_dependent_weights():
    from modules.cssc import DynamicScaleGate

    gate = DynamicScaleGate(channels=4, scales=(2, 4), hidden=8)
    x1 = torch.zeros(2, 4, 8, 8)
    x2 = torch.ones(2, 4, 8, 8)

    a1 = gate(x1, height=8, width=8)
    a2 = gate(x2, height=8, width=8)

    assert a1.shape == (2, 2)
    assert torch.all(a1 >= 0)
    assert torch.all(a1 <= 1)
    assert not torch.allclose(a1, a2)


def test_scale_consistency_loss_is_non_negative_and_differentiable():
    from train_gates import scale_consistency_loss

    fused_hr = torch.rand(2, 1, 16, 16, requires_grad=True)
    fused_lr = torch.rand(2, 1, 8, 8)

    loss = scale_consistency_loss(fused_hr, fused_lr, edge_weight=0.5)
    loss.backward()

    assert loss.item() >= 0
    assert fused_hr.grad is not None
    assert torch.isfinite(fused_hr.grad).all()


def test_extended_metrics_report_structure_color_and_statistics():
    from eval_metrics_extended import evaluate_extended_pair, paired_statistics

    ir = np.zeros((8, 8), dtype=np.float32)
    vi = np.zeros((8, 8), dtype=np.float32)
    vi[:, 4:] = 255
    baseline = np.full((8, 8), 64, dtype=np.float32)
    candidate = vi.copy()
    vi_rgb = np.dstack([vi, np.full_like(vi, 80), np.full_like(vi, 120)])
    cand_rgb = np.dstack([candidate, np.full_like(vi, 82), np.full_like(vi, 118)])

    metrics = evaluate_extended_pair(ir, vi, candidate, vi_rgb=vi_rgb, fused_rgb=cand_rgb)

    for key in ["EOR", "ECS", "OBR", "RGB_SSIM", "CIEDE2000", "Colorfulness"]:
        assert key in metrics
        assert np.isfinite(metrics[key])

    rows_base = [{"name": "a.png", "SF": 1.0}, {"name": "b.png", "SF": 2.0}]
    rows_cand = [{"name": "a.png", "SF": 2.0}, {"name": "b.png", "SF": 3.0}]
    stats = paired_statistics(rows_base, rows_cand, metric="SF", seed=0, n_boot=32)
    assert stats["metric"] == "SF"
    assert stats["mean_delta"] == 1.0
    assert "wilcoxon_p" in stats
    assert "effect_size" in stats


def test_model_wiring_smoke_with_explicit_selective_scan_fallback():
    env = os.environ.copy()
    env["S4FUSION_ALLOW_SCAN_FALLBACK"] = "1"
    code = (
        "import torch; "
        "from modules.selective_scan_compat import HAS_MAMBA_SELECTIVE_SCAN; "
        "from S4Fusion import MambaNet; "
        "device='cuda' if HAS_MAMBA_SELECTIVE_SCAN and torch.cuda.is_available() else 'cpu'; "
        "model=MambaNet(ss2d_global_scales=(), fusion_global_scales=(4,), gate_type='dynamic', dynamic_gate_hidden=8).to(device); "
        "x=torch.rand(1,1,37,37,device=device); "
        "y=torch.rand(1,1,37,37,device=device); "
        "out=model(x,y); "
        "print(device, tuple(out.shape)); "
        "assert out.shape == (1,1,37,37)"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reviewer_report_treats_existing_rgb_summary_as_evidence():
    rebuild = _legacy_rebuild()

    assert rebuild.rgb_evidence_ready()

    report_path = rebuild.build_reviewer_risk_report()
    text = report_path.read_text(encoding="utf-8")

    assert "RGB color evidence" in text
    assert "absent RGB color metrics" not in text
    assert "generate RGB color evidence" not in text


def test_sota_log_includes_local_metafusion_and_emma_evidence():
    rebuild = _legacy_rebuild()

    rows = rebuild.collect_sota_mean_rows()
    methods = {row["Method"] for row in rows}

    assert {"CDDFuse", "MetaFusion", "EMMA"} <= methods
    assert rebuild.sota_gap_status() == "partial_local_cddfuse_metafusion_emma_missing_fusionmamba_ivif_weight"

    log_path = rebuild.build_sota_availability_log()
    text = log_path.read_text(encoding="utf-8")

    assert "MetaFusion" in text
    assert "EMMA" in text
    assert "runnable_evaluated" in text
    assert "source_available_missing_local_checkpoint" in text
    assert "not be claimed as locally evaluated" in text


def test_complexity_rows_are_compact_for_manuscript_tables():
    rebuild = _legacy_rebuild()

    rows = [
        {
            "tag": "fusion_only_opt_w1e-3_g0p75_0p5",
            "checkpoint": "./model/model_ms_fusion_only_w1e-3_e10.pkl",
            "ss2d_scales": "none",
            "fusion_scales": "4,8",
            "height": "507",
            "width": "507",
            "params_m": "3.562999",
            "fps": "3.971299509624511",
            "max_mem_mb": "589.4375",
        }
    ]

    compact = rebuild.compact_complexity_rows(rows)

    assert list(compact[0]) == ["tag", "resolution", "SS2D", "Fusion", "params_M", "FPS", "mem_MB"]
    assert "checkpoint" not in compact[0]
    assert compact[0]["tag"] == "fusion_only"
    assert compact[0]["resolution"] == "507x507"
