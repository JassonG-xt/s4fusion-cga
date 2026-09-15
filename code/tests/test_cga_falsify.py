"""Tests for H5 CGA conflict-override falsification hooks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modules.cga import ConflictGatedArbitration


def test_uniform_override():
    """uniform -> all-ones conflict -> CGA applies everywhere."""
    m = ConflictGatedArbitration(hidden=4)
    with torch.no_grad():
        m.gate_scale.fill_(1.0)
    m.conflict_override = "uniform"
    ir = torch.rand(1, 1, 32, 32)
    vi = torch.rand(1, 1, 32, 32)
    fused = torch.rand(1, 1, 32, 32)
    _, aux = m(ir, vi, fused)
    per_pixel = (aux["fused_cga"] - fused).abs().squeeze()
    assert (per_pixel > 1e-6).all(), "some pixels unchanged under uniform override"
    print("PASS: uniform modifies all pixels")


def test_shuffle_direct():
    """Directly verify _apply_conflict_override preserves sum but breaks spatial pattern."""
    m = ConflictGatedArbitration(hidden=4)
    m.conflict_override = "shuffle"  # MUST set before calling
    B, C, H, W = 1, 1, 64, 64
    x = torch.zeros(B, C, H, W)
    x[0, 0, :16, :] = 0.9   # top band high
    x[0, 0, 48:, :] = 0.1   # bottom band low
    shuffled = m._apply_conflict_override(x.clone())
    assert abs(shuffled.sum().item() - x.sum().item()) < 1e-2
    assert not torch.equal(shuffled, x), "shuffled map identical to original"
    print("PASS: shuffle preserves sum, breaks spatial layout")


def test_shuffle_deterministic():
    """Same seed -> same permutation across calls and instances."""
    m1 = ConflictGatedArbitration(hidden=4); m1.conflict_override = "shuffle"
    m2 = ConflictGatedArbitration(hidden=4); m2.conflict_override = "shuffle"
    x = torch.rand(1, 1, 32, 32)
    s1a = m1._apply_conflict_override(x.clone())
    s1b = m1._apply_conflict_override(x.clone())
    s2 = m2._apply_conflict_override(x.clone())
    assert torch.equal(s1a, s1b), "not deterministic within instance"
    assert torch.equal(s1a, s2), "different instances produce different permutations"
    print("PASS: shuffle is deterministic across calls and instances")


def test_uniform_direct():
    m = ConflictGatedArbitration(hidden=4)
    m.conflict_override = "uniform"
    x = torch.rand(1, 1, 8, 8)
    u = m._apply_conflict_override(x.clone())
    assert torch.allclose(u, torch.ones_like(u)), "uniform override did not produce all-ones"
    print("PASS: uniform produces all-ones")


def test_none_is_noop():
    m = ConflictGatedArbitration(hidden=4)
    assert getattr(m, "conflict_override", "x") == "none", "default must be none"
    ir = torch.rand(1, 1, 16, 16)
    vi = torch.rand(1, 1, 16, 16)
    fused = torch.rand(1, 1, 16, 16)
    _, aux_a = m(ir, vi, fused)
    _, aux_b = m(ir, vi, fused)
    assert torch.equal(aux_a["fused_cga"], aux_b["fused_cga"]), "deterministic when override=none"
    # Also verify output is unchanged from no-CGA baseline when gate=0
    with torch.no_grad():
        m.gate_scale.fill_(0.0)
        _, aux_c = m(ir, vi, fused)
        assert torch.equal(aux_c["fused_cga"], fused), "zero gate must reproduce baseline exactly"
    print("PASS: default (none) is deterministic no-op; zero gate reproduces baseline")


def test_cli_flag_exists():
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluate_brss.py")
    with open(eval_path) as f:
        content = f.read()
    assert "--cga-conflict-override" in content, "CLI flag missing"
    assert "_apply_cga_conflict_override" in content, "apply function call missing"
    print("PASS: CLI flag exists in evaluate_brss.py")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_uniform_override()
    test_shuffle_direct()
    test_shuffle_deterministic()
    test_uniform_direct()
    test_none_is_noop()
    test_cli_flag_exists()
    print("\nALL TESTS PASSED")
