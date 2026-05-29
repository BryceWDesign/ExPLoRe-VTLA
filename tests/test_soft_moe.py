"""SoftMoELayer correctness tests (CPU-only).

Validates the routing-weight contract on which loss-coupling depends:
  - Combine weights softmax over experts (per-token distribution over experts)
  - Dispatch weights softmax over tokens (per-expert distribution over tokens)
  - Gradients flow back through the dispatch weights to the router (loss-coupling)

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_soft_moe.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.models.soft_moe import SoftMoELayer


# ── Fixtures ─────────────────────────────────────────────────────────────────
B, N, D = 2, 50, 16


@pytest.fixture
def moe():
    """2-expert, 1-slot-per-expert MoE layer."""
    return SoftMoELayer(dim=D, num_experts=2, slots_per_expert=1, mlp_ratio=2.0)


# ── Tests ────────────────────────────────────────────────────────────────────
def test_output_shape(moe):
    """Forward returns (B, N, D) tensor."""
    x = torch.randn(B, N, D)
    out, _ = moe(x, return_weights=True)
    assert out.shape == (B, N, D), f"expected ({B}, {N}, {D}), got {out.shape}"


def test_weights_dict_keys(moe):
    """Forward returns dict with 'combine' and 'dispatch' keys."""
    x = torch.randn(B, N, D)
    _, w = moe(x, return_weights=True)
    assert set(w.keys()) == {"combine", "dispatch"}, f"unexpected keys: {set(w.keys())}"


def test_combine_softmax_over_experts(moe):
    """Combine weights are a per-token distribution over experts (sum=1 over expert axis)."""
    x = torch.randn(B, N, D)
    _, w = moe(x, return_weights=True)
    c = w["combine"]  # (B, N, num_slots)
    assert c.shape == (B, N, 2)
    sums = c.sum(dim=-1)  # sum over experts/slots
    assert torch.allclose(sums, torch.ones(B, N), atol=1e-5), (
        f"combine doesn't sum to 1 per token: range [{sums.min():.6f}, {sums.max():.6f}]"
    )


def test_dispatch_softmax_over_tokens(moe):
    """Dispatch weights are a per-expert distribution over tokens (sum=1 over token axis)."""
    x = torch.randn(B, N, D)
    _, w = moe(x, return_weights=True)
    d = w["dispatch"]  # (B, N, num_slots)
    assert d.shape == (B, N, 2)
    sums = d.sum(dim=1)  # sum over tokens
    assert torch.allclose(sums, torch.ones(B, 2), atol=1e-5), (
        f"dispatch doesn't sum to 1 per expert: range [{sums.min():.6f}, {sums.max():.6f}]"
    )


def test_gradient_flow_through_dispatch(moe):
    """Loss on dispatch weights produces non-zero gradients on the router parameter phi.

    This is the loss-coupling contract: gradients can flow through dispatch back
    to the router, enabling content-dependent specialization without explicit
    supervision (paper §4.4.4).
    """
    x = torch.randn(B, N, D, requires_grad=True)
    _, w = moe(x, return_weights=True)
    # Use dispatch weights as the loss target
    loss = (w["dispatch"] ** 2).sum()
    loss.backward()

    assert moe.phi.grad is not None, "router param phi has no gradient"
    assert moe.phi.grad.abs().sum() > 0, "phi gradient is identically zero"


def test_gradient_flow_through_combine(moe):
    """Combine weights are differentiable too (needed for full forward backprop)."""
    x = torch.randn(B, N, D, requires_grad=True)
    _, w = moe(x, return_weights=True)
    loss = (w["combine"] ** 2).sum()
    loss.backward()
    assert moe.phi.grad is not None
    assert moe.phi.grad.abs().sum() > 0


def test_return_weights_false_returns_none(moe):
    """return_weights=False returns None for the weights slot."""
    x = torch.randn(B, N, D)
    out, w = moe(x, return_weights=False)
    assert out.shape == (B, N, D)
    assert w is None


def test_multi_slot_per_expert_shape():
    """slots_per_expert > 1 multiplies the slot axis."""
    moe = SoftMoELayer(dim=D, num_experts=2, slots_per_expert=3)
    x = torch.randn(B, N, D)
    _, w = moe(x, return_weights=True)
    # num_slots = num_experts * slots_per_expert = 6
    assert w["combine"].shape == (B, N, 6)
    assert w["dispatch"].shape == (B, N, 6)


def test_higher_expert_count():
    """E=64 (the SOTA config) instantiates and forward-passes correctly."""
    moe = SoftMoELayer(dim=D, num_experts=64, slots_per_expert=1)
    x = torch.randn(B, N, D)
    out, w = moe(x, return_weights=True)
    assert out.shape == (B, N, D)
    assert w["dispatch"].shape == (B, N, 64)
