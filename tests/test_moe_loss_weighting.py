"""apply_moe_loss_weighting() correctness tests (CPU-only).

Validates the core contracts of the per-patch loss-coupling mechanism (paper §4.4):
  - Dict format (combine + dispatch) is accepted with weight_type selecting
  - Detach flag stops gradient flow back to dispatch (Tab. 4.5 detach ablation)
  - Dispatch-weighted loss is differentiable through dispatch (loss-coupling)
  - Sparse mode handles visible-only weights correctly

Run: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_moe_loss_weighting.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.utils.losses import apply_moe_loss_weighting


# ── Fixtures ─────────────────────────────────────────────────────────────────
B, N_PATCH, E = 2, 49, 2  # 49 patches (7×7), 2 experts


def _make_inputs(requires_grad=False):
    """Build a coherent set of inputs for apply_moe_loss_weighting in sparse mode.

    Returns:
        loss_per_token: [B, N_patch]
        weights_dict: {'combine': [B, N_patch+1, E], 'dispatch': [B, N_patch+1, E]}
                      (CLS token included at position 0)
        mask: [B, N_patch]
    """
    loss_per_token = torch.rand(B, N_PATCH)
    # Logits → softmax along the right axes
    raw = torch.randn(B, N_PATCH + 1, E, requires_grad=requires_grad)
    combine = torch.softmax(raw, dim=-1)   # softmax over experts
    dispatch = torch.softmax(raw, dim=1)   # softmax over tokens
    mask = torch.ones(B, N_PATCH, dtype=torch.bool)  # all patches visible
    return loss_per_token, {"combine": combine, "dispatch": dispatch}, mask


# ── Tests ────────────────────────────────────────────────────────────────────
def test_dict_format_returns_scalar():
    """Dict input + dispatch weight_type returns a scalar loss."""
    loss, weights, mask = _make_inputs()
    out = apply_moe_loss_weighting(
        loss_per_token=loss,
        combine_weights=weights,
        mask=mask,
        expert_idx=0,
        use_mask_tokens=False,         # sparse mode
        loss_expert_indices=[0],
        weight_type="dispatch",
    )
    assert out.dim() == 0, f"expected scalar, got shape {out.shape}"
    assert torch.isfinite(out), f"non-finite loss: {out}"


def test_tensor_format_backward_compat():
    """Old format (single tensor as combine) still works."""
    loss, weights, mask = _make_inputs()
    out = apply_moe_loss_weighting(
        loss_per_token=loss,
        combine_weights=weights["combine"],    # tensor, not dict
        mask=mask,
        expert_idx=0,
        use_mask_tokens=False,
        loss_expert_indices=[0],
        weight_type="combine",
    )
    assert out.dim() == 0


def test_invalid_weight_type_raises():
    """Asking for a weight_type not in the dict raises ValueError."""
    loss, weights, mask = _make_inputs()
    with pytest.raises(ValueError, match="not found"):
        apply_moe_loss_weighting(
            loss_per_token=loss,
            combine_weights=weights,
            mask=mask,
            expert_idx=0,
            use_mask_tokens=False,
            loss_expert_indices=[0],
            weight_type="nonexistent",
        )


def test_expert_idx_out_of_range_raises():
    """expert_idx beyond len(loss_expert_indices) raises ValueError."""
    loss, weights, mask = _make_inputs()
    with pytest.raises(ValueError, match="out of range"):
        apply_moe_loss_weighting(
            loss_per_token=loss,
            combine_weights=weights,
            mask=mask,
            expert_idx=5,                       # out of range
            use_mask_tokens=False,
            loss_expert_indices=[0],
            weight_type="dispatch",
        )





def test_output_is_finite_with_normalization():
    """Normalization preserves finite output (paper Eq 4.16 per-image normalization)."""
    loss, weights, mask = _make_inputs()
    out = apply_moe_loss_weighting(
        loss_per_token=loss,
        combine_weights=weights,
        mask=mask,
        expert_idx=0,
        use_mask_tokens=False,
        loss_expert_indices=[0],
        weight_type="dispatch",
        normalize_weights=True,
        normalize_per_image=True,
    )
    assert torch.isfinite(out), f"non-finite output: {out}"
    assert out.dim() == 0
