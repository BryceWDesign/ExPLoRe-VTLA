"""Tests for MultiBlockWeightAggregator and related helper functions.

Tests the multi-block MoE loss weighting feature which allows per-loss
block selection and aggregation of routing weights across transformer blocks.

All tests run on CPU only (no GPU required).

Run with:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_multi_block_aggregator.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.medic_model import MEDiCModel, MultiBlockWeightAggregator

# Access the static/instance methods via the class
_parse_block_config = MEDiCModel._parse_block_config


def _get_weights_for_loss(combine_weights_dict, block_indices, aggregator=None):
    """Standalone wrapper around MEDiCModel._get_weights_for_loss for testing.

    The real method is an instance method but only uses its arguments
    (no self-dependent state), so we can call it with a None self.
    """
    return MEDiCModel._get_weights_for_loss(
        None, combine_weights_dict, block_indices, aggregator
    )


# ---------------------------------------------------------------------------
# Constants matching the MEDiC ViT-Base architecture
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
NUM_PATCHES = 196  # 14x14 for 224x224 images with patch_size=16
NUM_SLOTS = 2  # V2 dense config: num_experts * slots_per_expert = 2 * 1
# Alternating MoE placement in 12-layer ViT: blocks 1, 3, 5, 7, 9, 11
AVAILABLE_MOE_BLOCKS = {1, 3, 5, 7, 9, 11}
ALL_MOE_BLOCKS_SORTED = [1, 3, 5, 7, 9, 11]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def weight_tensor():
    """Single weight tensor [B, N, E]."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)


@pytest.fixture
def weight_tensors_3():
    """List of 3 weight tensors [B, N, E] (blocks 7, 9, 11)."""
    torch.manual_seed(42)
    return [
        torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        for _ in range(3)
    ]


@pytest.fixture
def weight_tensors_6():
    """List of 6 weight tensors [B, N, E] (all MoE blocks)."""
    torch.manual_seed(42)
    return [
        torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        for _ in range(6)
    ]


@pytest.fixture
def weight_tensors_1():
    """List of 1 weight tensor [B, N, E] (single block)."""
    torch.manual_seed(42)
    return [torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)]


@pytest.fixture
def weight_dicts_3():
    """List of 3 weight dicts with 'combine' and 'dispatch' keys."""
    torch.manual_seed(42)
    return [
        {
            "combine": torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS),
            "dispatch": torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS),
        }
        for _ in range(3)
    ]


@pytest.fixture
def combine_weights_dict():
    """Mock combine_weights_dict as produced by the encoder.

    Maps MoE block index -> weight tensor [B, N, E].
    """
    torch.manual_seed(42)
    return {
        idx: torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        for idx in ALL_MOE_BLOCKS_SORTED
    }


# =========================================================================
# 1. MultiBlockWeightAggregator Tests
# =========================================================================


class TestMultiBlockWeightAggregatorMean:
    """Tests for the 'mean' aggregation strategy."""

    def test_output_shape(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, contributions = agg(weight_tensors_3)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    def test_computes_arithmetic_mean(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, contributions = agg(weight_tensors_3)
        expected = torch.stack(weight_tensors_3, dim=0).mean(dim=0)
        torch.testing.assert_close(result, expected)

    def test_contributions_is_none(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        _, contributions = agg(weight_tensors_3)
        assert contributions is None

    def test_no_learnable_params(self):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        params = list(agg.parameters())
        assert len(params) == 0

    def test_gradient_flow_to_inputs(self):
        """Mean should pass gradients through to input tensors."""
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS, requires_grad=True)
            for _ in range(3)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = agg(tensors)
        loss = result.sum()
        loss.backward()
        for t in tensors:
            assert t.grad is not None
            assert t.grad.shape == t.shape

    def test_identical_inputs_returns_same(self):
        """When all inputs are identical, mean should return the same tensor."""
        single = torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        tensors = [single.clone() for _ in range(4)]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=4, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = agg(tensors)
        torch.testing.assert_close(result, single)


class TestMultiBlockWeightAggregatorLearnedScalar:
    """Tests for the 'learned_scalar' aggregation strategy."""

    def test_output_shape(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        result, contributions = agg(weight_tensors_3)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    def test_initial_equal_weights(self):
        """Block logits initialized to zeros -> softmax produces equal weights."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        expected_weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
        actual_weights = F.softmax(agg.block_logits, dim=0)
        torch.testing.assert_close(actual_weights, expected_weights)

    def test_block_logits_initialized_to_zeros(self):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=6, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        torch.testing.assert_close(
            agg.block_logits, torch.zeros(6)
        )

    def test_gradient_updates_block_logits(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        result, _ = agg(weight_tensors_3)
        loss = result.sum()
        loss.backward()
        assert agg.block_logits.grad is not None
        assert agg.block_logits.grad.shape == (3,)

    def test_block_contributions_returned(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        _, contributions = agg(weight_tensors_3)
        assert contributions is not None
        assert contributions.shape == (3,)
        # Should be detached (no gradient)
        assert not contributions.requires_grad

    def test_contributions_sum_to_one(self, weight_tensors_3):
        """Softmax-normalized contributions should sum to 1."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        _, contributions = agg(weight_tensors_3)
        torch.testing.assert_close(
            contributions.sum(), torch.tensor(1.0)
        )

    def test_has_learnable_params(self):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=6, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        params = list(agg.parameters())
        assert len(params) == 1  # block_logits only
        assert params[0].shape == (6,)

    def test_gradient_flow_to_inputs(self):
        """Learned scalar should also pass gradients through to input tensors."""
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS, requires_grad=True)
            for _ in range(3)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        result, _ = agg(tensors)
        loss = result.sum()
        loss.backward()
        for t in tensors:
            assert t.grad is not None


class TestMultiBlockWeightAggregatorLinearProjection:
    """Tests for the 'linear_projection' aggregation strategy."""

    def test_output_shape(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        result, contributions = agg(weight_tensors_3)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    def test_softmax_applied(self, weight_tensors_3):
        """Output should have softmax applied: sums to 1 along expert dim."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        result, _ = agg(weight_tensors_3)
        sums = result.sum(dim=-1)  # [B, N]
        expected = torch.ones(BATCH_SIZE, NUM_PATCHES)
        torch.testing.assert_close(sums, expected)

    def test_gradient_updates_projection_weights(self, weight_tensors_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        result, _ = agg(weight_tensors_3)
        loss = result.sum()
        loss.backward()
        assert agg.projection.weight.grad is not None
        assert agg.projection.bias.grad is not None

    def test_projection_input_dim(self):
        """Linear projection input dim = num_blocks * num_slots."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        assert agg.projection.in_features == 3 * NUM_SLOTS
        assert agg.projection.out_features == NUM_SLOTS

    def test_contributions_is_none(self, weight_tensors_3):
        """Linear projection does not return per-block contributions."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        _, contributions = agg(weight_tensors_3)
        assert contributions is None

    def test_has_learnable_params(self):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        params = list(agg.parameters())
        assert len(params) == 2  # weight + bias

    def test_output_values_in_valid_range(self, weight_tensors_3):
        """Softmax output should be in [0, 1] range."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="linear_projection"
        )
        result, _ = agg(weight_tensors_3)
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestMultiBlockWeightAggregatorShapes:
    """Tests for shape handling across all strategies."""

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_single_block_input(self, strategy):
        """K=1 (single block) should work for all strategies."""
        tensors = [torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=1, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(tensors)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_six_blocks_input(self, strategy, weight_tensors_6):
        """K=6 (all MoE blocks) should work for all strategies."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=6, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(weight_tensors_6)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_various_batch_sizes(self, strategy, batch_size):
        """Different batch sizes should all produce correct output shapes."""
        tensors = [
            torch.randn(batch_size, NUM_PATCHES, NUM_SLOTS)
            for _ in range(3)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(tensors)
        assert result.shape == (batch_size, NUM_PATCHES, NUM_SLOTS)

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    @pytest.mark.parametrize("num_slots", [2, 4, 8, 16])
    def test_various_slot_counts(self, strategy, num_slots):
        """Different slot counts (experts * slots_per_expert) should all work."""
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, num_slots)
            for _ in range(3)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=num_slots, strategy=strategy
        )
        result, _ = agg(tensors)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, num_slots)


class TestMultiBlockWeightAggregatorDictFormat:
    """Tests for dict-format weight inputs."""

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_dict_input_returns_dict(self, strategy, weight_dicts_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, contributions = agg(weight_dicts_3)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_dict_output_has_same_keys(self, strategy, weight_dicts_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(weight_dicts_3)
        assert set(result.keys()) == {"combine", "dispatch"}

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_dict_values_correct_shape(self, strategy, weight_dicts_3):
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(weight_dicts_3)
        for key in ["combine", "dispatch"]:
            assert result[key].shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    def test_dict_keys_aggregated_independently_mean(self, weight_dicts_3):
        """With mean strategy, each key should be averaged independently."""
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = agg(weight_dicts_3)

        # Verify 'combine' is mean of combine tensors
        expected_combine = torch.stack(
            [d["combine"] for d in weight_dicts_3], dim=0
        ).mean(dim=0)
        torch.testing.assert_close(result["combine"], expected_combine)

        # Verify 'dispatch' is mean of dispatch tensors
        expected_dispatch = torch.stack(
            [d["dispatch"] for d in weight_dicts_3], dim=0
        ).mean(dim=0)
        torch.testing.assert_close(result["dispatch"], expected_dispatch)


class TestMultiBlockWeightAggregatorInvalidStrategy:
    """Tests for invalid strategy handling."""

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            MultiBlockWeightAggregator(
                num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="invalid"
            )

    def test_empty_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            MultiBlockWeightAggregator(
                num_moe_blocks=3, num_slots=NUM_SLOTS, strategy=""
            )


# =========================================================================
# 2. _parse_block_config() Tests
# =========================================================================


class TestParseBlockConfig:
    """Tests for MEDiCModel._parse_block_config() static method."""

    def test_last_returns_max_block(self):
        result = _parse_block_config("last", AVAILABLE_MOE_BLOCKS)
        assert result == [11]

    def test_all_returns_sorted_blocks(self):
        result = _parse_block_config("all", AVAILABLE_MOE_BLOCKS)
        assert result == ALL_MOE_BLOCKS_SORTED

    def test_int_returns_single_block(self):
        result = _parse_block_config(7, AVAILABLE_MOE_BLOCKS)
        assert result == [7]

    def test_int_first_block(self):
        result = _parse_block_config(1, AVAILABLE_MOE_BLOCKS)
        assert result == [1]

    def test_int_last_block(self):
        result = _parse_block_config(11, AVAILABLE_MOE_BLOCKS)
        assert result == [11]

    def test_list_returns_sorted(self):
        result = _parse_block_config([7, 9, 11], AVAILABLE_MOE_BLOCKS)
        assert result == [7, 9, 11]

    def test_list_unsorted_input_returns_sorted(self):
        result = _parse_block_config([11, 7, 9], AVAILABLE_MOE_BLOCKS)
        assert result == [7, 9, 11]

    def test_single_element_list(self):
        result = _parse_block_config([11], AVAILABLE_MOE_BLOCKS)
        assert result == [11]

    def test_two_element_list(self):
        result = _parse_block_config([5, 11], AVAILABLE_MOE_BLOCKS)
        assert result == [5, 11]

    def test_all_blocks_as_list(self):
        result = _parse_block_config(
            [1, 3, 5, 7, 9, 11], AVAILABLE_MOE_BLOCKS
        )
        assert result == ALL_MOE_BLOCKS_SORTED

    # --- Validation error tests ---

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError, match="Block 6"):
            _parse_block_config(6, AVAILABLE_MOE_BLOCKS)

    def test_invalid_int_zero_raises(self):
        with pytest.raises(ValueError, match="Block 0"):
            _parse_block_config(0, AVAILABLE_MOE_BLOCKS)

    def test_invalid_in_list_raises(self):
        with pytest.raises(ValueError, match="Block 6"):
            _parse_block_config([1, 6, 11], AVAILABLE_MOE_BLOCKS)

    def test_error_message_lists_available_blocks(self):
        with pytest.raises(ValueError, match="Available"):
            _parse_block_config(6, AVAILABLE_MOE_BLOCKS)

    def test_invalid_int_not_in_range_raises(self):
        with pytest.raises(ValueError, match="Block 12"):
            _parse_block_config(12, AVAILABLE_MOE_BLOCKS)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty block list"):
            _parse_block_config([], AVAILABLE_MOE_BLOCKS)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid block config value"):
            _parse_block_config("first", AVAILABLE_MOE_BLOCKS)

    # --- Different MoE configurations ---

    def test_last_with_different_blocks(self):
        """Different MoE block sets should use their own max."""
        alt_blocks = {0, 2, 4}
        result = _parse_block_config("last", alt_blocks)
        assert result == [4]

    def test_all_with_different_blocks(self):
        alt_blocks = {0, 4, 8}
        result = _parse_block_config("all", alt_blocks)
        assert result == [0, 4, 8]

    def test_tuple_input_accepted(self):
        """Tuple input should work like list."""
        result = _parse_block_config((7, 9, 11), AVAILABLE_MOE_BLOCKS)
        assert result == [7, 9, 11]


# =========================================================================
# 3. _get_weights_for_loss() Tests
# =========================================================================


class TestGetWeightsForLoss:
    """Tests for MEDiCModel._get_weights_for_loss() method."""

    def test_single_block_returns_exact_tensor(self, combine_weights_dict):
        """Single block: returns the exact tensor, no aggregation."""
        result, contributions = _get_weights_for_loss(
            combine_weights_dict, block_indices=[11], aggregator=None
        )
        torch.testing.assert_close(result, combine_weights_dict[11])

    def test_single_block_contributions_none(self, combine_weights_dict):
        """Single block: contributions should be None."""
        _, contributions = _get_weights_for_loss(
            combine_weights_dict, block_indices=[11], aggregator=None
        )
        assert contributions is None

    def test_single_block_no_aggregator_needed(self, combine_weights_dict):
        """Single block should work even when aggregator is None."""
        result, _ = _get_weights_for_loss(
            combine_weights_dict, block_indices=[7], aggregator=None
        )
        torch.testing.assert_close(result, combine_weights_dict[7])

    def test_multi_block_mean_aggregation(self, combine_weights_dict):
        """Multi block with mean aggregator returns averaged weights."""
        block_indices = [7, 9, 11]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = _get_weights_for_loss(
            combine_weights_dict, block_indices=block_indices, aggregator=agg
        )

        expected = torch.stack(
            [combine_weights_dict[idx] for idx in block_indices], dim=0
        ).mean(dim=0)
        torch.testing.assert_close(result, expected)

    def test_multi_block_learned_scalar_aggregation(self, combine_weights_dict):
        """Multi block with learned_scalar aggregator returns weighted combination."""
        block_indices = [7, 9, 11]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=3, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )
        result, contributions = _get_weights_for_loss(
            combine_weights_dict, block_indices=block_indices, aggregator=agg
        )

        # Should return a valid tensor of correct shape
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        # Should return contributions for logging
        assert contributions is not None

    def test_multi_block_all_blocks(self, combine_weights_dict):
        """Aggregating all 6 MoE blocks."""
        block_indices = ALL_MOE_BLOCKS_SORTED
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=6, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = _get_weights_for_loss(
            combine_weights_dict, block_indices=block_indices, aggregator=agg
        )
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)

    def test_multi_block_two_blocks(self, combine_weights_dict):
        """Aggregating exactly 2 blocks."""
        block_indices = [9, 11]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=2, num_slots=NUM_SLOTS, strategy="mean"
        )
        result, _ = _get_weights_for_loss(
            combine_weights_dict, block_indices=block_indices, aggregator=agg
        )
        expected = (combine_weights_dict[9] + combine_weights_dict[11]) / 2
        torch.testing.assert_close(result, expected)


# =========================================================================
# 4. Backward Compatibility Tests
# =========================================================================


class TestBackwardCompatibility:
    """Tests verifying backward-compatible behavior with default config."""

    def test_default_config_uses_last_block(self):
        """Default 'last' config should resolve to the last MoE block index."""
        result = _parse_block_config("last", AVAILABLE_MOE_BLOCKS)
        assert result == [max(AVAILABLE_MOE_BLOCKS)]
        assert result == [11]

    def test_single_block_no_aggregator_created(self):
        """When only 1 block is selected, no aggregator is needed.

        The logic should detect len(block_indices) == 1 and skip aggregation.
        We test this via _get_weights_for_loss with aggregator=None.
        """
        torch.manual_seed(42)
        cwd = {
            idx: torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for idx in ALL_MOE_BLOCKS_SORTED
        }
        # Single block: aggregator=None, should work fine
        result, contributions = _get_weights_for_loss(
            cwd, block_indices=[11], aggregator=None
        )
        assert contributions is None
        torch.testing.assert_close(result, cwd[11])

    def test_explicit_last_matches_default(self):
        """Explicit block index 11 should match 'last' result."""
        last_result = _parse_block_config("last", AVAILABLE_MOE_BLOCKS)
        explicit_result = _parse_block_config(11, AVAILABLE_MOE_BLOCKS)
        assert last_result == explicit_result

    def test_single_element_list_matches_int(self):
        """[11] should produce same block indices as 11."""
        list_result = _parse_block_config([11], AVAILABLE_MOE_BLOCKS)
        int_result = _parse_block_config(11, AVAILABLE_MOE_BLOCKS)
        assert list_result == int_result

    def test_single_element_list_matches_last(self):
        """[11] should produce same block indices as 'last'."""
        list_result = _parse_block_config([11], AVAILABLE_MOE_BLOCKS)
        last_result = _parse_block_config("last", AVAILABLE_MOE_BLOCKS)
        assert list_result == last_result


# =========================================================================
# 5. Integration-Style Tests (CPU)
# =========================================================================


class TestIntegration:
    """Integration tests verifying end-to-end behavior on CPU."""

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_forward_pass_cpu(self, strategy):
        """Create aggregator, run forward, verify output on CPU."""
        torch.manual_seed(42)
        K = 3
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for _ in range(K)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(tensors)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_output_differentiable(self, strategy):
        """Output should support backward pass for gradient computation."""
        torch.manual_seed(42)
        K = 3
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS, requires_grad=True)
            for _ in range(K)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy=strategy
        )
        result, _ = agg(tensors)
        loss = result.sum()
        loss.backward()
        # All input tensors should have gradients
        for t in tensors:
            assert t.grad is not None

    def test_learned_scalar_equal_logits_matches_mean(self):
        """When logits are zeros (equal weights), learned_scalar == mean."""
        torch.manual_seed(42)
        K = 4
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for _ in range(K)
        ]

        mean_agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy="mean"
        )
        learned_agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )

        # Verify logits are zeros (default init)
        assert (learned_agg.block_logits == 0).all()

        mean_result, _ = mean_agg(tensors)
        learned_result, _ = learned_agg(tensors)

        torch.testing.assert_close(mean_result, learned_result)

    def test_learned_scalar_nonuniform_differs_from_mean(self):
        """With non-zero logits, learned_scalar should differ from mean."""
        torch.manual_seed(42)
        K = 3
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for _ in range(K)
        ]

        mean_agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy="mean"
        )
        learned_agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=NUM_SLOTS, strategy="learned_scalar"
        )

        # Set non-uniform logits
        with torch.no_grad():
            learned_agg.block_logits.copy_(torch.tensor([2.0, 0.0, -2.0]))

        mean_result, _ = mean_agg(tensors)
        learned_result, _ = learned_agg(tensors)

        # Results should differ when weights are non-uniform
        assert not torch.allclose(mean_result, learned_result)

    def test_full_pipeline_parse_then_aggregate(self):
        """End-to-end: parse config, select weights, aggregate."""
        torch.manual_seed(42)
        # Simulate encoder output
        cwd = {
            idx: torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for idx in ALL_MOE_BLOCKS_SORTED
        }

        # Parse config
        block_indices = _parse_block_config([7, 9, 11], AVAILABLE_MOE_BLOCKS)
        assert block_indices == [7, 9, 11]

        # Create aggregator
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=len(block_indices),
            num_slots=NUM_SLOTS,
            strategy="learned_scalar",
        )

        # Get weights for loss
        result, contributions = _get_weights_for_loss(
            cwd, block_indices=block_indices, aggregator=agg
        )

        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        assert contributions is not None
        assert contributions.shape == (len(block_indices),)

    def test_full_pipeline_single_block_no_aggregator(self):
        """End-to-end: single block config skips aggregation."""
        torch.manual_seed(42)
        cwd = {
            idx: torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for idx in ALL_MOE_BLOCKS_SORTED
        }

        # Parse config
        block_indices = _parse_block_config("last", AVAILABLE_MOE_BLOCKS)
        assert len(block_indices) == 1

        # No aggregator for single block
        result, contributions = _get_weights_for_loss(
            cwd, block_indices=block_indices, aggregator=None
        )

        assert result.shape == (BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
        assert contributions is None
        torch.testing.assert_close(result, cwd[11])

    def test_two_losses_independent_aggregators(self):
        """Each loss should use its own independent aggregator."""
        torch.manual_seed(42)
        cwd = {
            idx: torch.randn(BATCH_SIZE, NUM_PATCHES, NUM_SLOTS)
            for idx in ALL_MOE_BLOCKS_SORTED
        }

        # Head loss uses all blocks
        head_indices = _parse_block_config("all", AVAILABLE_MOE_BLOCKS)
        head_agg = MultiBlockWeightAggregator(
            num_moe_blocks=len(head_indices),
            num_slots=NUM_SLOTS,
            strategy="learned_scalar",
        )

        # Pixel loss uses last 3 blocks
        pixel_indices = _parse_block_config([7, 9, 11], AVAILABLE_MOE_BLOCKS)
        pixel_agg = MultiBlockWeightAggregator(
            num_moe_blocks=len(pixel_indices),
            num_slots=NUM_SLOTS,
            strategy="learned_scalar",
        )

        head_result, head_contributions = _get_weights_for_loss(
            cwd, block_indices=head_indices, aggregator=head_agg
        )
        pixel_result, pixel_contributions = _get_weights_for_loss(
            cwd, block_indices=pixel_indices, aggregator=pixel_agg
        )

        # Different block selections produce different results
        assert head_result.shape == pixel_result.shape
        # Aggregators are independent (different block_logits counts)
        assert head_agg.block_logits.shape != pixel_agg.block_logits.shape
        assert head_contributions.shape == (6,)
        assert pixel_contributions.shape == (3,)

    @pytest.mark.parametrize("strategy", ["mean", "learned_scalar", "linear_projection"])
    def test_aggregator_with_sparse_slots(self, strategy):
        """Test with 8 slots (V1 sparse config: 8 experts * 1 slot_per_expert)."""
        torch.manual_seed(42)
        num_slots_sparse = 8
        K = 3
        tensors = [
            torch.randn(BATCH_SIZE, NUM_PATCHES, num_slots_sparse)
            for _ in range(K)
        ]
        agg = MultiBlockWeightAggregator(
            num_moe_blocks=K, num_slots=num_slots_sparse, strategy=strategy
        )
        result, _ = agg(tensors)
        assert result.shape == (BATCH_SIZE, NUM_PATCHES, num_slots_sparse)


# ── Entropy Regularization Block Selection Tests ─────────────────────

class TestEntropyBlockSelection:
    """Tests for the entropy loss smart block selection fix.

    When moe_regularize_all_blocks=false, entropy regularization should target
    the block(s) used for loss weighting (via moe_head_loss_block/moe_pixel_loss_block),
    NOT always default to the last block.
    """

    def _make_combine_weights_dict(self, num_experts=2):
        """Create a combine_weights_dict with distinguishable per-block weights."""
        cwd = {}
        for block_idx in [1, 3, 5, 7, 9, 11]:
            cwd[block_idx] = {
                'combine': torch.randn(4, 119, num_experts).abs(),
                'dispatch': torch.randn(4, 119, num_experts).abs(),
            }
            # Normalize so they look like probabilities
            cwd[block_idx]['dispatch'] = F.softmax(cwd[block_idx]['dispatch'], dim=1)
            cwd[block_idx]['combine'] = F.softmax(cwd[block_idx]['combine'], dim=1)
        return cwd

    def _make_base_cfg(self, head_block=None, pixel_block=None, regularize_all=False):
        """Create a config dict for testing entropy block selection."""
        cfg = {
            'model': {'student': {'use_mask_tokens': False, 'moe_num_experts': 2}},
            'losses': {
                'use_head_loss': True,
                'use_decoder_loss': False,
                'use_cls_loss': False,
                'head_loss_weight': 1.0,
                'pixel_loss_weight': 1.0,
                'cls_loss_weight': 1.0,
                'loss_weighting_method': 'literal',
                'head': {'type': 'smooth_l1', 'beta': 1.0},
                'decoder': {'type': 'l2'},
                'cls': {'type': 'cross_entropy'},
                'decoder_norm_pix_loss': True,
                'normalize_targets': True,
                'normalize_predictions': False,
                'normalization_method': 'variance',
                'moe_weight_head_loss': False,
                'moe_weight_pixel_loss': False,
                'moe_normalize_weights': True,
                'moe_normalize_per_image': True,
                'moe_weight_type': 'dispatch',
                'moe_loss_expert_indices': [0],
                'moe_regularize_all_blocks': regularize_all,
                'moe_regularize_all_experts': False,
                'use_importance_loss': False,
                'importance_loss_weight': 0.1,
                'use_dispatch_entropy_loss': True,
                'dispatch_entropy_loss_weight': 5.0,
                'moe_entropy_weight_type': 'dispatch',
                'moe_entropy_aggregation': 'separate',
            },
        }
        if head_block is not None:
            cfg['losses']['moe_head_loss_block'] = head_block
        if pixel_block is not None:
            cfg['losses']['moe_pixel_loss_block'] = pixel_block
        return cfg

    def test_no_block_config_uses_last_block(self):
        """Without moe_head_loss_block, entropy should use last block (backward compat)."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=None)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)  # all masked
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        # Should have last_block_only indicator
        assert 'entropy_last_block_only' in loss_dict

    def test_head_block_config_regularizes_that_block(self):
        """With moe_head_loss_block=5, entropy should regularize block 5."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        # Should have loss_blocks indicator (not last_block_only)
        assert 'entropy_loss_blocks' in loss_dict
        assert loss_dict['entropy_loss_blocks'] == 1.0  # 1 block
        # Should have block_5 specific metrics
        assert 'block_5_dispatch_entropy' in loss_dict
        # Should NOT have block_11 specific metrics
        assert 'block_11_dispatch_entropy' not in loss_dict

    def test_head_and_pixel_blocks_regularize_both(self):
        """With both head and pixel blocks set, entropy should regularize both."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5, pixel_block=9)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        assert 'entropy_loss_blocks' in loss_dict
        assert loss_dict['entropy_loss_blocks'] == 2.0  # 2 blocks
        assert 'block_5_dispatch_entropy' in loss_dict
        assert 'block_9_dispatch_entropy' in loss_dict

    def test_regularize_all_blocks_overrides_block_config(self):
        """moe_regularize_all_blocks=true should still regularize ALL blocks."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5, regularize_all=True)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        # Should regularize all blocks, not just the head block
        assert 'entropy_num_blocks' in loss_dict
        assert loss_dict['entropy_num_blocks'] == 6.0


class TestImportanceBlockSelection:
    """Test that importance loss targets the correct blocks based on config.

    Mirrors TestEntropyBlockSelection but for the importance loss path.
    Importance loss uses the same smart block selection pattern as entropy loss.
    """

    def _make_combine_weights_dict(self, num_experts=2):
        """Create a combine_weights_dict with distinguishable per-block weights."""
        cwd = {}
        for block_idx in [1, 3, 5, 7, 9, 11]:
            cwd[block_idx] = {
                'combine': torch.randn(4, 119, num_experts).abs(),
                'dispatch': torch.randn(4, 119, num_experts).abs(),
            }
            cwd[block_idx]['dispatch'] = F.softmax(cwd[block_idx]['dispatch'], dim=1)
            cwd[block_idx]['combine'] = F.softmax(cwd[block_idx]['combine'], dim=1)
        return cwd

    def _make_base_cfg(self, head_block=None, pixel_block=None, regularize_all=False):
        """Create config with importance loss ENABLED (entropy disabled)."""
        cfg = {
            'model': {'student': {'use_mask_tokens': False, 'moe_num_experts': 2}},
            'losses': {
                'use_head_loss': True,
                'use_decoder_loss': False,
                'use_cls_loss': False,
                'head_loss_weight': 1.0,
                'pixel_loss_weight': 1.0,
                'cls_loss_weight': 1.0,
                'loss_weighting_method': 'literal',
                'head': {'type': 'smooth_l1', 'beta': 1.0},
                'decoder': {'type': 'l2'},
                'cls': {'type': 'cross_entropy'},
                'decoder_norm_pix_loss': True,
                'normalize_targets': True,
                'normalize_predictions': False,
                'normalization_method': 'variance',
                'moe_weight_head_loss': False,
                'moe_weight_pixel_loss': False,
                'moe_normalize_weights': True,
                'moe_normalize_per_image': True,
                'moe_weight_type': 'dispatch',
                'moe_loss_expert_indices': [0],
                'moe_regularize_all_blocks': regularize_all,
                'moe_regularize_all_experts': False,
                'use_importance_loss': True,
                'importance_loss_weight': 0.1,
                'use_dispatch_entropy_loss': False,
                'dispatch_entropy_loss_weight': 0.0,
                'moe_entropy_weight_type': 'dispatch',
                'moe_entropy_aggregation': 'separate',
            },
        }
        if head_block is not None:
            cfg['losses']['moe_head_loss_block'] = head_block
        if pixel_block is not None:
            cfg['losses']['moe_pixel_loss_block'] = pixel_block
        return cfg

    def test_no_block_config_uses_last_block(self):
        """Without moe_head_loss_block, importance should use last block (backward compat)."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=None)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        assert 'L_importance' in loss_dict
        assert 'importance_last_block_only' in loss_dict

    def test_head_block_config_regularizes_that_block(self):
        """With moe_head_loss_block=5, importance should regularize block 5."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        assert 'L_importance' in loss_dict
        assert 'importance_loss_blocks' in loss_dict
        assert loss_dict['importance_loss_blocks'] == 1.0
        assert 'block_5_importance_cv' in loss_dict
        assert 'block_11_importance_cv' not in loss_dict

    def test_head_and_pixel_blocks_regularize_both(self):
        """With both head and pixel blocks set, importance should regularize both."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5, pixel_block=9)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        assert 'importance_loss_blocks' in loss_dict
        assert loss_dict['importance_loss_blocks'] == 2.0
        assert 'block_5_importance_cv' in loss_dict
        assert 'block_9_importance_cv' in loss_dict

    def test_regularize_all_blocks_overrides_block_config(self):
        """moe_regularize_all_blocks=true should still regularize ALL blocks."""
        from src.utils.losses import compute_three_losses

        cwd = self._make_combine_weights_dict()
        cfg = self._make_base_cfg(head_block=5, regularize_all=True)

        B, N, D = 4, 119, 768
        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, loss_dict = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        assert 'L_importance' in loss_dict
        # Should have per-block metrics for all 6 MoE blocks
        assert 'block_5_importance_cv' in loss_dict
        assert 'block_11_importance_cv' in loss_dict


class TestFusionConfigIntegration:
    """Test fusion experiment configs produce correct aggregator behavior."""

    def _make_combine_weights_dict(self, blocks, batch=2, tokens=10, experts=2):
        """Create mock combine_weights_dict matching ViT-Base MoE blocks."""
        return {
            idx: {
                'combine': torch.randn(batch, tokens, experts).softmax(dim=-1),
                'dispatch': torch.randn(batch, tokens, experts).softmax(dim=-1),
            }
            for idx in blocks
        }

    # --- learned_scalar [7, 9, 11] ---

    def test_ls_3blk_parse_config(self):
        """learned_scalar [7,9,11] config parses to correct block indices."""
        available = [1, 3, 5, 7, 9, 11]
        result = MEDiCModel._parse_block_config([7, 9, 11], available)
        assert result == [7, 9, 11]

    def test_ls_3blk_aggregator_creation(self):
        """learned_scalar with 3 blocks creates aggregator with 3 block_logits."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="learned_scalar")
        assert hasattr(agg, 'block_logits')
        assert agg.block_logits.shape == (3,)
        assert torch.allclose(agg.block_logits, torch.zeros(3))

    def test_ls_3blk_forward_shape(self):
        """learned_scalar [7,9,11] produces correct output shape."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="learned_scalar")
        weights_dict = self._make_combine_weights_dict([7, 9, 11])
        weights_list = [weights_dict[idx] for idx in [7, 9, 11]]
        result, contributions = agg(weights_list)
        assert 'combine' in result
        assert 'dispatch' in result
        assert result['combine'].shape == (2, 10, 2)
        assert contributions is not None
        assert contributions.shape == (3,)
        assert abs(contributions.sum().item() - 1.0) < 1e-5

    def test_ls_3blk_gradient_flow(self):
        """learned_scalar block_logits receive gradients."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="learned_scalar")
        weights_list = [torch.randn(2, 10, 2).softmax(dim=-1) for _ in range(3)]
        result, _ = agg._aggregate(weights_list)
        loss = result.sum()
        loss.backward()
        assert agg.block_logits.grad is not None
        assert agg.block_logits.grad.shape == (3,)

    # --- learned_scalar "all" (6 blocks) ---

    def test_ls_all_parse_config(self):
        """'all' config parses to all 6 MoE blocks."""
        available = [1, 3, 5, 7, 9, 11]
        result = MEDiCModel._parse_block_config("all", available)
        assert result == [1, 3, 5, 7, 9, 11]

    def test_ls_all_aggregator_creation(self):
        """learned_scalar with 6 blocks creates aggregator with 6 block_logits."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=6, num_slots=2, strategy="learned_scalar")
        assert agg.block_logits.shape == (6,)

    def test_ls_all_forward_shape(self):
        """learned_scalar all blocks produces correct output shape."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=6, num_slots=2, strategy="learned_scalar")
        weights_list = [torch.randn(2, 10, 2).softmax(dim=-1) for _ in range(6)]
        result, contributions = agg._aggregate(weights_list)
        assert result.shape == (2, 10, 2)
        assert contributions.shape == (6,)
        assert abs(contributions.sum().item() - 1.0) < 1e-5

    # --- linear_projection [7, 9, 11] ---

    def test_lp_3blk_aggregator_creation(self):
        """linear_projection with 3 blocks creates projection layer (6 -> 2)."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="linear_projection")
        assert hasattr(agg, 'projection')
        assert agg.projection.in_features == 6
        assert agg.projection.out_features == 2

    def test_lp_3blk_forward_shape(self):
        """linear_projection [7,9,11] produces correct output shape with softmax."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="linear_projection")
        weights_list = [torch.randn(2, 10, 2).softmax(dim=-1) for _ in range(3)]
        result, contributions = agg._aggregate(weights_list)
        assert result.shape == (2, 10, 2)
        row_sums = result.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        assert contributions is None

    def test_lp_3blk_gradient_flow(self):
        """linear_projection parameters receive gradients."""
        agg = MultiBlockWeightAggregator(num_moe_blocks=3, num_slots=2, strategy="linear_projection")
        weights_list = [torch.randn(2, 10, 2).softmax(dim=-1) for _ in range(3)]
        result, _ = agg._aggregate(weights_list)
        loss = result.sum()
        loss.backward()
        assert agg.projection.weight.grad is not None

    # --- V1 sparse safety: no pixel aggregator ---

    def test_v1_sparse_no_pixel_aggregator(self):
        """V1 sparse (use_mask_tokens=false) should NOT create pixel aggregator."""
        cfg = {
            'model': {
                'student': {
                    'name': 'vit_base_patch16', 'img_size': 224, 'patch_size': 16,
                    'embed_dim': 768, 'depth': 12, 'num_heads': 12,
                    'use_mask_tokens': False,
                    'use_soft_moe': True, 'moe_num_experts': 2,
                    'moe_slots_per_expert': 1, 'moe_mlp_ratio': 4.0,
                    'moe_placement': 'alternating',
                    'drop_path_rate': 0.0, 'init_values': None,
                    'use_abs_pos_emb': True, 'use_rel_pos_bias': False,
                    'use_shared_rel_pos_bias': False,
                },
                'teacher': {'name': 'ViT-B/16', 'embed_dim': 768},
                'decoder': {
                    'decoder_embed_dim': 512, 'decoder_depth': 8,
                    'decoder_num_heads': 16, 'replace_mask_tokens': True,
                    'use_sincos_pos_emb': True,
                },
            },
            'losses': {
                'moe_head_loss_block': [7, 9, 11],
                'moe_loss_block_aggregation': 'learned_scalar',
                'moe_weight_pixel_loss': False,
            },
            'mask': {'mask_ratio': 0.4, 'mask_type': 'block'},
        }
        model = MEDiCModel(cfg)
        assert model.head_weight_aggregator is not None
        assert model.head_block_indices == [7, 9, 11]
        assert model.pixel_weight_aggregator is None
        assert model.pixel_block_indices is None

    # --- Entropy loss block-count invariance ---

    def test_entropy_loss_mean_across_blocks(self):
        """Entropy loss uses mean (not sum) across blocks, making it block-count invariant."""
        from src.utils.losses import compute_three_losses

        B, N, D = 4, 119, 768
        all_blocks = [1, 3, 5, 7, 9, 11]
        cwd = {}
        for block_idx in all_blocks:
            cwd[block_idx] = {
                'combine': torch.randn(B, N, 2).abs(),
                'dispatch': torch.randn(B, N, 2).abs(),
            }
            cwd[block_idx]['dispatch'] = F.softmax(cwd[block_idx]['dispatch'], dim=1)
            cwd[block_idx]['combine'] = F.softmax(cwd[block_idx]['combine'], dim=1)

        base_cfg = {
            'model': {'student': {'use_mask_tokens': False, 'moe_num_experts': 2}},
            'losses': {
                'use_head_loss': True, 'use_decoder_loss': False, 'use_cls_loss': False,
                'head_loss_weight': 1.0, 'pixel_loss_weight': 1.0, 'cls_loss_weight': 1.0,
                'loss_weighting_method': 'literal',
                'head': {'type': 'smooth_l1', 'beta': 1.0},
                'decoder': {'type': 'l2'}, 'cls': {'type': 'cross_entropy'},
                'decoder_norm_pix_loss': True, 'normalize_targets': True,
                'normalize_predictions': False, 'normalization_method': 'variance',
                'moe_weight_head_loss': False, 'moe_weight_pixel_loss': False,
                'moe_normalize_weights': True, 'moe_normalize_per_image': True,
                'moe_weight_type': 'dispatch', 'moe_loss_expert_indices': [0],
                'moe_regularize_all_blocks': False, 'moe_regularize_all_experts': False,
                'use_importance_loss': False, 'importance_loss_weight': 0.1,
                'use_dispatch_entropy_loss': True, 'dispatch_entropy_loss_weight': 5.0,
                'moe_entropy_weight_type': 'dispatch', 'moe_entropy_aggregation': 'separate',
                'moe_head_loss_block': 11,
            },
        }
        cfg_multi = {
            'model': base_cfg['model'],
            'losses': {**base_cfg['losses'], 'moe_head_loss_block': [7, 9, 11]}
        }

        pred_tok = torch.randn(B, N + 1, D)
        T = torch.randn(B, N + 1, D)
        mask = torch.ones(B, 196)
        img = torch.randn(B, 3, 224, 224)

        _, l_single = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=base_cfg, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )
        _, l_multi = compute_three_losses(
            pred_tok=pred_tok, pred_pix=None, T=T, img=img, mask=mask,
            cfg=cfg_multi, use_mask_tokens=False,
            combine_weights=cwd[11]['dispatch'],
            combine_weights_dict=cwd,
        )

        assert 'L_dispatch_entropy' in l_single
        assert 'L_dispatch_entropy' in l_multi
        ent_single = l_single['L_dispatch_entropy']
        ent_multi = l_multi['L_dispatch_entropy']
        # Handle both tensor and float returns
        val_single = ent_single.item() if hasattr(ent_single, 'item') else float(ent_single)
        val_multi = ent_multi.item() if hasattr(ent_multi, 'item') else float(ent_multi)
        assert val_single != 0, "Single-block entropy loss should be non-zero"
        assert val_multi != 0, "Multi-block entropy loss should be non-zero"
        # If using mean aggregation (not sum), the multi-block entropy loss should be
        # in the same order of magnitude as single-block, not 3x larger.
        ratio = abs(val_multi) / abs(val_single)
        assert 0.1 < ratio < 10.0, f"Entropy loss ratio {ratio} suggests sum instead of mean"
