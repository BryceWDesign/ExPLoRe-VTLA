"""
Configuration validation utilities.
Validates configuration options at load time to catch errors early.
"""

from typing import Dict, Any, List, Optional


def validate_loss_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate loss configuration options.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    losses_cfg = cfg.get("losses", {})

    # Validate decoder loss type
    if losses_cfg.get("use_decoder_loss", False):
        pixel_loss_type = losses_cfg.get("decoder", {}).get("type", "l2")
        valid_pixel_types = ["l1", "l2", "smooth_l1"]
        if pixel_loss_type not in valid_pixel_types:
            errors.append(
                f"Invalid decoder loss type: '{pixel_loss_type}'. "
                f"Must be one of {valid_pixel_types}"
            )

    # Validate CLS loss type
    if losses_cfg.get("use_cls_loss", False):
        cls_loss_type = losses_cfg.get("cls", {}).get("type", "cross_entropy")
        valid_cls_types = ["cross_entropy", "cosine"]
        if cls_loss_type not in valid_cls_types:
            errors.append(
                f"Invalid CLS loss type: '{cls_loss_type}'. "
                f"Must be one of {valid_cls_types}"
            )

    # Validate loss weighting method
    loss_weighting_method = losses_cfg.get("loss_weighting_method", "literal")
    valid_weighting_methods = ["literal", "random", "gradnorm"]
    if loss_weighting_method not in valid_weighting_methods:
        errors.append(
            f"Invalid loss weighting method: '{loss_weighting_method}'. "
            f"Must be one of {valid_weighting_methods}"
        )

    # Validate normalization method
    norm_method = losses_cfg.get("normalization_method", "variance")
    valid_norm_methods = ["variance", "none"]
    if norm_method not in valid_norm_methods:
        errors.append(
            f"Invalid normalization method: '{norm_method}'. "
            f"Must be one of {valid_norm_methods}"
        )

    # Validate mask type
    mask_cfg = cfg.get("mask", {})
    mask_type = mask_cfg.get("mask_type", "block")
    valid_mask_types = ["block", "random", "evolved"]
    if mask_type not in valid_mask_types:
        errors.append(
            f"Invalid mask type: '{mask_type}'. "
            f"Must be one of {valid_mask_types}"
        )

    # Validate evolved masking specific options
    if mask_type == "evolved":
        evolved_cfg = mask_cfg.get("evolved", {})
        attention_source = evolved_cfg.get("attention_source", "student_online")
        valid_attention_sources = ["student_online", "student_ema", "clip_teacher"]
        if attention_source not in valid_attention_sources:
            errors.append(
                f"Invalid evolved masking attention source: '{attention_source}'. "
                f"Must be one of {valid_attention_sources}"
            )

    return errors


def validate_model_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration options.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    student_cfg = cfg.get("model", {}).get("student", {})

    # Validate position embedding compatibility
    use_mask_tokens = student_cfg.get("use_mask_tokens", True)
    use_rel_pos_bias = student_cfg.get("use_rel_pos_bias", False)
    use_shared_rel_pos_bias = student_cfg.get("use_shared_rel_pos_bias", False)

    # Check for conflicting position embeddings
    if use_rel_pos_bias and use_shared_rel_pos_bias:
        errors.append(
            "Cannot use both 'use_rel_pos_bias' and 'use_shared_rel_pos_bias' simultaneously"
        )

    # Note: Sparse mode + relative position bias compatibility was investigated
    # and found to be working correctly (Task 1), so no validation needed

    return errors


def validate_moe_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate MoE (Mixture of Experts) configuration options.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    student_cfg = cfg.get("model", {}).get("student", {})
    losses_cfg = cfg.get("losses", {})

    # Check if MoE is enabled
    use_soft_moe = student_cfg.get("use_soft_moe", False)
    if not use_soft_moe:
        # No MoE validations needed if MoE is disabled
        return errors

    # Get MoE configuration
    moe_num_experts = student_cfg.get("moe_num_experts", 2)
    moe_placement = student_cfg.get("moe_placement", "alternating")
    depth = student_cfg.get("depth", 12)
    use_mask_tokens = student_cfg.get("use_mask_tokens", True)

    # Get loss weighting flags
    moe_weight_head_loss = losses_cfg.get("moe_weight_head_loss", False)
    moe_weight_pixel_loss = losses_cfg.get("moe_weight_pixel_loss", False)

    # Validation 1: MoE loss weighting requires corresponding loss to be enabled
    if moe_weight_head_loss and not losses_cfg.get("use_head_loss", True):
        errors.append(
            "Configuration error: 'moe_weight_head_loss=true' requires 'use_head_loss=true'. "
            "Cannot weight a loss that is disabled."
        )

    if moe_weight_pixel_loss and not losses_cfg.get("use_decoder_loss", False):
        errors.append(
            "Configuration error: 'moe_weight_pixel_loss=true' requires 'use_decoder_loss=true'. "
            "Cannot weight a loss that is disabled."
        )

    # Validation 2: Check loss expert indices are valid
    loss_expert_indices = losses_cfg.get("moe_loss_expert_indices", None)

    if (moe_weight_head_loss or moe_weight_pixel_loss):
        # If loss weighting is enabled, validate expert configuration
        if loss_expert_indices is None:
            # Default behavior: use all experts for loss (backward compatible)
            loss_expert_indices = list(range(moe_num_experts))

        # Check that we have at least 1 expert for loss weighting
        if len(loss_expert_indices) == 0:
            errors.append(
                f"Configuration error: MoE loss weighting enabled but "
                f"moe_loss_expert_indices is empty. Need at least 1 expert for loss."
            )

        # Check that all indices are valid
        for idx in loss_expert_indices:
            if idx >= moe_num_experts:
                errors.append(
                    f"Configuration error: moe_loss_expert_indices contains invalid index {idx}. "
                    f"Model has {moe_num_experts} experts (valid indices: 0-{moe_num_experts-1})."
                )

        # Check for duplicates
        if len(loss_expert_indices) != len(set(loss_expert_indices)):
            errors.append(
                f"Configuration error: moe_loss_expert_indices contains duplicates: {loss_expert_indices}"
            )

    # Validation 3: MoE placement requires sufficient depth
    min_depth_required = {
        "alternating": 2,  # Need at least 2 blocks to alternate
        "second_half": 2,  # Need at least 2 blocks to have a second half
        "all": 1,  # Need at least 1 block
    }

    if moe_placement in min_depth_required:
        min_depth = min_depth_required[moe_placement]
        if depth < min_depth:
            errors.append(
                f"Configuration error: moe_placement='{moe_placement}' requires "
                f"depth >= {min_depth}. Got depth={depth}."
            )
    else:
        # Unknown placement strategy
        valid_placements = list(min_depth_required.keys())
        errors.append(
            f"Configuration error: Invalid moe_placement='{moe_placement}'. "
            f"Must be one of {valid_placements}."
        )

    # Validation 4: Sparse mode cannot use MoE weighting for pixel loss
    if moe_weight_pixel_loss and not use_mask_tokens:
        errors.append(
            "Configuration error: 'moe_weight_pixel_loss=true' is incompatible with "
            "'use_mask_tokens=false' (sparse mode). In sparse mode, the encoder only "
            "processes visible patches, but pixel loss operates on masked patches "
            "(disjoint sets). MoE weights are not available for masked patches. "
            "Either set use_mask_tokens=true (dense mode) or set moe_weight_pixel_loss=false."
        )

    # Validation 5: Per-expert entropy loss weights
    if losses_cfg.get('use_dispatch_entropy_loss', False):
        entropy_weight = losses_cfg.get('dispatch_entropy_loss_weight', 0.01)
        if isinstance(entropy_weight, list):
            # Per-expert weights provided
            if len(entropy_weight) != moe_num_experts:
                errors.append(
                    f"Configuration error: dispatch_entropy_loss_weight list length ({len(entropy_weight)}) "
                    f"must match moe_num_experts ({moe_num_experts})"
                )
            # Check all weights are non-negative
            if any(w < 0 for w in entropy_weight):
                errors.append(
                    "Configuration error: All dispatch_entropy_loss_weight values must be non-negative"
                )
        elif isinstance(entropy_weight, (int, float)):
            # Scalar weight - check it's non-negative
            if entropy_weight < 0:
                errors.append(
                    f"Configuration error: dispatch_entropy_loss_weight must be non-negative, got {entropy_weight}"
                )
        else:
            # Invalid type
            errors.append(
                f"Configuration error: dispatch_entropy_loss_weight must be a number or list of numbers, "
                f"got {type(entropy_weight).__name__}"
            )

    return errors


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate entire configuration and raise error if invalid.

    Args:
        cfg: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    all_errors = []

    # Validate loss configuration
    loss_errors = validate_loss_config(cfg)
    all_errors.extend(loss_errors)

    # Validate model configuration
    model_errors = validate_model_config(cfg)
    all_errors.extend(model_errors)

    # Validate MoE configuration
    moe_errors = validate_moe_config(cfg)
    all_errors.extend(moe_errors)

    # Raise error if any validation failed
    if all_errors:
        error_msg = "Configuration validation failed:\n"
        for i, error in enumerate(all_errors, 1):
            error_msg += f"  {i}. {error}\n"
        raise ValueError(error_msg)


def print_config_warnings(cfg: Dict[str, Any]) -> None:
    """
    Print warnings for potentially problematic but valid configurations.

    Args:
        cfg: Configuration dictionary
    """
    warnings = []
    losses_cfg = cfg.get("losses", {})

    # Warn about GradNorm with gradient accumulation
    if losses_cfg.get("loss_weighting_method") == "gradnorm":
        accum_iter = cfg.get("optim", {}).get("accum_iter", 1)
        if accum_iter > 1:
            warnings.append(
                f"Using GradNorm with gradient accumulation (accum_iter={accum_iter}). "
                "Make sure you're using the updated implementation that supports accumulation."
            )

    # Warn about evolved masking with high mask ratio
    mask_cfg = cfg.get("mask", {})
    if mask_cfg.get("mask_type") == "evolved":
        mask_ratio = mask_cfg.get("mask_ratio", 0.4)
        if mask_ratio < 0.7:
            warnings.append(
                f"Evolved masking typically works better with higher mask ratios (>=0.75). "
                f"Current ratio: {mask_ratio}"
            )

    # Print warnings if any
    if warnings:
        print("\n" + "="*60)
        print("CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
        print("="*60 + "\n")