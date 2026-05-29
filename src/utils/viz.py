"""
Visualization utilities for MEDiC pretraining.

Provides:
- Learning rate schedule plotting
- Masked image visualization (original + masked overlay)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional


# ImageNet normalization constants
IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])
MASK_GRAY_VALUE = 0.5
MASK_ALPHA = 0.8


def plot_scheduler(schedule: List[float], niter_per_ep: int,
                   caption: str = '') -> Figure:
    """
    Plot learning rate schedule (per-iteration and per-epoch).

    Args:
        schedule: List of LR values per iteration
        niter_per_ep: Number of iterations per epoch
        caption: Figure title

    Returns:
        matplotlib Figure
    """
    total_steps = len(schedule)
    full_epochs = total_steps // niter_per_ep
    remaining_iters = total_steps % niter_per_ep
    total_epochs = full_epochs + 1 if remaining_iters > 0 else full_epochs
    epochs_range = np.arange(0, total_epochs)

    lr_per_epoch: List[float] = []
    if full_epochs > 0:
        lr_per_epoch = [schedule[i * niter_per_ep] for i in range(full_epochs)]
        if remaining_iters > 0:
            lr_per_epoch.append(schedule[-1])
    elif remaining_iters > 0:
        lr_per_epoch = [schedule[-1]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    # Per-iteration plot
    ax1.plot(schedule, label="LR per Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Learning Rate Schedule (Iterations)")

    # Per-epoch plot
    if lr_per_epoch:
        ax2.plot(epochs_range, lr_per_epoch, label="LR per Epoch", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule (Epochs)")

    # Annotations for iteration plot
    if schedule:
        max_x, max_y = int(np.argmax(schedule)), max(schedule)
        min_x, min_y = int(np.argmin(schedule)), min(schedule)

        points = [
            ("First Iter", 0, schedule[0], "red"),
            ("Last Iter", total_steps - 1, schedule[-1], "green"),
            ("Max LR Iter", max_x, max_y, "purple"),
            ("Min LR Iter", min_x, min_y, "blue"),
        ]

        for label, x, y, color in points:
            ax1.plot(x, y, 'o', color=color, label=f"{label} ({x}, {y:.6f})")
            ax1.axhline(y=y, color='grey', linestyle='--', linewidth=0.5)
            ax1.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

    # Annotations for epoch plot
    if lr_per_epoch:
        max_x_ep, max_y_ep = int(np.argmax(lr_per_epoch)), max(lr_per_epoch)
        min_x_ep, min_y_ep = int(np.argmin(lr_per_epoch)), min(lr_per_epoch)

        points_epoch = [
            ("First Epoch", 0, lr_per_epoch[0], "red"),
            ("Last Epoch", total_epochs - 1, lr_per_epoch[-1], "green"),
            ("Max LR Epoch", max_x_ep, max_y_ep, "purple"),
            ("Min LR Epoch", min_x_ep, min_y_ep, "blue"),
        ]

        for label, x, y, color in points_epoch:
            ax2.plot(x, y, 'o', color=color, label=f"{label} ({x}, {y:.6f})")
            ax2.axhline(y=y, color='grey', linestyle='--', linewidth=0.5)
            ax2.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.suptitle(caption, fontsize=16)
    plt.tight_layout()
    return fig


@torch.no_grad()
def get_reconstruction_fig(model, batch, mask, epoch, n_images=4) -> Figure:
    """
    Generates a figure showing original and masked images.

    Shows 2 columns: Original | Masked (with gray overlay on masked patches).
    When pixel decoder is available, also shows reconstructed patches.

    Args:
        model: MEDiCModel instance
        batch: (images, labels) tuple
        mask: Patch-level mask [B, N] (True = masked)
        epoch: Current epoch number
        n_images: Number of images to display

    Returns:
        matplotlib Figure
    """
    model.eval()

    if batch is None or not batch or len(batch) < 2:
        raise ValueError("Invalid batch format. Expected (images, labels) tuple.")

    images, _ = batch
    if images is None or images.numel() == 0:
        raise ValueError("Invalid images in batch.")

    images = images[:n_images].to(mask.device)
    mask = mask[:n_images]

    # Handle 3D masks (pixel-level) by converting to patch-level
    if mask.dim() == 3:
        patch_size = model.student.patch_embed.patch_size[0]
        batch_size, height, width = mask.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        mask = mask.float().view(batch_size, num_patches_h, patch_size, num_patches_w, patch_size)
        mask = mask.mean(dim=(2, 4))
        mask = mask.view(batch_size, num_patches_h * num_patches_w)
        mask = (mask > 0.5).float()

    # Denormalize images for display
    orig_imgs = (
        torch.einsum("nchw->nhwc", images).cpu() * IMAGENET_DEFAULT_STD
        + IMAGENET_DEFAULT_MEAN
    )

    # Create mask overlay
    patch_size = model.student.patch_embed.patch_size[0]
    mask_expanded = mask.unsqueeze(-1).float().repeat(1, 1, patch_size**2 * 3)
    unpatchified_mask = model.student.unpatchify(mask_expanded)
    mask_img = torch.einsum("nchw->nhwc", unpatchified_mask).cpu()

    # Ensure mask matches image size
    target_size = orig_imgs.shape[1:3]
    if mask_img.shape[1:3] != target_size:
        mask_img = torch.nn.functional.interpolate(
            mask_img.permute(0, 3, 1, 2),
            size=target_size,
            mode='nearest',
        ).permute(0, 2, 3, 1)

    # Overlay: visible patches shown normally, masked patches grayed out
    masked_imgs = (
        orig_imgs * (1 - mask_img * MASK_ALPHA)
        + (MASK_GRAY_VALUE * MASK_ALPHA) * mask_img
    )

    min_batch = min(orig_imgs.shape[0], mask_img.shape[0])

    fig, axs = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))
    axs = np.atleast_2d(axs)
    fig.suptitle(f"Epoch {epoch} Masking Visualization", fontsize=14)

    for i in range(n_images):
        for j in range(2):
            if i < min_batch:
                if j == 0:
                    axs[i, j].imshow(torch.clip(orig_imgs[i], 0, 1).numpy())
                    axs[i, j].set_title("Original")
                else:
                    axs[i, j].imshow(torch.clip(masked_imgs[i], 0, 1).numpy())
                    axs[i, j].set_title("Masked")
            axs[i, j].axis("off")

    return fig


def fig_to_pil(fig) -> "Image.Image":
    """Convert a matplotlib figure to a PIL Image."""
    from io import BytesIO
    from PIL import Image as PILImage
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return PILImage.open(buf)


@torch.no_grad()
def log_attention_plots_to_wandb(
    model_without_ddp: torch.nn.Module,
    viz_images: torch.Tensor,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
):
    """Generate and log attention visualizations to W&B."""
    import wandb
    from .attention_viz import (
        AttentionCapture,
        plot_attention_matrices_comparison,
        plot_head_evolution_by_layer,
        plot_layer_evolution,
    )
    from torchvision.transforms import ToPILImage

    viz_cfg = cfg.get("visualizations", {})
    if not viz_cfg.get("enable", False):
        return

    student_model = model_without_ddp.student
    attention_capture = AttentionCapture(student_model)

    mean = torch.tensor([0.485, 0.456, 0.406], device=viz_images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=viz_images.device).view(1, 3, 1, 1)

    all_attention_maps = {}
    pil_images = []

    for i in range(viz_images.shape[0]):
        img_tensor = viz_images[i].unsqueeze(0)
        img_name = f"image_{i}"

        img_denorm = (img_tensor * std) + mean
        img_pil = ToPILImage()(img_denorm.squeeze(0).cpu())
        pil_images.append(img_pil)

        try:
            with torch.autocast(
                device_type="cuda" if img_tensor.device.type == "cuda" else "cpu",
                enabled=True,
            ):
                _, attention_maps = attention_capture.forward(img_tensor)
            all_attention_maps[img_name] = attention_maps
        except NotImplementedError as e:
            if "fused attention" in str(e).lower():
                return  # Skip if fused attention blocks capture
            raise

    layers = viz_cfg.get("layers_to_analyze", [0, 5, 11])
    plots_to_run = viz_cfg.get("plots", {})

    if plots_to_run.get("raw_attention", False):
        fig = plot_attention_matrices_comparison(
            all_attention_maps,
            layers_to_analyze=layers,
            suptitle=f"Epoch {epoch}: Attention Matrix Comparison",
            mode="wandb",
        )
        if fig:
            wandb.log({"Attention/Matrices": wandb.Image(fig_to_pil(fig))}, step=global_step)
            plt.close(fig)

    for i, img_name in enumerate(all_attention_maps.keys()):
        viz_img_name = f"image_{i}"

        if plots_to_run.get("layer_evolution", False):
            fig = plot_layer_evolution(
                all_attention_maps[img_name],
                pil_images[i],
                layers_to_analyze=layers,
                suptitle=f"Epoch {epoch}: {img_name} Mean Attention Evolution",
                mode="wandb",
            )
            if fig:
                wandb.log(
                    {f"Layer_Evolution/{viz_img_name}/mean_attention_evolution": wandb.Image(fig_to_pil(fig))},
                    step=global_step,
                )
                plt.close(fig)

        if plots_to_run.get("head_comparison", False):
            heads = viz_cfg.get("attention_heads_to_show", 4)
            num_model_heads = student_model.blocks[0].attn.num_heads
            heads_to_show = np.linspace(
                0, num_model_heads - 1, min(heads, num_model_heads), dtype=int
            ).tolist()

            figs = plot_head_evolution_by_layer(
                all_attention_maps[img_name],
                pil_images[i],
                layers_to_analyze=layers,
                heads_to_show=heads_to_show,
                suptitle=f"Epoch {epoch}: {img_name} Head Evolution",
                mode="wandb",
            )
            if figs and isinstance(figs, dict):
                for key, fig in figs.items():
                    wandb.log(
                        {f"Head_Evolution/{viz_img_name}/{key}": wandb.Image(fig_to_pil(fig))},
                        step=global_step,
                    )
                    plt.close(fig)


def get_config_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    Example: get_config_value(cfg, "model.student.use_mask_tokens", True)
    """
    keys = key.split('.')
    value = cfg
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def get_moe_weight_fig(
    model, batch, mask, epoch: int, cfg: Dict[str, Any], n_images: int = 4,
    show_visible: bool = True, masked_indicator: str = 'diagonal',
    apply_loss_normalization: bool = False
) -> Dict[int, Figure]:
    """
    Generates separate figures for each MoE block visualizing expert weights per patch.

    Shows per-patch expert weights with inferno colormap overlay (same as Head Evolution).
    By default only visualizes masked patches where MoE weighting is actually applied in loss computation.
    Visible patches remain transparent showing the original image.

    Args:
        model: MEDiC_Model (unwrapped, not DDP)
        batch: (images, labels) tuple
        mask: Binary mask [B, N] indicating masked patches (1 = masked, 0 = visible)
        epoch: Current epoch number
        cfg: Configuration dictionary
        n_images: Number of images to visualize (default: 4)
        show_visible: If True, show weight overlay for visible patches too (lighter alpha) (default: False)
        masked_indicator: Visual indicator for masked patches: 'border', 'diagonal', 'both', 'none' (default: 'none')
        apply_loss_normalization: If True, apply patch-level normalization (as done in loss computation) (default: False)

    Returns:
        Dictionary mapping block_idx -> matplotlib Figure (one figure per MoE block)
        Returns None if MoE is not enabled or no experts to visualize
    """
    # Check if MoE is enabled
    use_soft_moe = cfg.get("model", {}).get("student", {}).get("use_soft_moe", False)
    if not use_soft_moe:
        return None

    # Get loss expert indices from config
    loss_expert_indices = get_config_value(cfg, "losses.moe_loss_expert_indices", None)
    num_experts_cfg = get_config_value(cfg, "model.student.moe_num_experts", 2)

    # Only visualize semantically meaningful experts (first 2: distillation + reconstruction)
    # Additional experts (2, 3, ..., N-1) are for collapse prevention and lack semantic interpretation
    if loss_expert_indices is None:
        # Default: first 2 experts only, regardless of total expert count
        # This ensures consistent behavior whether loss weighting is enabled or disabled
        loss_expert_indices = list(range(min(2, num_experts_cfg)))

    experts_to_show = loss_expert_indices

    # Generate expert names (all shown experts are loss-weighted)
    expert_names = []
    for expert_idx in experts_to_show:
        if expert_idx == 0:
            expert_names.append(f"Expert {expert_idx} (Distillation)")
        elif expert_idx == 1:
            expert_names.append(f"Expert {expert_idx} (Reconstruction)")
        else:
            expert_names.append(f"Expert {expert_idx}")

    # Extract images from batch
    if batch is None or len(batch) < 2:
        raise ValueError("Invalid batch format. Expected (images, labels) tuple.")

    images, _ = batch
    if images is None or images.numel() == 0:
        raise ValueError("Invalid images in batch.")

    # Limit to n_images
    images = images[:n_images].to(mask.device)
    mask = mask[:n_images]

    # VIZ-SPECIFIC: Directly call student encoder to get weights from ALL MoE blocks
    # This bypasses medic_model wrapper's loss-dependent logic
    # No impact on training performance - this only runs during visualization
    model.eval()
    with torch.no_grad():
        # Call encoder directly with return_combine_weights=True
        # Encoder returns: (embeddings, mask/ids_restore, combine_weights_last_block, combine_weights_dict)
        _, _, _, combine_weights_dict = model.student(
            images, mask,
            return_combine_weights=True  # Always collect for visualization
        )

    # Validate combine_weights_dict
    if combine_weights_dict is None or not isinstance(combine_weights_dict, dict):
        print("Warning: combine_weights_dict is None despite MoE being enabled")
        return None

    if len(combine_weights_dict) == 0:
        print("Warning: combine_weights_dict is empty")
        return None

    # Sort block indices for consistent visualization
    block_indices = sorted(combine_weights_dict.keys())
    num_blocks = len(block_indices)

    # Helper function to extract weights from dict or tensor (backward compatibility)
    def extract_weights(weights_or_dict, weight_type='combine'):
        """Extract weights, handling both dict and tensor formats."""
        if isinstance(weights_or_dict, dict):
            # New format: dict with 'combine' and 'dispatch'
            if weight_type not in weights_or_dict:
                # Fallback to combine if requested type not found
                weight_type = 'combine'
            return weights_or_dict[weight_type]
        else:
            # Old format: tensor (treat as combine weights)
            return weights_or_dict

    # Get weight type from config (default to 'combine' for visualization)
    # Use same weight type as loss weighting if configured, otherwise use combine
    weight_type_for_viz = get_config_value(cfg, "losses.moe_weight_type", "combine")

    # DEBUG: Log statistics about MoE weights across all blocks
    print(f"\nMoE Weight Statistics (All {num_blocks} Blocks, weight_type={weight_type_for_viz}):")
    print(f"   Block indices: {block_indices}")
    for block_idx in block_indices:
        weights = extract_weights(combine_weights_dict[block_idx], weight_type_for_viz)
        print(f"   Block {block_idx}: shape={weights.shape}, "
              f"range=[{weights.min():.4f}, {weights.max():.4f}], "
              f"Expert 0 mean={weights[:, :, 0].mean():.4f}, "
              f"Expert 1 mean={weights[:, :, 1].mean():.4f}")

    # Get use_mask_tokens to determine if sparse mode
    use_mask_tokens = get_config_value(cfg, "model.student.use_mask_tokens", True)
    B, N = mask.shape

    # Get shuffling-related info from encoder (if sparse + shuffling)
    ids_restore = None
    visible_positions_shuffled = None
    if not use_mask_tokens:  # Sparse mode
        # Check if shuffling is enabled
        if hasattr(model.student, 'ids_restore') and model.student.ids_restore is not None:
            ids_restore = model.student.ids_restore
            # Get visible positions in shuffled array (needed for correct unshuffling)
            if hasattr(model.student, 'visible_positions_shuffled'):
                visible_positions_shuffled = model.student.visible_positions_shuffled

    # Process ALL blocks' weights: handle CLS token removal, unshuffling, and sparse mode expansion
    processed_weights = {}
    for block_idx in block_indices:
        # Extract weights (handle both dict and tensor formats)
        dispatch_weights = extract_weights(combine_weights_dict[block_idx], weight_type_for_viz)

        # Remove CLS token (always at position 0 in both dense and sparse modes)
        # Encoder always adds CLS (vision_transformer_mim.py:686-687, 779-780)
        if dispatch_weights.size(1) > 0:
            dispatch_weights = dispatch_weights[:, 1:, :]  # Remove CLS token

        # Handle sparse mode + shuffling: unshuffle weights to spatial order
        # This fixes the bug where block masking + shuffling causes weights to be in shuffled order
        if not use_mask_tokens and ids_restore is not None and visible_positions_shuffled is not None:
            # Sparse + shuffling mode: weights are in shuffled-visible order
            # Need to: 1) Create full shuffled array, 2) Scatter visible weights, 3) Unshuffle
            num_visible = dispatch_weights.size(1)
            num_experts = dispatch_weights.size(2)

            # Create full shuffled array filled with zeros [B, N, num_experts]
            weights_full_shuffled = torch.zeros(B, N, num_experts, device=dispatch_weights.device, dtype=dispatch_weights.dtype)

            # Scatter visible weights at their correct positions in shuffled array
            # visible_positions_shuffled: [B, num_visible] - positions in shuffled array
            batch_indices = torch.arange(B, device=weights_full_shuffled.device).unsqueeze(1).expand(-1, num_visible)
            weights_full_shuffled[batch_indices, visible_positions_shuffled] = dispatch_weights

            # Unshuffle to spatial order using ids_restore
            # ids_restore: [N] (with shuffling) or [B, N] (without shuffling)
            # We need to gather from shuffled array using ids_restore
            # Handle both 1D (shuffling enabled) and 2D (no shuffling) cases
            if ids_restore.dim() == 1:
                ids_restore_batch = ids_restore.unsqueeze(0).expand(B, -1)  # [N] -> [B, N]
            else:
                ids_restore_batch = ids_restore  # Already [B, N]
            dispatch_weights = torch.gather(
                weights_full_shuffled,
                dim=1,
                index=ids_restore_batch.unsqueeze(-1).expand(-1, -1, num_experts)
            )
            # Now dispatch_weights is [B, N, num_experts] in SPATIAL order

        # Handle sparse mode without shuffling: expand dispatch_weights to full patch size
        elif not use_mask_tokens and dispatch_weights.size(1) < N:
            # Sparse mode (no shuffling): dispatch_weights only contains visible patches
            # Need to expand to full size by inserting zeros for masked positions
            num_visible = dispatch_weights.size(1)
            num_experts = dispatch_weights.size(2)

            # Create full-size weight tensor filled with zeros
            full_weights = torch.zeros(B, N, num_experts, device=dispatch_weights.device, dtype=dispatch_weights.dtype)

            # Get indices of visible patches (mask == 0 for visible)
            visible_mask = (mask == 0).float()  # [B, N]

            # For each batch, insert dispatch_weights at visible positions
            for b in range(B):
                visible_indices = torch.where(visible_mask[b] == 1)[0]  # Get visible patch indices
                if len(visible_indices) == num_visible:
                    full_weights[b, visible_indices, :] = dispatch_weights[b, :, :]

            dispatch_weights = full_weights

        # Validate shapes match after expansion
        assert dispatch_weights.size(1) == N, \
            f"Block {block_idx}: After expansion, dispatch_weights {dispatch_weights.shape} must match mask size [B, {N}]"

        processed_weights[block_idx] = dispatch_weights

    # Apply loss-level normalization if requested (same as in losses.py:854-879)
    if apply_loss_normalization:
        # For each block and each loss expert, normalize weights to MEAN=1.0 across masked patches
        # This EXACTLY matches how weights are applied in loss computation

        # Read normalization mode from config (matches losses.py line 1270)
        normalize_per_image = cfg.get("losses", {}).get("moe_normalize_per_image", True)

        mask_tensor = mask  # [B, N]
        B = mask_tensor.size(0)
        use_mask_tokens = cfg.get("model", {}).get("student", {}).get("use_mask_tokens", True)

        for block_idx in block_indices:
            weights = processed_weights[block_idx]  # [B, N, num_experts]
            normalized_weights = weights.clone()

            for expert_idx in loss_expert_indices:
                # Extract this expert's weights for all batches
                expert_weights = weights[:, :, expert_idx]  # [B, N]

                # Apply mask: select patches where loss is computed
                # Dense mode: weight MASKED patches (mask==1, head loss on masked)
                # Sparse mode: weight VISIBLE patches (mask==0, head loss on visible)
                if use_mask_tokens:
                    expert_weights_masked = expert_weights * mask_tensor.float()  # Dense: masked patches
                else:
                    expert_weights_masked = expert_weights * (~mask_tensor).float()  # Sparse: visible patches

                # Normalize to MEAN=1.0 (not sum=1.0) to match loss calculation
                # This preserves the magnitude of the loss
                if normalize_per_image:
                    # PER-IMAGE NORMALIZATION (matches losses.py lines 887-922)
                    # Each image normalized independently for stable training
                    expert_weights_normalized = expert_weights_masked.clone()

                    # Count masked/visible patches per image
                    if use_mask_tokens:
                        mask_count_per_image = mask_tensor.float().sum(dim=1, keepdim=True)  # [B, 1]
                    else:
                        mask_count_per_image = (~mask_tensor).float().sum(dim=1, keepdim=True)  # [B, 1]

                    # Normalize weights so mean = 1.0 per image
                    weight_mean = expert_weights_masked.sum(dim=1, keepdim=True) / (mask_count_per_image + 1e-8)  # [B, 1]

                    # Check for valid normalization per image
                    valid_images = (mask_count_per_image.squeeze(1) > 0) & (weight_mean.squeeze(1) > 1e-8)

                    if valid_images.all():
                        expert_weights_normalized = expert_weights_masked / (weight_mean + 1e-8)
                    elif valid_images.any():
                        # Some images have no masked/visible patches - handle separately
                        expert_weights_normalized[valid_images] = (
                            expert_weights_masked[valid_images] / weight_mean[valid_images]
                        )
                        # For invalid images, use uniform weighting as fallback
                        invalid_images = ~valid_images
                        if invalid_images.any():
                            fallback_mask = mask_tensor.float() if use_mask_tokens else (~mask_tensor).float()
                            expert_weights_normalized[invalid_images] = fallback_mask[invalid_images]
                    else:
                        # All images invalid - use uniform weighting
                        expert_weights_normalized = mask_tensor.float() if use_mask_tokens else (~mask_tensor).float()
                else:
                    # BATCH-WIDE NORMALIZATION (legacy mode, matches losses.py lines 924-944)
                    # All images share same normalization factor
                    if use_mask_tokens:
                        mask_count = mask_tensor.float().sum()
                    else:
                        mask_count = (~mask_tensor).float().sum()

                    if mask_count > 0:
                        weight_mean = expert_weights_masked.sum() / mask_count
                        if weight_mean > 1e-8:
                            expert_weights_normalized = expert_weights_masked / weight_mean
                        else:
                            expert_weights_normalized = expert_weights_masked
                    else:
                        expert_weights_normalized = expert_weights_masked

                # Put normalized weights back
                normalized_weights[:, :, expert_idx] = expert_weights_normalized

            processed_weights[block_idx] = normalized_weights

        # Validation: Verify normalization matches expected behavior
        print(f"\n[MoE Normalization Validation] Block {block_idx}")
        print(f"  Mode: {'PER-IMAGE' if normalize_per_image else 'BATCH-WIDE'}")

        for expert_idx in loss_expert_indices:
            expert_weights_norm = normalized_weights[:, :, expert_idx]  # [B, N]

            if normalize_per_image:
                # Validate per-image means = 1.0
                for img_idx in range(B):
                    if use_mask_tokens:
                        masked_weights_img = expert_weights_norm[img_idx] * mask_tensor[img_idx].float()
                        mask_count_img = mask_tensor[img_idx].float().sum()
                    else:
                        masked_weights_img = expert_weights_norm[img_idx] * (~mask_tensor[img_idx]).float()
                        mask_count_img = (~mask_tensor[img_idx]).float().sum()

                    if mask_count_img > 0:
                        per_image_mean = masked_weights_img.sum() / mask_count_img
                        print(f"  Expert {expert_idx}, Image {img_idx}: mean = {per_image_mean:.6f} (expected 1.0)")
                        if abs(per_image_mean - 1.0) > 0.01:
                            print(f"    WARNING: Per-image normalization off by {abs(per_image_mean - 1.0):.4f}")
            else:
                # Validate batch-wide mean = 1.0
                if use_mask_tokens:
                    masked_weights = expert_weights_norm * mask_tensor.float()
                    mask_count_val = mask_tensor.float().sum()
                else:
                    masked_weights = expert_weights_norm * (~mask_tensor).float()
                    mask_count_val = (~mask_tensor).float().sum()

                if mask_count_val > 0:
                    actual_mean = masked_weights.sum() / mask_count_val
                    print(f"  Expert {expert_idx}: batch-wide mean = {actual_mean:.6f} (expected 1.0)")
                    if abs(actual_mean - 1.0) > 0.01:
                        print(f"    WARNING: Batch-wide normalization off by {abs(actual_mean - 1.0):.4f}")

    # Get dimensions
    B, _, H, W = images.shape
    patch_size = 16  # Standard ViT patch size
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Denormalize images for visualization
    mean = IMAGENET_DEFAULT_MEAN.view(1, 3, 1, 1).to(images.device)
    std = IMAGENET_DEFAULT_STD.view(1, 3, 1, 1).to(images.device)
    imgs_denorm = images * std + mean
    imgs_denorm = torch.clamp(imgs_denorm, 0, 1)
    imgs_np = imgs_denorm.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]

    # Reshape mask to grid (weights will be reshaped per-block in the loop)
    mask_grid = mask.view(B, grid_h, grid_w).cpu().numpy()

    # Use inferno colormap, but trim the lower range to make purple start at 0.0
    # Standard inferno: 0.0=black, 0.2=purple, 1.0=yellow
    # Trimmed inferno: 0.0=purple, 1.0=yellow (more visible low weights)
    from matplotlib.colors import LinearSegmentedColormap
    inferno_full = plt.get_cmap('inferno')
    # Sample from 0.2 to 1.0 of the original colormap
    colors = inferno_full(np.linspace(0.2, 1.0, 256))
    cmap = LinearSegmentedColormap.from_list('inferno_trimmed', colors)

    # Get MoE placement info for titles
    moe_placement = getattr(model.student, 'moe_placement', 'unknown')

    # Show first image only
    img_idx = 0

    # Create one figure per MoE block
    figures = {}

    for block_idx in block_indices:
        # Create figure: 1 row, 2 columns (one per expert)
        n_cols = len(experts_to_show)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))

        # Handle single expert case
        if n_cols == 1:
            axes = [axes]

        # Get weights for this block
        patch_weights_grid_block = processed_weights[block_idx].view(B, grid_h, grid_w, -1).cpu().numpy()

        # Compute PER-EXPERT min/max for independent color scaling
        # This makes each expert's variation visible regardless of absolute magnitude
        # For loss-normalized weights: only consider patches where loss is computed
        # Dense mode: masked patches (mask==1), Sparse mode: visible patches (mask==0)
        # For softmax weights: consider all patches
        expert_vmin = {}  # Per-expert min values
        expert_vmax = {}  # Per-expert max values
        mask_map = mask_grid[img_idx, :, :]  # [grid_h, grid_w]
        use_mask_tokens = cfg.get("model", {}).get("student", {}).get("use_mask_tokens", True)

        for expert_idx in experts_to_show:
            weight_map = patch_weights_grid_block[img_idx, :, :, expert_idx]

            if apply_loss_normalization:
                # Select patches where loss is computed
                if use_mask_tokens:
                    relevant_weights = weight_map[mask_map == 1]  # Dense: masked patches
                else:
                    relevant_weights = weight_map[mask_map == 0]  # Sparse: visible patches
                masked_weights = relevant_weights
                if len(masked_weights) > 0:
                    expert_vmin[expert_idx] = masked_weights.min()
                    expert_vmax[expert_idx] = masked_weights.max()
                else:
                    expert_vmin[expert_idx] = 0.0
                    expert_vmax[expert_idx] = 1.0
            else:
                # Consider all patches (softmax weights are meaningful for all patches)
                expert_vmin[expert_idx] = weight_map.min()
                expert_vmax[expert_idx] = weight_map.max()

        # Compute mean across all loss experts for annotation
        all_weights = []
        for expert_idx in experts_to_show:
            weight_map = patch_weights_grid_block[img_idx, :, :, expert_idx]
            if apply_loss_normalization:
                # Only include masked patches in mean calculation
                masked_weights = weight_map[mask_map == 1]
                if len(masked_weights) > 0:
                    all_weights.append(masked_weights.flatten())
            else:
                # Include all patches
                all_weights.append(weight_map.flatten())

        global_mean = np.mean(np.concatenate(all_weights)) if all_weights else 0.0

        for col_idx, expert_idx in enumerate(experts_to_show):
            ax = axes[col_idx]

            # Show original image as background
            ax.imshow(imgs_np[img_idx])

            # Create discrete per-patch weight visualization [grid_h, grid_w]
            weight_map = patch_weights_grid_block[img_idx, :, :, expert_idx]
            mask_map = mask_grid[img_idx, :, :]

            # Create RGB overlay with per-patch coloring (NO interpolation)
            # Each patch gets a single color from the inferno colormap based on its weight
            overlay = np.zeros((H, W, 4))  # RGBA

            for i in range(grid_h):
                for j in range(grid_w):
                    # Get weight for this patch
                    weight = weight_map[i, j]

                    # Normalize weight to [0, 1] based on THIS EXPERT'S min/max (per-expert color scaling)
                    expert_vmin_val = expert_vmin[expert_idx]
                    expert_vmax_val = expert_vmax[expert_idx]
                    if expert_vmax_val > expert_vmin_val:
                        weight_normalized = (weight - expert_vmin_val) / (expert_vmax_val - expert_vmin_val)
                    else:
                        weight_normalized = 0.5  # Fallback if all weights are identical

                    # Get color from inferno colormap
                    color = cmap(weight_normalized)  # Returns RGBA tuple

                    # Patch coordinates
                    y_start = i * patch_size
                    y_end = (i + 1) * patch_size
                    x_start = j * patch_size
                    x_end = (j + 1) * patch_size

                    # Fill this patch with the color
                    overlay[y_start:y_end, x_start:x_end] = color

            # Apply different alpha for masked vs visible patches
            for i in range(grid_h):
                for j in range(grid_w):
                    y_start = i * patch_size
                    y_end = (i + 1) * patch_size
                    x_start = j * patch_size
                    x_end = (j + 1) * patch_size

                    if mask_map[i, j] == 1:  # Masked patch
                        # Full opacity for masked patches
                        overlay[y_start:y_end, x_start:x_end, 3] = overlay[y_start:y_end, x_start:x_end, 3] * 0.5
                    else:  # Visible patch
                        if show_visible:
                            # Lighter opacity for visible patches
                            overlay[y_start:y_end, x_start:x_end, 3] = overlay[y_start:y_end, x_start:x_end, 3] * 0.3
                        else:
                            # No overlay for visible patches (original behavior)
                            overlay[y_start:y_end, x_start:x_end, 3] = 0.0

            # Overlay on image
            ax.imshow(overlay)

            # Add visual indicators for MASKED patches
            if masked_indicator in ['border', 'both']:
                # Draw borders around MASKED patches
                for i in range(grid_h):
                    for j in range(grid_w):
                        if mask_map[i, j] == 1:  # Masked patch
                            y_start = i * patch_size
                            x_start = j * patch_size
                            rect = plt.Rectangle((x_start - 0.5, y_start - 0.5), patch_size, patch_size,
                                                fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                            ax.add_patch(rect)

            if masked_indicator in ['diagonal', 'both']:
                # Draw diagonal lines on MASKED patches
                for i in range(grid_h):
                    for j in range(grid_w):
                        if mask_map[i, j] == 1:  # Masked patch
                            y_start = i * patch_size
                            y_end = (i + 1) * patch_size
                            x_start = j * patch_size
                            x_end = (j + 1) * patch_size

                            # Draw single diagonal line
                            ax.plot([x_start, x_end], [y_start, y_end],
                                   color='red', linewidth=0.8, alpha=0.7)

            # Draw grid lines
            for i in range(grid_h + 1):
                ax.axhline(y=i * patch_size - 0.5, color='white', linewidth=0.5, alpha=0.3)
            for j in range(grid_w + 1):
                ax.axvline(x=j * patch_size - 0.5, color='white', linewidth=0.5, alpha=0.3)

            # Calculate per-expert statistics for this image
            if apply_loss_normalization:
                # Only consider masked patches
                masked_weights = weight_map[mask_map == 1]
                if len(masked_weights) > 0:
                    exp_mean = masked_weights.mean()
                    exp_std = masked_weights.std()
                    exp_min = masked_weights.min()
                    exp_max = masked_weights.max()
                else:
                    exp_mean = exp_std = exp_min = exp_max = 0.0
            else:
                # Consider all patches
                exp_mean = weight_map.mean()
                exp_std = weight_map.std()
                exp_min = weight_map.min()
                exp_max = weight_map.max()

            # Title with per-expert statistics
            ax.set_title(
                f"{expert_names[col_idx]}\n"
                f"mu={exp_mean:.3f} sigma={exp_std:.3f} [{exp_min:.3f}, {exp_max:.3f}]",
                fontsize=11, fontweight='bold', color='green'
            )

            # Add green border to all subplots
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)

            # Add individual colorbar for THIS expert with its own range
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            expert_vmin_val = expert_vmin[expert_idx]
            expert_vmax_val = expert_vmax[expert_idx]
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=expert_vmin_val, vmax=expert_vmax_val))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'Weight\n[{expert_vmin_val:.4f}, {expert_vmax_val:.4f}]',
                         rotation=270, labelpad=20)

            ax.axis('off')

        # Add enhanced figure title with statistics
        if apply_loss_normalization:
            # For loss-normalized: weights are normalized to mean=1.0 across masked patches
            uniform_weight = 1.0  # Target mean after normalization
            norm_desc = "Loss-Normalized (Batch-Wide Mean=1.0)"
            note = "Per-image mu shows expert specialization: <1.0 = low weight, >1.0 = high weight"
        else:
            # For softmax: uniform baseline is 1/num_experts across all patches
            uniform_weight = 1.0 / num_experts_cfg
            norm_desc = "Softmax-Normalized (All Patches)"
            note = f"Uniform baseline: {uniform_weight:.4f}"

        # Compute per-expert ranges for title
        expert_ranges_str = " | ".join([f"E{i}:[{expert_vmin[i]:.3f}, {expert_vmax[i]:.3f}]"
                                        for i in experts_to_show])

        fig.suptitle(
            f"Epoch {epoch} | Block {block_idx} | MoE Routing ({norm_desc})\n"
            f"Per-Expert Ranges: {expert_ranges_str} | Mean: {global_mean:.4f}\n"
            f"{note} | Each expert uses independent color scaling",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        # Store figure in dict
        figures[block_idx] = fig

    return figures
