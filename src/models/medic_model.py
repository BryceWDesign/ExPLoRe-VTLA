import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from . import build_student, build_pixel_decoder


class MultiBlockWeightAggregator(nn.Module):
    """Aggregates MoE routing weights across multiple transformer blocks.

    When multiple MoE blocks are selected for a given loss, this module
    combines their routing weights into a single set of weights.

    Strategies:
        - "mean": Simple average across blocks (no learnable params)
        - "learned_scalar": One learnable weight per block, softmax-normalized
        - "linear_projection": Projects [num_blocks * num_slots] -> [num_slots] + softmax
    """
    def __init__(self, num_moe_blocks: int, num_slots: int, strategy: str = "mean"):
        super().__init__()
        self.strategy = strategy
        self.num_moe_blocks = num_moe_blocks
        self.num_slots = num_slots

        if strategy == "learned_scalar":
            # Initialized to zeros = equal weights under softmax
            self.block_logits = nn.Parameter(torch.zeros(num_moe_blocks))
        elif strategy == "linear_projection":
            self.projection = nn.Linear(num_moe_blocks * num_slots, num_slots)
        elif strategy != "mean":
            raise ValueError(
                f"Unknown aggregation strategy '{strategy}'. "
                f"Must be one of: 'mean', 'learned_scalar', 'linear_projection'"
            )

    def forward(self, weights_list: List) -> Tuple:
        """
        Aggregate routing weights from multiple MoE blocks.

        Args:
            weights_list: List of weight dicts or tensors, one per selected block.
                Each element is either:
                - A dict like {'combine': [B,N,E], 'dispatch': [B,N,E]}
                - A tensor [B,N,E]

        Returns:
            aggregated: Same format as input elements (dict or tensor)
            block_contributions: [num_blocks] softmax'd weights for logging (or None for mean/projection)
        """
        if isinstance(weights_list[0], dict):
            # Dict format: aggregate each weight type independently
            result = {}
            contributions = None
            for key in weights_list[0].keys():
                tensors = [w[key] for w in weights_list]
                result[key], contributions = self._aggregate(tensors)
            return result, contributions
        else:
            # Raw tensor format
            return self._aggregate(weights_list)

    def _aggregate(self, tensor_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Aggregate a list of tensors using the configured strategy."""
        stacked = torch.stack(tensor_list, dim=0)  # [K, B, N, E]

        if self.strategy == "mean":
            return stacked.mean(dim=0), None

        elif self.strategy == "learned_scalar":
            w = F.softmax(self.block_logits, dim=0)  # [K]
            aggregated = torch.einsum('k,kbne->bne', w, stacked)
            return aggregated, w.detach()

        elif self.strategy == "linear_projection":
            K, B, N, E = stacked.shape
            # Reshape: [K, B, N, E] -> [B, N, K*E]
            flat = stacked.permute(1, 2, 0, 3).reshape(B, N, K * E)
            raw = self.projection(flat)  # [B, N, E]
            # Apply softmax to maintain routing weight semantics (sums to 1 over experts)
            aggregated = F.softmax(raw, dim=-1)
            return aggregated, None

        # Should not reach here due to __init__ validation
        raise RuntimeError(f"Unknown strategy: {self.strategy}")


class MEDiC_Model(nn.Module):
    """
    A unified model for MEDiC that encapsulates the student, distillation head,
    and pixel decoder to simplify DDP wrapping.

    Supports both block masking (default) and evolved masking strategies.
    """
    def __init__(self, cfg, mask_generator=None):
        super().__init__()
        self.cfg = cfg
        self.student = build_student(cfg)
        self.mask_generator = mask_generator

        # --- Multi-block MoE loss weight aggregation ---
        # Parse per-loss block selection and create aggregators if needed.
        # Defaults to "last" for both losses = identical to previous behavior.
        self.head_block_indices: Optional[List[int]] = None
        self.pixel_block_indices: Optional[List[int]] = None
        self.head_weight_aggregator: Optional[MultiBlockWeightAggregator] = None
        self.pixel_weight_aggregator: Optional[MultiBlockWeightAggregator] = None

        losses_cfg_init = cfg.get("losses", {})
        if self.student.use_soft_moe and self.student.moe_blocks:
            available_moe_blocks = self.student.moe_blocks
            student_cfg = cfg.get("model", {}).get("student", {})
            num_experts = student_cfg.get("moe_num_experts", 1)
            slots_per_expert = student_cfg.get("moe_slots_per_expert", 1)
            num_slots = num_experts * slots_per_expert

            aggregation_strategy = losses_cfg_init.get("moe_loss_block_aggregation", "mean")

            # Parse head loss block config
            head_block_value = losses_cfg_init.get("moe_head_loss_block", "last")
            self.head_block_indices = self._parse_block_config(
                head_block_value, available_moe_blocks
            )
            if len(self.head_block_indices) > 1:
                self.head_weight_aggregator = MultiBlockWeightAggregator(
                    num_moe_blocks=len(self.head_block_indices),
                    num_slots=num_slots,
                    strategy=aggregation_strategy,
                )

            # Parse pixel loss block config (only for dense mode)
            if self.student.use_mask_tokens:
                pixel_block_value = losses_cfg_init.get("moe_pixel_loss_block", "last")
                self.pixel_block_indices = self._parse_block_config(
                    pixel_block_value, available_moe_blocks
                )
                if len(self.pixel_block_indices) > 1:
                    self.pixel_weight_aggregator = MultiBlockWeightAggregator(
                        num_moe_blocks=len(self.pixel_block_indices),
                        num_slots=num_slots,
                        strategy=aggregation_strategy,
                    )

        # Track if CLIP teacher is required for evolved masking
        self.requires_clip_teacher = False

        # Validate evolved masking configuration early
        if mask_generator is not None and hasattr(mask_generator, 'attention_source'):
            if mask_generator.attention_source == 'clip_teacher':
                self.requires_clip_teacher = True
                # Ensure teacher will be provided when using CLIP attention
                # We can't check if teacher exists now (it's created later),
                # but we can check the config
                if not cfg.get("model", {}).get("teacher"):
                    raise ValueError(
                        "Teacher configuration required when using CLIP attention source. "
                        "Add 'teacher' section to model config."
                    )
                print(f"ℹ️ Evolved masking configured with CLIP teacher attention - "
                      f"teacher model MUST be passed to forward()")

        self.distill_head: Optional[nn.Module] = None
        # The distillation head is needed for both head loss and CLS loss.
        losses_cfg = cfg.get("losses", {})
        if losses_cfg.get("use_head_loss", True) or losses_cfg.get("use_cls_loss", False):
            in_embed_dim = cfg["model"]["student"]["embed_dim"]
            # Handle missing teacher config gracefully - default to student embed_dim
            teacher_cfg = cfg.get("model", {}).get("teacher", {})
            out_embed_dim = teacher_cfg.get("embed_dim", in_embed_dim)
            self.distill_head = nn.Linear(in_embed_dim, out_embed_dim)

        self.pixel_decoder: Optional[nn.Module] = None
        if losses_cfg.get("use_decoder_loss", False):
            self.pixel_decoder = build_pixel_decoder(cfg)

        # Initialize GradNorm buffers if GradNorm is enabled
        # This is cleaner than lazy initialization during forward pass
        if losses_cfg.get("loss_weighting_method") == "gradnorm":
            # Count the number of active losses
            num_losses = 0
            if losses_cfg.get("use_head_loss", True):
                num_losses += 1
            if losses_cfg.get("use_cls_loss", False):
                num_losses += 1
            if losses_cfg.get("use_decoder_loss", False):
                num_losses += 1

            # Only initialize if we have multiple losses (GradNorm is for balancing)
            if num_losses > 1:
                # Register GradNorm buffers (not parameters - updated manually)
                # Initialize without explicit device - PyTorch will handle device placement
                # when the model is moved to GPU with .to(device) or .cuda()
                self.register_buffer('gradnorm_weights', torch.ones(num_losses))
                self.register_buffer('gradnorm_weight_sum', torch.tensor(float(num_losses)))
                self.register_buffer('gradnorm_initial_losses', torch.zeros(num_losses))
                self.register_buffer('gradnorm_weights_grad', torch.zeros(num_losses))
                self.register_buffer('gradnorm_initted', torch.tensor(False))

                # Buffers for gradient accumulation support
                self.register_buffer('gradnorm_accumulated_losses', torch.zeros(num_losses))
                self.register_buffer('gradnorm_accumulation_started', torch.tensor(False))

    def forward(self, img, mask=None, mask_generator=None, epoch=None, model_ema=None, teacher=None,
                collect_moe_weights=False):
        """
        Forward pass supporting both block and evolved masking.

        Args:
            img: Input images [B, 3, H, W]
            mask: Pre-generated mask [B, N] (optional)
            mask_generator: Optional mask generator (legacy parameter)
            epoch: Current training epoch (for evolved masking)
            model_ema: EMA model for evolved masking
            teacher: Teacher model for CLIP attention (optional)
            collect_moe_weights: If True, collect MoE combine weights for visualization/logging (default: False)

        Returns:
            pred_tok: Token predictions [B, N+1, D] or None
            pred_pix: Pixel predictions [B, num_masked, patch_size²*3] or None
            mask_out: Applied mask [B, N]
            combine_weights: Last MoE block's weights dict or None (for regularization)
            ids_restore: Restore indices for sparse mode [B, N] or None
            combine_weights_dict: All MoE blocks' weights {block_idx: weights_dict} or None
            head_combine_weights: Per-loss selected/aggregated weights for head loss, or None
            pixel_combine_weights: Per-loss selected/aggregated weights for pixel loss, or None
        """
        # Early validation: Check if CLIP teacher is required but not provided
        if self.requires_clip_teacher and teacher is None:
            raise ValueError(
                "This model requires a CLIP teacher for evolved masking but none was provided. "
                "Pass the teacher model to forward(): model(img, mask, teacher=clip_teacher_model)"
            )

        # Generate mask using configured generator
        if mask is None and self.mask_generator is not None:
            # Check if this is evolved masking (needs patch embeddings)
            is_evolved_masking = (
                hasattr(self.mask_generator, '__class__') and
                self.mask_generator.__class__.__name__ == 'EvolvedMaskingGenerator'
            ) or hasattr(self.mask_generator, '_compute_evolution_alpha')

            if is_evolved_masking:
                # Check if we need CLIP teacher for attention
                if hasattr(self.mask_generator, 'attention_source') and self.mask_generator.attention_source == 'clip_teacher':
                    # CLIP attention: pass images and CLIP teacher
                    # Note: teacher must be provided when using CLIP attention
                    if teacher is None:
                        raise ValueError(
                            "CLIP teacher required but not provided to forward(). "
                            "When using evolved masking with attention_source='clip_teacher', "
                            "you must pass the teacher model to forward(). "
                            "Example: model(img, mask, teacher=clip_teacher_model)"
                        )
                    mask = self.mask_generator(
                        x=None,  # Not needed for CLIP
                        epoch=epoch,
                        model_ema=model_ema,
                        clip_teacher=teacher,
                        images=img
                    )
                else:
                    # Student attention: need patch embeddings
                    x = self.student.patch_embed(img)
                    # Generate semantic mask based on attention patterns
                    mask = self.mask_generator(x, epoch=epoch, model_ema=model_ema)

                # All masking generators now return torch.bool tensors
                # No conversion needed
            else:
                # Block masking: generate random block masks
                # Support both consistent and per-sample shuffling strategies

                # Check if per-sample shuffling is enabled
                per_sample_shuffling = getattr(self.mask_generator, 'per_sample_shuffling', False)

                if hasattr(self.mask_generator, 'shuffle_patches') and self.mask_generator.shuffle_patches:
                    if per_sample_shuffling:
                        # Per-sample shuffling: each sample gets unique shuffle indices
                        # This is more similar to MAE's approach and allows more diversity
                        masks = []
                        shuffle_indices = []
                        restore_indices = []

                        for _ in range(img.shape[0]):
                            mask_tensor = self.mask_generator()
                            masks.append(mask_tensor)
                            shuffle_indices.append(self.mask_generator.ids_shuffle.clone())
                            restore_indices.append(self.mask_generator.ids_restore.clone())

                        # Stack all shuffle/restore indices for per-sample processing
                        # Shape becomes [B, N] instead of [N]
                        self.mask_generator.ids_shuffle = torch.stack(shuffle_indices, dim=0)
                        self.mask_generator.ids_restore = torch.stack(restore_indices, dim=0)
                    else:
                        # Consistent shuffling: all samples use same shuffle indices
                        # This is more efficient (avoids for-loop in encoder)
                        # Generate ONE mask to get shuffle/restore indices
                        first_mask_array = self.mask_generator()
                        # Store the shuffle/restore indices from first call
                        stored_ids_shuffle = self.mask_generator.ids_shuffle
                        stored_ids_restore = self.mask_generator.ids_restore

                        # Generate masks for the rest of the batch
                        masks = [first_mask_array]
                        for _ in range(1, img.shape[0]):
                            mask_tensor = self.mask_generator()
                            masks.append(mask_tensor)

                        # Restore the first shuffle/restore indices to ensure consistency
                        self.mask_generator.ids_shuffle = stored_ids_shuffle
                        self.mask_generator.ids_restore = stored_ids_restore
                else:
                    # No shuffling - standard generation
                    masks = []
                    for _ in range(img.shape[0]):
                        mask_tensor = self.mask_generator()
                        masks.append(mask_tensor)

                mask = torch.stack(masks, dim=0).to(img.device)
        elif mask is None:
            raise ValueError("Either mask or mask_generator must be provided")

        # Check if we need combine weights for MoE loss weighting, regularization, or visualization
        losses_cfg = self.cfg.get("losses", {})
        need_combine_weights = (
            self.student.use_soft_moe and
            (collect_moe_weights or  # For visualization/logging
             losses_cfg.get("moe_weight_head_loss", False) or  # For loss weighting
             losses_cfg.get("moe_weight_pixel_loss", False) or  # For loss weighting
             losses_cfg.get("importance_loss_weight", 0) > 0 or  # For MoE regularization
             losses_cfg.get("dispatch_entropy_loss_weight", 0) > 0)  # For MoE regularization
        )

        # Forward pass through student (handles both BEiT and MAE modes)
        z, aux_output, combine_weights, combine_weights_dict = self.student(
            img, mask, mask_generator=mask_generator,
            return_combine_weights=need_combine_weights
        )

        # In sparse mode, aux_output is ids_restore, not mask
        # We need to return the actual mask, not ids_restore
        mask_out = mask

        # Distillation head forward
        pred_tok = None
        if self.distill_head is not None:
            pred_tok = self.distill_head(z)  # return includes cls token

        # Pixel decoder forward
        pred_pix = None
        ids_restore = None  # Initialize for return
        if self.pixel_decoder is not None:
            # Pass ids_restore if using sparse encoding
            # In sparse mode, aux_output contains ids_restore
            ids_restore = aux_output if not self.student.use_mask_tokens else None

            # Pass visible_positions_shuffled to decoder for correct unshuffling with block masking
            # This fixes the bug where block masking + shuffling causes patches to be at wrong positions
            visible_positions_shuffled = getattr(self.student, 'visible_positions_shuffled', None)

            pred_pix = self.pixel_decoder(z, mask_out, ids_restore=ids_restore,
                                         visible_positions_shuffled=visible_positions_shuffled)

        # --- Per-loss MoE weight extraction/aggregation ---
        # Extract and optionally aggregate routing weights for each loss independently.
        # combine_weights (last block) is kept for backward-compatible regularization.
        head_combine_weights = None
        pixel_combine_weights = None

        if combine_weights_dict and self.head_block_indices is not None:
            head_combine_weights, _ = self._get_weights_for_loss(
                combine_weights_dict, self.head_block_indices, self.head_weight_aggregator
            )

        if combine_weights_dict and self.pixel_block_indices is not None:
            pixel_combine_weights, _ = self._get_weights_for_loss(
                combine_weights_dict, self.pixel_block_indices, self.pixel_weight_aggregator
            )

        # Return ids_restore for MoE loss weighting alignment
        # Return combine_weights_dict (all MoE blocks) for per-block regularization
        # Return head/pixel_combine_weights for per-loss selected/aggregated MoE weights
        return (pred_tok, pred_pix, mask_out, combine_weights, ids_restore,
                combine_weights_dict, head_combine_weights, pixel_combine_weights)


    def get_masking_info(self, epoch=None):
        """
        Get information about the current masking strategy.

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary with masking information
        """
        info = {
            'mask_generator_type': type(self.mask_generator).__name__ if self.mask_generator else 'None',
            'has_mask_generator': self.mask_generator is not None,
        }

        if hasattr(self.mask_generator, 'get_evolution_info'):
            # Evolved masking info
            info.update(self.mask_generator.get_evolution_info(epoch))
            # Add attention source info if available
            if hasattr(self.mask_generator, 'attention_source'):
                info['attention_source'] = self.mask_generator.attention_source
        elif hasattr(self.mask_generator, 'get_shape'):
            # Block masking info
            info.update({
                'grid_shape': self.mask_generator.get_shape(),
                'mask_ratio': getattr(self.mask_generator, 'mask_ratio', 'unknown'),
                'num_patches': getattr(self.mask_generator, 'num_patches', 'unknown'),
            })

        return info

    @staticmethod
    def _parse_block_config(value, available_moe_blocks: List[int]) -> List[int]:
        """
        Parse a block selection config value into a sorted list of block indices.

        Args:
            value: Config value - one of:
                - "last": Use only the last (highest index) MoE block
                - "all": Use all MoE blocks
                - int: Use a single specific block index
                - list[int]: Use multiple specific block indices
            available_moe_blocks: List of valid MoE block indices (e.g. [1, 3, 5, 7, 9, 11])

        Returns:
            Sorted list of block indices.

        Raises:
            ValueError: If any specified block index is not a valid MoE block.
        """
        if value == "last":
            return [max(available_moe_blocks)]
        elif value == "all":
            return sorted(available_moe_blocks)
        elif isinstance(value, int):
            if value not in available_moe_blocks:
                raise ValueError(
                    f"Block {value} is not an MoE block. "
                    f"Available MoE blocks: {sorted(available_moe_blocks)}"
                )
            return [value]
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError(
                    "Empty block list provided. Must specify at least one MoE block index."
                )
            for idx in value:
                if idx not in available_moe_blocks:
                    raise ValueError(
                        f"Block {idx} is not an MoE block. "
                        f"Available MoE blocks: {sorted(available_moe_blocks)}"
                    )
            return sorted(value)
        else:
            raise ValueError(
                f"Invalid block config value: {value!r}. "
                f"Expected 'last', 'all', an integer, or a list of integers."
            )

    def _get_weights_for_loss(
        self,
        combine_weights_dict: Dict[int, Any],
        block_indices: List[int],
        aggregator: Optional[MultiBlockWeightAggregator] = None,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Extract and optionally aggregate MoE routing weights for a specific loss.

        Args:
            combine_weights_dict: Dict mapping block index -> weight dict/tensor
            block_indices: Which block(s) to use
            aggregator: MultiBlockWeightAggregator instance if multi-block, else None

        Returns:
            weights: Aggregated weight dict/tensor for the loss
            block_contributions: Softmax'd per-block weights for logging (or None)
        """
        weights_list = [combine_weights_dict[idx] for idx in block_indices]
        if len(weights_list) == 1:
            return weights_list[0], None  # Single block, no aggregation needed
        else:
            assert aggregator is not None, (
                f"MultiBlockWeightAggregator required for {len(weights_list)} blocks, but got None"
            )
            return aggregator(weights_list)  # Multi-block aggregation

    def no_weight_decay(self):
        """
        Specifies which parameters should not have weight decay.
        """
        student_no_decay = {'student.' + k for k in self.student.no_weight_decay()}

        # Handle optional distill_head
        distill_head_no_decay = set()
        if self.distill_head is not None:
            # Linear layers typically don't have special no_weight_decay patterns,
            # but we include this for completeness
            distill_head_no_decay = set()

        decoder_no_decay = set()
        if self.pixel_decoder is not None:
            decoder_no_decay = {'pixel_decoder.' + k
                                for k in self.pixel_decoder.no_weight_decay()}

        # Learned scalar aggregators: block_logits should not have weight decay
        # (bias-like parameters initialized to zero for equal weighting)
        aggregator_no_decay = set()
        if (self.head_weight_aggregator is not None
                and hasattr(self.head_weight_aggregator, 'block_logits')):
            aggregator_no_decay.add('head_weight_aggregator.block_logits')
        if (self.pixel_weight_aggregator is not None
                and hasattr(self.pixel_weight_aggregator, 'block_logits')):
            aggregator_no_decay.add('pixel_weight_aggregator.block_logits')

        return (student_no_decay
                .union(distill_head_no_decay)
                .union(decoder_no_decay)
                .union(aggregator_no_decay))

def build_medic_model(cfg, mask_generator=None):
    """
    Build MEDiC model with optional mask generator.

    Args:
        cfg: Configuration dictionary
        mask_generator: Optional masking generator (block or evolved)

    Returns:
        MEDiC_Model instance
    """
    return MEDiC_Model(cfg, mask_generator=mask_generator)
