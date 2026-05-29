# --------------------------------------------------------
# MEDiC: Masked Encoder-Decoder with soft Mixture-of-Experts
# Semantic Segmentation Backbone Adapter
# Based on BEiT and CAE implementations
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

try:
    from mmcv_custom import load_checkpoint
    from mmseg.utils import get_root_logger
    from mmseg.models.builder import BACKBONES
    MMSEG_AVAILABLE = True
except ImportError:
    print("Warning: mmsegmentation not installed. Install with: pip install mmsegmentation==0.11.0 mmcv-full==1.3.0")
    BACKBONES = None
    load_checkpoint = None
    get_root_logger = None
    MMSEG_AVAILABLE = False


class MEDiC(nn.Module):
    """MEDiC backbone adapter for semantic segmentation.

    This adapter wraps the pretrained MEDiC VisionTransformerMIM model
    and adds FPN layers to produce multi-scale features for dense prediction.

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0
        qkv_bias (bool): Enable bias for qkv. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        init_values (float): Layer scale init value. Default: 0.1
        use_abs_pos_emb (bool): Use absolute position embedding. Default: False
        use_sincos_pos_emb (bool): Use fixed sincos position embedding. Default: False
        use_rel_pos_bias (bool): Use relative position bias. Default: True
        use_shared_rel_pos_bias (bool): Share rel pos bias across layers. Default: True
        use_checkpoint (bool): Use gradient checkpointing. Default: False
        use_soft_moe (bool): Use Mixture-of-Experts. Default: False
        moe_num_experts (int): Number of experts. Default: 2
        moe_slots_per_expert (int): Slots per expert. Default: 1
        moe_mlp_ratio (float): MLP ratio for MoE blocks. Default: None (use mlp_ratio)
        moe_placement (str): MoE placement strategy. Default: "alternating"
        out_indices (list): Indices of blocks to output features. Default: [3, 5, 7, 11]
        out_with_norm (bool): Apply norm before output. Default: False
    """

    @staticmethod
    def _load_checkpoint_config(pretrained_path):
        """Load configuration from checkpoint folder BEFORE creating backbone.

        This is critical for MoE models: the architecture must be configured
        correctly BEFORE VisionTransformerMIM is instantiated, because the
        MoE blocks are created during __init__ and cannot be changed after.

        Args:
            pretrained_path (str): Path to pretrained checkpoint file

        Returns:
            dict: Student configuration from checkpoint, or empty dict if not found
        """
        import os
        import glob
        import yaml

        if pretrained_path is None:
            return {}

        checkpoint_folder = os.path.dirname(pretrained_path)
        config_files = glob.glob(os.path.join(checkpoint_folder, "config_*.yaml"))

        if not config_files:
            print(f"Warning: No config file found in {checkpoint_folder}")
            return {}

        try:
            with open(config_files[0], 'r') as f:
                cfg = yaml.safe_load(f)
            student_cfg = cfg.get("model", {}).get("student", {})
            print(f"Loaded checkpoint config from {config_files[0]}")
            return student_cfg
        except Exception as e:
            print(f"Warning: Failed to load config from {config_files[0]}: {e}")
            return {}

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 init_values=0.1,
                 use_abs_pos_emb=False,
                 use_sincos_pos_emb=False,
                 use_rel_pos_bias=True,
                 use_shared_rel_pos_bias=True,
                 use_checkpoint=False,
                 # MoE parameters
                 use_soft_moe=False,
                 moe_num_experts=2,
                 moe_slots_per_expert=1,
                 moe_mlp_ratio=None,
                 moe_placement="alternating",
                 # Output parameters
                 out_indices=[3, 5, 7, 11],
                 out_with_norm=False,
                 # Pretrained checkpoint path
                 pretrained=None):
        super().__init__()

        # Import VisionTransformerMIM from MEDiC codebase
        import sys
        import os
        medic_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if medic_root not in sys.path:
            sys.path.insert(0, medic_root)
        from src.models.vision_transformer_mim import VisionTransformerMIM

        # =====================================================================
        # CRITICAL FIX: Load checkpoint config BEFORE creating backbone
        # =====================================================================
        # For MoE models, the architecture (which blocks have SoftMoELayer vs MLP)
        # is determined at construction time. If we build with use_soft_moe=False
        # and then try to load MoE weights, all expert weights will be discarded!
        #
        # This fix loads the config from the checkpoint folder FIRST, then uses
        # those values to configure the backbone correctly.
        # =====================================================================
        if pretrained is not None:
            checkpoint_cfg = self._load_checkpoint_config(pretrained)
            if checkpoint_cfg:
                # Override MoE parameters with checkpoint config
                # This ensures architecture matches the pretrained weights
                cfg_use_soft_moe = checkpoint_cfg.get("use_soft_moe", use_soft_moe)
                cfg_moe_num_experts = checkpoint_cfg.get("moe_num_experts", moe_num_experts)
                cfg_moe_slots_per_expert = checkpoint_cfg.get("moe_slots_per_expert", moe_slots_per_expert)
                cfg_moe_mlp_ratio = checkpoint_cfg.get("moe_mlp_ratio", moe_mlp_ratio)
                cfg_moe_placement = checkpoint_cfg.get("moe_placement", moe_placement)

                # Also load ALL position embedding config from checkpoint
                # This is critical - we must match the checkpoint's position embedding type
                cfg_use_abs_pos_emb = checkpoint_cfg.get("use_abs_pos_emb", use_abs_pos_emb)
                cfg_use_sincos_pos_emb = checkpoint_cfg.get("use_sincos_pos_emb", use_sincos_pos_emb)
                cfg_use_rel_pos_bias = checkpoint_cfg.get("use_rel_pos_bias", use_rel_pos_bias)

                # Log if MoE settings are being applied
                if cfg_use_soft_moe:
                    print(f"[MEDiC Backbone] Configuring MoE architecture from checkpoint:")
                    print(f"  - use_soft_moe: {cfg_use_soft_moe}")
                    print(f"  - moe_num_experts: {cfg_moe_num_experts}")
                    print(f"  - moe_placement: {cfg_moe_placement}")

                # Log position embedding configuration
                print(f"[MEDiC Backbone] Position embedding config from checkpoint:")
                print(f"  - use_abs_pos_emb: {cfg_use_abs_pos_emb}")
                print(f"  - use_sincos_pos_emb: {cfg_use_sincos_pos_emb}")
                print(f"  - use_rel_pos_bias: {cfg_use_rel_pos_bias}")

                # Use checkpoint config values
                use_soft_moe = cfg_use_soft_moe
                moe_num_experts = cfg_moe_num_experts
                moe_slots_per_expert = cfg_moe_slots_per_expert
                moe_mlp_ratio = cfg_moe_mlp_ratio
                moe_placement = cfg_moe_placement
                # NOTE: Do NOT override position embedding settings from pretrain config.
                # The mmseg config specifies the correct position embedding type for
                # downstream (use_rel_pos_bias=True → converted to sincos for segmentation).
                # Overriding from pretrain config breaks eval of trained checkpoints.

        # =====================================================================
        # POSITION EMBEDDING HANDLING FOR SEGMENTATION
        # =====================================================================
        # For segmentation (variable image sizes, no masking), we cannot use
        # relative position bias because it requires fixed sequence length.
        #
        # The checkpoint position embedding type determines what we use:
        # 1. If checkpoint has use_abs_pos_emb=True: Use absolute (interpolatable)
        # 2. If checkpoint has use_rel_pos_bias=True: Convert to sincos (interpolatable)
        # 3. If checkpoint has use_sincos_pos_emb=True: Use sincos directly
        #
        # Key insight: Absolute position embeddings can be interpolated for different
        # image sizes, which is exactly what we need for segmentation at 512x512.
        # =====================================================================
        if use_rel_pos_bias:
            # For segmentation, use per-layer relative position bias (not shared).
            # During training: init_weights() converts shared→per-layer via interpolation.
            # During eval: the semseg checkpoint already has per-layer tables.
            print(f"[MEDiC Backbone] Using per-layer relative position bias for segmentation")
            actual_use_sincos = False
            actual_use_abs = False
            actual_use_rel_pos = True
        elif use_abs_pos_emb:
            print(f"[MEDiC Backbone] Using absolute position embeddings (interpolated 14x14 → 32x32)")
            actual_use_sincos = False
            actual_use_abs = True
            actual_use_rel_pos = False
        else:
            actual_use_sincos = use_sincos_pos_emb
            actual_use_abs = False
            actual_use_rel_pos = False

        # Create the backbone model with CORRECT architecture from checkpoint
        self.backbone = VisionTransformerMIM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            use_abs_pos_emb=actual_use_abs,
            use_sincos_pos_emb=actual_use_sincos,
            use_rel_pos_bias=actual_use_rel_pos,  # Per-layer bias for semseg, not shared
            use_shared_rel_pos_bias=False,
            # use_mask_tokens=True when using rel_pos_bias (validation requires it).
            # Semseg doesn't use masking, but mask_tokens=True avoids the validation error.
            # The mask token parameter will just be unused.
            use_mask_tokens=actual_use_rel_pos,
            use_soft_moe=use_soft_moe,
            moe_num_experts=moe_num_experts,
            moe_slots_per_expert=moe_slots_per_expert,
            moe_mlp_ratio=moe_mlp_ratio,
            moe_placement=moe_placement,
            num_classes=0,  # No classification head
        )

        self.num_features = self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size

        # Output normalization
        if not out_with_norm:
            self.norm = nn.Identity()
        else:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)

        # Feature Pyramid Network (FPN) layers
        # Creates multi-scale features from single-scale ViT output
        # For patch_size=16: strides [4, 8, 16, 32]
        if patch_size == 16:
            # FPN level 1: stride 4 (upsample 4x from stride 16)
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            # FPN level 2: stride 8 (upsample 2x from stride 16)
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            # FPN level 3: stride 16 (no change)
            self.fpn3 = nn.Identity()
            # FPN level 4: stride 32 (downsample 2x from stride 16)
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            # For patch_size=8: strides [4, 8, 16, 32]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        else:
            raise ValueError(f"Unsupported patch_size={patch_size}. Use 8 or 16.")

        # Initialize weights if pretrained path is provided
        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained MEDiC checkpoint.

        Note:
            MoE configuration is now loaded in __init__() BEFORE the backbone
            is created. This method only handles weight loading and validation.
        """
        if isinstance(pretrained, str):
            if load_checkpoint is not None:
                logger = get_root_logger() if get_root_logger is not None else None

                def log_info(msg):
                    if logger is not None:
                        logger.info(msg)
                    else:
                        print(msg)

                def log_warning(msg):
                    if logger is not None:
                        logger.warning(msg)
                    else:
                        print(f"WARNING: {msg}")

                def log_error(msg):
                    if logger is not None:
                        logger.error(msg)
                    else:
                        print(f"ERROR: {msg}")

                # Load checkpoint with strict=False to allow missing FPN weights
                # PyTorch 2.6 fix: set weights_only=False for numpy compatibility
                checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)

                # Extract student weights from MEDiC checkpoint
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Filter to student weights only (remove teacher, decoder, etc.)
                student_state_dict = {}
                for k, v in state_dict.items():
                    # Handle different checkpoint formats
                    if k.startswith('module.student.'):
                        new_key = k.replace('module.student.', 'backbone.')
                    elif k.startswith('student.'):
                        new_key = k.replace('student.', 'backbone.')
                    elif k.startswith('module.'):
                        new_key = k.replace('module.', 'backbone.')
                    else:
                        new_key = 'backbone.' + k

                    # Skip decoder, teacher, and head weights
                    if any(skip in new_key for skip in ['decoder', 'teacher', 'head', 'fc_norm', 'distill']):
                        continue

                    # Handle position embedding interpolation for different image sizes
                    if 'pos_embed' in new_key and v is not None:
                        # Get current and checkpoint position embedding shapes
                        current_pos_embed = self.backbone.pos_embed
                        if current_pos_embed is not None:
                            checkpoint_pos_embed = v
                            embedding_size = checkpoint_pos_embed.shape[2]  # Should be 768

                            # Calculate grid sizes
                            num_extra_tokens = 1  # CLS token
                            orig_size = int((checkpoint_pos_embed.shape[1] - num_extra_tokens) ** 0.5)
                            new_size = int((current_pos_embed.shape[1] - num_extra_tokens) ** 0.5)

                            if orig_size != new_size:
                                log_info(f"Position embedding interpolation from {orig_size}x{orig_size} to {new_size}x{new_size}")

                                # Separate CLS token and position embeddings
                                extra_tokens = checkpoint_pos_embed[:, :num_extra_tokens]
                                pos_tokens = checkpoint_pos_embed[:, num_extra_tokens:]

                                # Reshape to 2D grid
                                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                                # Interpolate position embeddings
                                pos_tokens = torch.nn.functional.interpolate(
                                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                                # Reshape back
                                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                                # Concatenate CLS token and interpolated position embeddings
                                v = torch.cat((extra_tokens, pos_tokens), dim=1)

                    student_state_dict[new_key] = v

                # =====================================================================
                # ARCHITECTURE MISMATCH VALIDATION
                # =====================================================================
                # Check if checkpoint contains MoE weights but model doesn't have MoE
                # This catches the critical bug where MoE weights would be silently discarded
                checkpoint_has_moe = any(
                    'experts' in k or 'mlp.phi' in k or 'mlp.scale' in k
                    for k in student_state_dict.keys()
                )
                model_has_moe = getattr(self.backbone, 'use_soft_moe', False)

                if checkpoint_has_moe and not model_has_moe:
                    moe_keys = [k for k in student_state_dict.keys()
                               if 'experts' in k or 'mlp.phi' in k or 'mlp.scale' in k]
                    log_error("=" * 70)
                    log_error("CRITICAL ARCHITECTURE MISMATCH DETECTED!")
                    log_error("=" * 70)
                    log_error(f"Checkpoint contains {len(moe_keys)} MoE weight keys:")
                    for key in moe_keys[:10]:  # Show first 10
                        log_error(f"  - {key}")
                    if len(moe_keys) > 10:
                        log_error(f"  ... and {len(moe_keys) - 10} more MoE keys")
                    log_error("")
                    log_error("BUT the model was built with use_soft_moe=False!")
                    log_error("All MoE weights will be DISCARDED, causing:")
                    log_error("  - 50% of MLP capacity using random weights")
                    log_error("  - Significant performance degradation")
                    log_error("")
                    log_error("This should not happen with the fixed code.")
                    log_error("Please check that _load_checkpoint_config() found the config file.")
                    log_error("=" * 70)
                    raise ValueError(
                        f"Architecture mismatch: checkpoint has MoE ({len(moe_keys)} keys) "
                        f"but model built without MoE. This would discard all expert weights!"
                    )

                # Load with strict=False (FPN weights will be randomly initialized)
                msg = self.load_state_dict(student_state_dict, strict=False)

                # =====================================================================
                # DETAILED UNEXPECTED KEY LOGGING
                # =====================================================================
                log_info(f"Loaded pretrained MEDiC from: {pretrained}")
                log_info(f"Missing keys: {len(msg.missing_keys)}")
                log_info(f"Unexpected keys: {len(msg.unexpected_keys)}")

                # Check for MoE keys in unexpected (safety check)
                if msg.unexpected_keys:
                    moe_unexpected = [k for k in msg.unexpected_keys
                                     if 'expert' in k or 'phi' in k or 'mlp.scale' in k]
                    if moe_unexpected:
                        log_error("=" * 70)
                        log_error(f"WARNING: {len(moe_unexpected)} MoE keys were NOT loaded!")
                        log_error("=" * 70)
                        for key in moe_unexpected[:10]:
                            log_error(f"  - {key}")
                        if len(moe_unexpected) > 10:
                            log_error(f"  ... and {len(moe_unexpected) - 10} more")
                        log_error("This indicates an architecture mismatch that was not caught earlier.")
                        log_error("=" * 70)

                    # Log non-MoE unexpected keys (usually position bias related)
                    other_unexpected = [k for k in msg.unexpected_keys if k not in moe_unexpected]
                    if other_unexpected:
                        log_info(f"Other unexpected keys (usually rel_pos_bias, expected): {len(other_unexpected)}")
                        for key in other_unexpected[:5]:
                            log_info(f"  - {key}")

                # Log expected missing keys (FPN layers)
                if msg.missing_keys:
                    fpn_missing = [k for k in msg.missing_keys if 'fpn' in k]
                    other_missing = [k for k in msg.missing_keys if 'fpn' not in k]
                    if fpn_missing:
                        log_info(f"FPN layers will be randomly initialized: {len(fpn_missing)} keys")
                    if other_missing:
                        log_warning(f"Non-FPN missing keys (may need attention): {len(other_missing)}")
                        for key in other_missing[:5]:
                            log_warning(f"  - {key}")

                # Final confirmation of MoE status
                if model_has_moe:
                    log_info(f"✓ Model built with MoE (use_soft_moe=True)")
                    log_info(f"  - Experts: {getattr(self.backbone, 'moe_num_experts', 'N/A')}")
                    log_info(f"  - Placement: {getattr(self.backbone, 'moe_placement', 'N/A')}")

            else:
                print(f"Warning: mmcv not available. Cannot load checkpoint: {pretrained}")
        elif pretrained is None:
            # Random initialization (default)
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        """Get number of transformer blocks."""
        return len(self.backbone.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Specify parameters that should not have weight decay."""
        return {'backbone.pos_embed', 'backbone.cls_token', 'backbone.mask_token'}

    def forward_features(self, x):
        """Extract multi-scale features from input images.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            tuple: Multi-scale features at different strides
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.backbone.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # Calculate spatial dimensions after patching
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # Add CLS token
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # Get relative position bias if using it
        rel_pos_bias = None
        if self.backbone.rel_pos_bias is not None:
            rel_pos_bias = self.backbone.rel_pos_bias()

        # Forward through transformer blocks and collect features
        features = []
        for i, blk in enumerate(self.backbone.blocks):
            if self.use_checkpoint:
                # Gradient checkpointing to save memory
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)

            # Extract features at specified indices
            if i in self.out_indices:
                # Remove CLS token and reshape to spatial format
                # x[:, 0] is CLS token, x[:, 1:] are patch tokens
                xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        # Apply FPN to create multi-scale pyramid
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            tuple: Multi-scale features for segmentation decoder
        """
        return self.forward_features(x)


# Register with mmseg if available
if MMSEG_AVAILABLE:
    MEDiC = BACKBONES.register_module()(MEDiC)
