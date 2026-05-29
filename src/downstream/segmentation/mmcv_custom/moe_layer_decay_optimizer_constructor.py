"""
MoE-aware Layer Decay Optimizer Constructor for Semantic Segmentation.

This extends the standard layer decay optimizer to support different learning rates
for MoE components (router, experts) vs standard components.

Key MoE-specific options:
- moe_router_lr_scale: Scale factor for router (phi, scale) parameters
- moe_expert_lr_scale: Scale factor for expert MLP parameters

Typical settings:
- Router: 0.1x base LR (sensitive to perturbation)
- Experts: 1.0x base LR (same as other MLPs)
"""

import json
from mmcv.runner import DefaultOptimizerConstructor
from mmcv.runner import get_dist_info

# Import BOTH registries to register with both
from mmcv.runner import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmseg.core.builder import OPTIMIZER_BUILDERS as MMSEG_OPTIMIZER_BUILDERS


def get_num_layer_for_vit(var_name, num_max_layer):
    """Get layer index for layer decay."""
    # Handle both backbone.X and backbone.backbone.X (MEDiC wrapper adds extra prefix)
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed",
                    "backbone.backbone.cls_token", "backbone.backbone.mask_token", "backbone.backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed") or var_name.startswith("backbone.backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks") or var_name.startswith("backbone.backbone.blocks"):
        parts = var_name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks':
                layer_id = int(parts[i + 1])
                return layer_id + 1
    return num_max_layer - 1


def is_moe_router_param(var_name):
    """Check if parameter is an MoE router parameter (phi, scale)."""
    # Router parameters: phi (slot projections) and scale (sharpness)
    return any(x in var_name for x in ['mlp.phi', 'mlp.scale'])


def is_moe_expert_param(var_name):
    """Check if parameter is an MoE expert parameter."""
    # Expert parameters: inside mlp.experts ModuleList
    return 'mlp.experts' in var_name


@MMSEG_OPTIMIZER_BUILDERS.register_module(force=True)
class MoELayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Layer Decay Optimizer with MoE-specific learning rate scaling.

    Args in paramwise_cfg:
        num_layers (int): Number of transformer layers
        layer_decay_rate (float): Layer decay multiplier (e.g., 0.85)
        moe_router_lr_scale (float): LR scale for router params (default: 0.1)
        moe_expert_lr_scale (float): LR scale for expert params (default: 1.0)
    """

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list with MoE-aware LR scaling."""
        parameter_groups = {}

        # Get config
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.85)
        moe_router_lr_scale = self.paramwise_cfg.get('moe_router_lr_scale', 0.1)
        moe_expert_lr_scale = self.paramwise_cfg.get('moe_expert_lr_scale', 1.0)

        print(f"Build MoELayerDecayOptimizerConstructor:")
        print(f"  - layer_decay_rate: {layer_decay_rate}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - moe_router_lr_scale: {moe_router_lr_scale}")
        print(f"  - moe_expert_lr_scale: {moe_expert_lr_scale}")

        weight_decay = self.base_wd
        moe_router_count = 0
        moe_expert_count = 0

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            # Weight decay grouping
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
                decay_group = "no_decay"
                this_weight_decay = 0.
            else:
                decay_group = "decay"
                this_weight_decay = weight_decay

            # Layer index for layer decay
            layer_id = get_num_layer_for_vit(name, num_layers)
            layer_scale = layer_decay_rate ** (num_layers - layer_id - 1)

            # MoE-specific scaling
            if is_moe_router_param(name):
                moe_scale = moe_router_lr_scale
                moe_group = "router"
                moe_router_count += 1
            elif is_moe_expert_param(name):
                moe_scale = moe_expert_lr_scale
                moe_group = "expert"
                moe_expert_count += 1
            else:
                moe_scale = 1.0
                moe_group = "normal"

            # Combined scale
            total_scale = layer_scale * moe_scale

            # Group name includes layer, decay status, and MoE status
            group_name = f"layer_{layer_id}_{decay_group}_{moe_group}"

            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": total_scale,
                    "group_name": group_name,
                    "lr": total_scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        # Log summary
        rank, _ = get_dist_info()
        if rank == 0:
            print(f"MoE parameter counts:")
            print(f"  - Router params (phi, scale): {moe_router_count}")
            print(f"  - Expert params: {moe_expert_count}")

            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"][:3],  # Show first 3
                    "num_params": len(parameter_groups[key]["param_names"]),
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())
