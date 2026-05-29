import argparse
import datetime
import os
import time
import random
import yaml
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, cast
import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DistributedSampler, DataLoader
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

# Local imports
from .utils.utils import (
    NativeScalerWithGradNormCount,
    is_main_process,
    cosine_scheduler,
    init_distributed_mode,
    get_rank,
    get_world_size,
    get_grad_norm_,
)
from .utils.optim_factory import get_parameter_groups, get_num_layer_for_vit, LayerDecayValueAssigner
from .utils.losses import compute_three_losses
from .utils.viz import get_reconstruction_fig, plot_scheduler, log_attention_plots_to_wandb, fig_to_pil, get_moe_weight_fig
from .data.loader import build_loader, MEDiCDataset
from .data.transforms import build_transform
from .models.medic_model import build_medic_model
from .models import build_teacher, build_student
from .utils.masking_generator import BlockMaskingGenerator, create_masking_generator, get_masking_generator_info

import numpy as np
from src.utils.config_tracker import ConfigUsageTracker

def get_args() -> Dict[str, Any]:
    """Parses command-line arguments and loads the configuration file."""
    parser = argparse.ArgumentParser("MEDiC pre-train")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--config-tracker", action="store_true",
                       help="Enable config usage tracking and validation")
    parser.add_argument("--strict-config", action="store_true",
                       help="Enable strict config validation (raise errors for unused/invalid keys)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--test-mode", action="store_true",
                       help="Enable test mode for quick verification (5 steps only)")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum number of training steps (for testing)")

    # WandB sweep hyperparameter overrides
    parser.add_argument("--dispatch_entropy_loss_weight", type=str, default=None,
                       help="Per-expert entropy loss weights (format: '0.3,1.0' for 2 experts)")
    parser.add_argument("--dispatch_entropy_weight_expert_0", type=float, default=None,
                       help="Entropy loss weight for expert 0 (continuous sweep parameter)")
    parser.add_argument("--dispatch_entropy_weight_expert_1", type=float, default=None,
                       help="Entropy loss weight for expert 1 (continuous sweep parameter)")
    parser.add_argument("--moe_entropy_aggregation", type=str, default=None,
                       help="Entropy aggregation method: mean, min, mean+var, separate")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of training epochs (useful for sweeps)")

    args = parser.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate configuration early to catch errors
    from src.utils.config_validation import validate_config, print_config_warnings
    validate_config(cfg)
    print_config_warnings(cfg)

    # Store the original config path and filename for later copying
    cfg["_config_path"] = args.cfg
    cfg["_config_filename"] = Path(args.cfg).name  # e.g., "config_v4.yaml"

    # Add resume path to config
    if args.resume:
        cfg["resume_path"] = args.resume

    # Add test mode settings
    if args.test_mode:
        cfg["test_mode"] = True
        cfg["max_steps"] = 5  # Default for test mode
        cfg["epochs"] = 1     # Single epoch for testing
        if is_main_process():
            print("Test mode enabled: 5 steps maximum")
    elif args.max_steps:
        cfg["max_steps"] = args.max_steps
        if is_main_process():
            print(f"Maximum steps set to: {args.max_steps}")

    # WandB sweep hyperparameter overrides
    # Priority: Per-expert continuous parameters > comma-separated string
    if args.dispatch_entropy_weight_expert_0 is not None and args.dispatch_entropy_weight_expert_1 is not None:
        # Continuous sweep format: separate float parameters for each expert
        weights = [args.dispatch_entropy_weight_expert_0, args.dispatch_entropy_weight_expert_1]
        cfg["losses"]["dispatch_entropy_loss_weight"] = weights
        if is_main_process():
            print(f"Sweep override (continuous): dispatch_entropy_loss_weight = {weights}")
    elif args.dispatch_entropy_loss_weight:
        # Backward compatibility: comma-separated string format (e.g., "0.3,1.0" → [0.3, 1.0])
        weights = [float(x.strip()) for x in args.dispatch_entropy_loss_weight.split(",")]
        cfg["losses"]["dispatch_entropy_loss_weight"] = weights
        if is_main_process():
            print(f"Sweep override (string): dispatch_entropy_loss_weight = {weights}")

    if args.moe_entropy_aggregation:
        cfg["losses"]["moe_entropy_aggregation"] = args.moe_entropy_aggregation
        if is_main_process():
            print(f"Sweep override: moe_entropy_aggregation = {args.moe_entropy_aggregation}")

    if args.epochs:
        cfg["epochs"] = args.epochs
        if is_main_process():
            print(f"Sweep override: epochs = {args.epochs}")

    # Initialize config tracker if requested
    if args.config_tracker:
        tracker = ConfigUsageTracker(cfg, strict_mode=args.strict_config)
        cfg["_config_tracker"] = tracker
        if is_main_process():
            print("Config usage tracking enabled")

    return cfg


def get_config_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get config value with optional tracking."""
    if cfg is None:
        return default
    if "_config_tracker" in cfg:
        return cfg["_config_tracker"].get(key, default)
    else:
        # Fallback to direct access
        keys = key.split('.')
        current = cfg
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

def setup_environment() -> Dict[str, Any]:
    """Sets up DDP, directories, seeds, and broadcasts the config."""
    cfg = get_args()
    init_distributed_mode(cfg)

    local_gpu = get_config_value(cfg, "gpu")
    local_rank = get_config_value(cfg, "rank")

    cfg_list_to_broadcast: List[Optional[Dict[str, Any]]] = [None]
    if is_main_process():
        base_output_dir = Path(get_config_value(cfg, "output_dir", "output"))
        print(f"Number of GPUs: {get_world_size()}")
        accum_iter = get_config_value(cfg, "optim.accum_iter", 1)
        batch_per_gpu = get_config_value(cfg, "data.batch_per_gpu")
        effective_batch_size = batch_per_gpu * accum_iter * get_world_size()
        print(f"Batch size per GPU: {batch_per_gpu}")
        print(f"Gradient accumulation steps: {accum_iter}")
        print(f"Effective batch size: {effective_batch_size} (= {batch_per_gpu} * {accum_iter} * {get_world_size()})")

        data_root_suffix = Path(get_config_value(cfg, "data.data_path")).name

        # Handle teacher name extraction for run naming with fallback support
        try:
            teacher_name = get_config_value(cfg, "model.teacher.name").replace('/', '-')
        except (KeyError, AttributeError):
            # Fallback to old config structure or default
            teacher_name = get_config_value(cfg, "teacher_model", "no-teacher").replace('/', '-')

        head_name = "h-dual" if get_config_value(cfg, "losses.use_decoder_loss", False) else "h-fc"
        # Create the full component string for documentation purposes
        components = [
            f"e-{get_config_value(cfg, 'epochs')}",
            f"t-{teacher_name}",
            f"s-{get_config_value(cfg, 'model.student.name')}",
            head_name,
            f"d-{data_root_suffix}",
        ]
        full_components = "_".join(components)

        # Create base name (without the long suffix) for wandb and folder
        base_name = get_config_value(cfg, 'wandb_meta.name')

        # Add timestamp to ensure uniqueness and traceability
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H")

        # Checkpoint folder uses base name + timestamp only (shorter)
        checkpoint_folder_name = f"{base_name}_{timestamp}"

        # Store the full name for reference and documentation
        full_reference_name = f"{base_name}_{full_components}_{timestamp}"

        output_dir = base_output_dir / checkpoint_folder_name
        # No need for suffix logic anymore since timestamp ensures uniqueness

        output_dir.mkdir(parents=True, exist_ok=True)
        cfg["output_dir"] = str(output_dir)

        # Store both names in config for later use
        cfg["_checkpoint_folder_name"] = checkpoint_folder_name
        cfg["_full_reference_name"] = full_reference_name
        cfg["_base_name"] = base_name

        # Copy config file to checkpoint directory
        config_filename = cfg.get("_config_filename", "config.yaml")
        config_src = Path(cfg["_config_path"])  # This was stored in get_args()
        config_dst = output_dir / config_filename
        shutil.copy2(config_src, config_dst)
        print(f"Config copied to checkpoint: {config_dst}")

        # Update config with scaled learning rates
        original_lr = get_config_value(cfg, "optim.lr")
        scaled_lr = original_lr * effective_batch_size / 2048.0
        cfg["optim"]["lr"] = scaled_lr
        print(f"Scaled learning rate to: {scaled_lr:.6f}")

        original_min_lr = float(get_config_value(cfg, "schedule.min_lr"))
        scaled_min_lr = original_min_lr * effective_batch_size / 2048.0
        # Set the scaled min_lr in the schedule section where it belongs
        if "schedule" not in cfg:
            cfg["schedule"] = {}
        cfg["schedule"]["min_lr"] = scaled_min_lr
        print(f"Scaled min LR to: {scaled_min_lr:.6f}")

        cfg_list_to_broadcast = [cfg]

    if cfg.get("distributed"):
        dist.broadcast_object_list(cfg_list_to_broadcast, src=0)

    cfg = cfg_list_to_broadcast[0]
    assert cfg is not None

    cfg["gpu"] = local_gpu
    cfg["rank"] = local_rank
    if not cfg.get("cpu", False) and torch.cuda.is_available():
        torch.cuda.set_device(cfg["gpu"])

    seed = get_config_value(cfg, "seed", 42) + get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    return cfg


def init_wandb(cfg: Dict[str, Any], output_dir: Path,
               lr_schedule: np.ndarray, niter_per_ep: int):
    """Initializes wandb and logs the learning rate schedule."""
    if not is_main_process():
        return

    print(f"Using output directory: {output_dir}")
    wandb_cfg = get_config_value(cfg, "wandb_meta")

    # Use the shorter base name for wandb run name
    wandb_run_name = cfg.get("_base_name", output_dir.name)

    data_name = Path(get_config_value(cfg, "data.data_path")).name
    tags = [
        f"E:{get_config_value(cfg, 'epochs')}",
        f"B:{get_config_value(cfg, 'data.batch_per_gpu') * get_world_size()}",
        f"LR:{get_config_value(cfg, 'optim.lr'):.1e}",
        f"TEACH:{get_config_value(cfg, 'model.teacher.name', get_config_value(cfg, 'teacher_model', 'no-teacher'))}",
        f"STDNT:{get_config_value(cfg, 'model.student.name')}",
        f"MASK:{get_config_value(cfg, 'mask.mask_ratio')}",
        f"DATA:{data_name}",
    ]
    # Use /tmp for wandb local storage to avoid filling home directory quota
    # Data is synced to wandb.ai in real-time, so local files are just cache
    import tempfile
    wandb_dir = tempfile.gettempdir()  # /tmp on compute nodes

    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        config=cfg,
        name=wandb_run_name,  # Use shorter name for cleaner wandb UI
        dir=wandb_dir,  # Use /tmp instead of output_dir to save quota
        tags=tags,
        job_type=cfg.get('mask.mask_type', 'block'),
        group=data_name,
    )

    # Update run_mappings.json with this new run
    if wandb.run is not None:
        mapping_file = Path('run_mappings.json')
        mappings = {}
        if mapping_file.exists():
            # Retry read operation to handle race conditions
            max_read_retries = 3
            for read_attempt in range(max_read_retries):
                try:
                    with open(mapping_file, 'r') as f:
                        mappings = json.load(f)
                    break  # Success
                except (json.JSONDecodeError, IOError) as e:
                    if read_attempt < max_read_retries - 1:
                        # Retry after brief delay
                        time.sleep(0.3)
                    else:
                        # Final attempt failed - use empty mappings
                        if is_main_process():
                            print(f"Warning: Could not read run_mappings.json after {max_read_retries} attempts: {e}")
                            print("Continuing with empty mappings - file may be corrupted or locked.")
                        mappings = {}

        # Add mapping for this run's output directory
        folder_name = output_dir.name  # This is the checkpoint folder name (short)
        base_name = cfg.get("_base_name", folder_name)
        full_reference = cfg.get("_full_reference_name", folder_name)

        mappings[folder_name] = {
            'folder_name': folder_name,  # Checkpoint folder name for KNN to find
            'full_reference_name': full_reference,  # Full name with all components for reference
            'parent_pretrain': {
                'run_id': wandb.run.id,
                'run_name': wandb.run.name,  # This should be the short base_name
                'run_url': wandb.run.url,
                'group': base_name,  # Use base name as group for cleaner wandb UI
                'project': wandb.run.project,
                'entity': wandb.run.entity
            },
            'key_configs': {
                'abs_pos_embed': cfg.get('model', {}).get('student', {}).get('abs_pos_embed', False),
                'encoder_mode': cfg.get('model', {}).get('student', {}).get('encoder_mode', 'dense'),
                'shared_rel_pos_bias': cfg.get('model', {}).get('student', {}).get('shared_rel_pos_bias', False),
                'mask_ratio': cfg.get('mask', {}).get('mask_ratio', 0.4),
                'mask_type': cfg.get('mask', {}).get('mask_type', 'block'),
                'head_loss_weight': cfg.get('losses', {}).get('head_loss_weight', 1.0),
                'cls_loss_weight': cfg.get('losses', {}).get('cls_loss_weight', 0.0),
                'pixel_loss_weight': cfg.get('losses', {}).get('pixel_loss_weight', 0.0),
            }
        }

        # Save updated mappings with retry logic for race conditions
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Write to temp file first, then atomic rename to avoid corruption
                temp_file = mapping_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(mappings, f, indent=2)
                temp_file.replace(mapping_file)

                if is_main_process():
                    print(f"Updated run_mappings.json with new run: {folder_name} -> {wandb.run.id}")
                    print(f"  Wandb name: {wandb.run.name}")
                    print(f"  Group: {base_name}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if is_main_process():
                        print(f"Warning: Failed to write run_mappings.json (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(0.5)  # Brief delay before retry
                else:
                    if is_main_process():
                        print(f"Error: Could not update run_mappings.json after {max_retries} attempts: {e}")
                        print("Continuing without updating mappings file.")

    caption = f"Iters per epoch: {niter_per_ep}, Total steps: {len(lr_schedule)}"
    fig = plot_scheduler(lr_schedule.tolist(), niter_per_ep, caption)
    wandb.log({"Scheduler": wandb.Image(fig_to_pil(fig))}, step=0)
    plt.close(fig)


def build_models_optimizer_and_schedules(cfg: Dict[str, Any], device: torch.device,
                                         num_training_steps_per_epoch: int):
    """Builds models, optimizer, and LR/WD schedules."""

    # Check if teacher model is needed based on loss configuration
    need_teacher = (get_config_value(cfg, "losses.use_head_loss", True) or
                   get_config_value(cfg, "losses.use_cls_loss", False))

    # Conditional teacher model building
    teacher = None
    if need_teacher:
        # Validate teacher configuration exists
        try:
            # Try new config structure first
            teacher_name = get_config_value(cfg, "model.teacher.name")
        except (KeyError, AttributeError):
            # Try old config structure as fallback
            try:
                teacher_name = get_config_value(cfg, "teacher_model", None)
                if teacher_name is None:
                    raise ValueError("Teacher model is required when using head or CLS loss, but no teacher configuration found")
                # Update config to new structure for consistency
                if "model" not in cfg:
                    cfg["model"] = {}
                if "teacher" not in cfg["model"]:
                    cfg["model"]["teacher"] = {}
                cfg["model"]["teacher"]["name"] = teacher_name
            except Exception:
                raise ValueError("Teacher model is required when using head or CLS loss, but teacher configuration is missing or invalid")

        teacher = build_teacher(cfg).to(device).eval()

    # Create masking generator using factory pattern
    print("Creating masking generator...")
    mask_info = get_masking_generator_info(cfg)
    print(f"Masking configuration: {mask_info}")

    # Build student model first
    student_model = build_student(cfg)

    # Create mask generator
    mask_generator = create_masking_generator(cfg, student_model=student_model)
    print(f"Created {type(mask_generator).__name__}")

    # Build full MEDiC model with mask generator
    model = build_medic_model(cfg, mask_generator=mask_generator).to(device)

    print(f"Using GPU index: {get_config_value(cfg, 'gpu')}")
    print(torch.cuda.is_available(), torch.cuda.device_count())
    # Device selection logic: use CPU if requested, otherwise use GPU index
    if cfg.get("cpu", False) or cfg["gpu"] == -1:
        model = model.to("cpu")
    else:
        model = model.to(get_config_value(cfg, "gpu"))
    model_without_ddp = model
    if cfg.get("distributed"):
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg["gpu"]], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Validate configuration compatibility
    use_mask_tokens = get_config_value(cfg, "model.student.use_mask_tokens", True)

    # Check for incompatible configurations
    if not use_mask_tokens:
        # MAE sparse mode validations
        decoder_replace_mask = get_config_value(cfg, "model.decoder.replace_mask_tokens", True)
        if decoder_replace_mask == False:
            raise ValueError(
                "Cannot use encoder mask tokens (replace_mask_tokens=False) "
                "when encoder doesn't use mask tokens (use_mask_tokens=False). "
                "The decoder needs fresh mask tokens in sparse mode."
            )

    # EMA model is only used for evolved masking; ExPLoRe uses block masking only.
    model_ema = None

    # --- Start of Layer-wise Decay Implementation ---
    num_layers = model_without_ddp.student.get_num_layers()
    layer_decay = get_config_value(cfg, "optim.layer_decay", 1.0)

    # Create the mapping from layer id to learning rate scale
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    assigner = LayerDecayValueAssigner(layer_scales)

    # Function to get layer ID for a given parameter name
    def get_num_layer(name):
        return get_num_layer_for_vit(name, num_layers)

    # Function to get the learning rate scale for a given layer ID
    def get_layer_scale(layer_id):
        return assigner.get_scale(layer_id)
    # --- End of Layer-wise Decay Implementation ---

    param_groups = get_parameter_groups(
        model_without_ddp,
        get_config_value(cfg, "optim.wd"),
        get_num_layer=get_num_layer,
        get_layer_scale=get_layer_scale
    )

    # MIRROR BEIT v2: Set top-level weight_decay to 0.0
    # The actual WD is set in each param_group.
    opt = AdamW(
        param_groups,
        lr=get_config_value(cfg, "optim.lr"),
        betas=tuple(get_config_value(cfg, "optim.betas")),
        weight_decay=0.0,
    )

    # MIRROR BEIT v2: Pre-compute schedules and return them
    # CRITICAL FIX: Use proper warmup start value to avoid zero LR at first step
    warmup_start_lr = get_config_value(cfg, "schedule.warmup_start_lr", 1e-6)

    # Handle test mode with limited steps
    max_steps = cfg.get("max_steps")
    if max_steps is not None:
        # For test mode, create a simple constant schedule for the limited steps
        total_steps = max_steps + 1  # Add buffer
        lr_schedule = np.full(total_steps, get_config_value(cfg, "optim.lr"), dtype=np.float32)
        wd_schedule = np.full(total_steps, get_config_value(cfg, "optim.wd"), dtype=np.float32)
    else:
        # Normal full training schedule
        lr_schedule = cosine_scheduler(
            get_config_value(cfg, "optim.lr"), get_config_value(cfg, "schedule.min_lr"), get_config_value(cfg, "epochs"),
            num_training_steps_per_epoch, warmup_epochs=get_config_value(cfg, "schedule.warmup_epochs"),
            start_warmup_value=warmup_start_lr,
        )
        wd_schedule = cosine_scheduler(
            get_config_value(cfg, "optim.wd"), 0, get_config_value(cfg, "epochs"), num_training_steps_per_epoch
        )

    return model, model_without_ddp, teacher, opt, lr_schedule, wd_schedule, model_ema


def load_visualization_images(cfg: Dict[str, Any],
                              device: torch.device) -> Optional[torch.Tensor]:
    """Loads and transforms images from the images/ directory for visualization."""
    if not is_main_process():
        return None

    viz_transform = cast(Compose, build_transform(is_train=False, cfg=cfg))
    img_paths = sorted(glob.glob("images/*.JPEG"))
    if not img_paths:
        print("No images found in images/ for visualization.")
        return None

    try:
        images = [Image.open(p).convert("RGB") for p in img_paths]
        viz_images_list = cast(List[torch.Tensor],
                               [viz_transform(img) for img in images])
        viz_images = torch.stack(viz_images_list).to(device)
        print(f"Loaded {len(viz_images)} images for visualization.")
        return viz_images
    except Exception as e:
        print(f"Error loading visualization images: {e}")
        return None


def handle_checkpointing(
    current_loss: float, epoch: int, model: nn.Module,
    opt: Optimizer, cfg: Dict[str, Any],
    output_dir: Path, best_loss: float, top_k_checkpoints: List[Tuple[float, Path]],
    specific_save_epochs: set, scaler: NativeScalerWithGradNormCount,
    global_step: int, teacher: Optional[nn.Module] = None,
    lr_schedule: Optional[np.ndarray] = None, wd_schedule: Optional[np.ndarray] = None
) -> Tuple[float, List[Tuple[float, Path]]]:
    """Saves last, best-k, and specific epoch checkpoints with complete training state.

    Prevents duplicate saves when the same checkpoint qualifies for multiple categories:
    - If last epoch is in specific_save_epochs, only save once
    - If best checkpoint epoch is in specific_save_epochs, save as best (more informative name)
    """
    # Track what checkpoints we've saved this call to prevent duplicates
    saved_checkpoints = set()

    def save_checkpoint(path: Path):
        # Skip if we've already saved this exact checkpoint
        if path in saved_checkpoints:
            return

        to_save = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "top_k_checkpoints": top_k_checkpoints,
            "config": cfg,
        }

        # NOTE: We don't save teacher (CLIP) state since:
        # 1. CLIP models are pretrained and frozen
        # 2. They can be reloaded from HuggingFace/OpenAI checkpoints
        # 3. This saves ~400MB per checkpoint
        # If you need teacher weights, load them separately from the original source

        # Add schedule states if provided
        if lr_schedule is not None:
            to_save["lr_schedule"] = lr_schedule
        if wd_schedule is not None:
            to_save["wd_schedule"] = wd_schedule

        # Add wandb run information if available
        if wandb.run is not None:
            to_save["wandb_run_id"] = wandb.run.id
            to_save["wandb_run_url"] = wandb.run.url
            to_save["wandb_run_name"] = wandb.run.name
            to_save["wandb_group"] = output_dir.name  # Use folder name as group for eval_knn grouping
            to_save["wandb_project"] = wandb.run.project
            to_save["wandb_entity"] = wandb.run.entity

        torch.save(to_save, path, _use_new_zipfile_serialization=False)
        saved_checkpoints.add(path)

    # Determine total epochs to check if this is the last epoch
    total_epochs = cfg.get('epochs', cfg.get('optim', {}).get('epochs', 300))
    is_last_epoch = (epoch == total_epochs - 1)

    # Track whether we need to save as last/specific epoch
    save_as_last = True
    save_as_specific = epoch in specific_save_epochs

    # Check if current epoch qualifies as best
    # is_new_best = current_loss < best_loss
    is_new_best = False  # deactivate best epoch saving

    # Save as specific epoch if needed
    if save_as_specific:
        save_checkpoint(output_dir / f"checkpoint-epoch{epoch:04d}.pth")

    # Update best_loss if this is a new best (regardless of whether we save it)
    if is_new_best:
        best_loss = current_loss

    # Only save as "best" checkpoint if NOT already in specific_save_epochs
    if is_new_best and epoch not in specific_save_epochs:
        new_path = output_dir / f"checkpoint-best-epoch{epoch:04d}-loss{current_loss:.4f}.pth"
        save_checkpoint(new_path)

        # Update top-k checkpoints list
        top_k_checkpoints.append((current_loss, new_path))
        top_k_checkpoints.sort(key=lambda x: x[0])
        if len(top_k_checkpoints) > 3:
            _, path_to_remove = top_k_checkpoints.pop()
            path_to_remove.unlink(missing_ok=True)

    return best_loss, top_k_checkpoints


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: Optimizer,
    scaler: NativeScalerWithGradNormCount, teacher: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scaler: Gradient scaler to load state into
        teacher: Teacher model to load weights into (optional)
        device: Device to load tensors on

    Returns:
        Dictionary containing restored training state
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint to CPU first to avoid GPU OOM during loading
    # Use weights_only=False because checkpoints contain numpy arrays (lr/wd schedules)
    # The model's load_state_dict will handle moving tensors to the correct device
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Validate checkpoint structure
    required_keys = ["model", "optimizer", "scaler", "epoch", "global_step", "best_loss", "config"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

    # Load model state
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model state from epoch {checkpoint['epoch']}")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded optimizer state")

    # Load scaler state
    scaler.load_state_dict(checkpoint["scaler"])
    print(f"Loaded scaler state")

    # Load teacher state if available (for backwards compatibility with old checkpoints)
    if teacher is not None and "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
        print(f"Loaded teacher state from checkpoint (legacy)")
    elif teacher is not None:
        print(f"Teacher state not in checkpoint (expected - CLIP weights are loaded separately)")

    # Extract training state
    training_state = {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_loss": checkpoint["best_loss"],
        "top_k_checkpoints": checkpoint.get("top_k_checkpoints", []),
        "config": checkpoint["config"],
        "lr_schedule": checkpoint.get("lr_schedule"),
        "wd_schedule": checkpoint.get("wd_schedule"),
    }

    print(f"Resuming from epoch {training_state['epoch']}, step {training_state['global_step']}")
    print(f"Best loss so far: {training_state['best_loss']:.6f}")

    return training_state


def validate_checkpoint_completeness(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate that a checkpoint contains all necessary components for resuming training.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary mapping component names to validation status
    """
    if not os.path.exists(checkpoint_path):
        return {"file_exists": False}

    try:
        # Use weights_only=False because checkpoints contain numpy arrays (lr/wd schedules)
        # This is safe since we're loading our own trusted checkpoints
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        validation_results = {
            "file_exists": True,
            "model_state": "model" in checkpoint,
            "optimizer_state": "optimizer" in checkpoint,
            "scaler_state": "scaler" in checkpoint,
            "epoch": "epoch" in checkpoint,
            "global_step": "global_step" in checkpoint,
            "best_loss": "best_loss" in checkpoint,
            "config": "config" in checkpoint,
            "teacher_state": "teacher" in checkpoint,
            "lr_schedule": "lr_schedule" in checkpoint,
            "wd_schedule": "wd_schedule" in checkpoint,
        }

        # Check state dict completeness
        if "model" in checkpoint:
            validation_results["model_keys"] = len(checkpoint["model"]) > 0
        if "optimizer" in checkpoint:
            validation_results["optimizer_keys"] = len(checkpoint["optimizer"]) > 0
        if "scaler" in checkpoint:
            validation_results["scaler_keys"] = len(checkpoint["scaler"]) > 0

        return validation_results

    except Exception as e:
        return {
            "file_exists": True,
            "loadable": False,
            "error": str(e)
        }


def log_visualizations(
    model_without_ddp: nn.Module, viz_images: Optional[torch.Tensor],
    epoch: int, device: torch.device, global_step: int, cfg: Dict[str, Any],
    model_ema: Optional[nn.Module] = None
):
    """Generates and logs all visualizations to wandb.

    Returns:
        dict: MoE statistics including expert STDs, or None if not available
    """
    if not is_main_process() or viz_images is None:
        return None

    import numpy as np  # Import at function level for use throughout

    # Initialize MoE stats (will be populated if MoE visualization runs)
    moe_stats = None

    # --- 1. Masking/Reconstruction Visualization ---
    # This visualization adapts based on whether decoder is present:
    # - With decoder (v2, v4, v6): Shows Original, Masked, Reconstruction, Pasted (4 columns)
    # - Without decoder (v0, v1, v3): Shows Original, Masked only (2 columns) to visualize masking patterns
    has_decoder = hasattr(model_without_ddp, 'pixel_decoder') and model_without_ddp.pixel_decoder is not None
    if has_decoder:
        print(f"Starting reconstruction visualization at epoch {epoch}, step {global_step} (4-column mode)")
    else:
        print(f"Starting masking visualization at epoch {epoch}, step {global_step} (2-column mode - no decoder)")

    # Store the current RNG state to ensure visualization does not affect training
    rng_state = np.random.get_state()
    try:
        # Set a fixed seed for deterministic mask generation for this plot
        np.random.seed(42)

        # Use the model's actual mask generator instead of hardcoding BlockMaskingGenerator
        if hasattr(model_without_ddp, 'mask_generator') and model_without_ddp.mask_generator is not None:
            mask_generator = model_without_ddp.mask_generator

            # Block masking: generate masks for each sample
            masks_list = []
            for _ in range(viz_images.shape[0]):
                mask = mask_generator()
                # Handle both numpy and tensor outputs
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                masks_list.append(mask)
            viz_mask = torch.stack(masks_list).to(device)
        else:
            # Fallback to block masking if no generator found (backward compatibility)
            print("Warning: No mask generator found in model, falling back to BlockMaskingGenerator")
            fallback_generator = BlockMaskingGenerator(
                grid_h=get_config_value(cfg, "model.student.img_size") // get_config_value(cfg, "model.student.patch_size"),
                grid_w=get_config_value(cfg, "model.student.img_size") // get_config_value(cfg, "model.student.patch_size"),
                mask_ratio=get_config_value(cfg, "mask.mask_ratio"),
            )
            masks_list = []
            for _ in range(viz_images.shape[0]):
                mask = fallback_generator()
                # Handle both numpy and tensor outputs
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                masks_list.append(mask)
            viz_mask = torch.stack(masks_list).to(device)

        viz_batch = (viz_images, None)

        fig = get_reconstruction_fig(
            model_without_ddp, viz_batch, viz_mask,
            epoch, n_images=viz_images.shape[0],
            masking_info=None,
        )
        wandb.log({"Reconstructions": wandb.Image(fig_to_pil(fig))}, step=global_step)
        plt.close(fig)

        if has_decoder:
            print(f"  Successfully logged reconstruction visualization to wandb")
        else:
            print(f"  Successfully logged masking visualization to wandb")
    except Exception as e:
        print(f"ERROR: Could not generate visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore the original RNG state to not affect training stochasticity
        np.random.set_state(rng_state)

    # --- 2. New Attention Visualization Plots ---
    try:
        log_attention_plots_to_wandb(
            model_without_ddp, viz_images, cfg, epoch, global_step
        )
        print(f"Completed attention visualizations at epoch {epoch}")
    except Exception as e:
        print(f"ERROR: Could not generate attention visualizations: {e}")
        import traceback
        traceback.print_exc()

    # --- 3. MoE Weight Visualization ---
    try:
        use_soft_moe = get_config_value(cfg, "model.student.use_soft_moe", False)
        if use_soft_moe:
            print(f"Starting MoE weight visualization at epoch {epoch}, step {global_step}")

            # Get combine weights for logging (last MoE block only)
            model_without_ddp.eval()
            with torch.no_grad():
                _, _, _, combine_weights, _, combine_weights_dict, _, _ = model_without_ddp(viz_images, viz_mask, collect_moe_weights=True)

            # Log per-block weight statistics using combine_weights_dict (all blocks)
            # This replaces the old MoE_LastBlock_Weights which only logged block 11
            weight_type = get_config_value(cfg, "losses.moe_weight_type", "combine")
            if combine_weights_dict is not None and len(combine_weights_dict) > 0:
                log_dict_aggregated = {}
                last_block_expert_stds = []  # For sweep optimization (last block only)

                # Get loss expert indices for highlighting in logs
                first_block_weights = next(iter(combine_weights_dict.values()))
                if isinstance(first_block_weights, dict):
                    sample_weights = first_block_weights.get(weight_type, first_block_weights.get('combine'))
                else:
                    sample_weights = first_block_weights
                num_experts = sample_weights.size(-1)

                loss_expert_indices = get_config_value(cfg, "losses.moe_loss_expert_indices", None)
                if loss_expert_indices is None:
                    loss_expert_indices = list(range(min(2, num_experts)))

                last_block_idx = max(combine_weights_dict.keys())
                for block_idx in sorted(combine_weights_dict.keys()):
                    block_weights = combine_weights_dict[block_idx]
                    if isinstance(block_weights, dict):
                        moe_weights = block_weights.get(weight_type, block_weights.get('combine'))
                    else:
                        moe_weights = block_weights

                    # Remove CLS token and use first image only
                    if moe_weights.size(1) > 0:
                        moe_weights = moe_weights[:, 1:, :]
                    patch_weights = moe_weights[0].cpu()  # [N, num_experts]

                    for expert_idx in range(num_experts):
                        expert_weights = patch_weights[:, expert_idx].cpu().numpy()
                        expert_std = float(expert_weights.std())
                        prefix = f"MoE_WeightStats/Block_{block_idx}/expert_{expert_idx}"
                        log_dict_aggregated[f"{prefix}/mean"] = float(expert_weights.mean())
                        log_dict_aggregated[f"{prefix}/std"] = expert_std
                        log_dict_aggregated[f"{prefix}/max"] = float(expert_weights.max())
                        log_dict_aggregated[f"{prefix}/min"] = float(expert_weights.min())
                        log_dict_aggregated[f"{prefix}/median"] = float(np.median(expert_weights))

                        # Track last block STDs for sweep optimization (backward compatible)
                        if block_idx == last_block_idx:
                            last_block_expert_stds.append(expert_std)

                wandb.log(log_dict_aggregated, step=global_step)
                print(f"Logged per-block MoE weight statistics for {len(combine_weights_dict)} blocks, {num_experts} experts (loss experts: {loss_expert_indices})")

                # Store MoE stats for return (last block STDs for sweep metric)
                moe_stats = {"expert_stds": last_block_expert_stds}

            elif combine_weights is not None:
                # Fallback: no combine_weights_dict, use combine_weights (last block only)
                if isinstance(combine_weights, dict):
                    moe_weights = combine_weights.get(weight_type, combine_weights.get('combine'))
                else:
                    moe_weights = combine_weights
                if moe_weights.size(1) > 0:
                    moe_weights = moe_weights[:, 1:, :]
                patch_weights = moe_weights[0].cpu()
                num_experts = patch_weights.size(1)
                loss_expert_indices = get_config_value(cfg, "losses.moe_loss_expert_indices", None)
                if loss_expert_indices is None:
                    loss_expert_indices = list(range(min(2, num_experts)))
                log_dict_aggregated = {}
                expert_stds = []
                for expert_idx in range(num_experts):
                    expert_weights = patch_weights[:, expert_idx].cpu().numpy()
                    expert_std = float(expert_weights.std())
                    expert_stds.append(expert_std)
                    prefix = f"MoE_WeightStats/Block_last/expert_{expert_idx}"
                    log_dict_aggregated[f"{prefix}/mean"] = float(expert_weights.mean())
                    log_dict_aggregated[f"{prefix}/std"] = expert_std
                    log_dict_aggregated[f"{prefix}/max"] = float(expert_weights.max())
                    log_dict_aggregated[f"{prefix}/min"] = float(expert_weights.min())
                    log_dict_aggregated[f"{prefix}/median"] = float(np.median(expert_weights))
                wandb.log(log_dict_aggregated, step=global_step)
                print(f"Logged MoE weight statistics for last block, {num_experts} experts (loss experts: {loss_expert_indices})")
                moe_stats = {"expert_stds": expert_stds}

            # Generate and log visualization figures (one per MoE block)
            # 1. Softmax-normalized weights (what router produces)
            figures_softmax = get_moe_weight_fig(
                model_without_ddp, viz_batch, viz_mask, epoch, cfg,
                n_images=viz_images.shape[0],
                apply_loss_normalization=False
            )
            if figures_softmax is not None and len(figures_softmax) > 0:
                log_dict = {}
                for block_idx, fig in figures_softmax.items():
                    log_dict[f"MoE/Weights_Softmax/Block_{block_idx}"] = wandb.Image(fig_to_pil(fig))
                    plt.close(fig)
                wandb.log(log_dict, step=global_step)
                print(f"Logged {len(figures_softmax)} softmax-normalized MoE visualizations (blocks: {sorted(figures_softmax.keys())})")

            # 2. Loss-normalized weights (only if normalization is actually enabled in training)
            moe_normalize_weights = get_config_value(cfg, "losses.moe_normalize_weights", True)
            figures_loss = None
            if moe_normalize_weights:
                # Only generate loss-normalized viz if it's actually being applied during training
                figures_loss = get_moe_weight_fig(
                    model_without_ddp, viz_batch, viz_mask, epoch, cfg,
                    n_images=viz_images.shape[0],
                    apply_loss_normalization=True
                )
                if figures_loss is not None and len(figures_loss) > 0:
                    log_dict = {}
                    for block_idx, fig in figures_loss.items():
                        log_dict[f"MoE/Weights_LossNorm/Block_{block_idx}"] = wandb.Image(fig_to_pil(fig))
                        plt.close(fig)
                    wandb.log(log_dict, step=global_step)
                    print(f"Logged {len(figures_loss)} loss-normalized MoE visualizations (blocks: {sorted(figures_loss.keys())})")
            else:
                print("Skipping MoE/Weights_LossNorm visualization (moe_normalize_weights=False)")

            if (figures_softmax is None or len(figures_softmax) == 0) and (figures_loss is None or len(figures_loss) == 0):
                print(f"MoE visualization skipped (no active experts or not enabled)")

            # --- 4. Log MoE Scale Parameters ---
            try:
                use_soft_moe = get_config_value(cfg, "model.student.use_soft_moe", False)
                if use_soft_moe:
                    print(f"[MoE Scale] Logging at epoch {epoch}, step {global_step}")

                    # Extract scale parameters from model
                    # Parameter name pattern: "blocks.X.mlp.scale" where X is block index
                    scale_values = []
                    scale_by_block = {}
                    for name, param in model_without_ddp.student.named_parameters():
                        if name.endswith('mlp.scale') and name.startswith('blocks.'):
                            # Extract block number (e.g., "blocks.11.mlp.scale" -> 11)
                            parts = name.split('.')
                            if len(parts) >= 2 and parts[1].isdigit():
                                block_idx = int(parts[1])
                                scale_val = param.item()
                                scale_values.append(scale_val)
                                scale_by_block[block_idx] = scale_val

                    # Log if scales found
                    if scale_values:
                        log_data_scale = {
                            "MoE/Scale/mean": np.mean(scale_values),
                            "MoE/Scale/std": np.std(scale_values),
                            "MoE/Scale/max": np.max(scale_values),
                            "MoE/Scale/min": np.min(scale_values),
                        }

                        # Add per-block scales
                        for block_idx, scale_val in scale_by_block.items():
                            log_data_scale[f"MoE/Scale/block_{block_idx}"] = scale_val

                        # Add ratio if min > 0
                        max_val = np.max(scale_values)
                        min_val = np.min(scale_values)
                        if min_val > 0:
                            log_data_scale["MoE/Scale/max_min_ratio"] = max_val / min_val

                        # Log with global_step for monotonic consistency
                        wandb.log(log_data_scale, step=global_step)
                        print(f"  Scales by block: {scale_by_block} | mean={np.mean(scale_values):.4f}")
                    else:
                        print("  ⚠️  Warning: No MoE scale parameters found!")

            except Exception as e:
                print(f"  ❌ ERROR logging MoE scale: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"ERROR: Could not generate MoE weight visualizations: {e}")
        import traceback
        traceback.print_exc()

    return moe_stats


def train_one_epoch(
    model: nn.Module,
    teacher: Optional[nn.Module],
    train_loader: DataLoader,
    opt: Optimizer,
    lr_schedule_values: np.ndarray,
    wd_schedule_values: np.ndarray,
    scaler: NativeScalerWithGradNormCount,
    epoch: int,
    global_step: int,
    dtype: torch.dtype,
    device: torch.device,
    cfg: Dict[str, Any],
    model_ema: Optional[Any] = None,
) -> Tuple[int, Dict[str, float]]:
    """Runs a single training epoch."""
    model.train()

    epoch_start_time = time.time()
    iter_times = []
    epoch_samples = 0

    # Get gradient accumulation steps
    accum_iter = get_config_value(cfg, "optim.accum_iter", 1)

    # Zero gradients at the beginning of epoch
    opt.zero_grad()

    for i, (img, mask) in enumerate(train_loader):
        batch_start_time = time.time()
        it = len(train_loader) * epoch + i

        # Check for max steps limit (test mode)
        max_steps = cfg.get("max_steps")
        if max_steps is not None and global_step >= max_steps:
            if is_main_process():
                print(f"Reached maximum steps limit: {max_steps}")
            break

        # MIRROR BEIT v2: Manually set LR and WD at accumulation boundaries
        # We update LR per accumulation step, not per data batch
        if i % accum_iter == 0:
            # Use the accumulation step index for LR scheduling
            accum_step = (it // accum_iter)
            for param_group in opt.param_groups:
                param_group["lr"] = lr_schedule_values[min(accum_step, len(lr_schedule_values)-1)] * param_group.get("lr_scale", 1.0)
                if param_group["weight_decay"] > 0: # only update wd for groups that have it
                    param_group["weight_decay"] = wd_schedule_values[min(accum_step, len(wd_schedule_values)-1)]

        img = img.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        epoch_samples += img.shape[0]

        assert isinstance(train_loader.dataset, MEDiCDataset)

        with torch.autocast(device_type='cuda', dtype=dtype):
            # Check if using sparse encoder
            use_mask_tokens = get_config_value(cfg, "model.student.use_mask_tokens", True)

            # Only compute teacher forward if teacher model exists
            T = None
            if teacher is not None:
                # Teacher always processes full image
                T = teacher(img, return_cls=True)
                if get_config_value(cfg, "losses.normalize_targets", True):
                    mean = T.mean(dim=-1, keepdim=True)
                    var = T.var(dim=-1, keepdim=True)
                    T = (T - mean) / (var + 1.0e-6) ** 0.5

            # unpack all return values including per-loss MoE weights (7th, 8th)
            pred_tok, pred_pix, mask_out, combine_weights, ids_restore, combine_weights_dict, head_combine_weights, pixel_combine_weights = model(img, mask, epoch=epoch, model_ema=model_ema, teacher=teacher)
            if mask is None:
                mask = mask_out

            # Debug: Check shapes before sparse mode handling (disabled for performance)
            # if pred_tok is not None and T is not None and i == 0:  # Only print once per epoch
            #     print(f"[Debug] Before sparse handling - Teacher shape: {T.shape}, Student shape: {pred_tok.shape}, use_mask_tokens: {use_mask_tokens}")

            # Handle sparse mode: extract visible positions from teacher (FULLY VECTORIZED)
            if not use_mask_tokens and T is not None:
                # MAE sparse mode: need to extract visible positions from teacher
                visible_mask = ~mask  # [B, N]
                batch_size = T.shape[0]
                embed_dim = T.shape[-1]
                device = T.device

                # Extract CLS token (always present)
                T_cls = T[:, :1, :]  # [B, 1, D]
                T_patches = T[:, 1:, :]  # [B, N, D]

                # Count visible patches per sample
                num_visible_per_sample = visible_mask.sum(dim=1)  # [B]

                # Check if all samples have the same number of visible patches (FAST PATH)
                if torch.all(num_visible_per_sample == num_visible_per_sample[0]):
                    # FAST PATH: All samples have same number of visible patches
                    # This is common with block masking at fixed ratio
                    num_visible = num_visible_per_sample[0].item()

                    if num_visible > 0:
                        # Get visible positions as (batch_idx, patch_idx) pairs
                        visible_indices = visible_mask.nonzero(as_tuple=False)  # [total_visible, 2]

                        # Extract patch indices PER BATCH to handle correct ordering
                        # The nonzero() operation returns indices in row-major order,
                        # so we need to group by batch to ensure correct assignment
                        patch_indices_list = []
                        for b in range(batch_size):
                            batch_mask = visible_indices[:, 0] == b
                            batch_patch_indices = visible_indices[batch_mask, 1]
                            patch_indices_list.append(batch_patch_indices)

                        # Stack into tensor [B, num_visible]
                        patch_indices = torch.stack(patch_indices_list, dim=0)

                        # Use gather for efficient batch-parallel extraction
                        T_visible = torch.gather(T_patches, dim=1,
                                               index=patch_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
                    else:
                        # Edge case: all patches masked
                        T_visible = torch.zeros(batch_size, 0, embed_dim, device=device, dtype=T.dtype)
                else:
                    # This should NEVER happen with fixed-ratio masking
                    raise RuntimeError(
                        f"SLOW PATH triggered! This should never happen with fixed-ratio masking.\n"
                        f"Visible patches per sample: {num_visible_per_sample.tolist()}\n"
                        f"Expected all samples to have {num_visible_per_sample[0].item()} visible patches.\n"
                        f"Check mask generator configuration."
                    )

                # Combine CLS with visible patches
                T = torch.cat([T_cls, T_visible], dim=1)  # [B, num_visible+1, D]

                # CRITICAL FIX: Align teacher features with student's shuffled order
                # When student uses shuffling, the visible patches are in a different order
                # than what we just extracted. We need to reorder teacher features to match.
                model_to_check = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_check, 'student') and hasattr(model_to_check.student, 'ids_shuffle'):
                    ids_shuffle = model_to_check.student.ids_shuffle
                    if ids_shuffle is not None:
                        # Student has shuffled patches - need to reorder teacher to match
                        # CORRECT APPROACH: Apply same shuffle to teacher, then extract visible

                        T_patches_aligned = []
                        for b in range(batch_size):
                            # Get shuffle indices for this batch
                            if ids_shuffle.dim() == 1:
                                # Same shuffle for all batches
                                batch_shuffle = ids_shuffle
                            else:
                                # Per-batch shuffle
                                batch_shuffle = ids_shuffle[b]

                            # Apply shuffle to ALL teacher patches first
                            T_patches_shuffled = T_patches[b, batch_shuffle, :]  # [N, D]

                            # Now extract visible patches from the SHUFFLED order
                            # The mask was also shuffled in the student, so apply same shuffle
                            if ids_shuffle.dim() == 1:
                                mask_shuffled = mask[b, batch_shuffle]
                            else:
                                # For per-batch, mask was already shuffled in encoder
                                # Use the shuffled mask to extract visible
                                mask_shuffled = torch.gather(mask[b].float(), dim=0, index=batch_shuffle).bool()

                            visible_mask_shuffled = ~mask_shuffled

                            # Extract visible patches from shuffled teacher
                            T_patches_visible_shuffled = T_patches_shuffled[visible_mask_shuffled]

                            # Add batch dimension back
                            T_patches_aligned.append(T_patches_visible_shuffled.unsqueeze(0))

                        # Stack all batches
                        T_visible_aligned = torch.cat(T_patches_aligned, dim=0)

                        # Update T with properly aligned visible patches
                        T = torch.cat([T_cls, T_visible_aligned], dim=1)

            # --- Start of Runtime Safety Assertions ---
            batch_size = img.shape[0]
            num_patches = (get_config_value(cfg, 'model.student.img_size') // get_config_value(cfg, 'model.student.patch_size')) ** 2
            embed_dim = get_config_value(cfg, 'model.student.embed_dim')

            if T is not None:
                if use_mask_tokens:
                    assert T.shape == (batch_size, num_patches + 1, embed_dim), f"Teacher tensor shape incorrect: got {T.shape}"
                else:
                    # In sparse mode, teacher shape matches student shape (variable length)
                    assert T.shape[0] == batch_size, f"Teacher batch size incorrect: got {T.shape[0]}"
                    assert T.shape[1] == pred_tok.shape[1], f"Teacher/student sequence length mismatch: {T.shape[1]} vs {pred_tok.shape[1]}"
                assert T.device == device, f"Teacher tensor is on wrong device: {T.device}"
            if pred_tok is not None:
                if use_mask_tokens:
                    assert pred_tok.shape == (batch_size, num_patches + 1, embed_dim), f"Student tensor shape incorrect: got {pred_tok.shape}"
                assert pred_tok.device == device, f"Student tensor is on wrong device: {pred_tok.device}"
            if mask is not None:
                assert mask.shape == (batch_size, num_patches), f"Mask shape incorrect: got {mask.shape}"
            # --- End of Runtime Safety Assertions ---

            # Calculate current micro-step within accumulation cycle
            current_micro_step = i % accum_iter

            loss, l_dict = compute_three_losses(
                pred_tok=pred_tok,
                pred_pix=pred_pix,
                T=T,
                img=img,
                mask=mask,
                cfg=cfg,
                model=model,
                accum_iter=accum_iter,
                current_micro_step=current_micro_step,
                use_mask_tokens=use_mask_tokens,
                combine_weights=combine_weights,
                ids_restore=ids_restore,
                combine_weights_dict=combine_weights_dict,
                head_combine_weights=head_combine_weights,
                pixel_combine_weights=pixel_combine_weights,
            )

            # Scale loss by accumulation steps to maintain same gradient magnitude
            loss = loss / accum_iter

        # Determine if we should update gradients this iteration
        should_update_grad = ((i + 1) % accum_iter == 0) or ((i + 1) == len(train_loader))

        # Standard backward pass and optimization
        grad_norm = scaler(
            loss,
            opt,
            parameters=model.parameters(),
            clip_grad=get_config_value(cfg, "optim.grad_clip"),
            update_grad=should_update_grad,
            create_graph=False,
        )

        # Zero gradients after optimizer step when accumulation complete
        if should_update_grad:
            opt.zero_grad()

        torch.cuda.synchronize()
        iter_times.append(time.time() - batch_start_time)

        # Only log at accumulation boundaries when gradients are actually updated
        if is_main_process() and should_update_grad and (it % 20 == 0 or it == 0):
            # Note: loss needs to be unscaled for logging (multiply back by accum_iter)
            log_data = {
                "train_loss": loss.item() * accum_iter,  # Unscale for logging
                "train_head_loss": l_dict.get("L_head", 0),
                "train_decoder_loss": l_dict.get("L_pix", 0),
                "train_cls_loss": l_dict.get("L_cls", 0),
                "Optimizer/grad_norm": grad_norm,
                "step": global_step,
                "epoch": epoch,
            }

            # Log unweighted losses if MoE weighting is enabled
            if l_dict.get("L_head_unweighted") is not None:
                log_data["train_head_loss_unweighted"] = l_dict.get("L_head_unweighted", 0)
            if l_dict.get("L_pix_unweighted") is not None:
                log_data["train_decoder_loss_unweighted"] = l_dict.get("L_pix_unweighted", 0)

            # Log MoE regularization losses (importance and entropy)
            if "L_importance" in l_dict:
                log_data["train_importance_loss"] = l_dict["L_importance"]
                log_data["MoE/importance_cv"] = l_dict.get("importance_cv", 0)
                # Also log additional importance metrics
                if "importance_cv_squared" in l_dict:
                    log_data["MoE/importance_cv_squared"] = l_dict["importance_cv_squared"]
                # Log per-expert importance scores for load balancing monitoring
                for key in l_dict.keys():
                    if key.startswith("expert_") and key.endswith("_importance"):
                        log_data[f"MoE/{key}"] = l_dict[key]
            if "L_dispatch_entropy" in l_dict:
                log_data["train_dispatch_entropy_loss"] = l_dict["L_dispatch_entropy"]
                log_data["MoE/dispatch_entropy"] = l_dict.get("dispatch_entropy", 0)
                log_data["MoE/dispatch_uniformity"] = l_dict.get("dispatch_uniformity", 0)
                # Log additional entropy metrics
                if "dispatch_entropy_max" in l_dict:
                    log_data["MoE/dispatch_entropy_max"] = l_dict["dispatch_entropy_max"]
                if "entropy_variance" in l_dict:
                    log_data["MoE/entropy_variance"] = l_dict["entropy_variance"]
                # Log per-expert entropy dynamically for any number of experts
                for key in l_dict.keys():
                    if key.startswith("expert_") and key.endswith("_entropy"):
                        log_data[f"MoE/{key}"] = l_dict[key]
                    # Also log block-level metrics (entropy and uniformity per block)
                    elif key.startswith("block_"):
                        log_data[f"MoE/{key}"] = l_dict[key]

            # Log multi-block aggregator weights if using learned_scalar
            model_for_agg = model.module if hasattr(model, 'module') else model
            if hasattr(model_for_agg, 'head_weight_aggregator') and model_for_agg.head_weight_aggregator is not None:
                agg = model_for_agg.head_weight_aggregator
                if hasattr(agg, 'block_logits'):
                    contributions = torch.softmax(agg.block_logits, dim=0).detach()
                    for idx, block_id in enumerate(model_for_agg.head_block_indices):
                        log_data[f"MoE/head_block_{block_id}_weight"] = contributions[idx].item()
            if hasattr(model_for_agg, 'pixel_weight_aggregator') and model_for_agg.pixel_weight_aggregator is not None:
                agg = model_for_agg.pixel_weight_aggregator
                if hasattr(agg, 'block_logits'):
                    contributions = torch.softmax(agg.block_logits, dim=0).detach()
                    for idx, block_id in enumerate(model_for_agg.pixel_block_indices):
                        log_data[f"MoE/pixel_block_{block_id}_weight"] = contributions[idx].item()

            for ind, param_group in enumerate(opt.param_groups):
                log_data[f"Optimizer/lr_scale_{ind}"] = param_group["lr_scale"]
                log_data[f"Optimizer/lr_{ind}"] = param_group["lr"]
                log_data[f"Optimizer/weight_decay_{ind}"] = param_group["weight_decay"]

            wandb.log(log_data, step=global_step)

        # Only increment global step when gradients are actually updated
        if should_update_grad:
            global_step += 1

    epoch_time = time.time() - epoch_start_time
    throughput = epoch_samples / epoch_time if epoch_time > 0 else 0
    avg_iter_time = sum(iter_times) / len(iter_times) if iter_times else 0

    epoch_stats = {
        "train/epoch_time": epoch_time,
        "train/throughput": throughput,
        "train/avg_iter_time": avg_iter_time,
    }
    return global_step, epoch_stats


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    teacher: Optional[nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    cfg: Dict[str, Any],
    epoch: int = 0,
    model_ema: Optional[Any] = None,
) -> Dict[str, float]:
    """Runs a single validation epoch."""
    model.eval()

    val_stats: Dict[str, torch.Tensor] = {
        "val_loss": torch.tensor(0.0, device=device),
        "val_head_loss": torch.tensor(0.0, device=device),
        "val_decoder_loss": torch.tensor(0.0, device=device),
        "val_cls_loss": torch.tensor(0.0, device=device),
        "val_head_loss_unweighted": torch.tensor(0.0, device=device),
        "val_decoder_loss_unweighted": torch.tensor(0.0, device=device),
    }
    val_batches = 0
    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        assert isinstance(val_loader.dataset, MEDiCDataset)

        with torch.autocast(device_type='cuda', dtype=dtype):
            # Check if using sparse encoder
            use_mask_tokens = get_config_value(cfg, "model.student.use_mask_tokens", True)

            # Only compute teacher forward if teacher model exists
            T = None
            if teacher is not None:
                T = teacher(img, return_cls=True)
                # CRITICAL FIX: Normalize teacher targets following reference MEDiC
                if get_config_value(cfg, "losses.normalize_targets", True):
                    mean = T.mean(dim=-1, keepdim=True)
                    var = T.var(dim=-1, keepdim=True)
                    T = (T - mean) / (var + 1.0e-6) ** 0.5

            pred_tok, pred_pix, mask_out, combine_weights, ids_restore, combine_weights_dict, head_combine_weights, pixel_combine_weights = model(img, mask, epoch=epoch, model_ema=model_ema, teacher=teacher)
            if mask is None:
                mask = mask_out

            # Handle sparse mode: extract visible positions from teacher (FULLY VECTORIZED)
            use_mask_tokens = get_config_value(cfg, "model.student.use_mask_tokens", True)
            if not use_mask_tokens and T is not None:
                # MAE sparse mode: need to extract visible positions from teacher
                visible_mask = ~mask  # [B, N]
                batch_size = T.shape[0]
                embed_dim = T.shape[-1]
                device = T.device

                # Extract CLS token (always present)
                T_cls = T[:, :1, :]  # [B, 1, D]
                T_patches = T[:, 1:, :]  # [B, N, D]

                # Count visible patches per sample
                num_visible_per_sample = visible_mask.sum(dim=1)  # [B]

                # Check if all samples have the same number of visible patches (FAST PATH)
                if torch.all(num_visible_per_sample == num_visible_per_sample[0]):
                    # FAST PATH: All samples have same number of visible patches
                    num_visible = num_visible_per_sample[0].item()

                    if num_visible > 0:
                        # Get visible positions as (batch_idx, patch_idx) pairs
                        visible_indices = visible_mask.nonzero(as_tuple=False)  # [total_visible, 2]

                        # Extract patch indices PER BATCH to handle correct ordering
                        # The nonzero() operation returns indices in row-major order,
                        # so we need to group by batch to ensure correct assignment
                        patch_indices_list = []
                        for b in range(batch_size):
                            batch_mask = visible_indices[:, 0] == b
                            batch_patch_indices = visible_indices[batch_mask, 1]
                            patch_indices_list.append(batch_patch_indices)

                        # Stack into tensor [B, num_visible]
                        patch_indices = torch.stack(patch_indices_list, dim=0)

                        # Use gather for efficient batch-parallel extraction
                        T_visible = torch.gather(T_patches, dim=1,
                                               index=patch_indices.unsqueeze(-1).expand(-1, -1, embed_dim))
                    else:
                        # Edge case: all patches masked
                        T_visible = torch.zeros(batch_size, 0, embed_dim, device=device, dtype=T.dtype)
                else:
                    # This should NEVER happen with fixed-ratio masking
                    raise RuntimeError(
                        f"SLOW PATH triggered! This should never happen with fixed-ratio masking.\n"
                        f"Visible patches per sample: {num_visible_per_sample.tolist()}\n"
                        f"Expected all samples to have {num_visible_per_sample[0].item()} visible patches.\n"
                        f"Check mask generator configuration."
                    )

                # Combine CLS with visible patches
                T = torch.cat([T_cls, T_visible], dim=1)  # [B, num_visible+1, D]

                # CRITICAL FIX: Align teacher features with student's shuffled order (same as training)
                model_to_check = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_check, 'student') and hasattr(model_to_check.student, 'ids_shuffle'):
                    ids_shuffle = model_to_check.student.ids_shuffle
                    if ids_shuffle is not None:
                        # Apply same shuffle alignment as in training
                        visible_indices_orig = visible_mask.nonzero(as_tuple=False)
                        T_patches_aligned = []
                        for b in range(batch_size):
                            batch_mask = visible_indices_orig[:, 0] == b
                            batch_visible_orig = visible_indices_orig[batch_mask, 1]

                            if ids_shuffle.dim() == 1:
                                batch_shuffle = ids_shuffle
                            else:
                                batch_shuffle = ids_shuffle[b]

                            shuffled_positions = batch_shuffle[batch_visible_orig]
                            sorted_indices = torch.argsort(shuffled_positions)
                            T_patches_aligned.append(T_visible[b:b+1, sorted_indices, :])

                        T_visible_aligned = torch.cat(T_patches_aligned, dim=0)
                        T = torch.cat([T_cls, T_visible_aligned], dim=1)

            if get_config_value(cfg, "losses.normalize_predictions", False) and pred_tok is not None:
                mean = pred_tok.mean(dim=-1, keepdim=True)
                var = pred_tok.var(dim=-1, keepdim=True)
                pred_tok = (pred_tok - mean) / (var + 1.0e-6) ** 0.5

        loss, l_dict = compute_three_losses(
            pred_tok=pred_tok,
            pred_pix=pred_pix,
            T=T,
            img=img,
            mask=mask,
            cfg=cfg,
            model=model,
            accum_iter=1,  # No accumulation in validation
            current_micro_step=0,
            use_mask_tokens=use_mask_tokens,
            combine_weights=combine_weights,
            ids_restore=ids_restore,
            combine_weights_dict=combine_weights_dict,
            head_combine_weights=head_combine_weights,
            pixel_combine_weights=pixel_combine_weights,
        )
        val_batches += 1
        val_stats["val_loss"] += loss
        val_stats["val_head_loss"] += l_dict.get("L_head", 0.0)
        val_stats["val_decoder_loss"] += l_dict.get("L_pix", 0.0)
        val_stats["val_cls_loss"] += l_dict.get("L_cls", 0.0)
        # Accumulate unweighted losses if available
        if l_dict.get("L_head_unweighted") is not None:
            val_stats["val_head_loss_unweighted"] += l_dict.get("L_head_unweighted", 0.0)
        if l_dict.get("L_pix_unweighted") is not None:
            val_stats["val_decoder_loss_unweighted"] += l_dict.get("L_pix_unweighted", 0.0)

        # Accumulate MoE losses if present
        if "L_importance" in l_dict:
            if "val_importance_loss" not in val_stats:
                val_stats["val_importance_loss"] = torch.tensor(0.0, device=device)
            val_stats["val_importance_loss"] += l_dict["L_importance"]
            # Also accumulate importance diagnostic metrics
            if "importance_cv" in l_dict:
                if "val_importance_cv" not in val_stats:
                    val_stats["val_importance_cv"] = torch.tensor(0.0, device=device)
                val_stats["val_importance_cv"] += l_dict["importance_cv"]
        if "L_dispatch_entropy" in l_dict:
            if "val_dispatch_entropy_loss" not in val_stats:
                val_stats["val_dispatch_entropy_loss"] = torch.tensor(0.0, device=device)
            val_stats["val_dispatch_entropy_loss"] += l_dict["L_dispatch_entropy"]
            # Also accumulate entropy diagnostic metrics
            if "dispatch_entropy" in l_dict:
                if "val_dispatch_entropy" not in val_stats:
                    val_stats["val_dispatch_entropy"] = torch.tensor(0.0, device=device)
                val_stats["val_dispatch_entropy"] += l_dict["dispatch_entropy"]
            if "dispatch_uniformity" in l_dict:
                if "val_dispatch_uniformity" not in val_stats:
                    val_stats["val_dispatch_uniformity"] = torch.tensor(0.0, device=device)
                val_stats["val_dispatch_uniformity"] += l_dict["dispatch_uniformity"]

    if get_config_value(cfg, "distributed", False):
        for k in val_stats:
            dist.all_reduce(val_stats[k], op=dist.ReduceOp.SUM)

    world_size = get_world_size() if get_config_value(cfg, "distributed", False) else 1
    avg_val_batches = val_batches * world_size

    final_stats = {k: v.item() / avg_val_batches for k, v in val_stats.items()}
    return final_stats

def copy_source_files(output_dir: Path):
    """Copy key source files to checkpoint for reproducibility."""
    if not is_main_process():
        return

    src_snapshot = output_dir / "src_snapshot"
    src_snapshot.mkdir(exist_ok=True)

    # Define files to preserve
    files_to_copy = [
        "src/train.py",
        "src/utils/losses.py",
    ]

    # Copy individual files
    for file_path in files_to_copy:
        src = Path(file_path)
        if src.exists():
            dst = src_snapshot / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Copy entire models directory
    models_src = Path("src/models")
    models_dst = src_snapshot / "models"
    if models_src.exists():
        shutil.copytree(models_src, models_dst, dirs_exist_ok=True)

    print(f"Source snapshot saved to: {src_snapshot}")

def main():
    """
    Main entry point for the MEDiC pre-training script.

    Initializes the distributed environment, builds the models, sets up the optimizer
    and schedulers, and runs the main training and validation loop.
    """
    cfg = setup_environment()
    output_dir = Path(cfg["output_dir"])

    # Log training paths early
    if is_main_process():
        print("="*80)
        print("TRAINING PATH INFORMATION")
        print("="*80)
        print(f"Checkpoint Directory: {output_dir.absolute()}")

        # Log SLURM information if available
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_job_name = os.environ.get("SLURM_JOB_NAME")
        if slurm_job_id:
            log_file = Path.cwd() / "logs" / "pretrain" / slurm_job_name / f"{slurm_job_id}.log"
            print(f"SLURM Log File: {log_file}")
        print("="*80)

        # Copy source files for reproducibility
        copy_source_files(output_dir)

    use_cpu = cfg.get("cpu", False) or cfg["gpu"] == -1
    device = torch.device("cpu" if use_cpu else f"cuda:{cfg['gpu']}")
    dtype = torch.bfloat16 if get_config_value(cfg, "precision", "bf16") == "bf16" else torch.float16

    # 1. Build loaders first
    train_loader, val_loader = build_loader(cfg)

    # 2. Build models and schedules using loader info
    model, model_without_ddp, teacher, opt, lr_schedule, wd_schedule, model_ema = \
        build_models_optimizer_and_schedules(cfg, device, len(train_loader))

    # 3. Init wandb
    init_wandb(cfg, output_dir, lr_schedule, len(train_loader))

    viz_images = load_visualization_images(cfg, device)

    scaler = NativeScalerWithGradNormCount()

    # Initialize training state
    global_step = 0
    best_loss = float("inf")
    top_k_checkpoints: List[Tuple[float, Path]] = []
    start_epoch = 0

    # Resume training if checkpoint provided
    if cfg.get("resume_path"):
        resume_path = get_config_value(cfg, "resume_path")
        print(f"Attempting to resume training from: {resume_path}")

        # Validate checkpoint before loading
        validation_results = validate_checkpoint_completeness(resume_path)
        print("Checkpoint validation results:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")

        if not validation_results.get("file_exists", False):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        if not validation_results.get("loadable", True):
            raise ValueError(f"Cannot load checkpoint: {validation_results.get('error', 'Unknown error')}")

        # Load checkpoint and restore training state
        training_state = load_checkpoint(
            resume_path, model, opt, scaler, teacher, device
        )

        # Restore training state
        start_epoch = training_state["epoch"] + 1  # Start from next epoch
        global_step = training_state["global_step"]
        best_loss = training_state["best_loss"]
        top_k_checkpoints = training_state["top_k_checkpoints"]

        # Use saved schedules if available, otherwise keep current ones
        if training_state.get("lr_schedule") is not None:
            lr_schedule = training_state["lr_schedule"]
            print("Restored saved learning rate schedule")
        if training_state.get("wd_schedule") is not None:
            wd_schedule = training_state["wd_schedule"]
            print("Restored saved weight decay schedule")

        print(f"Resumed training from epoch {start_epoch}, step {global_step}")
        print(f"Best loss so far: {best_loss:.6f}")
    else:
        print(f"Starting fresh training for {get_config_value(cfg, 'epochs')} epochs")

    start_time = time.time()
    total_epochs = get_config_value(cfg, "epochs")
    specific_save_epochs = set(
        list(range(20, 181, 20))            # 20, 40, 60, 80, 100, 120, 140, 160, 180  (rolling early-epoch checkpoints, only most recent kept by external cleanup script)
        + list(range(200, 291, 10))         # 200, 210, 220, 230, 240, 250, 260, 270, 280, 290
        + [295, total_epochs - 1]           # 295, 299
    )
    # Remove epochs we don't want
    specific_save_epochs.discard(210)
    specific_save_epochs.discard(230)

    for epoch in range(start_epoch, total_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if is_main_process():
            wandb.log({"epoch": epoch, "step": global_step}, step=global_step)

        global_step, train_stats = train_one_epoch(
            model, teacher, train_loader, opt, lr_schedule,
            wd_schedule, scaler, epoch, global_step, dtype, device, cfg, model_ema
        )
        if is_main_process():
            wandb.log({**train_stats, "epoch": epoch, "step": global_step}, step=global_step)

        val_stats = validate_one_epoch(model, teacher, val_loader, device, dtype, cfg, epoch, model_ema)

        # Compute combined validation loss for sweep optimization
        if "val_head_loss_unweighted" in val_stats and "val_decoder_loss_unweighted" in val_stats:
            val_combined_loss = val_stats["val_head_loss_unweighted"] + val_stats["val_decoder_loss_unweighted"]
            val_stats["val_combined_loss_unweighted"] = val_combined_loss

        if is_main_process():
            # Log visualizations and get MoE stats
            moe_stats = log_visualizations(model_without_ddp, viz_images, epoch, device, global_step, cfg, model_ema)

            # Compute dual-objective sweep metric combining validation loss and MoE weight uniformity
            # Higher STD = less uniform expert usage = worse (we want to minimize both)
            if "val_combined_loss_unweighted" in val_stats and moe_stats is not None and "expert_stds" in moe_stats:
                expert_stds = moe_stats["expert_stds"]
                # For 2 experts, sum their STDs (higher STD = less uniform distribution)
                # Scale STD by 100 to make it comparable in magnitude to validation loss (~1-10 range)
                # This makes both components have roughly equal importance in the optimization
                moe_std_sum = sum(expert_stds[:2]) if len(expert_stds) >= 2 else sum(expert_stds)
                sweep_metric = val_stats["val_combined_loss_unweighted"] + (moe_std_sum * 100)
                val_stats["sweep_metric_dual_objective"] = sweep_metric
                print(f"Sweep dual-objective metric: loss={val_stats['val_combined_loss_unweighted']:.4f} + std={moe_std_sum:.6f}*100 = {sweep_metric:.4f}")

            wandb.log({**val_stats, "epoch": epoch, "step": global_step}, step=global_step)

            current_loss = val_stats.get("val_loss", float("inf"))
            best_loss, top_k_checkpoints = handle_checkpointing(
                current_loss, epoch, model, opt, cfg, output_dir,
                best_loss, top_k_checkpoints, specific_save_epochs,
                scaler, global_step, teacher, lr_schedule, wd_schedule
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Generate config usage report if tracking was enabled
    if "_config_tracker" in cfg:
        print("\n" + "="*50)
        print("CONFIG USAGE REPORT")
        print("="*50)
        cfg["_config_tracker"].print_report()
        print("="*50)


if __name__ == "__main__":
    main()
