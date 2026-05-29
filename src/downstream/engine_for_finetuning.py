# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
import os
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

# MoE entropy loss for finetuning stability (based on ST-MoE/FLAN-MoE research)
# Import from src/utils (not the local utils module)
try:
    # Need to import using absolute path to avoid conflict with local utils module
    import importlib.util
    _losses_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'losses.py')
    _spec = importlib.util.spec_from_file_location("src_utils_losses", _losses_path)
    _losses_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_losses_module)
    compute_dispatch_entropy_loss = _losses_module.compute_dispatch_entropy_loss
    MOE_ENTROPY_AVAILABLE = True
except Exception as e:
    MOE_ENTROPY_AVAILABLE = False
    print(f"Warning: MoE entropy loss not available: {e}")


def train_class_batch(model, samples, target, criterion, return_moe_weights=False):
    """Forward pass with optional MoE weight collection for entropy loss.

    Args:
        model: The model to run forward pass on
        samples: Input samples
        target: Target labels
        criterion: Loss criterion
        return_moe_weights: If True, attempt to get MoE weights from model

    Returns:
        loss: Classification loss
        outputs: Model outputs (logits)
        moe_weights: MoE routing weights if available, else None
    """
    moe_weights = None

    if return_moe_weights:
        # Try to get MoE weights from the model
        # VisionTransformerMIM supports return_combine_weights parameter
        try:
            outputs = model(samples, return_combine_weights=True)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # Model returned (logits, moe_weights) or (logits, aux, moe_weights)
                logits = outputs[0]
                moe_weights = outputs[-1]  # Last element is MoE weights
            else:
                logits = outputs
        except TypeError:
            # Model doesn't support return_combine_weights
            logits = model(samples)
    else:
        logits = model(samples)

    loss = criterion(logits, target)
    return loss, logits, moe_weights


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    moe_entropy_weight: float = 0.0):
    """Train for one epoch with optional MoE entropy regularization.

    Args:
        moe_entropy_weight: Weight for dispatch entropy loss (0=disabled).
            Helps maintain routing diversity during finetuning.
            Recommended: 1.0-2.5 (pretraining typically uses 5.0).
    """
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Check if MoE entropy loss should be used
    use_moe_entropy = moe_entropy_weight > 0 and MOE_ENTROPY_AVAILABLE
    if moe_entropy_weight > 0 and not MOE_ENTROPY_AVAILABLE:
        print("Warning: moe_entropy_weight > 0 but MoE entropy loss not available")

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass with optional MoE weight collection
        entropy_loss_value = 0.0
        if loss_scaler is None:
            samples = samples.half()
            loss, output, moe_weights = train_class_batch(
                model, samples, targets, criterion, return_moe_weights=use_moe_entropy)
        else:
            with torch.amp.autocast('cuda'):
                loss, output, moe_weights = train_class_batch(
                    model, samples, targets, criterion, return_moe_weights=use_moe_entropy)

        # Add MoE entropy loss if enabled (maintains routing diversity during finetuning)
        # moe_weights is a list of dicts (one per MoE block)
        if use_moe_entropy and moe_weights is not None and len(moe_weights) > 0:
            try:
                # Compute entropy loss for each MoE block and average
                block_entropy_losses = []
                for block_weights in moe_weights:
                    block_entropy_loss, _ = compute_dispatch_entropy_loss(
                        block_weights,
                        weight_type="dispatch",
                        aggregation="separate"
                    )
                    block_entropy_losses.append(block_entropy_loss)

                # Average across all MoE blocks
                entropy_loss = torch.stack(block_entropy_losses).mean()
                loss = loss + moe_entropy_weight * entropy_loss
                entropy_loss_value = entropy_loss.item()
            except Exception as e:
                # Silently handle if entropy computation fails (e.g., wrong weight format)
                if epoch == 0 and it == 0:
                    print(f"Warning: MoE entropy loss computation failed: {e}")

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        if use_moe_entropy:
            metric_logger.update(moe_entropy_loss=entropy_loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            # Log training metrics with proper namespacing
            log_writer.update(train_loss=loss_value, step=it)
            if class_acc is not None:
                log_writer.update(train_acc=class_acc, step=it)

            # Log MoE entropy loss if enabled
            if use_moe_entropy:
                log_writer.update(moe_entropy_loss=entropy_loss_value, head="moe", step=it)

            # Log optimization metrics
            log_writer.update(loss_scale=loss_scale_value, head="opt", step=it)
            log_writer.update(lr=max_lr, head="opt", step=it)
            log_writer.update(min_lr=min_lr, head="opt", step=it)
            log_writer.update(weight_decay=weight_decay_value, head="opt", step=it)
            log_writer.update(grad_norm=grad_norm, head="opt", step=it)

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
