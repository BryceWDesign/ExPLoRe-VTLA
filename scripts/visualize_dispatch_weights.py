#!/usr/bin/env python3
"""Reproduce paper Fig. 4.7: per-patch dispatch-weight heatmaps overlaid on images.

Loads a trained ExPLoRe checkpoint and a directory of input images, runs
inference, extracts dispatch weights at the loss block (default: block 11),
and saves a grid figure showing Expert 0 / Expert 1 routing patterns as
warm/cool heatmap overlays per the paper.

Usage:
    python scripts/visualize_dispatch_weights.py \\
        --checkpoint output/pretrain_explore_2exp/checkpoint-299.pth \\
        --config configs/pretrain_explore_2exp.yaml \\
        --image_dir assets/sample_images/ \\
        --output_dir output/fig_4_7/ \\
        --block_idx 11

For Fig. 4.7's 2-expert visualization, use the 2-exp config. For other expert
counts, just change --block_idx if needed (default 11 matches paper).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medic_model import build_medic_model  # noqa: E402


def load_images(image_dir, image_size=224):
    """Load all images from a directory, return raw (for viz) + normalized (for model)."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    crop = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    raw, norm, names = [], [], []
    for p in paths:
        t = crop(Image.open(p).convert('RGB'))
        raw.append(t)
        norm.append(normalize(t))
        names.append(p.stem)
    return torch.stack(raw), torch.stack(norm), names


def load_checkpoint(model, ckpt_path):
    """Load a training checkpoint into the model."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    # Strip "module." DDP prefix if present
    state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
             for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print(f"  missing: {len(msg.missing_keys)} keys (first: {msg.missing_keys[:3]})")
    if msg.unexpected_keys:
        print(f"  unexpected: {len(msg.unexpected_keys)} keys (first: {msg.unexpected_keys[:3]})")
    return model


def plot_overlay(raw, dispatch, block_idx, names, output_path):
    """Plot a (N_images, 1+E) grid: input + per-expert heatmap overlays.

    raw: (N, 3, H, W) unnormalized images
    dispatch: (N, P, E) dispatch weights per patch per expert (CLS stripped)
    """
    n_img, n_patch, n_exp = dispatch.shape
    grid_side = int(round(n_patch ** 0.5))
    img_size = raw.shape[-1]

    fig, axes = plt.subplots(n_img, 1 + n_exp,
                             figsize=(2.5 * (1 + n_exp), 2.5 * n_img))
    if n_img == 1:
        axes = axes[None, :]

    for i in range(n_img):
        # Input
        axes[i, 0].imshow(raw[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(names[i] if i == 0 else '', fontsize=9)
        axes[i, 0].set_ylabel(names[i], fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        # Per-expert dispatch heatmaps
        for e in range(n_exp):
            heat = dispatch[i, :, e].reshape(grid_side, grid_side).numpy()
            # Normalize per-image for visual contrast
            norm = Normalize(vmin=heat.min(), vmax=heat.max())
            axes[i, 1 + e].imshow(raw[i].permute(1, 2, 0).numpy(), alpha=0.4)
            axes[i, 1 + e].imshow(heat, alpha=0.6, cmap='RdYlBu_r',
                                  norm=norm, extent=(0, img_size, img_size, 0),
                                  interpolation='bilinear')
            if i == 0:
                axes[i, 1 + e].set_title(f'Expert {e}', fontsize=10)
            axes[i, 1 + e].set_xticks([]); axes[i, 1 + e].set_yticks([])

    fig.suptitle(f'Dispatch weights at block {block_idx}', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--output_dir', default='output/dispatch_viz')
    ap.add_argument('--block_idx', type=int, default=11,
                    help='Which MoE block to extract weights from (default: 11, the loss block)')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Building model from {args.config}")
    model = build_medic_model(cfg).to(args.device).eval()
    print(f"Loading checkpoint {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)

    print(f"Loading images from {args.image_dir}")
    raw, norm, names = load_images(args.image_dir)
    print(f"  {len(names)} images: {names}")

    with torch.no_grad():
        out = model(norm.to(args.device), mask=None, return_combine_weights=True)
        # 8-tuple: pred_tok, pred_pix, mask, combine_weights, ids_restore,
        #          combine_weights_dict, head_combine_weights, pixel_combine_weights
        cw_dict = out[5]

    if args.block_idx not in cw_dict:
        raise KeyError(f"Block {args.block_idx} not in MoE blocks {list(cw_dict.keys())}")

    dispatch = cw_dict[args.block_idx].cpu()  # (B, P, E)
    # Strip CLS token (paper Fig 4.7 shows patch dispatch only)
    if dispatch.shape[1] in (197, 198):  # 196 + CLS (+1 if other tokens)
        dispatch = dispatch[:, 1:1 + 196, :]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f'dispatch_block{args.block_idx}.png'
    plot_overlay(raw, dispatch, args.block_idx, names, out_path)


if __name__ == "__main__":
    main()
