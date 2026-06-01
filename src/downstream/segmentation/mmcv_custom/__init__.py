# --------------------------------------------------------
# MEDiC: Masked Encoder-Decoder with soft Mixture-of-Experts
# Custom MMCV utilities
# --------------------------------------------------------

import torch

# Fix PyTorch 2.6+ weights_only default change
# mmcv's checkpoint loading doesn't pass weights_only, causing failures
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor
# Import (and thus register) the MoE-aware constructor with mmseg's OPTIMIZER_BUILDERS.
# Without this, mmseg's homonymous LayerDecayOptimizerConstructor wins over our custom
# one and crashes with NotImplementedError on MoE params.
from .moe_layer_decay_optimizer_constructor import MoELayerDecayOptimizerConstructor

__all__ = [
    'load_checkpoint',
    'LayerDecayOptimizerConstructor',
    'MoELayerDecayOptimizerConstructor',
]
