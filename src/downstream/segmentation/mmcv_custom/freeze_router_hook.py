"""
N→R (Navigate then Reach) Hook for MMSegmentation.

Freezes MoE router parameters (phi, scale) at a specified iteration,
implementing the two-phase N→R finetuning schedule:
  Phase 1 (iter 0 to transition_iter): Router trainable at reduced LR
  Phase 2 (transition_iter to end): Router frozen

Reference: "Training MoE with Proper Guidance" (EMNLP 2025)
"""

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeRouterHook(Hook):
    """Freeze MoE router parameters at a specified training iteration.

    Args:
        transition_iter (int): Iteration at which to freeze router params.
    """

    def __init__(self, transition_iter):
        self.transition_iter = transition_iter
        self.frozen = False

    def after_train_iter(self, runner):
        if not self.frozen and runner.iter >= self.transition_iter:
            frozen_count = 0
            frozen_params = []
            for name, param in runner.model.named_parameters():
                if '.phi' in name or '.scale' in name:
                    param.requires_grad = False
                    frozen_count += 1
                    frozen_params.append(name)

            runner.logger.info(f"\n{'='*60}")
            runner.logger.info(
                f"N→R Phase 2: Frozen {frozen_count} router params at iter {runner.iter}")
            for p in frozen_params[:6]:
                runner.logger.info(f"  - {p}")
            if len(frozen_params) > 6:
                runner.logger.info(f"  ... and {len(frozen_params) - 6} more")
            runner.logger.info(f"{'='*60}\n")
            self.frozen = True
