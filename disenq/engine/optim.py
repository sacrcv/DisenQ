"""Optimiser and learning-rate schedule.

The paper: "AdamW is used as the optimizer with weight decay of 5e-2 and base
learning rate of 1e-4 with beta values as [0.9, 0.999]."
"""

import math
from typing import List

import torch

from disenq.utils.logger import get_logger


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled for biases, norms and query tokens."""
    run_cfg = cfg.run
    weight_decay = run_cfg.get("weight_decay", 5e-2)
    base_lr = run_cfg.get("lr", 1e-4)
    betas = tuple(run_cfg.get("betas", [0.9, 0.999]))
    # Optional separate (usually larger) LR for the randomly initialised parts.
    head_lr_mult = run_cfg.get("head_lr_mult", 1.0)

    head_prefixes = (
        "id_head.", "action_head.", "non_biometrics_norm.",
        "adaptive_similarity.", "temporal_pool.", "temporal_pos.",
    )

    groups = {
        "decay": {"params": [], "weight_decay": weight_decay, "lr": base_lr},
        "no_decay": {"params": [], "weight_decay": 0.0, "lr": base_lr},
        "head_decay": {"params": [], "weight_decay": weight_decay, "lr": base_lr * head_lr_mult},
        "head_no_decay": {"params": [], "weight_decay": 0.0, "lr": base_lr * head_lr_mult},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith(head_prefixes)
        # 1-D tensors are norms/biases; query tokens are embeddings-like.
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "query_tokens" in name
        key = ("head_" if is_head else "") + ("no_decay" if no_decay else "decay")
        groups[key]["params"].append(param)

    param_groups = [g for g in groups.values() if g["params"]]
    get_logger().info(
        "optimizer groups: "
        + ", ".join(
            f"{name}={len(g['params'])} tensors (lr={g['lr']:.2e}, wd={g['weight_decay']})"
            for name, g in groups.items()
            if g["params"]
        )
    )

    return torch.optim.AdamW(param_groups, lr=base_lr, betas=betas)


class WarmupCosineSchedule:
    """Linear warmup then cosine decay, stepped once per iteration.

    Each parameter group decays from its own initial LR, so the ``head_lr_mult``
    ratio is preserved throughout training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        warmup_start_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(min(warmup_steps, self.total_steps - 1), 0)
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_ratio = warmup_start_ratio
        self.base_lrs: List[float] = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0
        self._apply()

    def _ratio(self) -> float:
        step = self.step_count
        if step < self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            return self.warmup_start_ratio + (1.0 - self.warmup_start_ratio) * progress
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self) -> None:
        ratio = self._ratio()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * ratio

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]
        self._apply()


def build_scheduler(cfg, optimizer, steps_per_epoch: int) -> WarmupCosineSchedule:
    run_cfg = cfg.run
    max_epoch = run_cfg.get("max_epoch", 60)
    warmup_epochs = run_cfg.get("warmup_epochs", 1)
    accum = max(run_cfg.get("accum_grad_iters", 1), 1)
    steps_per_epoch = max(steps_per_epoch // accum, 1)
    return WarmupCosineSchedule(
        optimizer,
        total_steps=max_epoch * steps_per_epoch,
        warmup_steps=int(warmup_epochs * steps_per_epoch),
        min_lr_ratio=run_cfg.get("min_lr_ratio", 0.01),
        warmup_start_ratio=run_cfg.get("warmup_start_ratio", 0.01),
    )
