"""Small shared helpers: seeding, checkpoint IO, parameter counting."""

import os
import random
from typing import Any, Dict

import numpy as np
import torch

from disenq.utils.dist import is_main_process
from disenq.utils.logger import get_logger


def set_seed(seed: int, rank: int = 0, deterministic: bool = False) -> None:
    """Seed every RNG. Each rank gets a distinct stream via ``seed + rank``."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base_seed: int = 0) -> None:
    seed = base_seed + worker_id
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, best: Any = None, cfg=None) -> None:
    """Save trainable weights plus optimiser state (rank 0 only).

    Frozen submodules (ViT / text encoder) are excluded so checkpoints stay
    small; they are rebuilt from their own pretrained sources on load.
    """
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    net = unwrap_model(model)

    trainable_names = {n for n, p in net.named_parameters() if p.requires_grad}
    state_dict = {}
    for k, v in net.state_dict().items():
        # Keep buffers (e.g. BNNeck running stats) for modules that are trained.
        if k in trainable_names:
            state_dict[k] = v
        elif not (k.startswith("visual_encoder.") or k.startswith("text_encoder.")):
            state_dict[k] = v

    payload = {
        "model": state_dict,
        "epoch": epoch,
        "best": best,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if cfg is not None:
        from omegaconf import OmegaConf

        payload["config"] = OmegaConf.to_container(cfg, resolve=True)

    torch.save(payload, path)
    get_logger().info(f"saved checkpoint -> {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    strict: bool = False, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    net = unwrap_model(model)
    logger = get_logger()

    # ``strict=False`` tolerates missing/unexpected keys but still raises on a
    # shape mismatch. That bites whenever the checkpoint's label space differs
    # from the current dataset's -- notably the cross-domain evaluation of
    # Appendix C.2 (train on NTU RGB-AB, evaluate on Charades-AB), where the
    # identity and action classifiers are sized differently. Retrieval only
    # needs the backbone and the BNNeck statistics, never the classifiers, so
    # drop the incompatible tensors and carry on.
    if not strict:
        current = net.state_dict()
        mismatched = [
            k for k, v in state_dict.items()
            if k in current and hasattr(v, "shape") and v.shape != current[k].shape
        ]
        if mismatched:
            state_dict = {k: v for k, v in state_dict.items() if k not in mismatched}
            logger.warning(
                f"dropped {len(mismatched)} tensor(s) whose shape differs from the "
                f"current model (label space changed): {mismatched[:6]}"
                + (" ..." if len(mismatched) > 6 else "")
            )

    msg = net.load_state_dict(state_dict, strict=strict)
    missing = [k for k in msg.missing_keys
               if not (k.startswith("visual_encoder.") or k.startswith("text_encoder."))]
    if missing:
        logger.warning(f"missing keys when loading {path}: {len(missing)} (first: {missing[:5]})")
    if msg.unexpected_keys:
        logger.warning(
            f"unexpected keys when loading {path}: {len(msg.unexpected_keys)} "
            f"(first: {msg.unexpected_keys[:5]})"
        )
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    logger.info(f"loaded checkpoint {path} (epoch {ckpt.get('epoch')})")
    return ckpt
