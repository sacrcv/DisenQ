#!/usr/bin/env python
"""Train DisenQ.

Single GPU::

    python train.py --cfg configs/ntu_rgb_ab.yaml

Multi GPU::

    torchrun --nproc_per_node=4 train.py --cfg configs/ntu_rgb_ab.yaml

Any config value can be overridden on the command line::

    python train.py --cfg configs/ntu_rgb_ab.yaml \
        run.max_epoch=10 run.batch_size=16 loss.orth_weight=0.1
"""

import argparse
import os
import sys

import torch

from disenq.data import build_datasets, build_eval_loader, build_train_loader
from disenq.engine import Trainer
from disenq.models import DisenQModel
from disenq.utils import (
    config_to_str,
    count_parameters,
    destroy_distributed,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    load_config,
    save_config,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DisenQ")
    parser.add_argument("--cfg", required=True, help="path to a YAML config")
    parser.add_argument("--output", default=None, help="override run.output_dir")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "overrides", nargs="*", help="config overrides as key=value (dotted keys)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg, args.overrides)

    distributed, rank, world_size, local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output or cfg.run.get("output_dir", "output/disenq")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "train.log") if is_main_process() else None)
    logger.info(f"output dir: {output_dir}")
    logger.info(f"world size: {world_size}  device: {device}")
    logger.info("config:\n" + config_to_str(cfg))
    if is_main_process():
        save_config(cfg, os.path.join(output_dir, "config.yaml"))

    set_seed(cfg.run.get("seed", 42), rank=rank, deterministic=cfg.run.get("deterministic", False))

    datasets, meta = build_datasets(cfg)
    if "train" not in datasets:
        logger.error("the annotation file has no 'train' split")
        return 1
    logger.info(
        f"label space: {meta['num_identities']} train identities, "
        f"{meta['num_actions']} actions (multilabel={meta['multilabel']})"
    )

    # Charades-AB style datasets need the multi-label action objective.
    if meta["multilabel"] and cfg.get("loss", {}).get("action_mode", "ce") != "bce":
        logger.warning("dataset is multi-label; forcing loss.action_mode=bce")
        cfg.loss.action_mode = "bce"

    model = DisenQModel.from_config(
        cfg, num_identities=meta["num_identities"], num_actions=meta["num_actions"]
    ).to(device)
    params = count_parameters(model)
    logger.info(
        f"parameters: {params['total'] / 1e6:.1f}M total, "
        f"{params['trainable'] / 1e6:.1f}M trainable, {params['frozen'] / 1e6:.1f}M frozen"
    )

    train_loader = build_train_loader(cfg, datasets["train"])
    query_loader = build_eval_loader(cfg, datasets["query"]) if "query" in datasets else None
    gallery_loader = (
        build_eval_loader(cfg, datasets["gallery"]) if "gallery" in datasets else None
    )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=cfg.run.get("find_unused_parameters", True),
        )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        device=device,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        output_dir=output_dir,
    )

    resume = args.resume or cfg.run.get("resume", None)
    if resume:
        trainer.resume(resume)

    trainer.run()
    destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
