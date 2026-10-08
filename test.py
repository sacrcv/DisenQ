#!/usr/bin/env python
"""Evaluate a trained DisenQ checkpoint.

    python test.py --cfg configs/ntu_rgb_ab.yaml --checkpoint output/disenq/checkpoint_best.pth

Ablations from Table 3 (bottom) -- the discriminative power of each
disentangled feature on its own:

    python test.py --cfg ... --checkpoint ... --feature biometrics
    python test.py --cfg ... --checkpoint ... --feature motion
    python test.py --cfg ... --checkpoint ... --feature non_biometrics

Fixed instead of adaptive weighting (the alpha_1 = alpha_2 = 0.5 baseline):

    python test.py --cfg ... --checkpoint ... --no-adaptive
"""

import argparse
import json
import os
import sys

import torch

from disenq.data import build_datasets, build_eval_loader
from disenq.engine import resolve_amp_dtype
from disenq.evaluation import PROTOCOLS, evaluate
from disenq.models import DisenQModel
from disenq.utils import (
    config_to_str,
    destroy_distributed,
    init_distributed_mode,
    is_main_process,
    load_checkpoint,
    load_config,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DisenQ")
    parser.add_argument("--cfg", required=True, help="path to a YAML config")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to evaluate")
    parser.add_argument("--output", default=None, help="where to write results.json")
    parser.add_argument(
        "--feature", default="fused",
        choices=["fused", "biometrics", "motion", "non_biometrics"],
        help="which feature to retrieve with",
    )
    # Comma-separated rather than nargs="*", which would greedily swallow the
    # trailing positional config overrides.
    parser.add_argument(
        "--protocols", default=None,
        help="comma-separated subset of " + ",".join(sorted(PROTOCOLS))
             + "; defaults to run.protocols",
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="use fixed alpha_1 = alpha_2 = 0.5 instead of the MLP",
    )
    parser.add_argument(
        "--no-motion", action="store_true", help="retrieve on biometrics only"
    )
    parser.add_argument("overrides", nargs="*", help="config overrides as key=value")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg, args.overrides)

    _, rank, world_size, local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output or cfg.run.get("output_dir", "output/disenq")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "test.log") if is_main_process() else None)
    logger.info(f"world size: {world_size}  device: {device}")
    logger.info("config:\n" + config_to_str(cfg))

    set_seed(cfg.run.get("seed", 42), rank=rank)

    datasets, meta = build_datasets(cfg)
    if "query" not in datasets or "gallery" not in datasets:
        logger.error("the annotation file needs both a 'query' and a 'gallery' split")
        return 1

    model = DisenQModel.from_config(
        cfg, num_identities=meta["num_identities"], num_actions=meta["num_actions"]
    ).to(device)

    checkpoint = args.checkpoint or cfg.run.get("checkpoint", None)
    if checkpoint:
        load_checkpoint(checkpoint, model, strict=False)
    else:
        logger.warning(
            "no --checkpoint given: evaluating the freshly initialised model, "
            "which only sanity-checks the pipeline"
        )

    query_loader = build_eval_loader(cfg, datasets["query"])
    gallery_loader = build_eval_loader(cfg, datasets["gallery"])

    protocols = (
        [p.strip() for p in args.protocols.split(",") if p.strip()]
        if args.protocols
        else list(cfg.run.get("protocols", ["same_activity"]))
    )
    results = evaluate(
        model,
        query_loader,
        gallery_loader,
        device,
        protocols=protocols,
        amp_dtype=resolve_amp_dtype(cfg.run.get("amp", "bf16")),
        use_motion=not args.no_motion and cfg.run.get("use_motion", True),
        adaptive=not args.no_adaptive and cfg.run.get("adaptive_similarity", True),
        fixed_alpha=cfg.run.get("fixed_alpha", 0.5),
        max_rank=cfg.run.get("max_rank", 50),
        print_freq=cfg.run.get("print_freq", 20),
        feature=args.feature,
    )

    if is_main_process():
        payload = {
            "dataset": meta["name"],
            "checkpoint": checkpoint,
            "feature": args.feature,
            "adaptive": not args.no_adaptive,
            "use_motion": not args.no_motion,
            "results": results,
        }
        suffix = args.feature if args.feature != "fused" else "fused"
        path = os.path.join(output_dir, f"results_{suffix}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {path}")

    destroy_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
