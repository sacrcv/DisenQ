#!/usr/bin/env python
"""Pre-compute frozen ViT features for every clip.

The visual encoder is frozen, so its output never changes during training.
Caching it turns each epoch into a Q-Former-only forward pass, which is the
single biggest speedup available for this model. The cache stores the
*post-``ln_vision``* tokens, which is exactly what ``DisenQModel.encode_video``
expects in its ``vit_features`` argument.

    python tools/extract_vit_features.py --cfg configs/ntu_rgb_ab.yaml \
        --out data/ntu_rgb_ab/vit_features

Then point the config at it::

    data:
      cached_vit_features_root: data/ntu_rgb_ab/vit_features

Be aware of the size: 257 tokens x 1408 dims x 2 bytes (fp16) is about 724KB
per frame, so cache only the frames you will sample.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disenq.data.build import build_datasets  # noqa: E402
from disenq.models.visual_encoder import build_visual_encoder  # noqa: E402
from disenq.utils import load_config, setup_logger  # noqa: E402


def main():
    args = parse_args()
    logger = setup_logger()
    cfg = load_config(args.cfg, args.overrides)
    # Make sure the datasets load pixels rather than an existing cache.
    cfg.data.cached_vit_features_root = None

    device = torch.device(args.device)
    encoder, ln_vision, width = build_visual_encoder(
        name=cfg.model.get("visual_encoder", "eva_clip_g"),
        image_size=cfg.model.get("image_size", 224),
        precision=cfg.model.get("vit_precision", "fp16"),
    )
    encoder = encoder.to(device).eval()
    ln_vision = ln_vision.to(device).eval()
    for param in list(encoder.parameters()) + list(ln_vision.parameters()):
        param.requires_grad = False

    if args.ln_vision_from:
        checkpoint = torch.load(args.ln_vision_from, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        ln_sd = {k[len("ln_vision.") :]: v for k, v in state.items()
                 if k.startswith("ln_vision.")}
        if ln_sd:
            ln_vision.load_state_dict(ln_sd)
            logger.info(f"loaded ln_vision from {args.ln_vision_from}")

    datasets, _ = build_datasets(cfg)
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    splits = (
        [s.strip() for s in args.splits.split(",") if s.strip()]
        if args.splits
        else list(datasets)
    )
    for split in splits:
        dataset = datasets.get(split)
        if dataset is None:
            logger.warning(f"split '{split}' not present; skipping")
            continue

        logger.info(f"extracting '{split}' ({len(dataset)} clips) -> {args.out}")
        for sample in tqdm(dataset.samples, desc=split):
            out_path = os.path.join(args.out, f"{sample['id']}.pt")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            total = dataset._num_frames(sample)
            if args.all_frames:
                indices = np.arange(total)
            else:
                # Cache a dense uniform grid so random training-time sampling
                # still has choices, rather than one fixed window.
                count = min(args.cache_frames, total)
                indices = np.linspace(0, total - 1, count).round().astype(np.int64)

            if sample.get("frames_dir"):
                clip = dataset._load_frames_from_dir(sample, indices)
            else:
                clip = dataset._load_frames_from_video(sample, indices)
            clip = dataset.transform(clip).to(device)

            features = []
            for start in range(0, clip.shape[0], args.batch_size):
                block = clip[start : start + args.batch_size]
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"
                             and cfg.model.get("vit_precision", "fp16") == "fp16"),
                ):
                    tokens = encoder(block)
                features.append(ln_vision(tokens.float()).to(dtype).cpu())

            # Record which original frame each cached row came from so the
            # dataset can map training-time frame ids onto the cache.
            torch.save(
                {
                    "features": torch.cat(features, dim=0),
                    "frame_indices": indices.astype(np.int64),
                    "num_frames": int(total),
                },
                out_path,
            )

    logger.info(f"done -> {args.out} (token width {width}, dtype {args.dtype})")
    logger.info("set data.cached_vit_features_root to this path to use the cache")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out", required=True)
    # Comma-separated rather than nargs="*", which would greedily swallow the
    # trailing positional config overrides.
    parser.add_argument("--splits", default=None,
                        help="comma-separated splits to extract; defaults to all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-frames", type=int, default=32,
                        help="uniformly sampled frames to cache per clip")
    parser.add_argument("--all-frames", action="store_true",
                        help="cache every frame (large)")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ln-vision-from", default=None,
                        help="checkpoint to take the trained ln_vision weights from")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
