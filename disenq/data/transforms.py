"""Clip-level transforms.

Augmentation decisions are made once per clip, not per frame, so temporal
consistency is preserved. Following the paper: "Each frame is resized to
224x224 and horizontal flipping is used for data augmentation, following
[18, 3]."
"""

import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF

#: CLIP / EVA-CLIP normalisation, which is what the BLIP-2 ViT was trained with.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipTransform:
    """Resize, augment and normalise a clip tensor ``[T, 3, H, W]`` in ``[0, 1]``.

    Args:
        image_size: output side length (square).
        train: enables the stochastic augmentations.
        horizontal_flip: probability of flipping the whole clip.
        random_crop_pad: if > 0, pad by this many pixels and take a random crop
            of ``image_size`` (the standard re-ID 10-pixel pad-and-crop). Set to
            0 for the paper's flip-only recipe.
        random_erasing: probability of erasing one rectangle, applied at the
            same location in every frame.
        mean / std: channel normalisation statistics.
    """

    def __init__(
        self,
        image_size: int = 224,
        train: bool = False,
        horizontal_flip: float = 0.5,
        random_crop_pad: int = 0,
        random_erasing: float = 0.0,
        mean: Sequence[float] = CLIP_MEAN,
        std: Sequence[float] = CLIP_STD,
    ):
        self.image_size = image_size
        self.train = train
        self.horizontal_flip = horizontal_flip
        self.random_crop_pad = random_crop_pad
        self.random_erasing = random_erasing
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dtype != torch.float32:
            clip = clip.float()
        if clip.max() > 1.5:  # came in as uint8-valued floats
            clip = clip / 255.0

        size = self.image_size
        if self.train and self.random_crop_pad > 0:
            padded = size + 2 * self.random_crop_pad
            clip = TF.resize(clip, [padded, padded], antialias=True)
            top = random.randint(0, padded - size)
            left = random.randint(0, padded - size)
            clip = TF.crop(clip, top, left, size, size)
        else:
            clip = TF.resize(clip, [size, size], antialias=True)

        if self.train and random.random() < self.horizontal_flip:
            clip = TF.hflip(clip)

        clip = TF.normalize(clip, mean=self.mean, std=self.std)

        if self.train and self.random_erasing > 0 and random.random() < self.random_erasing:
            clip = self._erase(clip)
        return clip

    def _erase(self, clip: torch.Tensor) -> torch.Tensor:
        """Erase one rectangle at the same position across all frames."""
        _, _, height, width = clip.shape
        for _ in range(10):
            area = height * width * random.uniform(0.02, 0.2)
            ratio = random.uniform(0.3, 3.3)
            erase_h = int(round((area * ratio) ** 0.5))
            erase_w = int(round((area / ratio) ** 0.5))
            if erase_h < height and erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                clip[:, :, top : top + erase_h, left : left + erase_w] = torch.randn(
                    clip.shape[0], clip.shape[1], erase_h, erase_w
                )
                return clip
        return clip


def build_transform(cfg, train: bool) -> ClipTransform:
    """Build a :class:`ClipTransform` from the ``data`` config block."""
    aug = cfg.get("augmentation", {}) or {}
    return ClipTransform(
        image_size=cfg.get("image_size", 224),
        train=train,
        horizontal_flip=aug.get("horizontal_flip", 0.5),
        random_crop_pad=aug.get("random_crop_pad", 0),
        random_erasing=aug.get("random_erasing", 0.0),
    )
