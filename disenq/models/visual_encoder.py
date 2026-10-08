"""Frame-level visual encoders.

The paper uses ViT-G/14 from EVA-CLIP (best of the three it ablates). SigLIP-L
and generic HuggingFace vision towers are also supported so the encoder
ablation in Table 5 (top) can be reproduced.
"""

from typing import Tuple

import torch
import torch.nn as nn

from disenq.models.eva_vit import create_eva_vit_g
from disenq.utils.logger import get_logger


class LayerNormFp32(nn.LayerNorm):
    """LayerNorm that always computes in fp32 and casts back.

    Needed because ``create_eva_vit_g(precision="fp16")`` emits half-precision
    activations while the Q-Former runs in fp32.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)


class HFVisionEncoder(nn.Module):
    """Wraps a HuggingFace vision tower (SigLIP, CLIP, DINOv2, ...).

    Exposes the same contract as the EVA ViT: ``forward(pixel_values) ->
    [B, num_tokens, num_features]`` and a ``num_features`` attribute.
    """

    def __init__(self, model_name: str, image_size: int = 224):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name)
        # Vision-language checkpoints nest the tower under ``vision_config``.
        if hasattr(config, "vision_config"):
            from transformers import AutoModel as _AutoModel

            full = _AutoModel.from_pretrained(model_name)
            self.backbone = getattr(full, "vision_model", full)
            self.num_features = config.vision_config.hidden_size
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.num_features = config.hidden_size
        self.image_size = image_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state


def build_visual_encoder(
    name: str = "eva_clip_g",
    image_size: int = 224,
    drop_path_rate: float = 0.0,
    use_grad_checkpoint: bool = False,
    precision: str = "fp16",
    vis_width: int = 1408,
) -> Tuple[nn.Module, nn.Module, int]:
    """Build a frame encoder and its output LayerNorm.

    Args:
        name: ``eva_clip_g`` for EVA-CLIP ViT-G/14 (paper default),
            ``hf:<model_id>`` for any HuggingFace vision tower (e.g.
            ``hf:google/siglip-large-patch16-224``), or ``none`` to skip the
            encoder entirely when training from pre-extracted features.
        vis_width: token width, only used when ``name="none"``.

    Returns:
        ``(encoder, ln_vision, width)``. ``encoder`` is ``None`` for
        ``name="none"``.
    """
    logger = get_logger()

    if name in ("none", None):
        # The visual encoder is frozen, so a cached-feature run never needs it.
        # Skipping it avoids loading 2GB of ViT weights.
        logger.info(
            f"visual encoder disabled (width={vis_width}); "
            f"pre-extracted features are required"
        )
        return None, LayerNormFp32(vis_width), vis_width

    if name == "eva_clip_g":
        encoder = create_eva_vit_g(
            img_size=image_size,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_grad_checkpoint,
            precision=precision,
        )
        width = encoder.num_features  # 1408
    elif name.startswith("hf:"):
        encoder = HFVisionEncoder(name[3:], image_size=image_size)
        width = encoder.num_features
    else:
        raise ValueError(
            f"unknown visual encoder '{name}'; expected 'eva_clip_g', "
            f"'hf:<model_id>' or 'none'"
        )

    ln_vision = LayerNormFp32(width)
    logger.info(f"built visual encoder '{name}' (width={width}, image_size={image_size})")
    return encoder, ln_vision, width
