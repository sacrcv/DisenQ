"""Temporal position embedding and temporal attention pooling (Section 3.1).

From the paper: "each visual feature f_i has temporal ordering information
associated with it through a position embedding layer. Finally, temporal
attention pooling is applied on all frame features to get a global video-level
feature F."
"""

import torch
import torch.nn as nn


class TemporalPositionEmbedding(nn.Module):
    """Learned per-frame embedding broadcast over the spatial token axis.

    Input/output shape: ``[B, T, N, D]``. Zero-initialised so the frozen ViT
    features are unperturbed at step 0.
    """

    def __init__(self, max_frames: int, dim: int):
        super().__init__()
        self.max_frames = max_frames
        self.embedding = nn.Embedding(max_frames, dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_frames, _, _ = x.shape
        if num_frames > self.max_frames:
            raise ValueError(
                f"got {num_frames} frames but temporal position embedding was built "
                f"for at most {self.max_frames}; raise model.max_frames"
            )
        idx = torch.arange(num_frames, device=x.device)
        pos = self.embedding(idx)              # [T, D]
        return x + pos[None, :, None, :]       # broadcast over batch and tokens


class TemporalAttentionPooling(nn.Module):
    """Pool ``[B, T, N, D]`` over the temporal axis into ``[B, N, D]``.

    A small MLP scores every (frame, token) position; scores are softmaxed over
    the temporal axis so each spatial token aggregates its own weighted average
    across frames. ``padding_mask`` marks frames to ignore (for clips shorter
    than ``T``).
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 64)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start close to uniform averaging.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, N, D]`` frame token features.
            padding_mask: ``[B, T]`` with 1 for valid frames, 0 for padding.

        Returns:
            ``[B, N, D]`` video-level token features.
        """
        logits = self.score(x)  # [B, T, N, 1]
        if padding_mask is not None:
            invalid = (padding_mask < 0.5)[:, :, None, None]
            logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1)
        return (weights * x).sum(dim=1)
