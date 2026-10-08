"""Training signal for the adaptive similarity MLP (Eq. 8).

The paper describes the MLP architecture and how it is used at retrieval time
but does not state how it is optimised. We train it on the in-batch
probe/gallery grid: every pair of clips in the batch is a probe-gallery pair
whose ground truth ("same identity or not") is known, so the MLP learns when
motion similarity helps and when it should defer to biometrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseSimilarityLoss(nn.Module):
    """Supervise the fused similarity on in-batch pairs.

    ``mode="ranking"`` (default)
        Batch-hard soft-margin ranking: ``softplus(s_hardest_neg -
        s_hardest_pos)`` per anchor. Scale free and directly aligned with the
        rank-1 metric.

    ``mode="bce"``
        Binary cross entropy on ``scale * similarity`` against the pair label.
    """

    def __init__(self, mode: str = "ranking", scale: float = 10.0):
        super().__init__()
        if mode not in ("ranking", "bce"):
            raise ValueError(f"similarity loss mode must be 'ranking' or 'bce', got '{mode}'")
        self.mode = mode
        self.scale = scale

    def forward(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarity: ``[B, B]`` fused similarity matrix for the batch.
            labels: ``[B]`` identity labels.
        """
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        if self.mode == "bce":
            keep = ~eye
            if not keep.any():
                return similarity.sum() * 0.0
            logits = self.scale * similarity[keep]
            target = positive_mask[keep].to(logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, target)

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return similarity.sum() * 0.0

        min_val = torch.finfo(similarity.dtype).min
        max_val = torch.finfo(similarity.dtype).max
        # Hardest positive = least similar same-identity pair.
        hardest_pos = similarity.masked_fill(~positive_mask, max_val).min(dim=1).values
        # Hardest negative = most similar different-identity pair.
        hardest_neg = similarity.masked_fill(~negative_mask, min_val).max(dim=1).values

        return F.softplus(hardest_neg[valid] - hardest_pos[valid]).mean()
