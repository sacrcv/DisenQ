"""Identification head, action head and the adaptive similarity weighting MLP
(Section 3.3 of the paper)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeck(nn.Module):
    """BNNeck classification head, the standard re-ID setup used by [3, 19].

    The pre-BN feature feeds the triplet loss, the post-BN feature feeds the
    classifier (and is what retrieval uses at test time).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        cosine_classifier: bool = False,
        cosine_scale: float = 16.0,
    ):
        super().__init__()
        self.bottleneck = nn.BatchNorm1d(in_dim)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.zeros_(self.bottleneck.bias)
        # Classic BNNeck: the shift is frozen so the feature stays centred.
        self.bottleneck.bias.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cosine_classifier = cosine_classifier
        self.cosine_scale = cosine_scale

        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, in_dim]`` pooled feature.

        Returns:
            ``(logits [B, num_classes], bn_feature [B, in_dim])``.
        """
        feat = self.bottleneck(x)
        logits_in = self.dropout(feat)
        if self.cosine_classifier:
            logits = self.cosine_scale * F.linear(
                F.normalize(logits_in, dim=-1), F.normalize(self.classifier.weight, dim=-1)
            )
        else:
            logits = self.classifier(logits_in)
        return logits, feat


class AdaptiveSimilarity(nn.Module):
    """Adaptive weighting of biometrics vs. motion similarity (Eq. 8).

    "Instead of fixed weights, we use a lightweight MLP to dynamically adjust
    the contribution of biometrics and motion features based on their
    relevance. Given a probe identity A and gallery identity B, we compute
    cosine similarities for both biometrics and motion features, concatenate
    them, and pass them through the MLP with ReLU activations and a softmax
    function."

    The final layer is zero initialised, so training starts from the fixed
    ``alpha_1 = alpha_2 = 0.5`` baseline the paper compares against.
    """

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(max(num_layers - 1, 1)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 2)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, sim_biometrics: torch.Tensor, sim_motion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sim_biometrics: cosine similarities of any shape ``[...]``.
            sim_motion: same shape as ``sim_biometrics``.

        Returns:
            ``(similarity [...], alpha [..., 2])`` where ``alpha`` sums to 1
            along the last axis.
        """
        pair = torch.stack([sim_biometrics, sim_motion], dim=-1)
        flat = pair.reshape(-1, 2)
        alpha = self.mlp(flat).softmax(dim=-1).reshape(pair.shape)
        similarity = (alpha * pair).sum(dim=-1)
        return similarity, alpha
