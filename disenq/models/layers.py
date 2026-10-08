"""Minimal layer helpers that the vendored EVA ViT needs.

Upstream LAVIS imports ``drop_path``, ``to_2tuple`` and ``trunc_normal_`` from
``timm.models.layers``, and the checkpoint cache from ``timm.models.hub``. Both
of those module paths were removed in modern timm, so depending on them would
pin this repo to ``timm==0.4.12`` forever. They are a few lines each, so they
are reproduced here instead and timm is not a dependency at all.

``drop_path`` follows the standard stochastic-depth formulation from
`timm <https://github.com/huggingface/pytorch-image-models>`_ (Apache-2.0).
"""

import collections.abc
from itertools import repeat

import torch


def to_ntuple(n):
    """Return a function that broadcasts a scalar into an ``n``-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                return tuple(repeat(x[0], n))
            if len(x) != n:
                raise ValueError(f"expected 1 or {n} values, got {len(x)}")
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = to_ntuple(2)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth: drop whole residual branches per sample.

    Shapes the mask as ``[B, 1, 1, ...]`` so it applies per sample regardless of
    tensor rank, and rescales the survivors so the expected value is preserved.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)
    return x * mask


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal initialisation (thin wrapper over torch's own)."""
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
