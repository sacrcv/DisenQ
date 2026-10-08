"""Orthogonality constraint between biometrics and non-biometrics (Eq. 6)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(nn.Module):
    r"""L_Orth = || F_b^T F_b_hat ||.

    Two readings of the equation are supported:

    ``mode="gram"`` (default, matches the written form)
        With ``F_b, F_b_hat`` as ``[B, D]`` matrices, ``F_b^T F_b_hat`` is the
        ``[D, D]`` cross-covariance; the loss is its Frobenius norm, divided by
        the batch size so the value does not scale with ``B``.

        Note what this measures: ``Gram[a, b] = sum_i F_b[i, a] *
        F_b_hat[i, b]``, i.e. the *batch-wise correlation* between every
        biometrics dimension and every non-biometrics dimension. It is zero
        only when the two feature spaces are linearly uncorrelated across the
        batch -- a strictly stronger condition than each sample's two vectors
        being perpendicular. That is the behaviour the paper wants: it forbids
        the non-biometrics branch from encoding anything predictable from the
        biometrics branch.

    ``mode="dot"``
        Treats each row as a vector and penalises the mean absolute per-sample
        dot product. Cheaper, and a strictly weaker constraint: per-sample
        perpendicularity still permits the two spaces to be correlated across
        the batch.

    Features are L2 normalised first so the penalty measures direction rather
    than magnitude, and so it cannot be trivially minimised by shrinking
    ``F_b_hat`` to zero.
    """

    def __init__(self, mode: str = "gram", normalize: bool = True):
        super().__init__()
        if mode not in ("gram", "dot"):
            raise ValueError(f"orthogonality mode must be 'gram' or 'dot', got '{mode}'")
        self.mode = mode
        self.normalize = normalize

    def forward(self, biometrics: torch.Tensor, non_biometrics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            biometrics: ``[B, D]`` ``F_b``.
            non_biometrics: ``[B, D]`` ``F_b_hat``.
        """
        if self.normalize:
            biometrics = F.normalize(biometrics, p=2, dim=-1)
            non_biometrics = F.normalize(non_biometrics, p=2, dim=-1)

        batch_size = biometrics.shape[0]
        if self.mode == "dot":
            return (biometrics * non_biometrics).sum(dim=-1).abs().mean()

        gram = biometrics.t() @ non_biometrics  # [D, D]
        return torch.linalg.matrix_norm(gram, ord="fro") / batch_size
