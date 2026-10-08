from disenq.losses.classification import ActionLoss, CrossEntropyLabelSmooth
from disenq.losses.orthogonality import OrthogonalityLoss
from disenq.losses.similarity import PairwiseSimilarityLoss
from disenq.losses.triplet import TripletLoss

__all__ = [
    "CrossEntropyLabelSmooth",
    "ActionLoss",
    "TripletLoss",
    "OrthogonalityLoss",
    "PairwiseSimilarityLoss",
]
