"""Storage for the structured textual descriptions produced by the frozen VLM.

Section 3.1: "To maintain consistency, biometrics descriptions are generated
only once per unique identity and reused in all subsequent videos of the same
actor by storing and iteratively refining it by updating the stored description
using a running average. This prevents major description drift and ensures
stable identity representation across varied activities and appearances."

A running average over raw strings is not well defined, so the refinement is
done in embedding space: the store keeps a running-mean sentence embedding per
identity and, whenever a new candidate description arrives, keeps whichever
observed description sits closest to that running mean. The effect matches the
stated intent -- one stable, drift-resistant biometrics description per actor --
while remaining a plain string that the frozen text encoder can consume.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

#: The three prompt categories, matching ``disenq.models.disenq_former.BRANCHES``.
PROMPT_KEYS = ("biometrics", "motion", "non_biometrics")

#: Fallbacks when a sample has no generated prompt (Section C.1 shows even
#: minimal slot-filler prompts help, so these keep the text path meaningful).
DEFAULT_PROMPTS = {
    "biometrics": "a person with a distinctive body shape and posture",
    "motion": "performing an everyday activity",
    "non_biometrics": "wearing everyday clothing and accessories",
}


class IdentityBiometricsStore:
    """Per-identity biometrics description with embedding-space running average."""

    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self._text: Dict[str, str] = {}
        self._mean: Dict[str, np.ndarray] = {}
        self._count: Dict[str, int] = {}
        self._candidates: Dict[str, List[str]] = {}

    def update(self, identity: str, text: str, embedding: Optional[np.ndarray] = None) -> str:
        """Add an observation for ``identity`` and return the stored description."""
        identity = str(identity)
        text = (text or "").strip()
        if not text:
            return self._text.get(identity, DEFAULT_PROMPTS["biometrics"])

        self._count[identity] = self._count.get(identity, 0) + 1
        self._candidates.setdefault(identity, [])
        if text not in self._candidates[identity]:
            self._candidates[identity].append(text)

        if embedding is None:
            # No embedding available: first description wins, as the paper's
            # "generated only once per unique identity" describes.
            self._text.setdefault(identity, text)
            return self._text[identity]

        embedding = np.asarray(embedding, dtype=np.float64)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        if identity not in self._mean:
            self._mean[identity] = embedding
            self._text[identity] = text
            self._emb_cache = getattr(self, "_emb_cache", {})
            self._emb_cache.setdefault(identity, {})[text] = embedding
            return text

        # Running average of the identity's biometrics embedding.
        self._mean[identity] = (
            self.momentum * self._mean[identity] + (1.0 - self.momentum) * embedding
        )
        self._mean[identity] /= np.linalg.norm(self._mean[identity]) + 1e-8

        self._emb_cache = getattr(self, "_emb_cache", {})
        self._emb_cache.setdefault(identity, {})[text] = embedding

        # Keep the observed description closest to the running mean.
        cache = self._emb_cache[identity]
        best_text, best_score = self._text[identity], -2.0
        for candidate, candidate_emb in cache.items():
            score = float(candidate_emb @ self._mean[identity])
            if score > best_score:
                best_text, best_score = candidate, score
        self._text[identity] = best_text
        return best_text

    def get(self, identity: str) -> Optional[str]:
        return self._text.get(str(identity))

    def to_dict(self) -> Dict[str, Dict]:
        return {
            identity: {
                "biometrics": text,
                "count": self._count.get(identity, 0),
                "candidates": self._candidates.get(identity, []),
            }
            for identity, text in self._text.items()
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict], momentum: float = 0.9):
        store = cls(momentum=momentum)
        for identity, entry in payload.items():
            store._text[identity] = entry["biometrics"]
            store._count[identity] = entry.get("count", 1)
            store._candidates[identity] = entry.get("candidates", [entry["biometrics"]])
        return store


class PromptBank:
    """Read-only lookup used by the dataset at training time.

    Resolution order for a sample:

    * ``motion`` / ``non_biometrics`` come from the sample's own entry (they
      describe that specific clip);
    * ``biometrics`` comes from the per-identity store, falling back to the
      sample's own entry, then to a generic default.
    """

    def __init__(
        self,
        samples: Optional[Dict[str, Dict[str, str]]] = None,
        identities: Optional[Dict[str, Dict]] = None,
    ):
        self.samples = samples or {}
        self.identities = identities or {}

    @classmethod
    def load(cls, path: Optional[str]) -> "PromptBank":
        if not path or not os.path.exists(path):
            return cls()
        with open(path) as f:
            payload = json.load(f)
        return cls(samples=payload.get("samples", {}), identities=payload.get("identities", {}))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"samples": self.samples, "identities": self.identities}, f, indent=2)

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, sample_id: str, identity) -> Dict[str, str]:
        entry = self.samples.get(str(sample_id), {})
        identity_entry = self.identities.get(str(identity), {})

        biometrics = (
            identity_entry.get("biometrics")
            or entry.get("biometrics")
            or DEFAULT_PROMPTS["biometrics"]
        )
        return {
            "biometrics": biometrics,
            "motion": entry.get("motion") or DEFAULT_PROMPTS["motion"],
            "non_biometrics": entry.get("non_biometrics") or DEFAULT_PROMPTS["non_biometrics"],
        }
