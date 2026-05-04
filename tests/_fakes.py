"""Test doubles shared across the suite."""

from __future__ import annotations

import numpy as np


class FakeEmbedder:
    """Keyword-overlap embedder for tests.

    Strings sharing any of the keywords below produce aligned vectors,
    so cosine similarity matches our intuition (FlashAttention paper is
    "near" a flashattention query). Anything not in the keyword set hits
    a small uniform bias so cosine is well-defined for empty overlaps.

    We avoid loading Nomic v1.5 (~250 MB) just to verify plumbing, but
    we still want the dense leg to behave realistically enough that
    integration tests can assert on rankings.
    """

    _KEYWORDS = (
        "attention", "flashattention", "transformer", "gpu", "memory", "io",
        "awareness", "rlhf", "human", "feedback", "language", "survey", "llm",
        "reasoning", "multilingual",
    )
    dim = len(_KEYWORDS)

    @classmethod
    def _vec(cls, text: str) -> np.ndarray:
        text_l = text.lower()
        v = np.array(
            [1.0 if kw in text_l else 0.0 for kw in cls._KEYWORDS],
            dtype=np.float32,
        )
        if v.sum() == 0:
            v = np.full(cls.dim, 0.01, dtype=np.float32)
        n = np.linalg.norm(v) or 1.0
        return v / n

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
