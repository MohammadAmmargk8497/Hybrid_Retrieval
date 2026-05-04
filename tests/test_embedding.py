"""Tests for the embedder protocol behaviour.

We don't load Nomic v1.5 in tests — that would download ~250 MB of
weights. Instead, this file:
  * verifies our `FakeEmbedder` test double satisfies the `Embedder`
    protocol shape (so production callsites can swap them safely),
  * exercises the LRU cache pattern with a stub identical in shape to
    `NomicEmbedder` but using deterministic local vectors.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.embedding import Embedder
from tests._fakes import FakeEmbedder


def test_fake_embedder_matches_protocol():
    e: Embedder = FakeEmbedder()
    v = e.embed_query("attention")
    assert isinstance(v, np.ndarray)
    assert v.shape == (e.dim,)

    docs = e.embed_documents(["foo", "bar"])
    assert docs.shape == (2, e.dim)


def test_keyword_overlap_aligns_vectors():
    """Sanity check the heuristic: shared keywords → high cosine."""
    e = FakeEmbedder()
    qa = e.embed_query("flashattention GPU memory IO")
    paper_b = e.embed_documents(
        ["FlashAttention computes exact attention with IO-awareness on GPUs."]
    )[0]
    paper_d = e.embed_documents(
        ["A survey of LLMs across reasoning and multilingual tasks."]
    )[0]
    assert qa @ paper_b > qa @ paper_d


def test_lru_cache_pattern():
    """Mirror NomicEmbedder's per-instance closure-bound LRU."""
    calls = []

    class Stub:
        def __init__(self, cache_size: int = 4):
            @lru_cache(maxsize=cache_size)
            def _q(text: str) -> tuple[float, ...]:
                calls.append(text)
                return tuple(float(ord(c)) for c in text[:3])

            self._q = _q

        def embed_query(self, text: str) -> np.ndarray:
            return np.asarray(self._q(text), dtype=np.float32)

    s = Stub()
    s.embed_query("attention")
    s.embed_query("attention")
    s.embed_query("attention")
    assert calls == ["attention"]  # one underlying call

    s.embed_query("RLHF")
    assert calls == ["attention", "RLHF"]
    info = s._q.cache_info()
    assert info.hits == 2
    assert info.misses == 2
