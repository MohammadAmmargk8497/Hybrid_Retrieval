"""Embedder interface + Nomic Embed v1.5 implementation.

Why Nomic v1.5
--------------
- Matryoshka: 768d native, but the first N dims (256/512/...) remain
  cosine-meaningful. Truncating to 512 buys ~2× search speed at <1 nDCG
  point cost on most retrieval benchmarks. We default to 512.
- 8 k context — enough for our paragraph-scale chunks plus a future page-
  scale embedding without re-architecting.
- Trained with explicit task prefixes:
      "search_document: <text>"   # for indexed text
      "search_query: <text>"      # for queries
  Using them is **not** optional — without prefixes you get a generic
  embedding that materially under-performs on retrieval.

Caching
-------
Query embeddings hit a per-instance LRU (default 1024 entries). Document
embeddings are not LRU'd in-process — they're already cached implicitly
in Qdrant once upserted. The model itself is loaded **once** per process
on first instantiation (~3 s on M-series via MPS).

Interface
---------
``Embedder`` is a Protocol so tests can pass a fake without pulling in
torch / sentence-transformers (~700 MB of deps).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Minimal embedder contract used by the rest of the system."""

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Real Nomic-backed embedder
# ---------------------------------------------------------------------------

class NomicEmbedder:
    """Nomic Embed v1.5 with prefix tokens, MPS, and an LRU on queries.

    Heavy imports (torch, sentence-transformers) are deferred to
    ``__init__`` so importing this module is cheap. Tests that don't
    actually instantiate the class won't pay the import cost.
    """

    DOC_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        dim: int = 512,
        cache_size: int = 1024,
        device: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info("Loading %s on %s (truncate_dim=%d)", model_name, device, dim)
        # ``truncate_dim`` makes ``encode`` return 512-d vectors directly
        # (post-Matryoshka). Re-normalisation is handled internally.
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            truncate_dim=dim,
        )
        self.dim = dim
        self.model_name = model_name

        # Closure-bound LRU so the cache is per-instance (lru_cache on
        # methods leaks references to ``self`` if applied at class level).
        @lru_cache(maxsize=cache_size)
        def _q_cache(text: str) -> tuple[float, ...]:
            v = self._encode_query_raw(text)
            return tuple(float(x) for x in v)

        self._q_cache = _q_cache

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.DOC_PREFIX + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _encode_query_raw(self, text: str) -> np.ndarray:
        v = self.model.encode(
            [self.QUERY_PREFIX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._q_cache(text), dtype=np.float32)

    def cache_info(self):
        """Expose underlying ``functools.lru_cache`` stats — for the CLI's
        ``info`` command and future telemetry."""
        return self._q_cache.cache_info()
