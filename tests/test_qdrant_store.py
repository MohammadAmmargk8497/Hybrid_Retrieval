"""Tests for the Qdrant helpers — see docs/changes/0010."""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from src.models import Chunk, ChunkMetadata  # noqa: E402
from src.qdrant_store import (  # noqa: E402
    chunk_id_to_point_id,
    ensure_collection,
    get_qdrant_client,
    scroll_all_chunks,
    upsert_chunks,
)
from tests._fakes import FakeEmbedder  # noqa: E402


def _chunk(cid: str, text: str, **meta) -> Chunk:
    base = dict(source="x.pdf", doc_id="x", page_start=1, page_end=1, chunk_index=0)
    base.update(meta)
    return Chunk(id=cid, text=text, metadata=ChunkMetadata(**base))


def test_chunk_id_to_point_id_is_deterministic():
    a = chunk_id_to_point_id("paper_p3_c5")
    b = chunk_id_to_point_id("paper_p3_c5")
    assert a == b
    assert chunk_id_to_point_id("paper_p3_c6") != a


def test_upsert_and_scroll_roundtrip():
    embedder = FakeEmbedder()
    client = get_qdrant_client(":memory:")
    ensure_collection(client, dim=embedder.dim)

    chunks = [
        _chunk("a_p1_c0", "transformer attention"),
        _chunk("b_p2_c0", "rlhf human feedback", page_start=2, page_end=2),
    ]
    upsert_chunks(client, chunks, embedder)

    payloads = list(scroll_all_chunks(client))
    assert len(payloads) == 2
    by_id = {p["chunk_id"]: p for p in payloads}
    assert "a_p1_c0" in by_id and "b_p2_c0" in by_id
    assert by_id["a_p1_c0"]["text"] == "transformer attention"
    assert by_id["b_p2_c0"]["page_start"] == 2


def test_ensure_collection_is_idempotent():
    """Calling twice mustn't raise — important for `process` reruns."""
    client = get_qdrant_client(":memory:")
    ensure_collection(client, dim=8)
    ensure_collection(client, dim=8)


def test_upsert_overwrites_same_chunk_id():
    """Re-extracting the same chunk produces the same UUID5 → upsert."""
    embedder = FakeEmbedder()
    client = get_qdrant_client(":memory:")
    ensure_collection(client, dim=embedder.dim)

    upsert_chunks(client, [_chunk("k_p1_c0", "first version")], embedder)
    upsert_chunks(client, [_chunk("k_p1_c0", "second version")], embedder)

    payloads = list(scroll_all_chunks(client))
    assert len(payloads) == 1
    assert payloads[0]["text"] == "second version"
