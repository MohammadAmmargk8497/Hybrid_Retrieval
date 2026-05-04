"""Tests for the Pydantic data contract — see docs/changes/0008."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import Chunk, ChunkMetadata, SearchHit, SearchResponse


def _meta(**overrides) -> ChunkMetadata:
    base = dict(source="x.pdf", doc_id="x", page_start=1, page_end=1, chunk_index=0)
    base.update(overrides)
    return ChunkMetadata(**base)


def test_chunk_roundtrips_through_json():
    c = Chunk(id="x_p1_c0", text="hello", metadata=_meta())
    payload = c.model_dump_json()
    again = Chunk.model_validate_json(payload)
    assert again == c


def test_chunk_metadata_rejects_zero_page():
    with pytest.raises(ValidationError):
        _meta(page_start=0)


def test_chunk_metadata_rejects_unknown_field():
    """Frozen + extra='forbid' means typos fail loud at the boundary."""
    with pytest.raises(ValidationError):
        ChunkMetadata(  # type: ignore[call-arg]
            source="x.pdf",
            doc_id="x",
            page_start=1,
            page_end=1,
            chunk_index=0,
            sectionn="oops",  # typo
        )


def test_chunk_text_must_be_nonempty():
    with pytest.raises(ValidationError):
        Chunk(id="x", text="", metadata=_meta())


def test_chunk_metadata_dump_excludes_none_section():
    """Chroma payloads can't carry None — model_dump(exclude_none=True) must drop it."""
    dumped = _meta().model_dump(exclude_none=True)
    assert "section" not in dumped


def test_search_response_shape():
    hit = SearchHit(
        chunk_id="x_p2_c0",
        doc_id="x",
        source="x.pdf",
        page=2,
        score=0.99,
        snippet="foo...",
        text="foo bar baz",
    )
    resp = SearchResponse(query="q", top_k=5, hits=[hit], took_ms=12.3)
    assert resp.hits[0].page == 2
    # JSON-roundtrip is what FastAPI / the CLI's --json mode will rely on.
    assert SearchResponse.model_validate_json(resp.model_dump_json()) == resp
