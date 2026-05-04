"""Shared Pydantic data models.

These types are the contract between extraction, indexing, retrieval, the
CLI/GUI, and the upcoming FastAPI handlers. Keeping them in one module
means a refactor anywhere downstream is a typed change, not a guessing
game over dict keys.

Conventions
-----------
* All page numbers are **1-indexed** (matches what users see in PDF viewers
  and what `pypdf` returns when you ask for the human-facing page).
* `chunk_id` is globally unique and stable across reindexing as long as the
  source filename, page, and intra-page chunk index don't change.
* `doc_id` is a paper-level identifier — currently the source filename
  stem; will become the arXiv id once Darwin Research Agent integration
  lands (Stage 5).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Persisted alongside each chunk in the vector store and BM25 sidecar.

    Field set is deliberately flat (no nested models) because Chroma /
    Qdrant payload schemas prefer primitive-valued dicts. ``model_dump()``
    serialises straight into either backend.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(description="Source PDF filename (e.g. '2105.12723v4.pdf').")
    doc_id: str = Field(description="Paper-level id; usually the filename stem.")
    page_start: int = Field(ge=1, description="1-indexed page where this chunk starts.")
    page_end: int = Field(ge=1, description="1-indexed page where this chunk ends.")
    section: str | None = Field(
        default=None,
        description="Section title from structured extraction (GROBID, Stage 2). "
        "None when extracted via the PyPDF fallback.",
    )
    chunk_index: int = Field(ge=0, description="0-indexed position of this chunk within doc.")


class Chunk(BaseModel):
    """A retrievable text fragment plus the metadata needed to locate it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Globally unique chunk id.")
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class SearchHit(BaseModel):
    """One row of a search response.

    ``page`` is what the UI should open the PDF at — currently
    ``metadata.page_start`` of the matching chunk.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source: str
    page: int = Field(ge=1)
    score: float
    snippet: str
    text: str = Field(default="", repr=False, description="Full chunk text.")
    section: str | None = None


class SearchResponse(BaseModel):
    """Top-level response from ``/search`` (and the CLI `search` command)."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int
    hits: list[SearchHit]
    took_ms: float = Field(ge=0)
