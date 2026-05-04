"""End-to-end test for the bm25s + Qdrant + RRF pipeline.

Skipped if optional deps aren't installed. Uses an in-memory Qdrant
client and a tiny ``FakeEmbedder`` (see ``tests/_fakes.py``) so the
suite stays under ~5 seconds with no model downloads.
"""

from __future__ import annotations

import json
import shutil

import pytest

bm25s = pytest.importorskip("bm25s")
qdrant_client = pytest.importorskip("qdrant_client")

from src.models import Chunk, ChunkMetadata  # noqa: E402
from src.qdrant_store import (  # noqa: E402
    ensure_collection,
    get_qdrant_client,
    upsert_chunks,
)
from src.search import hybrid_search  # noqa: E402
from tests._fakes import FakeEmbedder  # noqa: E402


@pytest.fixture()
def populated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    import importlib

    import src.config as config_mod

    importlib.reload(config_mod)
    settings = config_mod.settings

    embedder = FakeEmbedder()
    client = get_qdrant_client(":memory:")
    ensure_collection(client, dim=embedder.dim)

    def _chunk(doc_id: str, text: str, page: int = 1) -> Chunk:
        return Chunk(
            id=f"{doc_id}_p{page}_c0",
            text=text,
            metadata=ChunkMetadata(
                source=f"{doc_id}.pdf",
                doc_id=doc_id,
                page_start=page,
                page_end=page,
                chunk_index=0,
                section="Method" if doc_id == "paperB" else None,
            ),
        )

    chunks = [
        _chunk("paperA", "Attention is all you need: transformer self-attention.", page=2),
        _chunk("paperB", "FlashAttention computes exact attention with IO-awareness on GPUs.", page=3),
        _chunk("paperC", "RLHF aligns large language models with human feedback.", page=4),
        _chunk("paperD", "A survey of LLMs across reasoning and multilingual tasks.", page=5),
    ]
    upsert_chunks(client, chunks, embedder)

    # Build BM25 sidecar.
    tokens = bm25s.tokenize([c.text for c in chunks], stopwords="en", show_progress=False)
    r = bm25s.BM25()
    r.index(tokens, show_progress=False)
    idx = settings.bm25_index_dir
    if idx.exists():
        shutil.rmtree(idx)
    idx.mkdir(parents=True)
    r.save(str(idx))
    settings.bm25_chunk_ids_path.write_text(json.dumps([c.id for c in chunks]))

    retriever = bm25s.BM25.load(str(idx), load_corpus=False)
    chunk_ids = json.loads(settings.bm25_chunk_ids_path.read_text())
    return client, embedder, retriever, chunk_ids


def test_hybrid_search_finds_topical_match(populated_index):
    """Sparse leg (BM25) drives this: 'flashattention' is a precise term."""
    client, embedder, retriever, chunk_ids = populated_index
    hits = hybrid_search(
        qdrant_client=client,
        embedder=embedder,
        bm25_retriever=retriever,
        bm25_chunk_ids=chunk_ids,
        query="flashattention GPU memory IO",
        top_k=3,
    )
    assert hits, "expected at least one hit"
    assert hits[0].source == "paperB.pdf"
    # Page metadata round-trips through the Qdrant payload.
    assert hits[0].page == 3
    assert hits[0].section == "Method"
    for h in hits:
        assert h.snippet
        assert h.text
        assert h.chunk_id


def test_hybrid_search_keyword_only(populated_index):
    """RLHF — surfaces paperC via the sparse leg."""
    client, embedder, retriever, chunk_ids = populated_index
    hits = hybrid_search(
        qdrant_client=client,
        embedder=embedder,
        bm25_retriever=retriever,
        bm25_chunk_ids=chunk_ids,
        query="RLHF",
        top_k=2,
    )
    sources = [h.source for h in hits]
    assert "paperC.pdf" in sources
