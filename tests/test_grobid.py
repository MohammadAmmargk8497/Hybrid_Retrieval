"""Tests for the GROBID client + TEI parser — see docs/changes/0009.

We don't want CI / local `pytest` runs to require a live GROBID. Instead:
  * `parse_tei_to_chunks` is a pure function — drive it with a small
    synthetic TEI XML fixture.
  * `GrobidClient` is exercised via monkeypatched `requests`.
"""

from __future__ import annotations

import pytest
import requests

from src.grobid import (
    GrobidClient,
    GrobidUnavailable,
    _first_page_from_coords,
    parse_tei_to_chunks,
)
from src.pdf_processing import _build_splitter, clean_text

SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <profileDesc>
      <abstract>
        <p>We propose a hybrid retrieval system combining BM25 with dense vectors.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p coords="1,72.0,711.6,453.7,8.5">Information retrieval has long studied the
        complementary strengths of lexical and semantic matching.</p>
      </div>
      <div>
        <head>Method</head>
        <p coords="3,72.0,500.0,453.7,8.5">We combine sparse BM25 with dense Nomic
        embeddings via reciprocal rank fusion.</p>
        <p coords="4,72.0,500.0,453.7,8.5">FlashAttention enables efficient training
        of the underlying encoder on long contexts.</p>
      </div>
      <div>
        <head>Results</head>
        <p>This paragraph has no coords — should default to page 1.</p>
      </div>
    </body>
  </text>
</TEI>
"""


def _parse():
    return parse_tei_to_chunks(
        SAMPLE_TEI,
        source="paper.pdf",
        doc_id="paper",
        text_splitter=_build_splitter(chunk_size=1200, chunk_overlap=150),
        clean_text=clean_text,
    )


def test_first_page_from_coords():
    assert _first_page_from_coords("3,72.0,711.6,453.7,8.5") == 3
    assert _first_page_from_coords("3,1.0;5,2.0") == 3
    assert _first_page_from_coords(None) is None
    assert _first_page_from_coords("") is None
    assert _first_page_from_coords("garbage") is None


def test_abstract_is_emitted_on_page_1():
    chunks = _parse()
    page1 = [c for c in chunks if c.metadata.page_start == 1]
    assert any("hybrid retrieval system" in c.text for c in page1)
    # The abstract chunk's section is "Abstract" (we set it before walking body).
    abstract_chunk = next(c for c in page1 if "hybrid retrieval system" in c.text)
    assert abstract_chunk.metadata.section == "Abstract"


def test_section_titles_propagate_as_soft_hints():
    chunks = _parse()
    by_page = {c.metadata.page_start: c.metadata.section for c in chunks}
    assert by_page[3] == "Method"
    assert by_page[4] == "Method"   # same section spanning pages
    # "Results" had no coords → defaulted to page 1, where Abstract was already set.
    # Soft-hint semantics: first section title wins per page.


def test_pages_extracted_from_coords():
    chunks = _parse()
    pages = sorted({c.metadata.page_start for c in chunks})
    assert pages == [1, 3, 4]


def test_chunk_ids_are_unique_and_well_formed():
    chunks = _parse()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.id == f"paper_p{c.metadata.page_start}_c{c.metadata.chunk_index}"


def test_parse_invalid_xml_returns_empty():
    out = parse_tei_to_chunks(
        "<not-valid-xml",
        source="x.pdf",
        doc_id="x",
        text_splitter=_build_splitter(1200, 150),
        clean_text=clean_text,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Client tests (monkeypatched requests)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, text="true"):
        self.status_code = status_code
        self.text = text


def test_is_alive_true(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200, "true"))
    assert GrobidClient("http://x").is_alive() is True


def test_is_alive_false_on_non_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(503, "down"))
    assert GrobidClient("http://x").is_alive() is False


def test_is_alive_false_on_network_error(monkeypatch):
    def raise_(*a, **k):
        raise requests.ConnectionError("nope")
    monkeypatch.setattr(requests, "get", raise_)
    assert GrobidClient("http://x").is_alive() is False


def test_process_fulltext_raises_on_http_error(monkeypatch, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500, "boom"))
    with pytest.raises(GrobidUnavailable):
        GrobidClient("http://x").process_fulltext(pdf)


def test_extract_text_from_pdfs_hard_fails_when_grobid_down(monkeypatch, tmp_path):
    """The contract is: GROBID unavailable → raise, not silently fall back."""
    from src.pdf_processing import extract_text_from_pdfs

    class _DeadClient:
        base_url = "http://x"

        def is_alive(self):
            return False

    with pytest.raises(GrobidUnavailable):
        extract_text_from_pdfs(str(tmp_path), [], grobid=_DeadClient())
