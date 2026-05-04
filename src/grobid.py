"""GROBID client and TEI → ``Chunk`` parser.

GROBID is a service that turns scholarly PDFs into structured TEI XML —
title, authors, abstract, sections (with `<head>` titles), paragraphs,
references, and **page-level coordinates** for each element. Compared to
PyPDF, the wins for retrieval are:

  * Section titles surface as soft metadata hints on every chunk.
  * Two-column layouts and headers/footers are correctly handled.
  * Author/affiliation extraction is good enough to feed into the future
    knowledge-graph layer without an additional LLM pass.

Run locally with:

    docker compose up -d

(see ``docker-compose.yml`` at the repo root). We intentionally use the
lightweight image — the ``-full`` variant adds DL-based header parsing
at the cost of ~5 GB extra image size, which we don't currently need.

If the service is unreachable, ``extract_chunks_from_pdf`` raises
``GrobidUnavailable``. The decision (per project direction) is to **hard
fail** rather than silently fall back to PyPDF — extraction quality is a
load-bearing input to retrieval quality, and a half-degraded index would
be worse than a clear error.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from src.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_TEI = TEI_NS["tei"]


class GrobidUnavailable(RuntimeError):  # noqa: N818 — name describes state, not error class
    """Raised when the GROBID service can't be reached or returned an error."""


class GrobidClient:
    """Thin wrapper around GROBID's HTTP API.

    Only the two endpoints we actually use are exposed: ``/api/isalive``
    for the up-front liveness probe, and ``/api/processFulltextDocument``
    which returns full TEI XML for a PDF.
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_alive(self) -> bool:
        """Probe ``/api/isalive``. Returns False on any network error."""
        try:
            r = requests.get(f"{self.base_url}/api/isalive", timeout=5)
        except requests.RequestException as e:
            logger.warning("GROBID is_alive probe failed: %s", e)
            return False
        return r.status_code == 200 and r.text.strip().lower() == "true"

    def process_fulltext(self, pdf_path: Path) -> str:
        """POST a PDF to ``/api/processFulltextDocument``; return TEI XML.

        We request ``teiCoordinates=p,head,s`` so paragraphs, section
        headings, and sentences carry their page-level coordinates —
        essential for the click-to-page UX.

        ``consolidateHeader=0`` and ``consolidateCitations=0`` skip the
        CrossRef enrichment passes; they're slow and we don't currently
        consume the metadata they fill in. Flip back on once the
        knowledge-graph layer needs DOIs.
        """
        with open(pdf_path, "rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            data = {
                "consolidateHeader": "0",
                "consolidateCitations": "0",
                "teiCoordinates": "p,head,s",
                "segmentSentences": "0",
            }
            try:
                r = requests.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise GrobidUnavailable(f"GROBID request failed: {e}") from e
        if r.status_code != 200:
            raise GrobidUnavailable(
                f"GROBID returned HTTP {r.status_code} for {pdf_path.name}: "
                f"{r.text[:200]}"
            )
        return r.text


# ---------------------------------------------------------------------------
# TEI parsing
# ---------------------------------------------------------------------------

def _first_page_from_coords(coords: str | None) -> int | None:
    """Pull the first page number out of a TEI ``coords`` attribute.

    GROBID emits things like ``"3,72.0,711.6,453.7,8.5;3,72.0,..."``
    where each semicolon-separated tuple is ``page,x,y,w,h``. We only
    care about the first page an element appears on.
    """
    if not coords:
        return None
    head = coords.split(";", 1)[0]
    page_str = head.split(",", 1)[0]
    try:
        return int(page_str)
    except ValueError:
        return None


def _paragraph_text(p_el: ET.Element) -> str:
    """Inline text of a `<p>`, including its `<s>`/`<ref>` children."""
    return " ".join(t for t in p_el.itertext() if t)


def parse_tei_to_chunks(
    tei_xml: str,
    *,
    source: str,
    doc_id: str,
    text_splitter: Any,
    clean_text: Callable[[str], str],
) -> list[Chunk]:
    """Parse TEI XML into ``Chunk``s, one set per page.

    Strategy:
      1. Pull the abstract (always associated with page 1).
      2. Walk the `<body>` div tree; for each `<p>`, derive a page from
         its ``coords`` (falling back to the first child sentence's
         coords, then to page 1).
      3. Group paragraph text by page, recording the first section title
         seen on each page as a *soft hint* (per project direction —
         section is metadata, not a chunking boundary).
      4. Run the recursive splitter over per-page text to produce final
         chunks.

    Sections that span pages contribute their head text only to the
    first page where their `<p>` lands; subsequent pages keep whatever
    earlier section was active. This is the deliberate "soft hints"
    semantic — section is informative, not authoritative.
    """
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as e:
        logger.error("TEI parse error for %s: %s", source, e)
        return []

    # page -> {"section": str|None, "parts": list[str]}
    by_page: dict[int, dict] = defaultdict(lambda: {"section": None, "parts": []})

    # 1. Abstract → page 1 (GROBID rarely emits coords on the abstract).
    abstract = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    if abstract is not None:
        text = clean_text(_paragraph_text(abstract))
        if text:
            slot = by_page[1]
            slot["section"] = "Abstract"
            slot["parts"].append(text)

    # 2. Body sections.
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is not None:
        for div in body.iter(f"{{{_TEI}}}div"):
            head_el = div.find("tei:head", TEI_NS)
            section_title = (
                (head_el.text or "").strip()
                if head_el is not None and head_el.text
                else None
            )

            for p in div.findall("tei:p", TEI_NS):
                page = _first_page_from_coords(p.get("coords"))
                if page is None:
                    for s in p.findall("tei:s", TEI_NS):
                        pg = _first_page_from_coords(s.get("coords"))
                        if pg is not None:
                            page = pg
                            break
                if page is None:
                    page = 1

                text = clean_text(_paragraph_text(p))
                if not text:
                    continue
                slot = by_page[page]
                if slot["section"] is None and section_title:
                    slot["section"] = section_title
                slot["parts"].append(text)

    # 3 + 4. Chunk per page.
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in sorted(by_page.keys()):
        slot = by_page[page]
        page_text = " ".join(slot["parts"]).strip()
        if not page_text:
            continue
        for piece in text_splitter.split_text(page_text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc_id}_p{page}_c{chunk_index}",
                    text=piece,
                    metadata=ChunkMetadata(
                        source=source,
                        doc_id=doc_id,
                        page_start=page,
                        page_end=page,
                        section=slot["section"],
                        chunk_index=chunk_index,
                    ),
                )
            )
            chunk_index += 1
    return chunks
