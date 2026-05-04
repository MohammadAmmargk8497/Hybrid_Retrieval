import logging
import os
import re
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, settings
from src.grobid import GrobidClient, GrobidUnavailable, parse_tei_to_chunks
from src.models import Chunk

# ==========================
# Helper Functions
# ==========================

# Control chars except \t \n \r — these survive PDF extraction but break
# tokenizers and downstream string handling.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text while preserving meaningful unicode.

    Previous implementation stripped all non-ASCII via ``[^\\x00-\\x7F]+``,
    which deleted Greek letters, math symbols, accented author names, and
    in-line equations — exactly the tokens that distinguish AI papers.

    What we do now:
      * NFKC-normalize so PDF ligatures (``ﬁ``, ``ﬀ``, ``ﬂ`` …) and
        compatibility forms collapse to their canonical equivalents
        (``fi``, ``ff``, ``fl``).
      * Drop only ASCII control characters (PDFs occasionally embed these).
      * Collapse runs of whitespace to a single space.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text

def load_pdfs_from_directory(directory: str):
    """
    Return a list of all PDF filenames in the given directory.
    """
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def load_failed_pdfs(filepath: str) -> set:
    """
    Loads the list of failed PDFs from the specified file and returns a set of filenames.
    """
    failed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    failed_pdfs.add(pdf)
        logging.info(f"Loaded {len(failed_pdfs)} failed PDFs from {filepath}.")
    else:
        logging.info(f"No existing failed PDFs found at {filepath}.")
    return failed_pdfs

def save_failed_pdfs(failed_pdfs: list, filepath: str):
    """
    Appends the list of failed PDFs to the specified file.
    """
    if not failed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in failed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(failed_pdfs)} failed PDFs to {filepath}.")

def load_processed_pdfs(filepath: str) -> set:
    """
    Loads the list of processed PDFs from the specified file.
    """
    processed_pdfs = set()
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                pdf = line.strip()
                if pdf:
                    processed_pdfs.add(pdf)
        logging.info(f"Loaded {len(processed_pdfs)} processed PDFs from {filepath}.")
    else:
        logging.info(f"No existing processed PDFs found at {filepath}.")
    return processed_pdfs

def save_processed_pdfs(processed_pdfs: list, filepath: str):
    """
    Appends the list of processed PDFs to the specified file.
    """
    if not processed_pdfs:
        return
    with open(filepath, 'a') as f:
        for pdf in processed_pdfs:
            f.write(f"{pdf}\n")
    logging.info(f"Added {len(processed_pdfs)} processed PDFs to {filepath}.")

def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )


def extract_text_from_pdfs(
    directory: str,
    filenames: list,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    grobid: GrobidClient | None = None,
) -> tuple[list[Chunk], list[str], list[str]]:
    """Extract, clean, and chunk text from PDFs via GROBID.

    Returns ``(chunks, failed_filenames, ok_filenames)``.

    **Hard fails** with ``GrobidUnavailable`` if the service isn't up. The
    decision (per project direction) is that a half-degraded index is
    worse than a clear "GROBID isn't running" error: extraction quality
    is the input that bounds retrieval quality, and we'd rather you fix
    the pipeline than ship corrupted recall.

    The ``grobid`` argument is mainly for tests — production calls pass
    ``None`` and we construct a client from ``settings.grobid_url``.
    """
    if grobid is None:
        grobid = GrobidClient(settings.grobid_url, timeout=settings.grobid_timeout_s)
    if not grobid.is_alive():
        raise GrobidUnavailable(
            f"GROBID is not reachable at {grobid.base_url}. "
            "Start it with: `docker compose up -d` (see docker-compose.yml)."
        )

    splitter = _build_splitter(chunk_size, chunk_overlap)

    all_chunks: list[Chunk] = []
    failed_pdfs: list[str] = []
    success_pdfs: list[str] = []

    for filename in filenames:
        filepath = Path(directory) / filename
        try:
            tei_xml = grobid.process_fulltext(filepath)
            chunks = parse_tei_to_chunks(
                tei_xml,
                source=filename,
                doc_id=filepath.stem,
                text_splitter=splitter,
                clean_text=clean_text,
            )
            if not chunks:
                logging.warning("GROBID parsed no chunks from %s.", filename)
                failed_pdfs.append(filename)
                continue
            all_chunks.extend(chunks)
            success_pdfs.append(filename)
            logging.info("Processed %s: %d chunks.", filename, len(chunks))

        except GrobidUnavailable:
            # Service died mid-batch — propagate up, don't silently mark
            # this PDF as failed and keep going.
            raise
        except Exception as e:
            logging.error("Error processing %s: %s. Skipping.", filename, e)
            failed_pdfs.append(filename)
            continue

    return all_chunks, failed_pdfs, success_pdfs
