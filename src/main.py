"""GUI + headless entry points.

This module used to be both the application entrypoint and the GUI. The
new CLI (``src/cli.py``) owns the entrypoint; this module now exposes
*headless* implementations of the three operations (process, search,
reindex) that take a tiny ``log(msg)`` callback so the GUI and the CLI
can share them.

The Tkinter GUI is retained for now and will be retired in Stage 3 when
the FastAPI service + web UI lands.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, scrolledtext

import bm25s

from src.config import settings
from src.models import SearchResponse
from src.pdf_processing import (
    extract_text_from_pdfs,
    load_failed_pdfs,
    load_pdfs_from_directory,
    load_processed_pdfs,
    save_failed_pdfs,
    save_processed_pdfs,
)
from src.qdrant_store import (
    COLLECTION,
    ensure_collection,
    get_qdrant_client,
    scroll_all_chunks,
    upsert_chunks,
)
from src.search import hybrid_search

logging.basicConfig(
    filename=str(settings.log_file),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Headless operations — usable from CLI, GUI, or future FastAPI handlers.
# ---------------------------------------------------------------------------

def _build_embedder():
    """Lazily construct the production embedder.

    Imported inside the function so module import doesn't pay the
    torch + sentence-transformers cost (~3 s, ~700 MB of deps). Tests
    that don't call ``process`` / ``search`` are unaffected.
    """
    from src.embedding import NomicEmbedder

    return NomicEmbedder(
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        cache_size=settings.embed_cache_size,
    )


def process_pdfs_headless(
    pdf_directory: str,
    persist_directory: str,
    log: LogFn = _noop,
) -> dict:
    """Index any new PDFs in ``pdf_directory`` into Qdrant + BM25.

    Returns a small summary dict — useful for CLI output and tests.
    """
    client = get_qdrant_client(settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    ensure_collection(client, dim=settings.embedding_dim)

    failed_pdfs_set = load_failed_pdfs(str(settings.failed_pdfs_path))
    processed_pdfs_set = load_processed_pdfs(str(settings.processed_pdfs_path))

    all_files = load_pdfs_from_directory(pdf_directory)
    new_filenames = [
        f for f in all_files if f not in processed_pdfs_set and f not in failed_pdfs_set
    ]

    log(f"Found {len(all_files)} PDFs, {len(new_filenames)} new to process.")

    new_files_processed = False
    n_chunks = 0
    n_failed = 0
    embedder = None  # built only if we have something to embed

    if new_filenames:
        chunks, failed_pdfs, success_pdfs = extract_text_from_pdfs(
            pdf_directory, new_filenames
        )
        n_chunks = len(chunks)
        n_failed = len(failed_pdfs)
        log(f"Extracted text from {len(success_pdfs)} PDFs ({n_chunks} chunks).")

        if failed_pdfs:
            save_failed_pdfs(failed_pdfs, str(settings.failed_pdfs_path))

        if chunks:
            log(f"Loading embedder ({settings.embedding_model}, {settings.embedding_dim}d)...")
            embedder = _build_embedder()
            log("Embedding + upserting to Qdrant...")
            upsert_chunks(client, chunks, embedder)
            save_processed_pdfs(list(set(success_pdfs)), str(settings.processed_pdfs_path))
            new_files_processed = True
        else:
            log("No valid text extracted; nothing stored.")

    index_dir = settings.bm25_index_dir
    if new_files_processed or not index_dir.exists():
        log("Building bm25s index from Qdrant payloads...")
        chunk_ids: list[str] = []
        corpus: list[str] = []
        for payload in scroll_all_chunks(client):
            chunk_ids.append(payload["chunk_id"])
            corpus.append(payload["text"])
        if corpus:
            # bm25s.tokenize handles lowercase, stopwords, regex word splits.
            # No stemming — preserves precise ML terminology (FlashAttention,
            # RoPE, ALiBi). Revisit when the reranker lands in 0011.
            tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
            retriever = bm25s.BM25()
            retriever.index(tokens, show_progress=False)

            if index_dir.exists():
                shutil.rmtree(index_dir)
            index_dir.mkdir(parents=True)
            retriever.save(str(index_dir))
            settings.bm25_chunk_ids_path.write_text(json.dumps(chunk_ids))
            log(f"bm25s index saved ({len(corpus)} docs).")
        else:
            log("Corpus is empty; skipping bm25s build.")

    log("Done.")
    return {
        "scanned": len(all_files),
        "new": len(new_filenames),
        "chunks_added": n_chunks,
        "failed": n_failed,
    }


# Embedder is held in module scope so repeated CLI/GUI calls within one
# process reuse the loaded model (and the LRU query cache).
_EMBEDDER_SINGLETON = None


def _get_embedder():
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is None:
        _EMBEDDER_SINGLETON = _build_embedder()
    return _EMBEDDER_SINGLETON


def run_search_headless(
    query: str,
    persist_directory: str,
    top_k: int | None = None,
) -> SearchResponse:
    """Run a hybrid search; returns a typed ``SearchResponse``."""
    import time

    top_k = top_k or settings.top_k

    index_dir = settings.bm25_index_dir
    if not index_dir.exists() or not settings.bm25_chunk_ids_path.exists():
        raise FileNotFoundError(
            f"bm25s index not found at {index_dir}. Run `process` first."
        )

    client = get_qdrant_client(settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    embedder = _get_embedder()
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    chunk_ids = json.loads(settings.bm25_chunk_ids_path.read_text())

    t0 = time.perf_counter()
    hits = hybrid_search(
        qdrant_client=client,
        embedder=embedder,
        bm25_retriever=retriever,
        bm25_chunk_ids=chunk_ids,
        query=query,
        top_k=top_k,
        candidate_k=settings.candidate_k,
    )
    took_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=query, top_k=top_k, hits=hits, took_ms=took_ms)


def reindex_headless(persist_directory: str, pdf_directory: str, log: LogFn = _noop) -> dict:
    """Wipe the index and rebuild from scratch.

    Removes:
      * Qdrant collection (dropped server-side via the client)
      * ``processed_pdfs.txt`` so every PDF is re-extracted with the
        current cleaner / chunker / extractor
      * BM25 artifacts in ``persist_directory``
      * Any leftover Chroma sqlite+segment files from the pre-0010 days

    Keeps:
      * ``failed_pdfs.txt`` — PDFs that failed extraction usually fail
        again; keeping the list avoids retry storms. Delete it manually
        to force retry (or once we move to GROBID-only for real, it's
        worth re-running once).
      * The log file.
    """
    persist = settings.persist_directory
    log(f"Wiping local index artifacts under {persist} ...")

    keep = {settings.failed_pdfs_path.name, settings.log_file.name}
    for entry in persist.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    log(f"Dropping Qdrant collection {COLLECTION!r} ...")
    client = get_qdrant_client(settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    try:
        client.delete_collection(collection_name=COLLECTION)
    except Exception as e:
        # Common case: collection didn't exist yet — fine.
        log(f"  (delete_collection skipped: {e})")

    log("Rebuilding...")
    return process_pdfs_headless(pdf_directory, persist_directory, log=log)


# ---------------------------------------------------------------------------
# GUI (legacy, to be retired in Stage 3).
# ---------------------------------------------------------------------------

def open_pdf(file_path: str) -> None:
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.call(["open", file_path])
        else:
            subprocess.call(["xdg-open", file_path])
        logging.info("Opened PDF: %s", file_path)
    except Exception as e:
        logging.error("Failed to open %s: %s", file_path, e)


def _display_results(response, output_widget, directory=None, top_k=5):
    output_widget.delete("1.0", tk.END)
    output_widget.insert(
        tk.END, f"{len(response.hits)} hits in {response.took_ms:.1f} ms\n\n"
    )
    opened: set[str] = set()
    for idx, hit in enumerate(response.hits):
        section = f" [{hit.section}]" if hit.section else ""
        output_widget.insert(
            tk.END,
            f"[{idx + 1}] {hit.source}  p.{hit.page}{section}  (score={hit.score:.5f})\n"
            f"    {hit.snippet}\n\n",
        )
        if idx < top_k and directory and hit.source not in opened:
            file_path = os.path.join(directory, hit.source)
            if os.path.exists(file_path):
                open_pdf(file_path)
                opened.add(hit.source)
            else:
                output_widget.insert(tk.END, f"File not found: {file_path}\n")


def _browse_into(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def launch_gui() -> None:
    root = tk.Tk()
    root.title("Hybrid_Retrieval")

    tk.Label(root, text="PDF Directory:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    pdf_dir_entry = tk.Entry(root, width=50)
    pdf_dir_entry.grid(row=0, column=1, padx=5, pady=5)
    pdf_dir_entry.insert(0, str(settings.pdf_directory))
    tk.Button(root, text="Browse", command=lambda: _browse_into(pdf_dir_entry)).grid(
        row=0, column=2, padx=5, pady=5
    )

    tk.Label(root, text="Persist Directory:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    persist_dir_entry = tk.Entry(root, width=50)
    persist_dir_entry.grid(row=1, column=1, padx=5, pady=5)
    persist_dir_entry.insert(0, str(settings.persist_directory))
    tk.Button(root, text="Browse", command=lambda: _browse_into(persist_dir_entry)).grid(
        row=1, column=2, padx=5, pady=5
    )

    output = scrolledtext.ScrolledText(root, width=80, height=20)
    output.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

    def gui_log(msg: str) -> None:
        output.insert(tk.END, msg + "\n")
        output.update_idletasks()

    tk.Button(
        root,
        text="Process PDFs",
        command=lambda: process_pdfs_headless(pdf_dir_entry.get(), persist_dir_entry.get(), gui_log),
    ).grid(row=3, column=0, padx=5, pady=5, sticky="e")

    tk.Label(root, text="Search Query:").grid(row=3, column=1, sticky="e", padx=5, pady=5)
    query_entry = tk.Entry(root, width=30)
    query_entry.grid(row=3, column=2, padx=5, pady=5, sticky="w")

    def do_search():
        try:
            response = run_search_headless(query_entry.get(), persist_dir_entry.get())
        except FileNotFoundError as e:
            gui_log(str(e))
            return
        _display_results(response, output, directory=pdf_dir_entry.get(), top_k=settings.top_k)

    tk.Button(root, text="Search", command=do_search).grid(row=4, column=2, padx=5, pady=5, sticky="e")

    root.mainloop()


def main() -> None:
    """Backward-compatible shim — preserves ``python -m src.main``."""
    launch_gui()


if __name__ == "__main__":
    main()
