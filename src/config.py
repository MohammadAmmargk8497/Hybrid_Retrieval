"""Application configuration.

All paths and tuning knobs live here. Values can be overridden by environment
variables (12-factor style) so deployments — including the upcoming FastAPI
service in Stage 3 — never need to edit code:

    HR_PDF_DIRECTORY=/some/folder \
    HR_PERSIST_DIRECTORY=/data/index \
    HR_TOP_K=10 \
        python -m src process

A `.env` file in the repo root is loaded automatically if present. Unknown
keys in `.env` are ignored (no `python-dotenv` dependency — see
``_load_dotenv`` below).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "HR_"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny `.env` loader — KEY=VALUE per line, `#` comments, no shell parsing.

    We avoid the `python-dotenv` dep for one ~20-line function. Existing env
    vars win over `.env` so shell overrides always work.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(ENV_PREFIX + name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Frozen, fully-resolved app config. Built once at import time."""

    pdf_directory: Path
    persist_directory: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_k: int | None  # None → search.py picks 4×top_k
    grobid_url: str
    grobid_timeout_s: float
    qdrant_url: str
    qdrant_timeout_s: float
    embedding_model: str
    embedding_dim: int
    embed_cache_size: int

    @property
    def failed_pdfs_path(self) -> Path:
        return self.persist_directory / "failed_pdfs.txt"

    @property
    def processed_pdfs_path(self) -> Path:
        return self.persist_directory / "processed_pdfs.txt"

    @property
    def log_file(self) -> Path:
        return self.persist_directory / "pdf_processing.log"

    @property
    def bm25_index_dir(self) -> Path:
        """Directory containing the bm25s index (multiple .npy / .json files)."""
        return self.persist_directory / "bm25_index"

    @property
    def bm25_chunk_ids_path(self) -> Path:
        """Sidecar JSON: chunk_ids aligned with bm25 doc indices."""
        return self.bm25_index_dir / "chunk_ids.json"

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        pdf_dir = Path(_env("PDF_DIRECTORY", str(Path.home() / "Downloads"))).expanduser()
        persist = Path(_env("PERSIST_DIRECTORY", str(Path.home() / ".hybrid_retrieval"))).expanduser()
        persist.mkdir(parents=True, exist_ok=True)
        candidate_raw = os.environ.get(ENV_PREFIX + "CANDIDATE_K")
        return cls(
            pdf_directory=pdf_dir,
            persist_directory=persist,
            chunk_size=_env_int("CHUNK_SIZE", 1200),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            top_k=_env_int("TOP_K", 5),
            candidate_k=int(candidate_raw) if candidate_raw else None,
            grobid_url=_env("GROBID_URL", "http://localhost:8070"),
            grobid_timeout_s=float(_env("GROBID_TIMEOUT_S", "60")),
            qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_timeout_s=float(_env("QDRANT_TIMEOUT_S", "30")),
            embedding_model=_env("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
            embedding_dim=_env_int("EMBEDDING_DIM", 512),
            embed_cache_size=_env_int("EMBED_CACHE_SIZE", 1024),
        )


# Process-wide singleton. Import this from anywhere:  ``from src.config import settings``
settings = Settings.from_env()


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants.
#
# Existing call sites do ``from src.config import PDF_DIRECTORY`` etc. We
# keep those working so the config refactor is non-breaking. New code should
# import ``settings`` instead.
# ---------------------------------------------------------------------------
PDF_DIRECTORY = str(settings.pdf_directory)
PERSIST_DIRECTORY = str(settings.persist_directory)
FAILED_PDFS_PATH = str(settings.failed_pdfs_path)
PROCESSED_PDFS_PATH = str(settings.processed_pdfs_path)
LOG_FILE = str(settings.log_file)
BM25_INDEX_DIR = str(settings.bm25_index_dir)
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K = settings.top_k
