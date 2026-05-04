# 0005 — Env / CLI config + `reindex` command

**Stage:** 1 (foundation fixes) **Files touched:** `src/config.py`,
`src/main.py`, `src/cli.py` (new), `src/__main__.py` (new),
`src/__init__.py` (new), `.env.example` (new), `pyproject.toml`

## What was wrong

- `config.py` hard-coded absolute paths from one machine
  (`/Users/ammar/Downloads/`, `/Users/ammar/Desktop/ProjectX`). The repo
  was unrunnable for anyone else without editing source.
- The only entrypoint was `python -m src.main`, which **always launched
  Tkinter**. There was no way to (a) index headlessly, (b) run a one-shot
  search from the shell, (c) wipe and rebuild the index — all needed for
  scripting and for integration with the upstream Darwin Research Agent.
- After changes 0002 and 0003 (unicode preservation + chunk shrink), the
  on-disk Chroma store was stale; we needed a clean way to rebuild it.

## Fix

### Config — env-driven `Settings` dataclass

`src/config.py` now exposes a frozen `Settings` dataclass populated from
`HR_*` environment variables, with defaults:

| Key | Default |
|---|---|
| `HR_PDF_DIRECTORY` | `~/Downloads` |
| `HR_PERSIST_DIRECTORY` | `~/.hybrid_retrieval` |
| `HR_CHUNK_SIZE` | `1200` |
| `HR_CHUNK_OVERLAP` | `150` |
| `HR_TOP_K` | `5` |
| `HR_CANDIDATE_K` | unset (search picks `4×top_k`) |

A `.env` file in repo root is auto-loaded by a 20-line parser; shell env
vars win over `.env` (so CI / one-shot overrides work). `python-dotenv`
is *not* a dependency — keeping it tiny.

The persist directory is auto-created on import.

The old module-level constants (`PDF_DIRECTORY`, `PERSIST_DIRECTORY`,
`CHUNK_SIZE`, …) are kept as backward-compat re-exports of
`settings.<field>`, so all existing imports continue to work without
edits. New code should `from src.config import settings`.

### Headless ops — `process / search / reindex`

`src/main.py` was both the CLI entrypoint and the GUI. Split:

- `process_pdfs_headless(pdf_dir, persist_dir, log)` — returns a summary
  dict (scanned / new / chunks_added / failed).
- `run_search_headless(query, persist_dir, top_k)` — returns the raw
  result dicts.
- `reindex_headless(persist_dir, pdf_dir, log)` — wipes everything in
  `persist_directory` *except* `failed_pdfs.txt` and the log, then
  re-runs `process`.
- `launch_gui()` — the Tkinter window (now a thin shell over the
  headless functions, sharing the same code paths).

All three accept a `log: Callable[[str], None]` so the GUI, CLI, and
future FastAPI handlers route output to the right place.

`reindex` deliberately **keeps** `failed_pdfs.txt` — those PDFs failed
extraction for a reason (corrupt, encrypted, scanned-image-only) and
will fail again. To force-retry, delete that file manually. We'll
revisit when we move to GROBID in Stage 2 since better extraction may
recover some.

### CLI — `python -m src ...`

`src/cli.py` adds an `argparse` CLI with five subcommands:

```text
hybrid-retrieval [-h] {process,search,gui,reindex,info}
```

- `process` — index any new PDFs.
- `search "<query>" [--top-k N] [--json]` — print ranked snippets.
- `gui` — launch the Tkinter window (legacy).
- `reindex` — wipe + rebuild.
- `info` — print resolved configuration (sanity-check env loading).

`src/__main__.py` makes `python -m src ...` work. A
`[project.scripts]` entry in `pyproject.toml` adds a `hybrid-retrieval`
shell binary on install.

### Dependencies

Removed two unused deps from `pyproject.toml`:

- `sentence-transformers` — only used by the deleted `embedding.py` (0004).
- `faiss-cpu` — never imported anywhere in `src/`.

Stayed conservative on the rest (`langchain==0.1.16`,
`langchain-community==0.0.38`) since `poetry.lock` is already pinned to
those and nothing new requires bumping. We'll re-evaluate when we swap
to GROBID in Stage 2.

## Verification

```bash
$ python -m src info
{ "pdf_directory": "/Users/ammar/Downloads", ... "chunk_size": 1200, ... }

$ HR_PDF_DIRECTORY=/tmp/foo HR_TOP_K=10 python -m src info
{ "pdf_directory": "/tmp/foo", ... "top_k": 10, ... }

$ python -m src --help
# ... lists all five subcommands.
```

## Re-index path now formalized

Anyone wanting to retroactively apply the changes from 0002 (unicode
preserve) and 0003 (chunk shrink) to PDFs already indexed can run:

```bash
python -m src reindex
```

This is the deferral mentioned in those two change docs — now resolved.

## Follow-ups not done here

- `pyproject.toml` still pins old `langchain` versions; bumping is
  deferred to Stage 2 (GROBID swap is the natural place).
- No tests yet — landing in 0007.
- `.env` is already gitignored (verified `.gitignore` line 7). No new
  `.gitignore` rule needed.
