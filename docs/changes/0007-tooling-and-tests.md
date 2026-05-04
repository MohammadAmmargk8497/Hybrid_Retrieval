# 0007 — Tooling: ruff + mypy + pytest + Makefile

**Stage:** 1 (foundation fixes)
**Files touched:** `pyproject.toml`, `Makefile` (new),
`tests/test_search.py` (new), `tests/test_pdf_processing.py` (new),
`tests/test_config.py` (new), `tests/test_bm25_integration.py` (new),
`tests/__init__.py` (new). Also auto-formatted: `src/main.py`,
`src/search.py`, `src/pdf_processing.py`, `src/vector_store.py`,
`src/config.py`.

## Why

The four prior changes (0001–0006) all needed manual verification at the
REPL because there was nothing to lean on. Before we move into Stage 2
(GROBID, Nomic, reranker — each a much bigger surface area), we need
mechanically-enforced guardrails:

- **Tests** — so refactors don't silently break the bug-fixes.
- **ruff** — a single formatter + linter. Catches dead imports, sort
  order, deprecated patterns, common bugs (`B`).
- **mypy** — light typing checks; catches the easy mistakes (None vs
  str, missing branches in unions).
- **Make targets** — one entrypoint each for the three.

## What landed

### `pyproject.toml`

```toml
[project.optional-dependencies]
dev = [
    "ruff (>=0.6,<1.0)",
    "mypy (>=1.10,<2.0)",
    "pytest (>=8,<9)",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "SIM"]
ignore = ["E501", "B008"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
...

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Also bumped `requires-python` from the poetry-flavored `^3.10` to the
PEP-621-compliant `>=3.10,<4.0` so `ruff` can parse the file.

### `Makefile`

Targets: `install` (`poetry install --with dev` with pip fallback),
`lint`, `format`, `type`, `test`, `check` (= lint + type + test),
`clean`. Auto-detects whether to invoke through `poetry run` or
directly.

### Tests

| File | Coverage |
|---|---|
| `tests/test_search.py` | RRF math: single-list formula, cross-list summing, sort order, generic id types (str + int), empty input. Pure tests, no Chroma / bm25s deps. |
| `tests/test_pdf_processing.py` | `clean_text`: Greek/math preservation, accented names, ligature folding (`ﬁ` → `fi`), control-char strip, whitespace collapse, empty input. |
| `tests/test_config.py` | Defaults, env override, persist dir auto-creation, `.env` loading, shell-env wins over `.env`. |
| `tests/test_bm25_integration.py` | End-to-end: tiny in-memory Chroma + bm25s + RRF. Asserts FlashAttention paper ranks #1 for an FA-related query, and that "RLHF" surfaces the RLHF paper. `pytest.importorskip` on Chroma + bm25s so the suite stays green even without optional deps. |

18 tests total, all passing.

### Auto-formatting pass

`ruff check --fix` cleaned up:

- Import sort order across all of `src/`.
- `Callable` / `Hashable` / `Iterable` / `Sequence` moved from `typing`
  to `collections.abc` (3.9+ canonical location).
- `Optional[X]` → `X | None` everywhere (PEP 604, target 3.10).
- `with open(...) as f` (no implicit `'r'` mode arg).
- `zip(..., strict=True)` on the one materialization in `search.py`.
- Removed an unused leftover `existing_docs_set` in `process_pdfs_headless`.

Two `B905` warnings were the only ones I had to address by hand; the
rest were auto-fixable.

## Running

```bash
make install   # pulls dev deps via poetry (or pip fallback)
make test      # 18 tests, ~3.5s
make lint      # ruff check
make format    # ruff format + autofix
make type      # mypy src
make check     # all three
```

## What's deliberately not here

- **CI.** No GitHub Actions yet — that's Stage 3 territory once the
  service shape stabilizes (a workflow that runs `make check` is one
  small file, easy to add when wanted).
- **Coverage.** Not measuring yet. The four bug-fix areas (RRF,
  clean_text, config, bm25s integration) are covered; raw % isn't useful
  on a code base this small.
- **mypy `--strict`.** Shooting for clean default mypy first. Tightening
  to strict is a Stage 2 task once the GROBID + embedder modules give us
  real data shapes worth pinning.
