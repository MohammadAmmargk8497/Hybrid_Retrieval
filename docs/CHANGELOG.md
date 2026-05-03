# Changelog

Newest first. Each entry links to a detailed change note in `docs/changes/`.

## Stage 1 — Foundation fixes

- **0001** — [Fix RRF fusion bug](changes/0001-fix-rrf-fusion-bug.md) — Hybrid scoring in `src/search.py` mixed rank indices with raw BM25 scores into the same `1/(k+x)` term, badly biasing results. Rewrote `reciprocal_rank_fusion` to consume per-ranker rank lists.
