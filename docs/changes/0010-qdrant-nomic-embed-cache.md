# 0010 — Qdrant + Nomic Embed v1.5 + LRU embed cache

**Stage:** 2 (extraction + embeddings overhaul)
**Files touched:** `src/embedding.py` (rewritten), `src/qdrant_store.py`
(new), `src/vector_store.py` (deleted), `src/search.py`, `src/main.py`,
`src/config.py`, `pyproject.toml`, `docker-compose.yml`,
`.env.example`, `tests/_fakes.py` (new), `tests/test_qdrant_store.py`
(new), `tests/test_embedding.py` (new), `tests/test_bm25_integration.py`.

## Why bundle Qdrant, Nomic, and the cache together

Three reasons not to land them serially:

1. **Chroma's default embedder** (ONNX MiniLM-L6-v2, 384-d) was wired
   into the storage layer. Swapping the embedder *without* swapping the
   store would have meant maintaining two embedding paths for one
   change cycle — wasted work since the plan was always to leave Chroma.
2. **Qdrant doesn't ship an embedder.** You bring your own. So adopting
   Qdrant forces the embedder decision.
3. **The cache slots in cleanly here** — the `Embedder` protocol and
   the per-instance LRU are part of the same module, and the rest of
   the system (CLI/GUI/tests) already speaks Pydantic models from 0008
   so we touched the same surface anyway.

## Stack changes

### Vector store: Chroma → **Qdrant**

| Aspect | Chroma (before) | Qdrant (now) |
|---|---|---|
| Process model | Embedded sqlite + segments | **Server** (Docker), gRPC + REST |
| Embedder | Bundled ONNX MiniLM (384d) | BYO — we use Nomic v1.5 (512d) |
| Filtering | Limited | Strong (used in Stage 4 KG) |
| Test mode | Persistent dir tmp_path | `QdrantClient(":memory:")` |

Production deployment runs as a docker-compose service alongside
GROBID:

```yaml
qdrant:
  image: qdrant/qdrant:v1.12.4
  ports: [ "6333:6333", "6334:6334" ]
  volumes:
    - qdrant_data:/qdrant/storage
  restart: unless-stopped
  healthcheck: ...
```

A named volume holds the index so wiping the container doesn't kill
the data.

### Embedder: Chroma's MiniLM → **Nomic Embed v1.5 (512d)**

`src/embedding.py` exposes:

- `Embedder` Protocol: `dim: int`, `embed_documents`, `embed_query`.
  Tests pass a `FakeEmbedder` (keyword-overlap, 15-d) that satisfies
  the same shape — production code remains type-correct without
  loading torch.
- `NomicEmbedder` — `sentence-transformers` wrapper around
  `nomic-ai/nomic-embed-text-v1.5`. Highlights:
  - **Prefix tokens** are mandatory: `search_document: <text>` for
    indexed text, `search_query: <text>` for queries. Skipping them
    ships generic embeddings that materially under-perform on
    retrieval.
  - **Matryoshka truncation** to 512d via the
    `truncate_dim=512` constructor arg. Sentence-Transformers handles
    re-normalisation. Default chosen for the speed/quality knee.
  - **Auto device selection** — MPS on Apple Silicon, CUDA on Nvidia,
    CPU otherwise.
  - **Heavy imports deferred** to `__init__`, so importing the module
    doesn't pay the torch + sentence-transformers cost (~3 s, ~700 MB
    on disk). Tests that never instantiate `NomicEmbedder` are
    unaffected.

### Caching

Three cache layers, none of them clever:

1. **Per-instance LRU on query embeddings** (default 1024). Closure-
   bound (the standard Python idiom — `lru_cache` on methods leaks
   `self`). Exposed via `embedder.cache_info()` for telemetry.
2. **Embedder singleton in `main.py`** so consecutive CLI/GUI calls
   within one process reuse the loaded model and the LRU. Loading
   Nomic is ~3 s on M-series; reusing the singleton saves that on
   every search after the first.
3. **Document embeddings are cached implicitly in Qdrant** — once a
   chunk is upserted, we never re-embed it. No process-local doc cache
   needed; that's just RAM the OS could be using for the page cache.

Future caches worth considering (none of these warrant it now):
- Reranker outputs (when 0011 lands) — `(query, [chunk_ids])` keys.
- Tokenized BM25 corpus — bm25s already serializes this to disk.

## Code architecture

### `src/qdrant_store.py`

- `chunk_id_to_point_id(chunk_id) -> uuid5(NS, chunk_id)`. Qdrant
  point ids must be int or UUID; ours are stable strings. Deterministic
  UUIDv5 means upserts are idempotent and we don't need a side table
  to map "what's the UUID for chunk X?". The original chunk id rides
  in `payload["chunk_id"]`.
- `get_qdrant_client(url)` — server mode in production
  (`http://localhost:6333`), `:memory:` in tests.
- `ensure_collection(client, dim)` — idempotent collection creation
  with cosine distance.
- `upsert_chunks(client, chunks, embedder, batch_size=64)` — embed and
  upsert in batches. Payload includes the full Pydantic
  `ChunkMetadata.model_dump(exclude_none=True)` plus the chunk text
  (Qdrant doesn't store text by default — we keep it in payload so
  search responses don't need a separate fetch).
- `scroll_all_chunks(client)` — generator over every payload, used to
  rebuild BM25 from the current Qdrant contents.

### `src/search.py`

`hybrid_search` now takes a `qdrant_client` + `Embedder` instead of a
Chroma collection. Flow:

1. Embed query (LRU usually hits) → dense `query_points` against
   Qdrant; collect ranked `chunk_ids` and full payloads.
2. BM25 retrieve → ranked `chunk_ids`.
3. RRF on the two id lists.
4. For top_k results: **reuse payloads** already returned by the dense
   leg; only `client.retrieve(...)` for chunk_ids that surfaced solely
   in the sparse leg. Most queries (where dense and sparse mostly
   agree) need zero hydration round-trips.

`reciprocal_rank_fusion` is unchanged from 0006/0008 — generic over
hashable id types.

### `src/main.py`

- Replaced Chroma calls with `get_qdrant_client` + `ensure_collection`
  + `upsert_chunks` + `scroll_all_chunks`.
- Added `_get_embedder()` singleton so the model is loaded once per
  process — meaningful CLI UX win when running several searches
  back-to-back.
- `reindex_headless` now also drops the Qdrant collection (server-side)
  in addition to wiping local sidecar files.

## Configuration

New env vars:

| Key | Default |
|---|---|
| `HR_QDRANT_URL` | `http://localhost:6333` |
| `HR_QDRANT_TIMEOUT_S` | `30` |
| `HR_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` |
| `HR_EMBEDDING_DIM` | `512` |
| `HR_EMBED_CACHE_SIZE` | `1024` |

`Settings` gained the corresponding fields.

## Dependencies

```toml
- "chromadb (>=1.3.5,<2.0.0)"
+ "qdrant-client (>=1.12,<2.0)"
+ "sentence-transformers (>=3.0,<6.0)"
+ "torch (>=2.1,<3.0)"
+ "einops (>=0.7,<1.0)"
```

`einops` is required by Nomic's remote modeling code (it's in the
ops graph). `torch` is pulled in by sentence-transformers but pinned
explicitly so we own the version we're testing against.

`numpy` constraint loosened from `==1.26.4` to `>=1.26,<3.0` —
sentence-transformers + torch don't tolerate the pin on newer
toolchains.

## Tests

42 tests total (was 35). New:

- `tests/_fakes.py::FakeEmbedder` — keyword-overlap embedder so
  integration tests can assert dense-vs-sparse rankings deterministically
  without loading Nomic.
- `tests/test_embedding.py` (3 tests) — Protocol shape, keyword-overlap
  cosine sanity, LRU cache behaviour mirrored from `NomicEmbedder`.
- `tests/test_qdrant_store.py` (4 tests) — UUID5 determinism, upsert +
  scroll round-trip, idempotent collection creation, upsert-as-overwrite
  for stable chunk ids.
- `tests/test_bm25_integration.py` rewritten — uses Qdrant in-memory
  + `FakeEmbedder`. Asserts FlashAttention paper ranks #1 for the
  topical query, page metadata round-trips through Qdrant payloads,
  section soft-hint surfaces in `SearchHit.section`.

CI / local `pytest` does **not** require live Qdrant or GROBID, and
does **not** download Nomic weights. The full suite runs in ~1.5 s.

## Knock-on effects

- **Existing Chroma data is dead.** Pre-0010 `~/.hybrid_retrieval/`
  contains `chroma.sqlite3` + segment dirs. After this change those are
  ignored; `python -m src reindex` (or just deleting the persist dir)
  cleans them up.
- **First `process` after upgrade is slow.** Nomic v1.5 weights download
  on first use (~250 MB, one-time, cached in `~/.cache/huggingface`).
  All subsequent runs are fast.
- **Sentence-Transformers + torch download size** is significant for
  fresh installs. Mitigated by Apple Silicon users having torch already
  via other tooling, and by pinning to versions that match prevailing
  CI defaults.

## Follow-ups not done here

- **Reranker (0011).** Cross-encoder over top-50 fused candidates.
  Will reuse the same Embedder Protocol pattern (different protocol,
  same approach).
- **Concurrent embedding.** `model.encode(batch_size=32)` is already
  GPU-batched. For huge first-time indexes we could pipeline
  GROBID-extract and Nomic-embed; not yet needed.
- **Qdrant payload indexes.** For the KG layer we'll want `keyword`
  filter indexes on `doc_id` and `source` so server-side filtering is
  cheap. Trivial to add when we get there.
- **gRPC.** `qdrant-client` supports `prefer_grpc=True` for ~1.5×
  throughput on bulk upserts. Worth flipping on once we measure index
  time on a real corpus.
