# 0004 — Delete dead `src/embedding.py`

**Stage:** 1 (foundation fixes) **Files touched:** `src/embedding.py` (removed)

## What was wrong

`src/embedding.py` exposed `get_embedding_model()` and `generate_embeddings()`
wrapping `SentenceTransformer(EMBEDDING_MODEL)` with the configured
`all-MiniLM-L6-v2`. Nothing imported it.

```bash
$ grep -rn "from src.embedding\|src\.embedding\|import embedding" src/ docs/
# (no matches)
```

The actual embedder used at runtime is **Chroma's default**, which is the
ONNX MiniLM bundled with `chromadb`. So `EMBEDDING_MODEL = 'all-MiniLM-L6-v2'`
in `config.py` was decorative — changing it had no effect.

## Why this matters

Two reasons it's actively harmful, not just dead weight:

1. **False configurability.** A reader (or future-me) looking at `config.py`
   would reasonably believe `EMBEDDING_MODEL` controls the embedder. It
   doesn't. Removing the dead module makes the actual behavior — *Chroma
   picks the embedder* — visible.
2. **Hidden cost vector.** `sentence-transformers` is a heavy dep
   (PyTorch). Keeping a code path that imports it but is never called
   means the dep can't be reasoned about ("is it required?") until we
   audit imports. Deleting clarifies.

## What I removed

- `src/embedding.py` (entire file).

## What I left for later

- The Stage-2 swap to **Nomic Embed v1.5** will reintroduce a real embedder
  module — but configured into Qdrant via a custom embedding function, not
  via Chroma's default path. So the right time to write a new `embedding.py`
  is when we wire Qdrant, not now.
- `EMBEDDING_MODEL` in `config.py` is left in place for one more change
  (removed in 0005 along with the broader config refactor) so this commit
  stays focused.
