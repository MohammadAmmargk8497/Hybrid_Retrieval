# 0002 — Preserve unicode in `clean_text`

**Stage:** 1 (foundation fixes) **Files touched:** `src/pdf_processing.py`

## What was wrong

```python
cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
```

This deleted **everything outside ASCII** before chunking. For arXiv ML
papers that means:

- **Greek letters** (`α`, `β`, `γ`, `λ`, `θ`, `μ`, `σ`) — central to ML
  notation. A query for "softmax over θ" can no longer match the paper
  introducing it.
- **Math operators** (`∇`, `∑`, `∫`, `∈`, `≈`, `≤`, `→`) — vanish from
  equations, leaving meaningless residue.
- **Accented author names** (`Schölkopf`, `Bengio`, `Lecun` with diacritics).
- **CJK / non-Latin** fragments embedded in references or affiliations.

The earlier code also did **not** handle PDF ligatures. PDF text extraction
commonly produces `ﬁ`, `ﬀ`, `ﬂ`, `ﬃ`, `ﬄ` (single codepoints
`U+FB01`–`U+FB04`). A query for "fine-tuning" then misses chunks containing
"ﬁne-tuning".

## Fix

```python
text = unicodedata.normalize("NFKC", text)   # ligatures → fi/ff/fl, etc.
text = _CONTROL_CHARS.sub(" ", text)         # strip ASCII control chars only
text = _WHITESPACE.sub(" ", text).strip()    # collapse whitespace
```

NFKC handles compatibility decomposition: ligatures, full-width digits,
super/subscripts, etc., all fold to their canonical forms while real
unicode payload (Greek, math, accents) is preserved.

The control-char regex keeps `\t \n \r` (the whitespace collapse handles
those) and removes the C0/C1 garbage that occasionally leaks out of PDF
streams.

## Verification

```python
>>> clean_text('The ﬁnal α-β model uses ∇f and Schölkopf et al. 2020.')
'The final α-β model uses ∇f and Schölkopf et al. 2020.'
```

## Re-index implication

Old chunks already indexed in Chroma were normalized with the broken cleaner
and have ASCII-only payload. They will continue to work but *won't benefit
from the fix*. New PDFs added going forward will be cleaned correctly.

A full re-index (delete the Chroma persist dir + `processed_pdfs.txt`, then
re-run "Process PDFs") will retroactively apply the fix to existing
documents. We'll formalize a re-index command when 0005 (config / CLI) lands.
