"""Tests for `clean_text` — see docs/changes/0002."""

from __future__ import annotations

from src.pdf_processing import clean_text


def test_preserves_greek_and_math():
    out = clean_text("The α-β model uses ∇f and Σ over θ.")
    assert "α" in out
    assert "β" in out
    assert "∇" in out
    assert "Σ" in out
    assert "θ" in out


def test_preserves_accented_names():
    out = clean_text("Schölkopf, Bengio, and Lecun.")
    assert "Schölkopf" in out


def test_folds_pdf_ligatures_via_nfkc():
    # Common PDF ligatures: ﬁ ﬀ ﬂ ﬃ ﬄ
    assert clean_text("ﬁne-tuning") == "fine-tuning"
    assert clean_text("eﬀective") == "effective"
    assert clean_text("ﬂoating") == "floating"


def test_strips_ascii_control_chars():
    raw = "foo\x00bar\x07baz"
    assert clean_text(raw) == "foo bar baz"


def test_collapses_whitespace_and_newlines():
    raw = "lots\t\tof   white\n\nspace"
    assert clean_text(raw) == "lots of white space"


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
