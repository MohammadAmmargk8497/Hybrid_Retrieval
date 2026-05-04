"""Pure tests for the RRF math — no Chroma, no bm25s required."""

from __future__ import annotations

import math

from src.search import reciprocal_rank_fusion


def test_rrf_single_list_matches_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"]], k=60))
    assert math.isclose(fused["a"], 1 / 61)
    assert math.isclose(fused["b"], 1 / 62)
    assert math.isclose(fused["c"], 1 / 63)


def test_rrf_sums_contributions_across_lists():
    # "x" appears at rank 0 in both lists → 2 * 1/61
    # "y" appears at rank 1 in list 1 only → 1/62
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["x", "z"]], k=60))
    assert math.isclose(fused["x"], 2 / 61)
    assert math.isclose(fused["y"], 1 / 62)
    assert math.isclose(fused["z"], 1 / 62)


def test_rrf_returns_sorted_descending():
    out = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    # tie expected: a and b each get 1/61 + 1/62
    expected = 1 / 61 + 1 / 62
    assert all(math.isclose(s, expected) for s in scores)


def test_rrf_works_with_int_ids():
    """RRF is generic over hashable id types — int corpus indices must work."""
    fused = dict(reciprocal_rank_fusion([[5, 3, 1], [3, 5, 4]], k=60))
    # 5 is at ranks 0 and 1 → 1/61 + 1/62
    # 3 is at ranks 1 and 0 → 1/62 + 1/61  (tie with 5)
    assert math.isclose(fused[5], fused[3])
    assert fused[5] > fused[1]


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
