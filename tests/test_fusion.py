"""Tests for blast_radius.retrieval.fusion."""

import itertools
import random

import pytest

from blast_radius.retrieval import fusion


def test_reciprocal_rank_fusion_sums_reciprocal_ranks_across_rankings():
  rankings = [[1, 2, 3], [3, 1]]

  fused = fusion.reciprocal_rank_fusion(rankings, k=60)

  assert [chunk_id for chunk_id, _ in fused] == [1, 3, 2]
  assert dict(fused) == pytest.approx(
      {1: 1 / 61 + 1 / 62, 3: 1 / 63 + 1 / 61, 2: 1 / 62}
  )


def test_reciprocal_rank_fusion_single_ranking_keeps_its_order():
  ranking = list(range(100))
  random.Random(7).shuffle(ranking)

  fused = fusion.reciprocal_rank_fusion([ranking])

  assert [chunk_id for chunk_id, _ in fused] == ranking


def test_reciprocal_rank_fusion_chunk_in_both_rankings_beats_the_top_of_one():
  keyword_ranking = [1, 2]
  dense_ranking = [3, 2]

  fused = fusion.reciprocal_rank_fusion([keyword_ranking, dense_ranking])

  assert fused[0][0] == 2


def test_reciprocal_rank_fusion_smaller_k_rewards_the_top_ranks_more():
  fused = fusion.reciprocal_rank_fusion([[8, 9]], k=1)

  assert fused == [(8, 1 / 2), (9, 1 / 3)]


def test_reciprocal_rank_fusion_equal_scores_are_ordered_by_chunk_id():
  fused = fusion.reciprocal_rank_fusion([[7, 5], [3, 9]])

  assert [chunk_id for chunk_id, _ in fused] == [3, 7, 5, 9]


def test_reciprocal_rank_fusion_same_ranks_in_any_order_tie_exactly():
  # Chunks 1, 2 and 3 each hold ranks 1, 2 and 5, met in a different order.
  # With k = 1 that is 1/2 + 1/3 + 1/6. A running float sum makes it
  # 0.9999999999999999 for chunk 2 alone, which would rank it last.
  rankings = [
      [1, 2, 101, 102, 3],
      [2, 3, 201, 202, 1],
      [3, 1, 301, 302, 2],
  ]

  fused = fusion.reciprocal_rank_fusion(rankings, k=1, limit=3)

  assert fused == [(1, 1.0), (2, 1.0), (3, 1.0)]


def test_reciprocal_rank_fusion_does_not_depend_on_the_order_of_rankings():
  rng = random.Random(11)
  rankings = [rng.sample(range(40), 25) for _ in range(4)]

  results = [
      fusion.reciprocal_rank_fusion(list(permutation))
      for permutation in itertools.permutations(rankings)
  ]

  assert all(result == results[0] for result in results)


def test_reciprocal_rank_fusion_repeated_id_counts_once_at_its_best_rank():
  with_repeat = fusion.reciprocal_rank_fusion([[4, 9, 4]])
  without = fusion.reciprocal_rank_fusion([[4, 9]])

  assert with_repeat == without


def test_reciprocal_rank_fusion_ids_after_a_repeat_move_up_a_rank():
  fused = fusion.reciprocal_rank_fusion([[4, 4, 9]], k=60)

  assert fused == [(4, 1 / 61), (9, 1 / 62)]


def test_reciprocal_rank_fusion_of_no_rankings_is_empty():
  assert not fusion.reciprocal_rank_fusion([])


def test_reciprocal_rank_fusion_ignores_empty_rankings():
  fused = fusion.reciprocal_rank_fusion([[], [6], []], k=60)

  assert fused == [(6, 1 / 61)]


def test_reciprocal_rank_fusion_limit_keeps_the_best():
  fused = fusion.reciprocal_rank_fusion([[1, 2, 3], [2, 3, 4]], limit=2)

  assert [chunk_id for chunk_id, _ in fused] == [2, 3]


def test_reciprocal_rank_fusion_limit_larger_than_the_result_keeps_it_all():
  fused = fusion.reciprocal_rank_fusion([[1, 2]], limit=10)

  assert [chunk_id for chunk_id, _ in fused] == [1, 2]


@pytest.mark.parametrize("limit", [0, -1])
def test_reciprocal_rank_fusion_limit_of_zero_or_less_is_empty(limit: int):
  assert not fusion.reciprocal_rank_fusion([[1, 2]], limit=limit)


@pytest.mark.parametrize("k", [0, -60])
def test_reciprocal_rank_fusion_k_not_positive_is_rejected(k: int):
  with pytest.raises(ValueError):
    fusion.reciprocal_rank_fusion([[1, 2]], k=k)
