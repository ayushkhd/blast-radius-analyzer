"""Tests for blast_radius.evaluation.metrics."""

import dataclasses

import pytest

from blast_radius.evaluation import metrics

_RANKED = ("a", "b", "c", "d", "e", "f")


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_with_a_gold_id_in_the_top_k_is_one():
  assert metrics.recall_at_k(_RANKED, {"c"}, 3) == 1.0


def test_recall_at_k_with_the_only_gold_id_just_below_k_is_zero():
  assert metrics.recall_at_k(_RANKED, {"d"}, 3) == 0.0


def test_recall_at_k_with_one_of_several_gold_ids_found_is_still_one():
  assert metrics.recall_at_k(_RANKED, {"a", "x", "y"}, 3) == 1.0


def test_recall_at_k_with_no_gold_id_ranked_is_zero():
  assert metrics.recall_at_k(_RANKED, {"x"}, 5) == 0.0


def test_recall_at_k_with_nothing_ranked_is_zero():
  assert metrics.recall_at_k([], {"a"}, 5) == 0.0


def test_recall_at_k_with_k_beyond_the_ranking_considers_all_of_it():
  assert metrics.recall_at_k(_RANKED, {"f"}, 50) == 1.0


def test_recall_at_k_without_gold_ids_raises():
  with pytest.raises(ValueError, match="gold"):
    metrics.recall_at_k(_RANKED, set(), 5)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_with_k_not_positive_raises(k: int):
  with pytest.raises(ValueError, match="k must be positive"):
    metrics.recall_at_k(_RANKED, {"a"}, k)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_is_the_gold_share_of_the_top_k():
  assert metrics.precision_at_k(_RANKED, {"a", "c", "f"}, 5) == 0.4


def test_precision_at_k_ignores_gold_ids_below_k():
  assert metrics.precision_at_k(_RANKED, {"e", "f"}, 4) == 0.0


def test_precision_at_k_with_fewer_than_k_ranked_divides_by_what_was_ranked():
  assert metrics.precision_at_k(["a", "b", "x"], {"a", "b"}, 5) == (
      pytest.approx(2 / 3)
  )


def test_precision_at_k_with_nothing_ranked_is_zero():
  assert metrics.precision_at_k([], {"a"}, 5) == 0.0


def test_precision_at_k_without_gold_ids_is_zero():
  assert metrics.precision_at_k(_RANKED, set(), 5) == 0.0


def test_precision_at_k_accepts_gold_ids_as_a_list():
  assert metrics.precision_at_k(_RANKED, ["b", "a"], 2) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_with_k_not_positive_raises(k: int):
  with pytest.raises(ValueError, match="k must be positive"):
    metrics.precision_at_k(_RANKED, {"a"}, k)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_of_a_gold_id_in_first_place_is_one():
  assert metrics.reciprocal_rank(_RANKED, {"a"}) == 1.0


def test_reciprocal_rank_of_a_gold_id_in_fourth_place_is_a_quarter():
  assert metrics.reciprocal_rank(_RANKED, {"d"}) == 0.25


def test_reciprocal_rank_uses_the_first_of_several_gold_ids():
  assert metrics.reciprocal_rank(_RANKED, {"e", "b"}) == 0.5


def test_reciprocal_rank_without_a_gold_id_ranked_is_zero():
  assert metrics.reciprocal_rank(_RANKED, {"x"}) == 0.0


def test_reciprocal_rank_with_nothing_ranked_is_zero():
  assert metrics.reciprocal_rank([], {"a"}) == 0.0


def test_reciprocal_rank_without_gold_ids_is_zero():
  assert metrics.reciprocal_rank(_RANKED, set()) == 0.0


# ---------------------------------------------------------------------------
# set_precision_recall
# ---------------------------------------------------------------------------


def test_set_precision_recall_of_overlapping_sets():
  returned = {"h1", "h2", "h3", "h4"}
  gold = {"h3", "h4", "h5"}

  precision, recall = metrics.set_precision_recall(returned, gold)

  assert precision == 0.5
  assert recall == pytest.approx(2 / 3)


def test_set_precision_recall_of_equal_sets_is_perfect():
  assert metrics.set_precision_recall({"h1", "h2"}, {"h2", "h1"}) == (1.0, 1.0)


def test_set_precision_recall_of_disjoint_sets_is_zero():
  assert metrics.set_precision_recall({"h1"}, {"h2"}) == (0.0, 0.0)


def test_set_precision_recall_of_a_superset_loses_precision_only():
  assert metrics.set_precision_recall({"h1", "h2"}, {"h1"}) == (0.5, 1.0)


def test_set_precision_recall_of_a_subset_loses_recall_only():
  assert metrics.set_precision_recall({"h1"}, {"h1", "h2"}) == (1.0, 0.5)


def test_set_precision_recall_with_both_sets_empty_is_perfect():
  assert metrics.set_precision_recall(set(), set()) == (1.0, 1.0)


def test_set_precision_recall_with_nothing_returned_for_gold_hosts_is_zero():
  assert metrics.set_precision_recall(set(), {"h1", "h2"}) == (0.0, 0.0)


def test_set_precision_recall_with_hosts_returned_for_no_gold_is_zero():
  assert metrics.set_precision_recall({"h1"}, set()) == (0.0, 0.0)


def test_set_precision_recall_ignores_duplicate_ids():
  returned = ["h1", "h1", "h2"]
  gold = ["h1", "h3", "h3"]

  assert metrics.set_precision_recall(returned, gold) == (0.5, 0.5)


def test_set_precision_recall_never_lets_an_empty_answer_raise_the_mean():
  gold = {"h1", "h2"}
  half_right = metrics.set_precision_recall({"h1", "h9"}, gold)
  empty = metrics.set_precision_recall(set(), gold)

  precision = metrics.mean_of([half_right[0], empty[0]], binary=False)

  assert precision.value < half_right[0]


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0, 10.0), (25, 17.5), (50, 25.0), (95, 38.5), (100, 40.0)],
)
def test_percentile_interpolates_linearly_between_values(
    p: float, expected: float
):
  assert metrics.percentile([10.0, 20.0, 30.0, 40.0], p) == (
      pytest.approx(expected)
  )


def test_percentile_that_lands_on_a_value_returns_it_exactly():
  assert metrics.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 75) == 4.0


def test_percentile_does_not_depend_on_the_input_order():
  assert metrics.percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0


def test_percentile_leaves_the_input_unsorted():
  values = [3.0, 1.0, 2.0]

  metrics.percentile(values, 50)

  assert values == [3.0, 1.0, 2.0]


@pytest.mark.parametrize("p", [0, 37.5, 100])
def test_percentile_of_a_single_value_is_that_value(p: float):
  assert metrics.percentile([7.0], p) == 7.0


def test_percentile_of_no_values_raises():
  with pytest.raises(ValueError, match="without values"):
    metrics.percentile([], 50)


@pytest.mark.parametrize("p", [-0.1, 100.1, float("nan")])
def test_percentile_with_p_out_of_range_raises(p: float):
  with pytest.raises(ValueError, match="between 0 and 100"):
    metrics.percentile([1.0, 2.0], p)


# ---------------------------------------------------------------------------
# Mean and mean_of
# ---------------------------------------------------------------------------


def test_mean_value_is_the_total_over_the_count():
  assert metrics.Mean(total=3.0, count=4).value == 0.75


def test_mean_value_of_nothing_is_zero():
  assert metrics.Mean(total=0.0, count=0).value == 0.0


def test_mean_of_a_binary_metric_renders_successes_over_questions():
  mean = metrics.mean_of([1.0] * 37 + [0.0] * 3, binary=True)

  assert mean == metrics.Mean(total=37.0, count=40, binary=True)
  assert str(mean) == "0.93 (37/40)"


def test_mean_of_a_graded_metric_renders_the_count_alone():
  mean = metrics.mean_of([1.0, 0.5, 0.25, 0.25], binary=False)

  assert mean == metrics.Mean(total=2.0, count=4, binary=False)
  assert str(mean) == "0.50 (n=4)"


def test_mean_of_nothing_renders_a_zero_count():
  assert str(metrics.mean_of([], binary=True)) == "0.00 (0/0)"
  assert str(metrics.mean_of([], binary=False)) == "0.00 (n=0)"


def test_mean_of_accepts_a_generator():
  mean = metrics.mean_of((score for score in (1.0, 0.0)), binary=True)

  assert str(mean) == "0.50 (1/2)"


def test_mean_of_sums_without_accumulating_rounding_error():
  assert metrics.mean_of([0.1] * 10, binary=False).value == 0.1


@pytest.mark.parametrize("score", [0.5, 2.0, -1.0, float("nan")])
def test_mean_of_a_binary_metric_with_another_score_raises(score: float):
  with pytest.raises(ValueError, match="binary"):
    metrics.mean_of([1.0, score], binary=True)


def test_mean_is_immutable():
  mean = metrics.mean_of([1.0], binary=True)

  with pytest.raises(dataclasses.FrozenInstanceError):
    mean.count = 2
