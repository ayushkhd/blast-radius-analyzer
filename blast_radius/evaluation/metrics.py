"""Evaluation metrics: pure functions over ids and numbers.

The evaluation runner scores every question with the functions here and
averages the scores into one table. Nothing in this module knows about the
pipeline. A metric takes the ids that a stage returned and the ids that were
expected, so each can be checked by hand, and the caller decides what an id
is: a chunk, a CVE, a QID or a host.

Two things matter once scores are averaged over some forty questions, and
both are decided here rather than left to the caller. First, what a metric
is worth when its ratio is 0 / 0; each function states its convention.
Second, how a mean is shown: a rate over forty questions moves by 0.025 per
question, so ``Mean`` always renders with its count.
"""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import math


def _require_positive(k: int) -> None:
  """Raises ``ValueError`` unless the cut-off ``k`` is positive."""
  if k <= 0:
    raise ValueError(f"k must be positive, got {k}")


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
  """Returns 1.0 if any gold id is among the first ``k`` ranked, else 0.0.

  This is a hit rate, not the share of the gold ids that were found. Sibling
  CVEs under one QID lead to the same hosts, so one of them in the context
  is worth as much as all of them. Whether a question that spans several
  QIDs found every one of them shows in host recall instead.

  Args:
    ranked: The ids a retrieval stage returned, best first.
    gold: The ids that answer the question.
    k: How many of ``ranked`` to consider.

  Raises:
    ValueError: If ``k`` is not positive, or if ``gold`` is empty. A
      question without gold ids is a negative. It could never score above
      0 here, so it is scored by abstention and must stay out of this mean.
  """
  _require_positive(k)
  if not gold:
    raise ValueError("recall is undefined without gold ids")
  gold_ids = set(gold)
  return 1.0 if any(item in gold_ids for item in ranked[:k]) else 0.0


def precision_at_k(
    ranked: Sequence[str], gold: Collection[str], k: int
) -> float:
  """Returns the share of the first ``k`` ranked ids that are gold.

  The denominator is the number of ids considered, ``min(k, len(ranked))``,
  not ``k``: a stage that returns three candidates, all of them right, has
  a precision of 1, and is not marked down for slots it never filled. When
  nothing was ranked the result is 0.0, because an empty context is never
  what a question with gold ids wanted.

  Args:
    ranked: The ids a retrieval stage returned, best first, each id once.
    gold: The ids that answer the question. When it is empty nothing is
      gold and the result is 0.0, so negatives belong in this mean no more
      than in recall's.
    k: How many of ``ranked`` to consider.

  Raises:
    ValueError: If ``k`` is not positive.
  """
  _require_positive(k)
  considered = ranked[:k]
  if not considered:
    return 0.0
  gold_ids = set(gold)
  return sum(1 for item in considered if item in gold_ids) / len(considered)


def reciprocal_rank(ranked: Sequence[str], gold: Collection[str]) -> float:
  """Returns 1 / rank of the first gold id in ``ranked``, or 0.0 without one.

  Ranks are 1-based: a gold id in first place scores 1.0 and in fourth
  place 0.25. The mean of this over a question set is the MRR.

  Args:
    ranked: The ids a retrieval stage returned, best first.
    gold: The ids that answer the question. When it is empty the result is
      0.0, as for precision.
  """
  gold_ids = set(gold)
  for rank, item in enumerate(ranked, start=1):
    if item in gold_ids:
      return 1.0 / rank
  return 0.0


def set_precision_recall(
    returned: Collection[str], gold: Collection[str]
) -> tuple[float, float]:
  """Returns the precision and recall of a returned set against a gold set.

  For host sets, where order does not count. Duplicates are ignored. When
  a set is empty a ratio becomes 0 / 0, and it then takes the value that
  the answer deserves:

    returned   gold        precision, recall
    empty      empty       1.0, 1.0   nothing to find, nothing claimed
    empty      non-empty   0.0, 0.0   every host was missed
    non-empty  empty       0.0, 0.0   every host returned is a false alarm
    non-empty  non-empty   hits / returned, hits / gold

  The usual alternative scores an empty answer as perfectly precise, since
  nothing it returned was wrong. Averaged over a question set, that lets a
  configuration raise its precision by answering less. Here a wrong answer
  can never raise a mean. The price is that an empty answer lowers both
  columns and not recall alone, so how often that happened has to be read
  from the abstention metric, which counts exactly that.

  Args:
    returned: The ids the system returned.
    gold: The ids it should have returned.
  """
  returned_ids = set(returned)
  gold_ids = set(gold)
  if not returned_ids and not gold_ids:
    return 1.0, 1.0
  if not returned_ids or not gold_ids:
    return 0.0, 0.0
  hits = len(returned_ids & gold_ids)
  return hits / len(returned_ids), hits / len(gold_ids)


def percentile(values: Sequence[float], p: float) -> float:
  """Returns the ``p``-th percentile of ``values`` by linear interpolation.

  The sorted values are treated as evenly spaced from the 0th to the 100th
  percentile, and a percentile that falls between two of them is
  interpolated. With ``[10, 20, 30, 40]``:

    p = 50   ->  halfway between 20 and 30   = 25.0
    p = 95   ->  85% of the way from 30 to 40 = 38.5

  Args:
    values: The observations, in any order.
    p: The percentile, from 0 (the minimum) to 100 (the maximum).

  Raises:
    ValueError: If ``values`` is empty or ``p`` is outside 0-100.
  """
  if not values:
    raise ValueError("percentile is undefined without values")
  if not 0 <= p <= 100:
    raise ValueError(f"p must be between 0 and 100, got {p}")
  ordered = sorted(values)
  position = (len(ordered) - 1) * p / 100
  below = math.floor(position)
  above = math.ceil(position)
  weight = position - below
  return ordered[below] + (ordered[above] - ordered[below]) * weight


@dataclasses.dataclass(frozen=True)
class Mean:
  """The mean of one metric over a question set, kept with its count.

  Attributes:
    total: Sum of the per-question scores.
    count: Number of questions scored.
    binary: Whether every score is 0 or 1, so that ``total`` is a number of
      successes.
  """

  total: float
  count: int
  binary: bool = False

  @property
  def value(self) -> float:
    """Returns the mean, or 0.0 when nothing was scored."""
    return self.total / self.count if self.count else 0.0

  def __str__(self) -> str:
    """Returns the mean with its count, e.g. ``0.93 (37/40)``.

    A 0/1 metric shows successes over questions. Any other metric shows
    ``0.93 (n=40)``, because its total means nothing to a reader.
    """
    if self.binary:
      return f"{self.value:.2f} ({self.total:.0f}/{self.count})"
    return f"{self.value:.2f} (n={self.count})"


def mean_of(values: Iterable[float], *, binary: bool) -> Mean:
  """Returns the mean of ``values``.

  Args:
    values: One score per question.
    binary: Whether the metric is a 0/1 outcome, such as recall@k or an
      abstention, which decides how the mean is rendered.

  Raises:
    ValueError: If ``binary`` is set and a value is neither 0 nor 1: the
      rendered count of successes would be wrong.
  """
  scores = list(values)
  if binary:
    for score in scores:
      if score not in (0.0, 1.0):
        raise ValueError(f"a binary metric scored {score}, not 0 or 1")
  return Mean(total=math.fsum(scores), count=len(scores), binary=binary)
