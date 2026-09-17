"""Reciprocal rank fusion of the keyword and dense rankings.

Keyword search scores a chunk with BM25, which is unbounded and depends on
the number of query terms and on the statistics of the corpus. Dense search
scores it with a cosine similarity, which is confined to [-1, 1] and in
practice to a band a few tenths wide. The two are on unrelated scales, so
adding them, or normalising each list first, means choosing weights that
would have to be tuned again for every corpus and every model.

Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009) ignores the
scores and uses only the ranks: a chunk earns ``1 / (k + rank)`` from every
list that returns it, and the sums are sorted. It needs no tuning. ``k``
damps the advantage of the very top ranks, and the paper's value of 60 works
across collections. A chunk that both searches rank highly beats one that a
single search ranks first, which is the behaviour hybrid retrieval is after.
"""

from collections.abc import Sequence
import math

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[tuple[int, float]]:
  """Merges best-first rankings of chunk ids into one.

  A chunk's score is the sum of ``1 / (k + rank)`` over the rankings that
  contain it, with ranks starting at 1. A ranking is read with repeats
  removed, so an id that occurs twice counts once, at its better rank. Empty
  rankings contribute nothing.

  Args:
    rankings: Best-first lists of chunk ids, one per search.
    k: The damping constant; larger values flatten the difference between
      adjacent ranks.
    limit: Keep at most this many results; None keeps them all.

  Returns:
    ``(chunk_id, score)`` pairs, best first. Equal scores are ordered by
    ascending chunk id, so the result is deterministic.

  Raises:
    ValueError: If ``k`` is not positive.
  """
  if k <= 0:
    raise ValueError(f"k must be positive, got {k}")

  contributions: dict[int, list[float]] = {}
  for ranking in rankings:
    for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
      contributions.setdefault(chunk_id, []).append(1.0 / (k + rank))

  # fsum adds without intermediate rounding, so the same set of ranks gives
  # the same score in whatever order the rankings came. With three or more
  # rankings a running float sum would let rounding, not the chunk id, decide
  # between tied chunks.
  scores = {
      chunk_id: math.fsum(parts) for chunk_id, parts in contributions.items()
  }
  fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
  return fused if limit is None else fused[: max(limit, 0)]
