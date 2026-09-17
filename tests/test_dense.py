"""Tests for blast_radius.retrieval.dense."""

import numpy as np
import pytest

from blast_radius import embeddings
from blast_radius.retrieval import dense


def _index(vectors: dict[int, list[float]]) -> dense.DenseIndex:
  """Returns an index that maps each chunk id to its vector."""
  matrix = np.array(list(vectors.values()), dtype=np.float32)
  return dense.DenseIndex(list(vectors), matrix)


def _query(values: list[float]) -> embeddings.Vector:
  return np.array(values, dtype=np.float32)


def test_search_returns_chunk_ids_with_cosines_best_first():
  index = _index({10: [1.0, 0.0], 20: [0.0, 1.0], 30: [0.6, 0.8]})

  hits = index.search(_query([1.0, 0.0]), 3)

  assert [chunk_id for chunk_id, _ in hits] == [10, 30, 20]
  assert [score for _, score in hits] == pytest.approx([1.0, 0.6, 0.0])


def test_search_keeps_only_the_best_up_to_the_limit():
  index = _index({10: [1.0, 0.0], 20: [0.0, 1.0], 30: [0.6, 0.8]})

  hits = index.search(_query([1.0, 0.0]), 2)

  assert [chunk_id for chunk_id, _ in hits] == [10, 30]


def test_search_limit_beyond_the_index_returns_every_chunk():
  index = _index({10: [1.0, 0.0], 20: [0.0, 1.0]})

  hits = index.search(_query([0.0, 1.0]), 50)

  assert [chunk_id for chunk_id, _ in hits] == [20, 10]


@pytest.mark.parametrize("limit", [0, -3])
def test_search_limit_of_zero_or_less_is_empty(limit: int):
  index = _index({10: [1.0, 0.0]})

  assert not index.search(_query([1.0, 0.0]), limit)


def test_search_of_an_empty_index_is_empty():
  index = dense.DenseIndex([], np.zeros((0, 2), dtype=np.float32))

  assert not index.search(_query([1.0, 0.0]), 5)


def test_search_equal_cosines_are_ordered_by_chunk_id():
  index = _index({9: [1.0, 0.0], 4: [1.0, 0.0], 7: [1.0, 0.0]})

  hits = index.search(_query([1.0, 0.0]), 3)

  assert [chunk_id for chunk_id, _ in hits] == [4, 7, 9]


def test_search_tie_at_the_limit_goes_to_the_lowest_chunk_ids():
  index = _index(
      {
          8: [0.6, 0.8],
          5: [1.0, 0.0],
          6: [0.6, 0.8],
          2: [0.6, 0.8],
          3: [0.6, 0.8],
      }
  )

  hits = index.search(_query([1.0, 0.0]), 3)

  assert [chunk_id for chunk_id, _ in hits] == [5, 2, 3]


def test_search_allowed_restricts_the_search_to_those_chunks():
  index = _index({10: [1.0, 0.0], 20: [0.0, 1.0], 30: [0.6, 0.8]})

  hits = index.search(_query([1.0, 0.0]), 3, allowed={20, 30})

  assert [chunk_id for chunk_id, _ in hits] == [30, 20]


def test_search_allowed_ignores_chunk_ids_that_are_not_indexed():
  index = _index({10: [1.0, 0.0], 30: [0.6, 0.8]})

  hits = index.search(_query([1.0, 0.0]), 3, allowed=[30, 999])

  assert [chunk_id for chunk_id, _ in hits] == [30]


def test_search_allowed_nothing_is_empty():
  index = _index({10: [1.0, 0.0]})

  assert not index.search(_query([1.0, 0.0]), 3, allowed=set())


@pytest.mark.parametrize("limit", [1, 5, 17, 60])
def test_search_with_allowed_equals_a_full_search_filtered_afterwards(
    limit: int,
):
  rng = np.random.default_rng(3)
  matrix = rng.normal(size=(60, 8)).astype(np.float32)
  matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
  # Every vector occurs twice, and every third row is allowed, so tied pairs
  # sit inside the allowed set, in the results and across the cut.
  matrix[30:] = matrix[:30]
  chunk_ids = [int(chunk_id) for chunk_id in rng.permutation(1000)[:60]]
  index = dense.DenseIndex(chunk_ids, matrix)
  allowed = set(chunk_ids[::3])

  hits = index.search(matrix[6], limit, allowed=allowed)

  everything = index.search(matrix[6], len(index))
  assert hits == [hit for hit in everything if hit[0] in allowed][:limit]


def test_search_finds_the_text_that_shares_the_querys_words():
  embedder = embeddings.HashingEmbedder(dim=512)
  texts = {
      1: "Traefik is a cloud native application proxy.",
      2: "OpenSSH up to version 9.6 allows an authentication bypass.",
      3: "nfs: handle error of rpc_proc_register() in nfs_net_init().",
  }
  index = dense.DenseIndex(
      list(texts), embedder.embed_documents(list(texts.values()))
  )

  hits = index.search(embedder.embed_query("OpenSSH authentication bypass"), 1)

  assert hits[0][0] == 2


def test_len_and_dim_describe_the_matrix():
  index = dense.DenseIndex([1, 2, 3], np.zeros((3, 5), dtype=np.float32))

  assert len(index) == 3
  assert index.dim == 5


@pytest.mark.parametrize(
    "query",
    [
        np.zeros(3, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((), dtype=np.float32),
    ],
)
def test_search_query_of_the_wrong_shape_is_rejected(query: embeddings.Vector):
  index = _index({10: [1.0, 0.0]})

  with pytest.raises(ValueError, match="shape"):
    index.search(query, 1)


def test_matrix_that_is_not_two_dimensional_is_rejected():
  with pytest.raises(ValueError, match="two-dimensional"):
    dense.DenseIndex([1, 2], np.zeros(2, dtype=np.float32))


def test_matrix_that_is_not_float32_is_rejected():
  with pytest.raises(ValueError, match="float32"):
    dense.DenseIndex([1], np.zeros((1, 2), dtype=np.float64))


def test_matrix_without_one_row_per_chunk_id_is_rejected():
  with pytest.raises(ValueError, match="2 rows for 3 chunk ids"):
    dense.DenseIndex([1, 2, 3], np.zeros((2, 4), dtype=np.float32))


def test_repeated_chunk_id_is_rejected():
  with pytest.raises(ValueError, match="unique"):
    dense.DenseIndex([1, 1], np.zeros((2, 4), dtype=np.float32))
