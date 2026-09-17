"""Exact nearest-neighbour search over the chunk embeddings.

The index is one float32 matrix held in memory, one unit-length row per
embedded chunk. Because the rows and the query are unit vectors, a single
matrix product gives every cosine similarity at once. At the size of this
corpus (hundreds of vectors, about a megabyte) that takes well under a
millisecond, has no approximation error and needs no vector database.
"""

from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt

from blast_radius import embeddings


class DenseIndex:
  """Chunk embeddings searchable by cosine similarity.

  Attributes:
    dim: Dimensionality of the indexed vectors.
  """

  def __init__(
      self, chunk_ids: Sequence[int], matrix: embeddings.Matrix
  ) -> None:
    """Initialises the index.

    Args:
      chunk_ids: The chunk id of each row of ``matrix``, in row order.
      matrix: Unit-length embeddings, shape ``(len(chunk_ids), dim)``.

    Raises:
      ValueError: If ``matrix`` is not a two-dimensional float32 array with
        one row per chunk id, or if a chunk id is repeated.
    """
    if matrix.ndim != 2:
      raise ValueError(
          f"matrix must be two-dimensional, got shape {matrix.shape}"
      )
    if matrix.dtype != np.float32:
      raise ValueError(f"matrix must be float32, got {matrix.dtype}")
    if matrix.shape[0] != len(chunk_ids):
      raise ValueError(
          f"matrix has {matrix.shape[0]} rows for {len(chunk_ids)} chunk ids"
      )
    if len(set(chunk_ids)) != len(chunk_ids):
      raise ValueError("chunk ids must be unique")
    self._chunk_ids: npt.NDArray[np.int64] = np.asarray(
        chunk_ids, dtype=np.int64
    )
    self._matrix = matrix
    self.dim = int(matrix.shape[1])

  def __len__(self) -> int:
    """Returns the number of indexed chunks."""
    return len(self._chunk_ids)

  def search(
      self,
      query_vector: embeddings.Vector,
      limit: int,
      *,
      allowed: Collection[int] | None = None,
  ) -> list[tuple[int, float]]:
    """Returns the chunks most similar to a query.

    Args:
      query_vector: Unit-length query embedding, shape ``(dim,)``.
      limit: Maximum number of results; zero or less returns none.
      allowed: If given, only these chunk ids are searched. Ids that are not
        in the index are ignored, so a caller may pass every chunk of a
        document without knowing which of them were embedded.

    Returns:
      ``(chunk_id, cosine)`` pairs, best first. Equal similarities are
      ordered by ascending chunk id, so the result is deterministic.

    Raises:
      ValueError: If ``query_vector`` does not have shape ``(dim,)``.
    """
    if query_vector.shape != (self.dim,):
      raise ValueError(
          f"query has shape {query_vector.shape}, the index holds vectors of"
          f" shape ({self.dim},)"
      )
    rows = self._rows(allowed)
    if limit <= 0 or not rows.size:
      return []

    scores = (self._matrix @ query_vector)[rows]
    if limit < rows.size:
      # Select before sorting. The cut is made at the limit-th best score
      # and keeps every row that ties with it: argpartition alone would
      # choose among those ties arbitrarily, and the choice has to fall to
      # the chunk id.
      cutoff = np.partition(scores, -limit)[-limit]
      kept = np.flatnonzero(scores >= cutoff)
      rows, scores = rows[kept], scores[kept]
    order = np.lexsort((self._chunk_ids[rows], -scores))[:limit]
    return [(int(self._chunk_ids[rows[i]]), float(scores[i])) for i in order]

  def _rows(self, allowed: Collection[int] | None) -> npt.NDArray[np.intp]:
    """Returns the row numbers a search may consider.

    Args:
      allowed: Chunk ids to restrict the search to, or None for every row.
    """
    if allowed is None:
      return np.arange(len(self))
    wanted = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
    return np.flatnonzero(np.isin(self._chunk_ids, wanted))
