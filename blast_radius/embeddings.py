"""Text embedders for dense retrieval.

``Embedder`` is the interface ingest and the retriever depend on. Two
implementations follow it:

* ``FastEmbedEmbedder`` runs a small sentence-embedding model on CPU through
  fastembed and ONNX Runtime. It is what production uses.
* ``HashingEmbedder`` hashes word n-grams into a fixed-size vector. It needs
  no model download, is fully deterministic, and is good enough for tests
  and for running the service where the model cannot be fetched.

Both return L2-normalised float32 vectors, so a dot product is a cosine
similarity.
"""

from collections.abc import Iterable, Sequence
import pathlib
import re
from typing import Any, Protocol
import zlib

import numpy as np
import numpy.typing as npt

from blast_radius import config

Matrix = npt.NDArray[np.float32]
Vector = npt.NDArray[np.float32]

# Passages per ONNX Runtime call. fastembed pads a batch to its longest
# member, and on CPU the padding and the batch-sized attention buffers cost
# more than batching saves. Measured on the reference corpus on a laptop:
# one passage at a time embeds 29 chunks a second within 0.3 GB, 32 at a
# time 10 a second within 0.9 GB, and 64 at a time 9 a second within 1.5 GB.
# fastembed's own default is 256.
DOCUMENT_BATCH_SIZE = 1

_WORD = re.compile(r"[a-z0-9_]+")
_HASHING_PREFIX = "hashing-"
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")
_DIM_PROBE = "dimension probe"


class Embedder(Protocol):
  """Turns text into unit-length vectors.

  Attributes:
    name: Model name recorded in the artifact, so that a query is never
      embedded with a different model from the one that built the index.
    dim: Vector dimensionality.
  """

  name: str
  dim: int

  def embed_documents(self, texts: Sequence[str]) -> Matrix:
    """Returns one row per text, shape ``(len(texts), dim)``.

    Args:
      texts: Passages to index.
    """

  def embed_query(self, text: str) -> Vector:
    """Returns the vector for a search query, shape ``(dim,)``.

    Queries and documents are embedded differently by models trained with
    an instruction prefix, which is why this is a separate method.

    Args:
      text: The search query.
    """


class HashingEmbedder:
  """A model-free embedder that hashes words and word pairs into buckets.

  Each unigram and bigram adds or subtracts one from a bucket chosen by a
  stable hash (the signed "hashing trick"), and the result is normalised.
  Texts that share vocabulary get a high cosine similarity, which is all
  that tests and an offline smoke run need. It captures no meaning beyond
  word overlap and is not meant to be evaluated.

  Attributes:
    name: ``hashing-<dim>``, recorded in the artifact like any model name.
    dim: Number of buckets.
  """

  def __init__(self, dim: int = 256) -> None:
    """Initialises the embedder.

    Args:
      dim: Number of buckets, and so the dimensionality of every vector.

    Raises:
      ValueError: If ``dim`` is not positive.
    """
    if dim <= 0:
      raise ValueError(f"dim must be positive, got {dim}")
    self.name = f"hashing-{dim}"
    self.dim = dim

  def _embed(self, text: str) -> Vector:
    """Returns the unit vector for ``text``, or zeros if it has no words."""
    words = _WORD.findall(text.lower())
    features = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    vector = np.zeros(self.dim, dtype=np.float32)
    for feature in features:
      # crc32 is stable across processes, unlike the built-in hash().
      digest = zlib.crc32(feature.encode("utf-8"))
      sign = 1.0 if digest & 1 else -1.0
      vector[(digest >> 1) % self.dim] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector

  def embed_documents(self, texts: Sequence[str]) -> Matrix:
    """Returns one row per text, shape ``(len(texts), dim)``.

    Args:
      texts: Passages to index.
    """
    if not texts:
      return np.zeros((0, self.dim), dtype=np.float32)
    return np.stack([self._embed(text) for text in texts])

  def embed_query(self, text: str) -> Vector:
    """Returns the vector for a search query, shape ``(dim,)``.

    Args:
      text: The search query.
    """
    return self._embed(text)


class TextEmbeddingBackend(Protocol):
  """The part of ``fastembed.TextEmbedding`` that ``FastEmbedEmbedder`` uses.

  Naming it lets a test hand in a fake with these two methods and load no
  model.
  """

  def passage_embed(
      self, texts: Iterable[str], *, batch_size: int
  ) -> Iterable[npt.NDArray[Any]]:
    """Yields one vector per passage, in the order given.

    Args:
      texts: Passages to embed.
      batch_size: Passages that go through the model in one call.
    """

  def query_embed(self, query: str) -> Iterable[npt.NDArray[Any]]:
    """Yields the vector of one query.

    fastembed keeps this apart from ``passage_embed`` so that a model which
    embeds queries differently, with an instruction prefix say, can do so.

    Args:
      query: The search query.
    """


def _unit_rows(vectors: Iterable[npt.NDArray[Any]]) -> Matrix:
  """Returns ``vectors`` stacked as float32 rows of unit length.

  The model's own normalisation is not relied on: the dense index takes dot
  products for cosines, which is only true of unit vectors. A zero vector
  stays zero instead of becoming NaN.

  Args:
    vectors: Equal-length vectors, at least one.
  """
  matrix = np.stack(list(vectors)).astype(np.float32)
  norms = np.linalg.norm(matrix, axis=1, keepdims=True)
  norms[norms == 0.0] = 1.0
  return matrix / norms


class FastEmbedEmbedder:
  """A sentence-embedding model run on CPU through fastembed.

  Attributes:
    name: The fastembed model name, e.g. ``BAAI/bge-small-en-v1.5``.
    dim: Vector dimensionality, measured from the model itself.
  """

  def __init__(
      self,
      model_name: str,
      cache_dir: pathlib.Path,
      *,
      backend: TextEmbeddingBackend | None = None,
  ) -> None:
    """Initialises the embedder, loading the model unless one is handed in.

    Args:
      model_name: A model name that fastembed knows.
      cache_dir: Where the model files are cached. They are downloaded on
        first use and read from here afterwards.
      backend: A ready model to use instead of loading ``model_name``.
    """
    if backend is None:
      # fastembed brings in ONNX Runtime, so it is imported only when a model
      # is really loaded: importing this module, and every test that hands
      # in a backend, stays free of it.
      # Lazy on purpose, see above. pylint: disable-next=import-outside-toplevel
      import fastembed

      backend = fastembed.TextEmbedding(
          model_name=model_name, cache_dir=str(cache_dir)
      )
    self.name = model_name
    self._backend = backend
    self.dim = int(self.embed_query(_DIM_PROBE).shape[0])

  def embed_documents(self, texts: Sequence[str]) -> Matrix:
    """Returns one row per text, shape ``(len(texts), dim)``.

    Args:
      texts: Passages to index.

    Raises:
      RuntimeError: If the model does not return one vector per text.
    """
    if not texts:
      return np.zeros((0, self.dim), dtype=np.float32)
    matrix = _unit_rows(
        self._backend.passage_embed(texts, batch_size=DOCUMENT_BATCH_SIZE)
    )
    if matrix.shape != (len(texts), self.dim):
      raise RuntimeError(
          f"{self.name} returned shape {matrix.shape} for {len(texts)} texts"
          f" of dimension {self.dim}"
      )
    return matrix

  def embed_query(self, text: str) -> Vector:
    """Returns the vector for a search query, shape ``(dim,)``.

    Args:
      text: The search query.
    """
    vector: Vector = _unit_rows(self._backend.query_embed(text))[0]
    return vector


def create(settings: config.Settings) -> Embedder:
  """Returns the embedder that ``settings.embedding_model`` names.

  A name of the form ``hashing-<dim>``, which is how ``HashingEmbedder``
  reports itself, selects that embedder, so the service can run where no
  model can be downloaded. Any other name is loaded through fastembed.

  Args:
    settings: Supplies the model name and the model cache directory.

  Raises:
    ValueError: If the name starts with ``hashing-`` but does not end in a
      positive dimension.
  """
  name = settings.embedding_model
  if not name.startswith(_HASHING_PREFIX):
    return FastEmbedEmbedder(name, settings.model_cache_dir)
  dim = name.removeprefix(_HASHING_PREFIX)
  # No sign, space or leading zero is accepted: the artifact records the name
  # that the embedder reports, and that has to be the name in the settings.
  if not _POSITIVE_INTEGER.fullmatch(dim):
    raise ValueError(
        f"embedding model {name!r} is malformed: expected hashing-<dim> with"
        " a positive dimension, e.g. hashing-256"
    )
  return HashingEmbedder(int(dim))
