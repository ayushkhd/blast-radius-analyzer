"""Tests for blast_radius.embeddings.

No test loads a model: ``FastEmbedEmbedder`` is given a fake backend, or a
fake ``fastembed`` module where the point is how the real one is loaded.
"""

from collections.abc import Iterable, Iterator
import pathlib
import subprocess
import sys
import types
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from blast_radius import config
from blast_radius import embeddings

_MODEL = "BAAI/bge-small-en-v1.5"


class _FakeBackend:
  """Stands in for ``fastembed.TextEmbedding`` and records how it is used.

  Vectors are float64 and not of unit length, so that a test can see the
  embedder convert and normalise them. A text with no vector of its own,
  such as the dimension probe, embeds to all ones.
  """

  def __init__(
      self,
      passages: dict[str, list[float]] | None = None,
      queries: dict[str, list[float]] | None = None,
      dim: int = 2,
  ) -> None:
    self._passages = passages or {}
    self._queries = queries or {}
    self._dim = dim
    self.passage_calls: list[tuple[list[str], int]] = []
    self.query_calls: list[str] = []

  def _vector(
      self, vectors: dict[str, list[float]], text: str
  ) -> npt.NDArray[np.float64]:
    return np.array(vectors.get(text, [1.0] * self._dim), dtype=np.float64)

  def passage_embed(
      self, texts: Iterable[str], *, batch_size: int
  ) -> Iterator[npt.NDArray[np.float64]]:
    texts = list(texts)
    self.passage_calls.append((texts, batch_size))
    for text in texts:
      yield self._vector(self._passages, text)

  def query_embed(self, query: str) -> Iterator[npt.NDArray[np.float64]]:
    self.query_calls.append(query)
    yield self._vector(self._queries, query)


class _ShortBackend(_FakeBackend):
  """A backend that loses the vector of the last passage."""

  def passage_embed(
      self, texts: Iterable[str], *, batch_size: int
  ) -> Iterator[npt.NDArray[np.float64]]:
    yield from list(super().passage_embed(texts, batch_size=batch_size))[:-1]


def _fake_fastembed(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend
) -> list[dict[str, Any]]:
  """Replaces the ``fastembed`` module with one that hands out ``backend``.

  Args:
    monkeypatch: Undoes the replacement when the test ends.
    backend: What the fake ``TextEmbedding`` returns.

  Returns:
    A list that receives the keyword arguments of every model loaded.
  """
  loaded: list[dict[str, Any]] = []

  def text_embedding(**kwargs: Any) -> _FakeBackend:
    loaded.append(kwargs)
    return backend

  module = types.SimpleNamespace(TextEmbedding=text_embedding)
  monkeypatch.setitem(sys.modules, "fastembed", module)
  return loaded


def test_importing_the_module_does_not_import_fastembed():
  script = (
      "import sys\n"
      "from blast_radius import embeddings\n"
      "loaded = {'fastembed', 'onnxruntime'}.intersection(sys.modules)\n"
      "sys.exit(1 if loaded else 0)\n"
  )

  completed = subprocess.run([sys.executable, "-c", script], check=False)

  assert completed.returncode == 0


def test_fastembed_embedder_loads_the_named_model_from_the_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
  loaded = _fake_fastembed(monkeypatch, _FakeBackend())

  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path)

  assert loaded == [{"model_name": _MODEL, "cache_dir": str(tmp_path)}]
  assert embedder.name == _MODEL


def test_fastembed_embedder_measures_dim_from_the_backend(
    tmp_path: pathlib.Path,
):
  embedder = embeddings.FastEmbedEmbedder(
      _MODEL, tmp_path, backend=_FakeBackend(dim=7)
  )

  assert embedder.dim == 7


def test_embed_documents_returns_unit_float32_rows_in_input_order(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend(passages={"first": [3.0, 4.0], "second": [0.0, 5.0]})
  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path, backend=backend)

  matrix = embedder.embed_documents(["first", "second"])

  assert matrix.dtype == np.float32
  assert matrix == pytest.approx(np.array([[0.6, 0.8], [0.0, 1.0]]))


def test_embed_documents_hands_the_backend_every_text_in_one_batched_call(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend()
  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path, backend=backend)
  texts = [f"chunk {i}" for i in range(100)]

  matrix = embedder.embed_documents(texts)

  assert backend.passage_calls == [(texts, embeddings.DOCUMENT_BATCH_SIZE)]
  assert matrix.shape == (100, 2)


def test_embed_documents_of_no_texts_is_empty_and_skips_the_backend(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend(dim=5)
  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path, backend=backend)

  matrix = embedder.embed_documents([])

  assert matrix.shape == (0, 5)
  assert matrix.dtype == np.float32
  assert not backend.passage_calls


def test_embed_documents_zero_vector_stays_zero(tmp_path: pathlib.Path):
  backend = _FakeBackend(passages={"blank": [0.0, 0.0]})
  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path, backend=backend)

  matrix = embedder.embed_documents(["blank"])

  assert matrix.tolist() == [[0.0, 0.0]]


def test_embed_documents_missing_vector_is_an_error(tmp_path: pathlib.Path):
  embedder = embeddings.FastEmbedEmbedder(
      _MODEL, tmp_path, backend=_ShortBackend()
  )

  with pytest.raises(RuntimeError, match=r"shape \(2, 2\) for 3 texts"):
    embedder.embed_documents(["one", "two", "three"])


def test_embed_query_returns_the_backends_query_vector_normalised(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend(
      passages={"openssh": [1.0, 0.0]}, queries={"openssh": [6.0, 8.0]}
  )
  embedder = embeddings.FastEmbedEmbedder(_MODEL, tmp_path, backend=backend)

  vector = embedder.embed_query("openssh")

  assert vector.dtype == np.float32
  assert vector.tolist() == pytest.approx([0.6, 0.8])
  assert backend.query_calls[-1] == "openssh"


def test_hashing_embedder_reports_its_name_and_dim():
  embedder = embeddings.HashingEmbedder(dim=64)

  assert embedder.name == "hashing-64"
  assert embedder.dim == 64


@pytest.mark.parametrize("dim", [0, -8])
def test_hashing_embedder_dim_not_positive_is_rejected(dim: int):
  with pytest.raises(ValueError):
    embeddings.HashingEmbedder(dim=dim)


def test_hashing_embedder_returns_unit_float32_rows():
  embedder = embeddings.HashingEmbedder(dim=32)

  matrix = embedder.embed_documents(["OpenSSH auth bypass", "Traefik proxy"])

  assert matrix.shape == (2, 32)
  assert matrix.dtype == np.float32
  assert np.linalg.norm(matrix, axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_hashing_embedder_is_deterministic_across_instances():
  first = embeddings.HashingEmbedder().embed_query("nfs_net_init error path")
  second = embeddings.HashingEmbedder().embed_query("nfs_net_init error path")

  assert first.tolist() == second.tolist()


def test_hashing_embedder_scores_shared_vocabulary_higher():
  embedder = embeddings.HashingEmbedder()
  query = embedder.embed_query("OpenSSH authentication bypass")

  related, unrelated = embedder.embed_documents(
      [
          "An authentication bypass in OpenSSH before 9.6.",
          "Traefik is a cloud native application proxy.",
      ]
  )

  assert float(query @ related) > float(query @ unrelated)


def test_hashing_embedder_embeds_queries_and_documents_alike():
  embedder = embeddings.HashingEmbedder()

  query = embedder.embed_query("kernel use after free")
  document = embedder.embed_documents(["kernel use after free"])[0]

  assert query.tolist() == document.tolist()


def test_hashing_embedder_text_without_words_is_the_zero_vector():
  vector = embeddings.HashingEmbedder(dim=16).embed_query("?!")

  assert not vector.any()


def test_hashing_embedder_of_no_texts_is_empty():
  matrix = embeddings.HashingEmbedder(dim=16).embed_documents([])

  assert matrix.shape == (0, 16)


def test_create_hashing_name_returns_a_hashing_embedder_of_that_dim():
  settings = config.Settings(embedding_model="hashing-64")

  embedder = embeddings.create(settings)

  assert isinstance(embedder, embeddings.HashingEmbedder)
  assert embedder.name == "hashing-64"


@pytest.mark.parametrize(
    "name",
    [
        "hashing-",
        "hashing-abc",
        "hashing-0",
        "hashing--4",
        "hashing-1.5",
        "hashing-064",
        "hashing- 64",
        "hashing-64 ",
    ],
)
def test_create_malformed_hashing_name_is_rejected(name: str):
  settings = config.Settings(embedding_model=name)

  with pytest.raises(ValueError, match="hashing-<dim>"):
    embeddings.create(settings)


def test_create_any_other_name_loads_that_model_through_fastembed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
  loaded = _fake_fastembed(monkeypatch, _FakeBackend(dim=3))
  settings = config.Settings(embedding_model=_MODEL, model_cache_dir=tmp_path)

  embedder = embeddings.create(settings)

  assert isinstance(embedder, embeddings.FastEmbedEmbedder)
  assert loaded == [{"model_name": _MODEL, "cache_dir": str(tmp_path)}]
  assert (embedder.name, embedder.dim) == (_MODEL, 3)
