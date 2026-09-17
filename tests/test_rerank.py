"""Tests for blast_radius.retrieval.rerank.

No test loads a model: ``FastEmbedReranker`` is given a fake backend, or a
fake ``fastembed`` module where the point is how the real one is loaded.
"""

from collections.abc import Iterable, Iterator
import pathlib
import subprocess
import sys
import types
from typing import Any

import pytest

from blast_radius import config
from blast_radius.retrieval import rerank

_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class _FakeBackend:
  """Stands in for fastembed's ``TextCrossEncoder``: scripted scores."""

  def __init__(self, scores: list[float]) -> None:
    self._scores = scores
    self.calls: list[tuple[str, list[str], int]] = []

  def rerank(
      self, query: str, documents: Iterable[str], *, batch_size: int
  ) -> Iterator[float]:
    self.calls.append((query, list(documents), batch_size))
    yield from self._scores


def _fake_fastembed(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend
) -> list[dict[str, Any]]:
  """Replaces fastembed's cross-encoder module with one that gives ``backend``.

  Args:
    monkeypatch: Undoes the replacement when the test ends.
    backend: What the fake ``TextCrossEncoder`` returns.

  Returns:
    A list that receives the keyword arguments of every model loaded.
  """
  loaded: list[dict[str, Any]] = []

  def text_cross_encoder(**kwargs: Any) -> _FakeBackend:
    loaded.append(kwargs)
    return backend

  module = types.SimpleNamespace(TextCrossEncoder=text_cross_encoder)
  package = types.SimpleNamespace(cross_encoder=module)
  monkeypatch.setitem(sys.modules, "fastembed.rerank", package)
  monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)
  return loaded


def test_importing_the_module_does_not_import_fastembed():
  script = (
      "import sys\n"
      "from blast_radius.retrieval import rerank\n"
      "loaded = {'fastembed', 'onnxruntime'}.intersection(sys.modules)\n"
      "sys.exit(1 if loaded else 0)\n"
  )

  completed = subprocess.run([sys.executable, "-c", script], check=False)

  assert completed.returncode == 0


def test_fastembed_reranker_loads_the_named_model_from_the_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
  loaded = _fake_fastembed(monkeypatch, _FakeBackend([]))

  reranker = rerank.FastEmbedReranker(_MODEL, tmp_path)

  assert loaded == [{"model_name": _MODEL, "cache_dir": str(tmp_path)}]
  assert reranker.name == _MODEL


def test_fastembed_reranker_returns_the_backends_scores_in_order(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend([7.5, -11.25, 0.5])
  reranker = rerank.FastEmbedReranker(_MODEL, tmp_path, backend=backend)

  scores = reranker.score("openssh bypass", ["first", "second", "third"])

  assert scores == [7.5, -11.25, 0.5]


def test_fastembed_reranker_hands_the_backend_every_text_in_one_call(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend([1.0, 2.0])
  reranker = rerank.FastEmbedReranker(_MODEL, tmp_path, backend=backend)

  reranker.score("openssh bypass", ("first", "second"))

  assert backend.calls == [
      ("openssh bypass", ["first", "second"], rerank.BATCH_SIZE)
  ]


def test_fastembed_reranker_no_texts_is_empty_and_skips_the_backend(
    tmp_path: pathlib.Path,
):
  backend = _FakeBackend([1.0])
  reranker = rerank.FastEmbedReranker(_MODEL, tmp_path, backend=backend)

  assert not reranker.score("openssh bypass", [])
  assert not backend.calls


def test_fastembed_reranker_missing_score_is_an_error(tmp_path: pathlib.Path):
  backend = _FakeBackend([1.0, 2.0])
  reranker = rerank.FastEmbedReranker(_MODEL, tmp_path, backend=backend)

  with pytest.raises(RuntimeError, match="2 scores for 3 texts"):
    reranker.score("openssh bypass", ["first", "second", "third"])


def test_lexical_reranker_scores_the_fraction_of_query_words_in_the_text():
  reranker = rerank.LexicalReranker()

  scores = reranker.score(
      "openssh authentication bypass",
      [
          "An authentication bypass in OpenSSH.",
          "OpenSSH is a connectivity tool.",
          "Traefik is a proxy.",
      ],
  )

  assert scores == pytest.approx([1.0, 1 / 3, 0.0])


def test_lexical_reranker_ignores_case_and_punctuation():
  scores = rerank.LexicalReranker().score(
      "MOD_PROXY, smuggling!", ["request smuggling in mod_proxy"]
  )

  assert scores == [1.0]


def test_lexical_reranker_counts_a_repeated_query_word_once():
  scores = rerank.LexicalReranker().score(
      "kernel kernel kernel netfilter", ["a kernel bug"]
  )

  assert scores == [0.5]


def test_lexical_reranker_matches_whole_words_only():
  scores = rerank.LexicalReranker().score("ssh", ["OpenSSH up to 9.6"])

  assert scores == [0.0]


def test_lexical_reranker_query_without_words_scores_zero():
  scores = rerank.LexicalReranker().score("?!", ["anything", "at all"])

  assert scores == [0.0, 0.0]


def test_lexical_reranker_no_texts_is_empty():
  assert not rerank.LexicalReranker().score("openssh", [])


def test_lexical_reranker_is_named_lexical():
  assert rerank.LexicalReranker().name == "lexical"


def test_create_lexical_returns_the_lexical_reranker():
  settings = config.Settings(rerank_model="lexical")

  assert isinstance(rerank.create(settings), rerank.LexicalReranker)


def test_create_any_other_name_loads_that_model_through_fastembed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
  loaded = _fake_fastembed(monkeypatch, _FakeBackend([]))
  settings = config.Settings(rerank_model=_MODEL, model_cache_dir=tmp_path)

  reranker = rerank.create(settings)

  assert isinstance(reranker, rerank.FastEmbedReranker)
  assert loaded == [{"model_name": _MODEL, "cache_dir": str(tmp_path)}]
  assert reranker.name == _MODEL
