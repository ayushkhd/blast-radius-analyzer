"""Tests for blast_radius.services."""

import os
import pathlib
from typing import Any, NoReturn

import pytest

from blast_radius import config
from blast_radius import embeddings
from blast_radius import services as services_lib
from blast_radius import store as store_lib
from blast_radius.llm import anthropic_provider
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib
from tests import fakes

# A QID of the synthetic exports; tests/fixtures/make_fixtures.py defines it.
_QID_OPENSSH = "710001"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps the machine's own BLAST_* variables out of the settings."""
  for name in list(os.environ):
    if name.startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(artifact_path: pathlib.Path, **overrides: Any) -> config.Settings:
  """Returns model-free settings with every stage of retrieval switched on.

  Args:
    artifact_path: The artifact to serve.
    **overrides: Settings that a test wants different.
  """
  values: dict[str, Any] = {
      "artifact_path": artifact_path,
      "embedding_model": "hashing-256",
      "rerank_model": "lexical",
      "enable_keyword": True,
      "enable_dense": True,
      "enable_rerank": True,
      "llm_provider": "none",
  }
  return config.Settings(_env_file=None, **{**values, **overrides})


def _fail_if_called(*args: Any, **kwargs: Any) -> NoReturn:
  """Stands in for a factory that the code under test must not reach."""
  raise AssertionError(f"unexpected call with {args} {kwargs}")


def test_build_wires_a_pipeline_that_answers(artifact_path: pathlib.Path):
  services = services_lib.build(_settings(artifact_path))

  try:
    response = services.pipeline.analyze(f"QID {_QID_OPENSSH}")
  finally:
    services.close()

  assert response.status == "matched"
  assert response.hosts
  assert services.retriever.embedding_model == "hashing-256"
  assert services.retriever.rerank_model == rerank.LEXICAL_MODEL


def test_build_prefers_the_provider_it_is_given(artifact_path: pathlib.Path):
  provider = fakes.ScriptedProvider([])

  services = services_lib.build(_settings(artifact_path), provider=provider)

  try:
    assert services.pipeline.meta.llm_provider == provider.name
    assert services.pipeline.meta.llm_model == fakes.MODEL
  finally:
    services.close()


def test_build_with_llm_provider_none_has_no_provider(
    artifact_path: pathlib.Path,
):
  services = services_lib.build(_settings(artifact_path, llm_provider="none"))

  try:
    assert services.pipeline.meta.llm_provider is None
    assert services.pipeline.meta.llm_model is None
  finally:
    services.close()


def test_build_with_use_llm_false_never_creates_the_configured_provider(
    artifact_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setattr(anthropic_provider, "create", _fail_if_called)
  settings = _settings(artifact_path, llm_provider="anthropic")

  services = services_lib.build(settings, use_llm=False)

  try:
    assert services.pipeline.meta.llm_provider is None
  finally:
    services.close()


def test_build_missing_artifact_raises_artifact_not_found(
    tmp_path: pathlib.Path,
):
  settings = _settings(tmp_path / "missing.sqlite")

  with pytest.raises(store_lib.ArtifactNotFoundError, match="missing.sqlite"):
    services_lib.build(settings)


def test_build_mismatched_embedding_model_closes_the_store_it_opened(
    artifact_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  closed: list[store_lib.Store] = []
  original_close = store_lib.Store.close

  def recording_close(self: store_lib.Store) -> None:
    closed.append(self)
    original_close(self)

  monkeypatch.setattr(store_lib.Store, "close", recording_close)
  # The artifact was embedded with hashing-256.
  settings = _settings(artifact_path, embedding_model="hashing-128")

  with pytest.raises(retriever_lib.RetrieverConfigError, match="hashing-128"):
    services_lib.build(settings)

  assert len(closed) == 1


def test_build_retriever_loads_no_model_for_a_disabled_stage(
    store: store_lib.Store,
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setattr(embeddings, "create", _fail_if_called)
  monkeypatch.setattr(rerank, "create", _fail_if_called)
  settings = _settings(artifact_path, enable_dense=False, enable_rerank=False)

  retriever = services_lib.build_retriever(store, settings)

  assert retriever.embedding_model is None
  assert retriever.rerank_model is None
  assert retriever.search(["OpenSSH authentication bypass"])


def test_services_close_closes_the_store(artifact_path: pathlib.Path):
  services = services_lib.build(_settings(artifact_path))

  services.close()

  with pytest.raises(store_lib.StoreError, match="closed"):
    services.db.stats()
