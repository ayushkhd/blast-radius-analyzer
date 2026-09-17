"""Builds the long-lived objects of a process from its settings.

The API server, the ``ask`` command and the evaluation all need the same
object graph: an open artifact, the retrieval models, a retriever, an
optional language-model provider and the pipeline on top. It is built here,
once, so that the three cannot drift apart.

Everything in ``Services`` is safe to share between threads: the store keeps
a connection per thread, the dense index and the models are read-only after
construction, and the pipeline holds no per-request state.
"""

import dataclasses
import logging

from blast_radius import config
from blast_radius import embeddings
from blast_radius import pipeline as pipeline_lib
from blast_radius import store
from blast_radius.llm import base as llm_base
from blast_radius.llm import factory as llm_factory
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Services:
  """The object graph of one serving process.

  Attributes:
    db: The open index artifact.
    retriever: Search over the artifact.
    pipeline: The seven-step pipeline.
  """

  db: store.Store
  retriever: retriever_lib.Retriever
  pipeline: pipeline_lib.Pipeline

  def close(self) -> None:
    """Releases the artifact's connections."""
    self.db.close()


def build_retriever(
    db: store.Store, settings: config.Settings
) -> retriever_lib.Retriever:
  """Returns a retriever with exactly the models ``settings`` enables.

  A disabled stage never loads its model, so a keyword-only deployment
  starts without ONNX Runtime ever being imported.

  Args:
    db: The open index artifact.
    settings: Which stages run and which models they use.

  Raises:
    retriever_lib.RetrieverConfigError: If the artifact and the settings do
      not fit together.
  """
  embedder = embeddings.create(settings) if settings.enable_dense else None
  reranker = rerank.create(settings) if settings.enable_rerank else None
  return retriever_lib.Retriever(db, embedder, reranker, settings)


def build(
    settings: config.Settings,
    *,
    provider: llm_base.Provider | None = None,
    use_llm: bool = True,
) -> Services:
  """Opens the artifact and builds everything on top of it.

  Args:
    settings: The process's settings.
    provider: A language-model provider to use instead of the configured
      one. Tests pass a scripted provider here.
    use_llm: False forces no-LLM mode whatever the settings say. The
      evaluation uses it so that retrieval metrics never cost money.

  Raises:
    store.StoreError: If the artifact is missing or was built by an
      incompatible version.
    retriever_lib.RetrieverConfigError: If the artifact and the settings do
      not fit together.
  """
  db = store.Store(settings.artifact_path)
  try:
    retriever = build_retriever(db, settings)
    if provider is None and use_llm:
      provider = llm_factory.create(settings)
    pipeline = pipeline_lib.Pipeline(db, retriever, provider, settings)
  except Exception:
    db.close()
    raise
  _LOG.info(
      "services ready: embedding=%s rerank=%s llm=%s",
      retriever.embedding_model,
      retriever.rerank_model,
      provider.model if provider else None,
  )
  return Services(db=db, retriever=retriever, pipeline=pipeline)
