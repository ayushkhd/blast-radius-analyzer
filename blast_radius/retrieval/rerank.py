"""Cross-encoder reranking of fused candidates.

A bi-encoder embeds the query and each chunk separately, so hundreds of
kernel CVE descriptions that share their vocabulary land close together. A
cross-encoder reads the query and one candidate as a single input and scores
the pair, which is slower per candidate and much better at telling siblings
apart. It therefore runs last, over a few dozen candidates only.

``Reranker`` is the interface the retriever depends on. ``FastEmbedReranker``
is the production implementation; ``LexicalReranker`` is a deterministic
stand-in for tests that needs no model.
"""

from collections.abc import Iterable, Sequence
import pathlib
import re
from typing import Protocol

from blast_radius import config

# Pairs scored per ONNX Runtime call. As with the embedder, padding a batch
# to its longest member costs more on CPU than batching saves. Measured on a
# laptop, 30 candidates with the reference corpus's mix of lengths take
# about 0.7 s one at a time and 1.6 s as the single batch that fastembed's
# default of 64 makes of them. This is on the query path, so the difference
# is felt on every search.
BATCH_SIZE = 1

LEXICAL_MODEL = "lexical"

_WORD = re.compile(r"[a-z0-9]+")


class Reranker(Protocol):
  """Scores how well each candidate text answers a query.

  Attributes:
    name: Model name recorded in responses.
  """

  name: str

  def score(self, query: str, texts: Sequence[str]) -> list[float]:
    """Returns one relevance score per text, higher meaning more relevant.

    Scores are comparable within one call and across calls with the same
    model, which is what lets a fixed floor decide when to abstain. They are
    not probabilities.

    Args:
      query: The search query.
      texts: Candidate texts, in any order.
    """


class CrossEncoderBackend(Protocol):
  """The part of fastembed's ``TextCrossEncoder`` that is used here.

  Naming it lets a test hand in a fake with this one method and load no
  model.
  """

  def rerank(
      self, query: str, documents: Iterable[str], *, batch_size: int
  ) -> Iterable[float]:
    """Yields one score per document, in the order given.

    Args:
      query: The search query.
      documents: Candidate texts.
      batch_size: Pairs that go through the model in one call.
    """


class FastEmbedReranker:
  """A cross-encoder run on CPU through fastembed.

  Scores are the model's raw logits. For ``Xenova/ms-marco-MiniLM-L-6-v2``
  they run from about -11 for an unrelated text to about +10 for a direct
  answer, and the abstention floor in the settings is on that scale.

  Attributes:
    name: The fastembed model name.
  """

  def __init__(
      self,
      model_name: str,
      cache_dir: pathlib.Path,
      *,
      backend: CrossEncoderBackend | None = None,
  ) -> None:
    """Initialises the reranker, loading the model unless one is handed in.

    Args:
      model_name: A cross-encoder name that fastembed knows.
      cache_dir: Where the model files are cached. They are downloaded on
        first use and read from here afterwards.
      backend: A ready model to use instead of loading ``model_name``.
    """
    if backend is None:
      # fastembed brings in ONNX Runtime, so it is imported only when a model
      # is really loaded: importing this module, and every test that hands
      # in a backend, stays free of it.
      # Lazy on purpose, see above. pylint: disable-next=import-outside-toplevel
      from fastembed.rerank import cross_encoder

      backend = cross_encoder.TextCrossEncoder(
          model_name=model_name, cache_dir=str(cache_dir)
      )
    self.name = model_name
    self._backend = backend

  def score(self, query: str, texts: Sequence[str]) -> list[float]:
    """Returns one relevance score per text, higher meaning more relevant.

    Args:
      query: The search query.
      texts: Candidate texts, in any order.

    Raises:
      RuntimeError: If the model does not return one score per text.
    """
    if not texts:
      return []
    scores = list(self._backend.rerank(query, texts, batch_size=BATCH_SIZE))
    if len(scores) != len(texts):
      raise RuntimeError(
          f"{self.name} returned {len(scores)} scores for {len(texts)} texts"
      )
    return scores


class LexicalReranker:
  """A model-free reranker that scores by word overlap.

  The score is the fraction of the query's distinct words that occur in the
  text, from 0 to 1. It is deterministic and needs no download, which is
  what tests want. It knows nothing of meaning and is not meant to be
  evaluated.

  Attributes:
    name: ``lexical``, recorded in responses like any model name.
  """

  name = LEXICAL_MODEL

  def score(self, query: str, texts: Sequence[str]) -> list[float]:
    """Returns one relevance score per text, higher meaning more relevant.

    Args:
      query: The search query.
      texts: Candidate texts, in any order.
    """
    wanted = set(_WORD.findall(query.lower()))
    if not wanted:
      return [0.0] * len(texts)
    return [
        len(wanted.intersection(_WORD.findall(text.lower()))) / len(wanted)
        for text in texts
    ]


def create(settings: config.Settings) -> Reranker:
  """Returns the reranker that ``settings.rerank_model`` names.

  ``lexical`` selects ``LexicalReranker``, so the service can run where no
  model can be downloaded. Any other name is loaded through fastembed.

  Args:
    settings: Supplies the model name and the model cache directory.
  """
  if settings.rerank_model == LEXICAL_MODEL:
    return LexicalReranker()
  return FastEmbedReranker(settings.rerank_model, settings.model_cache_dir)
