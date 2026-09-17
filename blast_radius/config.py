"""Runtime settings, read from the environment.

Every knob the design calls configurable lives here: which retrieval stages
run, how many candidates each passes on, the ranking weights and the
language model. Variables take the ``BLAST_`` prefix, e.g.
``BLAST_ENABLE_RERANK=false``. A ``.env`` file in the working directory is
read if present.

The language-model API key is deliberately not a setting. The provider SDK
reads ``ANTHROPIC_API_KEY`` itself, so the key never passes through this
process's own configuration, logs or responses.
"""

import pathlib
from typing import Literal

import pydantic
import pydantic_settings

LlmProviderName = Literal["anthropic", "none"]


class Settings(pydantic_settings.BaseSettings):
  """Settings for ingest, retrieval, ranking, generation and serving.

  Attributes:
    data_dir: Directory holding the two scanner exports.
    artifact_path: The index artifact, a single SQLite file.
    model_cache_dir: Where embedding and reranking models are cached.
    embedding_model: fastembed model used for dense retrieval.
    rerank_model: fastembed cross-encoder used for reranking.
    chunk_max_chars: Upper bound on chunk length at ingest.
    chunk_overlap_sentences: Sentences repeated between adjacent chunks.
    enable_keyword: Whether BM25 keyword search runs.
    enable_dense: Whether embedding search runs.
    enable_rerank: Whether the cross-encoder reranks fused candidates.
    rerank_max_query_words: Only queries of at most this many words are
      reranked. The cross-encoder was trained on short search queries: on
      the evaluation set it sharpens short product queries and misranks
      long pasted advisories, which fusion alone handles better and about a
      second faster. With a language model configured, a long advisory is
      distilled into short queries first, and those are reranked.
    fusion_k: The constant in reciprocal rank fusion.
    candidate_count: Candidates each search stage returns and fusion keeps.
    context_count: Chunks kept after reranking.
    abstain_rerank_floor: With reranking on, retrieval abstains when the
      best cross-encoder score (a raw logit) is below this. Chosen by
      sweeping the evaluation set.
    abstain_dense_floor: With reranking off, retrieval abstains when the
      best cosine similarity is below this. Chosen by sweeping the
      evaluation set; cosine separates out-of-corpus questions poorly, so
      the floor is deliberately low and the limitation is documented.
    match_margin: A chunk supports a match only if its score is within this
      distance of the best chunk's score, in the units of the active floor.
      It keeps a strong hit from dragging in every weak sibling.
    match_margin_dense: The same margin, for cosine similarities.
    parse_min_words: Free text with at least this many words is distilled
      into search queries by the language model; shorter text is searched
      as typed.
    fix_chunk_count: Chunks the fix-evidence search adds to the context.
    max_refs_per_cve: Patch and advisory references kept per CVE.
    max_evidence_cves: When a QID was matched as a whole, its CVEs are
      consulted for fix evidence only if there are at most this many. An
      Ubuntu kernel update bundles hundreds of CVEs whose upstream commit
      links are no use to an analyst; the package update is the fix.
    max_context_items: Upper bound on sources handed to the language model.
    weight_severity: Weight of severity in the threat score.
    weight_epss: Weight of the EPSS percentile in the threat score.
    weight_known_exploited: Weight of KEV membership in the threat score.
    internet_facing_multiplier: Exposure multiplier for internet-facing
      hosts.
    baseline_criticality: The criticality that maps to an exposure of 1.
    group_example_count: Example host names listed per group.
    llm_provider: ``anthropic``, or ``none`` to run without a language
      model.
    llm_model: Model identifier passed to the provider.
    llm_effort: Reasoning effort passed to the provider.
    llm_max_tokens: Output token cap per call.
    llm_timeout_s: Per-attempt timeout in seconds.
    llm_max_retries: Extra attempts after a retryable failure.
    host: Interface the API binds to.
    port: Port the API binds to.
    log_level: Root log level.
  """

  model_config = pydantic_settings.SettingsConfigDict(
      env_prefix="BLAST_", env_file=".env", extra="ignore"
  )

  data_dir: pathlib.Path = pathlib.Path("data")
  artifact_path: pathlib.Path = pathlib.Path("artifacts/index.sqlite")
  model_cache_dir: pathlib.Path = pathlib.Path(".cache/models")

  embedding_model: str = "BAAI/bge-small-en-v1.5"
  rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
  chunk_max_chars: int = pydantic.Field(default=1200, gt=0)
  chunk_overlap_sentences: int = pydantic.Field(default=1, ge=0)

  enable_keyword: bool = True
  enable_dense: bool = True
  enable_rerank: bool = True
  rerank_max_query_words: int = pydantic.Field(default=12, gt=0)
  fusion_k: int = pydantic.Field(default=60, gt=0)
  candidate_count: int = pydantic.Field(default=30, gt=0)
  context_count: int = pydantic.Field(default=8, gt=0)
  abstain_rerank_floor: float = -4.0
  abstain_dense_floor: float = 0.6
  match_margin: float = pydantic.Field(default=1.0, ge=0)
  match_margin_dense: float = pydantic.Field(default=0.05, ge=0)
  parse_min_words: int = pydantic.Field(default=12, gt=0)
  fix_chunk_count: int = pydantic.Field(default=4, ge=0)
  max_refs_per_cve: int = pydantic.Field(default=2, ge=0)
  max_evidence_cves: int = pydantic.Field(default=10, ge=0)
  max_context_items: int = pydantic.Field(default=24, gt=0)

  weight_severity: float = pydantic.Field(default=0.5, ge=0)
  weight_epss: float = pydantic.Field(default=0.3, ge=0)
  weight_known_exploited: float = pydantic.Field(default=0.2, ge=0)
  internet_facing_multiplier: float = pydantic.Field(default=2.0, ge=1)
  baseline_criticality: int = pydantic.Field(default=3, gt=0)
  group_example_count: int = pydantic.Field(default=4, gt=0)

  llm_provider: LlmProviderName = "anthropic"
  llm_model: str = "claude-opus-5"
  llm_effort: str = "low"
  llm_max_tokens: int = pydantic.Field(default=16000, gt=0)
  llm_timeout_s: float = pydantic.Field(default=90.0, gt=0)
  llm_max_retries: int = pydantic.Field(default=2, ge=0)

  host: str = "127.0.0.1"
  port: int = 8000
  log_level: str = "INFO"

  @property
  def assets_path(self) -> pathlib.Path:
    """Returns the path of the asset export inside ``data_dir``."""
    return self.data_dir / "asset_data_scrubbed.json"

  @property
  def vulns_path(self) -> pathlib.Path:
    """Returns the path of the vulnerability export inside ``data_dir``."""
    return self.data_dir / "vulns_data_scrubbed.json"
