"""Scores each retrieval configuration against the saved question set.

One table, four rows: keyword only, embeddings only, both fused, and both
fused then reranked. Every row runs the same questions through the same
retriever code the API uses, with a different set of stages switched on, so
the table is the evidence for (or against) each stage.

What is measured, all without a language model:

* **Context recall@5, precision@5 and MRR**, at QID level. Hosts attach to
  QIDs, and sibling CVEs under one QID resolve to the same hosts, so a hit
  on any chunk of a gold QID counts. Recall@5 at CVE level is reported
  beside it for questions written from one specific CVE.
* **Host precision and recall**: the hosts of the QIDs the retriever chose
  to act on, against the hosts of the gold QIDs.
* **Abstention**: negatives that matched nothing, and positives that
  matched something.
* **Latency** of the retrieval step, p50 and p95.

Identifier questions never reach the search stages, so they score the same
in every row; the table reports them and the search questions separately.
A table over forty questions has wide intervals, which is why every rate is
printed with its counts.
"""

from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import sys
import time
from typing import Any

from blast_radius import config
from blast_radius import embeddings
from blast_radius import models
from blast_radius import pipeline
from blast_radius import schema
from blast_radius import store
from blast_radius.evaluation import metrics
from blast_radius.evaluation import questions as questions_lib
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

_LOG = logging.getLogger(__name__)

_K = 5

# Which stages each row of the table switches on.
CONFIGURATIONS: dict[str, dict[str, bool]] = {
    "bm25": {
        "enable_keyword": True,
        "enable_dense": False,
        "enable_rerank": False,
    },
    "dense": {
        "enable_keyword": False,
        "enable_dense": True,
        "enable_rerank": False,
    },
    "hybrid": {
        "enable_keyword": True,
        "enable_dense": True,
        "enable_rerank": False,
    },
    "hybrid+rerank": {
        "enable_keyword": True,
        "enable_dense": True,
        "enable_rerank": True,
    },
}


@dataclasses.dataclass(frozen=True)
class QuestionResult:
  """How one configuration did on one question.

  Attributes:
    question_id: The question's id.
    question_type: The question's type.
    ranked_qids: QIDs in ranked order, as far as the candidates go.
    matched_qids: QIDs the retriever chose to act on.
    top_score: The best candidate's score, for calibrating the floor.
    recall: Context recall@5 at QID level; None for a negative.
    precision: Context precision@5 at QID level; None for a negative.
    reciprocal_rank: 1 / rank of the first gold QID; None for a negative.
    cve_recall: Recall@5 at CVE level; None without gold CVEs.
    host_precision: Share of returned hosts that are gold; None for a
      negative.
    host_recall: Share of gold hosts that were returned; None for a
      negative.
    abstained_correctly: For a negative, that nothing matched; for a
      positive, that something did.
    latency_ms: Wall time of the retrieval step.
  """

  question_id: str
  question_type: str
  ranked_qids: list[str]
  matched_qids: list[str]
  top_score: float | None
  recall: float | None
  precision: float | None
  reciprocal_rank: float | None
  cve_recall: float | None
  host_precision: float | None
  host_recall: float | None
  abstained_correctly: bool
  latency_ms: float


def _unique(items: Sequence[str]) -> list[str]:
  """Returns ``items`` without repeats, first occurrences kept in order."""
  return list(dict.fromkeys(items))


def _ranked_ids(
    db: store.Store, result: models.RetrievalResult
) -> tuple[list[str], list[str]]:
  """Returns the ranked QIDs and the ranked CVE ids of one retrieval.

  An identifier lookup has no candidate list; its matches are its ranking.

  Args:
    db: The artifact, used to roll CVE chunks up to their QIDs.
    result: What the retriever returned.
  """
  qids: list[str] = []
  cve_ids: list[str] = []
  if not result.candidates:
    for match in result.matches:
      qids.append(match.qid)
      cve_ids.extend(match.matched_cve_ids)
    return qids, _unique(cve_ids)
  for item in result.candidates:
    chunk = item.chunk
    if chunk.doc_type == "qid":
      qids.append(chunk.doc_id)
    else:
      cve_ids.append(chunk.doc_id)
      qids.extend(db.qids_for_cve(chunk.doc_id))
  return _unique(qids), _unique(cve_ids)


def _top_score(result: models.RetrievalResult) -> float | None:
  """Returns the score the floor was held against, or None without one.

  That is the best rerank score when the candidates were reranked, else the
  best cosine. Candidates that were not reranked are in fused order, so the
  best cosine need not be the first candidate's.
  """
  candidates = result.candidates
  reranked = [c.rerank_score for c in candidates if c.rerank_score is not None]
  cosines = [c.dense_score for c in candidates if c.dense_score is not None]
  return max(reranked or cosines, default=None)


def score_question(
    db: store.Store,
    retriever: retriever_lib.Retriever,
    question: questions_lib.Question,
) -> QuestionResult:
  """Runs one question through one retriever and scores the outcome.

  Args:
    db: The artifact.
    retriever: The configuration under test.
    question: The question and its gold answer.
  """
  parsed, _ = pipeline.parse_without_llm(question.query)
  started = time.perf_counter()
  result = retriever.retrieve(parsed)
  latency_ms = (time.perf_counter() - started) * 1000

  ranked_qids, ranked_cves = _ranked_ids(db, result)
  matched = [match.qid for match in result.matches]
  scores: dict[str, float | None] = dict.fromkeys(
      ("recall", "precision", "rr", "cve_recall", "host_p", "host_r")
  )
  if not question.is_negative:
    gold = set(question.gold_qids)
    scores["recall"] = metrics.recall_at_k(ranked_qids, gold, _K)
    scores["precision"] = metrics.precision_at_k(ranked_qids, gold, _K)
    scores["rr"] = metrics.reciprocal_rank(ranked_qids, gold)
    if question.gold_cve_ids and question.type != "identifier":
      scores["cve_recall"] = metrics.recall_at_k(
          ranked_cves, set(question.gold_cve_ids), _K
      )
    returned_hosts = {host.id for host, _ in db.hosts_for_qids(matched)}
    gold_hosts = {host.id for host, _ in db.hosts_for_qids(sorted(gold))}
    scores["host_p"], scores["host_r"] = metrics.set_precision_recall(
        returned_hosts, gold_hosts
    )
  return QuestionResult(
      question_id=question.id,
      question_type=question.type,
      ranked_qids=ranked_qids[: 2 * _K],
      matched_qids=matched,
      top_score=_top_score(result),
      recall=scores["recall"],
      precision=scores["precision"],
      reciprocal_rank=scores["rr"],
      cve_recall=scores["cve_recall"],
      host_precision=scores["host_p"],
      host_recall=scores["host_r"],
      abstained_correctly=bool(matched) != question.is_negative,
      latency_ms=round(latency_ms, 2),
  )


def summarise(results: Sequence[QuestionResult]) -> dict[str, Any]:
  """Returns the aggregate row for one configuration and question subset.

  Args:
    results: Per-question results; negatives contribute to abstention and
      latency only.
  """

  def mean(field: str, *, binary: bool) -> metrics.Mean:
    values = [
        getattr(result, field)
        for result in results
        if getattr(result, field) is not None
    ]
    return metrics.mean_of(values, binary=binary)

  # A subset can be empty: a question set of identifiers alone has no search
  # questions. Its means render as "0.00 (0/0)", and its latencies follow.
  latencies = [result.latency_ms for result in results] or [0.0]
  return {
      "questions": len(results),
      "recall_at_5": str(mean("recall", binary=True)),
      "precision_at_5": str(mean("precision", binary=False)),
      "mrr": str(mean("reciprocal_rank", binary=False)),
      "cve_recall_at_5": str(mean("cve_recall", binary=True)),
      "host_precision": str(mean("host_precision", binary=False)),
      "host_recall": str(mean("host_recall", binary=False)),
      "abstention": str(
          metrics.mean_of(
              [float(result.abstained_correctly) for result in results],
              binary=True,
          )
      ),
      "latency_p50_ms": round(metrics.percentile(latencies, 50), 1),
      "latency_p95_ms": round(metrics.percentile(latencies, 95), 1),
  }


_COLUMNS = (
    ("recall_at_5", "Recall@5"),
    ("precision_at_5", "Precision@5"),
    ("mrr", "MRR"),
    ("cve_recall_at_5", "CVE recall@5"),
    ("host_precision", "Host P"),
    ("host_recall", "Host R"),
    ("abstention", "Abstention"),
    ("latency_p50_ms", "p50 ms"),
    ("latency_p95_ms", "p95 ms"),
)


def format_table(title: str, rows: dict[str, dict[str, Any]]) -> str:
  """Returns one Markdown table, a row per configuration.

  Args:
    title: What the table covers, printed above it.
    rows: Aggregates keyed by configuration name.
  """
  header = ["Configuration"] + [label for _, label in _COLUMNS]
  lines = [
      f"{title}",
      "",
      "| " + " | ".join(header) + " |",
      "|" + "|".join("---" for _ in header) + "|",
  ]
  for name, row in rows.items():
    cells = [f"`{name}`"] + [str(row[key]) for key, _ in _COLUMNS]
    lines.append("| " + " | ".join(cells) + " |")
  return "\n".join(lines) + "\n"


def run(
    settings: config.Settings,
    questions_path: pathlib.Path,
    results_path: pathlib.Path,
) -> int:
  """Evaluates every configuration, prints the tables and saves the details.

  Args:
    settings: Base settings; each configuration overrides the stage toggles.
    questions_path: The JSONL question set.
    results_path: Where the per-question results are written as JSON.

  Returns:
    0 on success, 1 when the question set is inconsistent with the artifact.
  """
  questions = questions_lib.load(questions_path)
  with store.Store(settings.artifact_path) as db:
    problems = questions_lib.validate(questions, db)
    if problems:
      for problem in problems:
        sys.stderr.write(f"{questions_path}: {problem}\n")
      return 1

    # The models are loaded once and shared by the configurations that use
    # them; only the retriever around them changes.
    embedder = embeddings.create(settings)
    reranker = rerank.create(settings)
    details: dict[str, list[QuestionResult]] = {}
    for name, toggles in CONFIGURATIONS.items():
      retriever = retriever_lib.Retriever(
          db, embedder, reranker, settings.model_copy(update=toggles)
      )
      details[name] = [
          score_question(db, retriever, question) for question in questions
      ]
      _LOG.info("evaluated %s on %d questions", name, len(questions))
    meta = db.meta()

  search_types = {"paraphrase", "product", "negative"}
  tables = {
      "All questions": {
          name: summarise(results) for name, results in details.items()
      },
      "Search questions only (paraphrase, product, negative)": {
          name: summarise(
              [r for r in results if r.question_type in search_types]
          )
          for name, results in details.items()
      },
  }
  for title, rows in tables.items():
    sys.stdout.write(format_table(title, rows) + "\n")

  results_path.parent.mkdir(parents=True, exist_ok=True)
  results_path.write_text(
      json.dumps(
          {
              "dataset_sha256": {
                  "assets": meta.get(schema.META_ASSETS_SHA256),
                  "vulns": meta.get(schema.META_VULNS_SHA256),
              },
              "embedding_model": embedder.name,
              "rerank_model": reranker.name,
              "abstain_rerank_floor": settings.abstain_rerank_floor,
              "abstain_dense_floor": settings.abstain_dense_floor,
              "match_margin": settings.match_margin,
              "match_margin_dense": settings.match_margin_dense,
              "tables": tables,
              "questions": {
                  name: [dataclasses.asdict(result) for result in results]
                  for name, results in details.items()
              },
          },
          indent=2,
      )
      + "\n",
      encoding="utf-8",
  )
  return 0
