"""Tests for blast_radius.evaluation.runner."""

import json
import os
import pathlib
from typing import Any

import pytest

from blast_radius import config
from blast_radius import embeddings
from blast_radius import pipeline
from blast_radius import store as store_lib
from blast_radius.evaluation import questions as questions_lib
from blast_radius.evaluation import runner
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

_FIXTURE_QUESTIONS = pathlib.Path(__file__).parent / "fixtures/questions.jsonl"

# The rows of the table, in the order they are evaluated and printed.
_CONFIGURATIONS = ["bm25", "dense", "hybrid", "hybrid+rerank"]

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_PROXY = "710003"
_QID_BROKER = "710005"
_CVE_QUARTZFS = "CVE-2099-1002"
_CVE_TIDAL = "CVE-2099-1003"
_CVE_PROXY = "CVE-2099-2001"
_CVE_BROKER = "CVE-2099-4001"

_QUARTZFS_QUERY = "quartzfs directory entries past the end of the block"
_UNCOVERED_QUERY = "PostgreSQL privilege escalation in logical replication"


@pytest.fixture(autouse=True)
def _no_blast_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps ``BLAST_*`` variables set on this machine out of the settings."""
  for name in list(os.environ):
    if name.upper().startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(**overrides: Any) -> config.Settings:
  """Returns settings that name the model-free scorers and suit their scores.

  ``LexicalReranker`` scores run from 0 to 1, and the cosines that
  ``HashingEmbedder`` gives the fixture corpus peak near 0.6, so the
  production floors would let everything or nothing through.

  Args:
    **overrides: Settings that a test sets itself, e.g. the artifact path.
  """
  values: dict[str, Any] = {
      "embedding_model": "hashing-256",
      "rerank_model": rerank.LEXICAL_MODEL,
      "abstain_rerank_floor": 0.5,
      "match_margin": 0.2,
      "abstain_dense_floor": 0.3,
      "match_margin_dense": 0.1,
  }
  return config.Settings(_env_file=None, **{**values, **overrides})


def _retriever(
    db: store_lib.Store, configuration: str, **overrides: Any
) -> retriever_lib.Retriever:
  """Returns the retriever of one row of the table, on ``db``."""
  toggles = runner.CONFIGURATIONS[configuration]
  return retriever_lib.Retriever(
      db,
      embeddings.HashingEmbedder(),
      rerank.LexicalReranker(),
      _settings(**toggles, **overrides),
  )


def _question(**overrides: Any) -> questions_lib.Question:
  """Returns a paraphrase of the proxy CVE, with fields overridden."""
  values: dict[str, Any] = {
      "id": "para-01",
      "type": "paraphrase",
      "query": "Trellis proxy removes forwarded headers",
      "gold_qids": [_QID_PROXY],
      "gold_cve_ids": [_CVE_PROXY],
      "note": "From CVE-2099-2001.",
  }
  return questions_lib.Question(**{**values, **overrides})


def _negative() -> questions_lib.Question:
  """Returns a question about something the fixture does not cover."""
  return _question(
      id="neg-01",
      type="negative",
      query=_UNCOVERED_QUERY,
      gold_qids=[],
      gold_cve_ids=[],
  )


def _result(**overrides: Any) -> runner.QuestionResult:
  """Returns the result of a positive question answered perfectly in 10 ms.

  Args:
    **overrides: Fields of the result that a test sets itself.
  """
  values: dict[str, Any] = {
      "question_id": "para-01",
      "question_type": "paraphrase",
      "ranked_qids": [_QID_PROXY],
      "matched_qids": [_QID_PROXY],
      "top_score": 0.9,
      "recall": 1.0,
      "precision": 1.0,
      "reciprocal_rank": 1.0,
      "cve_recall": 1.0,
      "host_precision": 1.0,
      "host_recall": 1.0,
      "abstained_correctly": True,
      "latency_ms": 10.0,
  }
  return runner.QuestionResult(**{**values, **overrides})


# ---------------------------------------------------------------------------
# score_question
# ---------------------------------------------------------------------------


def test_score_question_positive_scores_the_context_and_the_hosts(
    store: store_lib.Store,
):
  retriever = _retriever(store, "hybrid+rerank")

  result = runner.score_question(store, retriever, _question())

  assert result.matched_qids == [_QID_PROXY]
  assert result.ranked_qids[0] == _QID_PROXY
  assert (result.recall, result.reciprocal_rank) == (1.0, 1.0)
  # Five QIDs are in the first five places and one of them is gold.
  assert result.precision == pytest.approx(0.2)
  assert result.cve_recall == 1.0
  assert (result.host_precision, result.host_recall) == (1.0, 1.0)
  assert result.abstained_correctly
  assert result.top_score == pytest.approx(0.8)


def test_score_question_positive_that_matches_nothing_loses_its_hosts(
    store: store_lib.Store,
):
  # No lexical score reaches a floor above 1, so retrieval abstains.
  retriever = _retriever(store, "hybrid+rerank", abstain_rerank_floor=1.5)

  result = runner.score_question(store, retriever, _question())

  assert not result.matched_qids
  assert not result.abstained_correctly
  assert (result.host_precision, result.host_recall) == (0.0, 0.0)
  assert result.recall == 1.0


def test_score_question_negative_that_matches_nothing_abstained_correctly(
    store: store_lib.Store,
):
  retriever = _retriever(store, "hybrid+rerank")

  result = runner.score_question(store, retriever, _negative())

  assert not result.matched_qids
  assert result.abstained_correctly
  assert result.recall is None
  assert result.precision is None
  assert result.reciprocal_rank is None
  assert result.cve_recall is None
  assert (result.host_precision, result.host_recall) == (None, None)


def test_score_question_negative_that_matches_something_did_not_abstain(
    store: store_lib.Store,
):
  # Keyword search alone has no floor and trusts its top hit.
  retriever = _retriever(store, "bm25")

  result = runner.score_question(store, retriever, _negative())

  assert result.matched_qids
  assert not result.abstained_correctly


def test_score_question_identifier_ranks_its_matches_without_a_score(
    store: store_lib.Store,
):
  retriever = _retriever(store, "hybrid+rerank")
  question = _question(
      id="ident-01",
      type="identifier",
      query="cve-2099-4001",
      gold_qids=[_QID_BROKER],
      gold_cve_ids=[_CVE_BROKER],
  )

  result = runner.score_question(store, retriever, question)

  assert result.ranked_qids == [_QID_BROKER]
  assert (result.recall, result.precision) == (1.0, 1.0)
  assert (result.host_precision, result.host_recall) == (1.0, 1.0)
  assert result.cve_recall is None
  assert result.top_score is None


@pytest.mark.parametrize(
    ("gold_cve", "cve_recall"),
    [(_CVE_QUARTZFS, 1.0), (_CVE_TIDAL, 0.0)],
    ids=["retrieved-cve", "sibling-under-the-same-qid"],
)
def test_score_question_scores_cve_recall_apart_from_qid_recall(
    store: store_lib.Store, gold_cve: str, cve_recall: float
):
  retriever = _retriever(store, "hybrid+rerank")
  question = _question(
      query=_QUARTZFS_QUERY, gold_qids=[_QID_KERNEL], gold_cve_ids=[gold_cve]
  )

  result = runner.score_question(store, retriever, question)

  assert result.recall == 1.0
  assert result.cve_recall == cve_recall


def test_score_question_top_score_is_the_best_cosine_in_fused_order(
    store: store_lib.Store,
):
  retriever = _retriever(
      store, "hybrid", abstain_dense_floor=0.2, match_margin_dense=0.2
  )
  question = _question(
      type="product",
      query=(
          "OpenSSH authentication bypass memory bit flips Trellis proxy"
          " X-Forwarded-Host headers Connection header"
      ),
      gold_qids=[_QID_OPENSSH, _QID_PROXY],
      gold_cve_ids=[],
  )
  parsed, _ = pipeline.parse_without_llm(question.query)
  cosines = [
      item.dense_score
      for item in retriever.retrieve(parsed).candidates
      if item.dense_score is not None
  ]

  result = runner.score_question(store, retriever, question)

  # Fusion puts a candidate first whose cosine is not the best.
  assert cosines[0] < max(cosines)
  assert result.top_score == max(cosines)


# ---------------------------------------------------------------------------
# summarise and format_table
# ---------------------------------------------------------------------------


def test_summarise_averages_each_metric_over_the_questions_it_applies_to():
  missed = _result(
      recall=0.0,
      precision=0.0,
      reciprocal_rank=0.0,
      cve_recall=None,
      host_precision=0.0,
      host_recall=0.0,
      abstained_correctly=False,
      latency_ms=20.0,
  )
  negative = _result(
      question_type="negative",
      recall=None,
      precision=None,
      reciprocal_rank=None,
      cve_recall=None,
      host_precision=None,
      host_recall=None,
      latency_ms=30.0,
  )

  row = runner.summarise([_result(), missed, negative])

  assert row == {
      "questions": 3,
      "recall_at_5": "0.50 (1/2)",
      "precision_at_5": "0.50 (n=2)",
      "mrr": "0.50 (n=2)",
      "cve_recall_at_5": "1.00 (1/1)",
      "host_precision": "0.50 (n=2)",
      "host_recall": "0.50 (n=2)",
      "abstention": "0.67 (2/3)",
      "latency_p50_ms": 20.0,
      "latency_p95_ms": 29.0,
  }


def test_summarise_of_no_results_is_a_row_of_zeros():
  row = runner.summarise([])

  assert row["questions"] == 0
  assert row["recall_at_5"] == "0.00 (0/0)"
  assert row["abstention"] == "0.00 (0/0)"
  assert (row["latency_p50_ms"], row["latency_p95_ms"]) == (0.0, 0.0)


def test_format_table_has_a_markdown_row_for_each_configuration():
  rows = {"bm25": runner.summarise([_result()]), "dense": runner.summarise([])}

  table = runner.format_table("All questions", rows)

  lines = table.splitlines()
  assert lines[:2] == ["All questions", ""]
  assert lines[2].startswith("| Configuration | Recall@5 | Precision@5 |")
  assert set(lines[3]) == {"|", "-"}
  assert [line.split(" | ")[0] for line in lines[4:]] == [
      "| `bm25`",
      "| `dense`",
  ]
  assert len({line.count("|") for line in lines[2:]}) == 1
  assert table.endswith("|\n")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_saves_every_question_under_each_of_the_four_configurations(
    artifact_path: pathlib.Path, tmp_path: pathlib.Path
):
  results_path = tmp_path / "out" / "results.json"
  settings = _settings(artifact_path=artifact_path)

  status = runner.run(settings, _FIXTURE_QUESTIONS, results_path)

  assert status == 0
  saved = json.loads(results_path.read_text(encoding="utf-8"))
  asked = [question.id for question in questions_lib.load(_FIXTURE_QUESTIONS)]
  assert list(saved["questions"]) == _CONFIGURATIONS
  for results in saved["questions"].values():
    assert [result["question_id"] for result in results] == asked


def test_run_saves_both_tables_and_what_they_were_measured_with(
    artifact_path: pathlib.Path, tmp_path: pathlib.Path
):
  results_path = tmp_path / "results.json"
  settings = _settings(artifact_path=artifact_path)

  runner.run(settings, _FIXTURE_QUESTIONS, results_path)

  saved = json.loads(results_path.read_text(encoding="utf-8"))
  assert [list(rows) for rows in saved["tables"].values()] == (
      [_CONFIGURATIONS, _CONFIGURATIONS]
  )
  assert saved["embedding_model"] == "hashing-256"
  assert saved["rerank_model"] == "lexical"
  assert saved["abstain_rerank_floor"] == 0.5
  assert all(saved["dataset_sha256"].values())


def test_run_prints_a_table_of_all_questions_and_one_of_search_questions(
    artifact_path: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  settings = _settings(artifact_path=artifact_path)

  runner.run(settings, _FIXTURE_QUESTIONS, tmp_path / "results.json")

  printed = capsys.readouterr().out
  assert printed.startswith("All questions\n")
  assert "\nSearch questions only (paraphrase, product, negative)\n" in printed
  assert printed.count("| `hybrid+rerank` |") == 2


def test_run_on_a_question_set_without_search_questions_succeeds(
    artifact_path: pathlib.Path, tmp_path: pathlib.Path
):
  questions_path = tmp_path / "questions.jsonl"
  identifier = _question(
      id="ident-01",
      type="identifier",
      query=_CVE_BROKER,
      gold_qids=[_QID_BROKER],
      gold_cve_ids=[_CVE_BROKER],
  )
  questions_path.write_text(
      identifier.model_dump_json() + "\n", encoding="utf-8"
  )
  settings = _settings(artifact_path=artifact_path)

  status = runner.run(settings, questions_path, tmp_path / "results.json")

  assert status == 0


def test_run_with_gold_the_artifact_lacks_returns_1_and_says_why(
    artifact_path: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  questions_path = tmp_path / "questions.jsonl"
  drifted = _question(gold_qids=["999999"], gold_cve_ids=[])
  questions_path.write_text(drifted.model_dump_json() + "\n", encoding="utf-8")
  results_path = tmp_path / "results.json"
  settings = _settings(artifact_path=artifact_path)

  status = runner.run(settings, questions_path, results_path)

  assert status == 1
  assert capsys.readouterr().err == (
      f"{questions_path}: para-01: gold QID 999999 is not in the artifact\n"
  )
  assert not results_path.exists()
