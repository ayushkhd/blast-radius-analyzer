"""Tests for blast_radius.evaluation.questions."""

import collections
import json
import pathlib
from typing import Any

import pytest

from blast_radius import store as store_lib
from blast_radius.evaluation import questions

_REPOSITORY = pathlib.Path(__file__).parent.parent
_FIXTURE_QUESTIONS = pathlib.Path(__file__).parent / "fixtures/questions.jsonl"
_COMMITTED_QUESTIONS = _REPOSITORY / "eval/questions.jsonl"

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_QID_PROXY = "710003"
_QID_WEB_SERVER = "710004"
_CVE_PROXY = "CVE-2099-2001"


def _record(**overrides: Any) -> dict[str, Any]:
  """Returns a valid question as a JSON object, with fields overridden."""
  record = {
      "id": "para-01",
      "type": "paraphrase",
      "query": "A client can make the proxy drop the headers it adds.",
      "gold_qids": [_QID_PROXY],
      "gold_cve_ids": [_CVE_PROXY],
      "note": "From CVE-2099-2001.",
  }
  return {**record, **overrides}


def _question(**overrides: Any) -> questions.Question:
  """Returns a valid question, with fields overridden."""
  return questions.Question(**_record(**overrides))


def _write(path: pathlib.Path, *lines: str) -> pathlib.Path:
  """Writes ``lines`` to ``path``, one per line, and returns ``path``."""
  path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
  return path


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_returns_the_questions_in_file_order():
  loaded = questions.load(_FIXTURE_QUESTIONS)

  assert [question.id for question in loaded] == [
      "ident-01",
      "ident-02",
      "para-01",
      "para-02",
      "prod-01",
      "neg-01",
  ]
  assert loaded[2].gold_qids == [_QID_PROXY]
  assert loaded[2].gold_cve_ids == [_CVE_PROXY]


def test_load_ignores_blank_lines(tmp_path: pathlib.Path):
  first = json.dumps(_record(id="para-01"))
  second = json.dumps(_record(id="para-02"))
  path = _write(tmp_path / "q.jsonl", "", first, "   ", second, "")

  loaded = questions.load(path)

  assert [question.id for question in loaded] == ["para-01", "para-02"]


def test_load_rejects_malformed_json_naming_the_file_and_line(
    tmp_path: pathlib.Path,
):
  path = _write(tmp_path / "q.jsonl", json.dumps(_record()), "", '{"id": ')

  with pytest.raises(questions.QuestionSetError) as raised:
    questions.load(path)

  assert str(raised.value).startswith(f"{path}:3: Invalid JSON")


@pytest.mark.parametrize(
    ("overrides", "complaint"),
    [
        ({"type": "trivia"}, "type: Input should be"),
        ({"query": ""}, "query: String should have at least 1 character"),
        (
            {"gold_qids": [710003]},
            "gold_qids.0: Input should be a valid string",
        ),
        ({"gold_qids": ["710004", "710003"]}, "gold_qids: Value error, ids"),
        ({"gold_cve_ids": [_CVE_PROXY, _CVE_PROXY]}, "gold_cve_ids: Value"),
        ({"answer": "710003"}, "answer: Extra inputs are not permitted"),
    ],
)
def test_load_rejects_a_line_that_does_not_fit_the_schema(
    tmp_path: pathlib.Path, overrides: dict[str, Any], complaint: str
):
  path = _write(tmp_path / "q.jsonl", json.dumps(_record(**overrides)))

  with pytest.raises(questions.QuestionSetError) as raised:
    questions.load(path)

  assert str(raised.value).startswith(f"{path}:1: {complaint}")


def test_load_rejects_a_line_with_a_missing_field(tmp_path: pathlib.Path):
  record = _record()
  del record["gold_cve_ids"]
  path = _write(tmp_path / "q.jsonl", json.dumps(record))

  with pytest.raises(questions.QuestionSetError) as raised:
    questions.load(path)

  assert str(raised.value) == f"{path}:1: gold_cve_ids: Field required"


def test_load_rejects_a_repeated_id_naming_both_lines(tmp_path: pathlib.Path):
  line = json.dumps(_record(id="para-07"))
  path = _write(tmp_path / "q.jsonl", line, json.dumps(_record()), line)

  with pytest.raises(questions.QuestionSetError) as raised:
    questions.load(path)

  assert str(raised.value) == (
      f"{path}:3: id 'para-07' is already used on line 1"
  )


def test_load_of_a_missing_file_raises_the_operating_system_error(
    tmp_path: pathlib.Path,
):
  with pytest.raises(FileNotFoundError):
    questions.load(tmp_path / "absent.jsonl")


def test_committed_question_set_loads_with_the_documented_composition():
  loaded = questions.load(_COMMITTED_QUESTIONS)

  assert collections.Counter(question.type for question in loaded) == {
      "identifier": 10,
      "paraphrase": 20,
      "product": 5,
      "negative": 5,
  }


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------


def test_question_is_negative_only_for_the_negative_type():
  negative = _question(type="negative", gold_qids=[], gold_cve_ids=[])

  assert negative.is_negative
  assert not _question(type="product").is_negative


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_accepts_the_fixture_questions(store: store_lib.Store):
  loaded = questions.load(_FIXTURE_QUESTIONS)

  assert not questions.validate(loaded, store)


def test_validate_reports_a_gold_qid_that_the_artifact_lacks(
    store: store_lib.Store,
):
  question = _question(gold_qids=[_QID_PROXY, "799999"])

  assert questions.validate([question], store) == [
      "para-01: gold QID 799999 is not in the artifact"
  ]


def test_validate_reports_a_gold_cve_that_the_artifact_lacks(
    store: store_lib.Store,
):
  question = _question(gold_cve_ids=["CVE-2099-9999"])

  assert questions.validate([question], store) == [
      "para-01: gold CVE CVE-2099-9999 is not in the artifact"
  ]


def test_validate_reports_a_gold_cve_whose_qid_is_not_gold(
    store: store_lib.Store,
):
  question = _question(gold_qids=[_QID_WEB_SERVER])

  assert questions.validate([question], store) == [
      f"para-01: gold CVE {_CVE_PROXY} also maps to QID {_QID_PROXY}, which"
      " is not a gold QID"
  ]


def test_validate_allows_gold_qids_beyond_those_of_the_gold_cves(
    store: store_lib.Store,
):
  question = _question(gold_qids=[_QID_PROXY, _QID_WEB_SERVER])

  assert not questions.validate([question], store)


def test_validate_reports_a_negative_question_that_has_gold(
    store: store_lib.Store,
):
  question = _question(id="neg-01", type="negative")

  assert questions.validate([question], store) == [
      "neg-01: a negative question must have no gold ids"
  ]


def test_validate_reports_a_question_without_gold_that_is_not_negative(
    store: store_lib.Store,
):
  question = _question(
      id="prod-01", type="product", gold_qids=[], gold_cve_ids=[]
  )

  assert questions.validate([question], store) == [
      "prod-01: a product question needs at least one gold QID"
  ]


def test_validate_reports_an_id_used_twice_once(store: store_lib.Store):
  repeated = [_question(), _question(), _question(id="para-02")]

  assert questions.validate(repeated, store) == [
      "para-01: id is used by 2 questions"
  ]


def test_validate_returns_every_problem_in_question_order(
    store: store_lib.Store,
):
  broken = [
      _question(id="para-01", gold_qids=["799999"], gold_cve_ids=[]),
      _question(id="neg-01", type="negative", gold_cve_ids=[]),
  ]

  assert questions.validate(broken, store) == [
      "para-01: gold QID 799999 is not in the artifact",
      "neg-01: a negative question must have no gold ids",
  ]
