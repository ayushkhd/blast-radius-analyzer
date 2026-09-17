"""The evaluation question set: its shape, its loader and its checks.

``eval/questions.jsonl`` holds the questions that every retrieval
configuration is scored against, one JSON object per line. Every number the
evaluation reports rests on the gold answers in that file, so they are
checked twice:

* ``load`` rejects a file that is not well formed: a line that is not JSON,
  a missing or unknown field, a value of the wrong type, an id used twice.
  It needs no artifact, so it also runs where the scanner exports are not
  available.
* ``validate`` checks loaded questions against an index artifact, and
  returns every problem instead of stopping at the first, so that a question
  set that has drifted from the data can be repaired in one pass.

Hosts attach to QIDs, not to CVEs, so the gold answer of a question is a set
of QIDs, and the gold host set follows from it by SQL when the question is
scored. A question written from one specific CVE records that CVE as well,
which lets retrieval be scored at CVE level beside the QID level. A negative
question has no gold at all: the right answer to it is to match nothing.
"""

import collections
from collections.abc import Sequence
import pathlib
from typing import Literal

import pydantic

from blast_radius import models
from blast_radius import store

QuestionType = Literal["identifier", "paraphrase", "product", "negative"]


class QuestionSetError(Exception):
  """A question file is malformed; the message names the file and line."""


class Question(models.Model):
  """One evaluation question with its gold answer.

  Attributes:
    id: Stable identifier, unique within a question set, e.g. ``para-11``.
    type: ``identifier`` for CVE ids and QIDs that are looked up by primary
      key, ``paraphrase`` for one write-up retold in other words,
      ``product`` for a product-and-version query whose gold spans several
      QIDs, and ``negative`` for something the corpus does not cover.
    query: What the analyst types or pastes.
    gold_qids: The QIDs that answer the question, sorted as strings, each
      once. Empty for a negative question.
    gold_cve_ids: The CVEs the question was written from, sorted as strings,
      each once. Empty when it was written from a QID's diagnosis or from a
      whole product.
    note: What the question is derived from and what it is meant to probe.
  """

  id: str = pydantic.Field(min_length=1)
  type: QuestionType
  query: str = pydantic.Field(min_length=1)
  gold_qids: list[str]
  gold_cve_ids: list[str]
  note: str = pydantic.Field(min_length=1)

  @pydantic.field_validator("gold_qids", "gold_cve_ids")
  @classmethod
  def _require_canonical_order(cls, ids: list[str]) -> list[str]:
    """Rejects gold ids that repeat or are out of order.

    One canonical spelling of a gold set keeps a repeated id from slipping
    in unnoticed and keeps the diffs of the question file small.
    """
    if ids != sorted(set(ids)):
      raise ValueError("ids must be sorted and must not repeat")
    return ids

  @property
  def is_negative(self) -> bool:
    """Returns whether the right answer is to match nothing."""
    return self.type == "negative"


def _describe(error: pydantic.ValidationError) -> str:
  """Returns what is wrong with one line, as a single line of text.

  pydantic's own rendering runs over several lines and carries a
  documentation URL for every failure.
  """
  failures = []
  for failure in error.errors():
    field = ".".join(str(part) for part in failure["loc"])
    message = failure["msg"]
    failures.append(f"{field}: {message}" if field else message)
  return "; ".join(failures)


def load(path: pathlib.Path) -> list[Question]:
  """Returns the questions of a JSON Lines file, in file order.

  Args:
    path: A file with one JSON object per line. Blank lines are ignored.

  Raises:
    QuestionSetError: If a line is not valid JSON, does not fit ``Question``,
      or repeats the id of an earlier line.
    OSError: If the file cannot be read.
  """
  questions: list[Question] = []
  line_of_id: dict[str, int] = {}
  # Iterating the file splits on newlines only. str.splitlines() would also
  # split on U+2028, which JSON allows unescaped inside a string.
  with path.open(encoding="utf-8") as lines:
    for number, line in enumerate(lines, start=1):
      if not line.strip():
        continue
      try:
        question = Question.model_validate_json(line)
      except pydantic.ValidationError as error:
        raise QuestionSetError(
            f"{path}:{number}: {_describe(error)}"
        ) from error
      if question.id in line_of_id:
        raise QuestionSetError(
            f"{path}:{number}: id {question.id!r} is already used on line"
            f" {line_of_id[question.id]}"
        )
      line_of_id[question.id] = number
      questions.append(question)
  return questions


def _gold_problems(question: Question, db: store.Store) -> list[str]:
  """Returns what is wrong with the gold answer of one question.

  Args:
    question: The question to check.
    db: The artifact its gold ids must refer to.
  """
  problems = []
  if question.is_negative:
    if question.gold_qids or question.gold_cve_ids:
      problems.append("a negative question must have no gold ids")
  elif not question.gold_qids:
    problems.append(f"a {question.type} question needs at least one gold QID")

  for qid in question.gold_qids:
    if db.get_qid(qid) is None:
      problems.append(f"gold QID {qid} is not in the artifact")

  for cve_id in question.gold_cve_ids:
    if db.get_cve(cve_id) is None:
      problems.append(f"gold CVE {cve_id} is not in the artifact")
      continue
    missing = ", ".join(
        qid for qid in db.qids_for_cve(cve_id) if qid not in question.gold_qids
    )
    if missing:
      problems.append(
          f"gold CVE {cve_id} also maps to QID {missing}, which is not a gold"
          " QID"
      )
  return problems


def validate(questions: Sequence[Question], db: store.Store) -> list[str]:
  """Returns every inconsistency between a question set and an artifact.

  The rules, each of which guards a number the evaluation reports:

  * Ids are unique, so that a result can be traced to one question.
  * A negative question has no gold ids, and every other question has at
    least one gold QID. Negatives are scored on abstention and the rest on
    what was retrieved, so a question on the wrong side of that line
    distorts both.
  * Every gold QID is in the artifact. A QID without a write-up counts: its
    hosts are still found through the detections.
  * Every gold CVE is in the artifact, and every QID that the scanner maps
    it to is a gold QID. A CVE can sit under more than one QID, each with
    hosts of its own, so gold that names only one of them would mark a
    complete answer down as imprecise. The converse is not required: gold
    QIDs need not come from a gold CVE, because a question may also name a
    QID directly or ask about a whole product.

  Args:
    questions: The question set, typically the result of ``load``.
    db: The artifact the questions will be scored against.

  Returns:
    One human-readable line per problem, each starting with the id of the
    question it concerns. Empty when the set is consistent with ``db``.
  """
  problems = []
  uses = collections.Counter(question.id for question in questions)
  for question_id, count in uses.items():
    if count > 1:
      problems.append(f"{question_id}: id is used by {count} questions")
  for question in questions:
    problems.extend(
        f"{question.id}: {problem}" for problem in _gold_problems(question, db)
    )
  return problems
