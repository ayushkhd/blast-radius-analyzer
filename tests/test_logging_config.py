"""Tests for blast_radius.logging_config."""

from collections.abc import Iterator
import contextvars
import json
import logging
import pathlib
import re
from typing import Any

import pytest

from blast_radius import logging_config

_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
  """Puts the root logger back as it was, so no test affects another."""
  root = logging.getLogger()
  handlers = list(root.handlers)
  level = root.level
  yield
  for handler in list(root.handlers):
    root.removeHandler(handler)
  for handler in handlers:
    root.addHandler(handler)
  root.setLevel(level)


def _lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
  """Returns what stderr holds, parsed as one JSON object per line."""
  return [json.loads(line) for line in capsys.readouterr().err.splitlines()]


# ---------------------------------------------------------------------------
# The shape of a line
# ---------------------------------------------------------------------------


def test_configure_writes_to_stderr_and_not_to_stdout(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info("hello")

  captured = capsys.readouterr()
  assert captured.out == ""
  assert len(captured.err.splitlines()) == 1


def test_line_carries_timestamp_level_logger_and_message_in_that_order(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("blast_radius.ingest").info("built %d chunks", 3)

  (line,) = _lines(capsys)
  assert list(line) == ["ts", "level", "logger", "message"]
  assert _TIMESTAMP.fullmatch(line["ts"])
  assert line["level"] == "INFO"
  assert line["logger"] == "blast_radius.ingest"
  assert line["message"] == "built 3 chunks"


def test_timestamp_is_the_record_time_in_utc_with_milliseconds(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")
  record = logging.makeLogRecord(
      {
          "name": "app",
          "levelno": logging.INFO,
          "levelname": "INFO",
          "msg": "hello",
          "created": 1700000000.123,
      }
  )

  logging.getLogger().handle(record)

  (line,) = _lines(capsys)
  assert line["ts"] == "2023-11-14T22:13:20.123Z"


def test_each_record_is_one_line_even_when_the_message_has_newlines(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info("first\nsecond")
  logging.getLogger("app").info("third")

  assert [line["message"] for line in _lines(capsys)] == [
      "first\nsecond",
      "third",
  ]


# ---------------------------------------------------------------------------
# Extra fields
# ---------------------------------------------------------------------------


def test_extra_fields_are_added_after_the_fixed_keys(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info(
      "step done", extra={"step": "rank", "duration_ms": 12.5, "hosts": 341}
  )

  (line,) = _lines(capsys)
  assert list(line)[4:] == ["step", "duration_ms", "hosts"]
  assert line["step"] == "rank"
  assert line["duration_ms"] == 12.5
  assert line["hosts"] == 341


def test_extra_field_with_a_json_structure_keeps_it(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info(
      "retrieved",
      extra={"summary": {"qids": ["100", "200"], "abstained": False}},
  )

  (line,) = _lines(capsys)
  assert line["summary"] == {"qids": ["100", "200"], "abstained": False}


def test_extra_value_that_json_cannot_express_is_rendered_with_str(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info(
      "opened",
      extra={
          "artifact": pathlib.Path("artifacts/index.sqlite"),
          "stages": {"keyword"},
      },
  )

  (line,) = _lines(capsys)
  assert line["artifact"] == "artifacts/index.sqlite"
  assert line["stages"] == "{'keyword'}"


def test_extra_field_cannot_overwrite_a_fixed_key(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info(
      "hello",
      extra={"ts": "never", "level": "LOUD", "logger": "other", "exc": "none"},
  )

  (line,) = _lines(capsys)
  assert _TIMESTAMP.fullmatch(line["ts"])
  assert line["level"] == "INFO"
  assert line["logger"] == "app"
  assert "exc" not in line


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_logged_exception_is_rendered_into_exc_on_the_same_line(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  try:
    raise ValueError("boom")
  except ValueError:
    logging.getLogger("app").exception("step failed")

  (line,) = _lines(capsys)
  assert line["level"] == "ERROR"
  assert line["message"] == "step failed"
  assert line["exc"].startswith("Traceback (most recent call last):")
  assert line["exc"].endswith("ValueError: boom")


def test_line_without_an_exception_has_no_exc_key(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").error("failed, but nothing was raised")

  (line,) = _lines(capsys)
  assert "exc" not in line


def test_exc_info_outside_an_except_block_adds_no_exc_key(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").error("nothing is being handled", exc_info=True)

  (line,) = _lines(capsys)
  assert "exc" not in line


# ---------------------------------------------------------------------------
# request_id_var
# ---------------------------------------------------------------------------


def test_request_id_from_the_context_is_stamped_on_every_line(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  def handle_request() -> None:
    logging_config.request_id_var.set("req-42")
    logging.getLogger("app.parse").info("parsed")
    logging.getLogger("app.rank").info("ranked", extra={"step": "rank"})

  contextvars.copy_context().run(handle_request)

  lines = _lines(capsys)
  assert [line["request_id"] for line in lines] == ["req-42", "req-42"]
  assert list(lines[1])[4:] == ["request_id", "step"]


def test_line_logged_outside_a_request_has_no_request_id(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  logging.getLogger("app").info("starting up")

  (line,) = _lines(capsys)
  assert "request_id" not in line


def test_request_id_passed_as_extra_overrides_the_one_from_the_context(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")

  def handle_request() -> None:
    logging_config.request_id_var.set("req-42")
    logging.getLogger("app").info("retried", extra={"request_id": "req-7"})

  contextvars.copy_context().run(handle_request)

  (line,) = _lines(capsys)
  assert line["request_id"] == "req-7"


def test_request_id_var_is_unset_by_default():
  assert logging_config.request_id_var.get() is None


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


def test_configure_called_twice_writes_each_record_once(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("INFO")
  logging_config.configure("INFO")

  logging.getLogger("app").info("once")

  assert len(_lines(capsys)) == 1


def test_configure_called_twice_installs_a_single_handler():
  before = len(logging.getLogger().handlers)

  logging_config.configure("INFO")
  logging_config.configure("DEBUG")

  assert len(logging.getLogger().handlers) == before + 1


def test_configure_leaves_other_handlers_on_the_root_logger_alone():
  other = logging.NullHandler()
  logging.getLogger().addHandler(other)

  logging_config.configure("INFO")
  logging_config.configure("INFO")

  assert other in logging.getLogger().handlers


def test_configure_drops_records_below_the_level(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("WARNING")

  logging.getLogger("app").info("routine")
  logging.getLogger("app").warning("unusual")

  assert [line["message"] for line in _lines(capsys)] == ["unusual"]


def test_configure_called_again_applies_the_new_level(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("ERROR")
  logging_config.configure("DEBUG")

  logging.getLogger("app").debug("detail")

  assert [line["level"] for line in _lines(capsys)] == ["DEBUG"]


def test_configure_accepts_a_level_name_in_lower_case(
    capsys: pytest.CaptureFixture[str],
):
  logging_config.configure("debug")

  logging.getLogger("app").debug("detail")

  assert len(_lines(capsys)) == 1


@pytest.mark.parametrize("level", ["LOUD", "", "10"])
def test_configure_with_an_unknown_level_raises(level: str):
  with pytest.raises(ValueError, match="unknown log level"):
    logging_config.configure(level)


def test_configure_with_an_unknown_level_leaves_logging_as_it_was():
  root = logging.getLogger()
  handlers = list(root.handlers)
  level = root.level

  with pytest.raises(ValueError):
    logging_config.configure("LOUD")

  assert root.handlers == handlers
  assert root.level == level
