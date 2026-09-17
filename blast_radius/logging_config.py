"""Structured logging: one JSON object per line on stderr.

Every line carries ``ts``, ``level``, ``logger`` and ``message``, then
whatever the call site passed as ``extra=`` (``step``, ``duration_ms``, ...),
and ``exc`` with the traceback when an exception was logged. Lines are meant
for a log pipeline, not for reading raw: a traceback stays inside its line
instead of spilling over the next twenty.

``request_id_var`` ties the lines of one request together. The API sets it
once per request, and the formatter stamps it on every line logged in that
context, so no call site has to pass the id along.
"""

import contextvars
import datetime
import json
import logging
import sys
from typing import Any

# The id of the request being handled in the current context, if any.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Keys the formatter writes itself. An ``extra=`` field with one of these
# names is dropped, so that they mean the same thing on every line.
_FIXED_KEYS = frozenset({"ts", "level", "logger", "message", "exc"})

# Attributes that every log record has; anything else on a record came in
# through ``extra=``. They are read off a blank record, not listed by hand,
# because the set grows between Python versions. ``message`` and ``asctime``
# are added by the standard formatter, which another handler may have run on
# the same record first.
_RECORD_ATTRIBUTES = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
}


class _JsonFormatter(logging.Formatter):
  """Renders a log record as a single line of JSON."""

  def format(self, record: logging.LogRecord) -> str:
    """Returns ``record`` as a JSON object, fixed keys first.

    Args:
      record: The record to render.
    """
    timestamp = datetime.datetime.fromtimestamp(
        record.created, tz=datetime.timezone.utc
    )
    payload: dict[str, Any] = {
        "ts": timestamp.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    # Set before the extras, so that a call site that names another request
    # explicitly overrides the ambient id.
    request_id = request_id_var.get()
    if request_id is not None:
      payload["request_id"] = request_id
    for key, value in vars(record).items():
      if key not in _RECORD_ATTRIBUTES and key not in _FIXED_KEYS:
        payload[key] = value
    # ``exc_info=True`` outside an ``except`` block yields a triple of Nones.
    if record.exc_info and record.exc_info[0] is not None:
      payload["exc"] = self.formatException(record.exc_info)
    # A log call must never fail over what it was asked to log, so a value
    # that JSON cannot express, such as a path, is written as its ``str()``.
    return json.dumps(payload, default=str)


def configure(level: str) -> None:
  """Sends every log record to stderr as one JSON object per line.

  Safe to call more than once: the handler from an earlier call is replaced,
  never added to, and the new level applies. Handlers that other code
  attached to the root logger are left alone.

  Args:
    level: A standard level name such as ``INFO``, in either case.

  Raises:
    ValueError: If ``level`` is not a level name that ``logging`` knows.
  """
  number = logging.getLevelNamesMapping().get(level.upper())
  if number is None:
    raise ValueError(f"unknown log level: {level!r}")

  root = logging.getLogger()
  for stale in list(root.handlers):
    if isinstance(stale.formatter, _JsonFormatter):
      root.removeHandler(stale)
      stale.close()
  # Replaced rather than kept, so that the handler writes to whatever
  # ``sys.stderr`` is now, not to a stream that has since been swapped out.
  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(_JsonFormatter())
  root.addHandler(handler)
  root.setLevel(number)
