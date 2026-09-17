"""Builds the FTS5 ``MATCH`` expression for keyword search.

User text must never reach ``MATCH`` as written. FTS5 has a query language
of its own (``AND``, ``OR``, ``NOT``, ``NEAR``, quotes, ``*``, ``^``, ``-``,
parentheses, ``column:`` filters), so raw input is at best a different query
from the one intended and at worst an error: ``up to 9.6`` is a syntax
error, and ``ssh-agent`` fails with "no such column: agent".

``build_match_query`` therefore keeps only the words of the input. It
tokenises the text, drops what is not worth searching for, wraps every term
in double quotes and joins the terms with ``OR``. Inside double quotes FTS5
reads nothing as syntax, and a term can never hold a quote itself, so the
expression is safe whatever was typed. ``OR`` rather than the implicit
``AND`` lets BM25 rank the chunks that share the most, and the rarest, terms
with a paraphrased advisory, which no single chunk matches word for word.

This module only builds the expression. Running it is the store's job.
"""

import re

DEFAULT_MAX_TERMS = 32

# A term is a version number kept whole (digits, then dot-separated
# alphanumeric parts: "9.6", "2.4.55", "9.8p1"), or else a run of letters,
# digits and underscores. The table's unicode61 tokeniser splits on dots and
# underscores, so FTS5 reads the quoted terms "9.8p1" and "nfs_net_init" as
# the phrases 9 + 8p1 and nfs + net + init. A phrase matches only where its
# parts are adjacent, which is exactly how the indexed text was split, and
# that is far more selective than the parts OR-ed together.
_TERM = re.compile(r"\d+(?:\.[^\W_]+)+|\w+")

# Numbers shorter than this are dropped unless they are part of a version.
# Decided by querying the reference corpus in a real FTS5 table. The parts
# of a version are noise on their own: "9" and "6" each match around a
# hundred of its thousand chunks; the "55" of "2.4.55" matches twelve, eight
# of them kernel traces in which it is an opcode byte or a line number; the
# "59" of "2.4.59" matches a PID and a "#59" in kernel logs. The version as
# a phrase matches the chunks that state that version and nothing else.
# Numbers of three digits or more (ports, years, error codes) are rare
# enough to be worth a term.
_MIN_NUMBER_LENGTH = 3

# Lucene's default English stop set, plus the question words and pronouns
# of an analyst's question ("which hosts do we have with ..."). BM25 gives
# such words almost no weight, but each one matches most of the table and
# uses up one of the ``max_terms``. "can" is deliberately absent: it is
# also a kernel subsystem.
_STOP_WORDS = frozenset("""
    a an and any are as at be been but by did do does for from had has have
    how i if in into is it its me my no not of on or our such that the their
    then there these they this to us was we were what when where which who
    why will with would you your
    """.split())


def _is_searchable(term: str) -> bool:
  """Returns whether ``term`` is worth a place in the query."""
  if "." in term:
    return True  # A version number, which is specific whatever its parts.
  if len(term) < 2 or term in _STOP_WORDS:
    return False
  return not (term.isdigit() and len(term) < _MIN_NUMBER_LENGTH)


def build_match_query(
    text: str, *, max_terms: int = DEFAULT_MAX_TERMS
) -> str | None:
  """Returns an FTS5 ``MATCH`` expression for ``text``, or None.

  Terms are lower-cased and de-duplicated, and keep the order in which they
  were typed. The FTS5 keywords are words like any other here: ``AND``,
  ``OR`` and ``NOT`` go with the stop words, and ``NEAR`` is searched for.

  Args:
    text: Free text as typed: a question, or a pasted advisory.
    max_terms: Upper bound on the number of terms. A pasted advisory can run
      to hundreds of words, and every term is another posting list to scan.

  Returns:
    Double-quoted terms joined with ``OR``, or None when ``text`` holds
    nothing worth searching for.

  Raises:
    ValueError: If ``max_terms`` is not positive.
  """
  if max_terms <= 0:
    raise ValueError(f"max_terms must be positive, got {max_terms}")

  # unicode61 treats a leading or trailing underscore as a separator.
  terms = (term.strip("_") for term in _TERM.findall(text.lower()))
  searchable = dict.fromkeys(term for term in terms if _is_searchable(term))
  if not searchable:
    return None
  return " OR ".join(f'"{term}"' for term in list(searchable)[:max_terms])
