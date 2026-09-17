"""Tests for blast_radius.retrieval.keyword.

The expressions are run against a real FTS5 table built from the artifact's
schema, because what matters is how SQLite reads them.
"""

from collections.abc import Iterator
import re
import sqlite3

import pytest

from blast_radius import schema
from blast_radius.retrieval import keyword

_DOCUMENTS = [
    (
        1,
        "OpenSSH",
        "OpenSSH up to version 9.6 allows an authentication bypass.",
    ),
    (2, "Telnet", "The telnet daemon sends credentials in clear text."),
    (3, "Remote access", "Both ssh and telnet are enabled on the jump host."),
    (
        4,
        "Kernel nfs",
        "nfs: handle error of rpc_proc_register() in nfs_net_init().",
    ),
    (
        5,
        "Kernel netdev",
        "Do not init the net device before nfs, near the probe.",
    ),
    (6, "Exit codes", "The job failed with error 9 on line 6 of the script."),
    (7, "Release notes", "Version 6.9 changes how foo is parsed."),
]

_HOSTILE_INPUTS = [
    pytest.param('"ssh AND (', id="open-quote-and-parenthesis"),
    pytest.param("NEAR(a b", id="unclosed-near"),
    pytest.param("title:foo", id="column-filter"),
    pytest.param("nosuchcolumn:foo", id="unknown-column-filter"),
    pytest.param("{title body}: ssh", id="column-set-filter"),
    pytest.param('"unbalanced', id="unbalanced-quote"),
    pytest.param("a* -b ^c", id="prefix-exclusion-and-anchor"),
    pytest.param("ssh* -telnet ^openssh", id="operators-on-real-words"),
    pytest.param("'; DROP TABLE chunks;--", id="sql-injection"),
    pytest.param("\U0001f600 \U0001f525\U0001f4a5", id="emoji"),
    pytest.param("!@#$%^&*()" * 500, id="5000-characters-of-punctuation"),
    pytest.param('"' * 5000, id="5000-double-quotes"),
    pytest.param("the and of or not", id="only-stop-words"),
    pytest.param("AND", id="lone-and"),
    pytest.param("OR OR OR", id="only-or"),
    pytest.param("ssh NOT", id="dangling-not"),
    pytest.param("ssh NEAR/2 telnet", id="near-with-distance"),
    pytest.param("ssh-agent", id="hyphenated-word"),
    pytest.param("up to 9.6", id="version-number"),
    pytest.param("ssh + telnet", id="phrase-concatenation"),
    pytest.param("back\\slash", id="backslash"),
    pytest.param("ssh\x00telnet", id="nul-byte"),
    pytest.param("ssh \ud800 telnet", id="lone-surrogate"),
    pytest.param("ssh\n\ttelnet\r\n", id="control-whitespace"),
    pytest.param(" ".join(f"word{i}" for i in range(5000)), id="5000-words"),
    pytest.param(".".join(["1"] * 2500), id="2500-part-version"),
]

# The only shape an expression may take: quoted terms made of word
# characters and dots, joined by OR.
_QUOTED_TERMS = re.compile(r'"[\w.]+"(?: OR "[\w.]+")*')


@pytest.fixture(name="index")
def _index() -> Iterator[sqlite3.Connection]:
  """Yields a database with the artifact's schema and a few chunks indexed."""
  connection = sqlite3.connect(":memory:")
  connection.executescript(schema.SCHEMA)
  connection.executemany(
      "INSERT INTO chunks_fts(rowid, title, body) VALUES (?, ?, ?)", _DOCUMENTS
  )
  yield connection
  connection.close()


def _search(connection: sqlite3.Connection, expression: str) -> set[int]:
  """Returns the rowids of the chunks that ``expression`` matches."""
  rows = connection.execute(
      "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (expression,)
  )
  return {rowid for (rowid,) in rows}


def _executes(connection: sqlite3.Connection, expression: str) -> bool:
  """Returns whether FTS5 accepts ``expression``."""
  try:
    _search(connection, expression)
  except sqlite3.OperationalError:
    return False
  return True


def test_build_match_query_quotes_each_term_and_joins_them_with_or():
  expression = keyword.build_match_query("OpenSSH auth bypass")

  assert expression == '"openssh" OR "auth" OR "bypass"'


def test_build_match_query_removes_repeats_and_keeps_the_typed_order():
  expression = keyword.build_match_query("Telnet SSH telnet ssh TELNET")

  assert expression == '"telnet" OR "ssh"'


def test_build_match_query_drops_stop_words_and_single_characters():
  expression = keyword.build_match_query(
      "Which hosts are affected by a flaw in the X server?"
  )

  assert expression == '"hosts" OR "affected" OR "flaw" OR "server"'


@pytest.mark.parametrize(
    "text", ["", " \n\t ", "the and of", "a b c", "?!...", "9 6 55", "_ __"]
)
def test_build_match_query_nothing_searchable_is_none(text: str):
  assert keyword.build_match_query(text) is None


def test_build_match_query_keeps_a_version_number_whole():
  expression = keyword.build_match_query("OpenSSH before 9.6, 9.8p1 or 2.4.55")

  assert expression == '"openssh" OR "before" OR "9.6" OR "9.8p1" OR "2.4.55"'


def test_build_match_query_drops_short_bare_numbers_and_keeps_long_ones():
  expression = keyword.build_match_query("port 22 or 8080, error 404 in 2024")

  assert expression == '"port" OR "8080" OR "error" OR "404" OR "2024"'


def test_build_match_query_keeps_underscores_inside_a_term_only():
  expression = keyword.build_match_query("__init__ nfs_net_init _x_")

  assert expression == '"init" OR "nfs_net_init"'


def test_build_match_query_caps_the_number_of_terms():
  text = " ".join(f"word{i}" for i in range(100))

  expression = keyword.build_match_query(text)

  assert expression is not None
  assert expression.count('"') == 2 * keyword.DEFAULT_MAX_TERMS
  assert expression.startswith('"word0" OR "word1" OR')


def test_build_match_query_honours_max_terms():
  expression = keyword.build_match_query("alpha beta gamma delta", max_terms=2)

  assert expression == '"alpha" OR "beta"'


@pytest.mark.parametrize("max_terms", [0, -1])
def test_build_match_query_max_terms_not_positive_is_rejected(max_terms: int):
  with pytest.raises(ValueError):
    keyword.build_match_query("openssh", max_terms=max_terms)


@pytest.mark.parametrize("text", _HOSTILE_INPUTS)
def test_build_match_query_hostile_input_is_none_or_executes(
    index: sqlite3.Connection, text: str
):
  expression = keyword.build_match_query(text)

  assert expression is None or _executes(index, expression)


@pytest.mark.parametrize("text", _HOSTILE_INPUTS)
def test_build_match_query_emits_nothing_but_quoted_terms(text: str):
  expression = keyword.build_match_query(text)

  assert expression is None or _QUOTED_TERMS.fullmatch(expression)


@pytest.mark.parametrize(
    "text",
    [
        '"ssh AND (',
        "NEAR(a b",
        '"unbalanced',
        "a* -b ^c",
        "'; DROP TABLE chunks;--",
        "ssh-agent",
        "up to 9.6",
    ],
)
def test_text_as_typed_is_not_a_valid_match_expression(
    index: sqlite3.Connection, text: str
):
  assert not _executes(index, text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # As an operator AND would keep chunk 3 alone, which has both words.
        ("ssh AND telnet", {2, 3}),
        # As an operator NOT would drop chunk 3, which mentions ssh.
        ("telnet NOT ssh", {2, 3}),
        # As an operator a trailing OR is a syntax error.
        ("telnet OR", {2, 3}),
        # As an operator NEAR( is a syntax error; as a word it finds chunk 5.
        ("NEAR(ssh telnet", {2, 3, 5}),
    ],
)
def test_build_match_query_reads_fts5_keywords_as_words(
    index: sqlite3.Connection, text: str, expected: set[int]
):
  expression = keyword.build_match_query(text)

  assert expression is not None
  assert _search(index, expression) == expected


def test_build_match_query_ignores_a_column_filter(index: sqlite3.Connection):
  # As typed, "title:foo" searches titles only and finds nothing: "foo" is
  # in the body of chunk 7.
  expression = keyword.build_match_query("title:foo")

  assert expression is not None
  assert _search(index, expression) == {7}


def test_build_match_query_ignores_a_prefix_star(index: sqlite3.Connection):
  # As typed, "open*" is a prefix query and finds "OpenSSH" in chunk 1.
  expression = keyword.build_match_query("open*")

  assert expression is not None
  assert not _search(index, expression)


def test_version_term_matches_only_where_the_version_is_stated(
    index: sqlite3.Connection,
):
  # Chunk 6 holds a 9 and a 6 apart, and chunk 7 holds "6.9".
  expression = keyword.build_match_query("9.6")

  assert expression is not None
  assert _search(index, expression) == {1}


def test_underscored_term_matches_only_the_whole_identifier(
    index: sqlite3.Connection,
):
  # Chunk 5 holds "nfs", "net" and "init", scattered.
  expression = keyword.build_match_query("nfs_net_init")

  assert expression is not None
  assert _search(index, expression) == {4}
