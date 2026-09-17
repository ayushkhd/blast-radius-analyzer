"""Finds CVE ids and QIDs in what an analyst typed.

Identifiers are looked up by primary key, not searched for, so they are
taken out of the query before free-text search sees it. What is left, the
``remainder``, is the text that search runs on.

The patterns are forgiving about how an identifier was pasted
(``cve_2024_36971``, an en dash from a PDF, a line break inside the id) and
strict about what counts as one. A CVE id needs its year and a sequence
number of four to seven digits. A QID needs the word ``QID`` in front of it:
a bare number may be a port, a count or a version, and is left alone.
"""

import dataclasses
import re

from blast_radius import textproc

# The hyphen as typed, and the look-alikes that word processors, PDF viewers
# and chat clients put in its place: hyphen, non-breaking hyphen, figure
# dash, en dash, em dash and minus sign.
_DASHES = r"\-\u2010\u2011\u2012\u2013\u2014\u2212"

# What may stand between the parts of a CVE id: a dash or an underscore, with
# a line break on either side where the id was wrapped, or spaces alone. The
# two cases are separate alternatives, not optional spaces around an optional
# dash, so that a run of spaces can be matched in one way only. A pattern
# that can split such a run in several ways tries them all before giving up.
_SEPARATOR = rf"(?:\s*[{_DASHES}_]\s*|\s+)"

# The boundaries keep "XCVE-2024-1234" and "CVE-2024-36971abc" out, and stop
# a sequence number of eight digits from being read as its first seven.
_CVE = re.compile(
    rf"(?<![A-Za-z0-9])CVE{_SEPARATOR}(\d{{4}}){_SEPARATOR}(\d{{4,7}})"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# "QID" or "QIDs", optional punctuation, then one number or a list of them:
# "38919", "38919, 38913", "38919, 38913 and 105936". A number after "and"
# is taken for the next QID, since that is what a list looks like. The price
# is that "QID 38919 and 22 hosts" yields 22 as well.
_QID_LIST = r"\d+(?:\s*(?:,(?:\s*(?:and|or))?|and|or|&)\s*\d+)*"
_QID = re.compile(
    rf"(?<![A-Za-z0-9])QIDs?[\s:#{_DASHES}]*({_QID_LIST})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[A-Za-z0-9]+")

# Words that join identifiers and say nothing themselves: what remains of
# "CVE-2023-48795 and QID 38913" is not a search query.
_CONNECTIVES = frozenset({"and", "or"})


@dataclasses.dataclass(frozen=True)
class Identifiers:
  """The identifiers found in a query, and the text around them.

  Attributes:
    cve_ids: CVE ids in order of first appearance, as ``CVE-2024-36971``.
    qids: QIDs in order of first appearance, digits only.
    remainder: The query without its identifiers, whitespace tidied. Empty
      when the query was nothing but identifiers, connectives and
      punctuation.
  """

  cve_ids: list[str]
  qids: list[str]
  remainder: str


def _unique(items: list[str]) -> list[str]:
  """Returns ``items`` without repeats, in order of first appearance."""
  return list(dict.fromkeys(items))


def _tidy(text: str) -> str:
  """Returns ``text`` ready for search, or "" if it has nothing to search."""
  words = _WORD.findall(text.lower())
  if all(word in _CONNECTIVES for word in words):
    return ""
  return textproc.normalise_whitespace(text)


def extract(text: str) -> Identifiers:
  """Returns the identifiers in ``text`` and the text that is left.

  Args:
    text: The query as typed.
  """
  cve_ids = [f"CVE-{year}-{number}" for year, number in _CVE.findall(text)]
  text = _CVE.sub(" ", text)

  qids = [
      number
      for numbers in _QID.findall(text)
      for number in _NUMBER.findall(numbers)
  ]
  text = _QID.sub(" ", text)

  return Identifiers(
      cve_ids=_unique(cve_ids), qids=_unique(qids), remainder=_tidy(text)
  )
