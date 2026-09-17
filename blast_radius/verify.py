"""Step 7 of the pipeline: checks a brief against what the model was shown.

The language model writes prose. This module decides how much of that prose
can be trusted, by string matching alone: it is deterministic and calls no
model. Two questions are asked of every claim.

* Does each quote occur in the source it cites? A quote is looked for
  verbatim, then with whitespace normalised on both sides. That forgives
  line wrapping and nothing else: the match is case-sensitive, and a
  paraphrase fails.
* Does every identifier in the claim occur in what the model was shown? CVE
  ids, QIDs, IP addresses, EC2 instance ids and host names are the details
  an analyst acts on, and an invented one reads exactly like a real one.

A claim that fails is kept and flagged with the reasons, never removed. The
analyst should see what the model wrote and that it could not be backed up;
dropping the claim would hide that the model misbehaved. The summary and the
caveats carry no citations, so only their identifiers are checked.

What the check cannot see: an identifier that a poisoned source itself
contains counts as shown, and a host name that is in neither the inventory
nor the brief's sources has no pattern to be recognised by.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import re

from blast_radius import models
from blast_radius import textproc
from blast_radius.retrieval import identifiers

# Step 1 reads "QID 1, 2 and 3" as a list of three, which suits a query. In
# prose the number after the comma is more often a count ("QID 90001, 14 of
# them internet-facing"), and reading it as a QID would flag a sound claim.
# Only the number straight after the keyword counts here, and the write
# prompt asks for the keyword in front of every QID.
_QID = re.compile(r"(?<![A-Za-z0-9])QIDs?[\s:#-]*(\d+)", re.IGNORECASE)

# Four dotted groups that are not part of a longer dotted number. A version
# such as 1.2.3.4 matches too, and is held to the same standard: fine when
# the sources say it, flagged when they do not.
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)")

_INSTANCE_ID = re.compile(
    r"(?<![A-Za-z0-9])i-[0-9a-f]{8,17}(?![A-Za-z0-9])", re.IGNORECASE
)

# A host name is mentioned only as a whole: "web-1" is not mentioned by
# "web-10" or by "web-1.example.test". A full stop that ends the sentence is
# not part of the name.
_NAME_START = r"(?<![A-Za-z0-9_.-])"
_NAME_END = r"(?![A-Za-z0-9_-]|\.[A-Za-z0-9])"


@dataclasses.dataclass(frozen=True)
class _Shown:
  """What the model was shown, indexed for the identifier checks.

  Attributes:
    text: Everything the model was shown, lower-cased.
    identifiers: The CVE ids, QIDs, IP addresses and instance ids in it.
    host_names: The inventory's host names, keyed by their lower-case form,
      in alphabetical order so that reports come out the same every time.
  """

  text: str
  identifiers: frozenset[str]
  host_names: Mapping[str, str]


def _unique(items: Sequence[str]) -> list[str]:
  """Returns ``items`` without repeats, in order of first appearance."""
  return list(dict.fromkeys(items))


def _pattern_identifiers(text: str) -> list[str]:
  """Returns the identifiers in ``text`` that a pattern can recognise.

  Each comes back in one canonical spelling, so that ``cve-2024-0001`` in a
  claim equals ``CVE-2024-0001`` in a source.

  Args:
    text: Prose from the brief, or the text the model was shown.
  """
  return [
      *identifiers.extract(text).cve_ids,
      *(f"QID {number}" for number in _QID.findall(text)),
      *_IPV4.findall(text),
      *(found.lower() for found in _INSTANCE_ID.findall(text)),
  ]


def _mentions(text: str, name: str) -> bool:
  """Returns whether ``text`` mentions the host ``name`` as a whole name.

  Args:
    text: Lower-cased text.
    name: A lower-cased host name.
  """
  # The substring test settles nearly every name without a regex.
  if name not in text:
    return False
  return re.search(_NAME_START + re.escape(name) + _NAME_END, text) is not None


def _index_shown(
    context: Sequence[models.ContextItem],
    shown_text: str,
    inventory_names: Collection[str],
) -> _Shown:
  """Returns what the model was shown, ready for lookups.

  Args:
    context: The sources the brief could cite.
    shown_text: Whatever else the model was shown.
    inventory_names: Every host name in the inventory.
  """
  parts = [shown_text]
  for item in context:
    document = f"QID {item.doc_id}" if item.doc_type == "qid" else item.doc_id
    parts.extend([document, item.title, item.text])
  text = "\n".join(parts)

  host_names: dict[str, str] = {}
  for name in sorted(inventory_names):
    name = name.strip()
    # A name that is one plain word ("bastion") is left out: prose uses such
    # words, and the check could not tell the host from the noun.
    if name and not name.isalpha():
      host_names.setdefault(name.lower(), name)
  return _Shown(
      text=text.lower(),
      identifiers=frozenset(_pattern_identifiers(text)),
      host_names=host_names,
  )


def _unknown_identifiers(text: str, shown: _Shown) -> list[str]:
  """Returns the identifiers in ``text`` that the model was never shown.

  Args:
    text: Prose from the brief.
    shown: What the model was shown.
  """
  unknown = [
      identifier
      for identifier in _pattern_identifiers(text)
      if identifier not in shown.identifiers
  ]
  lowered = text.lower()
  for key, name in shown.host_names.items():
    if _mentions(lowered, key) and not _mentions(shown.text, key):
      unknown.append(name)
  return _unique(unknown)


def _quote_found(quote: str, source: str) -> bool:
  """Returns whether ``quote`` occurs in ``source``.

  A verbatim match is tried first, then a match with whitespace normalised
  on both sides, which forgives line wrapping without accepting paraphrase.

  Args:
    quote: The text the model claims to have copied.
    source: The text of the source it cites.
  """
  if quote in source:
    return True
  normalised_quote = textproc.normalise_whitespace(quote)
  return normalised_quote in textproc.normalise_whitespace(source)


def _check_citation(
    citation: models.Citation, sources: Mapping[str, models.ContextItem]
) -> tuple[models.Citation, str | None]:
  """Returns ``citation`` with ``verified`` set, and why it failed if it did.

  Args:
    citation: A citation as the model gave it.
    sources: The citable sources by id.
  """
  source = sources.get(citation.source_id)
  problem = None
  if source is None:
    problem = f"cites {citation.source_id!r}, which is not one of the sources"
  elif not citation.quote.strip():
    problem = f"quotes nothing from {citation.source_id}"
  elif not _quote_found(citation.quote, source.text):
    problem = f"the quote attributed to {citation.source_id} is not in it"
  return citation.model_copy(update={"verified": problem is None}), problem


def _check_claim(
    claim: models.Claim,
    sources: Mapping[str, models.ContextItem],
    shown: _Shown,
) -> tuple[models.Claim, list[str]]:
  """Returns ``claim`` with its verdict, and its unknown identifiers.

  Args:
    claim: A claim as the model gave it.
    sources: The citable sources by id.
    shown: What the model was shown.
  """
  problems = [] if claim.citations else ["cites no source"]
  citations = []
  for citation in claim.citations:
    checked, problem = _check_citation(citation, sources)
    citations.append(checked)
    if problem is not None:
      problems.append(problem)
  unknown = _unknown_identifiers(claim.text, shown)
  problems.extend(
      f"mentions {identifier}, which the model was not shown"
      for identifier in unknown
  )
  checked_claim = claim.model_copy(
      update={
          "citations": citations,
          "verified": not problems,
          "problems": problems,
      }
  )
  return checked_claim, unknown


def verify_answer(
    answer: models.Answer,
    context: Sequence[models.ContextItem],
    *,
    shown_text: str = "",
    inventory_names: Collection[str] = (),
) -> tuple[models.Answer, models.Verification]:
  """Checks every quote and identifier in ``answer``.

  A claim is verified when it has at least one citation, every citation
  names a source in ``context`` and quotes text that occurs in it, and every
  identifier in the claim occurs in what the model was shown.

  Args:
    answer: The brief as the model wrote it.
    context: The sources the brief could cite.
    shown_text: Whatever the model was shown besides ``context``, normally
      the write prompt, which holds the query and the host summary.
    inventory_names: Every host name in the inventory. A name in the prose
      that the model was not shown is reported as an unknown identifier.

  Returns:
    A copy of ``answer`` with ``verified`` set on every citation and claim
    and ``problems`` filled in on the claims that failed, and the tally.
    Nothing is removed.
  """
  sources = {item.id: item for item in context}
  shown = _index_shown(context, shown_text, inventory_names)

  unknown = _unknown_identifiers(answer.summary, shown)
  claims = []
  for claim in answer.claims:
    checked, claim_unknown = _check_claim(claim, sources, shown)
    claims.append(checked)
    unknown.extend(claim_unknown)
  for caveat in answer.caveats:
    unknown.extend(_unknown_identifiers(caveat, shown))

  verification = models.Verification(
      total_claims=len(claims),
      verified_claims=sum(1 for claim in claims if claim.verified),
      unknown_identifiers=_unique(unknown),
  )
  return answer.model_copy(update={"claims": claims}), verification
