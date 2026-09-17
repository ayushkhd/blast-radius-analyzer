"""The two prompts, and the defensive reading of what the model returns.

The pipeline talks to the language model at most twice per query. Step 1
("parse") turns a pasted advisory into a product, a version and search
queries; step 6 ("write") turns the pipeline's results into a cited brief.
For each call this module builds the request and reads the reply back into
the shared models.

Instructions and data are kept apart:

* The instructions are the files ``blast_radius/prompts/parse.md`` and
  ``write.md``, sent unchanged as the system prompt. They are frozen, and
  their SHA-256 goes into every response. That is where reproducibility
  comes from, since the model takes no temperature.
* Everything else is data and makes up the user message. The analyst's
  pasted query and the NVD and scanner write-ups are third-party text that
  may contain instructions of its own, and so may the labels and host names
  taken from the scanner. Long text is fenced between ``⟦BEGIN DATA id⟧``
  and ``⟦END DATA id⟧`` markers, short text sits on one line, and in both
  every structural token is defused first, so content can neither close its
  block nor open a new one.

The write prompt carries a summary of the affected hosts and never the list:
counts, and per group a few example names. The prompt stays small, less of
the inventory leaves the machine, and the model has no list to get wrong.

Nothing the model returns is trusted. ``parsed_query_from`` and
``answer_from`` keep what has the right shape and drop the rest without
raising; ``blast_radius.verify`` then checks the brief against its sources.
"""

from collections.abc import Sequence
import dataclasses
import functools
import hashlib
import importlib.resources
import re
from typing import Any

from blast_radius import models
from blast_radius import textproc

PARSE = "parse"
WRITE = "write"

ZERO_WIDTH_JOINER = "\u200d"

# Mathematical white square brackets frame the data markers. They are
# vanishingly rare in advisories and code, and easy to spot in a log.
DATA_OPEN = "\u27e6"  # ⟦
DATA_CLOSE = "\u27e7"  # ⟧
DATA_BEGIN = f"{DATA_OPEN}BEGIN DATA"
DATA_END = f"{DATA_OPEN}END DATA"

# The sections of a user message. The names are long-winded on purpose: a
# write-up may well discuss a "<source>" element, and text that collides with
# a tag is altered before the model reads it.
_TAG_QUERY = "analyst_query"
_TAG_MATCHES = "matched_qids"
_TAG_HOSTS = "affected_hosts"
_TAG_SOURCES = "citable_sources"
_TAG_SOURCE = "citable_source"
_TAG_CAVEATS = "data_caveats"
_TAGS = (
    _TAG_QUERY,
    _TAG_MATCHES,
    _TAG_HOSTS,
    _TAG_SOURCES,
    _TAG_SOURCE,
    _TAG_CAVEATS,
)

# Everything that gives a user message its structure. The opening bracket
# stands for both markers, so that a marker forged with other spacing or
# capitals is defused as well.
_STRUCTURAL_TOKENS: tuple[str, ...] = (
    DATA_OPEN,
    *(f"<{tag}" for tag in _TAGS),
    *(f"</{tag}" for tag in _TAGS),
)
_STRUCTURAL = re.compile(
    "|".join(re.escape(token) for token in _STRUCTURAL_TOKENS), re.IGNORECASE
)

_QUERY_BLOCK_ID = "query"

# Caps on what the write prompt lists. The counts are always given in full;
# only the enumeration stops.
MAX_EXAMPLE_HOSTS = 4
_MAX_GROUPS = 12
_MAX_LISTED_IDS = 10

# Caps on what is taken from the parse reply.
MAX_SEARCH_QUERIES = 3
_MAX_QUERY_CHARS = 200
_MAX_FIELD_CHARS = 120

_NULLABLE_STRING: dict[str, Any] = {
    "anyOf": [{"type": "string"}, {"type": "null"}]
}

# Structured outputs accept a subset of JSON Schema: every object is closed
# and requires all of its properties, a nullable field is an ``anyOf`` with
# ``null``, and an array may set ``minItems`` to 0 or 1 but has no
# ``maxItems``. Upper limits are therefore stated in the descriptions and
# enforced when the reply is read.
PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "product": {
            **_NULLABLE_STRING,
            "description": "The affected product as the text names it.",
        },
        "version": {
            **_NULLABLE_STRING,
            "description": "The affected version or range, as written.",
        },
        "search_queries": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": "One to three search queries, most specific first.",
        },
    },
    "required": ["product", "version", "search_queries"],
    "additionalProperties": False,
}

_CITATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_id": {
            "type": "string",
            "description": "The id of one of the provided sources.",
        },
        "quote": {
            "type": "string",
            "description": (
                "One unbroken span copied exactly from that source's text,"
                " about 200 characters at most."
            ),
        },
    },
    "required": ["source_id", "quote"],
    "additionalProperties": False,
}

_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "One or two sentences that make a single point.",
        },
        "citations": {
            "type": "array",
            "minItems": 1,
            "items": _CITATION_SCHEMA,
        },
    },
    "required": ["text", "citations"],
    "additionalProperties": False,
}

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Two to four sentences. It carries no citations.",
        },
        "claims": {
            "type": "array",
            "items": _CLAIM_SCHEMA,
            "description": "The body of the brief: at most about eight claims.",
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the analyst should not assume. May be empty.",
        },
    },
    "required": ["summary", "claims", "caveats"],
    "additionalProperties": False,
}


@dataclasses.dataclass(frozen=True)
class PromptRequest:
  """One call to a provider, ready to send.

  Attributes:
    name: ``parse`` or ``write``, which is also the key of the prompt's hash.
    system: The instructions: a prompt file, unchanged.
    prompt: The user message: the task's data, fenced and defused. It is
      everything the model is shown besides the instructions, which makes
      it the right ``shown_text`` for ``verify.verify_answer``.
    schema: JSON Schema the reply must satisfy.
  """

  name: str
  system: str
  prompt: str
  schema: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _PromptFile:
  """A prompt file as shipped.

  Attributes:
    text: The file's content.
    sha256: Digest of the file's bytes, as ``shasum -a 256`` prints it.
  """

  text: str
  sha256: str


# Prompt files are package data that cannot change while the process runs,
# so each is read and hashed once.
@functools.cache
def _load(name: str) -> _PromptFile:
  """Returns the prompt file called ``name``."""
  root = importlib.resources.files("blast_radius")
  data = root.joinpath("prompts", f"{name}.md").read_bytes()
  return _PromptFile(
      text=data.decode("utf-8"), sha256=hashlib.sha256(data).hexdigest()
  )


def prompt_hashes() -> dict[str, str]:
  """Returns the SHA-256 of each prompt file, keyed by prompt name."""
  return {name: _load(name).sha256 for name in (PARSE, WRITE)}


# ---------------------------------------------------------------------------
# Rendering untrusted text
# ---------------------------------------------------------------------------


def _neutralise(text: str) -> str:
  """Returns ``text`` with every structural token in it defused.

  A zero-width joiner goes in after the first character of each token. The
  text reads the same, to a person and to the model, but it no longer holds
  a marker or a tag, so it cannot close the block it sits in or open another.

  Args:
    text: Untrusted content.
  """
  return _STRUCTURAL.sub(
      lambda found: found.group()[0] + ZERO_WIDTH_JOINER + found.group()[1:],
      text,
  )


def _inline(text: str) -> str:
  """Returns untrusted ``text`` defused and on one line.

  Collapsing the whitespace keeps a label or a host name from starting a
  line of its own, where it could pass for part of the prompt's structure.

  Args:
    text: A label, a name or an identifier.
  """
  return _neutralise(textproc.normalise_whitespace(text))


def _attribute(text: str) -> str:
  """Returns untrusted ``text`` as a double-quoted attribute value."""
  escaped = _inline(text).replace("&", "&amp;").replace('"', "&quot;")
  escaped = escaped.replace("<", "&lt;").replace(">", "&gt;")
  return f'"{escaped}"'


def _fence(block_id: str, text: str) -> str:
  """Returns untrusted ``text`` between the data markers for ``block_id``."""
  block_id = _inline(block_id)
  return (
      f"{DATA_BEGIN} {block_id}{DATA_CLOSE}\n"
      f"{_neutralise(text)}\n"
      f"{DATA_END} {block_id}{DATA_CLOSE}"
  )


def _section(tag: str, body: str) -> str:
  """Returns ``body``, which is already safe, as the section ``tag``."""
  return f"<{tag}>\n{body}\n</{tag}>"


def _count(number: int, noun: str) -> str:
  """Returns e.g. ``"1 host"`` or ``"3 hosts"``."""
  return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _listing(items: Sequence[str], limit: int) -> str:
  """Returns the first ``limit`` items, and how many were left out."""
  listed = ", ".join(_inline(item) for item in items[:limit])
  hidden = len(items) - limit
  return f"{listed} and {hidden} more" if hidden > 0 else listed


# ---------------------------------------------------------------------------
# Step 1: parse
# ---------------------------------------------------------------------------


def build_parse_request(query: str) -> PromptRequest:
  """Returns the request that reads a free-text query.

  Args:
    query: The analyst's text, with identifiers already taken out.
  """
  return PromptRequest(
      name=PARSE,
      system=_load(PARSE).text,
      prompt=_section(_TAG_QUERY, _fence(_QUERY_BLOCK_ID, query)) + "\n",
      schema=PARSE_SCHEMA,
  )


def _short_text(value: Any, max_chars: int) -> str | None:
  """Returns ``value`` as one tidy line of at most ``max_chars``, or None.

  Args:
    value: A value from the model's JSON, of any type.
    max_chars: Where to cut text that is too long.
  """
  if not isinstance(value, str):
    return None
  text = textproc.normalise_whitespace(value)
  if len(text) > max_chars:
    # Cut after the last word that fits, so that search is not given half a
    # word. Text with no space in it is cut where the limit falls.
    text = text[: max_chars + 1].rsplit(" ", 1)[0][:max_chars]
  return text or None


def _search_queries(value: Any) -> list[str]:
  """Returns the usable queries in ``value``, most specific first.

  Args:
    value: The model's ``search_queries``, of any type.
  """
  if not isinstance(value, list):
    return []
  queries: dict[str, str] = {}
  for item in value:
    query = _short_text(item, _MAX_QUERY_CHARS)
    if query is not None:
      # Keyed by the lower-cased text, so that a query repeated in other
      # capitals does not take one of the three places.
      queries.setdefault(query.lower(), query)
  return list(queries.values())[:MAX_SEARCH_QUERIES]


def parsed_query_from(
    raw: dict[str, Any], base: models.ParsedQuery
) -> models.ParsedQuery:
  """Returns ``base`` with what the model's reply adds to it.

  The reply is read defensively. A field of the wrong type or an empty
  string is ignored, long text is cut, and queries beyond the third are
  dropped. Identifiers are never taken from the model: ``base`` has them
  from a regular expression, which does not make things up.

  Args:
    raw: The model's JSON, expected to follow ``PARSE_SCHEMA``.
    base: The parse so far.

  Returns:
    A copy of ``base`` with ``product``, ``version`` and ``search_queries``
    filled in where the reply had something usable, and ``used_llm`` set.
    ``base`` itself when the reply had nothing usable at all.
  """
  product = _short_text(raw.get("product"), _MAX_FIELD_CHARS)
  version = _short_text(raw.get("version"), _MAX_FIELD_CHARS)
  queries = _search_queries(raw.get("search_queries"))
  if product is None and version is None and not queries:
    return base
  return base.model_copy(
      update={
          "product": product or base.product,
          "version": version or base.version,
          "search_queries": queries or base.search_queries,
          "used_llm": True,
      }
  )


# ---------------------------------------------------------------------------
# Step 6: write
# ---------------------------------------------------------------------------


def _query_section(query: str, parsed: models.ParsedQuery) -> str:
  """Returns the analyst's query and what step 1 read out of it."""
  lines = [_fence(_QUERY_BLOCK_ID, query)]
  named = [*parsed.cve_ids, *(f"QID {qid}" for qid in parsed.qids)]
  if named:
    listed = _listing(named, _MAX_LISTED_IDS)
    lines.append(f"Identifiers read from it: {listed}")
  if parsed.product:
    lines.append(f"Product read from it: {_inline(parsed.product)}")
  if parsed.version:
    lines.append(f"Version read from it: {_inline(parsed.version)}")
  return _section(_TAG_QUERY, "\n".join(lines))


def _cve_note(match: models.QidMatch) -> str:
  """Returns how many CVEs ``match`` covers, and which ones matter here."""
  if not match.cves:
    return "no CVE attached"
  note = _count(len(match.cves), "CVE")
  if match.matched_cve_ids:
    matched = _listing(match.matched_cve_ids, _MAX_LISTED_IDS)
    return f"{note}, of which the query matched {matched}"
  if len(match.cves) <= _MAX_LISTED_IDS:
    every = _listing([cve.cve_id for cve in match.cves], _MAX_LISTED_IDS)
    return f"{note}: {every}"
  return note


def _match_line(match: models.QidMatch) -> str:
  """Returns one matched QID as a line of the prompt.

  Args:
    match: A QID that step 2 matched.
  """
  facts = []
  if match.category:
    facts.append(f"category {_inline(match.category)}")
  if match.severity is not None:
    facts.append(f"severity {match.severity} of 5")
  if match.matched_by == "identifier":
    facts.append("named in the query")
  else:
    facts.append("found by search")
  facts.append(_cve_note(match))
  label = _inline(match.label) or "no label"
  details = "; ".join(facts)
  return f"- QID {_inline(match.qid)}: {label} ({details})"


def _matches_section(matches: Sequence[models.QidMatch]) -> str:
  """Returns the matched QIDs, best match first."""
  lines = [_match_line(match) for match in matches]
  return _section(_TAG_MATCHES, "\n".join(lines) or "No QID matched.")


def _group_line(position: int, group: models.HostGroup) -> str:
  """Returns one host group as a line of the prompt.

  Args:
    position: The group's rank by priority, from 1.
    group: The group.
  """
  size = _count(group.count, "host")
  line = (
      f"{position}. {_inline(group.label)}: {size},"
      f" {group.internet_facing_count} internet-facing"
  )
  # The ranking step already keeps only a few examples. The cap is repeated
  # here because this is the line where host names leave the machine.
  examples = [_inline(name) for name in group.example_hosts[:MAX_EXAMPLE_HOSTS]]
  if examples:
    line += "; for example " + ", ".join(examples)
  return line


def _hosts_section(
    groups: Sequence[models.HostGroup], host_count: int, inactive_count: int
) -> str:
  """Returns the summary of the affected hosts: counts, never the list.

  Args:
    groups: The running affected hosts folded into groups, by priority.
    host_count: How many affected hosts are running.
    inactive_count: How many affected hosts are not running.
  """
  lines = [
      f"Affected hosts that are running: {host_count}",
      "Affected hosts that are not running (listed for the analyst, not"
      f" ranked): {inactive_count}",
  ]
  if groups:
    lines.append("Groups of running hosts, highest priority first:")
  lines.extend(
      _group_line(position, group)
      for position, group in enumerate(groups[:_MAX_GROUPS], start=1)
  )
  unlisted = groups[_MAX_GROUPS:]
  if unlisted:
    hidden_groups = _count(len(unlisted), "lower-priority group")
    hidden_hosts = _count(sum(group.count for group in unlisted), "host")
    lines.append(
        f"{hidden_groups} with {hidden_hosts} in total are not listed here."
    )
  return _section(_TAG_HOSTS, "\n".join(lines))


def _source_block(item: models.ContextItem) -> str:
  """Returns one citable source: its provenance, then its fenced text."""
  document = f"QID {item.doc_id}" if item.doc_type == "qid" else item.doc_id
  return (
      f"<{_TAG_SOURCE} id={_attribute(item.id)}"
      f" document={_attribute(document)} title={_attribute(item.title)}>\n"
      f"{_fence(item.id, item.text)}\n"
      f"</{_TAG_SOURCE}>"
  )


def _sources_section(context: Sequence[models.ContextItem]) -> str:
  """Returns every source the brief may cite."""
  blocks = [_source_block(item) for item in context]
  return _section(_TAG_SOURCES, "\n".join(blocks) or "No source was found.")


def _caveats_section(caveats: Sequence[str]) -> str:
  """Returns the data caveats, one per line."""
  lines = [f"- {_inline(caveat)}" for caveat in caveats]
  return _section(_TAG_CAVEATS, "\n".join(lines) or "None.")


def build_write_request(
    query: str,
    parsed: models.ParsedQuery,
    matches: Sequence[models.QidMatch],
    groups: Sequence[models.HostGroup],
    host_count: int,
    inactive_count: int,
    context: Sequence[models.ContextItem],
    caveats: Sequence[str],
) -> PromptRequest:
  """Returns the request that writes the brief.

  Args:
    query: The analyst's input as typed.
    parsed: Step 1's reading of it.
    matches: The matched QIDs, best first.
    groups: The running affected hosts folded into groups, by priority.
    host_count: How many affected hosts are running.
    inactive_count: How many affected hosts are not running.
    context: The sources the brief may cite.
    caveats: Limits of the data that bear on this answer.
  """
  sections = [
      _query_section(query, parsed),
      _matches_section(matches),
      _hosts_section(groups, host_count, inactive_count),
      _sources_section(context),
      _caveats_section(caveats),
  ]
  return PromptRequest(
      name=WRITE,
      system=_load(WRITE).text,
      prompt="\n\n".join(sections) + "\n",
      schema=WRITE_SCHEMA,
  )


def _citation_from(value: Any) -> models.Citation | None:
  """Returns ``value`` as a citation, or None if it is not shaped like one."""
  if not isinstance(value, dict):
    return None
  source_id = value.get("source_id")
  quote = value.get("quote")
  if not isinstance(source_id, str) or not isinstance(quote, str):
    return None
  return models.Citation(source_id=source_id.strip(), quote=quote)


def _claim_from(value: Any) -> models.Claim | None:
  """Returns ``value`` as a claim, or None if it has no text.

  A claim whose citations are all unusable is kept without them. Whether an
  uncited claim may stand is the verifier's decision, and it says no.

  Args:
    value: One entry of the model's ``claims``, of any type.
  """
  if not isinstance(value, dict):
    return None
  text = value.get("text")
  if not isinstance(text, str) or not text.strip():
    return None
  raw_citations = value.get("citations")
  if not isinstance(raw_citations, list):
    raw_citations = []
  citations = [_citation_from(item) for item in raw_citations]
  return models.Claim(
      text=text.strip(), citations=[c for c in citations if c is not None]
  )


def answer_from(raw: dict[str, Any]) -> models.Answer | None:
  """Returns the model's brief, or None if the reply is unusable.

  The reply is read defensively: entries of the wrong shape are dropped and
  nothing raises. A brief needs a summary and at least one claim; without
  either there is nothing worth verifying, and the caller falls back to the
  response it builds from the data alone.

  Args:
    raw: The model's JSON, expected to follow ``WRITE_SCHEMA``.
  """
  summary = raw.get("summary")
  if not isinstance(summary, str) or not summary.strip():
    return None
  raw_claims = raw.get("claims")
  if not isinstance(raw_claims, list):
    return None
  claims = [_claim_from(item) for item in raw_claims]
  kept = [claim for claim in claims if claim is not None]
  if not kept:
    return None
  raw_caveats = raw.get("caveats")
  if not isinstance(raw_caveats, list):
    raw_caveats = []
  caveats = [
      item.strip()
      for item in raw_caveats
      if isinstance(item, str) and item.strip()
  ]
  return models.Answer(summary=summary.strip(), claims=kept, caveats=caveats)
