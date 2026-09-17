"""Step 5 of the pipeline: what the data says about fixing the matched items.

Scanner exports are thin on remediation. In the reference dataset the field
meant for it, ``how_to_fix``, is "N/A" on all but a handful of rows. What
guidance exists is scattered:

* a diagnosis often has an "Affected Versions" section;
* an Ubuntu update QID names the package whose security update fixes it;
* NVD references are tagged "Patch" or as an advisory;
* a known-exploited CVE carries CISA's required action.

This module gathers those into ``models.FixEvidence`` items, each a short,
quotable sentence with a citable id, and runs the pipeline's second guided
retrieval: a search for fix-related text restricted to the documents step 2
matched. When nothing is found the result is simply empty, and the brief
says that the data holds no fix guidance instead of inventing some.
"""

from collections.abc import Sequence
import dataclasses
import re

from blast_radius import config
from blast_radius import models
from blast_radius import store
from blast_radius.retrieval import retriever as retriever_lib

# The words a fix is described with, searched for inside the matched
# write-ups only. The analyst's own query is searched alongside it, so the
# fused result favours fix text that is about their product.
_FIX_QUERY = (
    "fix patch upgrade update mitigation workaround remediation resolved"
    " affected versions"
)

_SECTION_HEADING = re.compile(
    r"^(?:affected versions?|note: .*patched at (?:the )?following"
    r" versions?)\s*:?$",
    re.I,
)
# What ends the section. Most headings end in a colon, but the one that
# usually follows is also written "QID Detection Logic:(Unauthenticated)" and
# "QID Detection Logic (Unauthenticated)", so it is known by its words too.
_NEXT_HEADING = re.compile(r"^qid detection logic\b|:$", re.I)
_PACKAGE_UPDATE = re.compile(
    r"Ubuntu has released a security update for \S+ to fix the"
    r" vulnerabilit(?:y|ies)\.?"
)
_MAX_SECTION_LINES = 8


@dataclasses.dataclass(frozen=True)
class FixFindings:
  """The outcome of step 5.

  Attributes:
    evidence: Structured fix evidence, most specific first.
    chunks: The best fix-related chunks of the matched write-ups.
  """

  evidence: list[models.FixEvidence]
  chunks: list[models.ScoredChunk]


def affected_versions(diagnosis: str) -> str | None:
  """Returns the "Affected Versions" section of a diagnosis as one sentence.

  Qualys diagnoses put the heading and each version in paragraphs of their
  own, ending at the next heading ("QID Detection Logic:"):

      Affected Versions:

      OpenSSH versions prior to 8.3

  becomes ``"Affected Versions: OpenSSH versions prior to 8.3"``.

  Args:
    diagnosis: A cleaned diagnosis, with paragraphs separated by blank lines.
  """
  paragraphs = [p.strip() for p in diagnosis.split("\n\n") if p.strip()]
  for i, paragraph in enumerate(paragraphs):
    if not _SECTION_HEADING.match(paragraph):
      continue
    lines = []
    for following in paragraphs[i + 1 : i + 1 + _MAX_SECTION_LINES]:
      if _NEXT_HEADING.search(following):
        break
      lines.append(following)
    if lines:
      heading = paragraph.rstrip(":")
      return f"{heading}: " + "; ".join(lines)
  return None


def package_update(diagnosis: str) -> str | None:
  """Returns the sentence naming the Ubuntu package update, if present."""
  found = _PACKAGE_UPDATE.search(diagnosis)
  return found.group(0) if found else None


def _cve_evidence(
    cve: models.Cve, max_refs: int
) -> list[tuple[models.EvidenceKind, str, str | None]]:
  """Returns ``(kind, text, url)`` for everything ``cve`` says about a fix.

  Args:
    cve: A CVE of a matched QID.
    max_refs: How many patch references and how many advisories to keep.
  """
  items: list[tuple[models.EvidenceKind, str, str | None]] = []
  if cve.kev_required_action:
    items.append(
        (
            "required_action",
            f"CISA KEV required action for {cve.cve_id}:"
            f" {cve.kev_required_action}",
            None,
        )
    )
  if cve.vendor_fix:
    items.append(
        ("vendor_fix", f"Fix for {cve.cve_id}: {cve.vendor_fix}", None)
    )
  for url in cve.patch_refs[:max_refs]:
    items.append(
        ("patch_reference", f"Patch reference for {cve.cve_id}: {url}", url)
    )
  for url in cve.advisory_refs[:max_refs]:
    items.append(
        ("advisory_reference", f"Advisory for {cve.cve_id}: {url}", url)
    )
  return items


def collect(
    db: store.Store,
    retriever: retriever_lib.Retriever,
    matches: Sequence[models.QidMatch],
    query: str,
    settings: config.Settings,
) -> FixFindings:
  """Gathers fix evidence for ``matches``.

  Only the CVEs the query actually hit are consulted. A QID matched as a
  whole has no such subset, so all of its CVEs are, unless it bundles more
  than ``settings.max_evidence_cves`` of them: for an Ubuntu kernel update
  the fix is the package update the diagnosis names, not hundreds of links
  to upstream commits.

  Args:
    db: The artifact.
    retriever: Runs the search restricted to the matched documents.
    matches: Step 2's matches.
    query: The analyst's free text, searched alongside the fix vocabulary.
      May be empty.
    settings: Caps on references per CVE and on chunks returned.
  """
  drafts: list[tuple[models.EvidenceKind, models.DocType, str, str, str | None]]
  drafts = []
  docs: list[tuple[models.DocType, str]] = []
  for match in matches:
    qid = db.get_qid(match.qid)
    if qid is not None and qid.diagnosis:
      docs.append(("qid", match.qid))
      versions = affected_versions(qid.diagnosis)
      if versions:
        drafts.append(("affected_versions", "qid", match.qid, versions, None))
      update = package_update(qid.diagnosis)
      if update:
        drafts.append(("package_update", "qid", match.qid, update, None))
    cve_ids = match.matched_cve_ids
    if not cve_ids and len(match.cves) <= settings.max_evidence_cves:
      cve_ids = [cve.cve_id for cve in match.cves]
    for cve_id in cve_ids:
      cve = db.get_cve(cve_id)
      # A CVE that sits under two matched QIDs is reported once.
      if cve is None or ("cve", cve_id) in docs:
        continue
      docs.append(("cve", cve_id))
      for kind, text, url in _cve_evidence(cve, settings.max_refs_per_cve):
        drafts.append((kind, "cve", cve_id, text, url))

  evidence = [
      models.FixEvidence(
          id=f"e{number}",
          kind=kind,
          doc_type=doc_type,
          doc_id=doc_id,
          text=text,
          url=url,
      )
      for number, (kind, doc_type, doc_id, text, url) in enumerate(
          drafts, start=1
      )
  ]

  chunks: list[models.ScoredChunk] = []
  if docs and settings.fix_chunk_count:
    found = retriever.search(
        [_FIX_QUERY, query], docs=docs, limit=settings.fix_chunk_count
    )
    chunks = [item for item in found if item.chunk.kind == "text"]
  return FixFindings(evidence=evidence, chunks=chunks)
