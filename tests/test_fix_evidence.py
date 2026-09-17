"""Tests for blast_radius.fix_evidence."""

import os
from typing import Any

import pytest

from blast_radius import config
from blast_radius import embeddings
from blast_radius import fix_evidence
from blast_radius import models
from blast_radius import store as store_lib
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_PROXY = "710003"
_QID_BROKER = "710005"
_QID_UNEXPLAINED = "710901"
_CVE_FXNET = "CVE-2099-1001"
_CVE_BROKER = "CVE-2099-4001"

_DETECTION_LOGIC = "This check reads the version from the service banner."


@pytest.fixture(autouse=True)
def _no_blast_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps ``BLAST_*`` variables set on this machine out of the settings."""
  for name in list(os.environ):
    if name.upper().startswith("BLAST_"):
      monkeypatch.delenv(name)


def _diagnosis(*paragraphs: str) -> str:
  """Returns a diagnosis as ingest stores it: paragraphs, blank lines."""
  return "\n\n".join(paragraphs)


def _settings(**overrides: Any) -> config.Settings:
  """Returns the default settings with ``overrides``, reading no ``.env``."""
  return config.Settings(_env_file=None, **overrides)


def _retriever(
    db: store_lib.Store, **overrides: Any
) -> retriever_lib.Retriever:
  """Returns a retriever on ``db`` that uses the model-free scorers."""
  return retriever_lib.Retriever(
      db,
      embeddings.HashingEmbedder(),
      rerank.LexicalReranker(),
      _settings(**overrides),
  )


def _collect(
    db: store_lib.Store,
    *,
    cve_ids: tuple[str, ...] = (),
    qids: tuple[str, ...] = (),
    **overrides: Any,
) -> fix_evidence.FixFindings:
  """Returns the findings for the matches of an identifier query."""
  retriever = _retriever(db)
  parsed = models.ParsedQuery(raw="", cve_ids=list(cve_ids), qids=list(qids))
  matches = retriever.retrieve(parsed).matches
  return fix_evidence.collect(
      db, retriever, matches, "", _settings(**overrides)
  )


def _kinds(findings: fix_evidence.FixFindings) -> list[str]:
  """Returns the kind of each piece of evidence, in order."""
  return [item.kind for item in findings.evidence]


# ---------------------------------------------------------------------------
# affected_versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heading", ["Affected Versions:", "Affected Versions"])
def test_affected_versions_joins_the_heading_and_its_value(heading: str):
  diagnosis = _diagnosis(
      "OpenSSH is a suite of tools for remote login.",
      heading,
      "OpenSSH up to version 9.6",
      "QID Detection Logic:",
      _DETECTION_LOGIC,
  )

  found = fix_evidence.affected_versions(diagnosis)

  assert found == "Affected Versions: OpenSSH up to version 9.6"


def test_affected_versions_joins_several_value_lines():
  diagnosis = _diagnosis(
      "Affected Versions:",
      "Trellis proxy before 2.8.4",
      "Trellis proxy 3.0.0 before 3.1.2",
      "QID Detection Logic (Unauthenticated):",
      _DETECTION_LOGIC,
  )

  found = fix_evidence.affected_versions(diagnosis)

  assert found == (
      "Affected Versions: Trellis proxy before 2.8.4; Trellis proxy 3.0.0"
      " before 3.1.2"
  )


@pytest.mark.parametrize(
    "next_heading",
    [
        "Workaround:",
        "QID Detection Logic:(Unauthenticated)",
        "QID Detection Logic (Unauthenticated)",
        "QID Detection Logic ( Un-Authenticated)",
    ],
)
def test_affected_versions_section_ends_at_the_next_heading(next_heading: str):
  diagnosis = _diagnosis(
      "Affected Versions:",
      "Marlin HTTP Server versions prior to 3.2.9",
      next_heading,
      _DETECTION_LOGIC,
  )

  found = fix_evidence.affected_versions(diagnosis)

  assert (
      found == "Affected Versions: Marlin HTTP Server versions prior to 3.2.9"
  )


def test_affected_versions_section_without_a_closing_heading_is_capped():
  versions = [f"Pylon agent 1.{minor}" for minor in range(12)]

  found = fix_evidence.affected_versions(
      _diagnosis("Affected Versions:", *versions)
  )

  assert found == "Affected Versions: " + "; ".join(versions[:8])


def test_affected_versions_reads_the_patched_versions_note():
  diagnosis = _diagnosis(
      "Ferrous broker deserialises join requests before authentication.",
      "Note: This CVE is patched at following versions",
      "4.1.2",
      "4.2.0",
      "QID Detection Logic:",
      _DETECTION_LOGIC,
  )

  found = fix_evidence.affected_versions(diagnosis)

  assert (
      found == "Note: This CVE is patched at following versions: 4.1.2; 4.2.0"
  )


@pytest.mark.parametrize(
    "diagnosis",
    [
        "",
        _diagnosis("Pylon agent writes a state file.", _DETECTION_LOGIC),
        # A sentence that mentions the words is not the heading.
        _diagnosis("Affected versions are listed in the vendor advisory."),
        # A heading with nothing under it.
        _diagnosis("Affected Versions:", "QID Detection Logic:"),
    ],
    ids=["empty", "no-section", "prose", "empty-section"],
)
def test_affected_versions_without_a_section_is_none(diagnosis: str):
  assert fix_evidence.affected_versions(diagnosis) is None


# ---------------------------------------------------------------------------
# package_update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Ubuntu has released a security update for linux to fix the"
        " vulnerabilities.",
        "Ubuntu has released a security update for openssl to fix the"
        " vulnerability.",
    ],
)
def test_package_update_returns_the_sentence_that_names_the_package(
    sentence: str,
):
  diagnosis = _diagnosis(sentence, "QID Detection Logic (Authenticated):")

  assert fix_evidence.package_update(diagnosis) == sentence


def test_package_update_of_another_kind_of_diagnosis_is_none():
  diagnosis = _diagnosis("OpenSSH is a suite of tools for remote login.")

  assert fix_evidence.package_update(diagnosis) is None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def test_collect_qid_without_cves_yields_its_affected_versions(
    store: store_lib.Store,
):
  findings = _collect(store, qids=(_QID_OPENSSH,))

  [item] = findings.evidence
  assert (item.kind, item.doc_type, item.doc_id) == (
      "affected_versions",
      "qid",
      _QID_OPENSSH,
  )
  assert item.text == "Affected Versions: OpenSSH up to version 9.6"
  assert item.url is None


def test_collect_affected_versions_stop_before_the_detection_logic(
    store: store_lib.Store,
):
  # This diagnosis writes both headings without a colon.
  findings = _collect(store, qids=(_QID_PROXY,), max_evidence_cves=0)

  [item] = findings.evidence
  assert item.text == (
      "Affected Versions: Trellis proxy before 2.8.4, and 3.0.0 before 3.1.2"
  )


def test_collect_small_bundle_matched_as_a_whole_consults_every_cve(
    store: store_lib.Store,
):
  findings = _collect(store, qids=(_QID_KERNEL,), max_evidence_cves=4)

  # Three of the four kernel CVEs have a reference that is tagged "Patch".
  assert _kinds(findings) == ["package_update"] + ["patch_reference"] * 3


def test_collect_large_bundle_matched_as_a_whole_yields_the_update_alone(
    store: store_lib.Store,
):
  findings = _collect(store, qids=(_QID_KERNEL,), max_evidence_cves=3)

  [item] = findings.evidence
  assert item.kind == "package_update"
  assert item.text == (
      "Ubuntu has released a security update for linux to fix the"
      " vulnerabilities."
  )


def test_collect_matched_cve_of_a_large_bundle_is_still_consulted(
    store: store_lib.Store,
):
  findings = _collect(store, cve_ids=(_CVE_FXNET,), max_evidence_cves=0)

  assert _kinds(findings) == ["package_update", "patch_reference"]
  assert findings.evidence[1].doc_id == _CVE_FXNET
  assert findings.evidence[1].url == (
      "https://git.example.org/linux/c/1001a1b2c3d4e5f6"
  )


def test_collect_known_exploited_cve_yields_required_action_and_vendor_fix(
    store: store_lib.Store,
):
  findings = _collect(store, cve_ids=(_CVE_BROKER,))

  by_kind = {item.kind: item for item in findings.evidence}
  assert by_kind["required_action"].text == (
      f"CISA KEV required action for {_CVE_BROKER}: Apply the vendor's"
      " update, or stop exposing the management listener until the update"
      " is applied."
  )
  assert by_kind["vendor_fix"].text == (
      f"Fix for {_CVE_BROKER}: Upgrade Ferrous broker to version 4.1.2 or"
      " later."
  )


def test_collect_numbers_evidence_from_e1_most_specific_first(
    store: store_lib.Store,
):
  findings = _collect(store, cve_ids=(_CVE_BROKER,))

  assert [(item.id, item.kind) for item in findings.evidence] == [
      ("e1", "affected_versions"),
      ("e2", "required_action"),
      ("e3", "vendor_fix"),
      ("e4", "patch_reference"),
      ("e5", "advisory_reference"),
  ]


def test_collect_max_refs_per_cve_of_zero_leaves_the_references_out(
    store: store_lib.Store,
):
  findings = _collect(store, cve_ids=(_CVE_BROKER,), max_refs_per_cve=0)

  assert _kinds(findings) == [
      "affected_versions",
      "required_action",
      "vendor_fix",
  ]


def test_collect_cve_under_two_matched_qids_is_reported_once(
    store: store_lib.Store,
):
  matches = [
      models.QidMatch(
          qid=qid, matched_by="identifier", matched_cve_ids=[_CVE_BROKER]
      )
      for qid in (_QID_BROKER, _QID_UNEXPLAINED)
  ]

  findings = fix_evidence.collect(
      store, _retriever(store), matches, "", _settings()
  )

  assert _kinds(findings).count("required_action") == 1
  assert _kinds(findings).count("patch_reference") == 1


def test_collect_searches_only_text_chunks_of_the_matched_documents(
    store: store_lib.Store,
):
  # Keyword search alone ranks a kernel-trace chunk of the CVE first for
  # these words, so the result shows that trace chunks are left out.
  retriever = _retriever(store, enable_dense=False, enable_rerank=False)
  query = "kasan_save_stack fxnet_ring_alloc kfree"
  matches = retriever.retrieve(
      models.ParsedQuery(raw=_CVE_FXNET, cve_ids=[_CVE_FXNET])
  ).matches

  findings = fix_evidence.collect(store, retriever, matches, query, _settings())

  chunks = [item.chunk for item in findings.chunks]
  assert chunks
  assert all(chunk.kind == "text" for chunk in chunks)
  assert {(chunk.doc_type, chunk.doc_id) for chunk in chunks} <= {
      ("qid", _QID_KERNEL),
      ("cve", _CVE_FXNET),
  }


def test_collect_with_a_fix_chunk_count_of_zero_skips_the_search(
    store: store_lib.Store,
):
  findings = _collect(store, qids=(_QID_OPENSSH,), fix_chunk_count=0)

  assert findings.evidence
  assert not findings.chunks


def test_collect_unexplained_qid_yields_nothing(store: store_lib.Store):
  findings = _collect(store, qids=(_QID_UNEXPLAINED,))

  assert not findings.evidence
  assert not findings.chunks
