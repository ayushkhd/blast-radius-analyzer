"""Tests for blast_radius.ingest, and for the synthetic exports it reads."""

from collections.abc import Sequence
import copy
import dataclasses
import datetime
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
import pytest

from blast_radius import embeddings
from blast_radius import ingest
from blast_radius import models
from blast_radius import schema
from blast_radius import store as store_lib
from tests.fixtures import make_fixtures

_Records = list[dict[str, Any]]

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_BASTION = "900000001"
_GATEWAY = "900000002"
_WORKER_A = "900000003"
_WORKER_B1 = "900000005"
_WEB = "900000008"

_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_UNEXPLAINED = "710901"

_CVE_LONG_KERNEL = "CVE-2099-1001"
_CVE_UNSCORED_KERNEL = "CVE-2099-1002"
_CVE_PROXY = "CVE-2099-2001"
_CVE_SMUGGLING = "CVE-2099-3001"
_CVE_TEMPLATE = "CVE-2099-3002"
_CVE_KNOWN_EXPLOITED = "CVE-2099-4001"
_CVE_AGENT = "CVE-2099-5001"

# Row 7 of vulns.json pairs the kernel finding on the first worker with
# CVE-2099-1002. Row 2 is the first row of CVE-2099-2001, and rows 0, 3 and 9
# are the first three rows of the OpenSSH finding.
_ROW = 7
_ROW_HOST = _WORKER_A
_ROW_DETECTION = "51000000011"
_FIRST_PROXY_ROW = 2
_LATER_OPENSSH_ROWS = (3, 9)


@pytest.fixture(name="assets")
def fixture_assets(assets_path: pathlib.Path) -> _Records:
  """Returns the records of the asset export, for a test to edit."""
  return json.loads(assets_path.read_text(encoding="utf-8"))


@pytest.fixture(name="vulns")
def fixture_vulns(vulns_path: pathlib.Path) -> _Records:
  """Returns the rows of the vulns export, for a test to edit."""
  return json.loads(vulns_path.read_text(encoding="utf-8"))


def _build_from(
    tmp_path: pathlib.Path,
    assets: Any,
    vulns: Any,
    *,
    chunk_max_chars: int = 1200,
    chunk_overlap_sentences: int = 1,
) -> ingest.IngestReport:
  """Builds ``tmp_path/out/index.sqlite`` from in-memory exports.

  Args:
    tmp_path: The test's temporary directory.
    assets: What to write as the asset export; normally a list of records.
    vulns: What to write as the vulns export; normally a list of rows.
    chunk_max_chars: Passed to ``build_artifact``.
    chunk_overlap_sentences: Passed to ``build_artifact``.
  """
  exports = tmp_path / "exports"
  exports.mkdir()
  (exports / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
  (exports / "vulns.json").write_text(json.dumps(vulns), encoding="utf-8")
  return ingest.build_artifact(
      exports / "assets.json",
      exports / "vulns.json",
      tmp_path / "out" / "index.sqlite",
      embeddings.HashingEmbedder(),
      chunk_max_chars=chunk_max_chars,
      chunk_overlap_sentences=chunk_overlap_sentences,
  )


def _asset(assets: _Records, host_id: str) -> dict[str, Any]:
  """Returns the asset record of one host."""
  return next(record for record in assets if record["id"] == host_id)


def _cve_row(vulns: _Records, cve_id: str) -> dict[str, Any]:
  """Returns the first vulns row that carries the given CVE."""
  return next(row for row in vulns if row.get("cve_id") == cve_id)


# ---------------------------------------------------------------------------
# The synthetic exports
# ---------------------------------------------------------------------------


def test_committed_fixtures_are_what_their_generator_writes(
    tmp_path: pathlib.Path, assets_path: pathlib.Path, vulns_path: pathlib.Path
):
  make_fixtures.write_fixtures(tmp_path)

  assert (tmp_path / "assets.json").read_bytes() == assets_path.read_bytes()
  assert (tmp_path / "vulns.json").read_bytes() == vulns_path.read_bytes()


# ---------------------------------------------------------------------------
# What a build writes
# ---------------------------------------------------------------------------


def test_build_artifact_reports_what_it_wrote(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  report = _build_from(tmp_path, assets, vulns)

  # All 45 detections of the asset file, of which the vulns file explains 26.
  assert dataclasses.replace(report, duration_s=0.0) == ingest.IngestReport(
      hosts=11,
      open_ports=36,
      detections=45,
      qids=8,
      explained_qids=6,
      cves=9,
      qid_cves=9,
      chunks=18,
      text_chunks=16,
      trace_chunks=2,
      embeddings=16,
      warnings=[],
      duration_s=0.0,
  )
  assert report.duration_s > 0


def test_build_artifact_writes_the_counts_it_reports(store: store_lib.Store):
  stats = store.stats()

  assert (stats.hosts, stats.total_detections, stats.total_qids) == (11, 45, 8)
  assert (stats.explained_qids, stats.cves, stats.chunks) == (6, 9, 18)


def test_meta_records_the_inputs_the_embedder_and_the_settings(
    store: store_lib.Store, assets_path: pathlib.Path, vulns_path: pathlib.Path
):
  meta = store.meta()

  built_at = datetime.datetime.fromisoformat(meta.pop(schema.META_BUILT_AT))
  assert built_at.utcoffset() == datetime.timedelta(0)
  assert meta == {
      schema.META_SCHEMA_VERSION: str(schema.SCHEMA_VERSION),
      schema.META_ASSETS_SHA256: hashlib.sha256(
          assets_path.read_bytes()
      ).hexdigest(),
      schema.META_VULNS_SHA256: hashlib.sha256(
          vulns_path.read_bytes()
      ).hexdigest(),
      schema.META_EMBEDDING_MODEL: "hashing-256",
      schema.META_EMBEDDING_DIM: "256",
      schema.META_CHUNK_MAX_CHARS: "1200",
      "chunk_overlap_sentences": "1",
  }


def test_chunk_settings_reach_the_chunker_and_the_meta_table(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  report = _build_from(
      tmp_path, assets, vulns, chunk_max_chars=300, chunk_overlap_sentences=0
  )

  with store_lib.Store(tmp_path / "out" / "index.sqlite") as opened:
    meta = opened.meta()
  assert report.chunks > 18
  assert meta[schema.META_CHUNK_MAX_CHARS] == "300"
  assert meta["chunk_overlap_sentences"] == "0"


# ---------------------------------------------------------------------------
# Hosts
# ---------------------------------------------------------------------------


def test_normalise_host_maps_every_field(assets: _Records):
  host = ingest.normalise_host(_asset(assets, _BASTION))

  assert host == models.Host(
      id=_BASTION,
      name="fx-bastion",
      os="Ubuntu Linux 20.04.6",
      criticality=5,
      state="RUNNING",
      internet_facing=True,
      public_ip="203.0.113.10",
      private_ip="10.20.1.10",
      region="us-west-2",
      vpc_id="vpc-00000000000000001",
      security_group="fx-bastion-sg",
      cluster=None,
      role="bastion",
      is_docker_host=False,
      last_scan="2099-03-01T06:00:00Z",
  )


def test_normalise_host_with_the_scanner_tag_alone_is_internet_facing(
    assets: _Records,
):
  record = _asset(assets, _BASTION)
  record["sourceInfo"]["publicIpAddress"] = None

  host = ingest.normalise_host(record)

  assert host.internet_facing
  assert host.public_ip is None


def test_normalise_host_with_a_public_ip_alone_is_internet_facing(
    assets: _Records,
):
  record = _asset(assets, _GATEWAY)
  assert [tag["name"] for tag in record["tags"]] == ["Fixture Cloud Connector"]

  host = ingest.normalise_host(record)

  assert host.internet_facing
  assert host.public_ip == "203.0.113.20"


@pytest.mark.parametrize("public_ip", [None, ""])
def test_normalise_host_with_neither_tag_nor_public_ip_is_internal(
    assets: _Records, public_ip: str | None
):
  record = _asset(assets, _WEB)
  record["sourceInfo"]["publicIpAddress"] = public_ip

  host = ingest.normalise_host(record)

  assert not host.internet_facing
  assert host.public_ip is None


def test_normalise_host_takes_the_cluster_from_the_eks_tag(assets: _Records):
  record = _asset(assets, _WORKER_A)
  record["sourceInfo"]["ec2InstanceTags"] = [
      {"key": "kubernetes.io/cluster/from-the-key", "value": "owned"},
      {"key": "aws:eks:cluster-name", "value": "from-the-eks-tag"},
  ]

  assert ingest.normalise_host(record).cluster == "from-the-eks-tag"


def test_normalise_host_falls_back_to_the_cluster_named_in_a_tag_key(
    assets: _Records,
):
  record = _asset(assets, _WORKER_B1)
  keys = [tag["key"] for tag in record["sourceInfo"]["ec2InstanceTags"]]
  assert "aws:eks:cluster-name" not in keys

  assert ingest.normalise_host(record).cluster == "fx-build"


def test_normalise_host_without_cluster_or_role_tags_has_neither(
    assets: _Records,
):
  host = ingest.normalise_host(_asset(assets, _WEB))

  assert host.cluster is None
  assert host.role is None


def test_normalise_host_reads_the_role_tag(assets: _Records):
  assert ingest.normalise_host(_asset(assets, _GATEWAY)).role == "nat-gateway"


def test_normalise_host_tolerates_an_ec2_tag_without_a_value(assets: _Records):
  record = _asset(assets, _WEB)
  record["sourceInfo"]["ec2InstanceTags"].append({"key": "role", "value": None})

  assert ingest.normalise_host(record).role is None


def test_normalise_host_parses_the_exports_strings(assets: _Records):
  record = _asset(assets, _WORKER_A)
  assert record["criticalityScore"] == "3"
  assert record["isDockerHost"] == "true"

  host = ingest.normalise_host(record)

  assert host.criticality == 3
  assert host.is_docker_host is True


def test_open_ports_are_stored_as_integers(store: store_lib.Store):
  ports = store.host_ports(_BASTION)

  assert ports == [models.OpenPort(port=22, protocol="TCP", service="ssh")]


def test_host_without_detections_is_still_ingested(store: store_lib.Store):
  host = store.get_host("900000011")

  assert host is not None
  assert host.last_scan is None
  assert not store.host_detections(host.id)


# ---------------------------------------------------------------------------
# QIDs and their labels
# ---------------------------------------------------------------------------


def test_unexplained_qid_is_recorded_without_a_write_up(
    store: store_lib.Store,
):
  assert store.get_qid(_QID_UNEXPLAINED) == models.Qid(
      qid=_QID_UNEXPLAINED, explained=False
  )
  assert not store.chunks_for_doc("qid", _QID_UNEXPLAINED)


def test_explained_qid_takes_its_fields_from_the_vulns_rows(
    store: store_lib.Store,
):
  qid = store.get_qid(_QID_OPENSSH)

  assert qid is not None
  assert (qid.category, qid.severity) == ("General remote services", 4)
  assert qid.pci_flag is True
  assert qid.explained


def test_diagnosis_is_cleaned_of_html(store: store_lib.Store):
  openssh = store.get_qid(_QID_OPENSSH)
  kernel = store.get_qid(_QID_KERNEL)

  assert openssh is not None and kernel is not None
  assert openssh.diagnosis is not None and kernel.diagnosis is not None
  assert "<" not in openssh.diagnosis
  assert "Affected Versions:\n\nOpenSSH up to version 9.6" in openssh.diagnosis
  assert 'such as "dpkg",' in kernel.diagnosis


def test_build_artifact_labels_every_explained_qid(store: store_lib.Store):
  labels = {}
  for number in range(710001, 710007):
    qid = store.get_qid(str(number))
    assert qid is not None
    labels[qid.qid] = qid.label

  assert labels == {
      "710001": (
          "OpenSSH may allow an authentication bypass on hardware prone to"
          " memory bit flips,..."
      ),
      "710002": "Ubuntu security update: linux",
      "710003": "Trellis Proxy Forwarded Header Removal Vulnerability",
      "710004": (
          "A request whose chunked body is followed by extra data is"
          " forwarded by mod_relay..."
      ),
      "710005": "Ferrous Broker Management Listener Remote Code Execution",
      "710006": "Pylon Agent World-Readable State File",
  }


def test_qid_label_of_a_qid_with_one_cve_is_the_cves_title():
  label = ingest.qid_label(
      "Trellis proxy is a reverse proxy. It drops headers.",
      ["Trellis Proxy Header Removal"],
  )

  assert label == "Trellis Proxy Header Removal"


def test_qid_label_of_an_ubuntu_update_names_the_package():
  diagnosis = (
      "Ubuntu has released a security update for python-zipp to fix the"
      " vulnerabilities.\n\nQID Detection Logic (Authenticated):"
  )

  assert ingest.qid_label(diagnosis, []) == (
      "Ubuntu security update: python-zipp"
  )
  assert ingest.qid_label(diagnosis, ["First CVE", "Second CVE"]) == (
      "Ubuntu security update: python-zipp"
  )


def test_qid_label_prefers_the_title_of_a_single_cve_to_the_package():
  diagnosis = (
      "Ubuntu has released a security update for busybox to fix the"
      " vulnerabilities."
  )

  assert ingest.qid_label(diagnosis, ["Busybox Stack Overflow"]) == (
      "Busybox Stack Overflow"
  )


def test_qid_label_skips_the_product_introduction_and_headings():
  diagnosis = (
      "OpenSSH (OpenBSD Secure Shell) is a set of computer programs.\n\n"
      "OpenSSH contains the following vulnerabilities:\n\n"
      "CVE-2099-0001: The client leaks the host key algorithm.\n\n"
      "Affected Versions:\n\nOpenSSH 5.7 to 8.6"
  )

  label = ingest.qid_label(diagnosis, [])

  assert label == "The client leaks the host key algorithm."


def test_qid_label_drops_a_cve_id_that_opens_the_sentence():
  diagnosis = (
      "Marlin HTTP Server is an open-source web server.\n\n"
      "CVE-2099-0001 - A crafted request body crashes the worker.\n\n"
      "CVE-2099-0002 - The status page discloses file paths."
  )

  label = ingest.qid_label(diagnosis, ["First CVE", "Second CVE"])

  assert label == "A crafted request body crashes the worker."


def test_qid_label_falls_back_to_the_first_sentence():
  diagnosis = "Marlin HTTP Server is a web server.\n\nAffected Versions:"

  assert ingest.qid_label(diagnosis, []) == (
      "Marlin HTTP Server is a web server."
  )


def test_qid_label_is_cut_at_a_word_boundary_to_ninety_characters():
  sentence = "The daemon " + "mishandles oversized frames and " * 5 + "crashes."

  label = ingest.qid_label(sentence, [])

  assert len(label) <= 90
  assert label.endswith("...")
  assert sentence.startswith(label.removesuffix("..."))
  assert sentence[len(label) - 3] == " "


def test_qid_label_of_an_empty_diagnosis_is_empty():
  assert ingest.qid_label("", []) == ""


def test_qid_rows_that_disagree_keep_the_first_and_warn_once(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  for index in _LATER_OPENSSH_ROWS:
    vulns[index]["severity_level"] = 1

  report = _build_from(tmp_path, assets, vulns)

  with store_lib.Store(tmp_path / "out" / "index.sqlite") as opened:
    qid = opened.get_qid(_QID_OPENSSH)
  assert qid is not None and qid.severity == 4
  assert report.warnings == [
      "vulns.json: rows of QID 710001 disagree on severity_level, first at"
      " row 3; kept the values of the QID's first row"
  ]


# ---------------------------------------------------------------------------
# CVEs
# ---------------------------------------------------------------------------


def test_parse_cve_maps_every_field(vulns: _Records):
  cve = ingest.parse_cve(_cve_row(vulns, _CVE_KNOWN_EXPLOITED))

  assert cve == models.Cve(
      cve_id=_CVE_KNOWN_EXPLOITED,
      title="Ferrous Broker Management Listener Remote Code Execution",
      description=(
          "Ferrous broker 4.0.0 through 4.1.1 deserialises cluster join"
          " requests on its management listener before authentication. A"
          " remote, unauthenticated attacker can send a crafted join request"
          " and execute arbitrary code with the privileges of the broker"
          " process. The issue is fixed in 4.1.2."
      ),
      cvss=9.8,
      epss=0.91433,
      epss_percentile=0.99,
      attack_vector="network",
      known_exploited=True,
      kev_required_action=(
          "Apply the vendor's update, or stop exposing the management"
          " listener until the update is applied."
      ),
      vendor_fix="Upgrade Ferrous broker to version 4.1.2 or later.",
      cogent_risk_score=9.6,
      published="2099-03-28T16:45:00.000000",
      cwes=["CWE-502"],
      patch_refs=["https://git.example.org/ferrous/broker/commit/d07be44"],
      advisory_refs=["https://ferrous.example/security/FSA-2099-01"],
  )


def test_parse_cve_maps_what_the_export_does_not_know_to_none(
    vulns: _Records,
):
  row = _cve_row(vulns, _CVE_AGENT)
  assert row["cvss_base_score"] is None
  assert row["attack_vector"] == ""
  assert row["how_to_fix"] == "N/A"
  assert row["known_exploit_json"] == {}
  assert "weaknesses" not in row["cve_json"]

  cve = ingest.parse_cve(row)

  assert cve.cvss is None
  assert cve.attack_vector is None
  assert cve.vendor_fix is None
  assert cve.kev_required_action is None
  assert not cve.known_exploited
  assert not cve.cwes


@pytest.mark.parametrize("how_to_fix", ["", "  ", "N/A", " N/A "])
def test_parse_cve_treats_an_empty_fix_as_no_fix(
    vulns: _Records, how_to_fix: str
):
  row = _cve_row(vulns, _CVE_KNOWN_EXPLOITED)
  row["how_to_fix"] = how_to_fix

  assert ingest.parse_cve(row).vendor_fix is None


def test_parse_cve_drops_cwe_placeholders_and_repeats(vulns: _Records):
  row = _cve_row(vulns, _CVE_TEMPLATE)
  values = [
      description["value"]
      for weakness in row["cve_json"]["weaknesses"]
      for description in weakness["description"]
  ]
  assert values == ["CWE-125", "NVD-CWE-Other", "CWE-125", "CWE-1284"]

  assert ingest.parse_cve(row).cwes == ["CWE-125", "CWE-1284"]


def test_parse_cve_sorts_references_by_their_tags(vulns: _Records):
  cve = ingest.parse_cve(_cve_row(vulns, _CVE_PROXY))

  # The third reference, the release page, has no tags and goes nowhere.
  assert cve.patch_refs == [
      "https://git.example.org/trellis/proxy/commit/5f2d9c1"
  ]
  assert cve.advisory_refs == ["https://trellis.example/security/TSA-2099-003"]


def test_parse_cve_lists_an_advisory_once_under_either_advisory_tag(
    vulns: _Records,
):
  row = _cve_row(vulns, _CVE_SMUGGLING)
  tags = [reference.get("tags") for reference in row["cve_json"]["references"]]
  assert tags == [
      ["Mailing List", "Third Party Advisory"],
      ["Patch"],
      ["Vendor Advisory"],
  ]

  assert ingest.parse_cve(row).advisory_refs == [
      "https://lists.example.org/marlin-announce/2099/0007"
  ]


def test_parse_cve_ignores_a_reference_without_tags(vulns: _Records):
  row = _cve_row(vulns, _CVE_UNSCORED_KERNEL)
  assert all("tags" not in ref for ref in row["cve_json"]["references"])

  cve = ingest.parse_cve(row)

  assert not cve.patch_refs
  assert not cve.advisory_refs


def test_parse_cve_keeps_the_description_verbatim(vulns: _Records):
  row = _cve_row(vulns, _CVE_LONG_KERNEL)

  cve = ingest.parse_cve(row)

  assert cve.description == row["description"]
  assert cve.description.startswith("In the Linux kernel, the following")
  assert "<TASK>" in cve.description


def test_first_row_of_a_cve_wins(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  rows = [row for row in vulns if row.get("cve_id") == _CVE_PROXY]
  assert len(rows) > 1
  for row in rows[1:]:
    row["title"] = "A title from a later row"

  _build_from(tmp_path, assets, vulns)

  with store_lib.Store(tmp_path / "out" / "index.sqlite") as opened:
    cve = opened.get_cve(_CVE_PROXY)
  assert cve is not None
  assert cve.title == "Trellis Proxy Forwarded Header Removal Vulnerability"


def test_row_without_enrichment_adds_no_cve(store: store_lib.Store):
  # The web-server diagnosis names CVE-2099-3003, which has a vulns row but
  # no enrichment, as CVE-2022-22721 does in the reference dataset.
  assert store.get_cve("CVE-2099-3003") is None
  assert [cve.cve_id for cve in store.cves_for_qid("710004")] == [
      _CVE_SMUGGLING,
      _CVE_TEMPLATE,
  ]


# ---------------------------------------------------------------------------
# Chunks, the keyword index and the vectors
# ---------------------------------------------------------------------------


def test_trace_chunks_are_searchable_but_not_embedded(store: store_lib.Store):
  chunks = store.chunks_for_doc("cve", _CVE_LONG_KERNEL)
  trace_ids = {chunk.id for chunk in chunks if chunk.kind == "trace"}
  assert trace_ids

  hits = store.keyword_search('"fxnet_reset_task"', 10)
  embedded_ids, _ = store.load_embeddings()

  assert {chunk_id for chunk_id, _ in hits} <= trace_ids
  assert hits
  assert not trace_ids & set(embedded_ids)


def test_every_text_chunk_is_embedded(store: store_lib.Store):
  embedded_ids, _ = store.load_embeddings()

  chunks = store.get_chunks(list(range(1, 19)))

  assert embedded_ids == [chunk.id for chunk in chunks if chunk.kind == "text"]


def test_embedded_text_is_the_title_then_the_chunk_text(
    store: store_lib.Store,
):
  embedded_ids, matrix = store.load_embeddings()
  (chunk,) = store.chunks_for_doc("cve", _CVE_PROXY)

  expected = embeddings.HashingEmbedder().embed_documents(
      [f"{chunk.title}\n{chunk.text}"]
  )

  np.testing.assert_array_equal(
      matrix[embedded_ids.index(chunk.id)], expected[0]
  )


def test_qid_chunks_are_titled_with_the_label_and_cve_chunks_with_the_title(
    store: store_lib.Store,
):
  (diagnosis,) = store.chunks_for_doc("qid", _QID_KERNEL)
  (description,) = store.chunks_for_doc("cve", _CVE_PROXY)

  assert diagnosis.title == "Ubuntu security update: linux"
  assert description.title == (
      "Trellis Proxy Forwarded Header Removal Vulnerability"
  )


def test_kernel_boilerplate_is_kept_in_the_cve_but_not_in_its_chunks(
    store: store_lib.Store,
):
  cve = store.get_cve(_CVE_UNSCORED_KERNEL)
  (chunk,) = store.chunks_for_doc("cve", _CVE_UNSCORED_KERNEL)

  assert cve is not None
  assert cve.description.startswith("In the Linux kernel, the following")
  assert chunk.text.startswith("quartzfs: reject directory entries")


def test_cve_description_is_never_cleaned_as_html(store: store_lib.Store):
  (template,) = store.chunks_for_doc("cve", _CVE_TEMPLATE)
  kernel = store.chunks_for_doc("cve", _CVE_LONG_KERNEL)

  assert "a file that contains <include self>." in template.text
  assert any("<TASK>" in chunk.text for chunk in kernel)


def test_title_is_searchable_in_the_keyword_index(store: store_lib.Store):
  (chunk,) = store.chunks_for_doc("cve", _CVE_AGENT)
  assert "world" in chunk.title.lower()
  assert "world" not in chunk.text.lower()

  hits = store.keyword_search('"world"', 10)

  assert chunk.id in {chunk_id for chunk_id, _ in hits}


# ---------------------------------------------------------------------------
# Atomic output
# ---------------------------------------------------------------------------


def test_build_creates_the_directory_and_leaves_only_the_artifact(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  _build_from(tmp_path, assets, vulns)

  assert [path.name for path in (tmp_path / "out").iterdir()] == [
      "index.sqlite"
  ]


def test_build_replaces_an_existing_artifact(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  artifact = tmp_path / "out" / "index.sqlite"
  artifact.parent.mkdir()
  artifact.write_bytes(b"the artifact of an older dataset")

  _build_from(tmp_path, assets, vulns)

  with store_lib.Store(artifact) as opened:
    assert opened.stats().hosts == 11
  assert list(artifact.parent.iterdir()) == [artifact]


def test_build_that_fails_while_writing_leaves_nothing_behind(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  # Two hosts with one id pass every check and fail on INSERT, by which
  # time the temporary database exists.
  assets.append(copy.deepcopy(assets[0]))

  with pytest.raises(ingest.IngestError, match="constraint.*hosts.id"):
    _build_from(tmp_path, assets, vulns)

  assert not list((tmp_path / "out").iterdir())


def test_build_that_fails_does_not_touch_the_existing_artifact(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  artifact = tmp_path / "out" / "index.sqlite"
  artifact.parent.mkdir()
  artifact.write_bytes(b"the artifact that is being served")
  assets.append(copy.deepcopy(assets[0]))

  with pytest.raises(ingest.IngestError):
    _build_from(tmp_path, assets, vulns)

  assert artifact.read_bytes() == b"the artifact that is being served"
  assert list(artifact.parent.iterdir()) == [artifact]


class _WrongShapeEmbedder:
  """Claims eight dimensions and returns four."""

  name = "wrong-shape"
  dim = 8

  def embed_documents(self, texts: Sequence[str]) -> embeddings.Matrix:
    return np.zeros((len(texts), 4), dtype=np.float32)

  def embed_query(self, text: str) -> embeddings.Vector:
    del text  # Unused: ingest embeds documents only.
    return np.zeros(4, dtype=np.float32)


def test_embedder_that_returns_the_wrong_shape_fails_the_build(
    tmp_path: pathlib.Path, assets_path: pathlib.Path, vulns_path: pathlib.Path
):
  with pytest.raises(ingest.IngestError, match=r"wrong-shape.*\(16, 4\)"):
    ingest.build_artifact(
        assets_path,
        vulns_path,
        tmp_path / "index.sqlite",
        _WrongShapeEmbedder(),
        chunk_max_chars=1200,
        chunk_overlap_sentences=1,
    )

  assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Join validation
# ---------------------------------------------------------------------------


def test_vulns_row_for_an_unknown_host_fails_join_validation(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  vulns[_ROW]["asset"]["id"] = "424242"

  with pytest.raises(
      ingest.JoinValidationError,
      match=(
          rf"vulns\.json: row 7 \(host 424242, detection {_ROW_DETECTION}\):"
          " the host is not in the asset file"
      ),
  ):
    _build_from(tmp_path, assets, vulns)


def test_vulns_row_for_an_unknown_detection_fails_join_validation(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  # A real detection id, but of another host.
  vulns[_ROW]["hostInstanceVulnId"] = "51000000002"

  with pytest.raises(
      ingest.JoinValidationError,
      match=(
          rf"row 7 \(host {_ROW_HOST}, detection 51000000002\): the asset"
          " file lists no such detection under this host"
      ),
  ):
    _build_from(tmp_path, assets, vulns)


def test_vulns_row_with_another_qid_fails_join_validation(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  vulns[_ROW]["qid"] = "710003"

  with pytest.raises(
      ingest.JoinValidationError,
      match=(
          rf"row 7 \(host {_ROW_HOST}, detection {_ROW_DETECTION}\): qid is"
          " '710003' here and '710002' in the asset file"
      ),
  ):
    _build_from(tmp_path, assets, vulns)


@pytest.mark.parametrize("field", ["firstFound", "lastFound"])
def test_vulns_row_with_another_date_fails_join_validation(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records, field: str
):
  vulns[_ROW][field] = "2001-01-01T00:00:00Z"

  with pytest.raises(
      ingest.JoinValidationError,
      match=(
          rf"row 7 \(host {_ROW_HOST}, detection {_ROW_DETECTION}\): {field}"
          " is '2001-01-01T00:00:00Z' here and '2099-0[23]-01T06:00:00Z' in"
          " the asset file"
      ),
  ):
    _build_from(tmp_path, assets, vulns)


def test_vulns_row_whose_host_copy_differs_fails_join_validation(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  vulns[_ROW]["asset"]["name"] = "renamed"
  vulns[_ROW]["asset"]["sourceInfo"]["groupName"] = "another-sg"

  with pytest.raises(
      ingest.JoinValidationError,
      match=(
          rf"row 7 \(host {_ROW_HOST}, detection {_ROW_DETECTION}\): the"
          " embedded host record differs from the asset file in name,"
          " sourceInfo"
      ),
  ):
    _build_from(tmp_path, assets, vulns)


def test_join_validation_error_is_an_ingest_error():
  assert issubclass(ingest.JoinValidationError, ingest.IngestError)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_missing_export_is_an_ingest_error(
    tmp_path: pathlib.Path, vulns_path: pathlib.Path
):
  with pytest.raises(ingest.IngestError, match="cannot read .*absent.json"):
    ingest.build_artifact(
        tmp_path / "absent.json",
        vulns_path,
        tmp_path / "index.sqlite",
        embeddings.HashingEmbedder(),
        chunk_max_chars=1200,
        chunk_overlap_sentences=1,
    )


def test_export_that_is_not_json_is_an_ingest_error(
    tmp_path: pathlib.Path, assets_path: pathlib.Path
):
  broken = tmp_path / "vulns.json"
  broken.write_text('[{"qid": "710001"', encoding="utf-8")

  with pytest.raises(ingest.IngestError, match="cannot read .*vulns.json"):
    ingest.build_artifact(
        assets_path,
        broken,
        tmp_path / "index.sqlite",
        embeddings.HashingEmbedder(),
        chunk_max_chars=1200,
        chunk_overlap_sentences=1,
    )


def test_export_that_is_not_an_array_is_an_ingest_error(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  with pytest.raises(
      ingest.IngestError, match=r"assets\.json: expected a JSON array.*dict"
  ):
    _build_from(tmp_path, {"hosts": assets}, vulns)


def test_row_that_is_not_an_object_is_named_in_the_error(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  vulns[4] = "a string where a row belongs"

  with pytest.raises(
      ingest.IngestError, match=r"vulns\.json: row 4 is not a JSON object"
  ):
    _build_from(tmp_path, assets, vulns)


@pytest.mark.parametrize(
    "key",
    [
        "id",
        "name",
        "os",
        "criticalityScore",
        "isDockerHost",
        "lastVulnScan",
        "tags",
        "sourceInfo",
        "openPorts",
        "vulnerabilities",
    ],
)
def test_asset_record_without_a_required_key_is_named_in_the_error(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records, key: str
):
  del assets[2][key]

  with pytest.raises(
      ingest.IngestError,
      match=rf"assets\.json: row 2 is malformed: KeyError\('{key}'\)",
  ):
    _build_from(tmp_path, assets, vulns)


def test_asset_record_with_a_criticality_that_is_not_a_number_is_named(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records
):
  assets[5]["criticalityScore"] = "high"

  with pytest.raises(
      ingest.IngestError, match=r"assets\.json: row 5 is malformed: ValueError"
  ):
    _build_from(tmp_path, assets, vulns)


@pytest.mark.parametrize(
    "key",
    [
        "asset",
        "hostInstanceVulnId",
        "qid",
        "firstFound",
        "lastFound",
        "category",
        "severity_level",
        "pci_flag",
        "diagnosis",
    ],
)
def test_vulns_row_without_a_required_key_is_named_in_the_error(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records, key: str
):
  del vulns[_ROW][key]

  with pytest.raises(
      ingest.IngestError,
      match=rf"vulns\.json: row 7 is malformed: KeyError\('{key}'\)",
  ):
    _build_from(tmp_path, assets, vulns)


@pytest.mark.parametrize(
    "key",
    [
        "title",
        "description",
        "cvss_base_score",
        "epss",
        "epss_percentile",
        "attack_vector",
        "known_exploit",
        "known_exploit_json",
        "how_to_fix",
        "cogent_risk_score",
        "publish_date",
        "cve_json",
    ],
)
def test_enriched_row_without_a_required_key_is_named_in_the_error(
    tmp_path: pathlib.Path, assets: _Records, vulns: _Records, key: str
):
  assert vulns[_FIRST_PROXY_ROW]["cve_id"] == _CVE_PROXY
  del vulns[_FIRST_PROXY_ROW][key]

  with pytest.raises(
      ingest.IngestError,
      match=rf"vulns\.json: row 2 is malformed: KeyError\('{key}'\)",
  ):
    _build_from(tmp_path, assets, vulns)
