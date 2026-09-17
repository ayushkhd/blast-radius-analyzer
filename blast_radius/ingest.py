"""Builds the index artifact from the two scanner exports.

``build_artifact`` is the whole write path. It reads the asset export and
the vulnerability export, checks that the second really explains the first,
normalises both into the tables of ``blast_radius.schema``, cleans, chunks
and embeds the write-ups, and moves the finished SQLite file into place.

Three properties matter more than speed here, because the build runs once
per dataset and the service trusts what it produces:

* The join is validated, not assumed. Every vulns row must match a
  detection under the same host in the asset file, and the host record the
  row embeds must equal the asset file's. Only then are the embedded
  copies, which make up most of the vulns file, dropped.
* Detections come from the asset file alone. The vulns file explains a
  small subset of them, and a QID it does not explain is still recorded,
  with ``explained = 0``, so the service can say what it cannot explain.
* The output is atomic. The database is built in a temporary directory
  beside the target and renamed over it only when it is complete.

The functions that turn one export record into a model (``normalise_host``,
``parse_cve``, ``qid_label``) are pure, so they can be tested without a
database. The INSERT statements live here because ingest is the only
writer; every query is in ``blast_radius.store``.
"""

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import hashlib
import itertools
import json
import logging
import os
import pathlib
import re
import sqlite3
import tempfile
import textwrap
import time
from typing import Any, NamedTuple

from blast_radius import chunking
from blast_radius import embeddings
from blast_radius import models
from blast_radius import schema
from blast_radius import store
from blast_radius import textproc

_Record = dict[str, Any]

_INTERNET_FACING_TAG = "Internet Facing Assets"
_EKS_CLUSTER_TAG = "aws:eks:cluster-name"
_K8S_CLUSTER_TAG_PREFIX = "kubernetes.io/cluster/"
_ROLE_TAG = "role"

# NVD uses these in place of a CWE id when it has not assigned one.
_CWE_PLACEHOLDER_PREFIX = "NVD-CWE-"
_NO_VENDOR_FIX = "N/A"
_PATCH_TAGS = frozenset({"Patch"})
_ADVISORY_TAGS = frozenset({"Vendor Advisory", "Third Party Advisory"})

# What a vulns row must share with the detection it explains.
_DETECTION_FIELDS = ("qid", "firstFound", "lastFound")

_LABEL_MAX_CHARS = 90
_UBUNTU_UPDATE = re.compile(
    r"Ubuntu has released a security update for (\S+) to fix"
)
_PRODUCT_INTRODUCTION = re.compile(r"^[^,;:]+? is (?:a|an|the) ")
_LEADING_CVE_ID = re.compile(r"^CVE-\d{4}-\d{4,}\s*[:-]?\s*")

_EMBED_BATCH_SIZE = 64

# ``schema`` names the other keys of ``meta``; it has none for this setting.

_INSERT_HOST = """
INSERT INTO hosts (
  id, name, os, criticality, state, internet_facing, public_ip, private_ip,
  region, vpc_id, security_group, cluster, role, is_docker_host, last_scan)
VALUES (
  :id, :name, :os, :criticality, :state, :internet_facing, :public_ip,
  :private_ip, :region, :vpc_id, :security_group, :cluster, :role,
  :is_docker_host, :last_scan)
"""

_INSERT_OPEN_PORT = """
INSERT INTO open_ports (host_id, port, protocol, service)
VALUES (:host_id, :port, :protocol, :service)
"""

_INSERT_QID = """
INSERT INTO qids (
  qid, label, category, severity, pci_flag, diagnosis, explained)
VALUES (
  :qid, :label, :category, :severity, :pci_flag, :diagnosis, :explained)
"""

_INSERT_DETECTION = """
INSERT INTO detections (id, host_id, qid, first_found, last_found)
VALUES (:id, :host_id, :qid, :first_found, :last_found)
"""

_INSERT_CVE = """
INSERT INTO cves (
  cve_id, title, description, cvss, epss, epss_percentile, attack_vector,
  known_exploited, kev_required_action, vendor_fix, cogent_risk_score,
  published, cwes, patch_refs, advisory_refs)
VALUES (
  :cve_id, :title, :description, :cvss, :epss, :epss_percentile,
  :attack_vector, :known_exploited, :kev_required_action, :vendor_fix,
  :cogent_risk_score, :published, :cwes, :patch_refs, :advisory_refs)
"""

_INSERT_QID_CVE = "INSERT INTO qid_cves (qid, cve_id) VALUES (?, ?)"

_INSERT_CHUNK = """
INSERT INTO chunks (id, doc_type, doc_id, kind, title, text)
VALUES (:id, :doc_type, :doc_id, :kind, :title, :text)
"""

# The keyword index shares its row ids with ``chunks``.
_INSERT_CHUNK_FTS = """
INSERT INTO chunks_fts (rowid, title, body) VALUES (:id, :title, :text)
"""

_INSERT_EMBEDDING = "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)"

_INSERT_META = "INSERT INTO meta (key, value) VALUES (?, ?)"


class IngestError(Exception):
  """The exports cannot be turned into an artifact."""


class JoinValidationError(IngestError):
  """A vulns row does not match the asset file it claims to explain."""


@dataclasses.dataclass(frozen=True)
class IngestReport:
  """What one build wrote.

  Attributes:
    hosts: Rows in ``hosts``.
    open_ports: Rows in ``open_ports``.
    detections: Rows in ``detections``.
    qids: Rows in ``qids``, explained or not.
    explained_qids: How many of ``qids`` have a write-up.
    cves: Rows in ``cves``.
    qid_cves: Rows in ``qid_cves``.
    chunks: Rows in ``chunks``, and so in ``chunks_fts``.
    text_chunks: Chunks of prose.
    trace_chunks: Chunks of kernel log, which are indexed but not embedded.
    embeddings: Rows in ``embeddings``: one vector per text chunk.
    warnings: Oddities in the input that did not stop the build.
    duration_s: Wall-clock seconds the build took.
  """

  hosts: int
  open_ports: int
  detections: int
  qids: int
  explained_qids: int
  cves: int
  qid_cves: int
  chunks: int
  text_chunks: int
  trace_chunks: int
  embeddings: int
  warnings: list[str]
  duration_s: float


@dataclasses.dataclass(frozen=True)
class _Export:
  """One export file as read from disk.

  Attributes:
    path: Where it was read from; error messages name it.
    sha256: Hex digest of the file's bytes.
    records: The file's content, a JSON array of objects.
  """

  path: pathlib.Path
  sha256: str
  records: list[_Record]


class _QidFields(NamedTuple):
  """The QID-level keys of a vulns row, which every row of a QID repeats.

  The field names are the export's keys.

  Attributes:
    category: Qualys category.
    severity_level: Qualys severity, 1-5.
    pci_flag: 1 when the finding fails PCI compliance, else 0.
    diagnosis: The check's write-up, an HTML fragment.
  """

  category: str
  severity_level: int
  pci_flag: int
  diagnosis: str


@dataclasses.dataclass(frozen=True)
class _Inventory:
  """The asset export, normalised.

  Attributes:
    hosts: One per asset record.
    ports: ``(host_id, port)`` pairs.
    detections: Every detection of every host.
  """

  hosts: list[models.Host]
  ports: list[tuple[str, models.OpenPort]]
  detections: list[models.Detection]


@dataclasses.dataclass(frozen=True)
class _Corpus:
  """The write-ups of the vulns export, normalised.

  Attributes:
    qids: One per detected QID, sorted by QID, explained or not.
    cves: One per CVE id, sorted by CVE id.
    qid_cves: The ``(qid, cve_id)`` pairs the export contains.
    warnings: Disagreements between rows, resolved by keeping the first.
  """

  qids: list[models.Qid]
  cves: list[models.Cve]
  qid_cves: list[tuple[str, str]]
  warnings: list[str]


# ---------------------------------------------------------------------------
# Reading the exports
# ---------------------------------------------------------------------------


def _read_export(path: pathlib.Path) -> _Export:
  """Reads and hashes one export.

  The hash and the records come from one open file, so the digest recorded
  in the artifact is that of the bytes that were actually parsed.

  Args:
    path: A JSON file holding an array of objects.

  Raises:
    IngestError: If the file cannot be read, is not JSON, or is not an array
      of objects.
  """
  try:
    with path.open("rb") as file:
      sha256 = hashlib.file_digest(file, "sha256").hexdigest()
      file.seek(0)
      data = json.load(file)
  except (OSError, ValueError) as err:
    raise IngestError(f"cannot read {path}: {err}") from err
  if not isinstance(data, list):
    raise IngestError(
        f"{path.name}: expected a JSON array of records, found"
        f" {type(data).__name__}"
    )
  for index, record in enumerate(data):
    if not isinstance(record, dict):
      raise IngestError(f"{path.name}: row {index} is not a JSON object")
  logging.info("read %d records from %s", len(data), path)
  return _Export(path=path, sha256=sha256, records=data)


@contextlib.contextmanager
def _named_row(export: _Export, index: int) -> Iterator[None]:
  """Turns what a malformed record breaks into an error that names it.

  The parsing functions index straight into the records. A missing key, a
  null where an object belongs or a number that does not parse surfaces as
  one of the three exceptions below, and is re-raised with the file and the
  row, which is what someone repairing an export needs.

  Args:
    export: The export the record belongs to.
    index: The record's position in the export.

  Raises:
    IngestError: If the body raises KeyError, TypeError or ValueError.
  """
  try:
    yield
  except (KeyError, TypeError, ValueError) as err:
    raise IngestError(
        f"{export.path.name}: row {index} is malformed: {err!r}"
    ) from err


# ---------------------------------------------------------------------------
# The asset export
# ---------------------------------------------------------------------------


def _cluster(ec2_tags: Mapping[str, str | None]) -> str | None:
  """Returns the Kubernetes cluster that a host's EC2 tags name, if any.

  EKS tags its nodes with the cluster name. Other nodes carry only the
  ``kubernetes.io/cluster/<name>`` key, whose value is an ownership marker
  and not a name.

  Args:
    ec2_tags: The host's EC2 tags, value by key.
  """
  eks_cluster = ec2_tags.get(_EKS_CLUSTER_TAG)
  if eks_cluster:
    return eks_cluster
  for key in ec2_tags:
    if key.startswith(_K8S_CLUSTER_TAG_PREFIX):
      return key.removeprefix(_K8S_CLUSTER_TAG_PREFIX)
  return None


def normalise_host(record: Mapping[str, Any]) -> models.Host:
  """Returns the host described by one record of the asset export.

  The export writes numbers and booleans as strings (``"5"``, ``"true"``),
  and keeps most of what matters under ``sourceInfo``, the cloud provider's
  view of the instance, which includes its EC2 tags.

  A host counts as internet-facing when the scanner tagged it so or when it
  has a public IP address. The two agree in the reference dataset; either
  alone is reason enough to treat a host as exposed.

  Args:
    record: One element of the asset export.
  """
  source = record["sourceInfo"]
  ec2_tags = {tag["key"]: tag["value"] for tag in source["ec2InstanceTags"]}
  public_ip = source["publicIpAddress"] or None
  tagged = any(tag["name"] == _INTERNET_FACING_TAG for tag in record["tags"])
  return models.Host(
      id=record["id"],
      name=record["name"],
      os=record["os"],
      criticality=int(record["criticalityScore"]),
      state=source["instanceState"],
      internet_facing=tagged or public_ip is not None,
      public_ip=public_ip,
      private_ip=source["privateIpAddress"],
      region=source["region"],
      vpc_id=source["vpcId"],
      security_group=source["groupName"],
      cluster=_cluster(ec2_tags),
      role=ec2_tags.get(_ROLE_TAG),
      is_docker_host=record["isDockerHost"] == "true",
      last_scan=record["lastVulnScan"],
  )


def _open_ports(
    record: Mapping[str, Any],
) -> list[tuple[str, models.OpenPort]]:
  """Returns the open ports of one asset record, each with its host id."""
  return [
      (
          record["id"],
          models.OpenPort(
              port=int(port["port"]),
              protocol=port["protocol"],
              service=port["serviceName"],
          ),
      )
      for port in record["openPorts"]
  ]


def _detections(record: Mapping[str, Any]) -> list[models.Detection]:
  """Returns the detections listed under one asset record."""
  return [
      models.Detection(
          id=detection["hostInstanceVulnId"],
          host_id=record["id"],
          qid=detection["qid"],
          first_found=detection["firstFound"],
          last_found=detection["lastFound"],
      )
      for detection in record["vulnerabilities"]
  ]


def _parse_inventory(assets: _Export) -> _Inventory:
  """Normalises the asset export.

  Args:
    assets: The asset export.

  Raises:
    IngestError: If a record is malformed.
  """
  hosts: list[models.Host] = []
  ports: list[tuple[str, models.OpenPort]] = []
  detections: list[models.Detection] = []
  for index, record in enumerate(assets.records):
    with _named_row(assets, index):
      hosts.append(normalise_host(record))
      ports.extend(_open_ports(record))
      detections.extend(_detections(record))
  logging.info(
      "parsed %d hosts, %d open ports and %d detections",
      len(hosts),
      len(ports),
      len(detections),
  )
  return _Inventory(hosts=hosts, ports=ports, detections=detections)


# ---------------------------------------------------------------------------
# The join between the two exports
# ---------------------------------------------------------------------------


def _join_problem(
    row: Mapping[str, Any],
    hosts: Mapping[str, _Record],
    detections: Mapping[tuple[str, str], _Record],
) -> str | None:
  """Returns how a vulns row contradicts the asset file, or None.

  Args:
    row: One row of the vulns export.
    hosts: The asset records by host id.
    detections: The asset file's detections by host id and detection id.
  """
  embedded = row["asset"]
  host = hosts.get(embedded["id"])
  if host is None:
    return "the host is not in the asset file"
  if embedded != host:
    differing = ", ".join(
        sorted(
            key
            for key in embedded.keys() | host.keys()
            if embedded.get(key) != host.get(key)
        )
    )
    return (
        f"the embedded host record differs from the asset file in {differing}"
    )
  detection = detections.get((host["id"], row["hostInstanceVulnId"]))
  if detection is None:
    return "the asset file lists no such detection under this host"
  for field in _DETECTION_FIELDS:
    if row[field] != detection[field]:
      return (
          f"{field} is {row[field]!r} here and {detection[field]!r} in the"
          " asset file"
      )
  return None


def _validate_join(assets: _Export, vulns: _Export) -> None:
  """Checks that every vulns row explains a detection in the asset file.

  Args:
    assets: The asset export.
    vulns: The vulns export, host copies still embedded.

  Raises:
    JoinValidationError: At the first row that does not match, naming the
      row, the host, the detection and what differed.
    IngestError: If a record of either export is malformed.
  """
  hosts: dict[str, _Record] = {}
  detections: dict[tuple[str, str], _Record] = {}
  for index, record in enumerate(assets.records):
    with _named_row(assets, index):
      hosts[record["id"]] = record
      for detection in record["vulnerabilities"]:
        detections[record["id"], detection["hostInstanceVulnId"]] = detection

  for index, row in enumerate(vulns.records):
    with _named_row(vulns, index):
      problem = _join_problem(row, hosts, detections)
      if problem:
        host_id = row["asset"]["id"]
        detection_id = row["hostInstanceVulnId"]
        raise JoinValidationError(
            f"{vulns.path.name}: row {index} (host {host_id}, detection"
            f" {detection_id}): {problem}"
        )
  logging.info(
      "validated %d vulns rows against %d hosts", len(vulns.records), len(hosts)
  )


def _drop_host_copies(vulns: _Export) -> None:
  """Deletes the host record embedded in every vulns row, in place.

  The copies are most of the vulns file. Once the join is validated they
  say nothing that the asset file does not, and dropping them frees their
  memory before the embedding model needs it.

  Args:
    vulns: The vulns export, already validated.
  """
  for row in vulns.records:
    del row["asset"]


# ---------------------------------------------------------------------------
# The vulns export
# ---------------------------------------------------------------------------


def _cwes(cve_json: Mapping[str, Any]) -> list[str]:
  """Returns the CWE ids NVD assigned, without its "no CWE" placeholders."""
  values = {
      description["value"]
      for weakness in cve_json.get("weaknesses", [])
      for description in weakness["description"]
  }
  return sorted(
      value for value in values if not value.startswith(_CWE_PLACEHOLDER_PREFIX)
  )


def _reference_urls(
    cve_json: Mapping[str, Any], wanted_tags: frozenset[str]
) -> list[str]:
  """Returns the URLs of NVD references that carry any of ``wanted_tags``.

  The order is NVD's, and a URL that is listed twice is returned once.

  Args:
    cve_json: The NVD record embedded in a vulns row.
    wanted_tags: NVD reference tags, such as "Patch".
  """
  urls = [
      reference["url"]
      for reference in cve_json["references"]
      if wanted_tags.intersection(reference.get("tags", []))
  ]
  return list(dict.fromkeys(urls))


def parse_cve(row: Mapping[str, Any]) -> models.Cve:
  """Returns the CVE that an enriched row of the vulns export describes.

  The description is kept exactly as it is. It is NVD plain text in which
  ``<TASK>`` or ``<linux/foo.h>`` is content, so it must never be cleaned
  as HTML. Values that mean "unknown" in the export, an empty attack vector
  and a ``how_to_fix`` of "N/A", become None.

  Args:
    row: A row of the vulns export that has a ``cve_id``.
  """
  cve_json = row["cve_json"]
  vendor_fix = row["how_to_fix"].strip()
  return models.Cve(
      cve_id=row["cve_id"],
      title=row["title"],
      description=row["description"],
      cvss=row["cvss_base_score"],
      epss=row["epss"],
      epss_percentile=row["epss_percentile"],
      attack_vector=row["attack_vector"] or None,
      known_exploited=row["known_exploit"],
      kev_required_action=row["known_exploit_json"].get("requiredAction"),
      vendor_fix=None if vendor_fix in ("", _NO_VENDOR_FIX) else vendor_fix,
      cogent_risk_score=row["cogent_risk_score"],
      published=row["publish_date"],
      cwes=_cwes(cve_json),
      patch_refs=_reference_urls(cve_json, _PATCH_TAGS),
      advisory_refs=_reference_urls(cve_json, _ADVISORY_TAGS),
  )


def _shorten(text: str) -> str:
  """Returns ``text`` cut at a word boundary to the length of a label."""
  return textwrap.shorten(text, width=_LABEL_MAX_CHARS, placeholder="...")


def qid_label(diagnosis: str, cve_titles: Sequence[str]) -> str:
  """Returns a short display name for a QID.

  The exports give a QID no title, only a diagnosis of several paragraphs,
  so one is derived. The rules, in order:

  1. A QID with exactly one CVE takes that CVE's title.
  2. A distribution update ("Ubuntu has released a security update for
     linux ...") becomes ``Ubuntu security update: linux``.
  3. Otherwise the label is the first sentence of the diagnosis that says
     something about the weakness. That excludes headings, which end in a
     colon ("Affected Versions:"), and sentences that introduce the product
     ("OpenSSH is a set of ..."). A CVE id that opens the sentence is
     dropped.
  4. If no sentence qualifies, the first sentence is used.

  The label is for display and for the title of the QID's chunks; nothing
  is decided by it. It is at most ``_LABEL_MAX_CHARS`` characters long.

  Args:
    diagnosis: The QID's diagnosis as plain text.
    cve_titles: The titles of the CVEs that the QID bundles.
  """
  if len(cve_titles) == 1 and cve_titles[0]:
    return _shorten(cve_titles[0])
  ubuntu_update = _UBUNTU_UPDATE.match(diagnosis)
  if ubuntu_update:
    return f"Ubuntu security update: {ubuntu_update.group(1)}"
  sentences = [
      _LEADING_CVE_ID.sub("", sentence)
      for sentence in textproc.split_sentences(diagnosis)
  ]
  for sentence in sentences:
    is_heading = not sentence or sentence.endswith(":")
    if not is_heading and not _PRODUCT_INTRODUCTION.match(sentence):
      return _shorten(sentence)
  return _shorten(sentences[0]) if sentences else ""


def _collect_cves(vulns: _Export) -> dict[str, models.Cve]:
  """Returns the CVEs of the vulns export by id; the first row of each wins.

  Args:
    vulns: The vulns export.

  Raises:
    IngestError: If an enriched row is malformed.
  """
  cves: dict[str, models.Cve] = {}
  for index, row in enumerate(vulns.records):
    cve_id = row.get("cve_id")
    if cve_id is not None and cve_id not in cves:
      with _named_row(vulns, index):
        cves[cve_id] = parse_cve(row)
  return cves


def _collect_qid_fields(
    vulns: _Export,
) -> tuple[dict[str, _QidFields], list[str]]:
  """Returns the QID-level fields of every explained QID, with warnings.

  The rows of one QID agree on these fields throughout the reference
  dataset. Should they ever not, the first row wins and the QID is
  reported, once.

  Args:
    vulns: The vulns export.

  Returns:
    The fields by QID, and one warning per QID whose rows disagree.

  Raises:
    IngestError: If a row lacks one of the fields.
  """
  fields_by_qid: dict[str, _QidFields] = {}
  warnings: dict[str, str] = {}
  for index, row in enumerate(vulns.records):
    with _named_row(vulns, index):
      qid = row["qid"]
      fields = _QidFields(*(row[name] for name in _QidFields._fields))
    first = fields_by_qid.setdefault(qid, fields)
    if fields != first and qid not in warnings:
      differing = ", ".join(
          name
          for name, a, b in zip(_QidFields._fields, first, fields)
          if a != b
      )
      warnings[qid] = (
          f"{vulns.path.name}: rows of QID {qid} disagree on {differing},"
          f" first at row {index}; kept the values of the QID's first row"
      )
  return fields_by_qid, list(warnings.values())


def _parse_corpus(vulns: _Export, detected_qids: set[str]) -> _Corpus:
  """Normalises the vulns export into QIDs, CVEs and the pairs between them.

  Args:
    vulns: The vulns export.
    detected_qids: Every QID that occurs in a detection. One that the vulns
      export does not explain still gets a row, marked unexplained.

  Raises:
    IngestError: If a row is malformed.
  """
  cves = _collect_cves(vulns)
  fields_by_qid, warnings = _collect_qid_fields(vulns)
  # A dict keeps the pairs in the export's order and drops the repeats.
  pairs = dict.fromkeys(
      (row["qid"], row["cve_id"]) for row in vulns.records if "cve_id" in row
  )
  titles_by_qid: dict[str, list[str]] = {}
  for qid, cve_id in pairs:
    titles_by_qid.setdefault(qid, []).append(cves[cve_id].title)

  qids = []
  for qid in sorted(detected_qids):
    if qid not in fields_by_qid:
      qids.append(models.Qid(qid=qid))
      continue
    fields = fields_by_qid[qid]
    # Diagnoses are HTML fragments. CVE descriptions are not.
    diagnosis = textproc.strip_html(fields.diagnosis)
    qids.append(
        models.Qid(
            qid=qid,
            label=qid_label(diagnosis, titles_by_qid.get(qid, [])),
            category=fields.category,
            severity=fields.severity_level,
            pci_flag=bool(fields.pci_flag),
            diagnosis=diagnosis,
            explained=True,
        )
    )
  logging.info(
      "collected %d QIDs (%d explained), %d CVEs and %d QID-CVE pairs",
      len(qids),
      len(fields_by_qid),
      len(cves),
      len(pairs),
  )
  return _Corpus(
      qids=qids,
      cves=[cves[cve_id] for cve_id in sorted(cves)],
      qid_cves=list(pairs),
      warnings=warnings,
  )


# ---------------------------------------------------------------------------
# The retrieval index
# ---------------------------------------------------------------------------


def _build_chunks(
    corpus: _Corpus, max_chars: int, overlap_sentences: int
) -> list[models.Chunk]:
  """Chunks every write-up, numbering the chunks from 1 in document order.

  A document is a QID's diagnosis, titled with the QID's label, or a CVE's
  description without the sentence that opens every kernel CVE, titled with
  the CVE's title. A document with no text yields no chunk.

  Args:
    corpus: The normalised write-ups.
    max_chars: Upper bound on chunk length.
    overlap_sentences: Sentences repeated between adjacent prose chunks.
  """
  documents: list[tuple[models.DocType, str, str, str]] = []
  for qid in corpus.qids:
    if qid.explained:
      documents.append(("qid", qid.qid, qid.label, qid.diagnosis or ""))
  for cve in corpus.cves:
    text = textproc.remove_kernel_boilerplate(cve.description)
    documents.append(("cve", cve.cve_id, cve.title or cve.cve_id, text))

  chunks: list[models.Chunk] = []
  for doc_type, doc_id, title, text in documents:
    pieces = chunking.chunk_document(
        text, max_chars=max_chars, overlap_sentences=overlap_sentences
    )
    for piece in pieces:
      chunks.append(
          models.Chunk(
              id=len(chunks) + 1,
              doc_type=doc_type,
              doc_id=doc_id,
              kind=piece.kind,
              title=title,
              text=piece.text,
          )
      )
  logging.info("built %d chunks from %d documents", len(chunks), len(documents))
  return chunks


def _embed_chunks(
    chunks: Sequence[models.Chunk], embedder: embeddings.Embedder
) -> list[tuple[int, bytes]]:
  """Embeds the prose chunks and serialises the vectors for storage.

  Trace chunks are skipped: a register dump is noise to an embedding model.
  Each chunk is embedded together with its document's title, which carries
  the product and the kind of weakness that a lone chunk often lacks.

  Args:
    chunks: Every chunk of the corpus.
    embedder: The embedding model.

  Returns:
    ``(chunk_id, vector)`` pairs, each vector as little-endian float32
    bytes.

  Raises:
    IngestError: If the embedder returns vectors of the wrong shape.
  """
  prose = [chunk for chunk in chunks if chunk.kind == "text"]
  vectors: list[tuple[int, bytes]] = []
  for batch in itertools.batched(prose, _EMBED_BATCH_SIZE):
    texts = [f"{chunk.title}\n{chunk.text}" for chunk in batch]
    matrix = embedder.embed_documents(texts)
    if matrix.shape != (len(batch), embedder.dim):
      raise IngestError(
          f"embedder {embedder.name} returned shape {matrix.shape} for"
          f" {len(batch)} texts of dimension {embedder.dim}"
      )
    for chunk, vector in zip(batch, matrix):
      vectors.append((chunk.id, vector.astype("<f4").tobytes()))
    logging.info("embedded %d of %d chunks", len(vectors), len(prose))
  return vectors


# ---------------------------------------------------------------------------
# Writing the artifact
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _new_artifact(artifact_path: pathlib.Path) -> Iterator[sqlite3.Connection]:
  """Yields a connection to a new artifact that replaces ``artifact_path``.

  The database is built in a temporary directory beside the target, which
  puts it on the same filesystem, so the final ``os.replace`` is atomic.
  Everything written through the connection is one transaction. If the body
  raises, the transaction is rolled back, the temporary directory is
  removed with whatever SQLite left in it, and the target is untouched.

  Args:
    artifact_path: Where the finished artifact goes. Its directory is
      created if need be.
  """
  artifact_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
      dir=artifact_path.parent, prefix=f".{artifact_path.name}."
  ) as tmp_dir:
    tmp_path = pathlib.Path(tmp_dir) / artifact_path.name
    with contextlib.closing(store.create(tmp_path)) as conn:
      with conn:
        yield conn
    os.replace(tmp_path, artifact_path)


def _cve_row(cve: models.Cve) -> dict[str, Any]:
  """Returns a CVE as ``cves`` stores it, its list columns as JSON arrays."""
  row = cve.model_dump()
  for column in ("cwes", "patch_refs", "advisory_refs"):
    row[column] = json.dumps(row[column])
  return row


def _write_tables(
    conn: sqlite3.Connection,
    inventory: _Inventory,
    corpus: _Corpus,
    chunks: Sequence[models.Chunk],
    vectors: Sequence[tuple[int, bytes]],
    meta: Mapping[str, str],
) -> None:
  """Fills every table of a new artifact.

  Rows are bound by column name from the models' own field names. Parents
  are written before the rows that reference them, because the connection
  enforces foreign keys.

  Args:
    conn: A connection from ``store.create``.
    inventory: Hosts, ports and detections.
    corpus: QIDs, CVEs and their pairs.
    chunks: The chunks of every write-up.
    vectors: ``(chunk_id, vector)`` pairs for the embedded chunks.
    meta: What the artifact was built from.

  Raises:
    IngestError: If the exports break a constraint of the schema, such as
      two hosts that share an id.
  """
  port_rows = (
      {"host_id": host_id, **port.model_dump()}
      for host_id, port in inventory.ports
  )
  chunk_rows = [chunk.model_dump() for chunk in chunks]
  try:
    conn.executemany(_INSERT_HOST, (h.model_dump() for h in inventory.hosts))
    conn.executemany(_INSERT_OPEN_PORT, port_rows)
    conn.executemany(_INSERT_QID, (q.model_dump() for q in corpus.qids))
    conn.executemany(
        _INSERT_DETECTION, (d.model_dump() for d in inventory.detections)
    )
    conn.executemany(_INSERT_CVE, (_cve_row(cve) for cve in corpus.cves))
    conn.executemany(_INSERT_QID_CVE, corpus.qid_cves)
    conn.executemany(_INSERT_CHUNK, chunk_rows)
    conn.executemany(_INSERT_CHUNK_FTS, chunk_rows)
    conn.executemany(_INSERT_EMBEDDING, vectors)
    conn.executemany(_INSERT_META, meta.items())
  except sqlite3.IntegrityError as err:
    raise IngestError(
        f"the exports break a constraint of the index: {err}"
    ) from err


def _build_meta(
    assets: _Export,
    vulns: _Export,
    embedder: embeddings.Embedder,
    chunk_max_chars: int,
    chunk_overlap_sentences: int,
) -> dict[str, str]:
  """Returns the ``meta`` table: what the artifact was built from and how."""
  built_at = datetime.datetime.now(datetime.timezone.utc)
  return {
      schema.META_SCHEMA_VERSION: str(schema.SCHEMA_VERSION),
      schema.META_BUILT_AT: built_at.isoformat(timespec="seconds"),
      schema.META_ASSETS_SHA256: assets.sha256,
      schema.META_VULNS_SHA256: vulns.sha256,
      schema.META_EMBEDDING_MODEL: embedder.name,
      schema.META_EMBEDDING_DIM: str(embedder.dim),
      schema.META_CHUNK_MAX_CHARS: str(chunk_max_chars),
      schema.META_CHUNK_OVERLAP_SENTENCES: str(chunk_overlap_sentences),
  }


def build_artifact(
    assets_path: pathlib.Path,
    vulns_path: pathlib.Path,
    artifact_path: pathlib.Path,
    embedder: embeddings.Embedder,
    *,
    chunk_max_chars: int,
    chunk_overlap_sentences: int,
) -> IngestReport:
  """Builds the index artifact and moves it into place.

  The vulns export is read with ``json.load``, which for the reference
  dataset takes a few seconds and well over a gigabyte of memory. That cost
  is paid here, once, so that the service never opens the raw exports.

  Args:
    assets_path: The asset export.
    vulns_path: The vulnerability export.
    artifact_path: Where to put the artifact. An existing artifact is
      replaced atomically, and only if the build succeeds.
    embedder: Embeds the prose chunks. Its name and dimension are recorded,
      so that queries can be embedded with the same model.
    chunk_max_chars: Upper bound on chunk length.
    chunk_overlap_sentences: Sentences repeated between adjacent chunks.

  Returns:
    What was written, with any warnings.

  Raises:
    JoinValidationError: If a vulns row does not match the asset file.
    IngestError: If an export is unreadable or malformed, or the embedder
      returns vectors of the wrong shape.
  """
  started = time.monotonic()
  assets = _read_export(assets_path)
  vulns = _read_export(vulns_path)
  inventory = _parse_inventory(assets)
  _validate_join(assets, vulns)
  _drop_host_copies(vulns)
  corpus = _parse_corpus(vulns, {d.qid for d in inventory.detections})
  for warning in corpus.warnings:
    logging.warning("%s", warning)
  chunks = _build_chunks(corpus, chunk_max_chars, chunk_overlap_sentences)
  vectors = _embed_chunks(chunks, embedder)
  meta = _build_meta(
      assets, vulns, embedder, chunk_max_chars, chunk_overlap_sentences
  )
  with _new_artifact(artifact_path) as conn:
    _write_tables(conn, inventory, corpus, chunks, vectors, meta)

  text_chunks = sum(chunk.kind == "text" for chunk in chunks)
  report = IngestReport(
      hosts=len(inventory.hosts),
      open_ports=len(inventory.ports),
      detections=len(inventory.detections),
      qids=len(corpus.qids),
      explained_qids=sum(qid.explained for qid in corpus.qids),
      cves=len(corpus.cves),
      qid_cves=len(corpus.qid_cves),
      chunks=len(chunks),
      text_chunks=text_chunks,
      trace_chunks=len(chunks) - text_chunks,
      embeddings=len(vectors),
      warnings=corpus.warnings,
      duration_s=time.monotonic() - started,
  )
  logging.info("wrote %s in %.1f s", artifact_path, report.duration_s)
  return report
