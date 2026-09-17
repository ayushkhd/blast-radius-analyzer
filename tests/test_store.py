"""Tests for blast_radius.store."""

from concurrent import futures
import contextlib
import pathlib
import sqlite3
import threading

import numpy as np
import pytest

from blast_radius import models
from blast_radius import schema
from blast_radius import store as store_lib

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_BASTION = "900000001"
_GATEWAY = "900000002"
_WORKER_A_FIRST = "900000003"
_WORKER_A_SECOND = "900000004"
_WORKER_B1 = "900000005"
_WORKER_B2 = "900000006"
_WORKER_TERMINATED = "900000007"
_WEB = "900000008"
_QUEUE = "900000009"
_IDLE = "900000011"

_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_PROXY = "710003"
_QID_UNEXPLAINED = "710901"

_CVE_PROXY = "CVE-2099-2001"
_CVE_LONG_KERNEL = "CVE-2099-1001"

# Chunk ids follow document order: the six QIDs, then the CVEs by id.
_CHUNK_OPENSSH_DIAGNOSIS = 1
_CHUNK_KERNEL_DIAGNOSIS = 2
_CHUNK_QUARTZFS_CVE = 11
_CHUNK_PROXY_CVE = 14


def _empty_artifact(path: pathlib.Path, meta: dict[str, str]) -> pathlib.Path:
  """Creates an artifact that holds nothing but the given ``meta`` rows."""
  with contextlib.closing(store_lib.create(path)) as conn:
    with conn:
      conn.executemany(
          "INSERT INTO meta (key, value) VALUES (?, ?)", meta.items()
      )
  return path


def _current_version() -> dict[str, str]:
  """Returns the ``meta`` row that makes an artifact readable."""
  return {schema.META_SCHEMA_VERSION: str(schema.SCHEMA_VERSION)}


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_applies_the_schema(tmp_path: pathlib.Path):
  with contextlib.closing(store_lib.create(tmp_path / "new.sqlite")) as conn:
    tables = {
        name
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

  assert tables >= {
      "hosts",
      "open_ports",
      "detections",
      "qids",
      "cves",
      "qid_cves",
      "chunks",
      "chunks_fts",
      "embeddings",
      "meta",
  }


def test_create_refuses_to_overwrite_an_existing_file(tmp_path: pathlib.Path):
  path = tmp_path / "index.sqlite"
  path.write_bytes(b"an artifact that is being served")

  with pytest.raises(FileExistsError):
    store_lib.create(path)

  assert path.read_bytes() == b"an artifact that is being served"


def test_create_returns_a_connection_that_enforces_foreign_keys(
    tmp_path: pathlib.Path,
):
  with contextlib.closing(store_lib.create(tmp_path / "new.sqlite")) as conn:
    with pytest.raises(sqlite3.IntegrityError):
      conn.execute(
          "INSERT INTO open_ports (host_id, port, protocol) VALUES (?, ?, ?)",
          ("no-such-host", 22, "TCP"),
      )


# ---------------------------------------------------------------------------
# Opening and closing
# ---------------------------------------------------------------------------


def test_store_on_a_missing_artifact_raises_artifact_not_found(
    tmp_path: pathlib.Path,
):
  with pytest.raises(
      store_lib.ArtifactNotFoundError, match="blast-radius ingest"
  ):
    store_lib.Store(tmp_path / "absent.sqlite")


def test_store_on_another_schema_version_raises_schema_version_error(
    tmp_path: pathlib.Path,
):
  other = str(schema.SCHEMA_VERSION + 1)
  path = _empty_artifact(
      tmp_path / "index.sqlite", {schema.META_SCHEMA_VERSION: other}
  )

  with pytest.raises(store_lib.SchemaVersionError, match=f"version {other}"):
    store_lib.Store(path)


def test_store_on_an_artifact_without_a_version_raises_schema_version_error(
    tmp_path: pathlib.Path,
):
  path = _empty_artifact(tmp_path / "index.sqlite", {})

  with pytest.raises(store_lib.SchemaVersionError):
    store_lib.Store(path)


def test_store_on_a_file_that_is_not_a_database_raises_store_error(
    tmp_path: pathlib.Path,
):
  path = tmp_path / "index.sqlite"
  path.write_text("this is not an SQLite database, however it is named")

  with pytest.raises(store_lib.StoreError, match="not an index artifact"):
    store_lib.Store(path)


def test_store_on_a_path_with_uri_characters_opens(
    tmp_path: pathlib.Path,
):
  directory = tmp_path / "odd ? name # here"
  directory.mkdir()
  path = _empty_artifact(directory / "index.sqlite", _current_version())

  with store_lib.Store(path) as opened:
    assert opened.meta() == _current_version()


def test_store_used_after_close_raises_store_error(
    artifact_path: pathlib.Path,
):
  opened = store_lib.Store(artifact_path)
  opened.close()

  with pytest.raises(store_lib.StoreError, match="closed"):
    opened.stats()


def test_store_closes_when_its_with_block_ends(artifact_path: pathlib.Path):
  with store_lib.Store(artifact_path) as opened:
    pass

  with pytest.raises(store_lib.StoreError, match="closed"):
    opened.stats()


def test_store_can_be_closed_twice(artifact_path: pathlib.Path):
  opened = store_lib.Store(artifact_path)
  opened.close()

  opened.close()


def test_store_serves_queries_from_other_threads(store: store_lib.Store):
  host_ids = [_BASTION, _GATEWAY, _WORKER_B1, _WEB, _QUEUE, _IDLE] * 4
  expected = [store.get_host(host_id) for host_id in host_ids]

  with futures.ThreadPoolExecutor(max_workers=4) as pool:
    hosts = list(pool.map(store.get_host, host_ids))

  assert hosts == expected


def test_close_closes_connections_that_other_threads_opened(
    artifact_path: pathlib.Path,
):
  opened = store_lib.Store(artifact_path)
  worker = threading.Thread(target=opened.stats)
  worker.start()
  worker.join()

  opened.close()

  with pytest.raises(store_lib.StoreError, match="closed"):
    opened.stats()


# ---------------------------------------------------------------------------
# meta and stats
# ---------------------------------------------------------------------------


def test_meta_returns_every_row_as_strings(store: store_lib.Store):
  meta = store.meta()

  assert meta[schema.META_SCHEMA_VERSION] == str(schema.SCHEMA_VERSION)
  assert meta[schema.META_EMBEDDING_MODEL] == "hashing-256"
  assert meta[schema.META_EMBEDDING_DIM] == "256"


def test_stats_counts_what_the_corpus_explains(store: store_lib.Store):
  assert store.stats() == models.CorpusStats(
      hosts=11,
      hosts_with_detections=10,
      total_qids=8,
      explained_qids=6,
      total_detections=45,
      explained_detections=26,
      cves=9,
      chunks=18,
  )


# ---------------------------------------------------------------------------
# QIDs and CVEs
# ---------------------------------------------------------------------------


def test_get_cve_returns_the_cve_with_its_lists_decoded(
    store: store_lib.Store,
):
  cve = store.get_cve(_CVE_PROXY)

  assert cve is not None
  assert cve.title == "Trellis Proxy Forwarded Header Removal Vulnerability"
  assert cve.cvss == 7.5
  assert cve.attack_vector == "network"
  assert cve.cwes == ["CWE-345", "CWE-348"]
  assert cve.patch_refs == [
      "https://git.example.org/trellis/proxy/commit/5f2d9c1"
  ]
  assert cve.advisory_refs == ["https://trellis.example/security/TSA-2099-003"]


def test_get_cve_of_an_unknown_id_is_none(store: store_lib.Store):
  assert store.get_cve("CVE-2021-44228") is None


def test_get_qid_returns_an_explained_qid(store: store_lib.Store):
  qid = store.get_qid(_QID_KERNEL)

  assert qid is not None
  assert qid.label == "Ubuntu security update: linux"
  assert qid.category == "Ubuntu"
  assert qid.severity == 4
  assert qid.pci_flag is True
  assert qid.diagnosis is not None and qid.diagnosis.startswith("Ubuntu has")
  assert qid.explained


def test_get_qid_of_an_unexplained_qid_has_only_its_id(
    store: store_lib.Store,
):
  assert store.get_qid(_QID_UNEXPLAINED) == models.Qid(
      qid=_QID_UNEXPLAINED,
      label="",
      category=None,
      severity=None,
      pci_flag=None,
      diagnosis=None,
      explained=False,
  )


def test_get_qid_of_an_unknown_id_is_none(store: store_lib.Store):
  assert store.get_qid("1") is None


def test_qids_for_cve_returns_the_qid_that_bundles_it(store: store_lib.Store):
  assert store.qids_for_cve(_CVE_PROXY) == [_QID_PROXY]


def test_qids_for_cve_lists_every_bundling_qid_in_text_order(
    tmp_path: pathlib.Path,
):
  path = _empty_artifact(tmp_path / "index.sqlite", _current_version())
  with contextlib.closing(sqlite3.connect(path)) as conn:
    with conn:
      conn.executemany(
          "INSERT INTO qids (qid, explained) VALUES (?, 1)", [("20",), ("100",)]
      )
      conn.execute("INSERT INTO cves (cve_id) VALUES ('CVE-2099-0001')")
      conn.executemany(
          "INSERT INTO qid_cves (qid, cve_id) VALUES (?, 'CVE-2099-0001')",
          [("20",), ("100",)],
      )

  with store_lib.Store(path) as opened:
    qids = opened.qids_for_cve("CVE-2099-0001")

  assert qids == ["100", "20"]


def test_qids_for_cve_of_an_unknown_id_is_empty(store: store_lib.Store):
  assert not store.qids_for_cve("CVE-2021-44228")


def test_cves_for_qid_orders_the_bundle_by_cve_id(store: store_lib.Store):
  cves = store.cves_for_qid(_QID_KERNEL)

  # The export lists them as 1003, 1001, 1004, 1002.
  assert [cve.cve_id for cve in cves] == [
      "CVE-2099-1001",
      "CVE-2099-1002",
      "CVE-2099-1003",
      "CVE-2099-1004",
  ]


def test_cves_for_qid_without_cves_is_empty(store: store_lib.Store):
  assert not store.cves_for_qid(_QID_OPENSSH)
  assert not store.cves_for_qid("1")


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def test_get_chunks_keeps_the_order_of_the_ids_given(store: store_lib.Store):
  chunks = store.get_chunks([14, 1, 11])

  assert [chunk.id for chunk in chunks] == [14, 1, 11]


def test_get_chunks_skips_ids_that_have_no_chunk(store: store_lib.Store):
  chunks = store.get_chunks([9999, 2, 0])

  assert [chunk.id for chunk in chunks] == [2]


def test_get_chunks_of_no_ids_is_empty(store: store_lib.Store):
  assert not store.get_chunks([])


def test_get_chunks_returns_the_chunk_with_its_documents_title(
    store: store_lib.Store,
):
  (chunk,) = store.get_chunks([_CHUNK_PROXY_CVE])

  assert chunk.doc_type == "cve"
  assert chunk.doc_id == _CVE_PROXY
  assert chunk.kind == "text"
  assert chunk.title == "Trellis Proxy Forwarded Header Removal Vulnerability"
  assert chunk.text.startswith("Trellis proxy before 2.8.4")


def test_chunks_for_doc_returns_a_documents_chunks_in_order(
    store: store_lib.Store,
):
  chunks = store.chunks_for_doc("cve", _CVE_LONG_KERNEL)

  assert [chunk.id for chunk in chunks] == [7, 8, 9, 10]
  assert [chunk.kind for chunk in chunks] == ["text", "trace", "trace", "text"]


def test_chunks_for_doc_tells_a_qid_from_a_cve(store: store_lib.Store):
  assert not store.chunks_for_doc("qid", _CVE_LONG_KERNEL)
  assert not store.chunks_for_doc("cve", _QID_KERNEL)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search_returns_the_best_match_first(store: store_lib.Store):
  results = store.keyword_search('"openssh" OR "proxy"', 10)

  scores = [score for _, score in results]
  assert results[0][0] == _CHUNK_OPENSSH_DIAGNOSIS
  assert scores == sorted(scores, reverse=True)
  assert all(score > 0 for score in scores)


def test_keyword_search_weights_a_title_hit_above_a_body_hit(
    store: store_lib.Store,
):
  # "listing" is in the title of the quartzfs CVE and, through its stem, in
  # the body of the kernel diagnosis ("lists the installed packages").
  by_default = store.keyword_search('"listing"', 10)
  body_only = store.keyword_search('"listing"', 10, title_weight=0.0)

  assert by_default[0][0] == _CHUNK_QUARTZFS_CVE
  assert body_only[0][0] == _CHUNK_KERNEL_DIAGNOSIS


def test_keyword_search_returns_at_most_limit_results(store: store_lib.Store):
  assert len(store.keyword_search('"openssh" OR "proxy"', 10)) == 4
  assert len(store.keyword_search('"openssh" OR "proxy"', 2)) == 2


def test_keyword_search_finds_a_symbol_in_a_trace_chunk(
    store: store_lib.Store,
):
  ((chunk_id, _),) = store.keyword_search('"fxnet_reset_task"', 10)

  (chunk,) = store.get_chunks([chunk_id])
  assert chunk.kind == "trace"


def test_keyword_search_without_a_match_is_empty(store: store_lib.Store):
  assert not store.keyword_search('"log4shell"', 10)


def test_keyword_search_in_docs_returns_only_their_chunks(
    store: store_lib.Store,
):
  docs: set[tuple[models.DocType, str]] = {
      ("cve", _CVE_PROXY),
      ("qid", _QID_OPENSSH),
  }

  results = store.keyword_search('"openssh" OR "proxy"', 10, docs=docs)

  assert [chunk_id for chunk_id, _ in results] == [
      _CHUNK_OPENSSH_DIAGNOSIS,
      _CHUNK_PROXY_CVE,
  ]


def test_keyword_search_in_no_docs_is_empty(store: store_lib.Store):
  assert not store.keyword_search('"openssh" OR "proxy"', 10, docs=[])


@pytest.mark.parametrize("match_query", ['"unterminated', "AND", "", "nope:x"])
def test_keyword_search_with_a_malformed_query_raises_store_error(
    store: store_lib.Store, match_query: str
):
  with pytest.raises(store_lib.StoreError, match="rejected the query"):
    store.keyword_search(match_query, 10)


@pytest.mark.parametrize("limit", [0, -1])
def test_keyword_search_with_a_limit_that_is_not_positive_is_rejected(
    store: store_lib.Store, limit: int
):
  with pytest.raises(ValueError, match="limit"):
    store.keyword_search('"openssh"', limit)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_load_embeddings_returns_unit_vectors_in_chunk_id_order(
    store: store_lib.Store,
):
  chunk_ids, matrix = store.load_embeddings()

  assert chunk_ids == sorted(chunk_ids)
  assert matrix.shape == (len(chunk_ids), 256)
  assert matrix.dtype == np.float32
  np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)


def test_load_embeddings_of_an_empty_index_keeps_the_dimension(
    tmp_path: pathlib.Path,
):
  meta = _current_version() | {schema.META_EMBEDDING_DIM: "384"}
  path = _empty_artifact(tmp_path / "index.sqlite", meta)

  with store_lib.Store(path) as opened:
    chunk_ids, matrix = opened.load_embeddings()

  assert not chunk_ids
  assert matrix.shape == (0, 384)


# ---------------------------------------------------------------------------
# Hosts
# ---------------------------------------------------------------------------


def test_hosts_for_qids_orders_hosts_by_name_and_then_id(
    store: store_lib.Store,
):
  hosts = store.hosts_for_qids([_QID_OPENSSH])

  # The two hosts named fx-k8s-worker-a are told apart by id, and
  # fx-nat-gateway sorts after every fx-k8s-worker.
  assert [host.id for host, _ in hosts] == [
      _BASTION,
      _WORKER_A_FIRST,
      _WORKER_A_SECOND,
      _WORKER_B1,
      _WORKER_B2,
      _WORKER_TERMINATED,
      _GATEWAY,
      _QUEUE,
      _WEB,
  ]


def test_hosts_for_qids_lists_the_sorted_qids_found_on_each_host(
    store: store_lib.Store,
):
  hosts = store.hosts_for_qids([_QID_PROXY, "1", _QID_OPENSSH])

  found = {host.id: qids for host, qids in hosts}
  assert found[_BASTION] == [_QID_OPENSSH]
  assert found[_GATEWAY] == [_QID_OPENSSH, _QID_PROXY]


def test_hosts_for_qids_reports_a_qid_detected_twice_once(
    store: store_lib.Store,
):
  detections = store.host_detections(_WORKER_B2)
  assert [d.qid for d in detections].count(_QID_PROXY) == 2

  hosts = store.hosts_for_qids([_QID_PROXY])

  assert [qids for host, qids in hosts if host.id == _WORKER_B2] == [
      [_QID_PROXY]
  ]


def test_hosts_for_qids_includes_hosts_that_are_not_running(
    store: store_lib.Store,
):
  hosts = store.hosts_for_qids([_QID_KERNEL])

  states = {host.id: host.state for host, _ in hosts}
  assert states[_WORKER_TERMINATED] == "TERMINATED"


def test_hosts_for_qids_resolves_an_unexplained_qid(store: store_lib.Store):
  hosts = store.hosts_for_qids([_QID_UNEXPLAINED])

  assert len(hosts) == 10


def test_hosts_for_qids_of_no_qids_or_unknown_qids_is_empty(
    store: store_lib.Store,
):
  assert not store.hosts_for_qids([])
  assert not store.hosts_for_qids(["1", "2"])


def test_get_host_returns_the_host(store: store_lib.Store):
  assert store.get_host(_WORKER_B1) == models.Host(
      id=_WORKER_B1,
      name="fx-k8s-worker-b1",
      os="Ubuntu Linux 20.04.6",
      criticality=3,
      state="RUNNING",
      internet_facing=False,
      public_ip=None,
      private_ip="10.20.5.10",
      region="us-west-2",
      vpc_id="vpc-00000000000000001",
      security_group="fx-k8s-workers-b-sg",
      cluster="fx-build",
      role="node",
      is_docker_host=True,
      last_scan="2099-03-01T06:00:00Z",
  )


def test_get_host_of_an_unknown_id_is_none(store: store_lib.Store):
  assert store.get_host("1") is None


def test_host_ports_are_ordered_by_port_and_then_protocol(
    store: store_lib.Store,
):
  ports = store.host_ports(_WORKER_B1)

  # The export lists them as 8080, 22, 111/UDP, 111/TCP, 80.
  assert ports == [
      models.OpenPort(port=22, protocol="TCP", service="ssh"),
      models.OpenPort(port=80, protocol="TCP", service="http"),
      models.OpenPort(port=111, protocol="TCP", service="rpc"),
      models.OpenPort(port=111, protocol="UDP", service="rpc_udp"),
      models.OpenPort(port=8080, protocol="TCP", service="proxy_http"),
  ]


def test_host_ports_keeps_a_port_with_no_recognised_service(
    store: store_lib.Store,
):
  ports = store.host_ports(_QUEUE)

  assert models.OpenPort(port=5672, protocol="TCP", service=None) in ports


def test_host_ports_of_a_host_without_ports_is_empty(store: store_lib.Store):
  assert not store.host_ports(_IDLE)
  assert not store.host_ports("1")


def test_host_detections_say_which_qids_are_explained(store: store_lib.Store):
  detections = store.host_detections(_BASTION)

  # Ordered by QID; the scanner lists the first unexplained QID first.
  assert detections == [
      models.Detection(
          id="51000000002",
          host_id=_BASTION,
          qid=_QID_OPENSSH,
          first_found="2099-02-01T06:00:00Z",
          last_found="2099-03-01T06:00:00Z",
          explained=True,
      ),
      models.Detection(
          id="51000000001",
          host_id=_BASTION,
          qid="710901",
          first_found="2099-02-01T06:00:00Z",
          last_found="2099-03-01T06:00:00Z",
          explained=False,
      ),
      models.Detection(
          id="51000000003",
          host_id=_BASTION,
          qid="710902",
          first_found="2099-02-01T06:00:00Z",
          last_found="2099-03-01T06:00:00Z",
          explained=False,
      ),
  ]


def test_host_detections_of_a_host_without_detections_is_empty(
    store: store_lib.Store,
):
  assert not store.host_detections(_IDLE)
  assert not store.host_detections("1")


def test_host_names_are_distinct_and_sorted(store: store_lib.Store):
  names = store.host_names()

  # Eleven hosts, two of which share the name fx-k8s-worker-a.
  assert names == [
      "fx-bastion",
      "fx-build-agent",
      "fx-idle-runner",
      "fx-k8s-worker-a",
      "fx-k8s-worker-b1",
      "fx-k8s-worker-b2",
      "fx-k8s-worker-b3",
      "fx-nat-gateway",
      "fx-queue-01",
      "fx-web-01",
  ]
