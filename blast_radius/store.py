"""Read access to the index artifact.

The artifact is the single SQLite file that ``blast_radius.ingest`` builds;
its layout is fixed in ``blast_radius.schema``. ``Store`` is the only way
the serving side reads it: every query the pipeline needs is a method here,
every method returns the models from ``blast_radius.models``, and no SQL is
written anywhere else on the read path. ``create`` is the one entry point
for the write path, which ingest and tests use to start an empty artifact.

Lists of ids reach SQL as one JSON array bound to a single parameter and
unpacked with ``json_each``. The statements therefore stay constant strings
with no placeholders spliced in, and a QID that bundles hundreds of CVEs
cannot run into SQLite's limit on bound parameters.
"""

from collections.abc import Collection, Sequence
import itertools
import json
import pathlib
import sqlite3
import threading
import types

import numpy as np

from blast_radius import embeddings
from blast_radius import models
from blast_radius import schema

_SELECT_META = "SELECT key, value FROM meta"

_SELECT_META_VALUE = "SELECT value FROM meta WHERE key = ?"

_SELECT_STATS = """
SELECT
  (SELECT count(*) FROM hosts) AS hosts,
  (SELECT count(DISTINCT host_id) FROM detections) AS hosts_with_detections,
  (SELECT count(*) FROM qids) AS total_qids,
  (SELECT count(*) FROM qids WHERE explained) AS explained_qids,
  (SELECT count(*) FROM detections) AS total_detections,
  (SELECT count(*)
   FROM detections AS d
   JOIN qids AS q ON q.qid = d.qid
   WHERE q.explained) AS explained_detections,
  (SELECT count(*) FROM cves) AS cves,
  (SELECT count(*) FROM chunks) AS chunks
"""

_SELECT_CVE = "SELECT * FROM cves WHERE cve_id = ?"

_SELECT_QID = "SELECT * FROM qids WHERE qid = ?"

_SELECT_QIDS_FOR_CVE = """
SELECT qid FROM qid_cves WHERE cve_id = ? ORDER BY qid
"""

_SELECT_CVES_FOR_QID = """
SELECT c.*
FROM qid_cves AS qc
JOIN cves AS c ON c.cve_id = qc.cve_id
WHERE qc.qid = ?
ORDER BY c.cve_id
"""

_SELECT_CHUNKS_BY_ID = """
SELECT * FROM chunks WHERE id IN (SELECT value FROM json_each(?))
"""

_SELECT_CHUNKS_FOR_DOC = """
SELECT * FROM chunks WHERE doc_type = ? AND doc_id = ? ORDER BY id
"""

# bm25() is smaller for a better match, so ascending order is best first.
# The rowid breaks ties, which keeps the result stable between runs. The
# alias avoids the name "rank", which FTS5 tables have as a hidden column.
_KEYWORD_SEARCH = """
SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts, ?, ?) AS weighted_bm25
FROM chunks_fts
WHERE chunks_fts MATCH ?
ORDER BY weighted_bm25, chunk_id
LIMIT ?
"""

_KEYWORD_SEARCH_IN_DOCS = """
SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts, ?, ?) AS weighted_bm25
FROM chunks_fts
JOIN chunks ON chunks.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?
  AND (chunks.doc_type, chunks.doc_id) IN (
    SELECT json_extract(value, '$[0]'), json_extract(value, '$[1]')
    FROM json_each(?))
ORDER BY weighted_bm25, chunk_id
LIMIT ?
"""

_SELECT_EMBEDDINGS = """
SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id
"""

# The blast-radius join. INDEXED BY is SQLite's way of asserting a query
# plan: if the statement could not be driven from the QID index it would
# fail to prepare, instead of quietly scanning every detection. DISTINCT is
# needed because the scanner can report one QID several times on one host.
_SELECT_HOSTS_FOR_QIDS = """
SELECT DISTINCT h.*, d.qid AS matched_qid
FROM detections AS d INDEXED BY detections_by_qid
JOIN hosts AS h ON h.id = d.host_id
WHERE d.qid IN (SELECT value FROM json_each(?))
ORDER BY h.name, h.id, d.qid
"""

_SELECT_HOST = "SELECT * FROM hosts WHERE id = ?"

_SELECT_HOST_PORTS = """
SELECT port, protocol, service
FROM open_ports
WHERE host_id = ?
ORDER BY port, protocol
"""

_SELECT_HOST_DETECTIONS = """
SELECT d.id, d.host_id, d.qid, d.first_found, d.last_found, q.explained
FROM detections AS d
JOIN qids AS q ON q.qid = d.qid
WHERE d.host_id = ?
ORDER BY d.qid, d.id
"""

_SELECT_HOST_NAMES = "SELECT DISTINCT name FROM hosts ORDER BY name"


class StoreError(Exception):
  """Base class for failures to read the index artifact."""


class ArtifactNotFoundError(StoreError):
  """There is no artifact at the given path; ingest has not run."""


class SchemaVersionError(StoreError):
  """The artifact was built for a different version of the schema."""


def create(path: pathlib.Path) -> sqlite3.Connection:
  """Creates an empty artifact at ``path`` and returns a connection to it.

  The connection is read-write and enforces the schema's foreign keys, so
  a build that references a missing host, QID, CVE or chunk fails instead
  of producing an inconsistent index. The caller owns the connection and
  closes it.

  Args:
    path: Where to create the database. Its directory must exist.

  Raises:
    FileExistsError: If ``path`` exists. An artifact is replaced by moving
      a complete new one over it, never by writing into it.
  """
  # Creating the file exclusively makes the refusal atomic. SQLite treats
  # the empty file as a new database.
  path.touch(exist_ok=False)
  conn = sqlite3.connect(path)
  conn.execute("PRAGMA foreign_keys = ON")
  conn.executescript(schema.SCHEMA)
  return conn


# Rows are read by column name, never by position, so the ``SELECT *``
# statements above do not depend on the order of the schema's columns.


def _host(row: sqlite3.Row) -> models.Host:
  """Returns the host held in a row of ``hosts``."""
  return models.Host(
      id=row["id"],
      name=row["name"],
      os=row["os"],
      criticality=row["criticality"],
      state=row["state"],
      internet_facing=bool(row["internet_facing"]),
      public_ip=row["public_ip"],
      private_ip=row["private_ip"],
      region=row["region"],
      vpc_id=row["vpc_id"],
      security_group=row["security_group"],
      cluster=row["cluster"],
      role=row["role"],
      is_docker_host=bool(row["is_docker_host"]),
      last_scan=row["last_scan"],
  )


def _qid(row: sqlite3.Row) -> models.Qid:
  """Returns the QID held in a row of ``qids``."""
  pci_flag = row["pci_flag"]
  return models.Qid(
      qid=row["qid"],
      label=row["label"],
      category=row["category"],
      severity=row["severity"],
      pci_flag=None if pci_flag is None else bool(pci_flag),
      diagnosis=row["diagnosis"],
      explained=bool(row["explained"]),
  )


def _cve(row: sqlite3.Row) -> models.Cve:
  """Returns the CVE held in a row of ``cves``."""
  return models.Cve(
      cve_id=row["cve_id"],
      title=row["title"],
      description=row["description"],
      cvss=row["cvss"],
      epss=row["epss"],
      epss_percentile=row["epss_percentile"],
      attack_vector=row["attack_vector"],
      known_exploited=bool(row["known_exploited"]),
      kev_required_action=row["kev_required_action"],
      vendor_fix=row["vendor_fix"],
      cogent_risk_score=row["cogent_risk_score"],
      published=row["published"],
      cwes=json.loads(row["cwes"]),
      patch_refs=json.loads(row["patch_refs"]),
      advisory_refs=json.loads(row["advisory_refs"]),
  )


def _chunk(row: sqlite3.Row) -> models.Chunk:
  """Returns the chunk held in a row of ``chunks``."""
  return models.Chunk(
      id=row["id"],
      doc_type=row["doc_type"],
      doc_id=row["doc_id"],
      kind=row["kind"],
      title=row["title"],
      text=row["text"],
  )


class Store:
  """Read-only, thread-safe access to a built artifact.

  The API server calls the store from a thread pool. Each thread gets its
  own SQLite connection the first time it runs a query, so queries never
  wait on one another and no cursor state is shared. Connections are
  opened read-only: nothing that holds a ``Store`` can change the index.

  Because threads connect lazily, an artifact must not be replaced while a
  store is open on it: a thread that connects afterwards would read the new
  file while the others still read the old one. The service picks up a new
  artifact when it restarts.

  A store is a context manager; leaving the block closes it.
  """

  def __init__(self, path: pathlib.Path) -> None:
    """Opens the artifact and checks that this build can read it.

    Args:
      path: The artifact written by ``blast_radius.ingest``.

    Raises:
      ArtifactNotFoundError: If there is no file at ``path``.
      SchemaVersionError: If the artifact records a schema version other
        than ``schema.SCHEMA_VERSION``.
      StoreError: If the file is not an index artifact at all.
    """
    if not path.is_file():
      raise ArtifactNotFoundError(
          f"no index artifact at {path}; build one with `blast-radius ingest`"
      )
    self._path = path
    # as_uri() percent-encodes the path, so a "?" or "#" in a directory
    # name cannot be mistaken for the start of the URI's query.
    self._uri = f"{path.resolve().as_uri()}?mode=ro"
    self._local = threading.local()
    self._lock = threading.Lock()
    self._connections: list[sqlite3.Connection] = []
    self._closed = False
    try:
      self._check_schema_version()
    except StoreError:
      self.close()
      raise

  def __enter__(self) -> "Store":
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> None:
    self.close()

  def close(self) -> None:
    """Closes every thread's connection. Closing twice is harmless."""
    with self._lock:
      self._closed = True
      for conn in self._connections:
        conn.close()
      self._connections.clear()

  def _check_open(self) -> None:
    """Refuses to go on once the store has been closed.

    Raises:
      StoreError: If ``close`` has been called.
    """
    if self._closed:
      raise StoreError(f"the store on {self._path} is closed")

  def _conn(self) -> sqlite3.Connection:
    """Returns the calling thread's connection, opening it on first use.

    Raises:
      StoreError: If the store has been closed.
    """
    conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
    if conn is None:
      # Opening and registering under the lock means close() cannot miss a
      # connection that another thread is just opening.
      with self._lock:
        self._check_open()
        # Only the owning thread ever queries a connection. The same-thread
        # check is lifted so that close() can close all of them from the
        # one thread that calls it.
        conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        self._connections.append(conn)
      conn.row_factory = sqlite3.Row
      self._local.conn = conn
    self._check_open()
    return conn

  def _check_schema_version(self) -> None:
    """Checks the version the artifact records against this build's.

    Raises:
      SchemaVersionError: If the versions differ.
      StoreError: If the file cannot be read as an artifact.
    """
    try:
      row = (
          self._conn()
          .execute(_SELECT_META_VALUE, (schema.META_SCHEMA_VERSION,))
          .fetchone()
      )
    except sqlite3.DatabaseError as err:
      raise StoreError(f"{self._path} is not an index artifact: {err}") from err
    found = row["value"] if row else None
    if found != str(schema.SCHEMA_VERSION):
      raise SchemaVersionError(
          f"{self._path} has schema version {found}, but this build reads"
          f" version {schema.SCHEMA_VERSION}; rebuild it with"
          " `blast-radius ingest`"
      )

  def meta(self) -> dict[str, str]:
    """Returns what the artifact was built from, keyed by ``schema.META_*``."""
    rows = self._conn().execute(_SELECT_META).fetchall()
    return {row["key"]: row["value"] for row in rows}

  def stats(self) -> models.CorpusStats:
    """Returns how much of what the scanner detected the corpus explains."""
    row = self._conn().execute(_SELECT_STATS).fetchone()
    return models.CorpusStats(
        hosts=row["hosts"],
        hosts_with_detections=row["hosts_with_detections"],
        total_qids=row["total_qids"],
        explained_qids=row["explained_qids"],
        total_detections=row["total_detections"],
        explained_detections=row["explained_detections"],
        cves=row["cves"],
        chunks=row["chunks"],
    )

  def get_cve(self, cve_id: str) -> models.Cve | None:
    """Returns the CVE with this id, or None if the corpus lacks it.

    Args:
      cve_id: A CVE identifier, upper-case as in the exports.
    """
    row = self._conn().execute(_SELECT_CVE, (cve_id,)).fetchone()
    return _cve(row) if row else None

  def get_qid(self, qid: str) -> models.Qid | None:
    """Returns the QID with this id, or None if no host has it.

    A QID that was detected but has no write-up is returned with
    ``explained`` false and its other fields empty.

    Args:
      qid: A Qualys check id.
    """
    row = self._conn().execute(_SELECT_QID, (qid,)).fetchone()
    return _qid(row) if row else None

  def qids_for_cve(self, cve_id: str) -> list[str]:
    """Returns the QIDs that the scanner maps to a CVE, sorted.

    Args:
      cve_id: A CVE identifier.
    """
    rows = self._conn().execute(_SELECT_QIDS_FOR_CVE, (cve_id,)).fetchall()
    return [row["qid"] for row in rows]

  def cves_for_qid(self, qid: str) -> list[models.Cve]:
    """Returns the CVEs that a QID bundles, ordered by CVE id.

    Args:
      qid: A Qualys check id.
    """
    rows = self._conn().execute(_SELECT_CVES_FOR_QID, (qid,)).fetchall()
    return [_cve(row) for row in rows]

  def get_chunks(self, chunk_ids: Sequence[int]) -> list[models.Chunk]:
    """Returns the chunks with the given ids, in the order they were given.

    Args:
      chunk_ids: Chunk ids, typically a ranked search result. Ids with no
        chunk are skipped.
    """
    rows = (
        self._conn()
        .execute(_SELECT_CHUNKS_BY_ID, (json.dumps(list(chunk_ids)),))
        .fetchall()
    )
    by_id = {row["id"]: _chunk(row) for row in rows}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

  def chunks_for_doc(
      self, doc_type: models.DocType, doc_id: str
  ) -> list[models.Chunk]:
    """Returns every chunk of one document, in document order.

    Args:
      doc_type: Whether the document is a CVE description or a QID
        diagnosis.
      doc_id: The CVE id or QID.
    """
    rows = (
        self._conn()
        .execute(_SELECT_CHUNKS_FOR_DOC, (doc_type, doc_id))
        .fetchall()
    )
    return [_chunk(row) for row in rows]

  def keyword_search(
      self,
      match_query: str,
      limit: int,
      *,
      docs: Collection[tuple[models.DocType, str]] | None = None,
      title_weight: float = 3.0,
      body_weight: float = 1.0,
  ) -> list[tuple[int, float]]:
    """Runs a BM25-ranked full-text search over the chunks.

    Args:
      match_query: An FTS5 ``MATCH`` expression. It must already be
        sanitised: FTS5 has its own query syntax, so text a user typed is
        never passed here as written.
      limit: The most results to return.
      docs: When given, only chunks of these ``(doc_type, doc_id)``
        documents are searched. An empty collection matches nothing.
      title_weight: BM25 weight of a hit in the document title.
      body_weight: BM25 weight of a hit in the chunk text.

    Returns:
      ``(chunk_id, score)`` pairs, best first. The score is the negated
      ``bm25()`` value, so that higher is better.

    Raises:
      ValueError: If ``limit`` is not positive.
      StoreError: If SQLite rejects ``match_query``.
    """
    if limit <= 0:
      raise ValueError(f"limit must be positive, got {limit}")
    params: tuple[float | str | int, ...]
    if docs is None:
      sql = _KEYWORD_SEARCH
      params = (title_weight, body_weight, match_query, limit)
    else:
      sql = _KEYWORD_SEARCH_IN_DOCS
      wanted = json.dumps(list(docs))
      params = (title_weight, body_weight, match_query, wanted, limit)
    try:
      rows = self._conn().execute(sql, params).fetchall()
    except sqlite3.OperationalError as err:
      raise StoreError(
          f"keyword search rejected the query {match_query!r}: {err}"
      ) from err
    return [(row["chunk_id"], -row["weighted_bm25"]) for row in rows]

  def load_embeddings(self) -> tuple[list[int], embeddings.Matrix]:
    """Returns every stored vector, for an in-memory similarity search.

    Returns:
      The chunk ids in ascending order, and a float32 matrix of shape
      ``(len(ids), dim)`` whose row ``i`` is the vector of chunk
      ``ids[i]``.
    """
    rows = self._conn().execute(_SELECT_EMBEDDINGS).fetchall()
    dim = int(self.meta()[schema.META_EMBEDDING_DIM])
    chunk_ids = [row["chunk_id"] for row in rows]
    # Vectors are stored little-endian; astype() yields native float32.
    flat = np.frombuffer(b"".join(row["vector"] for row in rows), dtype="<f4")
    return chunk_ids, flat.reshape(len(chunk_ids), dim).astype(np.float32)

  def hosts_for_qids(
      self, qids: Collection[str]
  ) -> list[tuple[models.Host, list[str]]]:
    """Returns every host on which any of the given QIDs was detected.

    This is the blast-radius join: one indexed query over ``detections``,
    whatever the number of hosts.

    Args:
      qids: Qualys check ids.

    Returns:
      ``(host, qids_found)`` pairs ordered by host name and then host id,
      where ``qids_found`` is the sorted subset of ``qids`` detected on
      that host.
    """
    rows = (
        self._conn()
        .execute(_SELECT_HOSTS_FOR_QIDS, (json.dumps(list(qids)),))
        .fetchall()
    )
    hosts = []
    for _, group in itertools.groupby(rows, key=lambda row: row["id"]):
      host_rows = list(group)
      found = [row["matched_qid"] for row in host_rows]
      hosts.append((_host(host_rows[0]), found))
    return hosts

  def get_host(self, host_id: str) -> models.Host | None:
    """Returns the host with this id, or None if there is none.

    Args:
      host_id: The scanner's asset id.
    """
    row = self._conn().execute(_SELECT_HOST, (host_id,)).fetchone()
    return _host(row) if row else None

  def host_ports(self, host_id: str) -> list[models.OpenPort]:
    """Returns a host's open ports, ordered by port and then protocol.

    Args:
      host_id: The scanner's asset id.
    """
    rows = self._conn().execute(_SELECT_HOST_PORTS, (host_id,)).fetchall()
    return [
        models.OpenPort(
            port=row["port"], protocol=row["protocol"], service=row["service"]
        )
        for row in rows
    ]

  def host_detections(self, host_id: str) -> list[models.Detection]:
    """Returns everything the scanner found on a host.

    Each detection's ``explained`` says whether the corpus has a write-up
    for its QID, so a caller can show explained and unexplained findings
    apart. The order is by QID and then detection id.

    Args:
      host_id: The scanner's asset id.
    """
    rows = self._conn().execute(_SELECT_HOST_DETECTIONS, (host_id,)).fetchall()
    return [
        models.Detection(
            id=row["id"],
            host_id=row["host_id"],
            qid=row["qid"],
            first_found=row["first_found"],
            last_found=row["last_found"],
            explained=bool(row["explained"]),
        )
        for row in rows
    ]

  def host_names(self) -> list[str]:
    """Returns the distinct host names in the inventory, sorted."""
    rows = self._conn().execute(_SELECT_HOST_NAMES).fetchall()
    return [row["name"] for row in rows]
