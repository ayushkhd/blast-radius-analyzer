"""The SQL schema of the index artifact.

The artifact is a single SQLite file, so that a rebuilt index can replace
the old one with one atomic rename. It holds four kinds of data:

* the inventory (``hosts``, ``open_ports``) and what the scanner found on
  it (``detections``);
* the write-ups that explain those findings (``qids``, ``cves`` and the
  many-to-many ``qid_cves``);
* the retrieval index over the write-ups: ``chunks``, the FTS5 table
  ``chunks_fts`` whose ``rowid`` is ``chunks.id``, and ``embeddings``, one
  float32 vector per chunk that is embedded;
* ``meta``, which records what the artifact was built from.

A QID that appears in ``detections`` but has no write-up still gets a row in
``qids``, with ``explained = 0``, so that the service can count what it
cannot explain. List-valued CVE columns (``cwes``, ``patch_refs``,
``advisory_refs``) are JSON arrays.
"""

# Bumped whenever a change to SCHEMA makes older artifacts unreadable.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE hosts (
  id              TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  os              TEXT,
  criticality     INTEGER NOT NULL,
  state           TEXT NOT NULL,
  internet_facing INTEGER NOT NULL,
  public_ip       TEXT,
  private_ip      TEXT,
  region          TEXT,
  vpc_id          TEXT,
  security_group  TEXT,
  cluster         TEXT,
  role            TEXT,
  is_docker_host  INTEGER NOT NULL,
  last_scan       TEXT
);

CREATE TABLE open_ports (
  host_id  TEXT NOT NULL REFERENCES hosts(id),
  port     INTEGER NOT NULL,
  protocol TEXT NOT NULL,
  service  TEXT
);
CREATE INDEX open_ports_by_host ON open_ports(host_id);
CREATE INDEX open_ports_by_service ON open_ports(service);

CREATE TABLE qids (
  qid       TEXT PRIMARY KEY,
  label     TEXT NOT NULL DEFAULT '',
  category  TEXT,
  severity  INTEGER,
  pci_flag  INTEGER,
  diagnosis TEXT,
  explained INTEGER NOT NULL
);

CREATE TABLE detections (
  id          TEXT PRIMARY KEY,
  host_id     TEXT NOT NULL REFERENCES hosts(id),
  qid         TEXT NOT NULL REFERENCES qids(qid),
  first_found TEXT,
  last_found  TEXT
);
CREATE INDEX detections_by_qid ON detections(qid);
CREATE INDEX detections_by_host ON detections(host_id);

CREATE TABLE cves (
  cve_id              TEXT PRIMARY KEY,
  title               TEXT NOT NULL DEFAULT '',
  description         TEXT NOT NULL DEFAULT '',
  cvss                REAL,
  epss                REAL,
  epss_percentile     REAL,
  attack_vector       TEXT,
  known_exploited     INTEGER NOT NULL DEFAULT 0,
  kev_required_action TEXT,
  vendor_fix          TEXT,
  cogent_risk_score   REAL,
  published           TEXT,
  cwes                TEXT NOT NULL DEFAULT '[]',
  patch_refs          TEXT NOT NULL DEFAULT '[]',
  advisory_refs       TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE qid_cves (
  qid    TEXT NOT NULL REFERENCES qids(qid),
  cve_id TEXT NOT NULL REFERENCES cves(cve_id),
  PRIMARY KEY (qid, cve_id)
);
CREATE INDEX qid_cves_by_cve ON qid_cves(cve_id);

CREATE TABLE chunks (
  id       INTEGER PRIMARY KEY,
  doc_type TEXT NOT NULL CHECK (doc_type IN ('cve', 'qid')),
  doc_id   TEXT NOT NULL,
  kind     TEXT NOT NULL CHECK (kind IN ('text', 'trace')),
  title    TEXT NOT NULL DEFAULT '',
  text     TEXT NOT NULL
);
CREATE INDEX chunks_by_doc ON chunks(doc_type, doc_id);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  title,
  body,
  tokenize = 'porter unicode61'
);

CREATE TABLE embeddings (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
  vector   BLOB NOT NULL
);

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

# Keys written to ``meta`` by ingest.
META_SCHEMA_VERSION = "schema_version"
META_BUILT_AT = "built_at"
META_ASSETS_SHA256 = "assets_sha256"
META_VULNS_SHA256 = "vulns_sha256"
META_EMBEDDING_MODEL = "embedding_model"
META_EMBEDDING_DIM = "embedding_dim"
META_CHUNK_MAX_CHARS = "chunk_max_chars"
META_CHUNK_OVERLAP_SENTENCES = "chunk_overlap_sentences"
