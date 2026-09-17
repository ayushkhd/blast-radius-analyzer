"""Data shapes shared by the pipeline steps and returned by the API.

Every step of the pipeline takes and returns the models defined here, and
``AnalyzeResponse`` is the HTTP contract the UI is written against. Models
are immutable: a step that adds information builds a new object with
``model_copy`` instead of changing one it was handed.

Identifiers (host ids, QIDs, CVE ids) are strings throughout, as they are
in the scanner exports.
"""

from typing import Any, Literal

import pydantic

DocType = Literal["cve", "qid"]
ChunkKind = Literal["text", "trace"]
MatchedBy = Literal["identifier", "search"]
AnalysisStatus = Literal["matched", "no_match"]
ReadinessStatus = Literal["ready", "not_ready"]
EvidenceKind = Literal[
    "affected_versions",
    "package_update",
    "patch_reference",
    "advisory_reference",
    "required_action",
    "vendor_fix",
]

STATE_RUNNING = "RUNNING"

# Long enough for a pasted advisory, short enough that one request cannot
# tie up the reranker.
MAX_QUERY_CHARS = 8000


class Model(pydantic.BaseModel):
  """Base class: immutable, and strict about unknown fields."""

  model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Inventory and corpus
# ---------------------------------------------------------------------------


class Host(Model):
  """One scanned host.

  Attributes:
    id: The scanner's asset id.
    name: Display name, usually the EC2 ``Name`` tag or a private DNS name.
    os: Operating system as the scanner reported it.
    criticality: Business criticality on the scanner's 1-5 scale.
    state: Instance state, e.g. ``RUNNING`` or ``TERMINATED``.
    internet_facing: True when the host is tagged internet-facing or has a
      public IP address.
    public_ip: Public IPv4 address, if any.
    private_ip: Private IPv4 address, if any.
    region: Cloud region.
    vpc_id: VPC the host sits in.
    security_group: Name of the host's security group; the main grouping key.
    cluster: Kubernetes cluster the host belongs to, from its tags.
    role: The host's ``role`` tag.
    is_docker_host: Whether the scanner found a container runtime.
    last_scan: ISO timestamp of the last vulnerability scan.
  """

  id: str
  name: str
  os: str | None = None
  criticality: int
  state: str
  internet_facing: bool
  public_ip: str | None = None
  private_ip: str | None = None
  region: str | None = None
  vpc_id: str | None = None
  security_group: str | None = None
  cluster: str | None = None
  role: str | None = None
  is_docker_host: bool = False
  last_scan: str | None = None

  @property
  def is_running(self) -> bool:
    """Returns whether the instance is up, and so worth prioritising."""
    return self.state == STATE_RUNNING


class OpenPort(Model):
  """A listening port the scanner found on a host."""

  port: int
  protocol: str
  service: str | None = None


class Detection(Model):
  """One scanner finding on one host.

  Attributes:
    id: The scanner's ``hostInstanceVulnId``.
    host_id: The host it was found on.
    qid: The Qualys check that fired.
    first_found: ISO timestamp of the first detection.
    last_found: ISO timestamp of the most recent detection.
    explained: Whether the corpus has a write-up for ``qid``.
  """

  id: str
  host_id: str
  qid: str
  first_found: str | None = None
  last_found: str | None = None
  explained: bool = False


class Qid(Model):
  """A Qualys check.

  Attributes:
    qid: The check's id.
    label: Short human-readable name derived at ingest.
    category: Qualys category, e.g. ``Ubuntu`` or ``CGI``.
    severity: Qualys severity, 1-5.
    pci_flag: Whether the finding fails PCI compliance.
    diagnosis: The check's description as plain text.
    explained: False for a QID that appears in detections but has no
      write-up anywhere in the data; every other field is then empty.
  """

  qid: str
  label: str = ""
  category: str | None = None
  severity: int | None = None
  pci_flag: bool | None = None
  diagnosis: str | None = None
  explained: bool = False


class Cve(Model):
  """A CVE with the enrichment the vulnerability export carries.

  Attributes:
    cve_id: The CVE identifier.
    title: Short title from the export.
    description: NVD description as plain text.
    cvss: CVSS base score, when the export has one.
    epss: EPSS probability of exploitation.
    epss_percentile: EPSS percentile, 0-1.
    attack_vector: ``network``, ``adjacent_network``, ``local``,
      ``physical``, or None when unknown.
    known_exploited: Whether the CVE is in CISA's KEV catalogue.
    kev_required_action: KEV's required action, when known-exploited.
    vendor_fix: The export's ``how_to_fix`` text, when it is not "N/A".
    cogent_risk_score: The export's own 0-10 risk score.
    published: ISO publication timestamp.
    cwes: CWE identifiers.
    patch_refs: URLs of NVD references tagged "Patch".
    advisory_refs: URLs of NVD references tagged as an advisory.
  """

  cve_id: str
  title: str = ""
  description: str = ""
  cvss: float | None = None
  epss: float | None = None
  epss_percentile: float | None = None
  attack_vector: str | None = None
  known_exploited: bool = False
  kev_required_action: str | None = None
  vendor_fix: str | None = None
  cogent_risk_score: float | None = None
  published: str | None = None
  cwes: list[str] = pydantic.Field(default_factory=list)
  patch_refs: list[str] = pydantic.Field(default_factory=list)
  advisory_refs: list[str] = pydantic.Field(default_factory=list)


class Chunk(Model):
  """A retrievable piece of one write-up.

  Attributes:
    id: Row id, shared by the ``chunks``, ``chunks_fts`` and ``embeddings``
      tables.
    doc_type: Whether the chunk comes from a CVE description or a QID
      diagnosis.
    doc_id: The CVE id or QID.
    kind: ``trace`` chunks hold kernel log excerpts; they are searchable by
      keyword but are never embedded or shown to the language model.
    title: The parent document's title, kept with every chunk for context.
    text: The chunk's content.
  """

  id: int
  doc_type: DocType
  doc_id: str
  kind: ChunkKind = "text"
  title: str = ""
  text: str


class CorpusStats(Model):
  """How much of what the scanner detected the corpus can explain."""

  hosts: int
  hosts_with_detections: int
  total_qids: int
  explained_qids: int
  total_detections: int
  explained_detections: int
  cves: int
  chunks: int


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class ParsedQuery(Model):
  """What step 1 extracted from the analyst's input.

  Attributes:
    raw: The input as typed.
    cve_ids: CVE identifiers found in the input, upper-cased.
    qids: QIDs found in the input (only when written as ``QID 12345``).
    product: Affected product, if the input names one.
    version: Affected version or range, if the input gives one.
    search_queries: Free-text queries for step 2, most specific first.
      Empty when the input was nothing but identifiers.
    used_llm: Whether a language model contributed to this parse.
  """

  raw: str
  cve_ids: list[str] = pydantic.Field(default_factory=list)
  qids: list[str] = pydantic.Field(default_factory=list)
  product: str | None = None
  version: str | None = None
  search_queries: list[str] = pydantic.Field(default_factory=list)
  used_llm: bool = False


class ScoredChunk(Model):
  """A chunk with the score each retrieval stage gave it.

  Ranks are 1-based. A field is None when its stage was disabled or did
  not return the chunk.
  """

  chunk: Chunk
  keyword_rank: int | None = None
  dense_rank: int | None = None
  dense_score: float | None = None
  fused_score: float | None = None
  rerank_score: float | None = None


class CveSummary(Model):
  """The fields of a CVE that ranking and display need."""

  cve_id: str
  title: str = ""
  cvss: float | None = None
  epss_percentile: float | None = None
  attack_vector: str | None = None
  known_exploited: bool = False
  cogent_risk_score: float | None = None


class QidMatch(Model):
  """A QID that answers the query, with the evidence for the match.

  Hosts attach to QIDs, so every retrieval hit is rolled up to one.

  Attributes:
    qid: The matched check.
    label: Short human-readable name of the check.
    category: Qualys category.
    severity: Qualys severity, 1-5.
    matched_by: ``identifier`` for a primary-key lookup, ``search`` for a
      retrieval hit.
    score: The best score among ``chunks`` from the last enabled stage;
      None for identifier matches, which are not ranked.
    cves: Every CVE the scanner maps to this QID.
    matched_cve_ids: The subset of ``cves`` that the query actually hit.
    chunks: The chunks that produced the match, best first.
  """

  qid: str
  label: str = ""
  category: str | None = None
  severity: int | None = None
  matched_by: MatchedBy
  score: float | None = None
  cves: list[CveSummary] = pydantic.Field(default_factory=list)
  matched_cve_ids: list[str] = pydantic.Field(default_factory=list)
  chunks: list[ScoredChunk] = pydantic.Field(default_factory=list)


class RetrievalResult(Model):
  """The outcome of step 2.

  Attributes:
    matches: Matched QIDs, identifier matches first, then by score.
    candidates: The ranked chunk list the matches were drawn from, kept for
      the trace and for the evaluation.
    unknown_identifiers: Identifiers in the query that the corpus lacks.
    abstained: True when search ran and nothing scored above the floor.
    reason: Why retrieval abstained, in a sentence.
  """

  matches: list[QidMatch] = pydantic.Field(default_factory=list)
  candidates: list[ScoredChunk] = pydantic.Field(default_factory=list)
  unknown_identifiers: list[str] = pydantic.Field(default_factory=list)
  abstained: bool = False
  reason: str | None = None


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class PriorityFactors(Model):
  """Every input to a host's priority, so the ordering can be explained.

  Attributes:
    severity: CVSS / 10 where a CVSS score exists, else Qualys severity / 5.
    epss_percentile: Highest EPSS percentile among the QID's CVEs.
    known_exploited: Whether any of the QID's CVEs is in KEV.
    threat: Weighted blend of the three fields above, 0-1.
    internet_facing: Copied from the host.
    criticality: Copied from the host.
    exposure: Multiplier derived from the two host fields above.
    driving_qid: The matched QID with the highest threat on this host.
    cogent_risk_score: Highest export risk score among the driving QID's
      CVEs, shown for comparison and not used in the calculation.
  """

  severity: float
  epss_percentile: float
  known_exploited: bool
  threat: float
  internet_facing: bool
  criticality: int
  exposure: float
  driving_qid: str
  cogent_risk_score: float | None = None


class RankedHost(Model):
  """An affected host with its priority.

  Attributes:
    host: The host.
    qids: Matched QIDs detected on this host.
    priority: ``factors.threat * factors.exposure``.
    factors: The inputs to ``priority``.
  """

  host: Host
  qids: list[str]
  priority: float
  factors: PriorityFactors


class HostGroup(Model):
  """Hosts that can be treated as one unit of work.

  Attributes:
    key: Stable grouping key, e.g. ``sg:dev-k8s-mgz-worker-sg``.
    label: What to show for the group.
    count: Number of hosts in the group.
    priority: Highest host priority in the group.
    internet_facing_count: How many of the hosts are internet-facing.
    example_hosts: A few host names, highest priority first.
    host_ids: Every host in the group, highest priority first.
  """

  key: str
  label: str
  count: int
  priority: float
  internet_facing_count: int = 0
  example_hosts: list[str] = pydantic.Field(default_factory=list)
  host_ids: list[str] = pydantic.Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fix evidence, generation and verification
# ---------------------------------------------------------------------------


class FixEvidence(Model):
  """Something the data says about fixing a matched item.

  Attributes:
    id: Citable id, ``e1``, ``e2``, ... within one response.
    kind: What sort of evidence this is.
    doc_type: Whether it comes from a CVE or a QID.
    doc_id: The CVE id or QID.
    text: The evidence as a quotable sentence.
    url: The reference URL, for ``patch_reference`` and
      ``advisory_reference``.
  """

  id: str
  kind: EvidenceKind
  doc_type: DocType
  doc_id: str
  text: str
  url: str | None = None


class ContextItem(Model):
  """One citable source handed to the language model.

  Attributes:
    id: ``c<chunk id>`` for a chunk, or a ``FixEvidence.id``.
    doc_type: Whether the source is a CVE or a QID write-up.
    doc_id: The CVE id or QID.
    title: The source document's title.
    text: The text a quote must come from.
  """

  id: str
  doc_type: DocType
  doc_id: str
  title: str = ""
  text: str


class Citation(Model):
  """A quote and the source it claims to come from.

  Attributes:
    source_id: A ``ContextItem.id``.
    quote: Text copied verbatim from that source.
    verified: Set by the verifier: whether the quote occurs in the source.
  """

  source_id: str
  quote: str
  verified: bool | None = None


class Claim(Model):
  """One statement in the brief.

  Attributes:
    text: The statement.
    citations: Its supporting quotes.
    verified: Set by the verifier: True when the claim has at least one
      citation, every citation verified, and every identifier in ``text``
      occurs in the context pack.
    problems: Why the claim failed verification, one entry per reason.
  """

  text: str
  citations: list[Citation] = pydantic.Field(default_factory=list)
  verified: bool | None = None
  problems: list[str] = pydantic.Field(default_factory=list)


class Answer(Model):
  """The language model's brief."""

  summary: str
  claims: list[Claim] = pydantic.Field(default_factory=list)
  caveats: list[str] = pydantic.Field(default_factory=list)


class Verification(Model):
  """The verifier's tally for one answer.

  Attributes:
    total_claims: Claims in the answer.
    verified_claims: Claims that passed every check.
    unknown_identifiers: Identifiers in the prose that occur nowhere in the
      context pack, de-duplicated.
  """

  total_claims: int
  verified_claims: int
  unknown_identifiers: list[str] = pydantic.Field(default_factory=list)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class TraceStep(Model):
  """What one pipeline step did, for the log and the UI's trace panel."""

  name: str
  duration_ms: float
  summary: dict[str, Any] = pydantic.Field(default_factory=dict)


class ResponseMeta(Model):
  """Provenance of a response: enough to reproduce or audit it.

  Attributes:
    version: Package version.
    artifact_built_at: When the index artifact was built.
    dataset_sha256: SHA-256 of each input file, keyed by file name.
    embedding_model: Embedding model name, or None when disabled.
    rerank_model: Reranker name, or None when disabled.
    llm_provider: Language-model provider name, or None in no-LLM mode.
    llm_model: Language-model identifier, or None in no-LLM mode.
    prompt_sha256: SHA-256 of each prompt file, keyed by prompt name.
  """

  version: str
  artifact_built_at: str | None = None
  dataset_sha256: dict[str, str] = pydantic.Field(default_factory=dict)
  embedding_model: str | None = None
  rerank_model: str | None = None
  llm_provider: str | None = None
  llm_model: str | None = None
  prompt_sha256: dict[str, str] = pydantic.Field(default_factory=dict)


class AnalyzeResponse(Model):
  """Everything ``POST /v1/analyze`` returns.

  Attributes:
    query: The input as typed.
    status: ``matched`` or ``no_match``.
    parsed: Step 1's reading of the query.
    matches: Matched QIDs with their supporting chunks.
    groups: Running affected hosts folded into groups, by priority.
    hosts: Running affected hosts, by priority.
    inactive_hosts: Affected hosts that are not running; listed, not ranked.
    fix_evidence: What the data says about fixing the matched items.
    context: Every source the brief was allowed to cite.
    summary: A deterministic one-paragraph summary built from the fields
      above. Always present, so the response reads the same with or without
      a language model.
    answer: The language model's cited brief, or None in no-LLM mode.
    verification: The verifier's tally for ``answer``.
    caveats: Limits of the data that bear on this answer.
    notices: Operational notes, e.g. why ``answer`` is missing.
    trace: One entry per pipeline step.
    meta: Provenance.
  """

  query: str
  status: AnalysisStatus
  parsed: ParsedQuery
  matches: list[QidMatch] = pydantic.Field(default_factory=list)
  groups: list[HostGroup] = pydantic.Field(default_factory=list)
  hosts: list[RankedHost] = pydantic.Field(default_factory=list)
  inactive_hosts: list[Host] = pydantic.Field(default_factory=list)
  fix_evidence: list[FixEvidence] = pydantic.Field(default_factory=list)
  context: list[ContextItem] = pydantic.Field(default_factory=list)
  summary: str = ""
  answer: Answer | None = None
  verification: Verification | None = None
  caveats: list[str] = pydantic.Field(default_factory=list)
  notices: list[str] = pydantic.Field(default_factory=list)
  trace: list[TraceStep] = pydantic.Field(default_factory=list)
  meta: ResponseMeta


# ---------------------------------------------------------------------------
# Other API shapes
# ---------------------------------------------------------------------------


class AnalyzeRequest(Model):
  """The body of ``POST /v1/analyze``.

  Attributes:
    query: A CVE id, a QID written as ``QID 12345``, or advisory text.
  """

  query: str = pydantic.Field(min_length=1, max_length=MAX_QUERY_CHARS)


class SearchResponse(Model):
  """What ``GET /v1/search`` returns: retrieval only, with per-stage scores."""

  query: str
  results: list[ScoredChunk] = pydantic.Field(default_factory=list)


class HostDetail(Model):
  """What ``GET /v1/hosts/{id}`` returns.

  Attributes:
    host: The host.
    ports: Its open ports.
    detections: Everything the scanner found on it. Each detection says
      whether the corpus can explain it, so that a view can keep the two
      kinds apart.
    qid_labels: The label of every explained QID detected on the host.
  """

  host: Host
  ports: list[OpenPort] = pydantic.Field(default_factory=list)
  detections: list[Detection] = pydantic.Field(default_factory=list)
  qid_labels: dict[str, str] = pydantic.Field(default_factory=dict)


class Readiness(Model):
  """What ``GET /readyz`` returns.

  Attributes:
    status: ``ready`` once the artifact and the models are loaded.
    reason: Why the service is not ready, e.g. that no artifact was found.
    meta: Provenance of the loaded artifact and models, when ready.
    stats: Size and coverage of the loaded corpus, when ready.
  """

  status: ReadinessStatus
  reason: str | None = None
  meta: ResponseMeta | None = None
  stats: CorpusStats | None = None
