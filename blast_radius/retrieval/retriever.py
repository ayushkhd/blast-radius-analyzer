"""Step 2 of the pipeline: from a parsed query to the QIDs it is about.

Hosts attach to QIDs, so whatever the analyst typed has to end up as a set
of QIDs before the blast radius can be computed. There are two ways there:

* **Identifiers** are resolved by primary key. A CVE id or a QID is a
  lookup, not a search problem, and a lookup cannot return a near miss.
* **Free text** goes through the search stages: BM25 over the FTS5 index,
  cosine similarity over the embeddings, reciprocal rank fusion of the two,
  and a cross-encoder rerank of the fused candidates. Each stage can be
  switched off in ``config.Settings``, which is what lets the evaluation
  compare them.

Search always returns *something*, so the last step decides which
candidates are good enough to act on. A chunk supports a match only when
its score clears an absolute floor and sits within a margin of the best
score; the floor is what lets the pipeline answer "nothing in the corpus
matches" instead of returning its nearest neighbour, and the margin keeps
one strong hit from dragging in every weak sibling. A false match here
becomes a list of hosts that are not affected, so the rule is deliberately
conservative. BM25 scores are not comparable between queries, so a
keyword-only configuration has no floor to apply and trusts only its top
hit.

Search runs only when no identifier resolved. An analyst who pastes
"CVE-2024-36971 kernel RCE" has already said exactly what they mean, and
the words around the identifier should not widen the answer.
"""

from collections.abc import Collection, Sequence
import dataclasses
import logging

from blast_radius import config
from blast_radius import embeddings
from blast_radius import models
from blast_radius import schema
from blast_radius import store
from blast_radius.retrieval import dense
from blast_radius.retrieval import fusion
from blast_radius.retrieval import keyword
from blast_radius.retrieval import rerank

Docs = Collection[tuple[models.DocType, str]]

_LOG = logging.getLogger(__name__)


class RetrieverConfigError(Exception):
  """The artifact, the models and the settings do not fit together."""


@dataclasses.dataclass
class _Draft:
  """A QID match while it is being assembled.

  Attributes:
    qid: The matched QID.
    matched_by: How the QID was reached.
    matched_cve_ids: CVEs of this QID that the query hit, in hit order.
    chunks: Supporting chunks, best first.
  """

  qid: str
  matched_by: models.MatchedBy
  matched_cve_ids: list[str] = dataclasses.field(default_factory=list)
  chunks: list[models.ScoredChunk] = dataclasses.field(default_factory=list)


def _best_ranks(best: dict[int, int], ranking: Sequence[int]) -> None:
  """Records each id's 1-based rank in ``ranking`` if it beats ``best``."""
  for rank, chunk_id in enumerate(ranking, start=1):
    if chunk_id not in best or rank < best[chunk_id]:
      best[chunk_id] = rank


def _best_per_document(
    scored: Sequence[models.ScoredChunk],
) -> list[models.ScoredChunk]:
  """Returns ``scored`` with only the best chunk of each document kept.

  A long description yields several chunks that share a title, and its
  kernel-trace chunks score as well as its prose because the title is read
  with them. Left alone, one document can fill the whole result and push
  its neighbours out; the ranking is about documents, so each gets one slot.

  Args:
    scored: Chunks in final order, best first.
  """
  seen: set[tuple[str, str]] = set()
  best = []
  for item in scored:
    doc = (item.chunk.doc_type, item.chunk.doc_id)
    if doc not in seen:
      seen.add(doc)
      best.append(item)
  return best


def _rerank_text(chunk: models.Chunk) -> str:
  """Returns what the cross-encoder reads for ``chunk``.

  The title goes first for the same reason it is embedded with the chunk:
  the second chunk of a long description rarely names its own subject.
  """
  return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


class Retriever:
  """Maps a parsed query to matched QIDs over one index artifact."""

  def __init__(
      self,
      db: store.Store,
      embedder: embeddings.Embedder | None,
      reranker: rerank.Reranker | None,
      settings: config.Settings,
  ) -> None:
    """Initialises the retriever and loads the dense index into memory.

    Args:
      db: The artifact to search.
      embedder: Embeds queries for dense search. Ignored when
        ``settings.enable_dense`` is false; required when it is true.
      reranker: Scores fused candidates. Ignored when
        ``settings.enable_rerank`` is false; required when it is true.
      settings: Which stages run and how many candidates each keeps.

    Raises:
      RetrieverConfigError: If no search stage is enabled, if an enabled
        stage was not given its model, or if the artifact was embedded with
        a different model from ``embedder``, in which case query and
        document vectors would not be comparable.
    """
    if not (settings.enable_keyword or settings.enable_dense):
      raise RetrieverConfigError(
          "at least one of keyword and dense search must be enabled"
      )
    self._db = db
    self._settings = settings
    self._embedder: embeddings.Embedder | None = None
    self._dense: dense.DenseIndex | None = None
    self._reranker: rerank.Reranker | None = None

    if settings.enable_dense:
      if embedder is None:
        raise RetrieverConfigError(
            "dense search is enabled without an embedder"
        )
      built_with = db.meta().get(schema.META_EMBEDDING_MODEL)
      if built_with != embedder.name:
        raise RetrieverConfigError(
            f"the artifact was embedded with {built_with!r} but queries would"
            f" be embedded with {embedder.name!r}; rebuild the artifact or"
            " change BLAST_EMBEDDING_MODEL"
        )
      chunk_ids, matrix = db.load_embeddings()
      self._embedder = embedder
      self._dense = dense.DenseIndex(chunk_ids, matrix)
      _LOG.info("loaded %d vectors for dense search", len(chunk_ids))

    if settings.enable_rerank:
      if reranker is None:
        raise RetrieverConfigError("reranking is enabled without a reranker")
      self._reranker = reranker

  @property
  def embedding_model(self) -> str | None:
    """Returns the embedding model's name, or None when dense is off."""
    return self._embedder.name if self._embedder else None

  @property
  def rerank_model(self) -> str | None:
    """Returns the reranker's name, or None when reranking is off."""
    return self._reranker.name if self._reranker else None

  # -------------------------------------------------------------------------
  # Search
  # -------------------------------------------------------------------------

  def search(
      self,
      queries: Sequence[str],
      *,
      docs: Docs | None = None,
      limit: int | None = None,
  ) -> list[models.ScoredChunk]:
    """Returns the best chunks for ``queries``, best first.

    Every query is run through every enabled search stage, and all of the
    resulting rankings are fused together, so a chunk that several phrasings
    agree on rises. Reranking and ``dense_score`` use the first query, which
    callers order most specific first.

    Args:
      queries: One or more phrasings of the same information need. Blank
        ones are ignored.
      docs: When given, only chunks of these documents are searched. This is
        how step 5 looks for fix guidance inside the matched write-ups.
      limit: How many chunks to return. Defaults to
        ``settings.context_count``.
    """
    queries = [query for query in queries if query.strip()]
    if not queries or (docs is not None and not docs):
      return []
    settings = self._settings
    allowed = self._chunk_ids_of(docs) if docs is not None else None

    rankings: list[list[int]] = []
    keyword_ranks: dict[int, int] = {}
    dense_ranks: dict[int, int] = {}
    for query in queries:
      if settings.enable_keyword:
        match_query = keyword.build_match_query(query)
        if match_query is not None:
          hits = self._db.keyword_search(
              match_query, settings.candidate_count, docs=docs
          )
          ranking = [chunk_id for chunk_id, _ in hits]
          rankings.append(ranking)
          _best_ranks(keyword_ranks, ranking)
      if self._dense is not None and self._embedder is not None:
        vector = self._embedder.embed_query(query)
        hits = self._dense.search(
            vector, settings.candidate_count, allowed=allowed
        )
        ranking = [chunk_id for chunk_id, _ in hits]
        rankings.append(ranking)
        _best_ranks(dense_ranks, ranking)

    fused = fusion.reciprocal_rank_fusion(
        rankings, k=settings.fusion_k, limit=settings.candidate_count
    )
    fused_scores = dict(fused)
    chunks = self._db.get_chunks([chunk_id for chunk_id, _ in fused])
    cosines = self._cosines(queries[0], [chunk.id for chunk in chunks])

    scored = [
        models.ScoredChunk(
            chunk=chunk,
            keyword_rank=keyword_ranks.get(chunk.id),
            dense_rank=dense_ranks.get(chunk.id),
            dense_score=cosines.get(chunk.id),
            fused_score=fused_scores[chunk.id],
        )
        for chunk in chunks
    ]
    if scored and self._should_rerank(queries[0]):
      scored = self._rerank(queries[0], scored)
    scored = _best_per_document(scored)
    return scored[: limit if limit is not None else settings.context_count]

  def _should_rerank(self, query: str) -> bool:
    """Returns whether the cross-encoder should score ``query``.

    It was trained on short search queries. Measured on the evaluation set,
    it sharpens those and misranks long pasted advisories, which also cost
    it the most time, so long text keeps its fused order.
    """
    if self._reranker is None:
      return False
    return len(query.split()) <= self._settings.rerank_max_query_words

  def _chunk_ids_of(self, docs: Docs) -> set[int]:
    """Returns the ids of every chunk of ``docs``."""
    return {
        chunk.id
        for doc_type, doc_id in docs
        for chunk in self._db.chunks_for_doc(doc_type, doc_id)
    }

  def _cosines(self, query: str, chunk_ids: Sequence[int]) -> dict[int, float]:
    """Returns the cosine of ``query`` with each embedded chunk of the ids.

    A chunk that only keyword search found still needs a cosine, because
    without a reranker the abstention floor is a cosine floor. Trace chunks
    have no vector and are absent from the result.
    """
    if self._dense is None or self._embedder is None or not chunk_ids:
      return {}
    vector = self._embedder.embed_query(query)
    return dict(self._dense.search(vector, len(chunk_ids), allowed=chunk_ids))

  def _rerank(
      self, query: str, scored: Sequence[models.ScoredChunk]
  ) -> list[models.ScoredChunk]:
    """Returns ``scored`` with rerank scores filled in, best first.

    Args:
      query: The query the cross-encoder reads beside each candidate.
      scored: Fused candidates.
    """
    assert self._reranker is not None
    scores = self._reranker.score(
        query, [_rerank_text(item.chunk) for item in scored]
    )
    reranked = [
        item.model_copy(update={"rerank_score": score})
        for item, score in zip(scored, scores, strict=True)
    ]
    # Ties fall back to chunk id so that the order never depends on the
    # order in which the search stages happened to return candidates.
    return sorted(
        reranked, key=lambda item: (-(item.rerank_score or 0.0), item.chunk.id)
    )

  # -------------------------------------------------------------------------
  # Retrieval
  # -------------------------------------------------------------------------

  def retrieve(self, parsed: models.ParsedQuery) -> models.RetrievalResult:
    """Returns the QIDs that ``parsed`` is about.

    Args:
      parsed: Step 1's reading of the query.

    Returns:
      Identifier matches in the order they were typed, or else search
      matches by descending score. ``abstained`` is true when search ran and
      nothing cleared the floor.
    """
    drafts: dict[str, _Draft] = {}
    unknown: list[str] = []
    for cve_id in parsed.cve_ids:
      qids = self._db.qids_for_cve(cve_id)
      if not qids:
        unknown.append(cve_id)
      for qid in qids:
        draft = drafts.setdefault(qid, _Draft(qid, "identifier"))
        draft.matched_cve_ids.append(cve_id)
    for qid in parsed.qids:
      if self._db.get_qid(qid) is None:
        unknown.append(f"QID {qid}")
      else:
        drafts.setdefault(qid, _Draft(qid, "identifier"))

    candidates: list[models.ScoredChunk] = []
    abstained = False
    reason = None
    if not drafts and parsed.search_queries:
      candidates = self.search(parsed.search_queries)
      supporting = self._supporting(candidates)
      if not supporting:
        abstained = True
        reason = self._abstention_reason(candidates)
      for item in supporting:
        self._add_search_hit(drafts, item)

    matches = [self._finish(draft) for draft in drafts.values()]
    # Candidates that were not reranked are in fused order, which need not be
    # the order of the cosines the matches are scored by. The sort is stable,
    # so identifier matches, which have no score, stay as they were typed.
    matches.sort(key=lambda match: -(match.score or 0.0))
    return models.RetrievalResult(
        matches=matches,
        candidates=candidates,
        unknown_identifiers=unknown,
        abstained=abstained,
        reason=reason,
    )

  def _gate(
      self, candidates: Sequence[models.ScoredChunk]
  ) -> tuple[str, float, float] | None:
    """Returns which score decides a match, with its floor and margin.

    The decision follows the scores the candidates actually carry, not the
    models the retriever holds, because a long query skips the reranker.

    Args:
      candidates: The output of ``search``.

    Returns:
      ``(field, floor, margin)`` where ``field`` names a ``ScoredChunk``
      attribute, or None when no candidate carries a comparable score: BM25
      scores are not comparable between queries, so a keyword-only search
      has nothing to hold a floor against.
    """
    settings = self._settings
    if any(item.rerank_score is not None for item in candidates):
      return (
          "rerank_score",
          settings.abstain_rerank_floor,
          settings.match_margin,
      )
    if any(item.dense_score is not None for item in candidates):
      return (
          "dense_score",
          settings.abstain_dense_floor,
          settings.match_margin_dense,
      )
    return None

  def _supporting(
      self, candidates: Sequence[models.ScoredChunk]
  ) -> list[models.ScoredChunk]:
    """Returns the candidates good enough to act on, best first.

    Args:
      candidates: The output of ``search``, best first.
    """
    if not candidates:
      return []
    gate = self._gate(candidates)
    if gate is None:
      # Nothing to hold a floor against: trust the top hit and nothing else.
      return [candidates[0]]
    field, floor, margin = gate
    scores = [getattr(item, field) for item in candidates]
    known = [score for score in scores if score is not None]
    if max(known) < floor:
      return []
    cutoff = max(floor, max(known) - margin)
    return [
        item
        for item, score in zip(candidates, scores, strict=True)
        if score is not None and score >= cutoff
    ]

  def _abstention_reason(self, candidates: Sequence[models.ScoredChunk]) -> str:
    """Returns one sentence saying why search matched nothing.

    Args:
      candidates: The output of ``search``, none of which cleared the floor.
    """
    gate = self._gate(candidates) if candidates else None
    if gate is None:
      return "No write-up shares any search term with the query."
    field, floor, _ = gate
    kind = "relevance score" if field == "rerank_score" else "similarity"
    best = max(
        score
        for score in (getattr(item, field) for item in candidates)
        if score is not None
    )
    return (
        f"The closest write-up has a {kind} of {best:.2f}, below the floor of"
        f" {floor:.2f}, so nothing is reported rather than a near miss."
    )

  def _add_search_hit(
      self, drafts: dict[str, _Draft], item: models.ScoredChunk
  ) -> None:
    """Rolls one supporting chunk up to the QIDs it belongs to.

    Args:
      drafts: Matches assembled so far, keyed by QID; updated in place.
      item: A chunk that cleared the floor.
    """
    chunk = item.chunk
    if chunk.doc_type == "qid":
      qids = [chunk.doc_id]
    else:
      qids = self._db.qids_for_cve(chunk.doc_id)
    for qid in qids:
      draft = drafts.setdefault(qid, _Draft(qid, "search"))
      draft.chunks.append(item)
      if chunk.doc_type == "cve" and chunk.doc_id not in draft.matched_cve_ids:
        draft.matched_cve_ids.append(chunk.doc_id)

  def _finish(self, draft: _Draft) -> models.QidMatch:
    """Returns the finished match for ``draft``.

    Args:
      draft: A QID with everything the query hit under it.
    """
    qid = self._db.get_qid(draft.qid)
    assert qid is not None, f"QID {draft.qid} vanished from the artifact"
    gate = self._gate(draft.chunks)
    field = gate[0] if gate else "fused_score"
    scores = [
        score
        for score in (getattr(item, field) for item in draft.chunks)
        if score is not None
    ]
    return models.QidMatch(
        qid=qid.qid,
        label=qid.label,
        category=qid.category,
        severity=qid.severity,
        matched_by=draft.matched_by,
        score=max(scores) if scores else None,
        cves=[
            models.CveSummary(
                cve_id=cve.cve_id,
                title=cve.title,
                cvss=cve.cvss,
                epss_percentile=cve.epss_percentile,
                attack_vector=cve.attack_vector,
                known_exploited=cve.known_exploited,
                cogent_risk_score=cve.cogent_risk_score,
            )
            for cve in self._db.cves_for_qid(qid.qid)
        ],
        matched_cve_ids=draft.matched_cve_ids,
        chunks=draft.chunks,
    )
