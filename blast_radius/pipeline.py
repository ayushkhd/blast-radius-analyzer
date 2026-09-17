"""The seven-step pipeline behind ``POST /v1/analyze``.

    1 parse        identifiers by regex; long free text distilled by the LLM
    2 retrieve     query -> matched QIDs (lookup, or hybrid search + rerank)
    3 resolve      matched QIDs -> affected hosts, by SQL join
    4 rank, group  deterministic priority; like hosts folded together
    5 fix evidence what the matched write-ups say about fixing it
    6 write        the LLM composes a cited brief from a context pack
    7 verify       every quote and identifier is checked against that pack

Steps 3 and 4 are the authority path: which hosts are affected, how many,
and in what order is decided by SQL and arithmetic before the language model
runs, and the model is shown only a summary of the result. It can therefore
neither add nor drop a host, and text injected into a write-up can at worst
spoil prose that step 7 then flags.

The language model is optional twice over. Without a provider, or when a
call fails, step 1 searches the text as typed and step 6 is skipped; the
response then carries everything except the brief, plus a notice saying
why. ``summary`` and ``caveats`` are always built from the data, so the
response reads the same either way.

Every step records its duration and a summary of what it produced in the
response's ``trace``.
"""

from collections.abc import Iterator, Sequence
import contextlib
import logging
import time
from typing import Any

import blast_radius
from blast_radius import config
from blast_radius import fix_evidence
from blast_radius import models
from blast_radius import ranking
from blast_radius import schema
from blast_radius import store
from blast_radius import verify
from blast_radius.llm import base as llm_base
from blast_radius.llm import prompts
from blast_radius.retrieval import identifiers
from blast_radius.retrieval import retriever as retriever_lib

_LOG = logging.getLogger(__name__)

# A QID that bundles more CVEs than this gets a caveat: the scanner reports
# the bundle, so the data cannot say which of them applies to which host.
_BUNDLE_CAVEAT_THRESHOLD = 10


class _Trace:
  """Collects one ``TraceStep`` per pipeline step."""

  def __init__(self) -> None:
    self.steps: list[models.TraceStep] = []

  @contextlib.contextmanager
  def step(self, name: str) -> Iterator[dict[str, Any]]:
    """Times a step and records the summary the caller fills in.

    Args:
      name: The step's name as it appears in the trace.

    Yields:
      A dict the step writes its summary into.
    """
    summary: dict[str, Any] = {}
    started = time.perf_counter()
    try:
      yield summary
    finally:
      duration_ms = round((time.perf_counter() - started) * 1000, 2)
      self.steps.append(
          models.TraceStep(name=name, duration_ms=duration_ms, summary=summary)
      )
      _LOG.info(
          "step %s finished in %.1f ms",
          name,
          duration_ms,
          extra={"step": name, "duration_ms": duration_ms},
      )


def parse_without_llm(query: str) -> tuple[models.ParsedQuery, str]:
  """Returns step 1's result when no language model takes part.

  Identifiers come out by regular expression, and whatever text is left is
  searched as typed. The evaluation calls this directly, so that retrieval
  is scored on exactly the parse the pipeline would use without a model.

  Args:
    query: The analyst's input.

  Returns:
    The parsed query, and the free text left after the identifiers.
  """
  found = identifiers.extract(query)
  parsed = models.ParsedQuery(
      raw=query,
      cve_ids=found.cve_ids,
      qids=found.qids,
      search_queries=[found.remainder] if found.remainder else [],
  )
  return parsed, found.remainder


def _plural(count: int, noun: str) -> str:
  """Returns e.g. ``"1 host"`` or ``"3 hosts"``."""
  return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _is_are(count: int) -> str:
  """Returns the verb that agrees with ``count`` things."""
  return "is" if count == 1 else "are"


class Pipeline:
  """Answers blast-radius questions over one index artifact."""

  def __init__(
      self,
      db: store.Store,
      retriever: retriever_lib.Retriever,
      provider: llm_base.Provider | None,
      settings: config.Settings,
  ) -> None:
    """Initialises the pipeline.

    Args:
      db: The index artifact.
      retriever: Step 2, and the restricted search of step 5.
      provider: The language model, or None to run without one.
      settings: Thresholds, caps and ranking weights.
    """
    self._db = db
    self._retriever = retriever
    self._provider = provider
    self._settings = settings
    # Read once: the artifact is immutable for the life of the process.
    self._stats = db.stats()
    self._inventory_names = frozenset(db.host_names())
    meta = db.meta()
    self._meta = models.ResponseMeta(
        version=blast_radius.__version__,
        artifact_built_at=meta.get(schema.META_BUILT_AT),
        dataset_sha256={
            "assets": meta.get(schema.META_ASSETS_SHA256, ""),
            "vulns": meta.get(schema.META_VULNS_SHA256, ""),
        },
        embedding_model=retriever.embedding_model,
        rerank_model=retriever.rerank_model,
        llm_provider=provider.name if provider else None,
        llm_model=provider.model if provider else None,
        prompt_sha256=prompts.prompt_hashes() if provider else {},
    )

  @property
  def meta(self) -> models.ResponseMeta:
    """Returns the provenance attached to every response."""
    return self._meta

  def analyze(self, query: str) -> models.AnalyzeResponse:
    """Runs the pipeline for one query.

    Args:
      query: A CVE id, a QID written as ``QID 12345``, or advisory text.

    Raises:
      ValueError: If ``query`` is blank.
    """
    query = query.strip()
    if not query:
      raise ValueError("query must not be blank")
    trace = _Trace()
    notices: list[str] = []

    with trace.step("parse") as summary:
      parsed, free_text = self._parse(query, notices)
      summary.update(
          cve_ids=parsed.cve_ids,
          qids=parsed.qids,
          search_queries=parsed.search_queries,
          used_llm=parsed.used_llm,
      )

    with trace.step("retrieve") as summary:
      retrieval = self._retriever.retrieve(parsed)
      summary.update(
          matched_qids=[match.qid for match in retrieval.matches],
          candidates=len(retrieval.candidates),
          abstained=retrieval.abstained,
          unknown_identifiers=retrieval.unknown_identifiers,
      )
    matches = retrieval.matches

    with trace.step("resolve") as summary:
      affected = self._db.hosts_for_qids([match.qid for match in matches])
      summary.update(affected_hosts=len(affected))

    with trace.step("rank") as summary:
      hosts, inactive = ranking.rank_hosts(affected, matches, self._settings)
      groups = ranking.group_hosts(hosts, self._settings)
      summary.update(
          running=len(hosts), inactive=len(inactive), groups=len(groups)
      )

    with trace.step("fix_evidence") as summary:
      findings = fix_evidence.collect(
          self._db, self._retriever, matches, free_text, self._settings
      )
      context = self._context(matches, findings)
      summary.update(
          evidence=len(findings.evidence),
          fix_chunks=len(findings.chunks),
          context_items=len(context),
      )

    caveats = self._caveats(retrieval, findings.evidence, inactive)
    answer: models.Answer | None = None
    verification: models.Verification | None = None
    if matches:
      with trace.step("write") as summary:
        request = prompts.build_write_request(
            query,
            parsed,
            matches,
            groups,
            len(hosts),
            len(inactive),
            context,
            caveats,
        )
        answer = self._write(request, notices, summary)
      if answer is not None:
        with trace.step("verify") as summary:
          answer, verification = verify.verify_answer(
              answer,
              context,
              shown_text=request.prompt,
              inventory_names=self._inventory_names,
          )
          summary.update(
              total_claims=verification.total_claims,
              verified_claims=verification.verified_claims,
              unknown_identifiers=verification.unknown_identifiers,
          )

    return models.AnalyzeResponse(
        query=query,
        status="matched" if matches else "no_match",
        parsed=parsed,
        matches=matches,
        groups=groups,
        hosts=hosts,
        inactive_hosts=inactive,
        fix_evidence=findings.evidence,
        context=context,
        summary=self._summary(retrieval, hosts, inactive, groups),
        answer=answer,
        verification=verification,
        caveats=caveats,
        notices=notices,
        trace=trace.steps,
        meta=self._meta,
    )

  # -------------------------------------------------------------------------
  # Step 1
  # -------------------------------------------------------------------------

  def _parse(
      self, query: str, notices: list[str]
  ) -> tuple[models.ParsedQuery, str]:
    """Returns the parsed query and the free text left after identifiers.

    Identifiers never need a model. Short free text is already a search
    query. Only a long advisory is worth distilling, because BM25 over forty
    words of prose matches on the prose, not on the product.

    Args:
      query: The analyst's input.
      notices: Receives a note if the language model was tried and failed.
    """
    parsed, free_text = parse_without_llm(query)
    has_identifier = bool(parsed.cve_ids or parsed.qids)
    is_long = len(free_text.split()) >= self._settings.parse_min_words
    if self._provider is None or has_identifier or not is_long:
      return parsed, free_text

    request = prompts.build_parse_request(free_text)
    response = self._provider.complete(
        request.system, request.prompt, request.schema
    )
    if not response.ok or response.parsed is None:
      notices.append(
          f"The language model could not parse the query ({response.kind});"
          " the text was searched as typed."
      )
      return parsed, free_text
    distilled = prompts.parsed_query_from(response.parsed, parsed)
    # The analyst's own words stay in as the last phrasing: a distilled
    # query can lose the one term that mattered.
    queries = [q for q in distilled.search_queries if q != free_text]
    return (
        distilled.model_copy(update={"search_queries": queries + [free_text]}),
        free_text,
    )

  # -------------------------------------------------------------------------
  # Steps 5 and 6
  # -------------------------------------------------------------------------

  def _context(
      self,
      matches: Sequence[models.QidMatch],
      findings: fix_evidence.FixFindings,
  ) -> list[models.ContextItem]:
    """Returns every source the brief may cite, most relevant first.

    The matched chunks come first, then the fix-related chunks, then the
    structured fix evidence. Kernel traces are left out: they are noise to
    a reader and the largest injection surface in the corpus. The list is
    capped so that a QID bundling hundreds of CVEs cannot flood the prompt,
    and the cap never squeezes out the evidence.

    Args:
      matches: Step 2's matches.
      findings: Step 5's evidence and fix-related chunks.
    """
    chunks: list[models.Chunk] = []
    for match in matches:
      if match.chunks:
        chunks.extend(item.chunk for item in match.chunks)
      else:
        chunks.extend(self._write_up_of(match))
    chunks.extend(item.chunk for item in findings.chunks)

    evidence_items = [
        models.ContextItem(
            id=evidence.id,
            doc_type=evidence.doc_type,
            doc_id=evidence.doc_id,
            title=evidence.kind.replace("_", " "),
            text=evidence.text,
        )
        for evidence in findings.evidence
    ]
    # Evidence is short and is what "what to do" rests on, so it keeps up to
    # half of the budget and the chunks take whatever is left.
    limit = self._settings.max_context_items
    evidence_items = evidence_items[: limit // 2]

    items: list[models.ContextItem] = []
    seen: set[str] = set()
    for chunk in chunks:
      item_id = f"c{chunk.id}"
      if chunk.kind == "text" and item_id not in seen:
        seen.add(item_id)
        items.append(
            models.ContextItem(
                id=item_id,
                doc_type=chunk.doc_type,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
            )
        )
    return items[: limit - len(evidence_items)] + evidence_items

  def _write_up_of(self, match: models.QidMatch) -> list[models.Chunk]:
    """Returns the write-up of a match that was reached by identifier.

    Search hands back the chunks that matched; a lookup hands back none, so
    the QID's diagnosis and the descriptions of the CVEs that were asked for
    are fetched here.

    Args:
      match: A match whose ``chunks`` is empty.
    """
    limit = self._settings.context_count
    chunks = self._db.chunks_for_doc("qid", match.qid)
    for cve_id in match.matched_cve_ids:
      chunks.extend(self._db.chunks_for_doc("cve", cve_id))
    return [chunk for chunk in chunks if chunk.kind == "text"][:limit]

  def _write(
      self,
      request: prompts.PromptRequest,
      notices: list[str],
      summary: dict[str, Any],
  ) -> models.Answer | None:
    """Returns the model's brief, or None with a notice saying why not.

    Args:
      request: The rendered write prompt.
      notices: Receives the reason when there is no brief.
      summary: The trace summary of the write step.
    """
    if self._provider is None:
      notices.append(
          "No language model is configured, so there is no written brief."
          " Everything else in this response is computed from the data."
      )
      summary.update(skipped="no provider")
      return None
    response = self._provider.complete(
        request.system, request.prompt, request.schema
    )
    summary.update(
        ok=response.ok,
        kind=response.kind,
        model=response.model,
        duration_s=round(response.duration_s, 2),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
    answer = prompts.answer_from(response.parsed) if response.parsed else None
    if answer is None:
      reason = response.kind or llm_base.KIND_PARSE
      notices.append(
          f"The language model did not return a usable brief ({reason}), so"
          " there is no written brief. Everything else in this response is"
          " computed from the data."
      )
    return answer

  # -------------------------------------------------------------------------
  # Deterministic prose
  # -------------------------------------------------------------------------

  def _coverage_sentence(self) -> str:
    """Returns the sentence that says how much the corpus cannot explain."""
    stats = self._stats
    unexplained = stats.total_qids - stats.explained_qids
    share = 0.0
    if stats.total_detections:
      share = 1 - stats.explained_detections / stats.total_detections
    return (
        f"{unexplained} of the {stats.total_qids} QIDs detected in this"
        f" environment ({share:.0%} of detections) have no write-up, so a"
        " missing match is not evidence that the environment is unaffected."
    )

  def _caveats(
      self,
      retrieval: models.RetrievalResult,
      evidence: Sequence[models.FixEvidence],
      inactive: Sequence[models.Host],
  ) -> list[str]:
    """Returns the limits of the data that bear on this answer.

    Args:
      retrieval: Step 2's result.
      evidence: Step 5's fix evidence.
      inactive: Affected hosts that are not running.
    """
    caveats: list[str] = []
    for identifier in retrieval.unknown_identifiers:
      caveats.append(f"{identifier} is not in the corpus.")
    if retrieval.reason:
      caveats.append(retrieval.reason)
    if not retrieval.matches:
      caveats.append(self._coverage_sentence())
      return caveats

    with_evidence = {item.doc_id for item in evidence}
    for match in retrieval.matches:
      cve_ids = {cve.cve_id for cve in match.cves}
      if match.severity is None and not match.cves:
        caveats.append(
            f"QID {match.qid} has no write-up in the corpus; its hosts come"
            " from the scanner's detections alone."
        )
      elif match.qid not in with_evidence and not cve_ids & with_evidence:
        caveats.append(f"The data holds no fix guidance for QID {match.qid}.")
      if len(match.cves) > _BUNDLE_CAVEAT_THRESHOLD:
        caveats.append(
            f"QID {match.qid} bundles {len(match.cves)} CVEs. The scanner"
            " reports the bundle, so the data cannot say which of them"
            " applies to which host."
        )
    if inactive:
      count = _plural(len(inactive), "affected host")
      caveats.append(
          f"{count} not running {_is_are(len(inactive))} listed separately"
          " and not ranked."
      )
    return caveats

  def _summary(
      self,
      retrieval: models.RetrievalResult,
      hosts: Sequence[models.RankedHost],
      inactive: Sequence[models.Host],
      groups: Sequence[models.HostGroup],
  ) -> str:
    """Returns the deterministic one-paragraph summary of the response.

    Args:
      retrieval: Step 2's result.
      hosts: Running affected hosts, by priority.
      inactive: Affected hosts that are not running.
      groups: The groups ``hosts`` fold into.
    """
    if not retrieval.matches:
      return (
          "No write-up in the corpus matches this query, so no affected"
          " hosts could be resolved."
      )
    names = ", ".join(
        f"QID {match.qid} ({match.label})"
        if match.label
        else f"QID {match.qid}"
        for match in retrieval.matches
    )
    facing = sum(1 for ranked in hosts if ranked.host.internet_facing)
    running = _plural(len(hosts), "running host")
    grouped = _plural(len(groups), "group")
    sentences = [
        f"Matched {names}.",
        f"{running} {_is_are(len(hosts))} affected, {facing} of them"
        f" internet-facing, in {grouped}.",
    ]
    if hosts:
      first = hosts[0]
      sentences.append(
          f"Highest priority: {first.host.name} (priority"
          f" {first.priority:.2f}, driven by QID {first.factors.driving_qid})."
      )
    if inactive:
      more = _plural(len(inactive), "more affected host")
      sentences.append(f"{more} {_is_are(len(inactive))} not running.")
    return " ".join(sentences)
