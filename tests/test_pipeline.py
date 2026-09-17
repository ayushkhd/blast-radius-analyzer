"""Tests for blast_radius.pipeline."""

from collections.abc import Callable, Iterator
import os
import pathlib
from typing import Any

import pytest

import blast_radius
from blast_radius import config
from blast_radius import pipeline as pipeline_lib
from blast_radius import services as services_lib
from blast_radius import store as store_lib
from blast_radius.llm import base as llm_base
from blast_radius.llm import prompts
from tests import fakes

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

_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_BROKER = "710005"
_QID_UNEXPLAINED = "710901"

_CVE_LONG_KERNEL = "CVE-2099-1001"
_CVE_PROXY = "CVE-2099-2001"
_CVE_UNKNOWN = "CVE-2099-9999"
_CVE_INVENTED = "CVE-2099-7777"

_OPENSSH_QUERY = f"QID {_QID_OPENSSH}"

_STEPS_WITHOUT_A_BRIEF = [
    "parse",
    "retrieve",
    "resolve",
    "rank",
    "fix_evidence",
]

# Long enough for step 1 to hand it to the language model, and worded so
# that no write-up comes near it: only a distilled query can match.
_VAGUE_ADVISORY = (
    "Somebody at our supplier mentioned last week that one queueing product"
    " we depend on lets outsiders take over its admin port, so please tell us"
    " which machines we ought to worry about first"
)
_PARSE_REPLY: dict[str, Any] = {
    "product": "Ferrous broker",
    "version": None,
    "search_queries": [
        "Ferrous broker management listener remote code execution",
        "Ferrous broker deserialisation",
    ],
}

# Words that occur only in the call trace of the long kernel CVE.
_TRACE_QUERY = "fxnet_wq fxnet_reset_task kasan_report dump_stack_lvl"

_NO_PROVIDER_NOTICE = (
    "No language model is configured, so there is no written brief."
    " Everything else in this response is computed from the data."
)

_BuildPipeline = Callable[..., pipeline_lib.Pipeline]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps the machine's own BLAST_* variables out of the settings."""
  for name in list(os.environ):
    if name.startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(artifact_path: pathlib.Path, **overrides: Any) -> config.Settings:
  """Returns model-free settings with every stage of retrieval switched on.

  Whatever the assertions below depend on is pinned here and not left to a
  default. The rerank floor and margin are on ``LexicalReranker``'s 0-1
  scale.

  Args:
    artifact_path: The artifact to serve.
    **overrides: Settings that a test wants different.
  """
  values: dict[str, Any] = {
      "artifact_path": artifact_path,
      "embedding_model": "hashing-256",
      "rerank_model": "lexical",
      "enable_keyword": True,
      "enable_dense": True,
      "enable_rerank": True,
      "abstain_rerank_floor": 0.5,
      "match_margin": 0.2,
      "abstain_dense_floor": 0.6,
      "parse_min_words": 12,
      "llm_provider": "none",
  }
  return config.Settings(_env_file=None, **{**values, **overrides})


@pytest.fixture(name="build_pipeline")
def fixture_build_pipeline(
    artifact_path: pathlib.Path,
) -> Iterator[_BuildPipeline]:
  """Yields a factory of pipelines on the session's artifact.

  The factory takes a provider, or None to run without one, and settings
  overrides as keywords. Whatever it built is closed when the test ends.

  Args:
    artifact_path: The session's artifact.
  """
  built: list[services_lib.Services] = []

  def build(
      provider: llm_base.Provider | None = None, **overrides: Any
  ) -> pipeline_lib.Pipeline:
    services = services_lib.build(
        _settings(artifact_path, **overrides), provider=provider
    )
    built.append(services)
    return services.pipeline

  yield build
  for services in built:
    services.close()


def _brief(*claims: dict[str, Any]) -> dict[str, Any]:
  """Returns a write reply with the given claims."""
  return {
      "summary": "The finding is widespread.",
      "claims": list(claims),
      "caveats": [],
  }


def _claim(text: str, source_id: str, quote: str) -> dict[str, Any]:
  """Returns one claim of a write reply, with a single citation."""
  return {
      "text": text,
      "citations": [{"source_id": source_id, "quote": quote}],
  }


# ---------------------------------------------------------------------------
# parse_without_llm
# ---------------------------------------------------------------------------


def test_parse_without_llm_separates_identifiers_from_free_text():
  parsed, free_text = pipeline_lib.parse_without_llm(
      f"header stripping {_CVE_PROXY} QID {_QID_OPENSSH}"
  )

  assert parsed.cve_ids == [_CVE_PROXY]
  assert parsed.qids == [_QID_OPENSSH]
  assert parsed.search_queries == ["header stripping"]
  assert free_text == "header stripping"
  assert not parsed.used_llm


def test_parse_without_llm_bare_identifier_leaves_nothing_to_search():
  parsed, free_text = pipeline_lib.parse_without_llm(_CVE_PROXY)

  assert parsed.raw == _CVE_PROXY
  assert parsed.search_queries == []
  assert free_text == ""


# ---------------------------------------------------------------------------
# analyze without a language model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "  \n\t "])
def test_analyze_blank_query_raises_value_error(
    build_pipeline: _BuildPipeline, query: str
):
  pipeline = build_pipeline()

  with pytest.raises(ValueError, match="blank"):
    pipeline.analyze(query)


def test_analyze_qid_query_matches_by_identifier(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(f"  {_OPENSSH_QUERY} ")

  assert response.query == _OPENSSH_QUERY
  assert response.status == "matched"
  assert [match.qid for match in response.matches] == [_QID_OPENSSH]
  assert response.matches[0].matched_by == "identifier"


def test_analyze_qid_query_splits_hosts_into_running_and_inactive(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  running = [ranked.host.id for ranked in response.hosts]
  # The two internet-facing hosts of criticality 5 outrank the rest.
  assert running[:2] == [_BASTION, _GATEWAY]
  assert sorted(running[2:]) == [
      _WORKER_A_FIRST,
      _WORKER_A_SECOND,
      _WORKER_B1,
      _WORKER_B2,
      _WEB,
      _QUEUE,
  ]
  assert [host.id for host in response.inactive_hosts] == [_WORKER_TERMINATED]


def test_analyze_qid_query_folds_running_hosts_into_groups(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  assert {group.key: group.count for group in response.groups} == {
      "sg:fx-bastion-sg": 1,
      "sg:fx-gateway-sg": 1,
      "sg:fx-k8s-workers-a-sg": 2,
      "sg:fx-k8s-workers-b-sg": 2,
      "sg:fx-data-sg": 1,
      "sg:fx-web-sg": 1,
  }


def test_analyze_qid_query_puts_fix_evidence_and_write_up_in_the_context(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  (evidence,) = response.fix_evidence
  assert evidence.kind == "affected_versions"
  assert evidence.text == "Affected Versions: OpenSSH up to version 9.6"
  assert [(item.doc_type, item.doc_id) for item in response.context] == [
      ("qid", _QID_OPENSSH),
      ("qid", _QID_OPENSSH),
  ]
  assert response.context[0].text.startswith("OpenSSH is a suite of tools")
  assert response.context[1].id == evidence.id


def test_analyze_summary_names_the_qid_and_the_counts(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  label = response.matches[0].label
  assert response.summary.startswith(f"Matched QID {_QID_OPENSSH} ({label}).")
  assert (
      "8 running hosts are affected, 2 of them internet-facing, in 6 groups."
      in response.summary
  )
  assert "Highest priority: fx-bastion" in response.summary
  assert response.summary.endswith("1 more affected host is not running.")


def test_analyze_openssh_qid_caveats_only_about_the_stopped_host(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  # The QID has no CVE, but it has a write-up and fix evidence, so neither
  # of those caveats applies.
  assert response.caveats == [
      "1 affected host not running is listed separately and not ranked."
  ]


def test_analyze_unexplained_qid_caveats_that_it_has_no_write_up(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(f"QID {_QID_UNEXPLAINED}")

  assert response.status == "matched"
  assert response.hosts
  assert response.context == []
  assert response.caveats[0] == (
      f"QID {_QID_UNEXPLAINED} has no write-up in the corpus; its hosts come"
      " from the scanner's detections alone."
  )


def test_analyze_without_provider_has_no_answer_and_says_why(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  assert response.answer is None
  assert response.verification is None
  assert response.notices == [_NO_PROVIDER_NOTICE]


def test_analyze_without_provider_traces_every_step_up_to_the_skipped_write(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  steps = {step.name: step.summary for step in response.trace}
  assert list(steps) == [*_STEPS_WITHOUT_A_BRIEF, "write"]
  assert steps["resolve"] == {"affected_hosts": 9}
  assert steps["rank"] == {"running": 8, "inactive": 1, "groups": 6}
  assert steps["write"] == {"skipped": "no provider"}
  assert all(step.duration_ms >= 0 for step in response.trace)


def test_analyze_meta_records_the_artifact_and_the_models(
    build_pipeline: _BuildPipeline,
):
  response = build_pipeline().analyze(_OPENSSH_QUERY)

  meta = response.meta
  assert meta.version == blast_radius.__version__
  assert meta.artifact_built_at is not None
  assert set(meta.dataset_sha256) == {"assets", "vulns"}
  assert all(len(digest) == 64 for digest in meta.dataset_sha256.values())
  assert meta.embedding_model == "hashing-256"
  assert meta.rerank_model == "lexical"
  assert meta.llm_provider is None
  assert meta.llm_model is None
  assert meta.prompt_sha256 == {}


def test_analyze_unknown_cve_is_no_match_with_the_coverage_caveat(
    build_pipeline: _BuildPipeline,
):
  # Scripted with nothing: any call to the model would fail the test.
  provider = fakes.ScriptedProvider([])

  response = build_pipeline(provider).analyze(_CVE_UNKNOWN)

  assert response.status == "no_match"
  assert response.hosts == []
  assert response.caveats == [
      f"{_CVE_UNKNOWN} is not in the corpus.",
      "2 of the 8 QIDs detected in this environment (42% of detections) have"
      " no write-up, so a missing match is not evidence that the environment"
      " is unaffected.",
  ]
  assert response.summary.startswith("No write-up in the corpus matches")
  assert [step.name for step in response.trace] == _STEPS_WITHOUT_A_BRIEF
  assert response.answer is None
  assert response.notices == []
  assert not provider.calls


# ---------------------------------------------------------------------------
# analyze with a language model: write and verify
# ---------------------------------------------------------------------------


def test_analyze_verifies_the_brief_and_keeps_the_claim_it_flags(
    build_pipeline: _BuildPipeline,
):
  # The context does not depend on the provider, so a run without one shows
  # what the model will be given to quote from.
  source = build_pipeline().analyze(_OPENSSH_QUERY).context[0]
  quote = source.text[:60]
  provider = fakes.ScriptedProvider(
      [
          _brief(
              _claim("The write-up describes the product.", source.id, quote),
              _claim(f"It is tracked as {_CVE_INVENTED}.", source.id, quote),
          )
      ]
  )

  response = build_pipeline(provider).analyze(_OPENSSH_QUERY)

  assert response.answer is not None
  good, invented = response.answer.claims
  assert good.verified
  assert good.citations[0].verified
  assert good.problems == []
  assert invented.verified is False
  assert invented.text == f"It is tracked as {_CVE_INVENTED}."
  assert invented.problems == [
      f"mentions {_CVE_INVENTED}, which the model was not shown"
  ]
  assert response.verification is not None
  assert response.verification.total_claims == 2
  assert response.verification.verified_claims == 1
  assert response.verification.unknown_identifiers == [_CVE_INVENTED]
  assert response.notices == []


def test_analyze_with_provider_traces_all_seven_steps_and_names_the_model(
    build_pipeline: _BuildPipeline,
):
  provider = fakes.ScriptedProvider(
      [_brief(_claim("It is OpenSSH.", "e1", "OpenSSH"))]
  )

  response = build_pipeline(provider).analyze(_OPENSSH_QUERY)

  assert [step.name for step in response.trace] == [
      *_STEPS_WITHOUT_A_BRIEF,
      "write",
      "verify",
  ]
  assert response.trace[-2].summary["ok"] is True
  assert response.trace[-1].summary["total_claims"] == 1
  assert response.meta.llm_provider == provider.name
  assert response.meta.llm_model == fakes.MODEL
  assert response.meta.prompt_sha256 == prompts.prompt_hashes()


@pytest.mark.parametrize(
    "failure, kind",
    [
        (fakes.timeout, "timeout"),
        (fakes.provider_error, "provider"),
        (fakes.refusal, "refusal"),
        (fakes.unparseable, "parse"),
    ],
)
def test_analyze_failed_write_call_degrades_to_a_notice(
    build_pipeline: _BuildPipeline,
    failure: Callable[[], llm_base.RawResponse],
    kind: str,
):
  provider = fakes.ScriptedProvider([failure()])

  response = build_pipeline(provider).analyze(_OPENSSH_QUERY)

  assert response.status == "matched"
  assert len(response.hosts) == 8
  assert response.answer is None
  assert response.verification is None
  (notice,) = response.notices
  assert f"did not return a usable brief ({kind})" in notice
  assert response.trace[-1].name == "write"
  assert response.trace[-1].summary["kind"] == kind


def test_analyze_brief_without_claims_degrades_to_a_notice(
    build_pipeline: _BuildPipeline,
):
  provider = fakes.ScriptedProvider([_brief()])

  response = build_pipeline(provider).analyze(_OPENSSH_QUERY)

  assert response.answer is None
  (notice,) = response.notices
  assert "did not return a usable brief (parse)" in notice


def test_analyze_write_prompt_names_example_hosts_and_never_the_host_list(
    build_pipeline: _BuildPipeline,
):
  provider = fakes.ScriptedProvider([fakes.refusal()])
  pipeline = build_pipeline(provider, group_example_count=1)

  response = pipeline.analyze(_OPENSSH_QUERY)

  assert len(provider.calls) == 1
  call = provider.calls[0]
  assert call.schema == prompts.WRITE_SCHEMA
  by_id = {ranked.host.id: ranked.host for ranked in response.hosts}
  assert by_id[_WORKER_B1].name in call.prompt
  # Running and affected, but second in its group, so not an example.
  assert by_id[_WORKER_B2].name not in call.prompt
  assert response.inactive_hosts[0].name not in call.prompt
  for host in by_id.values():
    assert host.id not in call.prompt
    assert host.private_ip is not None
    assert host.private_ip not in call.prompt


def test_analyze_kernel_trace_chunks_never_enter_the_context(
    build_pipeline: _BuildPipeline, store: store_lib.Store
):
  trace_ids = {
      chunk.id
      for chunk in store.chunks_for_doc("cve", _CVE_LONG_KERNEL)
      if chunk.kind == "trace"
  }

  response = build_pipeline().analyze(_TRACE_QUERY)

  (match,) = response.matches
  assert match.qid == _QID_KERNEL
  # The match rests on a trace chunk alone, which is what makes the context
  # worth checking.
  assert match.chunks
  assert {item.chunk.id for item in match.chunks} <= trace_ids
  shown = {item.id for item in response.context}
  assert shown
  assert not shown & {f"c{chunk_id}" for chunk_id in trace_ids}


# ---------------------------------------------------------------------------
# analyze with a language model: parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Trellis proxy forwarded headers",
        f"{_VAGUE_ADVISORY}, tracked as {_CVE_PROXY}",
    ],
    ids=["short free text", "long text with an identifier"],
)
def test_analyze_parse_call_is_skipped_unless_long_text_has_no_identifier(
    build_pipeline: _BuildPipeline, query: str
):
  provider = fakes.ScriptedProvider([fakes.refusal()])

  response = build_pipeline(provider).analyze(query)

  assert not response.parsed.used_llm
  assert [call.schema for call in provider.calls] == [prompts.WRITE_SCHEMA]


def test_analyze_long_free_text_is_searched_by_the_distilled_queries_first(
    build_pipeline: _BuildPipeline,
):
  provider = fakes.ScriptedProvider([_PARSE_REPLY, fakes.refusal()])

  response = build_pipeline(provider).analyze(_VAGUE_ADVISORY)

  parse_call = provider.calls[0]
  assert parse_call.schema == prompts.PARSE_SCHEMA
  assert _VAGUE_ADVISORY in parse_call.prompt
  assert response.parsed.used_llm
  assert response.parsed.product == "Ferrous broker"
  assert response.parsed.search_queries == [
      *_PARSE_REPLY["search_queries"],
      _VAGUE_ADVISORY,
  ]
  assert response.trace[0].summary["used_llm"] is True
  # The text as typed matches nothing (see the next test), so this match
  # can only come from the distilled queries.
  assert [match.qid for match in response.matches] == [_QID_BROKER]
  assert response.matches[0].matched_by == "search"


@pytest.mark.parametrize(
    "failure, kind",
    [(fakes.timeout, "timeout"), (fakes.unparseable, "parse")],
)
def test_analyze_failed_parse_call_searches_the_text_as_typed(
    build_pipeline: _BuildPipeline,
    failure: Callable[[], llm_base.RawResponse],
    kind: str,
):
  provider = fakes.ScriptedProvider([failure()])

  response = build_pipeline(provider).analyze(_VAGUE_ADVISORY)

  assert not response.parsed.used_llm
  assert response.parsed.search_queries == [_VAGUE_ADVISORY]
  assert response.notices == [
      f"The language model could not parse the query ({kind}); the text was"
      " searched as typed."
  ]
  assert response.status == "no_match"
  assert len(provider.calls) == 1
