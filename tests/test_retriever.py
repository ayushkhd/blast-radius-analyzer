"""Tests for blast_radius.retrieval.retriever."""

import os
from typing import Any

import pytest

from blast_radius import config
from blast_radius import embeddings
from blast_radius import models
from blast_radius import store as store_lib
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_QID_OPENSSH = "710001"
_QID_KERNEL = "710002"
_QID_PROXY = "710003"
_QID_WEB_SERVER = "710004"
_QID_UNEXPLAINED = "710901"
_CVE_FXNET = "CVE-2099-1001"
_CVE_QUARTZFS = "CVE-2099-1002"
_CVE_PROXY = "CVE-2099-2001"
_CVE_SMUGGLING = "CVE-2099-3001"
_CVE_TEMPLATE = "CVE-2099-3002"

# The stages that each scorer mode switches on.
_RERANK = {"enable_keyword": True, "enable_dense": True, "enable_rerank": True}
_HYBRID = {"enable_keyword": True, "enable_dense": True, "enable_rerank": False}
_DENSE = {"enable_keyword": False, "enable_dense": True, "enable_rerank": False}
_KEYWORD = {
    "enable_keyword": True,
    "enable_dense": False,
    "enable_rerank": False,
}

_PROXY_QUERY = "Trellis proxy removes forwarded headers"
_SMUGGLING_QUERY = "Marlin HTTP Server request smuggling mod_relay"
_UNCOVERED_QUERY = "PostgreSQL privilege escalation in logical replication"


@pytest.fixture(autouse=True)
def _no_blast_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps ``BLAST_*`` variables set on this machine out of the settings."""
  for name in list(os.environ):
    if name.upper().startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(**overrides: Any) -> config.Settings:
  """Returns settings whose floors suit the model-free scorers.

  ``LexicalReranker`` scores run from 0 to 1, and the cosines that
  ``HashingEmbedder`` gives the fixture corpus peak near 0.6. The production
  floors are on the scale of a cross-encoder's logits and of a sentence
  model's cosines, and would let everything or nothing through.

  Args:
    **overrides: Settings that a test sets itself, e.g. the stage toggles.
  """
  values: dict[str, Any] = {
      "abstain_rerank_floor": 0.5,
      "match_margin": 0.2,
      "abstain_dense_floor": 0.3,
      "match_margin_dense": 0.1,
  }
  return config.Settings(_env_file=None, **{**values, **overrides})


def _retriever(
    db: store_lib.Store, **overrides: Any
) -> retriever_lib.Retriever:
  """Returns a retriever on ``db`` that is handed both model-free scorers."""
  return retriever_lib.Retriever(
      db,
      embeddings.HashingEmbedder(),
      rerank.LexicalReranker(),
      _settings(**overrides),
  )


def _free_text(query: str) -> models.ParsedQuery:
  """Returns the parse of a query that holds no identifier."""
  return models.ParsedQuery(raw=query, search_queries=[query])


def _docs(
    items: list[models.ScoredChunk],
) -> list[tuple[models.DocType, str]]:
  """Returns the document of each scored chunk, in order."""
  return [(item.chunk.doc_type, item.chunk.doc_id) for item in items]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("embedder", "reranker", "toggles", "message"),
    [
        (
            embeddings.HashingEmbedder(),
            rerank.LexicalReranker(),
            {"enable_keyword": False, "enable_dense": False},
            "at least one of keyword and dense",
        ),
        (None, rerank.LexicalReranker(), _RERANK, "without an embedder"),
        (embeddings.HashingEmbedder(), None, _RERANK, "without a reranker"),
        # The artifact records hashing-256, the embedder that built it.
        (
            embeddings.HashingEmbedder(64),
            rerank.LexicalReranker(),
            _RERANK,
            "embedded with 'hashing-256'.*'hashing-64'",
        ),
    ],
    ids=["no-stage", "no-embedder", "no-reranker", "other-embedder"],
)
def test_init_with_settings_the_models_do_not_fit_raises(
    store: store_lib.Store,
    embedder: embeddings.Embedder | None,
    reranker: rerank.Reranker | None,
    toggles: dict[str, bool],
    message: str,
):
  settings = _settings(**toggles)

  with pytest.raises(retriever_lib.RetrieverConfigError, match=message):
    retriever_lib.Retriever(store, embedder, reranker, settings)


def test_init_ignores_the_models_of_disabled_stages(store: store_lib.Store):
  other_embedder = embeddings.HashingEmbedder(64)

  retriever = retriever_lib.Retriever(
      store, other_embedder, rerank.LexicalReranker(), _settings(**_KEYWORD)
  )

  assert retriever.embedding_model is None
  assert retriever.rerank_model is None


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def test_retrieve_cve_id_matches_its_qid_without_a_score(
    store: store_lib.Store,
):
  parsed = models.ParsedQuery(raw=_CVE_PROXY, cve_ids=[_CVE_PROXY])

  result = _retriever(store).retrieve(parsed)

  [match] = result.matches
  assert (match.qid, match.matched_by) == (_QID_PROXY, "identifier")
  assert match.matched_cve_ids == [_CVE_PROXY]
  assert match.score is None
  assert not result.abstained


def test_retrieve_qid_matches_as_a_whole_with_every_cve_listed(
    store: store_lib.Store,
):
  parsed = models.ParsedQuery(raw="QID 710004", qids=[_QID_WEB_SERVER])

  result = _retriever(store).retrieve(parsed)

  [match] = result.matches
  assert (match.qid, match.matched_by) == (_QID_WEB_SERVER, "identifier")
  assert [cve.cve_id for cve in match.cves] == [_CVE_SMUGGLING, _CVE_TEMPLATE]
  assert not match.matched_cve_ids


def test_retrieve_unexplained_qid_still_matches(store: store_lib.Store):
  parsed = models.ParsedQuery(raw="QID 710901", qids=[_QID_UNEXPLAINED])

  result = _retriever(store).retrieve(parsed)

  [match] = result.matches
  assert match.qid == _QID_UNEXPLAINED
  assert match.label == ""
  assert not match.cves
  assert not result.unknown_identifiers


def test_retrieve_unknown_identifiers_are_reported_and_match_nothing(
    store: store_lib.Store,
):
  parsed = models.ParsedQuery(
      raw="CVE-2099-9999 and QID 999999",
      cve_ids=["CVE-2099-9999"],
      qids=["999999"],
  )

  result = _retriever(store).retrieve(parsed)

  assert not result.matches
  assert result.unknown_identifiers == ["CVE-2099-9999", "QID 999999"]
  assert not result.abstained


def test_retrieve_skips_search_when_an_identifier_resolved(
    store: store_lib.Store,
):
  parsed = models.ParsedQuery(
      raw=f"{_CVE_PROXY} {_SMUGGLING_QUERY}",
      cve_ids=[_CVE_PROXY],
      search_queries=[_SMUGGLING_QUERY],
  )

  result = _retriever(store).retrieve(parsed)

  assert [match.qid for match in result.matches] == [_QID_PROXY]
  assert not result.candidates


def test_retrieve_searches_the_text_around_an_unknown_identifier(
    store: store_lib.Store,
):
  parsed = models.ParsedQuery(
      raw=f"CVE-2099-9999 {_SMUGGLING_QUERY}",
      cve_ids=["CVE-2099-9999"],
      search_queries=[_SMUGGLING_QUERY],
  )

  result = _retriever(store).retrieve(parsed)

  [match] = result.matches
  assert (match.qid, match.matched_by) == (_QID_WEB_SERVER, "search")
  assert result.unknown_identifiers == ["CVE-2099-9999"]


# ---------------------------------------------------------------------------
# Free text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "toggles",
    [_RERANK, _HYBRID, _DENSE, _KEYWORD],
    ids=["rerank", "hybrid", "dense", "keyword"],
)
def test_retrieve_free_text_finds_the_qid_in_every_scorer_mode(
    store: store_lib.Store, toggles: dict[str, bool]
):
  retriever = _retriever(store, **toggles)

  result = retriever.retrieve(_free_text(_PROXY_QUERY))

  [match] = result.matches
  assert (match.qid, match.matched_by) == (_QID_PROXY, "search")
  assert match.score is not None
  assert not result.abstained


def test_retrieve_keyword_only_trusts_only_its_top_hit(store: store_lib.Store):
  retriever = _retriever(store, **_KEYWORD)
  two_products = "Pylon agent state file Ferrous broker management listener"

  result = retriever.retrieve(_free_text(two_products))

  assert len(result.candidates) > 1
  [match] = result.matches
  assert match.chunks == result.candidates[:1]


@pytest.mark.parametrize(
    ("toggles", "query", "reason"),
    [
        (_RERANK, _UNCOVERED_QUERY, "relevance score of 0.17, below the floor"),
        (_HYBRID, _UNCOVERED_QUERY, "similarity of 0.07, below the floor"),
        (_KEYWORD, "zzzz qqqq", "shares any search term"),
    ],
    ids=["rerank", "hybrid", "keyword"],
)
def test_retrieve_abstains_with_a_reason_when_nothing_clears_the_floor(
    store: store_lib.Store, toggles: dict[str, bool], query: str, reason: str
):
  retriever = _retriever(store, **toggles)

  result = retriever.retrieve(_free_text(query))

  assert not result.matches
  assert result.abstained
  assert reason in (result.reason or "")


@pytest.mark.parametrize(
    ("margin", "matched_cve_ids"),
    [(0.2, [_CVE_SMUGGLING]), (0.5, [_CVE_SMUGGLING, _CVE_TEMPLATE])],
    ids=["narrow", "wide"],
)
def test_retrieve_margin_decides_whether_a_weaker_sibling_is_matched(
    store: store_lib.Store, margin: float, matched_cve_ids: list[str]
):
  retriever = _retriever(store, **_RERANK, match_margin=margin)

  result = retriever.retrieve(_free_text(_SMUGGLING_QUERY))

  # The sibling scores 0.57: above the floor of 0.5, and 0.43 below the best.
  [match] = result.matches
  assert match.qid == _QID_WEB_SERVER
  assert match.matched_cve_ids == matched_cve_ids


def test_retrieve_cve_chunk_hit_rolls_up_to_its_qid(store: store_lib.Store):
  query = "quartzfs directory entries past the end of the block"

  result = _retriever(store).retrieve(_free_text(query))

  [match] = result.matches
  assert match.qid == _QID_KERNEL
  assert match.matched_cve_ids == [_CVE_QUARTZFS]
  assert _docs(match.chunks) == [("cve", _CVE_QUARTZFS)]


def test_retrieve_orders_search_matches_by_score_not_by_fused_rank(
    store: store_lib.Store,
):
  retriever = _retriever(
      store, **_HYBRID, abstain_dense_floor=0.2, match_margin_dense=0.2
  )
  two_products = (
      "OpenSSH authentication bypass memory bit flips Trellis proxy"
      " X-Forwarded-Host headers Connection header"
  )

  result = retriever.retrieve(_free_text(two_products))

  # Fusion puts the OpenSSH diagnosis first, with the lower cosine of the two.
  assert _docs(result.candidates[:1]) == [("qid", _QID_OPENSSH)]
  assert [match.qid for match in result.matches] == [_QID_PROXY, _QID_OPENSSH]


def test_retrieve_query_within_the_word_limit_is_gated_on_rerank_scores(
    store: store_lib.Store,
):
  words = len(_PROXY_QUERY.split())
  retriever = _retriever(store, **_RERANK, rerank_max_query_words=words)

  result = retriever.retrieve(_free_text(_PROXY_QUERY))

  assert all(item.rerank_score is not None for item in result.candidates)
  [match] = result.matches
  assert match.score == result.candidates[0].rerank_score


def test_retrieve_query_over_the_word_limit_is_gated_on_cosines_instead(
    store: store_lib.Store,
):
  words = len(_PROXY_QUERY.split())
  retriever = _retriever(
      store,
      **_RERANK,
      rerank_max_query_words=words - 1,
      abstain_dense_floor=0.9,
  )

  result = retriever.retrieve(_free_text(_PROXY_QUERY))

  assert all(item.rerank_score is None for item in result.candidates)
  assert result.abstained
  assert "similarity" in (result.reason or "")


def test_retrieve_twice_returns_equal_results(store: store_lib.Store):
  retriever = _retriever(store)
  parsed = _free_text(_SMUGGLING_QUERY)

  assert retriever.retrieve(parsed) == retriever.retrieve(parsed)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_with_docs_returns_only_chunks_of_those_documents(
    store: store_lib.Store,
):
  docs: list[tuple[models.DocType, str]] = [
      ("qid", _QID_OPENSSH),
      ("cve", _CVE_FXNET),
  ]

  results = _retriever(store).search([_SMUGGLING_QUERY], docs=docs)

  assert sorted(_docs(results)) == sorted(docs)


def test_search_keeps_only_the_best_chunk_of_each_document(
    store: store_lib.Store,
):
  retriever = _retriever(store, **_DENSE)
  # Words of both text chunks of the fxnet description, which is the only
  # document of the fixture that is long enough to be split.
  query = "fxnet RX ring use-after-free fix by taking the ring lock"

  results = retriever.search([query], limit=30)

  docs = _docs(results)
  assert docs.count(("cve", _CVE_FXNET)) == 1
  assert len(docs) == len(set(docs))


@pytest.mark.parametrize("queries", [[], [""], ["  ", "\n"]])
def test_search_blank_queries_return_nothing(
    store: store_lib.Store, queries: list[str]
):
  assert not _retriever(store).search(queries)


def test_search_with_no_docs_to_search_returns_nothing(store: store_lib.Store):
  assert not _retriever(store).search([_SMUGGLING_QUERY], docs=[])
