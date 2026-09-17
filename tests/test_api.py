"""Tests for blast_radius.api."""

from collections.abc import Iterator
import os
import pathlib
import re
from typing import Any

from fastapi import testclient
import pytest

from blast_radius import api
from blast_radius import config
from blast_radius import models
from blast_radius import services as services_lib
from blast_radius import store as store_lib

# Ids in the synthetic exports; tests/fixtures/make_fixtures.py defines them.
_WORKER_B2 = "900000006"
_QID_OPENSSH = "710001"
_QID_PROXY = "710003"
_CVE_PROXY = "CVE-2099-2001"
_EXPLAINED_ON_WORKER_B2 = {"710001", "710002", "710003", "710006"}
_UNEXPLAINED_ON_WORKER_B2 = {"710901", "710902"}

# Spelt out, not imported: the policy is part of the contract with the UI.
_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self';"
    " connect-src 'self'; img-src 'self' data:; base-uri 'none';"
    " form-action 'none'; frame-ancestors 'none'"
)
_GENERATED_REQUEST_ID = re.compile(r"[0-9a-f]{32}")

_ANALYZE = {"query": f"QID {_QID_OPENSSH}"}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps the machine's own BLAST_* variables out of the settings."""
  for name in list(os.environ):
    if name.startswith("BLAST_"):
      monkeypatch.delenv(name)


def _settings(artifact_path: pathlib.Path) -> config.Settings:
  """Returns model-free settings with every stage of retrieval switched on.

  The rerank floor and margin are on ``LexicalReranker``'s 0-1 scale.

  Args:
    artifact_path: The artifact to serve.
  """
  return config.Settings(
      _env_file=None,
      artifact_path=artifact_path,
      embedding_model="hashing-256",
      rerank_model="lexical",
      enable_keyword=True,
      enable_dense=True,
      enable_rerank=True,
      abstain_rerank_floor=0.5,
      match_margin=0.2,
      llm_provider="none",
  )


@pytest.fixture(name="services")
def fixture_services(
    artifact_path: pathlib.Path,
) -> Iterator[services_lib.Services]:
  """Yields the object graph on the session's artifact, closed afterwards."""
  built = services_lib.build(_settings(artifact_path))
  yield built
  built.close()


@pytest.fixture(name="client")
def fixture_client(
    artifact_path: pathlib.Path, services: services_lib.Services
) -> Iterator[testclient.TestClient]:
  """Yields a client of a started app that serves ``services``."""
  app = api.create_app(_settings(artifact_path), services=services)
  with testclient.TestClient(app) as started:
    yield started


@pytest.fixture(name="client_without_artifact")
def fixture_client_without_artifact(
    tmp_path: pathlib.Path,
) -> Iterator[testclient.TestClient]:
  """Yields a client of an app whose settings point at no artifact."""
  app = api.create_app(_settings(tmp_path / "missing.sqlite"))
  with testclient.TestClient(app) as started:
    yield started


# ---------------------------------------------------------------------------
# Liveness and readiness
# ---------------------------------------------------------------------------


def test_healthz_answers_ok(client: testclient.TestClient):
  response = client.get("/healthz")

  assert response.status_code == 200
  assert response.json() == {"status": "ok"}


def test_readyz_reports_the_loaded_artifact_and_models(
    client: testclient.TestClient,
):
  response = client.get("/readyz")

  assert response.status_code == 200
  readiness = models.Readiness.model_validate(response.json())
  assert readiness.status == "ready"
  assert readiness.reason is None
  assert readiness.meta is not None
  assert readiness.meta.embedding_model == "hashing-256"
  assert readiness.meta.artifact_built_at is not None
  assert readiness.stats is not None
  assert readiness.stats.hosts == 11
  assert readiness.stats.explained_qids == 6


def test_app_without_artifact_starts_and_says_why_it_is_not_ready(
    client_without_artifact: testclient.TestClient,
):
  alive = client_without_artifact.get("/healthz")
  ready = client_without_artifact.get("/readyz")

  assert alive.status_code == 200
  assert ready.status_code == 503
  readiness = models.Readiness.model_validate(ready.json())
  assert readiness.status == "not_ready"
  assert readiness.reason is not None
  assert "missing.sqlite" in readiness.reason
  assert readiness.meta is None


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("POST", "/v1/analyze", _ANALYZE),
        ("GET", "/v1/search?q=openssh", None),
        ("GET", f"/v1/hosts/{_WORKER_B2}", None),
    ],
)
def test_analysis_endpoints_without_artifact_answer_503_with_the_reason(
    client_without_artifact: testclient.TestClient,
    method: str,
    path: str,
    body: dict[str, str] | None,
):
  response = client_without_artifact.request(method, path, json=body)

  assert response.status_code == 503
  assert "missing.sqlite" in response.json()["detail"]


def test_app_builds_its_own_services_and_closes_them_on_shutdown(
    artifact_path: pathlib.Path,
):
  app = api.create_app(_settings(artifact_path))

  with testclient.TestClient(app) as started:
    ready = started.get("/readyz")

  assert ready.status_code == 200
  with pytest.raises(store_lib.StoreError, match="closed"):
    app.state.services.db.stats()


def test_app_leaves_services_it_was_given_open(
    artifact_path: pathlib.Path, services: services_lib.Services
):
  app = api.create_app(_settings(artifact_path), services=services)

  with testclient.TestClient(app):
    pass

  assert services.db.stats().hosts == 11


# ---------------------------------------------------------------------------
# POST /v1/analyze
# ---------------------------------------------------------------------------


def test_analyze_returns_the_pipeline_response(client: testclient.TestClient):
  response = client.post("/v1/analyze", json=_ANALYZE)

  assert response.status_code == 200
  analysis = models.AnalyzeResponse.model_validate(response.json())
  assert analysis.status == "matched"
  assert [match.qid for match in analysis.matches] == [_QID_OPENSSH]
  assert len(analysis.hosts) == 8
  assert len(analysis.inactive_hosts) == 1
  assert analysis.answer is None


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": " \n\t "},
        {"query": "x" * (models.MAX_QUERY_CHARS + 1)},
    ],
    ids=["missing", "empty", "blank", "over-long"],
)
def test_analyze_unusable_query_is_rejected_with_422(
    client: testclient.TestClient, body: dict[str, str]
):
  response = client.post("/v1/analyze", json=body)

  assert response.status_code == 422


def test_unhandled_error_answers_500_with_the_request_id_and_no_internals(
    artifact_path: pathlib.Path,
    services: services_lib.Services,
    monkeypatch: pytest.MonkeyPatch,
):
  def explode(query: str) -> models.AnalyzeResponse:
    raise RuntimeError(f"secret detail about {query}")

  monkeypatch.setattr(services.pipeline, "analyze", explode)
  app = api.create_app(_settings(artifact_path), services=services)

  with testclient.TestClient(app, raise_server_exceptions=False) as started:
    response = started.post(
        "/v1/analyze", json=_ANALYZE, headers={"X-Request-ID": "upstream-42"}
    )

  assert response.status_code == 500
  assert response.json() == {
      "detail": "internal error",
      "request_id": "upstream-42",
  }
  assert "secret" not in response.text
  assert "Traceback" not in response.text
  assert response.headers["x-request-id"] == "upstream-42"
  assert response.headers["content-security-policy"] == _CONTENT_SECURITY_POLICY


# ---------------------------------------------------------------------------
# GET /v1/search
# ---------------------------------------------------------------------------


def test_search_returns_chunks_with_the_score_of_every_stage(
    client: testclient.TestClient,
):
  response = client.get(
      "/v1/search", params={"q": "Trellis proxy forwarded headers"}
  )

  assert response.status_code == 200
  found = models.SearchResponse.model_validate(response.json())
  assert found.query == "Trellis proxy forwarded headers"
  best = found.results[0]
  assert best.chunk.doc_id in {_QID_PROXY, _CVE_PROXY}
  assert best.keyword_rank is not None
  assert best.dense_rank is not None
  assert best.dense_score is not None
  assert best.fused_score is not None
  assert best.rerank_score == pytest.approx(1.0)


def test_search_limit_caps_the_number_of_results(
    client: testclient.TestClient,
):
  response = client.get("/v1/search", params={"q": "proxy", "limit": 1})

  assert response.status_code == 200
  assert len(response.json()["results"]) == 1


@pytest.mark.parametrize(
    "params, status",
    [
        ({"q": "proxy", "limit": 0}, 422),
        ({"q": "proxy", "limit": 50}, 200),
        ({"q": "proxy", "limit": 51}, 422),
        ({"q": ""}, 422),
        ({}, 422),
    ],
    ids=["limit 0", "limit 50", "limit 51", "empty q", "no q"],
)
def test_search_validates_its_parameters(
    client: testclient.TestClient, params: dict[str, Any], status: int
):
  response = client.get("/v1/search", params=params)

  assert response.status_code == status


# ---------------------------------------------------------------------------
# GET /v1/hosts/{id}
# ---------------------------------------------------------------------------


def test_host_detail_lists_ports_detections_and_labels(
    client: testclient.TestClient,
):
  response = client.get(f"/v1/hosts/{_WORKER_B2}")

  assert response.status_code == 200
  detail = models.HostDetail.model_validate(response.json())
  assert detail.host.name == "fx-k8s-worker-b2"
  assert [(port.port, port.protocol) for port in detail.ports] == [
      (22, "TCP"),
      (80, "TCP"),
      (111, "TCP"),
      (111, "UDP"),
      (8080, "TCP"),
  ]
  # Seven detections: the scanner reported the proxy finding twice.
  assert len(detail.detections) == 7
  explained = {d.qid for d in detail.detections if d.explained}
  unexplained = {d.qid for d in detail.detections if not d.explained}
  assert explained == _EXPLAINED_ON_WORKER_B2
  assert unexplained == _UNEXPLAINED_ON_WORKER_B2
  assert set(detail.qid_labels) == _EXPLAINED_ON_WORKER_B2
  assert all(detail.qid_labels.values())


def test_host_detail_unknown_host_is_404(client: testclient.TestClient):
  response = client.get("/v1/hosts/no-such-host")

  assert response.status_code == 404
  assert response.json() == {"detail": "no such host"}


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, path, status",
    [
        ("GET", "/healthz", 200),
        ("GET", "/", 200),
        ("GET", "/static/app.js", 200),
        ("GET", "/v1/hosts/no-such-host", 404),
        ("POST", "/v1/analyze", 422),
    ],
)
def test_every_response_carries_a_request_id_and_the_security_headers(
    client: testclient.TestClient, method: str, path: str, status: int
):
  response = client.request(method, path)

  assert response.status_code == status
  assert _GENERATED_REQUEST_ID.fullmatch(response.headers["x-request-id"])
  assert response.headers["content-security-policy"] == _CONTENT_SECURITY_POLICY
  assert response.headers["x-content-type-options"] == "nosniff"
  assert response.headers["referrer-policy"] == "no-referrer"


def test_well_formed_request_id_is_echoed(client: testclient.TestClient):
  response = client.get("/healthz", headers={"X-Request-ID": "trace_7.a-b"})

  assert response.headers["x-request-id"] == "trace_7.a-b"


@pytest.mark.parametrize(
    "supplied",
    ["has space", "semi;colon", "<script>", "x" * 65],
    ids=["space", "punctuation", "markup", "too long"],
)
def test_malformed_request_id_is_replaced(
    client: testclient.TestClient, supplied: str
):
  response = client.get("/healthz", headers={"X-Request-ID": supplied})

  assert _GENERATED_REQUEST_ID.fullmatch(response.headers["x-request-id"])


# ---------------------------------------------------------------------------
# The UI
# ---------------------------------------------------------------------------


def test_index_serves_the_page_that_loads_the_static_script(
    client: testclient.TestClient,
):
  page = client.get("/")
  script = client.get("/static/app.js")

  assert page.status_code == 200
  assert page.headers["content-type"].startswith("text/html")
  assert 'src="/static/app.js"' in page.text
  assert script.status_code == 200
  assert "javascript" in script.headers["content-type"]
