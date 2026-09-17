"""The HTTP API and the static UI.

    POST /v1/analyze      runs the pipeline
    GET  /v1/search?q=    retrieval only, with per-stage scores
    GET  /v1/hosts/{id}   one host and everything detected on it
    GET  /healthz         liveness
    GET  /readyz          readiness, with the loaded artifact's provenance
    GET  /                the single-page UI (static files under /static)

The service starts even when it cannot serve: a missing or incompatible
artifact is reported by ``/readyz`` and turns the analysis endpoints into
503s that say what to do, instead of a crash loop that says nothing. An
orchestrator keeps traffic away until ``/readyz`` is green.

Endpoints are plain ``def`` functions, so FastAPI runs them in its thread
pool. Retrieval is CPU-bound (ONNX Runtime) and SQLite calls block, and
neither belongs on the event loop.

Every response carries an ``X-Request-ID`` and a strict
Content-Security-Policy. The corpus is third-party text, so the page that
renders it may load nothing but its own script and stylesheet.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
import contextlib
import logging
import pathlib
import re
import time
from typing import Annotated
import uuid

import fastapi
from fastapi import responses
from fastapi import staticfiles

import blast_radius
from blast_radius import config
from blast_radius import logging_config
from blast_radius import models
from blast_radius import services as services_lib
from blast_radius import store
from blast_radius.retrieval import retriever as retriever_lib

_LOG = logging.getLogger(__name__)

_STATIC_DIR = pathlib.Path(__file__).parent / "static"

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self';"
        " connect-src 'self'; img-src 'self' data:; base-uri 'none';"
        " form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# A caller-supplied request id is echoed into logs and headers, so only a
# conservative alphabet is accepted; anything else gets a fresh id.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_MAX_SEARCH_RESULTS = 50

_NextHandler = Callable[[fastapi.Request], Awaitable[fastapi.Response]]


def _services(request: fastapi.Request) -> services_lib.Services:
  """Returns the process's services, or answers 503 when there are none.

  Args:
    request: The incoming request.

  Raises:
    fastapi.HTTPException: With status 503 and the reason the service is
      not ready.
  """
  services: services_lib.Services | None = request.app.state.services
  if services is None:
    raise fastapi.HTTPException(
        status_code=503, detail=request.app.state.not_ready_reason
    )
  return services


_Services = Annotated[services_lib.Services, fastapi.Depends(_services)]


def create_app(
    settings: config.Settings | None = None,
    *,
    services: services_lib.Services | None = None,
) -> fastapi.FastAPI:
  """Returns the ASGI application.

  Args:
    settings: The process's settings. Read from the environment when None.
    services: A ready object graph to serve instead of building one from
      ``settings``. Tests pass one built on the fixture artifact; the caller
      keeps ownership and closes it.
  """
  settings = settings or config.Settings()

  @contextlib.asynccontextmanager
  async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
    app.state.services = services
    app.state.not_ready_reason = None
    if services is None:
      try:
        app.state.services = services_lib.build(settings)
      except (store.StoreError, retriever_lib.RetrieverConfigError) as error:
        app.state.not_ready_reason = str(error)
        _LOG.error("not ready: %s", error)
    try:
      yield
    finally:
      if services is None and app.state.services is not None:
        app.state.services.close()

  app = fastapi.FastAPI(
      title="Blast Radius Analyzer",
      version=blast_radius.__version__,
      lifespan=lifespan,
  )
  app.middleware("http")(_observe)
  app.add_exception_handler(Exception, _internal_error)
  app.mount("/static", staticfiles.StaticFiles(directory=_STATIC_DIR))

  @app.get("/", include_in_schema=False)
  def index() -> responses.FileResponse:
    return responses.FileResponse(_STATIC_DIR / "index.html")

  @app.get("/healthz")
  def healthz() -> dict[str, str]:
    return {"status": "ok"}

  @app.get("/readyz", response_model=models.Readiness)
  def readyz(
      request: fastapi.Request, response: fastapi.Response
  ) -> models.Readiness:
    ready: services_lib.Services | None = request.app.state.services
    if ready is None:
      response.status_code = 503
      return models.Readiness(
          status="not_ready", reason=request.app.state.not_ready_reason
      )
    return models.Readiness(
        status="ready", meta=ready.pipeline.meta, stats=ready.db.stats()
    )

  @app.post("/v1/analyze", response_model=models.AnalyzeResponse)
  def analyze(
      body: models.AnalyzeRequest, ready: _Services
  ) -> models.AnalyzeResponse:
    if not body.query.strip():
      raise fastapi.HTTPException(
          status_code=422, detail="query must not be blank"
      )
    return ready.pipeline.analyze(body.query)

  @app.get("/v1/search", response_model=models.SearchResponse)
  def search(
      ready: _Services,
      q: Annotated[
          str, fastapi.Query(min_length=1, max_length=models.MAX_QUERY_CHARS)
      ],
      limit: Annotated[int, fastapi.Query(ge=1, le=_MAX_SEARCH_RESULTS)] = 10,
  ) -> models.SearchResponse:
    if not q.strip():
      raise fastapi.HTTPException(status_code=422, detail="q must not be blank")
    return models.SearchResponse(
        query=q, results=ready.retriever.search([q], limit=limit)
    )

  @app.get("/v1/hosts/{host_id}", response_model=models.HostDetail)
  def host_detail(host_id: str, ready: _Services) -> models.HostDetail:
    host = ready.db.get_host(host_id)
    if host is None:
      raise fastapi.HTTPException(status_code=404, detail="no such host")
    detections = ready.db.host_detections(host_id)
    labels = {}
    for qid_id in sorted({d.qid for d in detections if d.explained}):
      qid = ready.db.get_qid(qid_id)
      if qid is not None:
        labels[qid_id] = qid.label
    return models.HostDetail(
        host=host,
        ports=ready.db.host_ports(host_id),
        detections=detections,
        qid_labels=labels,
    )

  return app


async def _observe(
    request: fastapi.Request, call_next: _NextHandler
) -> fastapi.Response:
  """Tags the request with an id, times it, and hardens the response.

  The id is taken from ``X-Request-ID`` when the caller sent a well-formed
  one, so that a trace started upstream continues here. It is put in a
  context variable, which the thread pool copies, so every log line written
  while handling the request carries it.

  Args:
    request: The incoming request.
    call_next: The rest of the application.
  """
  supplied = request.headers.get("x-request-id", "")
  request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
  request.state.request_id = request_id
  token = logging_config.request_id_var.set(request_id)
  started = time.perf_counter()
  try:
    response = await call_next(request)
  finally:
    logging_config.request_id_var.reset(token)
  duration_ms = round((time.perf_counter() - started) * 1000, 2)
  response.headers["X-Request-ID"] = request_id
  for name, value in _SECURITY_HEADERS.items():
    response.headers.setdefault(name, value)
  _LOG.info(
      "%s %s -> %d",
      request.method,
      request.url.path,
      response.status_code,
      extra={
          "request_id": request_id,
          "method": request.method,
          "path": request.url.path,
          "status": response.status_code,
          "duration_ms": duration_ms,
      },
  )
  return response


async def _internal_error(
    request: fastapi.Request, error: Exception
) -> responses.JSONResponse:
  """Answers 500 without leaking internals, and logs the traceback.

  Args:
    request: The request that failed.
    error: The unhandled exception.
  """
  request_id = getattr(request.state, "request_id", None)
  _LOG.error(
      "unhandled error", exc_info=error, extra={"request_id": request_id}
  )
  return responses.JSONResponse(
      status_code=500,
      content={"detail": "internal error", "request_id": request_id},
      headers={"X-Request-ID": request_id or "", **_SECURITY_HEADERS},
  )
