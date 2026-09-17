# syntax=docker/dockerfile:1.7
#
# Two stages: `build` resolves dependencies with uv and downloads the
# embedding and reranking models, `runtime` carries only the virtualenv, the
# package and the models. The container therefore starts with no network
# access and runs as a non-root user on a read-only filesystem.
#
# The index artifact is not baked in: it is built from scanner exports that
# never enter the image, and is mounted at /data (see compose.yaml).

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, then the models, then the package: editing the code
# reinstalls nothing and downloads nothing.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra openai --extra anthropic \
    --no-install-project

# The defaults of BLAST_EMBEDDING_MODEL and BLAST_RERANK_MODEL in config.py.
# Override both here and at run time to ship different models.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
RUN /app/.venv/bin/python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/opt/models'); \
TextCrossEncoder('${RERANK_MODEL}', cache_dir='/opt/models')"

COPY blast_radius ./blast_radius
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra openai --extra anthropic --no-editable


FROM python:3.12-slim AS runtime
RUN useradd --system --uid 10001 --no-create-home blast
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/models /opt/models
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    BLAST_MODEL_CACHE_DIR=/opt/models \
    BLAST_ARTIFACT_PATH=/data/index.sqlite \
    BLAST_HOST=0.0.0.0 \
    BLAST_PORT=8000
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["blast-radius", "serve"]
