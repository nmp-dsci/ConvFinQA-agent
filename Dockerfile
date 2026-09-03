# The demo image: the public, read-only deployment.
#
# One container, no database, no secrets. Everything a visitor browses is baked
# in at build time — the dataset, the committed prediction CSVs, the recorded
# demo pack, the exported experiment snapshot — so what a given image serves is
# exactly what its commit says it serves.
#
# DEMO_MODE is set here rather than in infrastructure, deliberately: the image
# itself is the thing that cannot make model calls, so no Terraform mistake or
# console edit can turn a public URL into a billable one. There is no API key in
# the environment for it to use either.
#
#   docker build -t convfinqa-demo .
#   docker run --rm -p 8080:8080 convfinqa-demo    # no env file, no keys

# ── Stage 1: frontend bundle ────────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: runtime ────────────────────────────────────────────────────
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, as their own layer: they change far less often than the
# code, so an ordinary commit reuses this layer entirely.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
    && uv pip install --system -r requirements.txt \
    && rm requirements.txt

# Code, UI, and the frozen artifacts the read-only routes serve.
COPY src/ src/
COPY scripts/ scripts/
COPY --from=frontend /build/dist frontend/dist
COPY data/ data/
COPY evaluation/predictions/ evaluation/predictions/
COPY evaluation/diagnostics/ evaluation/diagnostics/
# The eval-loop split manifest: /eval/dataset reads it, and without it that
# route answered 500 on the public demo (2026-09-03).
COPY evaluation/splits/ evaluation/splits/
COPY evaluation/mlflow_snapshot.json evaluation/registry.json evaluation/
COPY runs/ runs/

# The build's git SHA, passed in by the deploy workflow. The container has no
# `.git`, so this is how an answer stays attributable to the exact build that
# produced it.
ARG CODE_SHA=unknown
ENV CONVFINQA_CODE_SHA=$CODE_SHA

ENV DEMO_MODE=1 \
    TRACE_CAPTURE_ENABLED=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8080

# --workers 1 is load-bearing, not a default: session state lives in process
# memory, so a second worker would lose half the conversations.
CMD ["python", "-m", "uvicorn", "convfinqa.serving.app:create_app", \
     "--factory", "--workers", "1", "--host", "0.0.0.0", "--port", "8080"]
