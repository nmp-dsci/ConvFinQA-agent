"""What the demo image must carry, and what a page URL must resolve to.

Two failures this file exists to stop, both of which happened on the public
deployment and neither of which fails a build, a route, or a test that only
exercises dev:

* **A committed artifact left out of the Dockerfile.** `evaluation/story.json`
  and `evaluation/readiness.json` were never copied in, so `/eval/campaigns`
  answered "no campaign has been recorded yet" and `/eval/readiness` 404'd. The
  routes are written to degrade rather than 500, which is right — and which is
  exactly why nothing shouted. The Runtimes page, the campaign track, the
  progression chart, the judge panel, the readiness strip and four of the
  landing HUD tiles all rendered their empty state on a deployment whose whole
  job is to show them.

* **A page URL that the API answers instead.** `/admin` is both an API prefix
  and a UI route prefix, and `GET /admin/experiments` is spelled the same on
  both sides. FastAPI wins, so opening that page in the container returned raw
  JSON. Dev hides it: Vite's proxy hands document requests back to the client
  router, so the collision is invisible until it is public.
"""

# ruff: noqa: D103

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from convfinqa.config import REPO_ROOT
from convfinqa.serving import app as api_app

DOCKERFILE = REPO_ROOT / "Dockerfile"

# Every committed artifact a read-only route opens, with the route that opens
# it. Add a row here when a route learns to read a new committed file — the
# point of the test is that the list and the image cannot drift apart.
SHIPPED_ARTIFACTS: dict[str, str] = {
    "evaluation/mlflow_snapshot.json": "GET /admin/experiments",
    "evaluation/registry.json": "GET /admin/registry, GET /healthz",
    "evaluation/story.json": "GET /eval/campaigns",
    "evaluation/readiness.json": "GET /eval/readiness",
    "evaluation/traces_snapshot.jsonl.gz": "GET /traces, GET /metrics/production",
    "evaluation/splits/": "GET /eval/dataset",
    "evaluation/predictions/": "GET /eval/answers, GET /eval/loop-runs",
    "evaluation/diagnostics/": "GET /admin/rules",
    "evaluation/judge/": "the judge's own record",
    "data/": "GET /reports",
    "runs/": "the GEPA overlay the pipeline runtime loads",
}


def _copied_paths() -> set[str]:
    """Every source path the Dockerfile's `COPY` lines carry into the image."""
    text = DOCKERFILE.read_text()
    # Fold `\`-continued lines so a multi-path COPY reads as one statement.
    text = re.sub(r"\\\n\s*", " ", text)
    copied: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        parts = stripped.split()[1:]
        if parts and parts[0].startswith("--from"):
            parts = parts[1:]
        copied.update(parts[:-1])  # the last token is the destination
    return copied


@pytest.mark.parametrize(("artifact", "reader"), sorted(SHIPPED_ARTIFACTS.items()))
def test_artifact_is_committed_and_shipped(artifact: str, reader: str) -> None:
    assert (REPO_ROOT / artifact).exists(), f"{artifact} is missing from the repo"
    assert artifact in _copied_paths(), (
        f"{artifact} is read by {reader} but the Dockerfile never copies it, "
        "so that route serves its empty state on the public demo"
    )


def test_dockerignore_does_not_exclude_a_shipped_artifact() -> None:
    ignored = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for artifact in SHIPPED_ARTIFACTS:
        assert artifact.rstrip("/") not in ignored
        assert artifact not in ignored


# ---------------------------------------------------------------------------
# The `/admin` page-vs-API collision
# ---------------------------------------------------------------------------


@pytest.fixture
def container_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The app as the container builds it: an SPA bundle mounted beside the API."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ConvFinQA Agent</title>")
    monkeypatch.setattr(api_app, "FRONTEND_DIST", dist)
    return TestClient(api_app.create_app())


def test_admin_page_url_serves_the_spa_for_a_browser(
    container_client: TestClient,
) -> None:
    """A navigation to the Experiments *page* must not be answered by the API."""
    response = container_client.get(
        "/admin/experiments",
        headers={
            "accept": "text/html,application/xhtml+xml",
            "sec-fetch-dest": "document",
        },
    )
    assert response.status_code == 200
    assert "ConvFinQA Agent" in response.text
    assert not response.text.lstrip().startswith("{")


def test_admin_api_url_still_serves_json_for_a_client(
    container_client: TestClient,
) -> None:
    """The same path fetched as an API call is unchanged: `fetch()` sends `*/*`."""
    response = container_client.get("/admin/experiments", headers={"accept": "*/*"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_sse_stream_is_not_mistaken_for_a_page(container_client: TestClient) -> None:
    """`text/event-stream` is a client, not a navigation."""
    assert not api_app._wants_document("empty", "text/event-stream")
    assert not api_app._wants_document(None, "text/event-stream")
    assert api_app._wants_document("document", "*/*")
