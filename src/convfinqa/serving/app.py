"""FastAPI application factory.

One build, two deployments. Nothing here is conditional on which one it is except
`/healthz`, which reports the mode so the *frontend* can configure itself — the
alternative, a separate demo build, is how a demo drifts away from the product it
is supposed to demonstrate.

In production this process also serves the built SPA, so there is one container,
one origin, and no CORS or proxy configuration in the deployment at all.

`--workers 1` is required: session state lives in this process's memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from convfinqa.config import REPO_ROOT, settings
from convfinqa.llm import demo_mode_enabled
from convfinqa.serving.limits import (
    InFlightLimiter,
    RateLimited,
    SlidingWindowRateLimiter,
    client_key,
)
from convfinqa.serving.models import HealthResponse
from convfinqa.serving.research import ResearchRunner
from convfinqa.serving.routes import admin, chat, evaluation, traces
from convfinqa.serving.sessions import SessionStore

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

# Paths the rate limiter applies to. Read-only browsing is deliberately exempt:
# a visitor scrolling the answers explorer should never be throttled, and those
# routes read cached frames rather than spending anything per request.
_RATE_LIMITED_PREFIXES = ("/sessions",)


def create_app(
    *,
    session_ttl_seconds: int = 1800,
    eviction_interval_seconds: int = 60,
) -> FastAPI:
    """Build the application."""
    store = SessionStore(
        ttl_seconds=session_ttl_seconds, valid_reports=set(chat.REPORT_IDS)
    )
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        housekeeping = asyncio.create_task(
            _housekeeping_loop(store, rate_limiter, eviction_interval_seconds)
        )
        try:
            yield
        finally:
            housekeeping.cancel()
            with suppress(asyncio.CancelledError):
                await housekeeping

    app = FastAPI(
        title="ConvFinQA Agent",
        description=(
            "Conversational financial QA over SEC filings, with its own "
            "evaluation, tracing and experiment-tracking surface."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            o.strip() for o in settings.frontend_origins.split(",") if o.strip()
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*", "x-owner-token"],
    )

    app.state.session_store = store
    app.state.session_ttl_seconds = session_ttl_seconds
    app.state.eviction_interval_seconds = eviction_interval_seconds
    app.state.inflight = InFlightLimiter(settings.max_inflight_turns)
    app.state.rate_limiter = rate_limiter
    app.state.research = ResearchRunner()

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    logfire.instrument_fastapi(app)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Any) -> Any:
        """Per-IP sliding window on the turn-producing routes."""
        if request.method == "POST" and request.url.path.startswith(
            _RATE_LIMITED_PREFIXES
        ):
            try:
                rate_limiter.check(client_key(request))
            except RateLimited as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=exc.headers or {},
                )
        response: Any = await call_next(request)
        return response

    @app.get("/healthz")
    async def healthz() -> HealthResponse:
        """Liveness plus the mode and bundle the frontend configures itself from."""
        from convfinqa.serving.demo_pack import replay
        from convfinqa.tracking import registry
        from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id

        fingerprint = bundle_fingerprint()
        return HealthResponse(
            ok=True,
            mode="demo" if demo_mode_enabled() else "dev",
            champion=registry.champion(),
            bundle_id=bundle_id(fingerprint),
            bundle=fingerprint,
            demo_reports=len(replay.packed_reports()),
        )

    app.include_router(chat.router)
    app.include_router(evaluation.router)
    app.include_router(traces.router)
    app.include_router(admin.router)

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from this process, when a build is present.

    Absent in dev — Vite serves the frontend there and proxies the API here — so
    this is a no-op on a working tree that has never run `npm run build`.
    """
    if not FRONTEND_DIST.is_dir():
        return

    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = FRONTEND_DIST / "index.html"
    root = FRONTEND_DIST.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        """Serve a real file when one exists, else the SPA shell.

        Registered last so every API route above wins; the client-side router
        owns whatever is left. The containment check keeps `../` out of a route
        that otherwise maps user input straight onto the filesystem.
        """
        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(index)


async def _housekeeping_loop(
    store: SessionStore,
    rate_limiter: SlidingWindowRateLimiter,
    interval_seconds: int,
) -> None:
    """Evict idle sessions and prune the rate-limiter's per-IP table."""
    while True:
        await asyncio.sleep(interval_seconds)
        store.evict_expired()
        rate_limiter.prune()


app = create_app()
