"""The admin portal API: experiments, registry, promotion, research, rules.

Reads resolve through one seam — `_experiments_source()` — which returns the live
MLflow store in dev and the committed snapshot in demo. The frontend never learns
which, so the experiments tab is the same component in both deployments.

Writes (promote, research launch) are owner-token gated *and* demo-gated. Both
checks, not either: the token could in principle be set on the demo, and a
promotion is not something a public URL should be able to do even with it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from convfinqa.config import DIAGNOSTICS_DIR, settings
from convfinqa.llm import demo_mode_enabled
from convfinqa.serving.limits import require_owner
from convfinqa.tracking import mlflow_log, registry, snapshot
from convfinqa.tracking.comparator import (
    accuracy,
    available_versions,
    compare,
    load_predictions,
    program_accuracy,
)

router = APIRouter(prefix="/admin")


def _demo_write_blocked() -> None:
    """Refuse state-changing admin actions on the public demo."""
    if demo_mode_enabled():
        raise HTTPException(
            status_code=501,
            detail={
                "code": "not_available_demo",
                "message": (
                    "This is the read-only demo. Promotion and research rounds "
                    "run in the dev deployment."
                ),
            },
        )


def _experiments_source() -> dict[str, Any]:
    """Live tracking store in dev, committed snapshot in demo.

    Also falls back to the snapshot in dev when no local `mlruns/` has been
    created yet — a fresh clone should still show the backfilled history rather
    than an empty table.
    """
    if not demo_mode_enabled():
        runs = mlflow_log.search_runs(limit=500)
        if runs:
            return {
                "source": "live",
                "runs": runs,
                "registry": registry.summary(),
                "versions": snapshot.read_snapshot().get("versions", []),
            }
    payload = snapshot.read_snapshot()
    return {
        "source": "snapshot",
        "runs": payload.get("runs", []),
        "registry": payload.get("registry", registry.summary()),
        "versions": payload.get("versions", []),
        "exported_at": payload.get("exported_at"),
    }


@router.get("/experiments")
async def list_experiments() -> dict[str, Any]:
    """Every tracked run, plus the registry and per-version accuracy."""
    payload = _experiments_source()
    payload["tracking"] = mlflow_log.env_summary()
    payload["mode"] = "demo" if demo_mode_enabled() else "dev"
    return payload


@router.get("/experiments/{run_id}")
async def get_experiment(run_id: str) -> dict[str, Any]:
    """One run in full."""
    runs: list[dict[str, Any]] = _experiments_source()["runs"]
    for run in runs:
        if run.get("run_id") == run_id:
            return run
    raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")


@router.get("/registry")
async def get_registry() -> dict[str, Any]:
    """Bundle versions, aliases, and the append-only promotion history."""
    payload = _experiments_source()
    reg: dict[str, Any] = payload.get("registry") or registry.summary()
    reg["mode"] = "demo" if demo_mode_enabled() else "dev"
    reg["can_promote"] = not demo_mode_enabled() and settings.owner_token is not None
    return reg


@router.get("/compare")
async def compare_versions(
    baseline: str = Query(...),
    candidate: str = Query(...),
) -> dict[str, Any]:
    """Question-by-question diff of two versions, with the pass→fail flip list."""
    try:
        return compare(baseline, candidate).as_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/versions")
async def list_versions() -> list[dict[str, Any]]:
    """Versions with committed predictions, with both accuracies side by side.

    `exe_acc` is the headline: did the final number come out right. `prog_acc`
    is the one that says how much of it was reasoning — the ConvFinQA paper
    reports both, and a version that gains execution accuracy while losing
    program accuracy has learned the answers rather than the method. Showing
    only the first would hide exactly the failure this system is meant to
    surface.

    Computed from the committed CSVs on the fly. No API calls, and no cached
    number to go stale against the predictions it describes.
    """
    out: list[dict[str, Any]] = []
    for version in available_versions():
        try:
            df = load_predictions(version)
        except (FileNotFoundError, ValueError):
            continue
        programs = program_accuracy(df)
        out.append(
            {
                "version": version,
                "exe_acc": round(accuracy(df), 6),
                "prog_acc": programs["program_accuracy"],
                "n_questions": int(len(df)),
                "n_program_turns": int(programs["n_program_turns"]),
                "n_program_correct": int(programs["n_program_correct"]),
            }
        )
    return out


class PromoteRequest(BaseModel):
    """Body for a promotion attempt."""

    version: str
    force: bool = False


@router.post("/registry/promote", dependencies=[Depends(require_owner)])
async def promote_version(body: PromoteRequest) -> dict[str, Any]:
    """Promote a version to champion, if the comparator allows it.

    Returns 409 rather than 200-with-a-flag when the comparator refuses: a
    refused promotion is a failed request, and the UI should not have to inspect
    a success payload to discover that nothing happened.
    """
    _demo_write_blocked()
    try:
        outcome = registry.promote(body.version, force=body.force, actor="admin-api")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not outcome.promoted:
        raise HTTPException(status_code=409, detail=outcome.as_dict())
    return outcome.as_dict()


class ChallengerRequest(BaseModel):
    """Body for pointing the challenger alias."""

    version: str


@router.post("/registry/challenger", dependencies=[Depends(require_owner)])
async def set_challenger(body: ChallengerRequest) -> dict[str, Any]:
    """Point the challenger alias at a registered version."""
    _demo_write_blocked()
    try:
        registry.set_alias(registry.CHALLENGER, body.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return registry.summary()


@router.get("/rules")
async def list_rules(variant: str = "") -> dict[str, Any]:
    """The s7 rule stores per agent — rule text, verified case, attempts.

    Currently only inspectable by reading JSONL by hand; this is the same data,
    joined to its attempt history.
    """
    target = variant or settings.variant
    out: dict[str, Any] = {"variant": target, "agents": {}}
    for agent in ("triage", "preprocess", "retriever", "calculator"):
        out["agents"][agent] = {
            "rules": _read_jsonl(DIAGNOSTICS_DIR / f"rules_{agent}_{target}.jsonl"),
            "attempts": _read_jsonl(
                DIAGNOSTICS_DIR / f"rule_attempts_{agent}_{target}.jsonl"
            ),
        }
    return out


@router.get("/rules/variants")
async def list_rule_variants() -> list[str]:
    """Variants that have an s7 rule store on disk."""
    if not DIAGNOSTICS_DIR.exists():
        return []
    return sorted(
        {
            path.stem.split("_", 2)[-1]
            for path in DIAGNOSTICS_DIR.glob("rules_*_*.jsonl")
        }
    )


class ResearchRequest(BaseModel):
    """Body for launching a research round."""

    kind: str = "s7"
    limit: int = 5
    retry_n: int = 1
    variant: str = ""
    skip_regression: bool = True


@router.get("/research/status")
async def research_status(request: Request) -> dict[str, Any]:
    """Current round and recent history. Readable in demo; launch is not."""
    status: dict[str, Any] = dict(request.app.state.research.status())
    status["can_launch"] = not demo_mode_enabled() and settings.owner_token is not None
    status["mode"] = "demo" if demo_mode_enabled() else "dev"
    return status


@router.post("/research/start", dependencies=[Depends(require_owner)])
async def research_start(body: ResearchRequest, request: Request) -> dict[str, Any]:
    """Launch an s7 round or a GEPA smoke run as a background job."""
    _demo_write_blocked()
    runner = request.app.state.research
    try:
        job = await runner.start(
            body.kind,
            {
                "limit": body.limit,
                "retry_n": body.retry_n,
                "variant": body.variant,
                "skip_regression": body.skip_regression,
            },
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "research_busy", "message": str(exc)}
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload: dict[str, Any] = job.as_dict()
    return payload


@router.post("/research/cancel", dependencies=[Depends(require_owner)])
async def research_cancel(request: Request) -> dict[str, Any]:
    """Terminate the running round."""
    _demo_write_blocked()
    cancelled = await request.app.state.research.cancel()
    return {"cancelled": cancelled}


@router.get("/research/stream")
async def research_stream(request: Request) -> StreamingResponse:
    """Live progress for the running round, over the same SSE the chat uses."""
    runner = request.app.state.research
    queue = runner.subscribe()

    async def gen() -> AsyncIterator[str]:
        try:
            yield _frame({"event": "status", **runner.status()})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keep-alive: App Runner drops idle connections, and a
                    # research round can be quiet for minutes at a time.
                    yield ": keep-alive\n\n"
                    continue
                yield _frame(event)
        finally:
            runner.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, default=str) + "\n\n"


def _read_jsonl(path: Any) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
