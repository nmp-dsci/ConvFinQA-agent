"""Backfill `sdk_session_cost_usd` onto sdk passes recorded before it existed.

The runner logs the price an sdk pass paid from 2026-10-15
(`cost.sdk_session_cost_usd`). Older passes carry only `sdk_cost_usd`, which
summed the CLI's running totals, but the figure they need survives in their
traces: each turn's `LLM` span holds the CLI's `total_cost_usd`, and a
conversation is one trace. Runs logged before the central server (2026-09-21)
are read from the archived store, opened read-only, and their record is the
committed snapshot; later runs are on the central server and take the metric
there. A run whose traces cannot be found is skipped and reported, never
estimated.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from convfinqa.config import REPO_ROOT, settings
from convfinqa.tracking import cost, mlflow_log, snapshot

ARCHIVE_DB = REPO_ROOT / ".mlflow" / "store" / "mlflow.db"
METRIC = "sdk_session_cost_usd"
SOURCE_TAG = "sdk_session_cost_source"


def _attr(value: Any) -> Any:
    """A span attribute as stored: JSON-encoded in the archive, plain in the client."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _sdk_model(run: dict[str, Any]) -> str | None:
    params = run.get("params") or {}
    tags = run.get("tags") or {}
    return params.get("sdk_model") or tags.get("sdk_model") or None


def _needs_backfill(run: dict[str, Any]) -> bool:
    metrics = run.get("metrics") or {}
    params = run.get("params") or {}
    return (
        params.get("runtime") == "agent_sdk"
        and "sdk_cost_usd" in metrics
        and METRIC not in metrics
    )


def archive_running_totals(db_path: Path, run_id: str) -> list[list[float]] | None:
    """Each conversation's running totals for `run_id`, from an archived store."""
    if not db_path.exists():
        return None
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        trace_ids = [
            row[0]
            for row in db.execute(
                "select request_id from trace_request_metadata "
                "where key = 'mlflow.sourceRun' and value = ?",
                (run_id,),
            )
        ]
        conversations: list[list[float]] = []
        for trace_id in trace_ids:
            totals: list[float] = []
            for (content,) in db.execute(
                "select content from spans where trace_id = ?", (trace_id,)
            ):
                attributes = json.loads(content).get("attributes") or {}
                value = _attr(attributes.get("total_cost_usd"))
                if isinstance(value, int | float):
                    totals.append(float(value))
            conversations.append(totals)
    finally:
        db.close()
    return conversations or None


def central_running_totals(run_id: str, experiment_id: str) -> list[list[float]] | None:
    """Each conversation's running totals for `run_id`, from the tracking server."""
    import mlflow

    mlflow.set_tracking_uri(mlflow_log.tracking_uri())
    traces = mlflow.search_traces(
        locations=[experiment_id], run_id=run_id, return_type="list"
    )
    conversations = [
        [
            float(value)
            for span in trace.data.spans
            if isinstance(
                value := _attr((span.attributes or {}).get("total_cost_usd")),
                int | float,
            )
        ]
        for trace in traces
    ]
    return conversations or None


def backfill_sdk_cost(
    *, archive: Path = ARCHIVE_DB, dry_run: bool = False
) -> dict[str, Any]:
    """Add the corrected cost to every sdk pass that lacks it; return what changed."""
    done: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    record = snapshot.read_snapshot()
    for run in record["runs"]:
        if not _needs_backfill(run):
            continue
        totals = archive_running_totals(archive, str(run["run_id"]))
        if totals is None:
            skipped.append(str(run.get("run_name")))
            continue
        value = cost.sdk_session_cost_usd(totals, _sdk_model(run))
        done[str(run.get("run_name"))] = {
            "sdk_cost_usd": run["metrics"]["sdk_cost_usd"],
            METRIC: value,
            "source": "archive",
        }
        run["metrics"][METRIC] = value
        run.setdefault("tags", {})[SOURCE_TAG] = "backfill:archive-traces"
    if done and not dry_run:
        path = snapshot.MLFLOW_SNAPSHOT_PATH
        path.write_text(json.dumps(record, indent=2) + "\n")

    if mlflow_log.reachable():
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_log.tracking_uri())
        experiment = client.get_experiment_by_name(settings.mlflow_experiment)
        live = mlflow_log.search_runs(limit=500) if experiment is not None else []
        for run in live:
            if not _needs_backfill(run) or str(run.get("run_name")) in done:
                continue
            totals = central_running_totals(
                str(run["run_id"]),
                str(experiment.experiment_id),  # type: ignore[union-attr]
            )
            if totals is None:
                skipped.append(str(run.get("run_name")))
                continue
            value = cost.sdk_session_cost_usd(totals, _sdk_model(run))
            done[str(run.get("run_name"))] = {
                "sdk_cost_usd": run["metrics"]["sdk_cost_usd"],
                METRIC: value,
                "source": "central",
            }
            if not dry_run:
                client.log_metric(str(run["run_id"]), METRIC, value)
                client.set_tag(str(run["run_id"]), SOURCE_TAG, "backfill:traces")
    return {"backfilled": done, "skipped": skipped, "dry_run": dry_run}
