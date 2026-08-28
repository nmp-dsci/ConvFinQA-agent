"""MLflow instrumentation: eval runs, GEPA runs, and s7 rounds.

The design rule that makes history trustworthy: **logging lives inside the
runners, not beside them.** An operator who forgets to wrap a run in a logging
call produces an unrecorded result, and an experiment history with holes in it is
worse than none — you cannot tell a gap from a failure. So `evaluation.runner`,
`optimization.gepa` and `diagnosis.harness` each open a run themselves, and every
future run is captured by construction.

Backend is a local `file:` store. No server to run, secure, or pay for, and the
demo reads a committed export of it instead (see `snapshot.py`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from convfinqa.config import MLRUNS_DIR, settings
from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id

log = logging.getLogger("convfinqa.tracking")


def tracking_uri() -> str:
    """Resolve the tracking URI: explicit setting, else the repo-local SQLite store.

    SQLite rather than the older `file:./mlruns` layout, which MLflow deprecated
    in February 2026. It is still a single local file with no server to run,
    secure, or pay for — the property that mattered about the file store is kept,
    without building on something already announced as going away.
    """
    if settings.mlflow_tracking_uri:
        return settings.mlflow_tracking_uri
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"


def artifacts_dir() -> Path:
    """Where run artifacts are written. Anchored, not cwd-relative."""
    return MLRUNS_DIR / "artifacts"


def _mlflow() -> Any:
    """Import mlflow lazily and point it at the configured store.

    Lazy because importing mlflow costs ~2s and pulls in a large dependency tree;
    the API process should not pay that at startup just to serve /reports.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    root = artifacts_dir()
    root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
        mlflow.create_experiment(
            settings.mlflow_experiment, artifact_location=root.as_uri()
        )
    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


def available() -> bool:
    """True when MLflow can be imported and a store is reachable."""
    try:
        _mlflow()
    except Exception:  # noqa: BLE001 — tracking is never load-bearing for serving
        return False
    return True


@contextlib.contextmanager
def run(
    name: str,
    *,
    kind: str,
    version: str | None = None,
    overlay: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Open an MLflow run stamped with the bundle fingerprint.

    Yields a small recorder with `.metric()`, `.artifact()` and `.dict_artifact()`,
    or a no-op recorder when MLflow is unavailable — the caller's code path is the
    same either way, which is what keeps the instrumentation out of the way of the
    thing being instrumented.
    """
    try:
        mlflow = _mlflow()
    except Exception:  # noqa: BLE001
        yield _NullRecorder()
        return

    fingerprint = bundle_fingerprint(version=version, overlay=overlay)
    with mlflow.start_run(run_name=name) as active:
        mlflow.set_tags(
            {
                "kind": kind,
                "bundle_id": bundle_id(fingerprint),
                **(tags or {}),
            }
        )
        mlflow.log_params({k: str(v) for k, v in fingerprint.items() if v is not None})
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
        yield _Recorder(mlflow, active.info.run_id)


class _Recorder:
    """Thin wrapper over the mlflow module, scoped to one active run.

    A failed write is logged rather than silently dropped: the whole point of
    logging inside the runners is that a gap in the history should mean the run
    never happened, not that a metric quietly vanished from one that looks
    complete.
    """

    def __init__(self, mlflow: Any, run_id: str) -> None:
        self._mlflow = mlflow
        self.run_id = run_id

    def metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log a single metric, optionally as a step in a trajectory."""
        try:
            self._mlflow.log_metric(key, float(value), step=step)
        except Exception:
            log.warning(
                "mlflow: failed to log metric %r for run %s",
                key,
                self.run_id,
                exc_info=True,
            )

    def metrics(self, values: dict[str, float]) -> None:
        """Log several metrics at once."""
        for key, value in values.items():
            self.metric(key, value)

    def artifact(self, path: Path | str) -> None:
        """Attach a file to the run, if it exists."""
        candidate = Path(path)
        if not candidate.exists():
            return
        try:
            self._mlflow.log_artifact(str(candidate))
        except Exception:
            log.warning(
                "mlflow: failed to log artifact %s for run %s",
                candidate,
                self.run_id,
                exc_info=True,
            )

    def dict_artifact(self, name: str, payload: dict[str, Any]) -> None:
        """Attach a JSON document to the run."""
        try:
            self._mlflow.log_dict(payload, name)
        except Exception:
            log.warning(
                "mlflow: failed to log dict artifact %r for run %s",
                name,
                self.run_id,
                exc_info=True,
            )


class _NullRecorder:
    """Recorder used when MLflow is unavailable. Every method is a no-op."""

    run_id = ""

    def metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Discard the metric."""

    def metrics(self, values: dict[str, float]) -> None:
        """Discard the metrics."""

    def artifact(self, path: Path | str) -> None:
        """Discard the artifact."""

    def dict_artifact(self, name: str, payload: dict[str, Any]) -> None:
        """Discard the artifact."""


def search_runs(limit: int = 200) -> list[dict[str, Any]]:
    """Return recent runs as plain dicts: params, metrics, tags, status.

    Shaped for the admin API rather than for pandas — the frontend wants a list
    of records, and the caller should not have to know MLflow's frame layout.
    """
    try:
        mlflow = _mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri())
        experiment = client.get_experiment_by_name(settings.mlflow_experiment)
        if experiment is None:
            return []
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    except Exception:  # noqa: BLE001
        return []

    out: list[dict[str, Any]] = []
    for item in runs:
        info = item.info
        out.append(
            {
                "run_id": info.run_id,
                "run_name": item.data.tags.get("mlflow.runName", ""),
                "kind": item.data.tags.get("kind", ""),
                "bundle_id": item.data.tags.get("bundle_id", ""),
                "status": info.status,
                "start_time": info.start_time,
                "end_time": info.end_time,
                "params": dict(item.data.params),
                "metrics": dict(item.data.metrics),
                "tags": {
                    k: v
                    for k, v in item.data.tags.items()
                    if not k.startswith("mlflow.")
                },
            }
        )
    del mlflow
    return out


def artifacts_root() -> Path:
    """Filesystem root of the local store, for backfill and export."""
    return MLRUNS_DIR


def env_summary() -> dict[str, Any]:
    """Where tracking is pointing, for /admin/experiments to report honestly."""
    return {
        "tracking_uri": tracking_uri(),
        "experiment": settings.mlflow_experiment,
        "registered_model": settings.registered_model_name,
        "available": available(),
        "store_exists": artifacts_root().exists(),
    }


def dumps(payload: Any) -> str:
    """JSON with stable ordering, for artifacts that get diffed across runs."""
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
