"""Export the tracking store to a committed JSON snapshot.

The demo container has no MLflow store: `mlruns/` is dev state and stays
gitignored. But the experiments tab is one of the surfaces the demo exists to
show, so what ships is a compact export — runs, params, metrics, registry,
promotion history, no heavy artifacts — baked into the image at build time.

The admin API reads the live store in dev and this file in demo, behind one
interface, so the frontend never learns which mode it is in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import MLFLOW_SNAPSHOT_PATH
from convfinqa.tracking import mlflow_log, registry
from convfinqa.tracking.comparator import accuracy, available_versions, load_predictions

SNAPSHOT_VERSION = 1


def build_snapshot() -> dict[str, Any]:
    """Assemble the snapshot payload from the live store and committed CSVs."""
    versions: list[dict[str, Any]] = []
    for version in available_versions():
        try:
            df = load_predictions(version)
        except (FileNotFoundError, ValueError):
            continue
        versions.append(
            {
                "version": version,
                "accuracy": round(accuracy(df), 6),
                "n_questions": int(len(df)),
                "slices": _slices(df),
            }
        )

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "runs": mlflow_log.search_runs(limit=500),
        "registry": registry.summary(),
        "versions": versions,
    }


def _slices(df: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for column in ("turn_type", "conv_type", "gold_turn_type", "gold_conv_type"):
        if column not in df.columns:
            continue
        values: dict[str, float] = {}
        for value, group in df.groupby(column):
            label = str(value)
            if label and label.lower() != "nan":
                values[label] = round(float(group["correct"].mean()), 6)
        if values:
            out[column] = values
    return out


def write_snapshot(path: Path | None = None) -> Path:
    """Write the snapshot to disk and return its path."""
    target = path or MLFLOW_SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_snapshot(), indent=2) + "\n")
    return target


def read_snapshot(path: Path | None = None) -> dict[str, Any]:
    """Read the committed snapshot, or an empty one when absent."""
    target = path or MLFLOW_SNAPSHOT_PATH
    if not target.exists():
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "exported_at": None,
            "runs": [],
            "registry": {"aliases": {}, "versions": [], "history": []},
            "versions": [],
        }
    try:
        payload: dict[str, Any] = json.loads(target.read_text())
    except json.JSONDecodeError:
        return {"snapshot_version": SNAPSHOT_VERSION, "runs": [], "versions": []}
    return payload
