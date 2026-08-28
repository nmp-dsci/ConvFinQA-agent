"""Reconstruct experiment history that predates the tracking layer.

Everything before this phase ran without MLflow, but the evidence survives in
git: prediction CSVs per version, GEPA artifacts under `runs/`, and the s7 rule
stores under `evaluation/diagnostics/`. This module replays that evidence into
the tracking store once, so the experiments tab opens with real history rather
than with one row.

Backfill is *idempotent* — re-running it does not duplicate versions, because
`register` refreshes an existing entry instead of appending a second one. It is
also the last time history is reconstructed by hand: from here on, logging lives
inside the runners themselves.
"""

from __future__ import annotations

import json
from typing import Any

from convfinqa.config import DIAGNOSTICS_DIR, RUNS_DIR
from convfinqa.tracking import mlflow_log, registry, snapshot
from convfinqa.tracking.comparator import (
    accuracy,
    available_versions,
    load_predictions,
)


def _slice_metrics(df: Any) -> dict[str, float]:
    """Per-slice accuracies, flattened into MLflow's metric namespace."""
    out: dict[str, float] = {}
    for column in ("turn_type", "conv_type", "gold_turn_type", "gold_conv_type"):
        if column not in df.columns:
            continue
        for value, group in df.groupby(column):
            label = str(value).strip().replace(" ", "_")
            if label and label.lower() != "nan":
                out[f"accuracy_{column}_{label}"] = round(
                    float(group["correct"].mean()), 6
                )
    return out


def backfill_evals() -> list[dict[str, Any]]:
    """Log one MLflow run per committed prediction CSV and register each version."""
    records: list[dict[str, Any]] = []
    for version in available_versions():
        try:
            df = load_predictions(version)
        except (FileNotFoundError, ValueError):
            continue
        overall = round(accuracy(df), 6)
        metrics = {
            "accuracy": overall,
            "n_questions": float(len(df)),
            **_slice_metrics(df),
        }
        with mlflow_log.run(
            f"eval-{version}",
            kind="eval",
            version=version,
            params={"backfilled": True, "source": "committed predictions CSV"},
            tags={"bundle_version": version, "backfilled": "true"},
        ) as recorder:
            recorder.metrics(metrics)
            recorder.artifact(_predictions_file(version))
            run_id = recorder.run_id
        registry.register(
            version,
            source="manual",
            run_id=run_id,
            metrics={"accuracy": overall, "n_questions": float(len(df))},
        )
        records.append({"version": version, "accuracy": overall, "run_id": run_id})
    return records


def _predictions_file(version: str) -> str:
    from convfinqa.tracking.comparator import predictions_path

    return str(predictions_path(version))


def backfill_gepa() -> list[dict[str, Any]]:
    """Log one MLflow run per committed GEPA run directory."""
    records: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return records
    for run_dir in sorted(RUNS_DIR.glob("gepa_*")):
        if not run_dir.is_dir():
            continue
        artifact = run_dir / "optimized_runner.json"
        if not artifact.exists():
            artifact = run_dir / "dspy_optimized_runner.json"
        mode = "real" if "_real_" in run_dir.name else "smoke"
        with mlflow_log.run(
            run_dir.name,
            kind="gepa",
            overlay=run_dir.name,
            params={"gepa_mode": mode, "backfilled": True},
            tags={"gepa_name": run_dir.name, "backfilled": "true"},
        ) as recorder:
            for metric_file in run_dir.glob("*.json"):
                if metric_file.name.startswith("optimized"):
                    continue
                payload = _read_json(metric_file)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, (int, float)):
                            recorder.metric(key, float(value))
            recorder.artifact(artifact)
            records.append({"gepa_run": run_dir.name, "run_id": recorder.run_id})
    return records


def backfill_s7() -> list[dict[str, Any]]:
    """Log one MLflow run per s7 variant found in the diagnostics directory."""
    records: list[dict[str, Any]] = []
    if not DIAGNOSTICS_DIR.exists():
        return records
    variants = sorted(
        {
            path.stem.split("case_results_")[-1]
            for path in DIAGNOSTICS_DIR.glob("case_results_*.jsonl")
        }
    )
    for variant in variants:
        cases = _read_jsonl(DIAGNOSTICS_DIR / f"case_results_{variant}.jsonl")
        resolved = sum(1 for case in cases if case.get("resolved") is True)
        rule_counts = {
            agent: len(_read_jsonl(DIAGNOSTICS_DIR / f"rules_{agent}_{variant}.jsonl"))
            for agent in ("triage", "preprocess", "retriever", "calculator")
        }
        with mlflow_log.run(
            f"s7-{variant}",
            kind="s7",
            version=variant,
            params={"variant": variant, "backfilled": True},
            tags={"bundle_version": variant, "backfilled": "true"},
        ) as recorder:
            recorder.metrics(
                {
                    "cases_diagnosed": float(len(cases)),
                    "cases_resolved": float(resolved),
                    "cases_unresolved": float(len(cases) - resolved),
                    **{f"rules_{a}": float(n) for a, n in rule_counts.items()},
                }
            )
            recorder.artifact(DIAGNOSTICS_DIR / f"diagnostic_results_{variant}.csv")
            records.append(
                {
                    "variant": variant,
                    "cases": len(cases),
                    "resolved": resolved,
                    "run_id": recorder.run_id,
                }
            )
    return records


def _read_json(path: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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


def backfill(champion: str | None = None) -> dict[str, Any]:
    """Run every backfill, seed the champion alias, and write the snapshot.

    The champion defaults to the highest-accuracy committed version rather than
    the newest — v3_1 scored *below* v2, and seeding the newest would have made
    the registry assert an improvement that did not happen.
    """
    evals = backfill_evals()
    gepa = backfill_gepa()
    s7 = backfill_s7()

    doc = registry.load()
    if doc.aliases.get(registry.CHAMPION) is None and evals:
        best = champion or max(evals, key=lambda r: r["accuracy"])["version"]
        registry.promote(best, actor="backfill")

    snapshot_path = snapshot.write_snapshot()
    return {
        "evals": evals,
        "gepa": gepa,
        "s7": s7,
        "champion": registry.champion(),
        "snapshot": str(snapshot_path),
    }
