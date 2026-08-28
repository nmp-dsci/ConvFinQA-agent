"""Cached readers for the committed evaluation artifacts.

Every one of these files is immutable once committed, so they are read once per
process and kept. Before this, `/eval/runs/<v>/predictions` re-parsed a 30 MB CSV
on every request, which is slow in dev and, on a 1 GB App Runner instance, is the
difference between a responsive demo and one that stalls whenever two people open
the answers tab at once.
"""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any

import pandas as pd

from convfinqa.config import PREDICTIONS_DIR

MODEL_CSV_PATTERN: dict[str, str] = {
    "dspy": "dspy_predictions_{v}_joined.csv",
    "pydantic": "pydantic_predictions_{v}_joined.csv",
    "api": "api_predictions_{v}_joined.csv",
}


def version_key(version: str) -> tuple[int, int]:
    """Sort key for version labels: `v1` → (1, 0), `v3_1` → (3, 1).

    Always returns a uniform `(int, int)` so mixed plain and variant versions
    order without comparing across types. Unparseable labels sort last.
    """
    body = version[1:] if version.startswith("v") else version
    parts = body.split("_")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (10_000, 0)
    return (major, minor)


@cache
def available_versions() -> list[str]:
    """Prompt versions with at least one joined predictions CSV."""
    if not PREDICTIONS_DIR.exists():
        return []
    versions: set[str] = set()
    for path in PREDICTIONS_DIR.iterdir():
        if not path.is_file() or path.suffix != ".csv":
            continue
        stem = path.stem
        if not stem.endswith("_joined"):
            continue
        base = stem[: -len("_joined")]
        for model in MODEL_CSV_PATTERN:
            prefix = f"{model}_predictions_"
            if base.startswith(prefix):
                versions.add(base[len(prefix) :])
                break
    return sorted(versions, key=version_key)


@lru_cache(maxsize=32)
def load_joined(version: str, model: str = "pydantic") -> pd.DataFrame | None:
    """Load a joined predictions CSV, normalised. None when absent."""
    pattern = MODEL_CSV_PATTERN.get(model)
    if pattern is None:
        return None
    path = PREDICTIONS_DIR / pattern.format(v=version)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
    if "q_order" in df.columns:
        df["q_order"] = df["q_order"].astype(float).astype(int)
    if "pred_program" not in df.columns:
        df["pred_program"] = ""
    if "turn_index" not in df.columns:
        df["turn_index"] = df.groupby("report_id").cumcount()
    df["turn_index"] = df["turn_index"].astype(int)
    return df


@cache
def gold_programs() -> dict[tuple[str, int], str]:
    """Gold DSL program per (report_id, q_order)."""
    from convfinqa.data.loader import qa_data

    return {
        (str(row.report_id), int(row.q_order)): str(row.turn_program)
        for row in qa_data.itertuples()
    }


@cache
def splits() -> dict[str, list[str]]:
    """Report-id membership for each dataset split.

    `train` is the 60% of the sampled conversations the optimizer was allowed to
    see; `holdout` is everything it was not. Surfacing this in the app is what
    turns the held-out claim into something a visitor can check rather than take
    on trust.
    """
    from convfinqa.data.loader import (
        sampled_report_ids,
        test_report_ids,
        train_report_ids,
    )

    return {
        "train": list(train_report_ids),
        "holdout": list(test_report_ids),
        "sampled": list(sampled_report_ids),
    }


@cache
def split_of() -> dict[str, str]:
    """Map each report id to the split it belongs to."""
    membership = splits()
    lookup = {rid: "holdout" for rid in membership["holdout"]}
    lookup.update({rid: "train" for rid in membership["train"]})
    return lookup


def slice_accuracy(df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Accuracy over a frame, in the shape the API returns."""
    n = len(df)
    correct = int(df["correct"].sum())
    return {
        "label": label,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "n_correct": correct,
        "n_total": n,
    }


def slices_by(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """Accuracy per distinct value of `column`."""
    if column not in df.columns:
        return []
    return [
        slice_accuracy(df[df[column] == value], str(value))
        for value in sorted(df[column].dropna().unique(), key=str)
    ]


def clear_caches() -> None:
    """Drop every cached read. For tests that write fixture CSVs."""
    available_versions.cache_clear()
    load_joined.cache_clear()
    gold_programs.cache_clear()
    splits.cache_clear()
    split_of.cache_clear()
