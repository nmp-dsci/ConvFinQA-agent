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
def optimizer_train_ids() -> frozenset[str]:
    """The conversations GEPA actually optimized against.

    Deliberately sourced from `backends.dspy`, not from `data.loader`. Both
    define a "60% train split" with seed 42, but by different means — a pandas
    `.sample()` in the loader, a `random.Random(42).shuffle()` in the DSPy
    backend — and they agree on only 78 of 120 conversations. GEPA ran against
    the DSPy one, so that is the only definition with a claim to being the set
    the optimizer saw. Reporting against the other would mislabel 42
    conversations in both directions.
    """
    from convfinqa.backends.dspy import conv_examples_train

    return frozenset(example.report_id for example in conv_examples_train)


@cache
def splits() -> dict[str, list[str]]:
    """Report-id membership per split, keyed by what the optimizer actually saw."""
    from convfinqa.data.loader import sampled_report_ids

    seen = optimizer_train_ids()
    return {
        "optimizer_train": sorted(seen),
        "never_seen": sorted(r for r in sampled_report_ids if r not in seen),
        "sampled": list(sampled_report_ids),
    }


@cache
def split_of() -> dict[str, str]:
    """Map each report id to the split it belongs to."""
    seen = optimizer_train_ids()
    return {
        rid: ("optimizer_train" if rid in seen else "never_seen")
        for rid in splits()["sampled"]
    }


def holdout_accuracy(df: pd.DataFrame) -> dict[str, Any]:
    """Accuracy restricted to conversations the optimizer never saw.

    The number that actually supports a generalisation claim. The full 770-row
    scored set spans all 200 sampled conversations, 120 of which GEPA trained
    on, so the overall figure is a mix of seen and unseen and cannot be
    described as held out.
    """
    seen = optimizer_train_ids()
    unseen = df[~df["report_id"].isin(seen)]
    return {
        "accuracy": round(float(unseen["correct"].mean()), 6) if len(unseen) else 0.0,
        "n_questions": int(len(unseen)),
        "n_conversations": int(unseen["report_id"].nunique()) if len(unseen) else 0,
    }


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
    optimizer_train_ids.cache_clear()
