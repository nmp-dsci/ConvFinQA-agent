"""Per-conversation prediction cache for ConvFinQA evaluations.

The three evaluation scripts (`pydantic_agent.py`, `api_eval.py`,
`dspy_agent.py`) all need the same caching pattern:

    1. Load an existing predictions CSV for the active prompt version.
    2. Identify which conversations are *fully* scored — every turn for that
       report_id is present in the CSV.
    3. Skip those, run only the missing conversations, and merge new results
       back in.

That pattern lives here so the three scripts can't drift apart on what
"fully cached" means.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class _HasReportIdAndQuestions(Protocol):
    """Minimal interface every example object needs to be cacheable."""

    report_id: str
    questions: list[str]


def load_cached_conversations(
    csv_path: Path,
    examples: list[Any],
    *,
    required_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Return `(cached_df, cached_rids)` for fully-scored conversations.

    A conversation (identified by `ex.report_id`) is considered fully cached
    only if every turn index `0..len(ex.questions)-1` appears in the CSV.

    If `required_columns` is supplied and any are missing from the CSV, the
    cache is treated as invalid (schema drift after a recent column addition).

    Returns an empty `DataFrame` + empty set if the file is missing,
    unreadable, has the wrong schema, or contains no fully-scored rows.
    """
    if not csv_path.exists():
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), set()
    if not {"report_id", "turn_index"}.issubset(df.columns):
        return pd.DataFrame(), set()
    if required_columns and (required_columns - set(df.columns)):
        return pd.DataFrame(), set()

    cached_rids = identify_cached_conversations(df, examples)
    return df, cached_rids


def identify_cached_conversations(df: pd.DataFrame, examples: list[Any]) -> set[str]:
    """Return the set of report_ids in `examples` whose every turn is in `df`.

    Pure helper around the in-memory DataFrame, useful when the CSV has
    already been loaded for other reasons.
    """
    cached_rids: set[str] = set()
    for ex in examples:
        present = set(df.loc[df["report_id"] == ex.report_id, "turn_index"].astype(int))
        if all(i in present for i in range(len(ex.questions))):
            cached_rids.add(ex.report_id)
    return cached_rids


def flush_csv_atomic(
    out_path: Path,
    rows: list[list[Any]],
    columns: list[str],
) -> None:
    """Write the CSV via `tmp + replace`. Rows are sorted by `(report_id, turn_index)`.

    Used by `api_eval.py` to flush after each completed conversation so an
    interrupted run can resume from the last persisted state.
    """
    sorted_rows = sorted(rows, key=lambda r: (str(r[0]), int(r[1])))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(sorted_rows)
    tmp.replace(out_path)
