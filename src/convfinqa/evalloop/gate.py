"""The gate: paired comparison of two eval-loop runs, and the promotion.

Uses the house comparator's rule — **net positive on the shared questions**
(more fixed than broken), with the exact McNemar p recorded on every verdict
and flagged when the sample cannot support significance. Promotion goes
through ``tracking.registry.promote``, which records the comparison on the
history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.tracking.comparator import (
    ComparisonResult,
    compare_frames,
    mcnemar_exact_p,
)


def load_run_csv(path: Path | str) -> pd.DataFrame:
    """Load an eval-loop predictions CSV into the comparison shape."""
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
    df["turn_index"] = df["turn_index"].astype(int)
    return df


def gate_runs(
    baseline_csv: Path | str,
    candidate_csv: Path | str,
    *,
    baseline_version: str,
    candidate_version: str,
) -> tuple[ComparisonResult, dict[str, Any]]:
    """Compare two run CSVs question by question; return (result, statistics)."""
    result = compare_frames(
        load_run_csv(baseline_csv),
        load_run_csv(candidate_csv),
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )
    pass_to_fail = len(result.regressions)
    fail_to_pass = len(result.improvements)
    cand = load_run_csv(candidate_csv)
    splits = sorted(
        set(str(v) for v in cand.get("split", pd.Series(dtype=str)).dropna())
    )
    stats = {
        "evidence_split": splits[0]
        if len(splits) == 1
        else ",".join(splits) or "unknown",
        "n_compared": result.n_compared,
        "baseline_accuracy": round(result.baseline_accuracy, 6),
        "candidate_accuracy": round(result.candidate_accuracy, 6),
        "accuracy_delta": round(result.accuracy_delta, 6),
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
        "mcnemar_p": round(mcnemar_exact_p(pass_to_fail, fail_to_pass), 6),
    }
    return result, stats


def promote_winner(
    result: ComparisonResult,
    stats: dict[str, Any],
    *,
    actor: str = "evalloop",
) -> dict[str, Any]:
    """Promote the gate's winner; the loser's champion is simply retained.

    The candidate wins by being net positive on the shared questions. The
    flips it broke and the McNemar p travel with the verdict onto the
    registry history, so a small-sample promotion is inspectable later.
    """
    from convfinqa.tracking import registry

    if result.promotable:
        outcome = registry.promote(
            result.candidate_version, comparison=result, actor=actor
        )
        return {
            "winner": result.candidate_version,
            "mcnemar_p": stats["mcnemar_p"],
            **outcome.as_dict(),
        }
    return {
        "winner": result.baseline_version,
        "promoted": False,
        "previous_champion": registry.champion(),
        "reason": f"baseline retained — {result.reason()}",
        "mcnemar_p": stats["mcnemar_p"],
        "comparison": result.as_dict(),
    }
