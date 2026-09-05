"""The gate: paired comparison of two eval-loop runs, and the promotion.

The promotion rule (campaign protocol, 2026-09-03) is **net positive on the
shared questions AND one-sided cluster-corrected McNemar p < 0.05**. Both halves
matter and neither is decorative:

- *Net positive* says the change helped more questions than it hurt.
- *One-sided* because the gate only ever promotes improvements, so spending half
  the rejection region on the direction it will never act in buys nothing — it
  detects what two-sided at α=0.10 detects without the false-positive cost.
- *Cluster-corrected* (Durkalski) because flips are not independent: a
  conversation's turns share a report, a history and usually an error, so four
  fixed turns in one report are one piece of evidence, not four.

Every verdict also carries the cluster bootstrap CI on Δ, which is what to read
when a p sits near the line. Promotion goes through ``tracking.registry.promote``,
which records the whole comparison on the history.

`load_run_csv` is the shared door every arm's evidence comes through, and it
**refuses a CSV with unscored rows** (`IncompleteRunError`). A paired
comparison over a set half of whose turns were never attempted is not a
comparison: the two arms have to face the same questions, which is the entire
reason the gate is paired. The fix is never a flag — it is to finish the pass
with ``convfinqa-evalloop run … --resume-from <csv>``. CSVs written before the
`unscored` column exist are read as all-scored, which is what they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.tracking.comparator import (
    ALPHA,
    ComparisonResult,
    cluster_bootstrap_ci,
    compare_frames,
    mcnemar_exact_p,
)


class IncompleteRunError(ValueError):
    """A predictions CSV carrying turns the runtime never answered."""


def _bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    """One column as booleans; a column that is not there is all-False.

    Trailing columns are added over time, and a CSV committed before one
    existed must keep loading — for `unscored`, "absent" and "no unscored
    rows" are the same claim.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].astype(str).str.lower().isin({"true", "1"})


def load_run_csv(path: Path | str, *, allow_unscored: bool = False) -> pd.DataFrame:
    """Load an eval-loop predictions CSV into the comparison shape.

    Refuses a frame with unscored rows — turns that were never answered, so
    neither arm has a counterpart for them. `allow_unscored` exists for
    bookkeeping that reads a CSV to *describe* a pass rather than to judge it;
    no gate may pass it.
    """
    df = pd.read_csv(path)
    df["correct"] = _bool_column(df, "correct")
    df["unscored"] = _bool_column(df, "unscored")
    df["turn_index"] = df["turn_index"].astype(int)
    n_unscored = int(df["unscored"].sum())
    if n_unscored and not allow_unscored:
        raise IncompleteRunError(
            f"{Path(path).name}: {n_unscored} of {len(df)} rows are unscored — "
            "turns the runtime never answered (the CLI refused them). A paired "
            "comparison needs both arms to face the same questions, so this run "
            "cannot be gated. Finish the pass first:\n"
            f"  convfinqa-evalloop run --split … --version … --resume-from {path}"
        )
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
    base, cand = load_run_csv(baseline_csv), load_run_csv(candidate_csv)
    splits = sorted(
        set(str(v) for v in cand.get("split", pd.Series(dtype=str)).dropna())
    )
    ci = cluster_bootstrap_ci(base, cand)
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
        "mcnemar_p_one_sided": round(result.mcnemar_p_one_sided, 6),
        "cluster_z": round(result.cluster_z, 4),
        "cluster_p_one_sided": result.cluster_p_one_sided,
        "n_flip_clusters": result.n_clusters,
        "alpha": ALPHA,
        "significant_one_sided": result.cluster_p_one_sided < ALPHA,
        "promotable": result.promotable_significant,
        "delta_ci_lo": ci["lo"],
        "delta_ci_hi": ci["hi"],
        "delta_p_positive": ci["p_positive"],
    }
    return result, stats


def gate_reason(stats: dict[str, Any]) -> str:
    """One line a CI log or a promotion record can carry unedited."""
    verdict = "PROMOTE" if stats["promotable"] else "REJECT"
    return (
        f"{verdict}: Δ {stats['accuracy_delta'] * 100:+.2f}pp on "
        f"{stats['n_compared']} shared {stats['evidence_split']} questions "
        f"({stats['fail_to_pass']} fixed vs {stats['pass_to_fail']} broken across "
        f"{stats['n_flip_clusters']} conversations); one-sided clustered "
        f"McNemar p={stats['cluster_p_one_sided']:.4f} "
        f"(α={stats['alpha']}); 95% CI "
        f"[{stats['delta_ci_lo'] * 100:+.2f}pp, {stats['delta_ci_hi'] * 100:+.2f}pp], "
        f"P(Δ>0)={stats['delta_p_positive']:.2f}"
    )


def promote_winner(
    result: ComparisonResult,
    stats: dict[str, Any],
    *,
    actor: str = "evalloop",
) -> dict[str, Any]:
    """Promote the gate's winner; the loser's champion is simply retained.

    The candidate wins by being net positive on the shared questions *and*
    clearing one-sided clustered McNemar at α=0.05. The flips it broke, the p
    and the bootstrap interval travel with the verdict onto the registry
    history, so any promotion stays inspectable long after the fact.

    Promotion goes through ``force=True`` because the registry's own comparator
    still applies the older net-positive rule; the significance requirement is
    strictly stronger, and the reason recorded on the history says so.
    """
    from convfinqa.tracking import registry

    if stats["promotable"]:
        outcome = registry.promote(
            result.candidate_version,
            comparison=result,
            actor=actor,
            force=True,
            reason=gate_reason(stats),
        )
        return {
            "winner": result.candidate_version,
            "cluster_p_one_sided": stats["cluster_p_one_sided"],
            **outcome.as_dict(),
        }
    return {
        "winner": result.baseline_version,
        "promoted": False,
        "previous_champion": registry.champion(),
        "reason": f"baseline retained — {gate_reason(stats)}",
        "cluster_p_one_sided": stats["cluster_p_one_sided"],
        "comparison": result.as_dict(),
    }
