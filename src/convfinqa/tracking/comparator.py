"""The regression comparator: what "better" has to mean before a promotion.

Overall accuracy going up is not sufficient evidence that a change is an
improvement. A prompt edit that fixes twelve number-retrieval turns and breaks
nine program turns nets out positive and is still a regression for anyone who
asks a multi-step question. So promotion requires both:

    1. overall accuracy >= champion (never trade the headline number away), and
    2. no per-question pass -> fail flips (never silently lose a capability).

Rule 2 is what makes the gate load-bearing. It is also why the comparison is
per-question rather than aggregate: the flip list is the evidence, and the CI
gate prints it so a failed merge tells you exactly which questions broke.

Everything here is deterministic and offline. It reads committed prediction CSVs
and makes zero API calls, which is what lets it run on every pull request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import PREDICTIONS_DIR

# Tolerance on the accuracy comparison. Scoring is deterministic, so this exists
# only to absorb float formatting in the CSV round-trip, not real movement.
ACCURACY_EPSILON = 1e-9


@dataclass
class Flip:
    """One question whose correctness changed between two runs."""

    report_id: str
    q_order: int
    question: str
    gold_answer: str
    baseline_answer: str
    candidate_answer: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API and the CI gate output."""
        return {
            "report_id": self.report_id,
            "q_order": self.q_order,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "baseline_answer": self.baseline_answer,
            "candidate_answer": self.candidate_answer,
        }


@dataclass
class ComparisonResult:
    """The verdict, plus every fact it was based on."""

    baseline_version: str
    candidate_version: str
    baseline_accuracy: float
    candidate_accuracy: float
    n_compared: int
    regressions: list[Flip] = field(default_factory=list)
    improvements: list[Flip] = field(default_factory=list)
    slice_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def accuracy_delta(self) -> float:
        """Candidate minus baseline overall accuracy."""
        return self.candidate_accuracy - self.baseline_accuracy

    @property
    def accuracy_ok(self) -> bool:
        """True when the candidate does not lose overall accuracy."""
        return self.accuracy_delta >= -ACCURACY_EPSILON

    @property
    def no_regressions(self) -> bool:
        """True when no question flipped from pass to fail."""
        return not self.regressions

    @property
    def promotable(self) -> bool:
        """Both conditions of the promotion contract, and a non-empty comparison."""
        return self.accuracy_ok and self.no_regressions and self.n_compared > 0

    def reason(self) -> str:
        """One line explaining the verdict, suitable for a CI log or a UI badge."""
        if self.n_compared == 0:
            return "no overlapping questions to compare"
        if not self.accuracy_ok:
            return (
                f"overall accuracy fell {self.baseline_accuracy:.1%} → "
                f"{self.candidate_accuracy:.1%} ({self.accuracy_delta:+.2%})"
            )
        if self.regressions:
            return (
                f"{len(self.regressions)} question(s) flipped pass→fail "
                f"despite {self.accuracy_delta:+.2%} overall"
            )
        return (
            f"accuracy {self.baseline_accuracy:.1%} → {self.candidate_accuracy:.1%} "
            f"({self.accuracy_delta:+.2%}), {len(self.improvements)} fixed, 0 broken"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API and the registry's promotion record."""
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "baseline_accuracy": round(self.baseline_accuracy, 6),
            "candidate_accuracy": round(self.candidate_accuracy, 6),
            "accuracy_delta": round(self.accuracy_delta, 6),
            "n_compared": self.n_compared,
            "accuracy_ok": self.accuracy_ok,
            "no_regressions": self.no_regressions,
            "promotable": self.promotable,
            "reason": self.reason(),
            "regressions": [f.as_dict() for f in self.regressions],
            "improvements": [f.as_dict() for f in self.improvements],
            "slice_deltas": self.slice_deltas,
            "notes": self.notes,
        }


def predictions_path(version: str, model: str = "pydantic") -> Path:
    """Path to a committed predictions CSV for `version`."""
    return PREDICTIONS_DIR / f"{model}_predictions_{version}.csv"


def load_predictions(version: str, model: str = "pydantic") -> pd.DataFrame:
    """Load and normalise a predictions CSV into the comparison shape."""
    path = predictions_path(version, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No committed predictions for version {version!r}: {path}"
        )
    df = pd.read_csv(path)
    required = {"report_id", "question", "gold_answer", "pred_answer", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
    # `turn_index` is the per-conversation position and is the stable join key;
    # q_order is not present in every historical CSV.
    if "turn_index" not in df.columns:
        df["turn_index"] = df.groupby("report_id").cumcount()
    df["turn_index"] = df["turn_index"].astype(int)
    return df


def accuracy(df: pd.DataFrame) -> float:
    """Overall turn accuracy of a predictions frame."""
    return float(df["correct"].mean()) if len(df) else 0.0


def _slice_accuracies(df: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in df.columns:
        return {}
    out: dict[str, float] = {}
    for value, group in df.groupby(column):
        label = str(value)
        if label and label.lower() != "nan":
            out[label] = float(group["correct"].mean())
    return out


def compare(
    baseline_version: str,
    candidate_version: str,
    *,
    model: str = "pydantic",
) -> ComparisonResult:
    """Compare two committed prediction runs question by question."""
    baseline = load_predictions(baseline_version, model)
    candidate = load_predictions(candidate_version, model)
    return compare_frames(
        baseline,
        candidate,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )


def compare_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    baseline_version: str,
    candidate_version: str,
) -> ComparisonResult:
    """Compare two already-loaded prediction frames on their shared questions.

    Only questions present in *both* runs are compared. A version evaluated over
    a different subset would otherwise show phantom regressions for questions it
    simply never attempted — the note records how many were dropped.
    """
    key = ["report_id", "turn_index"]
    merged = baseline.merge(candidate, on=key, how="inner", suffixes=("_base", "_cand"))

    notes: list[str] = []
    dropped = len(baseline) - len(merged)
    if dropped:
        notes.append(
            f"{dropped} question(s) in {baseline_version} had no counterpart in "
            f"{candidate_version} and were excluded from the comparison"
        )

    regressions: list[Flip] = []
    improvements: list[Flip] = []
    for row in merged.itertuples():
        before = bool(row.correct_base)
        after = bool(row.correct_cand)
        if before == after:
            continue
        flip = Flip(
            report_id=str(row.report_id),
            q_order=int(row.turn_index),
            question=str(getattr(row, "question_base", "")),
            gold_answer=str(getattr(row, "gold_answer_base", "")),
            baseline_answer=str(getattr(row, "pred_answer_base", "")),
            candidate_answer=str(getattr(row, "pred_answer_cand", "")),
        )
        (regressions if before and not after else improvements).append(flip)

    slice_deltas: dict[str, dict[str, float]] = {}
    for column in ("turn_type", "conv_type", "gold_turn_type", "gold_conv_type"):
        base_slices = _slice_accuracies(baseline, column)
        cand_slices = _slice_accuracies(candidate, column)
        shared = sorted(set(base_slices) & set(cand_slices))
        if shared:
            slice_deltas[column] = {
                label: round(cand_slices[label] - base_slices[label], 6)
                for label in shared
            }

    return ComparisonResult(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_accuracy=accuracy(baseline),
        candidate_accuracy=accuracy(candidate),
        n_compared=len(merged),
        regressions=regressions,
        improvements=improvements,
        slice_deltas=slice_deltas,
        notes=notes,
    )


def available_versions(model: str = "pydantic") -> list[str]:
    """Every version with a committed predictions CSV, oldest first."""
    if not PREDICTIONS_DIR.exists():
        return []
    prefix = f"{model}_predictions_"
    versions = [
        path.stem[len(prefix) :]
        for path in PREDICTIONS_DIR.glob(f"{prefix}*.csv")
        if not path.stem.endswith("_joined")
    ]
    return sorted(versions, key=_version_key)


def _version_key(version: str) -> tuple[int, int]:
    body = version[1:] if version.startswith("v") else version
    parts = body.split("_")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return (10_000, 0)
