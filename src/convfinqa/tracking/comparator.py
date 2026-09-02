"""The regression comparator: what "better" has to mean before a promotion.

Overall accuracy going up is not enough evidence on its own to trust blindly,
so the comparison is per-question rather than aggregate. Promotion requires a
**net-positive paired comparison on the shared question set**: more questions
fixed than broken (equivalently, the paired accuracy delta is positive), with
the exact McNemar p recorded on the verdict (flagged when not significant at
alpha=0.05). Individual pass -> fail flips no longer veto promotion on their
own (rule changed 2026-09-02 at the owner's direction) — every flip is still
listed and counted, and the CI gate prints the full list so a promotion or a
failed merge tells you exactly which questions moved and in which direction.

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


ALPHA = 0.05


def mcnemar_exact_p(pass_to_fail: int, fail_to_pass: int) -> float:
    """Two-sided exact McNemar p from the discordant-pair counts.

    Under "the two versions are equally good", each discordant question is a
    fair coin between the two flip directions; the p is the binomial
    probability of an imbalance at least this extreme.
    """
    from math import comb

    n = pass_to_fail + fail_to_pass
    if n == 0:
        return 1.0
    k = min(pass_to_fail, fail_to_pass)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


@dataclass
class ComparisonResult:
    """The verdict, plus every fact it was based on.

    The promotion rule is **net positive on the shared question set**: more
    questions fixed than broken (equivalently, the paired accuracy delta is
    positive). Individual pass→fail flips no longer veto on their own — they
    are listed, counted, and fed into the exact McNemar p, which is recorded
    on every verdict (and flagged when the sample cannot support significance)
    so a small-sample promotion is read as what it is.

    `baseline_accuracy`/`candidate_accuracy` (and the delta derived from them)
    are computed over the *shared* question set — the same population the flip
    counts use — so the rule's two halves describe one population.
    `baseline_accuracy_all`/`candidate_accuracy_all` keep the full-frame
    headline numbers for display; they never drive `promotable`.
    """

    baseline_version: str
    candidate_version: str
    baseline_accuracy: float
    candidate_accuracy: float
    baseline_accuracy_all: float
    candidate_accuracy_all: float
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
        """True when no question flipped from pass to fail. Informational."""
        return not self.regressions

    @property
    def pass_to_fail(self) -> int:
        """Questions the candidate broke."""
        return len(self.regressions)

    @property
    def fail_to_pass(self) -> int:
        """Questions the candidate fixed."""
        return len(self.improvements)

    @property
    def mcnemar_p(self) -> float:
        """Exact McNemar p over the discordant pairs."""
        return mcnemar_exact_p(self.pass_to_fail, self.fail_to_pass)

    @property
    def significant(self) -> bool:
        """True when the flip imbalance clears α = 0.05."""
        return self.mcnemar_p < ALPHA

    @property
    def promotable(self) -> bool:
        """Net positive on the shared set, and a non-empty comparison."""
        return self.accuracy_delta > ACCURACY_EPSILON and self.n_compared > 0

    def _p_note(self) -> str:
        p = self.mcnemar_p
        tag = "significant" if self.significant else "not significant"
        return f"McNemar p={p:.3f} ({tag} at α={ALPHA})"

    def reason(self) -> str:
        """One line explaining the verdict, suitable for a CI log or a UI badge."""
        if self.n_compared == 0:
            return "no overlapping questions to compare"
        flips = f"{self.fail_to_pass} fixed vs {self.pass_to_fail} broken"
        if not self.promotable:
            return (
                f"not net positive: {self.baseline_accuracy:.1%} → "
                f"{self.candidate_accuracy:.1%} ({self.accuracy_delta:+.2%}) "
                f"on the shared set, {flips}; {self._p_note()}"
            )
        return (
            f"net positive: {self.baseline_accuracy:.1%} → "
            f"{self.candidate_accuracy:.1%} ({self.accuracy_delta:+.2%}) "
            f"on the shared set, {flips}; {self._p_note()}"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API and the registry's promotion record."""
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "baseline_accuracy": round(self.baseline_accuracy, 6),
            "candidate_accuracy": round(self.candidate_accuracy, 6),
            "baseline_accuracy_all": round(self.baseline_accuracy_all, 6),
            "candidate_accuracy_all": round(self.candidate_accuracy_all, 6),
            "accuracy_delta": round(self.accuracy_delta, 6),
            "n_compared": self.n_compared,
            "accuracy_ok": self.accuracy_ok,
            "no_regressions": self.no_regressions,
            "pass_to_fail": self.pass_to_fail,
            "fail_to_pass": self.fail_to_pass,
            "mcnemar_p": round(self.mcnemar_p, 6),
            "significant": self.significant,
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
    """Overall turn accuracy of a predictions frame — the *execution* accuracy."""
    return float(df["correct"].mean()) if len(df) else 0.0


def _predicted_program(row: Any) -> str:
    """The best available numeric program for one row.

    `pred_program` is written by the preprocess stage over sub-question
    placeholders (`multiply(divide(C, B), 100)`), so on its own it cannot be
    compared to a gold program written over values. The calculator's recorded
    trajectory is that same program with the retrieved numbers substituted in,
    so it is the one to score when the placeholder form is all `pred_program`
    holds. Preferring the parseable one rather than picking a column outright
    keeps this working for runs from either shape.
    """
    from convfinqa.evaluation.metrics import parse_program, program_from_trajectory

    declared = getattr(row, "pred_program", "")
    parsed = parse_program(declared)
    if parsed and all(
        arg.startswith("#") or _is_numeric(arg) for _, args in parsed for arg in args
    ):
        return str(declared)

    raw = getattr(row, "calculator_io", "")
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return str(declared or "")
        if isinstance(payload, dict):
            return program_from_trajectory(payload.get("trajectory"))
    return str(declared or "")


def _is_numeric(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def program_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """Program accuracy of a predictions frame, beside its execution accuracy.

    Scored only over turns whose *gold* entry is a real program. A number-
    selection turn's gold "program" is the selected value itself, and the paper
    does not score those — folding them in would either inflate the metric (by
    counting trivial matches) or depress it (by counting them as misses),
    depending on the convention, and neither number would mean anything.

    Zero API calls: everything needed is already in the committed CSVs.
    """
    from convfinqa.evaluation.metrics import has_program, program_match

    n_scored = 0
    n_correct = 0
    for row in df.itertuples():
        gold = getattr(row, "gold_program", "")
        if not has_program(gold):
            continue
        n_scored += 1
        n_correct += int(program_match(_predicted_program(row), gold))

    return {
        "program_accuracy": round(n_correct / n_scored, 6) if n_scored else 0.0,
        "n_program_correct": float(n_correct),
        "n_program_turns": float(n_scored),
    }


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

    shared_baseline_accuracy = (
        float(merged["correct_base"].mean()) if len(merged) else 0.0
    )
    shared_candidate_accuracy = (
        float(merged["correct_cand"].mean()) if len(merged) else 0.0
    )

    return ComparisonResult(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_accuracy=shared_baseline_accuracy,
        candidate_accuracy=shared_candidate_accuracy,
        baseline_accuracy_all=accuracy(baseline),
        candidate_accuracy_all=accuracy(candidate),
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
