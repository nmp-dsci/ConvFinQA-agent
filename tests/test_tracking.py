"""The promotion contract, the trace store, and the bundle fingerprint.

The comparator tests are the important ones. They encode the rule that a change
which raises overall accuracy while breaking questions that used to pass is a
regression, not an improvement — which is the whole reason the gate exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from convfinqa.tracking import registry
from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id
from convfinqa.tracking.comparator import compare_frames
from convfinqa.tracking.traces import TraceStore


def _frame(rows: list[tuple[str, int, str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_id": r,
                "turn_index": t,
                "question": f"q{t}",
                "gold_answer": g,
                "pred_answer": p,
                "correct": c,
            }
            for r, t, g, p, c in rows
        ]
    )


# ---------------------------------------------------------------------------
# Comparator — the promotion contract
# ---------------------------------------------------------------------------


def test_strict_improvement_is_promotable() -> None:
    """Fixing a question and breaking none passes both conditions."""
    base = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "9", False)])
    cand = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "2", True)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.promotable
    assert result.accuracy_delta == pytest.approx(0.5)
    assert len(result.improvements) == 1
    assert result.regressions == []


def test_net_positive_with_a_regression_is_refused() -> None:
    """The case the flip check exists for: two fixed, one broken, net positive.

    Overall accuracy rises, so an accuracy-only gate would call this a win. It is
    not — a capability that used to work now does not.
    """
    base = _frame(
        [
            ("r1", 0, "1", "1", True),
            ("r1", 1, "2", "x", False),
            ("r1", 2, "3", "x", False),
        ]
    )
    cand = _frame(
        [
            ("r1", 0, "1", "x", False),
            ("r1", 1, "2", "2", True),
            ("r1", 2, "3", "3", True),
        ]
    )
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.accuracy_delta > 0
    assert result.accuracy_ok
    assert not result.no_regressions
    assert not result.promotable
    assert "flipped pass→fail" in result.reason()


def test_equal_accuracy_with_no_flips_is_promotable() -> None:
    """`>=`, not `>`: an identical-scoring bundle may still be promoted."""
    base = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "x", False)])
    cand = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "y", False)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.promotable


def test_accuracy_drop_is_refused() -> None:
    """Losing the headline number is refused regardless of flips."""
    base = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "2", True)])
    cand = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "x", False)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert not result.accuracy_ok
    assert not result.promotable


def test_non_overlapping_questions_are_excluded_and_noted() -> None:
    """A version scored on a different subset must not show phantom regressions."""
    base = _frame([("r1", 0, "1", "1", True), ("r2", 0, "1", "1", True)])
    cand = _frame([("r1", 0, "1", "1", True)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.n_compared == 1
    assert result.regressions == []
    assert result.notes and "no counterpart" in result.notes[0]


def test_empty_comparison_is_not_promotable() -> None:
    """Nothing compared means nothing demonstrated."""
    base = _frame([("r1", 0, "1", "1", True)])
    cand = _frame([("r9", 0, "1", "1", True)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.n_compared == 0
    assert not result.promotable


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_first_version_becomes_champion(tmp_path: Path) -> None:
    """A system with no champion cannot serve, so the first one is adopted."""
    path = tmp_path / "registry.json"
    registry.register("v1", path=path)
    outcome = registry.promote("v1", path=path)
    assert outcome.promoted
    assert registry.champion(path) == "v1"


def test_promotion_history_is_append_only(tmp_path: Path) -> None:
    """Every promotion is recorded; nothing is overwritten."""
    path = tmp_path / "registry.json"
    registry.register("v1", path=path)
    registry.register("v2", path=path)
    registry.promote("v1", path=path)
    registry.promote("v2", force=True, path=path)

    doc = registry.load(path)
    assert len(doc.history) == 2
    assert [h["version"] for h in doc.history] == ["v1", "v2"]
    assert doc.history[1]["previous_champion"] == "v1"
    assert doc.history[1]["forced"] is True
    # Both versions keep their spec — a superseded champion is never deleted.
    assert {v["version"] for v in doc.versions} == {"v1", "v2"}


def test_unregistered_version_cannot_be_promoted(tmp_path: Path) -> None:
    """Promotion requires registration first — no implicit creation."""
    with pytest.raises(ValueError):
        registry.promote("nope", path=tmp_path / "registry.json")


def test_registering_twice_refreshes_rather_than_duplicates(tmp_path: Path) -> None:
    """Backfill is idempotent, which requires this."""
    path = tmp_path / "registry.json"
    first = registry.register("v1", path=path, notes="one")
    registry.register("v1", path=path, notes="two")
    doc = registry.load(path)
    assert len(doc.versions) == 1
    assert doc.versions[0]["registered_at"] == first["registered_at"]
    assert doc.versions[0]["notes"] == "two"


# ---------------------------------------------------------------------------
# Bundle fingerprint
# ---------------------------------------------------------------------------


def test_bundle_id_is_stable_for_the_same_spec() -> None:
    """Same spec in, same id out — it is the join key across artifacts."""
    spec = bundle_fingerprint(version="v2")
    assert bundle_id(spec) == bundle_id(dict(reversed(list(spec.items()))))
    assert len(bundle_id(spec)) == 12


def test_bundle_id_changes_with_the_spec() -> None:
    """A different prompt version is a different bundle."""
    assert bundle_id(bundle_fingerprint(version="v1")) != bundle_id(
        bundle_fingerprint(version="v2")
    )


# ---------------------------------------------------------------------------
# Trace store
# ---------------------------------------------------------------------------


def test_trace_roundtrip(tmp_path: Path) -> None:
    """A recorded turn comes back with its per-stage capture intact."""
    store = TraceStore(tmp_path / "t.db")
    capture = {
        "triage": {"output": {"turn_type": "number"}, "metrics": {"latency_ms": 120.0}},
        "retriever": {"output": {"answers": []}, "metrics": {"latency_ms": 80.0}},
    }
    trace_id = store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture=capture,
        answer="42",
        gold_answer="42",
        correct=True,
    )
    record = store.get_turn(trace_id)
    assert record is not None
    assert record["answer"] == "42"
    assert record["correct"] == 1
    assert record["capture"]["triage"]["output"]["turn_type"] == "number"
    # Per-stage latency rolls up to the turn.
    assert record["latency_ms"] == pytest.approx(200.0)
    store.close()


def test_trace_recording_never_raises(tmp_path: Path) -> None:
    """Telemetry must not be able to fail the turn that produced it."""
    store = TraceStore(tmp_path / "t.db")
    unserialisable = {"triage": {"output": object()}}
    trace_id = store.record(
        report_id="r1", turn_index=0, question="q", capture=unserialisable
    )
    assert trace_id
    store.close()


def test_trace_listing_filters(tmp_path: Path) -> None:
    """Listing filters by report and source, newest first."""
    store = TraceStore(tmp_path / "t.db")
    store.record(report_id="r1", turn_index=0, question="a", capture={})
    store.record(report_id="r2", turn_index=0, question="b", capture={})
    store.record(report_id="r1", turn_index=1, question="c", capture={}, source="eval")

    assert len(store.list_turns()) == 3
    assert len(store.list_turns(report_id="r1")) == 2
    assert len(store.list_turns(source="eval")) == 1
    assert store.stats()["n_turns"] == 3
    store.close()


# ---------------------------------------------------------------------------
# Metric hygiene
# ---------------------------------------------------------------------------


def test_holdout_split_excludes_everything_the_optimizer_saw() -> None:
    """The two sets must not overlap, or "held out" means nothing.

    Guards a real bug this replaced: the app reported 770 questions as a
    held-out set when 461 of them came from conversations GEPA trained on.
    """
    from convfinqa.serving import evaldata

    seen = evaldata.optimizer_train_ids()
    never = set(evaldata.splits()["never_seen"])
    assert seen & never == set()
    assert seen | never == set(evaldata.splits()["sampled"])
    assert len(never) > 0


def test_split_source_is_the_one_the_optimizer_actually_used() -> None:
    """Sourced from the DSPy backend, not the loader's differently-seeded split.

    The two disagree on 42 of 120 conversations; only the DSPy one describes
    what GEPA trained on.
    """
    from convfinqa.backends.dspy import conv_examples_train
    from convfinqa.serving import evaldata

    assert evaldata.optimizer_train_ids() == {e.report_id for e in conv_examples_train}


def test_holdout_accuracy_is_reported_separately_from_overall() -> None:
    """They are different numbers and must never be conflated."""
    from convfinqa.serving.evaldata import holdout_accuracy
    from convfinqa.tracking.comparator import accuracy, load_predictions

    df = load_predictions("v2")
    held = holdout_accuracy(df)
    assert held["n_questions"] < len(df)
    assert held["accuracy"] != accuracy(df)
