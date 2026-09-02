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


def test_net_positive_with_a_regression_promotes_and_records_the_flip() -> None:
    """Two fixed, one broken, net positive: promotable under the net-positive
    rule — and the broken question and the McNemar p travel with the verdict,
    so the trade is recorded rather than hidden."""
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
    assert result.promotable
    assert result.pass_to_fail == 1 and result.fail_to_pass == 2
    assert "net positive" in result.reason()
    assert "McNemar" in result.reason()
    d = result.as_dict()
    assert d["mcnemar_p"] == pytest.approx(1.0)  # 2 vs 1 is pure coin-flip land
    assert d["significant"] is False


def test_equal_accuracy_is_not_net_positive() -> None:
    """Strictly `>`: a bundle that changes nothing has demonstrated nothing."""
    base = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "x", False)])
    cand = _frame([("r1", 0, "1", "1", True), ("r1", 1, "2", "y", False)])
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert not result.promotable
    assert "not net positive" in result.reason()


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


def test_accuracy_gate_uses_the_shared_question_set_not_full_frames() -> None:
    """The accuracy half of the gate must describe the same population as the
    flip check, not the two runs' full frames.

    Baseline has an extra question the candidate never scored (r3); candidate has
    an extra question the baseline never scored (r4, wrong). Neither extra row is
    part of what was actually compared. A full-frame accuracy comparison would
    let the candidate's unrelated r4 failure drag its overall accuracy down.
    Under the net-positive rule a clean tie is not promotable either way — but
    the verdict must say "no improvement", never "accuracy fell".
    """
    base = _frame(
        [
            ("r1", 0, "1", "1", True),
            ("r2", 0, "1", "1", True),
            ("r3", 0, "1", "1", True),
        ]
    )
    cand = _frame(
        [
            ("r1", 0, "1", "1", True),
            ("r2", 0, "1", "1", True),
            ("r4", 0, "1", "x", False),
        ]
    )
    result = compare_frames(base, cand, baseline_version="v1", candidate_version="v2")
    assert result.n_compared == 2
    assert result.baseline_accuracy == pytest.approx(1.0)
    assert result.candidate_accuracy == pytest.approx(1.0)
    assert result.accuracy_delta == pytest.approx(0.0)
    assert result.baseline_accuracy_all == pytest.approx(1.0)
    assert result.candidate_accuracy_all == pytest.approx(2 / 3)
    assert result.accuracy_ok
    assert result.no_regressions
    assert not result.promotable  # a tie is no improvement
    assert "not net positive" in result.reason()
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


# ---------------------------------------------------------------------------
# MLflow write failures
# ---------------------------------------------------------------------------


def test_recorder_logs_rather_than_swallows_a_failed_write(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A gap in MLflow history should be visible, not silent.

    Before this fix, `_Recorder` wrapped every write in `contextlib.suppress`,
    so a broken store dropped metrics without a trace. It must now log a
    warning that names the run and the failing key.
    """
    from convfinqa.tracking.mlflow_log import _Recorder

    class _BrokenMlflow:
        def log_metric(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("store unavailable")

    recorder = _Recorder(_BrokenMlflow(), run_id="run-123")
    with caplog.at_level("WARNING", logger="convfinqa.tracking"):
        recorder.metric("holdout_accuracy", 0.777)

    assert any(
        "failed to log metric" in r.message and "run-123" in r.message
        for r in caplog.records
    )


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


def test_prompts_version_follows_the_champion_not_the_newest_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/healthz` must not report a bundle nobody promoted.

    `v3_1` exists on disk because it was tried and *not* promoted. Resolving the
    unpinned version to "newest prompt module" made the health payload describe
    the champion and the bundle with two different versions, side by side.
    """
    from convfinqa.config import settings
    from convfinqa.tracking import bundle, registry

    monkeypatch.setattr(settings, "prompts_version", None, raising=False)
    monkeypatch.setattr(registry, "champion", lambda *a, **k: "v2")
    assert bundle.prompts_version() == "v2"


def test_explicit_prompts_version_still_beats_the_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override the registry could veto would not be an override."""
    from convfinqa.config import settings
    from convfinqa.tracking import bundle, registry

    monkeypatch.setattr(settings, "prompts_version", "v1", raising=False)
    monkeypatch.setattr(registry, "champion", lambda *a, **k: "v2")
    assert bundle.prompts_version() == "v1"


def test_prompts_version_falls_back_when_champion_has_no_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A champion alias pointing at something with no prompt file degrades."""
    import convfinqa.prompts as prompts_pkg
    from convfinqa.config import settings
    from convfinqa.tracking import bundle, registry

    monkeypatch.setattr(settings, "prompts_version", None, raising=False)
    monkeypatch.setattr(registry, "champion", lambda *a, **k: "v99_nonexistent")
    assert bundle.prompts_version() == prompts_pkg.latest()


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


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def test_turn_usage_rolls_up_stage_metrics() -> None:
    """Per-stage tokens and latency sum to a turn total with a price attached."""
    from convfinqa.tracking.cost import turn_usage

    capture = {
        "triage": {
            "metrics": {"input_tokens": 100, "output_tokens": 20, "latency_ms": 300.0}
        },
        "preprocess": {
            "metrics": {"input_tokens": 400, "output_tokens": 80, "latency_ms": 900.0}
        },
    }
    usage = turn_usage(capture)
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 100
    assert usage["latency_ms"] == pytest.approx(1200.0)
    assert usage["n_stages"] == 2
    assert usage["cost_usd"] > 0


def test_usage_degrades_on_captures_without_metrics() -> None:
    """Committed CSVs predate per-stage metrics; they must contribute zero, not fail."""
    from convfinqa.tracking.cost import aggregate, turn_usage

    assert turn_usage({"triage": {"output": {}}})["total_tokens"] == 0
    totals = aggregate([{}, {"triage": None}])
    assert totals["cost_usd"] == 0
    assert totals["n_turns"] == 2


def test_unknown_model_falls_back_rather_than_raising() -> None:
    """A model id with no published price must not break a run's accounting."""
    from convfinqa.tracking.cost import cost_usd

    assert cost_usd(1_000_000, 0, "some-future-model") > 0
