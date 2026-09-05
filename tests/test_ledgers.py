"""The three append-only ledgers (s10 P4a): schema, append, joins, wiring."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from convfinqa.evalloop import ledgers

REPO = Path(__file__).resolve().parents[1]
DIAG_DIR = REPO / "evaluation" / "diagnostics" / "evalloop"
PRED_DIR = REPO / "evaluation" / "predictions" / "evalloop"

# The two committed diagnosis passes the backfill test replays, and the CSV
# each was diagnosed from (the latest CSV of that version written before it).
FIXTURE_PAIRS = [
    (
        "diagnoses_v8_20260904_225729.jsonl",
        "evalloop-train100-v8·t2p2r5c2-20260904_225226.csv",
    ),
    ("diagnoses_v4_20260902_200345.jsonl", "evalloop-train10-v4-20260902_200228.csv"),
]


@pytest.fixture
def ledger_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "ledgers"
    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(target))
    return target


def _diag(**over: Any) -> dict[str, Any]:
    return {
        "diagnosis_id": "d-1",
        "version": "v8",
        "question_id": "R1_q1",
        "report_id": "R1",
        "turn_index": 1,
        "stage": "retriever",
        "derived_agent": "retriever",
        "label": "retriever/wrong-period",
        "sub_questions": ["a", "b"],
        **over,
    }


# ── the schema is frozen ──────────────────────────────────────────────────


def test_columns_are_pinned_literally() -> None:
    """A silent reorder or rename changes the table every reader depends on."""
    assert ledgers.DIAGNOSES_COLUMNS == (
        "diagnosis_id", "diagnosed_at", "runtime", "version", "prompt_hash",
        "eval_run_id", "diagnosis_run_id", "split", "draw_seed", "report_id",
        "question_id", "turn_index", "diagnoser_model",
        "question", "history_text", "gold_turn_type", "gold_answer", "gold_program",
        "pred_turn_type", "pred_answer", "pred_program", "sub_questions",
        "retrieved", "calc_trajectory",
        "triage_turn_type_ok", "preprocess_skeleton_ok", "preprocess_plan_ok",
        "retriever_operand_recall", "calc_ok", "derived_agent",
        "missing_gold_operands",
        "stage", "label", "what_went_wrong", "evidence", "attribution_reason",
        "fix_hint", "confidence", "gold_suspect", "attribution_disputed",
        "adjudicated", "adjudication_reason",
        "input_tokens", "output_tokens", "cost_usd", "latency_s",
    )  # fmt: skip
    assert ledgers.REWRITES_COLUMNS == (
        "edit_id", "rewrite_id", "proposed_at", "runtime", "campaign",
        "experiment_n", "base_version", "new_version", "prompt_hash_before",
        "prompt_hash_after", "teacher_run_id", "teacher_model",
        "target", "failure_class", "n_diagnoses", "diagnosis_ids", "wilson_lower",
        "rank", "evidence_summary", "prior_attempts",
        "change_kind", "edit_text", "diff", "rationale", "prompt_chars_before",
        "prompt_chars_after", "validate_ok",
        "input_tokens", "output_tokens", "cost_usd", "latency_s",
    )  # fmt: skip
    assert ledgers.GATES_COLUMNS == (
        "gate_id", "gated_at", "runtime", "campaign", "experiment_n", "rewrite_id",
        "baseline_version", "candidate_version", "baseline_hash", "candidate_hash",
        "split", "gate_run_id", "baseline_eval_run_id", "candidate_eval_run_id",
        "n_paired", "baseline_acc", "candidate_acc", "delta_pp", "fixed", "broken",
        "p_value", "ci_low", "ci_high", "flips_by_class", "panel_baseline",
        "panel_candidate",
        "promoted", "reason", "consecutive_rejections", "champion_after",
    )  # fmt: skip
    assert set(ledgers.COLUMNS) == {"diagnoses", "rewrites", "gates"}


def test_unknown_columns_are_refused(ledger_dir: Path) -> None:
    with pytest.raises(ValueError, match="no column"):
        ledgers.append("diagnoses", [_diag(surprise=1)])
    assert not ledgers.path("diagnoses").exists()
    with pytest.raises(ValueError, match="unknown ledger"):
        ledgers.append("verdicts", [{}])


# ── append never rewrites ─────────────────────────────────────────────────


def test_append_only_grows_and_never_touches_an_existing_line(
    ledger_dir: Path,
) -> None:
    ledgers.append("diagnoses", [_diag()])
    first = ledgers.path("diagnoses").read_bytes().splitlines()[0]
    ledgers.append("diagnoses", [_diag(diagnosis_id="d-2", question_id="R2_q0")])
    ledgers.append(
        "rewrites",
        [ledgers.rewrite_row(
            target="retriever", base_version="v8", new_version="v9",
            prompt_before="x", prompt_after="y", diff="", rationale="",
            diagnosis_ids=["d-1"],
        )],
    )  # fmt: skip
    ledgers.append(
        "gates",
        [ledgers.gate_row(
            {"accuracy_delta": 0.01}, baseline_version="v8", candidate_version="v9",
            promoted=False, reason="r", champion_after="v8",
        )],
    )  # fmt: skip
    lines = ledgers.path("diagnoses").read_bytes().splitlines()
    assert len(lines) == 2
    assert lines[0] == first
    assert len(ledgers.path("rewrites").read_text().splitlines()) == 1
    assert len(ledgers.path("gates").read_text().splitlines()) == 1
    # Nested values are JSON strings, so a row is one line and one row.
    row = json.loads(lines[0])
    assert row["sub_questions"] == '["a", "b"]'
    assert list(row) == list(ledgers.DIAGNOSES_COLUMNS)
    table = pd.read_json(ledgers.path("diagnoses"), lines=True)
    assert len(table) == 2


def test_old_lines_load_under_a_widened_schema_with_defaults(
    ledger_dir: Path,
) -> None:
    """A line written before a column existed must still read as a full row."""
    ledger_dir.mkdir(parents=True)
    old = {"diagnosis_id": "d-old", "version": "v2", "question_id": "R9_q0"}
    ledgers.path("diagnoses").write_text(json.dumps(old) + "\n")
    frame = ledgers.load("diagnoses")
    assert list(frame.columns) == list(ledgers.DIAGNOSES_COLUMNS)
    row = frame.iloc[0]
    assert row["runtime"] == "multi_agent"
    assert row["cost_usd"] == 0.0
    assert row["sub_questions"] == "[]"
    assert not bool(row["gold_suspect"])
    # Filters work on the widened frame, and a missing file is an empty table.
    assert len(ledgers.load("diagnoses", version="v2")) == 1
    assert len(ledgers.load("diagnoses", version="v3")) == 0
    assert list(ledgers.load("gates").columns) == list(ledgers.GATES_COLUMNS)
    assert ledgers.load("gates").empty


# ── joins ─────────────────────────────────────────────────────────────────


def test_trace_joins_a_case_to_its_edits_and_verdicts(ledger_dir: Path) -> None:
    ledgers.append(
        "diagnoses",
        [_diag(), _diag(diagnosis_id="d-2", question_id="R2_q0"), _diag(
            diagnosis_id="d-3", question_id="R3_q0")],
    )  # fmt: skip
    rw = ledgers.rewrite_row(
        target="retriever", base_version="v8", new_version="v9",
        prompt_before="x", prompt_after="y", diff="", rationale="",
        diagnosis_ids=["d-1", "d-2"],
    )  # fmt: skip
    other = ledgers.rewrite_row(
        target="preprocess", base_version="v8", new_version="v10",
        prompt_before="x", prompt_after="z", diff="", rationale="",
        diagnosis_ids=["d-3"],
    )  # fmt: skip
    ledgers.append("rewrites", [rw, other])
    ledgers.append(
        "gates",
        [
            ledgers.gate_row(
                {"accuracy_delta": 0.02}, baseline_version="v8",
                candidate_version="v9", promoted=True, reason="ok",
                rewrite_id=rw["rewrite_id"], champion_after="v9",
            ),
            ledgers.gate_row(
                {"accuracy_delta": -0.02}, baseline_version="v8",
                candidate_version="v10", promoted=False, reason="no",
                rewrite_id=other["rewrite_id"], champion_after="v8",
            ),
        ],
    )  # fmt: skip
    by_question = ledgers.trace(question_id="R1_q1")
    assert list(by_question["diagnoses"]["diagnosis_id"]) == ["d-1"]
    assert list(by_question["rewrites"]["rewrite_id"]) == [rw["rewrite_id"]]
    assert list(by_question["gates"]["candidate_version"]) == ["v9"]

    by_edit = ledgers.trace(edit_id=other["edit_id"])
    assert list(by_edit["diagnoses"]["diagnosis_id"]) == ["d-3"]
    assert list(by_edit["gates"]["promoted"]) == [False]
    with pytest.raises(ValueError):
        ledgers.trace()


def test_flips_by_class_arithmetic() -> None:
    flips = {
        "fixed": [{"report_id": "a", "q_order": 0}, {"report_id": "b", "q_order": 1}],
        "broken": [{"report_id": "c", "q_order": 0}, {"report_id": "d", "q_order": 2}],
    }
    classes = {
        ("fixed", "a", 0): "retriever",
        ("fixed", "b", 1): "retriever",
        ("broken", "c", 0): "preprocess",
    }

    def _of(flip: Any, side: str) -> str | None:
        return classes.get((side, flip["report_id"], flip["q_order"]))

    out = ledgers.flips_by_class(flips, _of)
    assert out == {
        "retriever": {"fixed": 2, "broken": 0},
        "preprocess": {"fixed": 0, "broken": 1},
        "unattributed": {"fixed": 0, "broken": 1},
    }
    # Totals always reconcile with the gate's counts.
    assert sum(v["fixed"] for v in out.values()) == 2
    assert sum(v["broken"] for v in out.values()) == 2
    assert ledgers.flips_by_class({}, _of) == {}


def test_attribution_from_frames_reads_the_arm_the_question_failed_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.evalloop import stage_scores

    monkeypatch.setattr(stage_scores, "report_documents", lambda: {})

    def _row(rid: str, ok: bool, retrieved: list[str]) -> dict[str, Any]:
        return {
            "report_id": rid, "turn_index": 0, "question": "q", "gold_answer": "132",
            "pred_answer": "132" if ok else "0", "correct": ok,
            "gold_turn_type": "Program", "pred_turn_type": "program",
            "gold_program": "subtract(243, 111)", "pred_program": "subtract(A, B)",
            "pred_sub_questions": json.dumps(["x", "y"]),
            "retriever_io": json.dumps({"output": {"answers": [
                {"question": "q", "answer": v} for v in retrieved]}}),
        }  # fmt: skip

    base = pd.DataFrame([_row("a", False, ["9", "8"]), _row("b", True, ["243", "111"])])
    cand = pd.DataFrame([_row("a", True, ["243", "111"]), _row("b", False, ["9", "8"])])
    of = ledgers.attribution_from_frames(base, cand)
    assert of({"report_id": "a", "q_order": 0}, "fixed") == "preprocess"
    assert of({"report_id": "b", "q_order": 0}, "broken") == "preprocess"
    assert of({"report_id": "zz", "q_order": 0}, "fixed") is None


# ── row builders ──────────────────────────────────────────────────────────


def test_diagnosis_row_maps_the_teacher_fields_and_reads_the_case() -> None:
    case = pd.Series(
        {
            "report_id": "R1", "question_id": "R1_q2", "turn_index": 2,
            "question": "what is it?", "history_text": "h", "gold_answer": "5",
            "gold_program": "divide(10, 2)", "pred_answer": "4",
            "pred_program": "divide(A, B)", "gold_turn_type": "Program",
            "pred_turn_type": "program", "run_id": "eval-1", "split": "train",
            "pred_sub_questions": json.dumps(["ten", "two"]),
            "retriever_io": json.dumps({"output": {"answers": [
                {"question": "ten", "answer": "10"}]}}),
            "calculator_io": json.dumps({"trajectory": [{"op": "divide"}]}),
            "triage_turn_type_ok": True, "preprocess_skeleton_ok": True,
            "preprocess_plan_ok": False, "retriever_operand_recall": 0.5,
            "calculator_ok": False, "prior_gold_answers": [],
        }
    )  # fmt: skip
    d = {
        "failed_agent": "retriever", "failure_mode": "retriever/wrong-value",
        "proposed_rule": "look harder", "what_went_wrong": "w", "evidence": "e",
        "attribution_reason": "a", "confidence": 0.8, "gold_suspect": False,
        "derived_agent": "retriever", "attribution_disputed": False,
        "adjudicated": True, "adjudication_reason": "asked",
    }  # fmt: skip
    usage = {"usage": {"input_tokens": 100, "output_tokens": 20},
             "total_cost_usd": 0.05, "duration_ms": 1500}  # fmt: skip
    row = ledgers.diagnosis_row(
        d, case, version="v8", prompt_hash="abcd1234", diagnosis_run_id="diag-1",
        usage=usage,
    )  # fmt: skip
    assert row["stage"] == "retriever"
    assert row["label"] == "retriever/wrong-value"
    assert row["fix_hint"] == "look harder"
    assert row["eval_run_id"] == "eval-1" and row["split"] == "train"
    assert row["sub_questions"] == ["ten", "two"]
    assert row["retrieved"] == [{"question": "ten", "answer": "10"}]
    assert row["calc_trajectory"] == [{"op": "divide"}]
    assert row["missing_gold_operands"] == ["2"]
    assert row["calc_ok"] is False and row["retriever_operand_recall"] == 0.5
    assert row["adjudicated"] and row["adjudication_reason"] == "asked"
    assert (row["input_tokens"], row["output_tokens"]) == (100, 20)
    assert row["cost_usd"] == 0.05 and row["latency_s"] == 1.5
    written = ledgers.normalise("diagnoses", row)
    assert json.loads(written["retrieved"]) == [{"question": "ten", "answer": "10"}]


def test_gate_row_converts_gate_statistics() -> None:
    stats = {
        "evidence_split": "test", "n_compared": 349, "baseline_accuracy": 0.7,
        "candidate_accuracy": 0.72, "accuracy_delta": 0.02, "fail_to_pass": 12,
        "pass_to_fail": 5, "cluster_p_one_sided": 0.03, "delta_ci_lo": 0.001,
        "delta_ci_hi": 0.04,
    }  # fmt: skip
    row = ledgers.gate_row(
        stats, baseline_version="v8", candidate_version="v9", promoted=True,
        reason="PROMOTE", label="c03-e02", campaign="c03", champion_after="v9",
    )  # fmt: skip
    assert row["delta_pp"] == 2.0 and row["n_paired"] == 349
    assert (row["fixed"], row["broken"]) == (12, 5)
    assert row["p_value"] == 0.03 and row["experiment_n"] == 2
    assert row["consecutive_rejections"] == 0 and row["champion_after"] == "v9"
    assert ledgers.experiment_number("c01-e10") == 10
    assert ledgers.experiment_number(None) is None


def test_rewrite_row_refuses_an_unknown_change_kind() -> None:
    with pytest.raises(ValueError):
        ledgers.rewrite_row(
            target="retriever", base_version="v8", new_version="v9",
            prompt_before="x", prompt_after="y", diff="", rationale="",
            change_kind="vibes",
        )  # fmt: skip


def test_log_rows_to_run_mirrors_the_batch_and_its_scalars() -> None:
    logged: dict[str, Any] = {"artifacts": [], "metrics": {}}

    class Rec:
        def artifact(self, p: Path) -> None:
            logged["artifacts"].append((p.name, p.read_text().count("\n")))

        def metrics(self, values: dict[str, float]) -> None:
            logged["metrics"].update(values)

    gate = ledgers.normalise(
        "gates",
        ledgers.gate_row(
            {"accuracy_delta": 0.015, "fail_to_pass": 3, "pass_to_fail": 1,
             "cluster_p_one_sided": 0.2},
            baseline_version="v8", candidate_version="v9", promoted=False,
            reason="r", champion_after="v8",
        ),
    )  # fmt: skip
    ledgers.log_rows_to_run(Rec(), [gate], "gates")
    assert logged["artifacts"] == [("ledger_rows.jsonl", 1)]
    assert logged["metrics"]["ledger_gates_n_rows"] == 1.0
    assert logged["metrics"]["ledger_delta_pp"] == 1.5
    assert logged["metrics"]["ledger_fixed"] == 3.0
    assert logged["metrics"]["ledger_p_value"] == 0.2
    ledgers.log_rows_to_run(Rec(), [], "gates")  # nothing to mirror, no error


# ── backfill ──────────────────────────────────────────────────────────────


@pytest.fixture
def backfill_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    """Two committed diagnosis passes and their CSVs, copied into a temp tree."""
    from convfinqa.evalloop import stage_scores

    diag = tmp_path / "diag"
    pred = tmp_path / "pred"
    diag.mkdir()
    pred.mkdir()
    for d, c in FIXTURE_PAIRS:
        shutil.copy(DIAG_DIR / d, diag / d)
        shutil.copy(PRED_DIR / c, pred / c)
    # A decoy: a *later* CSV of the same version must not be chosen.
    shutil.copy(
        PRED_DIR / FIXTURE_PAIRS[1][1], pred / "evalloop-train10-v4-20260902_230000.csv"
    )
    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(diag))
    monkeypatch.setattr(stage_scores, "report_documents", lambda: {})
    return {"diag": diag, "pred": pred}


def test_backfill_seeds_the_diagnoses_ledger_and_is_idempotent(
    backfill_fixture: dict[str, Path],
) -> None:
    diag, pred = backfill_fixture["diag"], backfill_fixture["pred"]
    n_lines = sum(
        len((DIAG_DIR / d).read_text().splitlines()) for d, _ in FIXTURE_PAIRS
    )
    first = ledgers.backfill_ledgers(
        diagnostics_dir=diag, predictions_dir=pred, use_mlflow=False
    )
    assert first["diagnoses"] == n_lines
    assert first["diagnoses_no_csv"] == 0
    assert first["mlflow_reachable"] == 0
    table = ledgers.load("diagnoses")
    assert len(table) == n_lines
    assert set(table["version"]) == {"v8", "v4"}
    assert set(table["runtime"]) == {"multi_agent"}
    # Inputs and gold flags came from the CSV, not the diagnosis file.
    v8 = table[table["version"] == "v8"]
    assert (v8["question"].str.len() > 0).all()
    assert v8["eval_run_id"].nunique() == 1 and v8["eval_run_id"].iloc[0]
    assert (v8["split"] == "train").all()
    assert v8["retriever_operand_recall"].notna().any()
    assert set(v8["diagnosed_at"]) == {"2026-09-04T22:57:29"}
    # The attributed agent's prompt hash; a verdict naming no agent has none.
    agents = ("triage", "preprocess", "retriever", "calculator")
    named = v8[v8["derived_agent"].isin(agents)]
    assert len(named) > 0 and (named["prompt_hash"].str.len() == 8).all()
    assert (v8[~v8["derived_agent"].isin(agents)]["prompt_hash"] == "").all()
    # The teacher's outputs carried over under their ledger names.
    assert v8["stage"].isin(("triage", "preprocess", "retriever", "calculator")).all()
    assert (v8["label"].str.len() > 0).all()

    before = ledgers.path("diagnoses").read_bytes()
    second = ledgers.backfill_ledgers(
        diagnostics_dir=diag, predictions_dir=pred, use_mlflow=False
    )
    assert second["diagnoses"] == 0
    assert second["diagnoses_existing"] == n_lines
    assert ledgers.path("diagnoses").read_bytes() == before


def test_backfill_skips_and_counts_a_pass_with_no_csv(
    backfill_fixture: dict[str, Path],
) -> None:
    diag, pred = backfill_fixture["diag"], backfill_fixture["pred"]
    for c in pred.glob("*-v4-*.csv"):
        c.unlink()
    counts = ledgers.backfill_ledgers(
        diagnostics_dir=diag, predictions_dir=pred, use_mlflow=False
    )
    assert counts["diagnoses_no_csv"] == 7
    assert counts["diagnoses"] == 48
    assert set(ledgers.load("diagnoses")["version"]) == {"v8"}


# ── the memory reads the ledgers first ────────────────────────────────────


def test_diagnoses_for_agent_and_fault_history_read_the_ledger_first(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import ledger

    hashes = {("v8", "retriever"): "r5hash", ("v9", "retriever"): "r6hash",
              ("v8", "preprocess"): "p2hash", ("v9", "preprocess"): "p2hash"}  # fmt: skip
    monkeypatch.setattr(ledger, "_agent_prompt_hash", lambda v, a: hashes.get((v, a)))

    def _boom() -> Any:
        raise AssertionError("the store must not be consulted when the ledger has rows")

    monkeypatch.setattr(ledger, "_client", _boom)
    rows = [
        _diag(diagnosis_id="1", version="v8", question_id="a", prompt_hash="r5hash",
              diagnosis_run_id="run-a", label="m1", fix_hint="fix1"),
        _diag(diagnosis_id="2", version="v8", question_id="b", prompt_hash="p2hash",
              derived_agent="preprocess", stage="preprocess", diagnosis_run_id="run-a"),
        _diag(diagnosis_id="3", version="v9", question_id="c", prompt_hash="p2hash",
              derived_agent="preprocess", stage="preprocess", diagnosis_run_id="run-b"),
        _diag(diagnosis_id="4", version="v9", question_id="d", prompt_hash="r6hash",
              diagnosis_run_id="run-b"),
        _diag(diagnosis_id="5", version="v9", question_id="e", prompt_hash="",
              derived_agent="gold_suspect", diagnosis_run_id="run-b"),
    ]  # fmt: skip
    ledgers.append("diagnoses", rows)

    got = ledger.diagnoses_for_agent("retriever", "v8")
    assert [g["report_id"] for g in got] == ["R1"]
    assert got[0]["failure_mode"] == "m1" and got[0]["proposed_rule"] == "fix1"
    # preprocess shares its prompt across v8 and v9, so both passes count.
    assert len(ledger.diagnoses_for_agent("preprocess", "v9")) == 2

    pooled = ledger.fault_history("v9")
    # run-b has 2 attributed cases (the gold_suspect is excluded); run-a has 2.
    pre = pooled["preprocess"]
    assert (pre["faults"], pre["cases"], pre["n_runs"]) == (2, 4, 2)
    assert pre["versions"] == ["v8", "v9"] and pre["rate"] == 0.5
    assert 0.0 < pre["score"] < pre["rate"]
    assert pooled["retriever"]["faults"] == 1 and pooled["retriever"]["cases"] == 2
    excluded = ledger.fault_history("v9", exclude_run_id="run-b")
    assert excluded["retriever"]["cases"] == 0
    assert excluded["preprocess"]["cases"] == 2


def test_attempts_reads_the_rewrites_and_gates_ledgers_first(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import ledger

    monkeypatch.setattr(ledger, "_client", lambda: (_ for _ in ()).throw(OSError()))
    rw = ledgers.rewrite_row(
        target="retriever", base_version="v8", new_version="v9", prompt_before="x",
        prompt_after="y", diff="", rationale="why", edit_text="what",
    )  # fmt: skip
    ledgers.append("rewrites", [rw])
    ledgers.append(
        "gates",
        [ledgers.gate_row(
            {"accuracy_delta": -0.013, "fail_to_pass": 4, "pass_to_fail": 9,
             "cluster_p_one_sided": 0.7},
            baseline_version="v8", candidate_version="v9", promoted=False,
            reason="REJECT", rewrite_id=rw["rewrite_id"], champion_after="v8",
            gate_run_id="gate-run",
        )],
    )  # fmt: skip
    rows = ledger.attempts(target_agent="retriever")
    assert len(rows) == 1
    got = rows[0]
    assert got["outcome"] == "rejected" and got["verdict"] == "REJECT"
    assert got["accuracy_delta"] == pytest.approx(-0.013)
    assert (got["fixed"], got["broken"]) == (4, 9)
    assert got["summary_of_changes"] == "what" and got["rationale"] == "why"
    assert ledger.attempts(target_agent="preprocess") == []
    text = ledger.ledger_text("retriever")
    assert "v9 — REJECTED" in text


# ── wiring: each step appends exactly its rows ───────────────────────────


class _FakeRun:
    class info:  # noqa: N801 — mirrors mlflow's own attribute name
        run_id = "r1"

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class _FakeMlflow:
    def start_run(self, run_name: str = "") -> _FakeRun:
        return _FakeRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        return None

    def log_params(self, params: dict[str, str]) -> None:
        return None

    def get_experiment_by_name(self, name: str) -> object:
        return object()

    def set_experiment(self, name: str) -> None:
        return None


@pytest.mark.asyncio
async def test_diagnose_run_appends_one_ledger_row_per_case(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import teacher
    from convfinqa.tracking import mlflow_log

    monkeypatch.setattr(mlflow_log, "_mlflow", lambda: _FakeMlflow())
    monkeypatch.setattr(teacher, "DIAGNOSTICS_DIR", tmp_path / "per-run")
    monkeypatch.setattr(teacher, "prior_diagnoses", lambda *a, **k: [])
    monkeypatch.setattr(ledgers, "agent_prompt_hash", lambda v, a: f"{a[:1]}hash")
    monkeypatch.setattr(ledgers, "eval_run_param", lambda r, k: "2027")

    cases = pd.DataFrame(
        [
            {
                "report_id": f"R{i}", "turn_index": i, "question_id": f"R{i}_q{i}",
                "question": "q", "gold_answer": "1", "pred_answer": "2",
                "gold_program": "add(1, 1)", "pred_program": "", "run_id": "eval-7",
                "split": "train", "prior_gold_answers": [],
            }
            for i in range(3)
        ]
    )  # fmt: skip
    monkeypatch.setattr(teacher, "first_wrong_cases", lambda _p: cases)
    monkeypatch.setattr(
        teacher, "case_payload", lambda row: {"derived_attribution": "preprocess"}
    )

    async def fake_case(payload: Any, memory: str, refs: Any = None) -> Any:
        return (
            SimpleNamespace(
                failed_agent="preprocess", failure_mode="m", attribution_reason="r",
                what_went_wrong="w", evidence="e", proposed_rule="p",
                confidence=0.9, gold_suspect=False,
                model_dump=lambda: {
                    "failed_agent": "preprocess", "failure_mode": "m",
                    "what_went_wrong": "w", "evidence": "e",
                    "attribution_reason": "r", "proposed_rule": "p",
                    "confidence": 0.9, "gold_suspect": False,
                },
            ),
            {"usage": {"input_tokens": 10, "output_tokens": 2}, "total_cost_usd": 0.01},
        )  # fmt: skip

    monkeypatch.setattr(teacher, "_diagnose_case", fake_case)

    summary = await teacher.diagnose_run("ignored.csv", "v2", concurrency=2)

    table = ledgers.load("diagnoses")
    assert len(table) == 3
    assert list(table["question_id"]) == ["R0_q0", "R1_q1", "R2_q2"]
    assert set(table["runtime"]) == {"multi_agent"}
    assert set(table["version"]) == {"v2"}
    assert set(table["prompt_hash"]) == {"phash"}
    assert set(table["eval_run_id"]) == {"eval-7"}
    assert set(table["diagnosis_run_id"]) == {"r1"}
    assert set(table["draw_seed"]) == {2027}
    assert set(table["stage"]) == {"preprocess"} and set(table["label"]) == {"m"}
    assert table["cost_usd"].sum() == pytest.approx(0.03)
    # The per-run file is still written, and carries the same ids.
    per_run = [
        json.loads(line)
        for line in Path(summary["diagnoses_path"]).read_text().splitlines()
    ]
    assert [d["diagnosis_id"] for d in per_run] == list(table["diagnosis_id"])
    assert ledgers.load("rewrites").empty and ledgers.load("gates").empty


def test_rewrite_ledger_row_from_a_proposal() -> None:
    """The propose step's row builder, tested without the SDK harness."""
    from convfinqa.evalloop import teacher

    output = teacher.PromptRewrite(
        prompt="new " * 60, rationale="why", summary_of_changes="what"
    )
    targeted = [
        {"diagnosis_id": "d-1", "failure_mode": "retriever/wrong-period"},
        {"diagnosis_id": "d-2", "failure_mode": "retriever/wrong-period"},
        {"diagnosis_id": "d-3", "failure_mode": "retriever/wrong-value"},
    ]
    pooled = {
        "retriever": {"score": 0.31}, "preprocess": {"score": 0.28},
        "triage": {"score": 0.0}, "calculator": {"score": 0.1},
    }  # fmt: skip
    row = teacher._rewrite_ledger_row(
        targeted, target="retriever", base_version="v8", new_version="v9",
        prompt_before="old prompt", output=output, diff="--- a\n+++ b\n",
        prior_attempts=[{"version": "v6", "outcome": "rejected"}], pooled=pooled,
        validate_ok=True, campaign="c03", label="c03-e04", teacher_run_id="t-1",
        usage={"usage": {"input_tokens": 5, "output_tokens": 1}, "total_cost_usd": 0.2},
    )  # fmt: skip
    assert row["failure_class"] == "retriever/wrong-period"
    assert row["diagnosis_ids"] == ["d-1", "d-2", "d-3"] and row["n_diagnoses"] == 3
    assert row["rank"] == 1 and row["wilson_lower"] == 0.31
    assert row["experiment_n"] == 4 and row["campaign"] == "c03"
    assert row["prior_attempts"] == [{"version": "v6", "outcome": "rejected"}]
    assert row["prompt_chars_before"] == len("old prompt")
    assert row["prompt_hash_before"] != row["prompt_hash_after"]
    assert row["edit_text"] == "what" and row["rationale"] == "why"
    assert row["validate_ok"] and row["cost_usd"] == 0.2
    ledgers.normalise("rewrites", row)  # every key is a column


def test_gate_path_appends_one_gates_row_with_flip_classes(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import stage_scores, teacher
    from convfinqa.tracking import mlflow_log

    monkeypatch.setattr(mlflow_log, "_mlflow", lambda: _FakeMlflow())
    monkeypatch.setattr(stage_scores, "report_documents", lambda: {})
    monkeypatch.setattr(ledgers, "bundle_hash", lambda v: f"{v}-hash")

    def _row(rid: str, ok: bool, retrieved: list[str]) -> dict[str, Any]:
        return {
            "report_id": rid, "turn_index": 0, "question": "q", "gold_answer": "132",
            "pred_answer": "132" if ok else "0", "correct": ok,
            "gold_turn_type": "Program", "pred_turn_type": "program",
            "gold_program": "subtract(243, 111)", "pred_program": "subtract(A, B)",
            "split": "test", "run_id": "eval-" + rid,
            "pred_sub_questions": json.dumps(["x", "y"]),
            "retriever_io": json.dumps({"output": {"answers": [
                {"question": "q", "answer": v} for v in retrieved]}}),
        }  # fmt: skip

    base = pd.DataFrame([_row("a", False, ["9", "8"]), _row("b", True, ["243", "111"]),
                         _row("c", True, ["243", "111"])])  # fmt: skip
    cand = pd.DataFrame([_row("a", True, ["243", "111"]), _row("b", True, ["243", "111"]),
                         _row("c", False, ["9", "8"])])  # fmt: skip
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    verdict, comparison = teacher.gate_targeted(
        a, b, target_agent="retriever", baseline_version="v3_1", candidate_version="v4"
    )
    run_id = teacher.log_gate_verdict(
        verdict, comparison=comparison, campaign="c09", label="c09-e01",
        rewrite_id="rw-42", consecutive_rejections=2, champion_after="v3_1",
    )  # fmt: skip
    assert run_id == "r1"
    table = ledgers.load("gates")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["rewrite_id"] == "rw-42" and row["gate_run_id"] == "r1"
    assert row["campaign"] == "c09" and row["experiment_n"] == 1
    assert (row["baseline_version"], row["candidate_version"]) == ("v3_1", "v4")
    assert (row["baseline_hash"], row["candidate_hash"]) == ("v3_1-hash", "v4-hash")
    assert row["split"] == "test" and row["n_paired"] == 3
    assert (row["fixed"], row["broken"]) == (1, 1)
    assert row["delta_pp"] == 0.0 and not bool(row["promoted"])
    assert row["consecutive_rejections"] == 2 and row["champion_after"] == "v3_1"
    assert (row["baseline_eval_run_id"], row["candidate_eval_run_id"]) == (
        "eval-a",
        "eval-a",
    )
    assert json.loads(row["flips_by_class"]) == {
        "preprocess": {"fixed": 1, "broken": 1}
    }
    assert "retriever_operand_recall" in json.loads(row["panel_candidate"])
    assert ledgers.load("diagnoses").empty and ledgers.load("rewrites").empty
