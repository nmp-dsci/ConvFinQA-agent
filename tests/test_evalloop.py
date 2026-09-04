"""The eval loop (M1): manifest determinism, run-identity traces, the gate."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from convfinqa.evalloop import gate, splits
from convfinqa.evalloop.runner import first_wrong_index

# ── splits ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def manifest() -> dict:
    return splits.build_manifest(target_questions=200, seed=2026)


def test_manifest_is_deterministic(manifest: dict) -> None:
    again = splits.build_manifest(target_questions=200, seed=2026)
    assert manifest["splits"] == again["splits"]
    assert manifest["dataset_hash"] == again["dataset_hash"]


def test_splits_are_disjoint_and_clean(manifest: dict) -> None:
    ids = [rid for s in splits.SPLIT_NAMES for rid in manifest["splits"][s]]
    assert len(ids) == len(set(ids)), "a report id landed in two splits"
    from convfinqa.data.loader import qa_data

    seen = set(qa_data["report_id"])
    assert not (set(ids) & seen), "an optimiser-seen conversation leaked in"
    for s in splits.SPLIT_NAMES:
        stat = manifest["stats"][s]
        assert stat["n_questions"] >= 200
        assert 0.0 < stat["type2_share"] < 1.0, "a split lost a whole stratum"


def test_manifest_refuses_overwrite(manifest: dict, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    splits.write_manifest(manifest, path)
    with pytest.raises(FileExistsError):
        splits.write_manifest(manifest, path)
    splits.write_manifest(manifest, path, force=True)
    loaded = json.loads(path.read_text())
    assert loaded["splits"] == manifest["splits"]
    assert splits.load_manifest(path)["seed"] == 2026


def test_split_report_ids_truncates_in_manifest_order(
    manifest: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "m.json"
    splits.write_manifest(manifest, path)
    ten = splits.split_report_ids("train", n_reports=10, path=path)
    assert ten == manifest["splits"]["train"][:10]
    with pytest.raises(ValueError, match="Unknown split"):
        splits.split_report_ids("dev", path=path)


def test_split_report_ids_by_question_budget(manifest: dict, tmp_path: Path) -> None:
    from convfinqa.data.loader import training_data

    path = tmp_path / "m.json"
    splits.write_manifest(manifest, path)
    counts = training_data().groupby("report_id")["question_id"].size().to_dict()

    fifty = splits.split_report_ids("train", n_questions=50, path=path)
    assert fifty == manifest["splits"]["train"][: len(fifty)]
    total = sum(counts[rid] for rid in fifty)
    assert total >= 50
    # dropping the last report must fall (or stay, if it alone met the budget) short
    assert total - counts[fifty[-1]] < 50

    with pytest.raises(ValueError, match="at most one"):
        splits.split_report_ids("train", n_reports=5, n_questions=50, path=path)


# ── runner pieces ───────────────────────────────────────────────────────


def test_first_wrong_index_marks_the_cascade_root() -> None:
    assert first_wrong_index([True, True, True]) is None
    assert first_wrong_index([True, False, True]) == 1
    assert first_wrong_index([False, False]) == 0


def test_trace_store_records_run_identity(tmp_path: Path) -> None:
    from convfinqa.tracking.traces import TraceStore

    store = TraceStore(tmp_path / "t.db")
    trace_id = store.record(
        report_id="r1",
        turn_index=0,
        question="q?",
        capture={},
        answer="42",
        source="eval",
        gold_answer="42",
        correct=True,
        bundle={"prompts_version": "v2"},
        run_id="run-abc",
        split="train",
        question_id="r1_q0",
        model_version_id="v2",
    )
    row = store._conn.execute(
        "SELECT run_id, split, question_id, model_version_id FROM turns "
        "WHERE trace_id = ?",
        (trace_id,),
    ).fetchone()
    store.close()
    assert tuple(row) == ("run-abc", "train", "r1_q0", "v2")


def test_old_trace_db_is_widened_not_broken(tmp_path: Path) -> None:
    """A pre-loop turns table gains the identity columns on open."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE turns (trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
        "source TEXT NOT NULL, session_id TEXT, report_id TEXT NOT NULL, "
        "turn_index INTEGER NOT NULL, question TEXT NOT NULL, answer TEXT, "
        "program TEXT, gold_answer TEXT, correct INTEGER, bundle_id TEXT, "
        "bundle TEXT, latency_ms REAL, total_tokens INTEGER, error TEXT, "
        "capture TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    from convfinqa.tracking.traces import TraceStore

    store = TraceStore(db)
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(turns)").fetchall()}
    store.close()
    assert {"run_id", "split", "question_id", "model_version_id"} <= cols


# ── the gate ────────────────────────────────────────────────────────────


def _frame(rows: list[tuple[str, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_id": r,
                "turn_index": t,
                "question": "q",
                "gold_answer": "1",
                "pred_answer": "1" if ok else "2",
                "correct": ok,
            }
            for r, t, ok in rows
        ]
    )


def test_mcnemar_exact_p() -> None:
    assert gate.mcnemar_exact_p(0, 0) == 1.0
    assert gate.mcnemar_exact_p(0, 3) == pytest.approx(0.25)
    assert gate.mcnemar_exact_p(1, 1) == pytest.approx(1.0)
    assert gate.mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)


def test_gate_counts_flips_from_csvs(tmp_path: Path) -> None:
    base = _frame([("a", 0, True), ("a", 1, False), ("b", 0, False)])
    cand = _frame([("a", 0, True), ("a", 1, True), ("b", 0, True)])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    result, stats = gate.gate_runs(
        a, b, baseline_version="v2", candidate_version="v3_1"
    )
    assert stats["pass_to_fail"] == 0
    assert stats["fail_to_pass"] == 2
    assert stats["mcnemar_p"] == pytest.approx(0.5)
    assert result.promotable


def test_promote_winner_retains_champion_on_a_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.tracking import registry

    reg = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg, raising=False)
    monkeypatch.setattr(registry, "_mirror_to_mlflow", lambda v: None)

    base = _frame([("a", 0, True), ("a", 1, False)])
    cand = _frame([("a", 0, False), ("a", 1, True)])  # one flip each way
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    result, stats = gate.gate_runs(
        a, b, baseline_version="v2", candidate_version="v3_1"
    )
    assert not result.promotable  # one flip each way: a tie, no net gain
    outcome = gate.promote_winner(result, stats)
    assert outcome["winner"] == "v2"
    assert outcome["promoted"] is False
    assert outcome["comparison"]["pass_to_fail"] == 1


# ── the teacher (M2) ────────────────────────────────────────────────────


def test_first_wrong_cases_picks_one_row_per_failing_report(tmp_path: Path) -> None:
    from convfinqa.evalloop import teacher

    df = pd.DataFrame(
        [
            {
                "report_id": "a",
                "turn_index": 0,
                "first_wrong_turn": 1.0,
                "correct": True,
                "gold_answer": "1",
            },
            {
                "report_id": "a",
                "turn_index": 1,
                "first_wrong_turn": 1.0,
                "correct": False,
                "gold_answer": "2",
            },
            {
                "report_id": "a",
                "turn_index": 2,
                "first_wrong_turn": 1.0,
                "correct": False,
                "gold_answer": "3",
            },
            {
                "report_id": "b",
                "turn_index": 0,
                "first_wrong_turn": None,
                "correct": True,
                "gold_answer": "4",
            },
        ]
    )
    path = tmp_path / "run.csv"
    df.to_csv(path, index=False)
    cases = teacher.first_wrong_cases(path)
    assert list(cases.report_id) == ["a"]
    assert list(cases.turn_index) == [1]


def _diag_file(tmp_path: Path, name: str, agents: list[str]) -> Path:
    import json as _json

    p = tmp_path / name
    p.write_text("".join(_json.dumps({"failed_agent": a}) + "\n" for a in agents))
    return p


def test_gate_targeted_reports_the_target_metric_but_promotes_on_significance(
    tmp_path: Path,
) -> None:
    """The per-agent metric is evidence, not a second route to promotion.

    Under M2 a moved target metric could promote a challenger on its own, and
    that is how three versions were promoted on evidence whose interval
    contained zero. Now one rule decides — net positive AND one-sided clustered
    McNemar p < 0.05 — and a single fixed question cannot clear it."""
    from convfinqa.evalloop import teacher

    base = _frame([("a", 0, True), ("a", 1, False), ("b", 0, False)])
    cand = _frame([("a", 0, True), ("a", 1, True), ("b", 0, False)])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    bd = _diag_file(tmp_path, "bd.jsonl", ["retriever", "retriever", "triage"])
    cd = _diag_file(tmp_path, "cd.jsonl", ["retriever", "triage"])
    verdict, comparison = teacher.gate_targeted(
        a,
        b,
        target_agent="retriever",
        baseline_diagnoses=bd,
        candidate_diagnoses=cd,
        baseline_version="v3_1",
        candidate_version="v4",
    )
    # these frames have no retriever_io, so recall is 0.0 on both sides: a tie
    assert verdict["target_metric_before"] == verdict["target_metric_after"] == 0.0
    assert not verdict["target_moved"]
    # one fixed question, zero broken: net positive, but p = 0.5 one-sided
    assert verdict["comparison"]["fail_to_pass"] == 1
    assert verdict["cluster_p_one_sided"] > 0.05
    assert not verdict["promotable"]
    # attribution evidence still recorded as secondary
    assert verdict["baseline_target_faults"] == 2
    assert verdict["candidate_target_faults"] == 1
    assert comparison.candidate_version == "v4"


def test_write_version_module_changes_exactly_one_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import teacher

    monkeypatch.setattr(teacher, "REPO_ROOT", tmp_path)
    (tmp_path / "src" / "convfinqa" / "prompts").mkdir(parents=True)
    path = teacher._write_version_module(
        "v99",
        base_version="v3_1",
        target="retriever",
        prompt='Return an answer. Backslash \\ and a "quote" survive.',
    )
    text = path.read_text()
    assert "do not hand-edit" in text
    assert "RETRIEVER_PROMPT = " in text
    assert "TRIAGE_PROMPT,\n" in text  # imported unchanged
    assert "_BASE" not in text  # replaced outright, not appended to
    with pytest.raises(SystemExit):  # refuses to overwrite
        teacher._write_version_module(
            "v99", base_version="v3_1", target="retriever", prompt="x" * 300
        )


def test_write_version_module_survives_quotes_and_backslashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated module must import, whatever the writer put in the prompt.

    The prompt is written into a triple-quoted literal for readability, so the
    two sequences that could terminate it early have to be neutralised — a
    rewrite containing a regex or an embedded docstring would otherwise produce
    a module that fails to parse, days after the run that made it."""
    import importlib.util

    monkeypatch.setattr(teacher_module(), "REPO_ROOT", tmp_path)
    (tmp_path / "src" / "convfinqa" / "prompts").mkdir(parents=True)
    nasty = 'Match \\d+ and never emit """ or a trailing quote: "'
    path = teacher_module()._write_version_module(
        "v98", base_version="v3_1", target="calculator", prompt=nasty
    )
    spec = importlib.util.spec_from_file_location("v98", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["v98_probe"] = module
    source = path.read_text()
    compile(source, str(path), "exec")  # the module parses
    assert "\\d+" in source


def teacher_module():  # noqa: ANN201 — test helper
    from convfinqa.evalloop import teacher

    return teacher


# ── per-agent versions + stage scores (M2.5) ────────────────────────────


def test_prompt_ledger_is_idempotent_and_orders_seqs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.tracking import prompt_ledger, registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "reg.json", raising=False)
    first = prompt_ledger.ensure("v3_1")
    again = prompt_ledger.ensure("v3_1")
    assert first == again  # same hashes, no new entries
    assert prompt_ledger.composition_string(first) == "t1.p1.r1.c1"
    v4 = prompt_ledger.ensure("v4")
    assert prompt_ledger.composition_string(v4) == "t1.p1.r2.c1"
    assert prompt_ledger.changed_agents("v3_1", "v4") == ["retriever"]
    doc = registry.load()
    r2 = doc.agent_prompts["retriever"][1]
    assert r2["parent"] == "r1" and r2["first_seen_in"] == "v4"


def test_stage_scores_panel_from_a_tiny_frame() -> None:
    import json as _json

    from convfinqa.evalloop import stage_scores

    retr = _json.dumps({"output": {"answers": [{"question": "q", "answer": "243"}]}})
    df = pd.DataFrame(
        [
            {  # program turn: skeleton right, one of two operands retrieved
                "report_id": "a",
                "turn_index": 0,
                "gold_answer": "132",
                "correct": False,
                "gold_turn_type": "Program",
                "pred_turn_type": "program",
                "gold_program": "subtract(243, 111)",
                "pred_program": "subtract(A, B)",
                "retriever_io": retr,
            },
            {  # number turn: retriever surfaced the gold value, answer right
                "report_id": "a",
                "turn_index": 1,
                "gold_answer": "243",
                "correct": True,
                "gold_turn_type": "Number",
                "pred_turn_type": "number",
                "gold_program": "243",
                "pred_program": "",
                "retriever_io": retr,
            },
        ]
    )
    m = stage_scores.run_metrics(df)
    assert m["acc_triage_turn_type"] == 1.0
    assert m["acc_preprocess_skeleton"] == 1.0
    assert m["retriever_operand_recall"] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert m["acc_calculator_exec"] == 0.0  # the one program turn was wrong


def test_gold_document_operands_drop_history_and_constants() -> None:
    from convfinqa.evalloop.stage_scores import gold_document_operands

    ops = gold_document_operands(
        "subtract(500, 300), divide(#0, const_100)", prior_gold_answers=["300"]
    )
    assert ops == ["500"]  # 300 came from history, const_100 and #0 are not lookups


def test_gate_targeted_uses_the_deterministic_metric(tmp_path: Path) -> None:
    import json as _json

    from convfinqa.evalloop import teacher

    def _row(rid, t, ok, retrieved):
        return {
            "report_id": rid,
            "turn_index": t,
            "question": "q",
            "gold_answer": "132",
            "pred_answer": "132" if ok else "0",
            "correct": ok,
            "gold_turn_type": "Program",
            "pred_turn_type": "program",
            "gold_program": "subtract(243, 111)",
            "pred_program": "subtract(A, B)",
            "split": "test",
            "retriever_io": _json.dumps(
                {
                    "output": {
                        "answers": [{"question": "q", "answer": v} for v in retrieved]
                    }
                }
            ),
        }

    base = pd.DataFrame(
        [_row("a", 0, False, ["9"]), _row("b", 0, True, ["243", "111"])]
    )
    cand = pd.DataFrame(
        [_row("a", 0, True, ["243", "111"]), _row("b", 0, True, ["243", "111"])]
    )
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    verdict, _ = teacher.gate_targeted(
        a,
        b,
        target_agent="retriever",
        baseline_version="v3_1",
        candidate_version="v4",
    )
    assert verdict["target_metric"] == "retriever_operand_recall"
    assert verdict["target_metric_after"] > verdict["target_metric_before"]
    assert verdict["evidence_split"] == "test"
    assert verdict["target_moved"]


# ── kappa + release (M2 trust, M3 gate) ─────────────────────────────────


def test_cohens_kappa_and_sheet_roundtrip(tmp_path: Path) -> None:
    import json as _json

    from convfinqa.evalloop import kappa

    assert kappa.cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    # 50% observed agreement over two balanced classes -> kappa 0
    assert kappa.cohens_kappa(
        ["a", "b", "a", "b"], ["a", "a", "b", "b"]
    ) == pytest.approx(0.0)

    d = tmp_path / "d.jsonl"
    d.write_text(
        "".join(
            _json.dumps(
                {
                    "report_id": f"r{i}",
                    "turn_index": 0,
                    "version": "v3_1",
                    "failed_agent": "retriever",
                    "failure_mode": "retriever/wrong-value",
                    "what_went_wrong": "x",
                }
            )
            + "\n"
            for i in range(5)
        )
    )
    sheet = kappa.make_sheet([d], out_path=tmp_path / "sheet.csv", n=3)
    df = pd.read_csv(sheet)
    assert len(df) == 3 and (df["human_agent"].isna() | (df["human_agent"] == "")).all()
    df["human_agent"] = ["retriever", "retriever", "calculator"]
    df.to_csv(sheet, index=False)
    scored = kappa.score_sheet(sheet)
    assert scored["n_labelled"] == 3
    assert scored["agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert len(scored["disagreements"]) == 1


async def test_release_refuses_a_reopened_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import release
    from convfinqa.tracking import registry

    reg = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg, raising=False)
    monkeypatch.setattr(registry, "_mirror_to_mlflow", lambda v: None)
    registry.register("v3_1")
    registry.promote("v3_1")
    doc = registry.load()
    doc.history.append({"event": "holdout_opened", "candidate": "v3_1"})
    registry.save(doc)

    with pytest.raises(SystemExit, match="already opened"):
        await release.run_release()


async def test_release_moves_the_released_alias_on_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import release, runner
    from convfinqa.tracking import registry

    reg = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg, raising=False)
    monkeypatch.setattr(registry, "_mirror_to_mlflow", lambda v: None)
    registry.register("v3_1")
    registry.promote("v3_1")

    csv = tmp_path / "h.csv"
    _frame([("a", 0, True)]).assign(split="holdout").to_csv(csv, index=False)

    async def fake_run_split(split, version, *, n_reports=None, concurrency=8):
        assert split == "holdout"
        return {"run_name": f"hold-{version}", "accuracy": 0.8, "csv": str(csv)}

    monkeypatch.setattr(runner, "run_split", fake_run_split)
    verdict = await release.run_release()
    assert verdict["passed"] and verdict["candidate"] == "v3_1"
    doc = registry.load()
    assert doc.aliases["released"] == "v3_1"
    assert doc.history[-1]["event"] == "holdout_opened"


# ── campaign protocol: attribution, manifest v2, the significance gate ──


def test_first_fault_walks_the_pipeline_in_order() -> None:
    """The first failing gold-derived check wins, whatever fails after it."""
    from convfinqa.evalloop import stage_scores

    every_check_fails = {
        "triage_turn_type_ok": False,
        "preprocess_skeleton_ok": False,
        "retriever_operand_recall": 0.0,
        "calculator_ok": False,
    }
    assert stage_scores.first_fault(every_check_fails) == "triage"
    assert (
        stage_scores.first_fault({**every_check_fails, "triage_turn_type_ok": True})
        == "preprocess"
    )
    assert (
        stage_scores.first_fault(
            {
                **every_check_fails,
                "triage_turn_type_ok": True,
                "preprocess_skeleton_ok": True,
            }
        )
        == "retriever"
    )
    # partial recall is a retriever fault, not a calculator one
    assert (
        stage_scores.first_fault(
            {
                "triage_turn_type_ok": True,
                "preprocess_skeleton_ok": True,
                "retriever_operand_recall": 0.5,
                "calculator_ok": False,
            }
        )
        == "retriever"
    )
    # a number turn has no skeleton and no calculator verdict — None, not a fault
    assert (
        stage_scores.first_fault(
            {
                "triage_turn_type_ok": True,
                "preprocess_skeleton_ok": None,
                "retriever_operand_recall": 1.0,
                "calculator_ok": None,
            }
        )
        is None
    )
    # ...and `attribute` gives that case to the calculator, which owns final form
    assert (
        stage_scores.attribute(
            {
                "triage_turn_type_ok": True,
                "preprocess_skeleton_ok": None,
                "retriever_operand_recall": 1.0,
                "calculator_ok": None,
            }
        )
        == "calculator"
    )


def test_one_sided_mcnemar_is_half_the_two_sided_p() -> None:
    from convfinqa.tracking.comparator import (
        mcnemar_exact_p,
        mcnemar_exact_p_one_sided,
    )

    # 12 fixed vs 8 broken — the v5 promotion's own numbers
    assert round(mcnemar_exact_p(8, 12), 4) == 0.5034
    assert round(mcnemar_exact_p_one_sided(8, 12), 4) == 0.2517
    # a clean sweep is significant one-sided and (just) not two-sided at 8 pairs
    assert mcnemar_exact_p_one_sided(0, 8) < 0.05


def test_clustering_weakens_evidence_concentrated_in_one_conversation() -> None:
    """Four fixes in one report are one piece of evidence, not four."""
    from convfinqa.tracking.comparator import durkalski_z, normal_sf

    spread = durkalski_z([(0, 1)] * 8)  # eight conversations, one fix each
    concentrated = durkalski_z([(0, 8)])  # one conversation, eight fixes
    assert spread > concentrated
    assert normal_sf(spread) < 0.05
    assert normal_sf(concentrated) > 0.05


def test_v5_promotion_is_refused_by_the_campaign_rule() -> None:
    """The committed evidence behind the last promotion, re-judged.

    v5 was promoted under the net-positive rule with McNemar p = 0.50. Under the
    campaign rule it is rejected, and this pins that — the rule change is the
    whole reason the lineage was rolled back."""
    from pathlib import Path

    from convfinqa.evalloop.gate import gate_runs

    root = Path("evaluation/predictions/evalloop")
    base = root / "evalloop-test50-v3_1·t3p3r3c3-20260902_220956.csv"
    cand = root / "evalloop-test50-v5·t3p4r3c3-20260902_221407.csv"
    if not base.exists() or not cand.exists():
        pytest.skip("committed evalloop CSVs not present")
    result, stats = gate_runs(
        base, cand, baseline_version="v3_1", candidate_version="v5"
    )
    assert result.promotable  # net positive, the old rule
    assert not stats["promotable"]  # ...and not significant, the new one
    assert stats["cluster_p_one_sided"] > 0.05
    assert stats["delta_ci_lo"] < 0 < stats["delta_ci_hi"]


def test_manifest_v2_is_a_superset_of_v1_and_disjoint(tmp_path: Path) -> None:
    """Extending a manifest must never invalidate evidence already recorded."""
    import json as _json

    from convfinqa.evalloop import splits

    v1 = splits.SPLITS_DIR / "eval_loop_v1.json"
    v2 = splits.SPLITS_DIR / "eval_loop_v2.json"
    if not v2.exists():
        pytest.skip("eval_loop_v2 not cut yet")
    a = _json.loads(v1.read_text())["splits"]
    b = _json.loads(v2.read_text())["splits"]
    assert set(a["train"]) <= set(b["train"])
    assert set(a["test"]) <= set(b["test"])
    assert not set(b["train"]) & set(b["test"])
    assert len(b["train"]) == len(b["test"]) == 100
    assert b["holdout"] == []  # deliberately unallocated during a campaign


def test_prompt_rewrite_must_keep_the_output_contract() -> None:
    from convfinqa.evalloop import teacher

    before = "Return turn_type and conv_type for each question."
    assert teacher.validate_prompt("triage", before, "x" * 400)  # drops both fields
    assert not teacher.validate_prompt(
        "triage", before, "Decide the turn_type and the conv_type. " + "x" * 400
    )
    # a collapsed prompt is refused whatever it says
    assert teacher.validate_prompt("triage", before, "turn_type conv_type")


# ── campaign caps, early stop, and the story ────────────────────────────


def test_campaign_rotates_off_an_agent_that_failed_twice() -> None:
    """Without rotation the loop gets stuck rewriting the same agent forever."""
    from convfinqa.evalloop import campaign

    past = [
        {"target_agent": "retriever", "promoted": False},
        {"target_agent": "retriever", "promoted": False},
        {"target_agent": "triage", "promoted": True},
    ]
    assert campaign.blocked_agents(past) == {"retriever"}
    counts = {"retriever": 9, "triage": 6, "preprocess": 4, "calculator": 2}
    agent, why = campaign.pick_target(counts, past)
    assert agent == "triage"  # not the most faults — the most faults is blocked
    assert "rotated past retriever" in why
    # ...and naming it explicitly does not get around the rule
    with pytest.raises(SystemExit):
        campaign.pick_target(counts, past, requested="retriever")
    # a promotion in the window clears the block
    assert (
        campaign.blocked_agents(
            past + [{"target_agent": "retriever", "promoted": True}]
        )
        == set()
    )


def test_campaign_refuses_a_sixth_experiment() -> None:
    from convfinqa.evalloop import campaign

    five = [{"target_agent": "triage", "promoted": False}] * 5
    with pytest.raises(SystemExit) as exc:
        campaign.check_capacity("c01", five)
    assert "cap is 5" in str(exc.value)
    campaign.check_capacity("c01", five[:4])  # four is fine


@pytest.mark.asyncio
async def test_early_stop_is_refused_on_the_gate_split() -> None:
    """A paired comparison needs a counterpart for every question."""
    from convfinqa.evalloop.runner import run_split

    with pytest.raises(ValueError, match="stop-at-first-wrong"):
        await run_split("test", "v2", stop_at_first_wrong=True)
    with pytest.raises(ValueError, match="train-seed"):
        await run_split("test", "v2", train_seed=7)


def test_train_draw_never_touches_the_gate_split() -> None:
    """The one property that makes every promotion afterwards meaningful."""
    from convfinqa.evalloop import splits

    if not (splits.SPLITS_DIR / "eval_loop_v2.json").exists():
        pytest.skip("eval_loop_v2 not cut yet")
    manifest = splits.load_manifest(splits.manifest_path("eval_loop_v2"))
    gate = set(manifest["splits"]["test"])
    for seed in (2026, 2027, 2028):
        ids, provenance = splits.draw_train(
            seed=seed,
            n_reports=100,
            path=splits.manifest_path("eval_loop_v2"),
        )
        assert len(ids) == 100
        assert not set(ids) & gate
        assert provenance["draw_seed"] == seed
    # different seeds really do draw different conversations
    a, _ = splits.draw_train(
        seed=1, n_reports=50, path=splits.manifest_path("eval_loop_v2")
    )
    b, _ = splits.draw_train(
        seed=2, n_reports=50, path=splits.manifest_path("eval_loop_v2")
    )
    assert set(a) != set(b)


def test_story_page_renders_from_a_minimal_record() -> None:
    """The published page must build from whatever the store actually holds."""
    from convfinqa.evalloop.story_page import render_page

    html = render_page(
        {
            "generated_at": "2026-09-04T00:00:00+00:00",
            "champion": "v6",
            "rule": "net positive AND one-sided clustered McNemar p < 0.05",
            "split": {
                "name": "eval_loop_v2",
                "gate_reports": 100,
                "gate_questions": 349,
            },
            "campaigns": [
                {
                    "name": "c01",
                    "experiments": [
                        {
                            "label": "c01-e01",
                            "target_agent": "retriever",
                            "baseline_version": "v2",
                            "candidate_version": "v6",
                            "promoted": True,
                            "accuracy_delta": 0.031,
                            "cluster_p_one_sided": 0.021,
                            "delta_ci": [0.004, 0.058],
                            "n_compared": 349,
                            "fixed": 18,
                            "broken": 7,
                            "panel_baseline": {"retriever": 0.7},
                            "panel_candidate": {"retriever": 0.78},
                            "summary_of_changes": "restructured around period selection",
                            "rationale": "The prompt buried the year rule.",
                            "diff": "@@ -1 +1 @@\n-old line\n+new line\n",
                            "prompt_chars": {"before": 1200, "after": 980},
                        }
                    ],
                }
            ],
            "lineage": [
                {
                    "at": "2026-09-04",
                    "version": "v6",
                    "previous_champion": "v2",
                    "reason": "gate passed",
                }
            ],
            "champion_track": [
                {"version": "v2", "accuracy": 0.62, "panel": {"retriever": 0.7}},
                {
                    "version": "v6",
                    "accuracy": 0.651,
                    "panel": {"retriever": 0.78},
                    "target_agent": "retriever",
                    "moved_by": "c01-e01",
                },
            ],
        }
    )
    assert "<!doctype html>" in html
    assert "c01-e01" in html
    assert "+3.10pp" in html
    assert "<svg" in html  # both figures rendered
    assert 'class="add"' in html  # the diff is coloured
    assert "&lt;script&gt;" not in html  # nothing unescaped leaked


def test_next_version_skips_the_numeric_prefix_of_an_existing_variant() -> None:
    """`v3_1` exists, so `v3` is taken — two bundles must not read as variants."""
    from convfinqa.evalloop.cycle import next_version

    picked = next_version("v2")
    assert picked.startswith("v")
    assert picked not in {"v3", "v4", "v5"}  # all have modules on disk


@pytest.mark.asyncio
async def test_run_structured_retries_a_transient_empty_reply() -> None:
    """The observed SDK failure returns no content and succeeds on the retry.

    One such call in fifty aborted a whole cycle and discarded twenty minutes of
    diagnosis, so the retry is the difference between a loop that runs
    unattended and one that does not."""
    from convfinqa.evalloop import sdk

    class Reply(BaseModel):
        ok: str

    calls = {"n": 0}

    async def flaky(prompt: str, **kwargs: object) -> tuple[Reply, dict[str, object]]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sdk.TeacherCallError("the SDK returned no content at all")
        return Reply(ok="yes"), {"usage": {}}

    original = sdk._run_structured_once
    sdk._run_structured_once = flaky  # type: ignore[assignment]
    try:
        out, _ = await sdk.run_structured(
            "hi", schema=Reply, system_prompt="s", max_turns=1
        )
    finally:
        sdk._run_structured_once = original  # type: ignore[assignment]
    assert out.ok == "yes"
    assert calls["n"] == 2

    # ...and a persistent failure still surfaces rather than being swallowed
    async def always(prompt: str, **kwargs: object) -> tuple[Reply, dict[str, object]]:
        raise sdk.TeacherCallError("still nothing")

    sdk._run_structured_once = always  # type: ignore[assignment]
    try:
        with pytest.raises(sdk.TeacherCallError, match="still nothing"):
            await sdk.run_structured("hi", schema=Reply, system_prompt="s", attempts=2)
    finally:
        sdk._run_structured_once = original  # type: ignore[assignment]


def test_rotation_note_only_claims_credit_when_it_changed_the_pick() -> None:
    """A blocked agent that ranked below the pick was never in contention.

    Saying "rotated past preprocess" when preprocess had fewer faults than the
    chosen agent credits the cap for a choice it did not make — and that note
    goes onto the experiment record, where it would be read as the reason."""
    from convfinqa.evalloop import campaign

    past = [
        {"target_agent": "preprocess", "promoted": False},
        {"target_agent": "preprocess", "promoted": False},
    ]
    # preprocess is blocked but ranks below retriever — no credit
    agent, why = campaign.pick_target(
        {"retriever": 16, "preprocess": 14, "triage": 7, "calculator": 7}, past
    )
    assert agent == "retriever"
    assert "rotated past" not in why

    # preprocess is blocked and would have won — the cap did change the pick
    agent, why = campaign.pick_target(
        {"retriever": 9, "preprocess": 20, "triage": 7, "calculator": 7}, past
    )
    assert agent == "retriever"
    assert "rotated past preprocess" in why


def test_ledger_text_names_the_questions_a_rejected_rewrite_broke(
    monkeypatch: Any,
) -> None:
    """Counts are not actionable; identities are.

    "Broke 18" tells the next writer to be careful. The eighteen questions tell
    it which behaviour to leave alone, which is the only version of that fact a
    prompt can be written against."""
    from convfinqa.evalloop import ledger

    monkeypatch.setattr(
        ledger,
        "attempts",
        lambda **_: [
            {
                "version": "v6",
                "outcome": "rejected",
                "accuracy_delta": 0.0229,
                "cluster_p_one_sided": 0.191,
                "fixed": 31,
                "broken": 23,
                "summary_of_changes": "added a greater-operator rule",
                "rationale": "",
                "broken_cases": [
                    {
                        "report_id": "AAP/2010/page_12.pdf",
                        "q_order": 2,
                        "question": "what was the change in operating income?",
                        "gold_answer": "4.2",
                        "baseline_answer": "4.2",
                        "candidate_answer": "greater(4.2, 0)",
                    }
                ],
            }
        ],
    )
    text = ledger.ledger_text("preprocess")
    assert "fixed 31 questions and broke 23" in text
    assert "BROKE AAP/2010/page_12.pdf q2" in text
    assert "before 4.2 -> after greater(4.2, 0)" in text


def test_diagnoses_for_agent_keys_on_the_prompt_hash_not_the_bundle_version(
    monkeypatch: Any,
) -> None:
    """A bundle is four prompts, so bundle version is the wrong key.

    v2 and v8 differ only in the retriever, so every preprocess failure filed
    under either bears on preprocess identically. Scoping by bundle version
    would hide half the record from the writer."""
    from convfinqa.evalloop import ledger

    hashes = {
        ("v2", "preprocess"): "pre-aaa",
        ("v8", "preprocess"): "pre-aaa",  # unchanged by the v8 retriever rewrite
        ("v9", "preprocess"): "pre-bbb",  # a different preprocess prompt
    }
    monkeypatch.setattr(ledger, "_agent_prompt_hash", lambda v, a: hashes.get((v, a)))

    class _Run:
        def __init__(self, version: str) -> None:
            self.data = SimpleNamespace(params={"prompts_version": version})
            self.info = SimpleNamespace(run_id=version)

    rows = {
        "v2": [{"derived_agent": "preprocess", "report_id": "A", "turn_index": 0}],
        "v8": [
            {"derived_agent": "preprocess", "report_id": "B", "turn_index": 1},
            {"derived_agent": "retriever", "report_id": "C", "turn_index": 0},
        ],
        "v9": [{"derived_agent": "preprocess", "report_id": "D", "turn_index": 0}],
    }

    class _Client:
        def download_artifacts(self, run_id: str, name: str) -> str:
            path = tmp / f"{run_id}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows[run_id]))
            return str(path)

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(ledger, "_client", lambda: _Client())
    monkeypatch.setattr(
        ledger, "_runs", lambda *a, **k: [_Run("v2"), _Run("v8"), _Run("v9")]
    )

    got = ledger.diagnoses_for_agent("preprocess", "v2")
    # v8 is included (same preprocess text), v9 excluded (different text),
    # and v8's retriever case is not preprocess's problem.
    assert {(d["report_id"], d["version"]) for d in got} == {("A", "v2"), ("B", "v8")}


@pytest.mark.asyncio
async def test_diagnosis_runs_concurrently_and_still_reports_in_case_order(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Independent calls should overlap; the record should not depend on that.

    Fifty sequential Opus calls at ~27s each was the largest single term in a
    cycle, and nothing about a diagnosis reads another's result. Concurrency is
    only safe if the artifact and the log stay in case order regardless of which
    call returns first — so the slowest case here is deliberately the first."""
    import asyncio as _asyncio

    from convfinqa.evalloop import teacher
    from convfinqa.tracking import mlflow_log

    class FakeRun:
        class info:  # noqa: N801 — mirrors mlflow's own attribute name
            run_id = "r1"

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *a: object) -> None:
            return None

    class FakeMlflow:
        def start_run(self, run_name: str = "") -> FakeRun:
            return FakeRun()

        def set_tags(self, tags: dict[str, str]) -> None:
            return None

        def log_params(self, params: dict[str, str]) -> None:
            return None

    monkeypatch.setattr(mlflow_log, "_mlflow", lambda: FakeMlflow())
    monkeypatch.setattr(teacher, "DIAGNOSTICS_DIR", tmp_path)
    monkeypatch.setattr(teacher, "prior_diagnoses", lambda *a, **k: [])

    cases = pd.DataFrame(
        [
            {
                "report_id": f"R{i}",
                "turn_index": i,
                "question": "q",
                "gold_answer": "1",
                "pred_answer": "2",
                "gold_program": "add(1, 1)",
                "pred_program": "",
            }
            for i in range(4)
        ]
    )
    monkeypatch.setattr(teacher, "first_wrong_cases", lambda _p: cases)
    monkeypatch.setattr(
        teacher, "case_payload", lambda row: {"derived_attribution": "preprocess"}
    )

    live = 0
    peak = 0

    async def fake_case(payload: Any, memory: str) -> tuple[Any, dict[str, Any]]:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        # R0 finishes last, so an order-of-completion assembly would invert.
        await _asyncio.sleep(0.05 if len(order_seen) == 0 else 0.01)
        order_seen.append(1)
        live -= 1
        return (
            SimpleNamespace(
                failed_agent="preprocess",
                failure_mode="m",
                attribution_reason="r",
                what_went_wrong="w",
                evidence="e",
                proposed_rule="p",
                confidence=0.9,
                gold_suspect=False,
                model_dump=lambda: {
                    "failed_agent": "preprocess",
                    "failure_mode": "m",
                    "confidence": 0.9,
                    "gold_suspect": False,
                },
            ),
            {},
        )

    order_seen: list[int] = []
    monkeypatch.setattr(teacher, "_diagnose_case", fake_case)

    summary = await teacher.diagnose_run("ignored.csv", "v2", concurrency=4)

    assert peak > 1, "cases ran one at a time — the gather is not doing anything"
    written = [
        json.loads(line)
        for line in (tmp_path / f"diagnoses_v2_{summary['run_name'][-15:]}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [d["report_id"] for d in written] == ["R0", "R1", "R2", "R3"]


def test_backfill_flips_refuses_when_the_recomputation_disagrees(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """A wrong flip record would be read as history by every future writer.

    The backfill re-derives flips from committed CSVs and pairs them to a run by
    version and split. If that pairing is wrong the arithmetic still succeeds and
    still produces plausible flips — so the recomputed counts are checked against
    the counts the verdict already recorded, and a mismatch skips the run."""
    from convfinqa.evalloop import ledger

    class _Run:
        data = SimpleNamespace(
            params={
                "baseline_version": "v2",
                "candidate_version": "v6",
                "evidence_split": "test",
            },
            tags={"mlflow.runName": "gate-v6-vs-v2"},
        )
        info = SimpleNamespace(run_id="g1")

    written: list[str] = []

    class _Client:
        def list_artifacts(self, run_id: str) -> list[Any]:
            return [SimpleNamespace(path="verdict.json")]

        def log_dict(self, run_id: str, payload: Any, name: str) -> None:
            written.append(name)

    monkeypatch.setattr(ledger, "_client", lambda: _Client())
    monkeypatch.setattr(ledger, "_runs", lambda *a, **k: [_Run()])
    # The verdict says 31 fixed / 23 broken; the recomputation will say 1/1.
    monkeypatch.setattr(
        ledger,
        "_artifact_json",
        lambda *a, **k: {"fail_to_pass": 31, "pass_to_fail": 23},
    )
    for v in ("v2", "v6"):
        (tmp_path / f"evalloop-test100-{v}·x-1.csv").write_text("x\n")
    monkeypatch.setattr(
        ledger,
        "_run_csv_for",
        lambda version, split, d: tmp_path / f"evalloop-test100-{version}·x-1.csv",
    )

    class _Result:
        improvements = [object()]
        regressions = [object()]

    monkeypatch.setattr(
        "convfinqa.tracking.comparator.compare_frames", lambda *a, **k: _Result()
    )
    monkeypatch.setattr("convfinqa.evalloop.gate.load_run_csv", lambda p: p)

    out = ledger.backfill_flips(predictions_dir=tmp_path)
    assert written == [], "wrote flips from a comparison that did not match the verdict"
    assert "disagrees" in out[0]["status"]


def test_pooled_targeting_outranks_a_single_noisy_draw() -> None:
    """One train draw is ~50 cases split four ways; the top two sit close.

    The real record: three v2 draws ranked preprocess 18, 26, 14 against
    retriever 15, 15, 16. Same prompts, and the winner depended on which reports
    were drawn — draw three picked retriever on a 16-vs-14 gap while the pooled
    evidence favoured preprocess 58-46."""
    from convfinqa.evalloop import campaign

    third_draw = {"preprocess": 14, "retriever": 16, "triage": 7, "calculator": 7}
    # Unpooled, the draw decides.
    agent, _ = campaign.pick_target(third_draw, [])
    assert agent == "retriever"

    pooled = {
        "preprocess": {"faults": 58, "cases": 150, "rate": 58 / 150, "n_runs": 3},
        "retriever": {"faults": 46, "cases": 150, "rate": 46 / 150, "n_runs": 3},
        "triage": {"faults": 24, "cases": 150, "rate": 24 / 150, "n_runs": 3},
        "calculator": {"faults": 22, "cases": 150, "rate": 22 / 150, "n_runs": 3},
    }
    agent, why = campaign.pick_target(third_draw, [], pooled=pooled)
    assert agent == "preprocess"
    assert "58/150" in why and "3 train draw(s)" in why
    assert "14 in this draw" in why


def test_pooled_targeting_ranks_on_rate_not_raw_totals() -> None:
    """Agents accumulate different numbers of draws, so totals are not comparable.

    An agent untouched since v2 has four draws behind it; one rewritten last
    cycle has one. Ranking on raw faults would pick whichever had gone longest
    without a rewrite — exactly backwards, since the freshly changed agent is the
    one we know least about."""
    from convfinqa.evalloop import campaign

    pooled = {
        # Four draws, 200 cases, 40 faults — a 20% rate on a long history.
        "triage": {"faults": 40, "cases": 200, "rate": 0.20, "n_runs": 4},
        # One draw, 50 cases, 20 faults — a 40% rate, and the higher priority.
        "retriever": {"faults": 20, "cases": 50, "rate": 0.40, "n_runs": 1},
        "preprocess": {"faults": 10, "cases": 200, "rate": 0.05, "n_runs": 4},
        "calculator": {"faults": 5, "cases": 200, "rate": 0.025, "n_runs": 4},
    }
    counts = {"triage": 8, "retriever": 20, "preprocess": 3, "calculator": 1}
    agent, why = campaign.pick_target(counts, [], pooled=pooled)
    assert agent == "retriever", "ranked on raw totals, favouring the stalest prompt"
    assert "40.0%" in why


def test_merge_draw_folds_the_current_pass_in_exactly_once() -> None:
    """A freshly rewritten agent has no pooled history — only this draw.

    Against v8 the retriever prompt r5 has never been diagnosed, so its pooled
    evidence is 0/0. Without the current draw merged in it would rank last on a
    rate of zero and could never be targeted, which is the opposite of what a
    just-changed agent deserves."""
    from convfinqa.evalloop import ledger

    pooled = {
        "triage": {"faults": 24, "cases": 150, "n_runs": 3, "versions": ["v2"]},
        "preprocess": {"faults": 58, "cases": 150, "n_runs": 3, "versions": ["v2"]},
        "retriever": {"faults": 0, "cases": 0, "n_runs": 0, "versions": []},
        "calculator": {"faults": 22, "cases": 150, "n_runs": 3, "versions": ["v2"]},
    }
    counts = {"triage": 5, "preprocess": 10, "retriever": 20, "calculator": 5}
    merged = ledger.merge_draw(pooled, counts, "v8")

    assert merged["retriever"]["faults"] == 20
    assert merged["retriever"]["cases"] == 40  # the draw's own denominator
    assert merged["retriever"]["rate"] == pytest.approx(0.5)
    assert merged["retriever"]["n_runs"] == 1
    # An agent with history gains exactly one draw, not two.
    assert merged["preprocess"]["n_runs"] == 4
    assert merged["preprocess"]["cases"] == 190
    assert merged["preprocess"]["faults"] == 68


async def test_agent_sdk_calls_open_a_traced_llm_span(monkeypatch: Any) -> None:
    """Teacher calls are invisible to MLflow unless this span is opened by hand.

    The pipeline agents are autologged because pydantic-ai runs in-process; the
    Agent SDK spawns the `claude` CLI as a subprocess, so no autologger can see
    a teacher call. The failure mode is silent — traces still appear, with one
    empty wrapper span and no prompt, reply, tokens or cost — so this pins the
    instrumentation rather than trusting a future refactor to notice."""
    import contextlib

    from convfinqa.evalloop import sdk
    from convfinqa.tracking import tracing

    class Ping(BaseModel):
        answer: str

    opened: list[dict[str, Any]] = []

    class _Handle:
        def __init__(self, rec: dict[str, Any]) -> None:
            self.rec = rec

        def set(self, **kw: Any) -> None:
            self.rec.setdefault("attrs", {}).update(kw)

        def inputs(self, v: Any) -> None:
            self.rec["inputs"] = v

        def outputs(self, v: Any) -> None:
            self.rec["outputs"] = v

    @contextlib.contextmanager
    def fake_span(name: str, **kw: Any) -> Any:
        rec = {"name": name, **kw}
        opened.append(rec)
        yield _Handle(rec)

    monkeypatch.setattr(tracing, "span", fake_span)

    async def fake_once(prompt: str, **kw: Any) -> tuple[Any, dict[str, Any]]:
        return Ping(answer="OK"), {
            "duration_ms": 1568,
            "num_turns": 2,
            "total_cost_usd": 0.14,
            "usage": {"input_tokens": 2, "output_tokens": 53},
            "tools_used": ["StructuredOutput"],
        }

    monkeypatch.setattr(sdk, "_run_structured_once", fake_once)

    out, _ = await sdk.run_structured("say ok", schema=Ping, system_prompt="reply json")
    assert out.answer == "OK"
    assert len(opened) == 1
    span = opened[0]
    assert span["span_type"] == "LLM", "span left UNKNOWN renders as an anonymous box"
    # The prompt and the reply are the point — attributes alone leave the trace
    # UI blank, which is what "no logs for the agent_sdk run" looked like.
    assert span["inputs"]["prompt"] == "say ok"
    assert span["outputs"] == {"answer": "OK"}
    assert span["attributes"]["model"]
    assert span["attrs"]["total_cost_usd"] == 0.14
    assert span["attrs"]["input_tokens"] == 2


async def test_a_retried_agent_sdk_call_records_both_attempts(
    monkeypatch: Any,
) -> None:
    """A transient empty reply should be visible, not swallowed by the retry.

    One call in fifty returns no content and the next identical call succeeds.
    If the retry shared a span the record would show only a slow success, hiding
    the flakiness that aborted a whole cycle before it was handled."""
    import contextlib

    from convfinqa.evalloop import sdk
    from convfinqa.tracking import tracing

    class Ping(BaseModel):
        answer: str

    opened: list[dict[str, Any]] = []

    class _Handle:
        def __init__(self, rec: dict[str, Any]) -> None:
            self.rec = rec

        def set(self, **kw: Any) -> None:
            self.rec.setdefault("attrs", {}).update(kw)

        def inputs(self, v: Any) -> None:
            pass

        def outputs(self, v: Any) -> None:
            pass

    @contextlib.contextmanager
    def fake_span(name: str, **kw: Any) -> Any:
        rec = {"name": name, **kw}
        opened.append(rec)
        yield _Handle(rec)

    monkeypatch.setattr(tracing, "span", fake_span)
    monkeypatch.setattr(sdk.asyncio, "sleep", lambda _s: asyncio_noop())

    async def asyncio_noop() -> None:
        return None

    calls = {"n": 0}

    async def flaky(prompt: str, **kw: Any) -> tuple[Any, dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sdk.TeacherCallError("the SDK returned no content at all")
        return Ping(answer="OK"), {}

    monkeypatch.setattr(sdk, "_run_structured_once", flaky)

    out, _ = await sdk.run_structured("x", schema=Ping, system_prompt="s")
    assert out.answer == "OK"
    assert len(opened) == 2, "the retry reused one span, hiding the failed attempt"
    assert opened[0]["attributes"]["attempt"] == 1
    assert "no content" in opened[0]["attrs"]["error"]
    assert opened[1]["attributes"]["attempt"] == 2
