"""The eval loop (M1): manifest determinism, run-identity traces, the gate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
            {"report_id": "a", "turn_index": 0, "first_wrong_turn": 1.0},
            {"report_id": "a", "turn_index": 1, "first_wrong_turn": 1.0},
            {"report_id": "a", "turn_index": 2, "first_wrong_turn": 1.0},
            {"report_id": "b", "turn_index": 0, "first_wrong_turn": None},
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


def test_gate_targeted_metric_tie_beats_fault_improvement(tmp_path: Path) -> None:
    """The deterministic metric outranks attribution: a tied metric refuses
    promotion even when the teacher's fault counts improved."""
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
    assert not verdict["target_improved"]
    assert not verdict["promotable_targeted"]
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
        "v99", base_version="v3_1", target="retriever", rules=["Rule one.", "Rule two."]
    )
    text = path.read_text()
    assert "do not hand-edit" in text
    assert "RETRIEVER_PROMPT = (" in text
    assert "TRIAGE_PROMPT,\n" in text  # imported unchanged
    assert "- Rule one." in text
    with pytest.raises(SystemExit):  # refuses to overwrite
        teacher._write_version_module(
            "v99", base_version="v3_1", target="retriever", rules=["x"]
        )


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
    assert verdict["promotable_targeted"]


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
