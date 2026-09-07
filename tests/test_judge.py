"""The confidence judge (s12): dataset, payload, metrics, distil, score, gate.

No test reaches a model or MLflow: `run_structured` is the scripted fake from
`test_sdk_teacher`, MLflow is the in-memory recorder, generated judge modules
go to a temp directory the prompts package also searches, and the judge's
committed record is redirected with `CONVFINQA_JUDGE_DIR`.

What is pinned: the judge never sees gold; the optimise split is balanced and
the three splits share no conversation; the metrics say what the plan says
they say (rule of three, coverage at the bound, failure capture); a distilled
prompt that mentions a reference answer is refused; a judge call that fails
fails *closed*; the gate rule; and the judge lineage stays apart from both
answering lineages.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from convfinqa.evalloop import judge
from tests.test_sdk_teacher import (  # noqa: F401 — fixtures
    DOC,
    REPORT_A,
    REPORT_B,
    REPORT_C,
    FakeSdkCalls,
    _row,
    fake_calls,
    fake_docs,
    fake_mlflow,
    registry_tmp,
)

REPORT_D = "FakeD/2020/page_1.pdf"
REPORT_E = "FakeE/2020/page_1.pdf"
REPORT_F = "FakeF/2020/page_1.pdf"
JUDGE_VERSION = "judge_j900"

GOOD_PROMPT = """## 1. Role
You are the confidence judge for a single-session financial Q&A agent.

## 2. What you are given
The question, the history, the report and the agent's trail: sub-questions,
program, retrieved values with sources, calculator trajectory, answer.

## 3. The six checks
Run operand_in_source, period_matches, reference_resolved, program_matches,
arithmetic_verified and unit_and_scale. Re-read the cited cell. Re-derive the
operation from the question.

## 4. Patterns that mean the answer is wrong
A cited cell that does not hold the value. A period the question did not name.
An answer that matches no tool return.

## 5. Patterns that mean the answer is right
Every operand is in its cited cell for the named period; the program is the
question's operation; the last tool return is the answer.

## 6. Deciding the band
Any failed check forces low. Two cannot_tell force low. Otherwise high, with
p_correct near 1. The high band must be at least 99% correct: when in doubt,
withhold.

## 7. Output contract
Return checks (one verdict per check), band (high or low), p_correct and a
one-line reason.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def judge_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Prompt modules and the judge's record both go to temp directories."""
    import convfinqa.prompts as prompts_pkg

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    monkeypatch.setattr(judge, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(prompts_pkg, "__path__", [*prompts_pkg.__path__, str(prompts)])
    monkeypatch.setenv(judge.JUDGE_DIR_ENV, str(tmp_path / "judge"))
    before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - before:
        if name.startswith("convfinqa.prompts.judge_j9"):
            del sys.modules[name]


@pytest.fixture
def all_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    from convfinqa.evalloop import stage_scores

    docs = {
        r: json.dumps(DOC)
        for r in (REPORT_A, REPORT_B, REPORT_C, REPORT_D, REPORT_E, REPORT_F)
    }
    monkeypatch.setattr(stage_scores, "report_documents", lambda: docs)


def _number(report: str, turn: int, answer: str, gold: str, *, correct: bool) -> dict[str, Any]:
    return _row(
        report, turn, f"what was revenue in 2020? ({report} q{turn})", gold, "",
        {"turn_type": "number", "conv_type": "Type I", "answer": answer,
         "retrieved": [{"question": "revenue in 2020", "answer": answer, "source": "table"}]},
        [], correct=correct,
    )  # fmt: skip


def _program(report: str, turn: int, answer: str, gold: str, *, correct: bool, program: str = "subtract(A, B)") -> dict[str, Any]:
    return _row(
        report, turn, f"what was the change in revenue? ({report} q{turn})", gold, "subtract(200, 50)",
        {"turn_type": "program", "conv_type": "Type I",
         "sub_questions": ["revenue in 2020", "revenue in 2019"], "program": program,
         "retrieved": [{"question": "revenue in 2020", "answer": "200", "source": "table"},
                       {"question": "revenue in 2019", "answer": "50", "source": "table"}],
         "answer": answer},
        [{"event": "tool_call", "tool": "subtract", "args": {"a": 200, "b": 50}},
         {"event": "tool_return", "tool": "subtract", "result": "150.0"}],
        correct=correct,
    )  # fmt: skip


def _write_run(path: Path, rows: list[dict[str, Any]], version: str = "sdk_v1") -> Path:
    from convfinqa.evalloop import stage_scores

    df = pd.DataFrame(rows)
    df["model_version_id"] = version
    stage_scores.score_rows(df)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def three_runs(tmp_path: Path, all_docs: None) -> dict[str, Path]:
    """optimise: 2 wrong (one gold-suspect) + 4 right; calibrate shares report A; test is F."""
    # Report D's wrong turn cites a gold operand (999) the document never
    # states → gold_suspect → excluded from the negatives.
    suspect = _row(
        REPORT_D, 0, "what was the change in revenue?", "949", "subtract(999, 50)",
        {"turn_type": "program", "conv_type": "Type I",
         "sub_questions": ["revenue in 2020", "revenue in 2019"], "program": "subtract(A, B)",
         "retrieved": [{"question": "revenue in 2020", "answer": "200", "source": "table"},
                       {"question": "revenue in 2019", "answer": "50", "source": "table"}],
         "answer": "150"},
        [{"event": "tool_call", "tool": "subtract", "args": {"a": 200, "b": 50}},
         {"event": "tool_return", "tool": "subtract", "result": "150.0"}],
        correct=False,
    )  # fmt: skip
    optimise = [
        _number(REPORT_A, 0, "50", "200", correct=False),
        _number(REPORT_A, 1, "200", "200", correct=True),
        _program(REPORT_B, 0, "150", "150", correct=True),
        _number(REPORT_C, 0, "200", "200", correct=True),
        _program(REPORT_C, 1, "150", "150", correct=True),
        suspect,
    ]
    calibrate = [
        _number(REPORT_A, 0, "200", "200", correct=True),
        _number(REPORT_E, 0, "50", "200", correct=False),
        _program(REPORT_E, 1, "150", "150", correct=True),
    ]
    test = [
        _number(REPORT_F, 0, "200", "200", correct=True),
        _program(REPORT_F, 1, "140", "150", correct=False),
    ]
    return {
        "optimise": _write_run(tmp_path / "opt.csv", optimise),
        "calibrate": _write_run(tmp_path / "cal.csv", calibrate),
        "test": _write_run(tmp_path / "test.csv", test),
    }


@pytest.fixture
def dataset(judge_dirs: Path, three_runs: dict[str, Path]) -> dict[str, Any]:
    return judge.build_dataset(
        optimise_csv=three_runs["optimise"],
        calibrate_csv=three_runs["calibrate"],
        test_csv=three_runs["test"],
    )


@pytest.fixture
def judge_module(judge_dirs: Path) -> str:
    judge._write_judge_module(JUDGE_VERSION, prompt=GOOD_PROMPT, header="test")
    return JUDGE_VERSION


# ---------------------------------------------------------------------------
# The judge never sees gold
# ---------------------------------------------------------------------------


def test_judge_payload_carries_no_gold(all_docs: None) -> None:
    row = _program(REPORT_B, 0, "140", "150", correct=False)
    payload = judge.judge_payload(row)
    assert not (judge.GOLD_KEYS & set(payload))
    assert not (judge.GOLD_KEYS & set(payload["trail"]))
    assert payload["trail"]["answer"] == "140"
    assert payload["trail"]["retrieved"][0]["source"] == "table"
    assert payload["report"] == DOC
    # And the text the judge is sent never contains the gold answer's field.
    assert "gold" not in judge.judge_prompt_text(payload)


def test_a_gold_key_reaching_the_payload_is_an_assertion(
    all_docs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import sdk_teacher

    monkeypatch.setattr(
        sdk_teacher, "sdk_flags", lambda row: {"gold_answer": "leak"}
    )
    payload = judge.judge_payload(_number(REPORT_A, 0, "200", "200", correct=True))
    # `sdk_flags` is nested, so a leak there is not a top-level key — the
    # guard is on the payload's own keys and the trail's. Prove the guard fires
    # when a top-level key appears.
    monkeypatch.setattr(judge, "GOLD_KEYS", frozenset({"question"}))
    with pytest.raises(AssertionError, match="gold reached the judge payload"):
        judge.judge_payload(_number(REPORT_A, 0, "200", "200", correct=True))
    assert payload["question"]


def test_a_live_capture_and_a_csv_row_build_the_same_payload(all_docs: None) -> None:
    from convfinqa.backends.agent_sdk import SdkTurnResult, result_to_capture

    result = {
        "turn_type": "program", "conv_type": "Type I",
        "sub_questions": ["revenue in 2020", "revenue in 2019"], "program": "subtract(A, B)",
        "retrieved": [{"question": "revenue in 2020", "answer": "200", "source": "table"},
                      {"question": "revenue in 2019", "answer": "50", "source": "table"}],
        "answer": "150",
    }  # fmt: skip
    trajectory = [
        {"event": "tool_call", "tool": "subtract", "args": {"a": 200, "b": 50}},
        {"event": "tool_return", "tool": "subtract", "result": "150.0"},
    ]
    capture = {"history_text": ""}
    capture.update(
        result_to_capture(
            SdkTurnResult.model_validate(result),
            question="what was the change in revenue?",
            history_text="",
            trajectory=trajectory,
            metrics={"num_turns": 2},
        )
    )
    live = judge.row_from_capture(
        capture,
        report_id=REPORT_B,
        turn_index=0,
        question="what was the change in revenue?",
        answer="150",
        program="subtract(A, B)",
    )
    csv_row = _row(
        REPORT_B, 0, "what was the change in revenue?", "150", "subtract(200, 50)",
        result, trajectory, correct=True,
    )  # fmt: skip
    assert judge.judge_payload(live) == judge.judge_payload(csv_row)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def test_dataset_balances_optimise_and_keeps_conversations_apart(
    dataset: dict[str, Any],
) -> None:
    balance = dataset["optimise_balance"]
    # Report D's miss cites a value the document never states → excluded.
    assert balance["n_excluded_non_agent"] == 1
    assert balance["excluded_question_ids"] == [f"{REPORT_D}_q0"]
    assert balance["n_negatives"] == 1
    assert balance["n_positives"] == 1
    assert dataset["stats"]["optimise"] == {"n": 2, "n_wrong": 1, "n_reports": 2}
    # Report A is in both draws → calibrate loses it, optimise keeps it.
    assert dataset["shared_reports_to_optimise"] == [REPORT_A]
    cal_ids = dataset["splits"]["calibrate"]
    assert all(not q.startswith(REPORT_A) for q in cal_ids)
    assert dataset["stats"]["calibrate"] == {"n": 2, "n_wrong": 1, "n_reports": 1}
    # The test split is the gate CSV, whole.
    assert dataset["stats"]["test"] == {"n": 2, "n_wrong": 1, "n_reports": 1}
    assert dataset["runtime_version"] == "sdk_v1"
    assert dataset["attribution_rule"]
    # Reproducible: the frame comes back in manifest order.
    frame = judge.split_frame(dataset, "optimise")
    assert list(frame["question_id"]) == dataset["splits"]["optimise"]
    assert judge.load_dataset() == dataset


def test_dataset_refuses_a_test_split_that_overlaps_a_draw(
    judge_dirs: Path, three_runs: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="test must be untouched"):
        judge.build_dataset(
            optimise_csv=three_runs["optimise"],
            calibrate_csv=three_runs["calibrate"],
            test_csv=three_runs["calibrate"],
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _scores(rows: list[tuple[str, bool, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"question_id": f"{r}_q{i}", "report_id": r, "correct": c, "band": b, "p_correct": p}
            for i, (r, c, b, p) in enumerate(rows)
        ]
    )


def test_selective_metrics_read_the_band_and_bound_the_unseen_error() -> None:
    df = _scores(
        [("A", True, "high", 0.99), ("A", True, "high", 0.95), ("B", False, "low", 0.2),
         ("B", True, "high", 0.9), ("C", False, "low", 0.4), ("C", True, "low", 0.6)]
    )  # fmt: skip
    m = judge.selective_metrics(df)
    assert m["n"] == 6 and m["n_wrong"] == 2
    assert m["coverage"] == pytest.approx(0.5)
    assert m["high_band_accuracy"] == 1.0 and m["high_band_error"] == 0.0
    assert m["failure_capture"] == 1.0 and m["n_false_alarms"] == 1
    assert m["meets_target"] is True
    # Zero errors in three answers bounds almost nothing; the bound shrinks
    # with n (the rule of three, ~3/n, holds once n is in the hundreds).
    assert 0.4 < m["high_band_error_upper95"] < 0.6
    assert m["auroc"] == 1.0  # every wrong answer scored below every right one
    # The curve: the best threshold inside a 1% error bound answers the four
    # highest-scored turns, all correct.
    assert m["coverage_at_target"] == pytest.approx(4 / 6)
    assert m["threshold_at_target"] == pytest.approx(0.6)


def test_a_high_band_miss_breaks_the_target() -> None:
    df = _scores([("A", True, "high", 0.9), ("A", False, "high", 0.8), ("B", False, "low", 0.1)])
    m = judge.selective_metrics(df)
    assert m["meets_target"] is False
    assert m["n_failures_missed"] == 1 and m["failure_capture"] == 0.5
    assert judge.wilson_upper(0, 300) == pytest.approx(0.009, abs=0.001)


def test_an_unscored_turn_is_neither_released_nor_withheld() -> None:
    """A rate-limited call is no verdict, so it leaves both bands alone.

    Scored closed to `low` instead, a spent subscription reads as a cautious
    judge: the j2 calibrate pass of 2026-09-07 withheld 175 of its 304 turns
    that way and reported a failure capture of 56.7% it had never earned.
    """
    df = _scores(
        [("A", True, "high", 0.99), ("A", False, "low", 0.2),
         ("B", False, "low", 0.1), ("B", True, "high", 0.9)]
    )  # fmt: skip
    df["unscored"] = False
    spent = df.copy()
    spent.loc[[2, 3], ["band", "p_correct", "unscored"]] = ["", None, True]
    m = judge.selective_metrics(spent)
    assert m["n"] == 2 and m["n_scored"] == 2 and m["n_unscored"] == 2
    assert m["complete"] is False
    # Only the two judged turns count, on both sides of every ratio.
    assert m["n_wrong"] == 1 and m["coverage"] == 0.5
    assert m["failure_capture"] == 1.0 and m["high_band_error"] == 0.0
    # And a complete frame is unaffected by the column existing.
    assert judge.selective_metrics(df)["n"] == 4
    assert judge.selective_metrics(df)["complete"] is True


def test_a_gate_refuses_a_scores_file_with_unjudged_turns(tmp_path: Path) -> None:
    df = _scores([("A", True, "high", 0.9), ("A", True, "high", 0.9), ("B", False, "low", 0.1)])
    df["unscored"] = False
    partial = df.copy()
    partial.loc[[2], ["band", "p_correct", "unscored"]] = ["", None, True]
    full, half = tmp_path / "full.csv", tmp_path / "half.csv"
    df.to_csv(full, index=False)
    partial.to_csv(half, index=False)
    with pytest.raises(judge.IncompleteJudgeScoresError, match="1 of 3 turns were never judged"):
        judge.gate_judges(full, half, baseline_version="judge_j1", candidate_version="judge_j2")
    with pytest.raises(judge.IncompleteJudgeScoresError):
        judge.gate_judges(half, full, baseline_version="judge_j1", candidate_version="judge_j2")
    # A CSV written before the column existed still says so in `error`, and it
    # is exactly the file that must not read as a complete pass.
    legacy = df.drop(columns=["unscored"])
    legacy["error"] = ["", "", "TeacherRateLimitError('rate_limited: session limit')"]
    old = tmp_path / "legacy.csv"
    legacy.to_csv(old, index=False)
    assert judge.selective_metrics(legacy)["n_unscored"] == 1
    with pytest.raises(judge.IncompleteJudgeScoresError):
        judge.gate_judges(full, old, baseline_version="judge_j1", candidate_version="judge_j2")


def test_gate_promotes_more_coverage_inside_the_bound_and_no_fewer_catches(
    tmp_path: Path,
) -> None:
    base = _scores([("A", True, "low", 0.5), ("A", True, "high", 0.9), ("B", False, "low", 0.1)])
    better = _scores([("A", True, "high", 0.9), ("A", True, "high", 0.9), ("B", False, "low", 0.1)])
    leaky = _scores([("A", True, "high", 0.9), ("A", True, "high", 0.9), ("B", False, "high", 0.7)])
    b, c, k = tmp_path / "b.csv", tmp_path / "c.csv", tmp_path / "k.csv"
    base.to_csv(b, index=False)
    better.to_csv(c, index=False)
    leaky.to_csv(k, index=False)
    verdict = judge.gate_judges(b, c, baseline_version="judge_j1", candidate_version="judge_j2")
    assert verdict["promotable"] is True
    assert verdict["flips"]["released"] == ["A_q0"] and verdict["n_withheld"] == 0
    refused = judge.gate_judges(b, k, baseline_version="judge_j1", candidate_version="judge_j2")
    assert refused["promotable"] is False
    assert "exceeds" in refused["reason"] and "failure capture fell" in refused["reason"]


# ---------------------------------------------------------------------------
# Distil
# ---------------------------------------------------------------------------


def test_validate_judge_prompt_refuses_gold_and_wrong_headings() -> None:
    assert judge.validate_judge_prompt(GOOD_PROMPT) == []
    with_gold = GOOD_PROMPT.replace("withhold.", "withhold. Compare to the gold answer.")
    assert any("no reference answer" in p for p in judge.validate_judge_prompt(with_gold))
    reordered = GOOD_PROMPT.replace("## 7. Output contract", "## 7. Output")
    assert any("headings" in p for p in judge.validate_judge_prompt(reordered))
    no_check = GOOD_PROMPT.replace("unit_and_scale", "units")
    assert any("unit_and_scale" in p for p in judge.validate_judge_prompt(no_check))
    # A draft at the wrong heading level is normalised, not refused.
    one_hash = GOOD_PROMPT.replace("## ", "# ")
    assert judge.validate_judge_prompt(one_hash) != []
    assert judge.validate_judge_prompt(judge.normalise_headings(one_hash)) == []
    assert judge.normalise_headings(one_hash) == GOOD_PROMPT


async def test_distil_writes_the_module_registers_the_lineage_and_promotes_the_first_judge(
    dataset: dict[str, Any],
    registry_tmp: Path,
    fake_mlflow: Any,
    fake_calls: FakeSdkCalls,
) -> None:
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry

    # "First" means first in this registry: the fixture copies the repo's own,
    # which carries the real j1/j2 lineage once a judge has been distilled.
    doc0 = registry.load()
    doc0.judge_prompts.clear()
    doc0.aliases.pop("judge_champion", None)
    registry.save(doc0)

    fake_calls.replies["JudgePromptDraft"] = {
        "prompt": GOOD_PROMPT,
        "sections": list(judge.JUDGE_HEADINGS),
        "notes": "first draft",
    }
    diagnoses = [
        {"outcome": "INCORRECT", "checks": "{}", "what_happened": "wrong cell",
         "evidence": "table", "detectable_without_gold": True, "judge_rule": "re-read the cell",
         "gold_suspect": False, "round": 1}
    ]  # fmt: skip
    out = await judge.distil_judge(new_version="judge_j901", diagnoses=diagnoses)
    assert out["promoted"] is True and out["seq"] == "j1"
    assert prompts_pkg.load_judge("judge_j901") == GOOD_PROMPT
    doc = registry.load()
    assert doc.aliases["judge_champion"] == "judge_j901"
    assert doc.judge_prompts and doc.judge_prompts[0]["first_seen_in"] == "judge_j901"
    assert doc.history[-1]["event"] == "promote_judge"
    assert doc.history[-1]["runtime_version"] == "sdk_v1"
    # The teacher's call went through the chokepoint with the teacher model.
    assert fake_calls.calls[-1]["model"]
    assert "judge_diagnoses" not in fake_calls.calls[-1]["prompt"]
    assert "re-read the cell" in fake_calls.calls[-1]["prompt"]

    # A revision: previous prompt goes in, a second seq comes out, champion stays.
    # (The lineage is keyed on the hash, so the revision has to differ.)
    fake_calls.replies["JudgePromptDraft"] = {
        "prompt": GOOD_PROMPT.replace("when in doubt", "whenever in doubt"),
        "sections": list(judge.JUDGE_HEADINGS),
        "notes": "revised",
    }
    out2 = await judge.distil_judge(
        new_version="judge_j902", base_version="judge_j901", diagnoses=diagnoses
    )
    assert out2["seq"] == "j2" and out2["promoted"] is False
    assert "You are the confidence judge for a single-session" in fake_calls.calls[-1]["prompt"]
    assert registry.judge_champion() == "judge_j901"

    # A distillation is never redone in place.
    with pytest.raises(SystemExit, match="never redone"):
        await judge.distil_judge(new_version="judge_j901", diagnoses=diagnoses)


async def test_a_draft_that_mentions_gold_is_refused_and_not_written(
    dataset: dict[str, Any],
    registry_tmp: Path,
    fake_mlflow: Any,
    fake_calls: FakeSdkCalls,
) -> None:
    fake_calls.replies["JudgePromptDraft"] = {
        "prompt": GOOD_PROMPT.replace("withhold.", "withhold. Compare to the gold answer."),
        "sections": [],
        "notes": "",
    }
    with pytest.raises(SystemExit, match="failed its contract"):
        await judge.distil_judge(new_version="judge_j903", diagnoses=[{"outcome": "CORRECT"}])
    assert not (judge.PROMPTS_DIR / "judge_j903.py").exists()
    assert "rejected_prompt.txt" in fake_mlflow.texts


# ---------------------------------------------------------------------------
# Score and diagnose
# ---------------------------------------------------------------------------


def _verdict(band: str, p: float) -> dict[str, Any]:
    return {
        "checks": dict.fromkeys(judge.CHECKS, "pass" if band == "high" else "fail"),
        "band": band,
        "p_correct": p,
        "reason": f"{band} because",
    }


async def test_score_split_fails_closed_and_records_the_pass(
    dataset: dict[str, Any],
    judge_module: str,
    registry_tmp: Path,
    fake_mlflow: Any,
    fake_calls: FakeSdkCalls,
) -> None:
    def reply(prompt: str) -> dict[str, Any]:
        if REPORT_E in prompt and "q0" in prompt:
            raise RuntimeError("the SDK returned no content at all")
        return _verdict("high", 0.95)

    fake_calls.replies["JudgeVerdict"] = reply
    out = await judge.score_split(version=judge_module, split="calibrate")
    scores = pd.read_csv(out["csv"])
    assert list(scores.columns) == list(judge.SCORE_COLUMNS)
    failed = scores[scores["question_id"] == f"{REPORT_E}_q0"].iloc[0]
    assert failed["band"] == "low" and "no content" in str(failed["error"])
    assert failed["p_correct"] == 0.0
    m = out["metrics"]
    assert m["n"] == 2 and m["n_judge_errors"] == 1
    # The failed call was the wrong answer, so failing closed caught it.
    assert m["failure_capture"] == 1.0 and m["coverage"] == 0.5
    assert fake_mlflow.metrics["coverage"] == 0.5
    assert fake_mlflow.params["judge_version"] == judge_module
    assert fake_mlflow.params["runtime_version"] == "sdk_v1"
    assert "metrics.json" in fake_mlflow.dicts and "risk_coverage.json" in fake_mlflow.dicts
    # Every judge call went out on the judge's model, not the teacher's.
    from convfinqa.llm import judge_model_name

    assert {c["model"] for c in fake_calls.calls} == {judge_model_name()}
    assert judge.latest_scores(judge_module, "calibrate") == Path(out["csv"])


async def test_round_two_diagnoses_only_the_misses_with_the_verdict_attached(
    dataset: dict[str, Any],
    judge_module: str,
    registry_tmp: Path,
    fake_mlflow: Any,
    fake_calls: FakeSdkCalls,
    tmp_path: Path,
) -> None:
    # A scores file where the wrong answer was released (a miss) and the right
    # one withheld (also a miss), plus nothing else.
    scores = pd.DataFrame(
        [
            {"question_id": f"{REPORT_E}_q0", "correct": False, "band": "high",
             "p_correct": 0.9, "reason": "looked fine", "checks": "{}"},
            {"question_id": f"{REPORT_E}_q1", "correct": True, "band": "high",
             "p_correct": 0.9, "reason": "fine", "checks": "{}"},
        ]
    )  # fmt: skip
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    fake_calls.replies["JudgeDiagnosis"] = {
        "checks": dict.fromkeys(judge.CHECKS, "fail"),
        "what_happened": "read the wrong cell",
        "evidence": "table",
        "detectable_without_gold": True,
        "judge_rule": "check the row label",
        "gold_suspect": False,
        "confidence": 0.9,
    }
    out = await judge.diagnose_split(
        split="calibrate", judge_scores=path, judge_version=judge_module
    )
    assert out["round"] == 2 and out["n_diagnosed"] == 1
    rows = judge.load_diagnoses(judge_module)
    assert len(rows) == 1
    row = rows[0]
    assert row["question_id"] == f"{REPORT_E}_q0" and row["outcome"] == "INCORRECT"
    assert row["judge_band"] == "high" and row["judge_reason"] == "looked fine"
    assert set(row) == set(judge.DIAGNOSIS_COLUMNS)
    # The teacher saw the outcome, the gold and the judge's verdict.
    prompt = fake_calls.calls[-1]["prompt"]
    assert '"outcome": "INCORRECT"' in prompt and "looked fine" in prompt
    assert re.search(r'"gold_answer": "?200"?', prompt)
    # Round 1 diagnoses every case and attaches no verdict.
    out1 = await judge.diagnose_split(split="optimise")
    assert out1["round"] == 1 and out1["n_diagnosed"] == 2
    assert all(r["judge_band"] == "" for r in judge.load_diagnoses("") if r["round"] == 1)


# ---------------------------------------------------------------------------
# Plumbing: models and lineages
# ---------------------------------------------------------------------------


async def test_run_structured_carries_the_model_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.evalloop import sdk
    from pydantic import BaseModel

    class Reply(BaseModel):
        ok: bool

    seen: dict[str, Any] = {}

    async def once(prompt: str, **kw: Any) -> tuple[Reply, dict[str, Any]]:
        seen.update(kw)
        return Reply(ok=True), {}

    monkeypatch.setattr(sdk, "_run_structured_once", once)
    await sdk.run_structured("p", schema=Reply, system_prompt="s", refs=None, model="claude-haiku-4-5-20251001")
    assert seen["model"] == "claude-haiku-4-5-20251001"
    await sdk.run_structured("p", schema=Reply, system_prompt="s", refs=None)
    from convfinqa.llm import teacher_model_name

    assert seen["model"] == teacher_model_name()


def test_teacher_and_judge_models_are_settings_with_pinned_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa import llm
    from convfinqa.config import settings
    from convfinqa.evalloop import teacher

    monkeypatch.setattr(settings, "teacher_model", None)
    monkeypatch.setattr(settings, "judge_model", None)
    assert llm.teacher_model_name() == llm.LM_TEACHER_MODEL == "claude-opus-5"
    assert llm.judge_model_name() == llm.LM_JUDGE_MODEL == "claude-haiku-4-5-20251001"
    monkeypatch.setattr(settings, "teacher_model", "claude-sonnet-5")
    assert teacher.teacher_model() == "claude-sonnet-5"
    monkeypatch.setattr(llm, "guard_llm_call", lambda: None)
    options = llm.teacher_options(system_prompt="s", model="claude-haiku-4-5-20251001")
    assert options.model == "claude-haiku-4-5-20251001"
    assert llm.teacher_options(system_prompt="s").model == "claude-sonnet-5"


def test_the_judge_lineage_stays_apart_from_the_answering_ones(
    judge_module: str, registry_tmp: Path
) -> None:
    from convfinqa.tracking import prompt_ledger, registry

    # The lineage is numbered from this registry, and the fixture copies the
    # repo's own — clear it so the sequence under test starts where it says.
    doc0 = registry.load()
    doc0.judge_prompts.clear()
    doc0.aliases.pop("judge_champion", None)
    registry.save(doc0)

    assert prompt_ledger.resolve_judge(judge_module)["seq"] == "j?"
    entry = prompt_ledger.ensure_judge(judge_module, source="test")
    assert entry["seq"] == "j1"
    assert prompt_ledger.ensure_judge(judge_module)["seq"] == "j1"
    registry.register(judge_module, source="test")
    with pytest.raises(ValueError, match="different runtimes"):
        registry.set_alias("sdk_champion", judge_module)
    with pytest.raises(ValueError, match="different runtimes"):
        registry.set_alias("judge_champion", "v2")
    with pytest.raises(ValueError, match="not a confidence-judge"):
        registry.promote_judge("v2")
    with pytest.raises(ValueError, match="never promoted on the gate split"):
        registry.promote_judge(judge_module, evidence_split="test")
    first = registry.promote_judge(judge_module)
    assert first.promoted and registry.judge_champion() == judge_module
    # The champion's fingerprint carries the judge composition, not a bundle's.
    from convfinqa.tracking.bundle import bundle_fingerprint

    spec = bundle_fingerprint(version=judge_module)
    assert spec["composition"] == "j1" and spec["v_judge"].startswith("j1@")
