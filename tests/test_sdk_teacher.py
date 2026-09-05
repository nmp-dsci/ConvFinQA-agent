"""The SDK arm's loop agents (s10 P4b): diagnose, rank classes, edit by section, distil.

No test reaches a model or MLflow: `run_structured` is replaced by a scripted
fake that dispatches on the schema it is asked for, the MLflow module by a
recorder that keeps artifacts in memory, and generated prompt modules are
written to a temp directory that is added to the prompts package path so
`prompts.load_sdk` round-trips them without touching `src/convfinqa/prompts/`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from convfinqa.evalloop import ledgers, sdk_teacher
from convfinqa.evalloop.sdk_teacher import HEADINGS

REPORT_A = "FakeA/2020/page_1.pdf"
REPORT_B = "FakeB/2020/page_1.pdf"
REPORT_C = "FakeC/2020/page_1.pdf"
DOC = {
    "pre_text": "revenue was 200 in 2020 and 50 in 2019.",
    "post_text": "",
    "table": {"revenue": {"2020": 200, "2019": 50}},
}

BASE_PROMPT = """## 1. Role
You answer a sequence of questions about one financial report in one session.
A number turn reads one value; a program turn computes from values.

## 2. Train of thought
number: triage -> retrieve. program: triage -> preprocess -> retrieve -> calculate.

## 3. Triage
Decide turn_type (number or program) and conv_type from the question and history.

## 4. Preprocess
Write sub_questions and a symbolic program (A, B bound in order; #0 refs;
const_100 for percentages) using add/subtract/multiply/divide/exp/greater.

## 5. Retrieve
Read each value from the table or text; keep units, signs and periods aligned.

## 6. Calculate
Always call mcp__cfq__add, mcp__cfq__subtract, mcp__cfq__multiply,
mcp__cfq__divide, mcp__cfq__exp or mcp__cfq__greater for every step.

## 7. Output contract
Return turn_type, conv_type, sub_questions, program, retrieved and answer.
"""

BASE_VERSION = "sdk_v900"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Generated modules go to a temp dir that the prompts package also searches."""
    import convfinqa.prompts as prompts_pkg

    target = tmp_path / "prompts"
    target.mkdir()
    monkeypatch.setattr(sdk_teacher, "PROMPTS_DIR", target)
    monkeypatch.setattr(prompts_pkg, "__path__", [*prompts_pkg.__path__, str(target)])
    before = set(sys.modules)
    yield target
    for name in set(sys.modules) - before:
        if name.startswith("convfinqa.prompts.sdk_v9"):
            del sys.modules[name]


@pytest.fixture
def base_module(prompts_tmp: Path) -> str:
    sdk_teacher._write_sdk_module(BASE_VERSION, prompt=BASE_PROMPT, header="test")
    return BASE_VERSION


@pytest.fixture
def registry_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from convfinqa.tracking import registry

    target = tmp_path / "registry.json"
    target.write_text(registry.REGISTRY_PATH.read_text())
    monkeypatch.setattr(registry, "REGISTRY_PATH", target)
    return target


class FakeMlflow:
    """Enough of the mlflow module for `mlflow_log.run` to yield a live recorder."""

    class _Run:
        class info:  # noqa: N801 — mirrors mlflow's attribute name
            run_id = "run-sdk-1"

        def __enter__(self) -> FakeMlflow._Run:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.dicts: dict[str, Any] = {}
        self.files: list[str] = []
        self.kinds: list[str] = []

    def start_run(self, run_name: str = "") -> FakeMlflow._Run:
        return self._Run()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)
        if "kind" in tags:
            self.kinds.append(tags["kind"])

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics[key] = value

    def log_artifact(self, path: str) -> None:
        self.files.append(Path(path).name)

    def log_text(self, text: str, name: str) -> None:
        self.texts[name] = text

    def log_dict(self, payload: Any, name: str) -> None:
        self.dicts[name] = payload

    def get_experiment_by_name(self, name: str) -> object:
        return object()

    def set_experiment(self, name: str) -> None:
        return None


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    from convfinqa.tracking import mlflow_log, tracing

    fake = FakeMlflow()
    monkeypatch.setattr(mlflow_log, "_mlflow", lambda: fake)
    monkeypatch.setattr(tracing, "enable", lambda: False)
    monkeypatch.setattr(ledgers, "eval_run_param", lambda r, k: None)
    return fake


@pytest.fixture
def fake_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    from convfinqa.evalloop import stage_scores

    docs = {r: json.dumps(DOC) for r in (REPORT_A, REPORT_B, REPORT_C)}
    monkeypatch.setattr(stage_scores, "report_documents", lambda: docs)


class FakeSdkCalls:
    """A scripted `run_structured`: dispatches on the schema, records every call."""

    def __init__(self) -> None:
        self.replies: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSdkCalls:
        from convfinqa.evalloop import sdk

        fake = self

        async def run_structured(prompt: str, **kw: Any) -> tuple[Any, dict[str, Any]]:
            schema = kw["schema"]
            fake.calls.append({"prompt": prompt, **kw})
            reply = fake.replies[schema.__name__]
            payload = reply(prompt) if callable(reply) else reply
            return (
                schema.model_validate(payload),
                {
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                    "total_cost_usd": 0.02,
                },
            )

        monkeypatch.setattr(sdk, "run_structured", run_structured)
        return self


@pytest.fixture
def fake_calls(monkeypatch: pytest.MonkeyPatch) -> FakeSdkCalls:
    return FakeSdkCalls().install(monkeypatch)


# ---------------------------------------------------------------------------
# A three-row SDK run: one number miss, one calculator skip, one empty program
# ---------------------------------------------------------------------------


def _row(
    report: str,
    turn: int,
    question: str,
    gold_answer: str,
    gold_program: str,
    result: dict[str, Any],
    trajectory: list[dict[str, Any]],
    *,
    correct: bool,
) -> dict[str, Any]:
    from convfinqa.backends.agent_sdk import SdkTurnResult, result_to_capture
    from convfinqa.evaluation.runner import _capture_to_row_fields

    cap = result_to_capture(
        SdkTurnResult.model_validate(result),
        question=question,
        history_text="",
        trajectory=trajectory,
        metrics={"num_turns": 2, "latency_ms": 10},
    )
    return {
        "report_id": report,
        "turn_index": turn,
        "question_id": f"{report}_q{turn}",
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": result["answer"],
        "correct": correct,
        "cascade": False,
        "first_wrong_turn": turn,
        "pred_program": result.get("program", ""),
        "gold_program": gold_program,
        "gold_turn_type": "program" if gold_program else "number",
        "gold_conv_type": "Type I",
        "run_id": "eval-sdk-1",
        "split": "train",
        **_capture_to_row_fields(cap),
    }


@pytest.fixture
def run_csv(tmp_path: Path) -> Path:
    from convfinqa.evalloop import stage_scores

    rows = [
        # A number turn that read the wrong cell → retriever.
        _row(
            REPORT_A, 0, "what was revenue in 2020?", "200", "",
            {"turn_type": "number", "conv_type": "Type I", "answer": "50",
             "retrieved": [{"question": "revenue in 2020", "answer": "50", "source": "table"}]},
            [], correct=False,
        ),
        # A program turn planned and retrieved correctly, no tool call → calculator.
        _row(
            REPORT_B, 0, "what was the change in revenue?", "150", "subtract(200, 50)",
            {"turn_type": "program", "conv_type": "Type I",
             "sub_questions": ["revenue in 2020", "revenue in 2019"],
             "program": "subtract(A, B)",
             "retrieved": [{"question": "revenue in 2020", "answer": "200", "source": "table"},
                           {"question": "revenue in 2019", "answer": "50", "source": "table"}],
             "answer": "140"},
            [], correct=False,
        ),
        # A program turn with an empty program → preprocess.
        _row(
            REPORT_C, 0, "what is the ratio of 2020 to 2019 revenue?", "4", "divide(200, 50)",
            {"turn_type": "program", "conv_type": "Type I",
             "sub_questions": ["revenue in 2020", "revenue in 2019"], "program": "",
             "retrieved": [{"question": "revenue in 2020", "answer": "200", "source": "table"},
                           {"question": "revenue in 2019", "answer": "50", "source": "table"}],
             "answer": "3"},
            [{"event": "tool_call", "tool": "divide", "args": {"a": 200, "b": 50}},
             {"event": "tool_return", "tool": "divide", "result": "4.0"}],
            correct=False,
        ),
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    stage_scores.score_rows(df)
    path = tmp_path / "sdk-run.csv"
    df.to_csv(path, index=False)
    return path


def _diagnosis(stage: str, label: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": label,
        "what_went_wrong": "w",
        "evidence": "e",
        "attribution_reason": "r",
        "fix_hint": "always re-read the column header",
        "gold_suspect": False,
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def test_payload_carries_flags_trail_and_trajectory(
    run_csv: Path, fake_docs: None
) -> None:
    from convfinqa.evalloop import teacher

    cases = {r.report_id: r for _, r in teacher.first_wrong_cases(run_csv).iterrows()}

    number = sdk_teacher.sdk_case_payload(cases[REPORT_A])
    assert number["derived_attribution"] == "retriever"
    assert number["trail"]["turn_type"] == "number"
    assert number["trail"]["sub_questions"] == [] and number["trail"]["program"] == ""
    assert number["trail"]["retrieved"] == [
        {"question": "what was revenue in 2020?", "answer": "50", "source": "table"}
    ]
    assert number["sdk_flags"] == {
        "stage_skips": [],
        "inline_arithmetic": False,
        "tool_calls": 0,
    }
    assert number["report"] == DOC
    assert number["gold_program"].startswith("(number selection")
    assert set(number["derived_checks"]) == {
        "triage_turn_type_ok",
        "preprocess_skeleton_ok",
        "preprocess_plan_ok",
        "retriever_operand_recall",
        "calculator_ok",
    }

    program = sdk_teacher.sdk_case_payload(cases[REPORT_B])
    assert program["derived_attribution"] == "calculator"
    assert program["sdk_flags"]["stage_skips"] == ["calculator"]
    assert program["calculator_trajectory"] == []
    assert program["missing_gold_operands"] == []
    assert [r["source"] for r in program["trail"]["retrieved"]] == ["table", "table"]

    empty = sdk_teacher.sdk_case_payload(cases[REPORT_C])
    assert empty["derived_attribution"] == "preprocess"
    assert empty["sdk_flags"]["stage_skips"] == ["preprocess"]
    assert empty["sdk_flags"]["tool_calls"] == 1
    # The answer (3) matches no tool return (4.0): arithmetic happened in-model.
    assert empty["sdk_flags"]["inline_arithmetic"] is True
    assert empty["calculator_trajectory"][0]["event"] == "tool_call"


# ---------------------------------------------------------------------------
# diagnose_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnose_run_appends_sdk_rows_and_scores_kappa(
    run_csv: Path,
    fake_docs: None,
    base_module: str,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from convfinqa.tracking.prompt_ledger import prompt_hash

    monkeypatch.setattr(sdk_teacher, "DIAGNOSTICS_DIR", tmp_path / "per-run")
    by_report = {
        REPORT_A: _diagnosis("retriever", "retriever/wrong-period"),
        REPORT_B: _diagnosis("calculator", "calculator/wrong-computation"),
        REPORT_C: _diagnosis(
            "retriever", "retriever/wrong-value"
        ),  # disputes preprocess
    }

    def reply(prompt: str) -> dict[str, Any]:
        payload = json.loads(prompt)
        return by_report[payload["report_id"]]

    fake_calls.replies["SdkDiagnosis"] = reply

    summary = await sdk_teacher.diagnose_run(
        run_csv, base_module, concurrency=2, campaign="s01", label="s01-e01"
    )

    table = ledgers.load("diagnoses")
    assert len(table) == 3 and summary["ledger_rows"] == 3
    assert set(table["runtime"]) == {"agent_sdk"}
    assert set(table["version"]) == {base_module}
    assert set(table["prompt_hash"]) == {prompt_hash(BASE_PROMPT)}
    assert list(table["report_id"]) == [REPORT_A, REPORT_B, REPORT_C]
    assert list(table["derived_agent"]) == ["retriever", "calculator", "preprocess"]
    assert list(table["stage"]) == ["retriever", "calculator", "retriever"]
    assert list(table["attribution_disputed"]) == [False, False, True]
    assert set(table["diagnosis_run_id"]) == {"run-sdk-1"}
    assert table["cost_usd"].sum() == pytest.approx(0.06)
    assert ledgers.load("diagnoses", runtime="multi_agent").empty

    # κ over the three agent-naming pairs: po = 2/3, pe = 1/3 → 0.5.
    assert summary["kappa_vs_attribution"] == pytest.approx(0.5)
    assert fake_mlflow.metrics["kappa_vs_attribution"] == pytest.approx(0.5)
    assert fake_mlflow.kinds == ["sdk_diagnose"]
    assert fake_mlflow.tags["runtime"] == "agent_sdk"
    assert fake_mlflow.tags["campaign"] == "s01"
    assert summary["counts"] == {
        "triage": 0,
        "preprocess": 1,
        "retriever": 1,
        "calculator": 1,
    }
    assert summary["labels"] == {
        "retriever/wrong-period": 1,
        "calculator/wrong-computation": 1,
        "retriever/wrong-value": 1,
    }
    assert summary["n_cases"] == 3 and summary["n_diagnosed"] == 3
    assert summary["run_id"] == "run-sdk-1" and summary["usage"][
        "cost_usd"
    ] == pytest.approx(0.06)

    # The per-run file keeps the pipeline teacher's names for its readers.
    per_run = [
        json.loads(line)
        for line in Path(summary["diagnoses_path"]).read_text().splitlines()
    ]
    assert [d["diagnosis_id"] for d in per_run] == list(table["diagnosis_id"])
    assert (
        per_run[0]["failed_agent"] == "retriever"
        and per_run[0]["runtime"] == "agent_sdk"
    )
    assert per_run[0]["proposed_rule"] == "always re-read the column header"

    # Every call went through the chokepoint with references, not text.
    assert len(fake_calls.calls) == 3
    refs = fake_calls.calls[0]["refs"]
    assert refs["system_prompt"]["name"] == "SDK_DIAGNOSE_PROMPT"
    assert refs["user_prompt"]["kind"] == "diagnose_case"
    assert refs["user_prompt"]["runtime"] == "agent_sdk"
    assert fake_calls.calls[0]["system_prompt"] is sdk_teacher.SDK_DIAGNOSE_PROMPT


@pytest.mark.asyncio
async def test_diagnose_memory_reads_the_sdk_ledger_only(
    run_csv: Path,
    fake_docs: None,
    base_module: str,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sdk_teacher, "DIAGNOSTICS_DIR", tmp_path / "per-run")
    ledgers.append(
        "diagnoses",
        [
            {"diagnosis_id": "d-sdk", "runtime": "agent_sdk", "version": "sdk_v1",
             "stage": "retriever", "label": "retriever/wrong-value", "fix_hint": "SDK HINT",
             "diagnosed_at": "2026-09-01T00:00:00+00:00"},
            {"diagnosis_id": "d-pipe", "runtime": "multi_agent", "version": "v8",
             "stage": "retriever", "label": "retriever/wrong-value", "fix_hint": "PIPE HINT",
             "diagnosed_at": "2026-09-02T00:00:00+00:00"},
        ],
    )  # fmt: skip
    fake_calls.replies["SdkDiagnosis"] = _diagnosis(
        "retriever", "retriever/wrong-value"
    )

    await sdk_teacher.diagnose_run(run_csv, base_module)

    prompt = fake_calls.calls[0]["prompt"]
    assert "SDK HINT" in prompt and "PIPE HINT" not in prompt
    assert "SDK HINT" in fake_mlflow.texts["diagnose_memory.txt"]
    assert (
        fake_calls.calls[0]["refs"]["user_prompt"]["memory_artifact"]
        == "diagnose_memory.txt"
    )


# ---------------------------------------------------------------------------
# rank_classes
# ---------------------------------------------------------------------------


def _ledger_diag(
    version: str, prompt_hash: str, label: str, i: int, **over: Any
) -> dict[str, Any]:
    return {
        "diagnosis_id": f"d-{version}-{i}",
        "runtime": "agent_sdk",
        "version": version,
        "prompt_hash": prompt_hash,
        "stage": label.split("/")[0],
        "derived_agent": label.split("/")[0],
        "label": label,
        **over,
    }


def test_rank_classes_pools_by_prompt_hash_and_ranks_by_wilson(
    prompts_tmp: Path, base_module: str
) -> None:
    from convfinqa.evalloop import ledger
    from convfinqa.tracking.prompt_ledger import prompt_hash

    same = prompt_hash(BASE_PROMPT)
    sdk_teacher._write_sdk_module("sdk_v901", prompt=BASE_PROMPT, header="same text")
    other_text = BASE_PROMPT.replace("Read each value", "Read every value")
    sdk_teacher._write_sdk_module("sdk_v902", prompt=other_text, header="other text")
    other = prompt_hash(other_text)

    rows = [
        *(_ledger_diag("sdk_v900", same, "retriever/wrong-period", i) for i in range(3)),
        *(_ledger_diag("sdk_v901", same, "retriever/wrong-period", i) for i in range(2)),
        *(_ledger_diag("sdk_v901", same, "calculator/wrong-scale", 10 + i) for i in range(2)),
        # Excluded from the population: a verdict naming no agent, and gold doubt.
        _ledger_diag("sdk_v900", same, "gold", 20, derived_agent="gold_suspect", stage="retriever"),
        _ledger_diag("sdk_v900", same, "retriever/wrong-value", 21, gold_suspect=True),
        # A different prompt text never pools.
        *(_ledger_diag("sdk_v902", other, "triage/wrong-turn-type", i) for i in range(9)),
        # Nor does a pipeline draw, whatever its hash.
        {**_ledger_diag("v8", same, "preprocess/wrong-operation", 30), "runtime": "multi_agent"},
    ]  # fmt: skip
    ledgers.append("diagnoses", rows)

    ranked = sdk_teacher.rank_classes("sdk_v900")
    assert list(ranked) == ["retriever/wrong-period", "calculator/wrong-scale"]
    top = ranked["retriever/wrong-period"]
    assert top["faults"] == 5 and top["n"] == 7 and top["rank"] == 1
    assert top["stages"] == {"retriever": 5}
    assert len(top["diagnosis_ids"]) == 5
    assert ranked["calculator/wrong-scale"]["rank"] == 2

    # The exact formula targeting uses, not a re-implementation of it.
    entry: dict[str, Any] = {"faults": 5, "cases": 7}
    ledger._score(entry)
    assert top["wilson_lower"] == pytest.approx(entry["score"], abs=1e-6)
    assert top["wilson_lower"] > ranked["calculator/wrong-scale"]["wilson_lower"] > 0

    # sdk_v901 shares the text, so it sees the same pool; sdk_v902 sees its own.
    assert sdk_teacher.rank_classes("sdk_v901") == ranked
    assert list(sdk_teacher.rank_classes("sdk_v902")) == ["triage/wrong-turn-type"]
    assert sdk_teacher.rank_classes("sdk_v902")["triage/wrong-turn-type"]["n"] == 9


# ---------------------------------------------------------------------------
# propose_version
# ---------------------------------------------------------------------------


def _edit(target: str, failure_class: str, body: str, ids: list[str]) -> dict[str, Any]:
    return {
        "target": target,
        "failure_class": failure_class,
        "change_kind": "rule",
        "diagnosis_ids": ids,
        "edit_text": f"edit {target}",
        "rationale": f"because {failure_class}",
        "new_section_body": body,
    }


def _diagnoses_file(tmp_path: Path) -> Path:
    rows = [
        {"diagnosis_id": "d-1", "stage": "retriever", "derived_agent": "retriever",
         "label": "retriever/wrong-period", "what_went_wrong": "w", "evidence": "e",
         "fix_hint": "h", "question": "q1"},
        {"diagnosis_id": "d-2", "stage": "retriever", "derived_agent": "retriever",
         "label": "retriever/wrong-period", "what_went_wrong": "w", "evidence": "e",
         "fix_hint": "h", "question": "q2"},
        {"diagnosis_id": "d-3", "stage": "calculator", "derived_agent": "calculator",
         "label": "calculator/wrong-scale", "what_went_wrong": "w", "evidence": "e",
         "fix_hint": "h", "question": "q3"},
    ]  # fmt: skip
    path = tmp_path / "diagnoses.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


POOLED = {
    "retriever/wrong-period": {"faults": 2, "n": 3, "wilson_lower": 0.2, "rank": 1},
    "calculator/wrong-scale": {"faults": 1, "n": 3, "wilson_lower": 0.05, "rank": 2},
}

RETRIEVE_BODY = (
    "Read each value from the table or text; keep units, signs and periods "
    "aligned.\nTranscribe the whole row across every period header before "
    "choosing a cell."
)
CALCULATE_BODY = (
    "Always call mcp__cfq__add, mcp__cfq__subtract, mcp__cfq__multiply,\n"
    "mcp__cfq__divide, mcp__cfq__exp or mcp__cfq__greater for every step.\n"
    "Multiply a ratio by const_100 only when the question asks for a percentage."
)


@pytest.mark.asyncio
async def test_propose_version_edits_only_the_named_sections(
    base_module: str,
    registry_tmp: Path,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    tmp_path: Path,
) -> None:
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry
    from convfinqa.tracking.prompt_ledger import prompt_hash

    fake_calls.replies["SdkRewrite"] = {
        "edits": [
            _edit(
                "## 5. Retrieve",
                "retriever/wrong-period",
                RETRIEVE_BODY,
                ["d-1", "d-2"],
            ),
            _edit("calculate", "calculator/wrong-scale", CALCULATE_BODY, ["d-3"]),
        ],
        "summary": "two sections",
    }

    out = await sdk_teacher.propose_version(
        _diagnoses_file(tmp_path),
        base_version=base_module,
        new_version="sdk_v903",
        pooled=POOLED,
        campaign="s01",
        label="s01-e02",
    )

    # The module round-trips, and only the two named sections changed.
    after = prompts_pkg.load_sdk("sdk_v903")
    before_s, after_s = (
        sdk_teacher.split_sections(BASE_PROMPT),
        sdk_teacher.split_sections(after),
    )
    assert list(after_s) == list(HEADINGS)
    for heading in HEADINGS:
        if heading in {"## 5. Retrieve", "## 6. Calculate"}:
            assert after_s[heading] != before_s[heading]
        else:
            assert after_s[heading] == before_s[heading], heading
    assert after_s["## 5. Retrieve"].strip() == RETRIEVE_BODY
    assert "Transcribe the whole row" in out["diff"]
    assert "const_100 only when" in out["diff"]
    # Two added lines in the whole-prompt diff (the calculate edit keeps two lines
    # verbatim): nothing outside the two sections moved.
    added = [
        line
        for line in out["diff"].splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert len(added) == 2, added
    assert (
        Path(out["module_path"])
        .read_text()
        .startswith('"""Generated by convfinqa.evalloop.sdk_teacher')
    )
    assert base_module in Path(out["module_path"]).read_text()
    assert out["validate_ok"] is True
    assert out["new_version"] == "sdk_v903" and out["base_version"] == base_module
    assert out["prompt_chars_before"] == len(BASE_PROMPT)
    assert out["prompt_chars_after"] == len(after)
    assert out["run_id"] == "run-sdk-1"

    # Two edits, one rewrite: two ledger rows sharing a rewrite_id.
    rewrites = ledgers.load("rewrites")
    assert len(rewrites) == 2 and rewrites["rewrite_id"].nunique() == 1
    assert set(rewrites["rewrite_id"]) == {out["rewrite_id"]}
    assert set(rewrites["runtime"]) == {"agent_sdk"}
    assert list(rewrites["target"]) == ["## 5. Retrieve", "## 6. Calculate"]
    assert list(rewrites["failure_class"]) == [
        "retriever/wrong-period",
        "calculator/wrong-scale",
    ]
    assert list(rewrites["rank"]) == [1, 2]
    assert list(rewrites["wilson_lower"]) == [0.2, 0.05]
    assert set(rewrites["change_kind"]) == {"rule"}
    assert set(rewrites["prompt_hash_before"]) == {prompt_hash(BASE_PROMPT)}
    assert set(rewrites["prompt_hash_after"]) == {prompt_hash(after)}
    assert set(rewrites["campaign"]) == {"s01"} and set(rewrites["experiment_n"]) == {2}
    assert json.loads(rewrites["diagnosis_ids"][0]) == ["d-1", "d-2"]
    # Each row's diff is the hunk for its own section.
    assert "Transcribe the whole row" in rewrites["diff"][0]
    assert "Transcribe the whole row" not in rewrites["diff"][1]
    assert [e["edit_id"] for e in out["edits"]] == list(rewrites["edit_id"])
    assert out["edits"][0]["diff"] == rewrites["diff"][0]
    assert {k for e in out["edits"] for k in e} == {
        "edit_id", "target", "failure_class", "change_kind",
        "diagnosis_ids", "edit_text", "diff", "rationale",
    }  # fmt: skip

    # The run carries the change list, the diff, the writer prompt; the lineage grew.
    assert fake_mlflow.kinds == ["sdk_propose"]
    assert fake_mlflow.tags["runtime"] == "agent_sdk"
    assert fake_mlflow.params["base_version"] == base_module
    assert fake_mlflow.params["new_version"] == "sdk_v903"
    assert [e["target"] for e in fake_mlflow.dicts["changes.json"]["edits"]] == [
        "## 5. Retrieve",
        "## 6. Calculate",
    ]
    assert fake_mlflow.dicts["prompt_diff.json"]["diff"] == out["diff"]
    assert "writer_prompt.txt" in fake_mlflow.texts
    assert fake_mlflow.metrics["n_edits"] == 2.0
    doc = registry.load()
    assert any(
        e["hash"] == prompt_hash(after) and e["source"] == "sdk_teacher"
        for e in doc.sdk_prompts or []
    )

    # What the writer was shown, and how the call was referenced.
    call = fake_calls.calls[0]
    shown = json.loads(call["prompt"].split("\n\n## Prior edits", 1)[0])
    assert list(shown["current_prompt_sections"]) == list(HEADINGS)
    assert shown["class_ranking"]["retriever/wrong-period"]["rank"] == 1
    assert set(shown["diagnoses_by_label"]) == {
        "retriever/wrong-period",
        "calculator/wrong-scale",
    }
    assert "This is the first rewrite" in call["prompt"]
    assert call["refs"]["system_prompt"]["name"] == "SDK_WRITER_PROMPT"
    assert call["refs"]["target_prompt"]["kind"] == "sdk_prompt"
    assert call["refs"]["user_prompt"]["name"] == "writer_prompt.txt"


@pytest.mark.asyncio
async def test_max_areas_one_keeps_only_the_top_ranked_edit(
    base_module: str,
    registry_tmp: Path,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    tmp_path: Path,
) -> None:
    import convfinqa.prompts as prompts_pkg

    # The writer lists the lower-ranked class first; the cap keeps rank 1.
    fake_calls.replies["SdkRewrite"] = {
        "edits": [
            _edit("## 6. Calculate", "calculator/wrong-scale", CALCULATE_BODY, ["d-3"]),
            _edit("## 5. Retrieve", "retriever/wrong-period", RETRIEVE_BODY, ["d-1"]),
        ],
        "summary": "two offered",
    }
    out = await sdk_teacher.propose_version(
        _diagnoses_file(tmp_path),
        base_version=base_module,
        new_version="sdk_v904",
        pooled=POOLED,
        max_areas=1,
    )
    assert [e["target"] for e in out["edits"]] == ["## 5. Retrieve"]
    after_s = sdk_teacher.split_sections(prompts_pkg.load_sdk("sdk_v904"))
    assert (
        after_s["## 6. Calculate"]
        == sdk_teacher.split_sections(BASE_PROMPT)["## 6. Calculate"]
    )
    assert len(ledgers.load("rewrites")) == 1
    assert fake_mlflow.params["max_areas"] == "1"


@pytest.mark.asyncio
async def test_an_edit_naming_an_unknown_heading_is_rejected(
    base_module: str,
    registry_tmp: Path,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    prompts_tmp: Path,
    tmp_path: Path,
) -> None:
    fake_calls.replies["SdkRewrite"] = {
        "edits": [
            _edit("## 8. Extras", "retriever/wrong-period", RETRIEVE_BODY, ["d-1"])
        ],
        "summary": "bad",
    }
    with pytest.raises(SystemExit, match="unknown target"):
        await sdk_teacher.propose_version(
            _diagnoses_file(tmp_path), base_version=base_module, new_version="sdk_v905"
        )
    assert not (prompts_tmp / "sdk_v905.py").exists()
    assert ledgers.load("rewrites").empty


@pytest.mark.asyncio
async def test_a_rewrite_breaking_the_contract_is_recorded_not_written(
    base_module: str,
    registry_tmp: Path,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
    prompts_tmp: Path,
    tmp_path: Path,
) -> None:
    fake_calls.replies["SdkRewrite"] = {
        "edits": [
            _edit("## 6. Calculate", "calculator/wrong-scale", "Do the maths.", ["d-3"])
        ],
        "summary": "drops the tools",
    }
    with pytest.raises(SystemExit, match="mcp__cfq__add"):
        await sdk_teacher.propose_version(
            _diagnoses_file(tmp_path), base_version=base_module, new_version="sdk_v906"
        )
    assert not (prompts_tmp / "sdk_v906.py").exists()
    rewrites = ledgers.load("rewrites")
    assert len(rewrites) == 1 and not bool(rewrites["validate_ok"][0])


# ---------------------------------------------------------------------------
# Section mechanics and validation
# ---------------------------------------------------------------------------


def test_split_and_join_round_trip_and_normalise_targets() -> None:
    sections = sdk_teacher.split_sections(BASE_PROMPT)
    assert list(sections) == list(HEADINGS)
    assert sdk_teacher.join_sections(sections) == BASE_PROMPT
    with_preamble = "intro\n\n" + BASE_PROMPT
    assert (
        sdk_teacher.join_sections(sdk_teacher.split_sections(with_preamble))
        == with_preamble
    )
    assert sdk_teacher.normalise_target("retrieve") == "## 5. Retrieve"
    assert sdk_teacher.normalise_target("7. Output contract") == "## 7. Output contract"
    assert sdk_teacher.normalise_target("3") == "## 3. Triage"
    with pytest.raises(ValueError):
        sdk_teacher.normalise_target("## 9. Nope")
    with pytest.raises(ValueError, match="more than once"):
        sdk_teacher.split_sections(BASE_PROMPT + "\n## 3. Triage\nagain\n")


def test_validate_sdk_prompt_rejects_missing_heading_plumbing_and_tools() -> None:
    assert sdk_teacher.validate_sdk_prompt(BASE_PROMPT, BASE_PROMPT) == []

    missing = BASE_PROMPT.replace("## 4. Preprocess", "## Preprocess")
    problems = sdk_teacher.validate_sdk_prompt(BASE_PROMPT, missing)
    assert any("missing section heading '## 4. Preprocess'" in p for p in problems)

    plumbing = BASE_PROMPT + "\n[[ ## answer ## ]]\n"
    assert any(
        "plumbing" in p for p in sdk_teacher.validate_sdk_prompt(BASE_PROMPT, plumbing)
    )

    no_tool = BASE_PROMPT.replace("mcp__cfq__exp", "exp")
    assert any(
        "mcp__cfq__exp" in p
        for p in sdk_teacher.validate_sdk_prompt(BASE_PROMPT, no_tool)
    )

    no_key = BASE_PROMPT.replace("sub_questions", "subquestions")
    assert any(
        "'sub_questions'" in p
        for p in sdk_teacher.validate_sdk_prompt(BASE_PROMPT, no_key)
    )

    swapped = BASE_PROMPT.replace("## 5. Retrieve", "## 6. Calculate", 1).replace(
        "## 6. Calculate\nAlways", "## 5. Retrieve\nAlways"
    )
    assert any(
        "out of order" in p
        for p in sdk_teacher.validate_sdk_prompt(BASE_PROMPT, swapped)
    )
    assert any(
        "floor" in p
        for p in sdk_teacher.validate_sdk_prompt(BASE_PROMPT, "## 1. Role\n")
    )


def test_taxonomy_is_sliced_from_the_pipeline_teachers_prompt() -> None:
    from convfinqa.evalloop import teacher

    assert sdk_teacher.TAXONOMY in teacher.TEACHER_PROMPT
    assert "retriever/wrong-period" in sdk_teacher.TAXONOMY
    assert sdk_teacher.TAXONOMY in sdk_teacher.SDK_DIAGNOSE_PROMPT


def test_sdk_teacher_prompts_resolve_by_reference() -> None:
    from convfinqa.evalloop import prompt_refs

    for name in ("SDK_DIAGNOSE_PROMPT", "SDK_WRITER_PROMPT", "SDK_DISTIL_PROMPT"):
        ref = prompt_refs.teacher_prompt_ref(name, getattr(sdk_teacher, name))
        assert prompt_refs.resolve(ref) == getattr(sdk_teacher, name)


# ---------------------------------------------------------------------------
# distil_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distil_writes_the_root_module_registers_it_and_refuses_twice(
    prompts_tmp: Path,
    registry_tmp: Path,
    fake_mlflow: FakeMlflow,
    fake_calls: FakeSdkCalls,
) -> None:
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry
    from convfinqa.tracking.prompt_ledger import prompt_hash

    fake_calls.replies["PromptDraft"] = {
        "prompt": BASE_PROMPT,
        "sections": list(HEADINGS),
        "dropped": ["[[ ## ]] plumbing", "Field guidance"],
        "notes": "kept the retrieval conventions",
    }
    out = await sdk_teacher.distil_prompt(source_version="v8", new_version="sdk_v907")

    assert prompts_pkg.load_sdk("sdk_v907") == BASE_PROMPT
    assert out["new_version"] == "sdk_v907" and out["prompt_chars"] == len(BASE_PROMPT)
    assert out["run_id"] == "run-sdk-1"
    header = Path(out["module_path"]).read_text()
    assert "Distilled from the four v8" in header and "run-sdk-1" in header
    doc = registry.load()
    entry = next(
        e for e in doc.sdk_prompts or [] if e["hash"] == prompt_hash(BASE_PROMPT)
    )
    assert entry["source"] == "distil" and entry["first_seen_in"] == "sdk_v907"

    rewrites = ledgers.load("rewrites")
    assert len(rewrites) == 1
    row = rewrites.iloc[0]
    assert (row["runtime"], row["target"], row["change_kind"]) == (
        "agent_sdk",
        "whole",
        "rewrite",
    )
    assert (row["base_version"], row["failure_class"]) == ("v8", "distil")
    assert row["prompt_hash_before"] == "" and row["prompt_hash_after"] == prompt_hash(
        BASE_PROMPT
    )
    assert fake_mlflow.kinds == ["sdk_distil"]
    assert fake_mlflow.texts["sdk_prompt.txt"] == BASE_PROMPT
    assert "distil_prompt.txt" in fake_mlflow.texts

    # What the distiller was shown, referenced by the four source prompts.
    call = fake_calls.calls[0]
    shown = json.loads(call["prompt"])
    assert set(shown["source_prompts"]) == {
        "triage",
        "preprocess",
        "retriever",
        "calculator",
    }
    assert shown["dataset_notes"].startswith("## ConvFinQA Dataset Characteristics")
    assert "Multi-Turn Dependency Chain" in shown["dataset_notes"]
    assert "## Development Commands" not in shown["dataset_notes"]
    assert shown["output_contract_schema"]["title"] == "SdkTurnResult"
    assert shown["headings"] == list(HEADINGS)
    assert call["refs"]["system_prompt"]["name"] == "SDK_DISTIL_PROMPT"
    assert call["refs"]["source_retriever"]["kind"] == "agent_prompt"
    assert call["refs"]["source_retriever"]["version"] == "v8"

    with pytest.raises(SystemExit, match="already exists"):
        await sdk_teacher.distil_prompt(source_version="v8", new_version="sdk_v907")
    assert len(fake_calls.calls) == 1, "the refusal happens before any model call"


# ---------------------------------------------------------------------------
# The writer's memory
# ---------------------------------------------------------------------------


def test_memory_flags_an_edit_whose_class_got_worse() -> None:
    ledgers.append(
        "rewrites",
        [
            {"edit_id": "e-1", "rewrite_id": "rw-1", "runtime": "agent_sdk",
             "base_version": "sdk_v1", "new_version": "sdk_v2", "target": "## 5. Retrieve",
             "failure_class": "retriever/wrong-period", "change_kind": "rule",
             "edit_text": "transcribe the row", "proposed_at": "2026-09-01T00:00:00+00:00"},
            {"edit_id": "e-2", "rewrite_id": "rw-1", "runtime": "agent_sdk",
             "base_version": "sdk_v1", "new_version": "sdk_v2", "target": "## 6. Calculate",
             "failure_class": "calculator/wrong-scale", "change_kind": "example",
             "edit_text": "a percent example", "proposed_at": "2026-09-01T00:00:00+00:00"},
            {"edit_id": "e-3", "rewrite_id": "rw-0", "runtime": "multi_agent",
             "base_version": "v8", "new_version": "v9", "target": "retriever",
             "failure_class": "retriever/wrong-period", "change_kind": "rewrite",
             "proposed_at": "2026-09-02T00:00:00+00:00"},
        ],
    )  # fmt: skip
    ledgers.append(
        "gates",
        [
            {"gate_id": "g-1", "runtime": "agent_sdk", "rewrite_id": "rw-1",
             "baseline_version": "sdk_v1", "candidate_version": "sdk_v2",
             "promoted": False, "reason": "not significant", "delta_pp": -0.5,
             "p_value": 0.4, "fixed": 4, "broken": 6,
             # Filed by class for the retriever edit, by stage for the calculator one.
             "flips_by_class": {"retriever/wrong-period": {"fixed": 1, "broken": 3},
                                "calculator": {"fixed": 3, "broken": 1}}},
        ],
    )  # fmt: skip

    attempts = sdk_teacher.sdk_attempts()
    assert [a["version"] for a in attempts] == [
        "sdk_v2",
        "sdk_v2",
    ]  # pipeline rows excluded
    by_class = {a["failure_class"]: a for a in attempts}
    worse = by_class["retriever/wrong-period"]
    assert worse["outcome"] == "rejected" and worse["revert_or_rethink"] is True
    assert worse["class_flips"] == {"fixed": 1, "broken": 3}
    better = by_class["calculator/wrong-scale"]
    assert better["class_flips"] == {"fixed": 3, "broken": 1}
    assert better["revert_or_rethink"] is False

    text = sdk_teacher.sdk_ledger_text()
    assert "REJECTED" in text and "Δ -0.50pp" in text
    assert text.count("← REVERT OR RETHINK") == 1
    assert "## 5. Retrieve · retriever/wrong-period (rule)" in text
