"""A rate-limited pass: unscored rows, a gate that refuses, and `--resume-from`.

The defect these tests exist for: on the subscription path the `claude` CLI
answered 176 of 349 turns with "You've hit your session limit · resets 5:40pm",
the backend treated each as an unparseable reply, and the pass reported 44.4%
accuracy — a number that reads like a measurement of the agent and is really
"half the turns were never answered".

Three properties are pinned here, all offline, driven by the fake SDK client of
`test_agent_sdk_backend`:

* a refused turn is written `unscored` and is absent from both sides of the
  accuracy fraction, with the pass marked incomplete everywhere it is recorded;
* neither gate will read a CSV with unscored rows, and a CSV written before the
  column existed still loads;
* `--resume-from` copies whole answered conversations through verbatim — same
  `run_id`, same `trace_id` — and re-runs partial ones from turn 0.
"""

from __future__ import annotations

# ruff: noqa: F811 — the fake SDK client and its fixtures are imported from
# `test_agent_sdk_backend` rather than copied. pytest resolves a fixture by the
# name in the module namespace, so an imported fixture is shadowed by the
# parameter of every test that requests it, which ruff reads as a redefinition.
import contextlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.test_agent_sdk_backend import (
    REPORT,
    SDK_VERSION,
    FakeSdk,
    Script,
    api_key,  # noqa: F401 — a fixture, used by name
    fake_sdk,  # noqa: F401
    number_turn,
    program_turn,
    registry_tmp,  # noqa: F401
    sdk_prompt_module,  # noqa: F401
)

REPORT_A = REPORT
REPORT_B = "Fake/2021/page_2.pdf"

SESSION_LIMIT = "You've hit your session limit · resets 5:40pm (Australia/Sydney)"


# ---------------------------------------------------------------------------
# A two-conversation split, run with the recorder and the SDK client faked
# ---------------------------------------------------------------------------


def _examples() -> list[Any]:
    """Conversation A (2 turns) and B (3 turns), in that order."""
    from convfinqa.data.schemas import ConvExample

    return [
        ConvExample(
            report_id=REPORT_A,
            questions=["revenue in 2020?", "and the change from 2019?"],
            gold_answers=["200", "150"],
            gold_programs=["200", "subtract(200, 50)"],
            gold_turn_types=["number", "program"],
            gold_conv_types=["Type I", "Type I"],
        ),
        ConvExample(
            report_id=REPORT_B,
            questions=["revenue in 2019?", "and in 2020?", "and the change?"],
            gold_answers=["50", "200", "150"],
            gold_programs=["50", "200", "subtract(200, 50)"],
            gold_turn_types=["number", "number", "program"],
            gold_conv_types=["Type I", "Type I", "Type I"],
        ),
    ]


@pytest.fixture
def split_of_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """Both reports in the loader, both examples as the whole train split."""
    from convfinqa.data import loader
    from convfinqa.data.schemas import Document
    from convfinqa.evalloop import runner
    from convfinqa.tracking import tracing

    document = Document(
        pre_text="revenue was 200 in 2020 and 50 in 2019.",
        post_text="",
        table={"revenue": {"2020": 200, "2019": 50}},
    )
    for report_id in (REPORT_A, REPORT_B):
        monkeypatch.setitem(loader._DOCS, report_id, document)
    examples = _examples()
    monkeypatch.setattr(
        runner, "split_report_ids", lambda *a, **k: [REPORT_A, REPORT_B]
    )
    monkeypatch.setattr(runner, "examples_for", lambda ids: list(examples))
    monkeypatch.setattr(runner, "PREDICTIONS_DIR", tmp_path / "preds")
    monkeypatch.setattr(tracing, "enable", lambda: False)
    return examples


def install_recorder(monkeypatch: pytest.MonkeyPatch, run_id: str) -> dict[str, Any]:
    """Replace `mlflow_log.run` with a recorder that keeps what it was told."""
    from convfinqa.tracking import mlflow_log

    logged: dict[str, Any] = {"params": {}, "tags": {}, "metrics": {}}

    class _Rec:
        def __init__(self) -> None:
            self.run_id = run_id

        def metrics(self, values: dict[str, float]) -> None:
            logged["metrics"].update(values)

        def metric(self, key: str, value: float, *, step: int | None = None) -> None:
            logged["metrics"][key] = value

        def artifact(self, path: Any) -> None:
            logged["artifact"] = str(path)

        def dict_artifact(self, name: str, payload: Any) -> None:
            logged.setdefault("dicts", {})[name] = payload

        def text_artifact(self, name: str, text: str) -> None:
            logged.setdefault("texts", {})[name] = text

        def tag(self, key: str, value: str) -> None:
            logged["tags"][key] = value

        def param(self, key: str, value: Any) -> None:
            logged["params"][key] = value

    @contextlib.contextmanager
    def fake_run(name: str, **kw: Any) -> Any:
        logged["name"] = name
        logged["params"].update(kw.get("params") or {})
        logged["tags"].update(kw.get("tags") or {})
        yield _Rec()

    monkeypatch.setattr(mlflow_log, "run", fake_run)
    return logged


def rate_limited_scripts() -> list[Script]:
    """A answered whole; B refused from its second turn onwards.

    With `concurrency=1` the conversations are walked in split order, so a flat
    script list is deterministic: A q0, A q1, B q0, then the refusal.
    """
    return [
        Script(number_turn("200")),
        Script(program_turn("150.0"), tool_calls=[("subtract", 200, 50)]),
        Script(number_turn("50")),
        Script(SESSION_LIMIT),
    ]


async def run_pass(
    fake: FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    scripts: list[Script],
    resume_from: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    from convfinqa.evalloop import runner

    logged = install_recorder(monkeypatch, run_id)
    fake.scripts = list(scripts)
    summary = await runner.run_split(
        "train",
        SDK_VERSION,
        runtime="agent_sdk",
        concurrency=1,
        resume_from=resume_from,
    )
    return summary, logged, pd.read_csv(summary["csv"])


# ---------------------------------------------------------------------------
# A rate-limited pass is incomplete, not inaccurate
# ---------------------------------------------------------------------------


async def test_a_rate_limited_pass_writes_unscored_rows_and_scores_only_the_rest(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.backends.agent_sdk import RATE_LIMIT_ERROR_PREFIX

    summary, logged, df = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )

    # Every question of the split is present — the gate has to be able to see
    # that the question exists and was not answered.
    assert list(df["question_id"]) == [
        f"{REPORT_A}_q0",
        f"{REPORT_A}_q1",
        f"{REPORT_B}_q0",
        f"{REPORT_B}_q1",
        f"{REPORT_B}_q2",
    ]
    assert list(df["unscored"]) == [False, False, False, True, True]
    refused, never = df.iloc[3], df.iloc[4]
    assert str(refused["error"]) == f"{RATE_LIMIT_ERROR_PREFIX}{SESSION_LIMIT}"
    assert str(never["error"]).startswith(RATE_LIMIT_ERROR_PREFIX)
    assert "turn not attempted" in str(never["error"])
    assert pd.isna(refused["pred_answer"]) and pd.isna(never["pred_answer"])
    assert not bool(refused["correct"]) and not bool(never["correct"])
    # An unscored turn is not a first-wrong turn either: B's history is clean.
    assert pd.isna(df.iloc[2]["first_wrong_turn"])

    # Accuracy is over the three turns that were answered, all of them right.
    assert summary["accuracy"] == 1.0
    assert summary["complete"] is False
    assert summary["n_unscored"] == 2
    assert summary["n_rate_limited"] == 1
    assert summary["n_scored"] == 3 and summary["n_questions"] == 5
    m = logged["metrics"]
    assert m["accuracy"] == 1.0
    assert (m["n_unscored"], m["n_rate_limited"], m["complete"]) == (2.0, 1.0, 0.0)
    assert (m["n_scored"], m["n_questions"], m["n_wrong"]) == (3.0, 5.0, 0.0)
    # The panel, program accuracy and the slices all read the scored subset.
    assert m["acc_calculator_exec"] == 1.0
    assert m["acc_triage_turn_type"] == 1.0
    assert m["program_accuracy"] == 1.0 and m["n_program_turns"] == 1.0
    assert m["accuracy_gold_turn_type_number"] == 1.0
    # And the store cannot mistake it for a finished pass.
    assert logged["tags"]["incomplete"] == "true"
    assert logged["params"]["unscored_rows"] == 2


async def test_a_rate_limited_conversation_keeps_the_turns_it_did_answer(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B q0 was answered before the refusal, so it is scored as itself."""
    _summary, _logged, df = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )
    answered = df[df["question_id"] == f"{REPORT_B}_q0"].iloc[0]
    assert not bool(answered["unscored"])
    assert float(answered["pred_answer"]) == 50.0 and bool(answered["correct"])
    assert str(answered["run_id"]) == "run-partial"
    # Nothing was copied into this pass, so the provenance column is empty.
    assert pd.isna(answered["resumed_from_run_id"])


# ---------------------------------------------------------------------------
# Neither gate reads incomplete evidence
# ---------------------------------------------------------------------------


async def test_both_gates_refuse_a_csv_with_unscored_rows(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from convfinqa.evalloop import gate, sdk_gate

    _s, _l, _df = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )
    partial = tmp_path / "partial.csv"
    shutil.copy(_s["csv"], partial)

    complete_scripts = [
        *rate_limited_scripts()[:3],
        Script(number_turn("200")),
        Script(program_turn("150.0"), tool_calls=[("subtract", 200, 50)]),
    ]
    whole, _l2, _df2 = await run_pass(
        fake_sdk, monkeypatch, run_id="run-whole", scripts=complete_scripts
    )
    assert whole["complete"] is True
    good = tmp_path / "complete.csv"
    shutil.copy(whole["csv"], good)

    with pytest.raises(gate.IncompleteRunError, match=r"partial\.csv: 2 of 5"):
        gate.load_run_csv(partial)
    with pytest.raises(gate.IncompleteRunError, match="resume-from"):
        gate.gate_runs(
            good,
            partial,
            baseline_version=SDK_VERSION,
            candidate_version=SDK_VERSION,
        )
    with pytest.raises(gate.IncompleteRunError, match=r"partial\.csv"):
        sdk_gate.gate_overall(
            partial,
            good,
            baseline_version=SDK_VERSION,
            candidate_version=SDK_VERSION,
        )

    # A complete pass goes through both doors.
    assert len(gate.load_run_csv(good)) == 5
    _result, stats = gate.gate_runs(
        good,
        good,
        baseline_version=SDK_VERSION,
        candidate_version=SDK_VERSION,
    )
    assert stats["n_compared"] == 5 and stats["accuracy_delta"] == 0.0
    verdict, _comparison = sdk_gate.gate_overall(
        good,
        good,
        baseline_version=SDK_VERSION,
        candidate_version=SDK_VERSION,
    )
    assert verdict["overall_delta"] == 0.0

    # And a CSV written before the column existed is read as all-scored.
    legacy = tmp_path / "legacy.csv"
    frame = pd.read_csv(good).drop(columns=["unscored", "resumed_from_run_id"])
    frame.to_csv(legacy, index=False)
    loaded = gate.load_run_csv(legacy)
    assert not bool(loaded["unscored"].any())


# ---------------------------------------------------------------------------
# --resume-from: whole conversations, or none of them
# ---------------------------------------------------------------------------


async def test_resume_reuses_whole_conversations_and_reruns_partial_ones(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, _logged, before = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )
    prior = tmp_path / "prior.csv"
    shutil.copy(first["csv"], prior)

    # B is re-run from turn 0 — all three of its questions, in one new session.
    summary, logged, df = await run_pass(
        fake_sdk,
        monkeypatch,
        run_id="run-resumed",
        scripts=[
            Script(number_turn("50")),
            Script(number_turn("200")),
            Script(program_turn("150.0"), tool_calls=[("subtract", 200, 50)]),
        ],
        resume_from=prior,
    )

    assert fake_sdk.clients == 3, "two sessions in the first pass, one in the second"
    assert summary["complete"] is True and summary["n_unscored"] == 0
    assert summary["n_reused_conversations"] == 1
    assert summary["n_reused_questions"] == 2
    assert summary["accuracy"] == 1.0
    assert logged["params"]["resumed_from"] == "prior.csv"
    assert logged["params"]["n_reused_conversations"] == 1
    assert logged["metrics"]["n_reused_questions"] == 2.0

    # Every question of the split exactly once, in report/turn order.
    assert list(df["question_id"]) == [
        f"{REPORT_A}_q0",
        f"{REPORT_A}_q1",
        f"{REPORT_B}_q0",
        f"{REPORT_B}_q1",
        f"{REPORT_B}_q2",
    ]
    assert list(df["turn_index"]) == [0, 1, 0, 1, 2]
    assert not bool(df["unscored"].any())

    # A's rows are the prior pass's rows, provenance intact.
    reused = df.iloc[:2]
    prior_a = before.iloc[:2]
    assert list(reused["run_id"]) == ["run-partial", "run-partial"]
    assert list(reused["trace_id"]) == list(prior_a["trace_id"])
    assert list(reused["resumed_from_run_id"]) == ["run-partial", "run-partial"]
    for column in ("pred_answer", "pred_program", "history_text", "calculator_io"):
        assert list(reused[column].fillna("")) == list(prior_a[column].fillna(""))

    # B's rows are this pass's, and say so.
    fresh = df.iloc[2:]
    assert list(fresh["run_id"]) == ["run-resumed"] * 3
    assert list(fresh["resumed_from_run_id"].fillna("")) == ["", "", ""]
    assert [float(v) for v in fresh["pred_answer"]] == [50.0, 200.0, 150.0]

    # The panel is recomputed over the whole frame, reused rows included.
    from convfinqa.evalloop import stage_scores

    assert list(df["triage_turn_type_ok"]) == [True] * 5
    assert stage_scores.run_metrics(df)["acc_calculator_exec"] == 1.0


async def test_resume_refuses_a_prior_csv_that_is_not_this_pass(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Split, version, runtime and report set are all checked before any call."""
    from convfinqa.evalloop import runner

    first, _logged, _df = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )
    prior = pd.read_csv(first["csv"])

    def written(frame: pd.DataFrame, name: str) -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    other_split = prior.copy()
    other_split["split"] = "test"
    other_version = prior.copy()
    other_version["model_version_id"] = "sdk_v9"
    other_runtime = prior.copy()
    other_runtime["runtime"] = "pipeline"
    wider = prior.copy()
    wider.loc[wider.index[0], "report_id"] = "Fake/2019/page_9.pdf"

    # The runtime of a CSV that does not name one is read off the run-name
    # prefix, and otherwise off the version's own lineage.
    assert runner.prior_runtime(prior, "sdk-evalloop-train2-x.csv") == "agent_sdk"
    assert runner.prior_runtime(prior, "whatever.csv") == "agent_sdk"
    pipeline_prior = prior.copy()
    pipeline_prior["model_version_id"] = "v2"
    assert runner.prior_runtime(pipeline_prior, "evalloop-train2-x.csv") == "pipeline"

    install_recorder(monkeypatch, "run-refused")
    for frame, name, message in (
        (other_split, "sdk-evalloop-other-split.csv", "is a test run"),
        (other_version, "sdk-evalloop-other-version.csv", "ran version sdk_v9"),
        (other_runtime, "sdk-evalloop-pipeline.csv", "ran runtime 'pipeline'"),
        (wider, "sdk-evalloop-wider.csv", "must be a superset"),
    ):
        with pytest.raises(ValueError, match=message):
            await runner.run_split(
                "train",
                SDK_VERSION,
                runtime="agent_sdk",
                concurrency=1,
                resume_from=written(frame, name),
            )
    assert fake_sdk.clients == 2, "no refused resume opened a session"


async def test_resume_refuses_a_different_train_draw(
    fake_sdk: FakeSdk,
    split_of_two: list[Any],
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--train-seed` draws its own reports; a resume may not cross draws."""
    from convfinqa.evalloop import runner

    first, _logged, _df = await run_pass(
        fake_sdk, monkeypatch, run_id="run-partial", scripts=rate_limited_scripts()
    )
    prior = pd.read_csv(first["csv"])
    prior.loc[prior.index[0], "report_id"] = "Fake/2019/page_9.pdf"
    path = tmp_path / "sdk-evalloop-drawn.csv"
    prior.to_csv(path, index=False)

    monkeypatch.setattr(
        runner, "draw_train", lambda **kw: ([REPORT_A, REPORT_B], {"seed": kw["seed"]})
    )
    install_recorder(monkeypatch, "run-refused")
    with pytest.raises(ValueError, match="--train-seed 7"):
        await runner.run_split(
            "train",
            SDK_VERSION,
            runtime="agent_sdk",
            concurrency=1,
            train_seed=7,
            resume_from=path,
        )


async def test_a_partial_conversation_is_never_stitched_mid_session(
    split_of_two: list[Any],
) -> None:
    """The unit under the resume rule: whole or nothing, per conversation."""
    from convfinqa.backends.agent_sdk import RATE_LIMIT_ERROR_PREFIX
    from convfinqa.evalloop import runner

    prior = pd.DataFrame(
        [
            # A: answered whole.
            {"report_id": REPORT_A, "turn_index": 0, "unscored": False, "error": ""},
            {"report_id": REPORT_A, "turn_index": 1, "unscored": False, "error": ""},
            # B: two of three turns, the second one refused.
            {"report_id": REPORT_B, "turn_index": 0, "unscored": False, "error": ""},
            {
                "report_id": REPORT_B,
                "turn_index": 1,
                "unscored": True,
                "error": f"{RATE_LIMIT_ERROR_PREFIX}{SESSION_LIMIT}",
            },
        ]
    )
    reusable = runner.reusable_conversations(prior, split_of_two)
    assert set(reusable) == {REPORT_A}

    # A conversation whose rows are all scored but not all present is not
    # reusable either — a short conversation is still a partial one.
    short = prior[prior["report_id"] == REPORT_A].iloc[:1]
    assert runner.reusable_conversations(short, split_of_two) == {}

    # Nor is one that ends in a rate-limited row that predates the column.
    legacy = prior.drop(columns=["unscored"]).copy()
    assert set(runner.reusable_conversations(legacy, split_of_two)) == {REPORT_A}
