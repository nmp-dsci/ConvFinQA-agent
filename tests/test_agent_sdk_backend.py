"""The single-session Agent SDK runtime, driven by a fake SDK client.

No test here reaches a model. The `claude_agent_sdk` module is real — its
message classes, `tool` and the options dataclass are what the runtime is
written against — but `ClaudeSDKClient` is replaced by a scripted fake that
runs the registered tool handlers itself, so the trajectory, the spans and the
capture are produced by the real code paths.

What is pinned: the capture the runtime writes is the pipeline's capture, so
`_capture_to_row_fields`, `stage_scores.score_rows` and `first_fault` read it
unchanged; arithmetic happens only in tools; the budget, the early stop and the
corrective retry behave as the runner expects; and the sdk prompt lineage,
references and aliases stay apart from the pipeline's.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import claude_agent_sdk
import pandas as pd
import pytest

REPORT = "Fake/2020/page_1.pdf"
SDK_VERSION = "sdk_v7"
SDK_PROMPT_TEXT = "You answer ConvFinQA turns in one session. Use the calculator tools."


# ---------------------------------------------------------------------------
# Fixtures: a prompt module in the sdk lineage, a document, a fake SDK client
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk_prompt_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Register `convfinqa.prompts.sdk_v7` in sys.modules without writing a file."""
    module = types.ModuleType(f"convfinqa.prompts.{SDK_VERSION}")
    module.SDK_PROMPT = SDK_PROMPT_TEXT  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, f"convfinqa.prompts.{SDK_VERSION}", module)
    yield SDK_VERSION


@pytest.fixture
def fake_document(monkeypatch: pytest.MonkeyPatch) -> str:
    from convfinqa.data import loader
    from convfinqa.data.schemas import Document

    monkeypatch.setitem(
        loader._DOCS,
        REPORT,
        Document(
            pre_text="revenue was 200 in 2020 and 50 in 2019.",
            post_text="",
            table={"revenue": {"2020": 200, "2019": 50}},
        ),
    )
    return REPORT


@pytest.fixture
def registry_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A private copy of registry.json so no test writes the committed one."""
    from convfinqa.tracking import registry

    target = tmp_path / "registry.json"
    # The committed registry now carries the real sdk lineage (sdk_v1 = s1);
    # these tests register a fake sdk version and assert it becomes s1, so the
    # copy starts with an empty sdk lineage.
    doc = json.loads(registry.REGISTRY_PATH.read_text())
    doc["sdk_prompts"] = []
    doc["aliases"] = {
        k: v for k, v in doc.get("aliases", {}).items() if k != "sdk_champion"
    }
    target.write_text(json.dumps(doc, indent=2))
    monkeypatch.setattr(registry, "REGISTRY_PATH", target)
    return target


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the runtime down the API-billed path, deliberately.

    `sdk_billing` defaults to `subscription` and `llm.sdk_child_env` refuses
    `api` unless the escape hatch is set (owner's instruction, 2026-09-05).
    These tests select the API path on purpose: it is the one whose child
    environment can be asserted from a placeholder key, with no login and no
    keychain involved. The refusal itself is pinned by `test_llm.py`.
    """
    from convfinqa.config import settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("SDK_ALLOW_API_BILLING", "1")
    monkeypatch.setattr(settings, "sdk_billing", "api", raising=False)


class Script:
    """What the fake session does for one question: tool calls, then a reply."""

    def __init__(
        self,
        payload: Any,
        *,
        tool_calls: list[tuple[str, float, float]] | None = None,
        usage: dict[str, int] | None = None,
        is_error: bool = False,
    ) -> None:
        self.payload = payload
        self.tool_calls = tool_calls or []
        self.usage = usage or {"input_tokens": 100, "output_tokens": 20}
        self.is_error = is_error


class FakeSdk:
    """The scripted stand-in for `ClaudeSDKClient` plus the captured tools."""

    def __init__(self) -> None:
        self.scripts: list[Script] = []
        self.prompts: list[str] = []
        self.options: list[Any] = []
        self.clients = 0
        self.tools: dict[str, Any] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSdk:
        fake = self

        def create_server(name: str, version: str = "1.0.0", tools: Any = None) -> Any:
            fake.tools = {t.name: t for t in (tools or [])}
            return {"type": "sdk", "name": name, "tools": fake.tools}

        class _Client:
            def __init__(self, options: Any = None, **_: Any) -> None:
                fake.clients += 1
                fake.options.append(options)

            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                fake.prompts.append(prompt)

            async def receive_response(self) -> Any:
                script = fake.scripts.pop(0)
                blocks: list[Any] = []
                for name, a, b in script.tool_calls:
                    blocks.append(
                        claude_agent_sdk.ToolUseBlock(
                            id="t", name=f"mcp__cfq__{name}", input={"a": a, "b": b}
                        )
                    )
                    await fake.tools[name].handler({"a": a, "b": b})
                if blocks:
                    yield claude_agent_sdk.AssistantMessage(content=blocks, model="m")
                yield claude_agent_sdk.ResultMessage(
                    subtype="success",
                    duration_ms=5,
                    duration_api_ms=4,
                    is_error=script.is_error,
                    num_turns=1 + len(script.tool_calls),
                    session_id="s",
                    total_cost_usd=0.01,
                    usage=script.usage,
                    structured_output=script.payload,
                    result=None if isinstance(script.payload, dict) else script.payload,
                )

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _Client)
        monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", create_server)
        return self


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch, api_key: None) -> FakeSdk:
    return FakeSdk().install(monkeypatch)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every `tracing.span` the runtime opens, with what it put on it."""
    from convfinqa.tracking import tracing

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
    return opened


def number_turn(answer: str = "200") -> dict[str, Any]:
    return {
        "turn_type": "number",
        "conv_type": "Type I",
        "answer": answer,
        "reasoning": "read it off the table",
    }


def program_turn(
    answer: Any = "150.0",
    *,
    program: str = "subtract(A, B)",
    retrieved: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "turn_type": "program",
        "conv_type": "Type I",
        "sub_questions": ["revenue in 2020", "revenue in 2019"],
        "program": program,
        "retrieved": retrieved
        if retrieved is not None
        else [
            {"question": "revenue in 2020", "answer": "200", "source": "table"},
            {"question": "revenue in 2019", "answer": 50, "source": "table"},
        ],
        "answer": answer,
        "reasoning": "difference",
    }


async def _run(
    fake: FakeSdk, questions: list[str], **kw: Any
) -> tuple[Any, Any, list[dict[str, Any]]]:
    from convfinqa.backends import agent_sdk

    captures: list[dict[str, Any]] = []
    preds, programs = await agent_sdk.run_conversation(
        REPORT,
        questions,
        system_prompt=SDK_PROMPT_TEXT,
        captures=captures,
        version=SDK_VERSION,
        **kw,
    )
    return preds, programs, captures


def _rows(
    captures: list[dict[str, Any]], gold: list[tuple[str, str, str]]
) -> pd.DataFrame:
    """Frame rows the way `run_split` builds them, from captures plus gold."""
    from convfinqa.evaluation.metrics import numeric_match
    from convfinqa.evaluation.runner import _capture_to_row_fields

    rows = []
    for i, ((question, gold_answer, gold_program), cap) in enumerate(
        zip(gold, captures, strict=True)
    ):
        answer = (cap.get("calculator") or cap.get("retriever") or {}).get("output", {})
        pred = (
            answer.get("answer")
            if "answer" in answer
            else (answer.get("answers") or [{}])[0].get("answer", "")
        )
        rows.append(
            {
                "report_id": REPORT,
                "turn_index": i,
                "question_id": f"{REPORT}_q{i}",
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred,
                "correct": numeric_match(pred, gold_answer),
                "cascade": False,
                "first_wrong_turn": None,
                "pred_program": ((cap.get("preprocess") or {}).get("output") or {}).get(
                    "program", ""
                ),
                "gold_program": gold_program,
                "gold_turn_type": "number"
                if not gold_program or "(" not in gold_program
                else "program",
                "gold_conv_type": "Type I",
                **_capture_to_row_fields(cap),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The capture contract
# ---------------------------------------------------------------------------


async def test_a_two_turn_conversation_writes_the_pipelines_capture(
    fake_sdk: FakeSdk,
    fake_document: str,
    sdk_prompt_module: str,
    spans: list[dict[str, Any]],
) -> None:
    """Turn 0 a number, turn 1 a program with two tool calls — one session."""
    from convfinqa.evalloop import stage_scores
    from convfinqa.evaluation.runner import _capture_to_row_fields
    from convfinqa.llm import SDK_ALLOWED_TOOLS

    fake_sdk.scripts = [
        Script(number_turn("200")),
        Script(
            program_turn("4.0", program="divide(A, B)"),
            tool_calls=[("subtract", 200, 50), ("divide", 200, 50)],
        ),
    ]
    preds, programs, captures = await _run(
        fake_sdk, ["what was revenue in 2020?", "and its ratio of change?"]
    )

    assert preds == ["200", "4.0"]
    assert programs == ["", "divide(A, B)"]
    assert fake_sdk.clients == 1, "one session per conversation"

    # The exact key set turn_events fills, with the pipeline's None rules.
    core = {"history_text", "triage", "preprocess", "retriever", "calculator"}
    for cap in captures:
        assert core <= set(cap)
        assert "error" not in cap
    number, program = captures
    assert number["preprocess"] is None and number["calculator"] is None
    assert number["retriever"]["output"]["answers"] == [
        {"question": "what was revenue in 2020?", "answer": "200"}
    ]
    assert number["triage"]["output"]["turn_type"] == "number"
    assert program["preprocess"]["output"]["sub_questions"] == [
        "revenue in 2020",
        "revenue in 2019",
    ]
    assert program["retriever"]["output"]["answers"] == [
        {"question": "revenue in 2020", "answer": "200"},
        {"question": "revenue in 2019", "answer": "50"},
    ]
    assert program["retriever"]["output"]["sources"] == ["table", "table"]

    # History is rendered exactly as the pipeline renders it.
    assert number["history_text"] == "(no prior turns)"
    assert program["history_text"] == (
        f"Q1 [report={REPORT}]: what was revenue in 2020?\nA1: 200"
    )

    # Arithmetic only through the tools: the calls are in the trajectory and
    # the answer is one of their returns.
    traj = program["calculator"]["trajectory"]
    assert [e["event"] for e in traj] == ["tool_call", "tool_return"] * 2
    assert traj[0] == {
        "event": "tool_call",
        "tool": "subtract",
        "args": {"a": 200, "b": 50},
    }
    assert traj[1] == {"event": "tool_return", "tool": "subtract", "result": "150.0"}
    returns = [e["result"] for e in traj if e["event"] == "tool_return"]
    assert "4.0" in returns
    assert program["sdk"]["tool_calls"] == 2
    assert program["sdk"]["stage_skips"] == []
    assert program["sdk"]["inline_arithmetic"] is False
    assert set(program["sdk"]) == {
        "num_turns",
        "tool_calls",
        "stage_skips",
        "inline_arithmetic",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
    }

    # The shared row builder and scorer read it without knowing the runtime.
    fields = _capture_to_row_fields(program)
    assert set(fields) == {
        "pred_turn_type",
        "pred_conv_type",
        "pred_sub_questions",
        "history_text",
        "triage_io",
        "preprocess_io",
        "retriever_io",
        "calculator_io",
        "error",
    }
    assert json.loads(fields["pred_sub_questions"]) == [
        "revenue in 2020",
        "revenue in 2019",
    ]
    df = _rows(
        captures,
        [
            ("what was revenue in 2020?", "200", "200"),
            ("and its ratio of change?", "4", "divide(200, 50)"),
        ],
    )
    scored = stage_scores.score_rows(df)
    assert list(scored["triage_turn_type_ok"]) == [True, True]
    assert list(scored["retriever_operand_recall"]) == [1.0, 1.0]
    assert scored["preprocess_plan_ok"].iloc[1] is True
    assert stage_scores.run_metrics(scored)["acc_calculator_exec"] == 1.0

    # The document goes in the first message only; later turns are the question.
    assert "[[ ## document ## ]]" in fake_sdk.prompts[0]
    assert "revenue was 200 in 2020" in fake_sdk.prompts[0]
    assert fake_sdk.prompts[0].endswith("what was revenue in 2020?")
    assert fake_sdk.prompts[1] == "[[ ## question ## ]]\nand its ratio of change?"

    # Options: only the six calculator tools, no built-ins, structured output.
    options = fake_sdk.options[0]
    assert options.allowed_tools == SDK_ALLOWED_TOOLS
    assert options.tools == []
    assert set(options.mcp_servers) == {"cfq"}
    assert options.output_format["type"] == "json_schema"
    assert options.setting_sources == []
    assert options.env["ANTHROPIC_API_KEY"] == "sk-ant-test-not-real"

    # Traced by hand: a question span, an LLM span per SDK turn with the prompt
    # by reference, and a TOOL span per call.
    llm = [s for s in spans if s.get("span_type") == "LLM"]
    assert len(llm) == 2
    assert llm[0]["inputs"]["refs"]["kind"] == "sdk_prompt"
    assert llm[0]["inputs"]["refs"]["version"] == SDK_VERSION
    assert llm[0]["inputs"]["prompt_head"].startswith("[[ ## report_id ## ]]")
    assert llm[1]["outputs"]["answer"] == "4.0"
    assert llm[1]["attrs"]["input_tokens"] == 100
    assert [s["name"] for s in spans if s.get("span_type") == "TOOL"] == [
        "subtract",
        "divide",
    ]
    assert [s["name"][:3] for s in spans if s["name"].startswith("q")] == ["q0:", "q1:"]


async def test_an_empty_program_is_a_preprocess_skip_and_a_preprocess_fault(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    from convfinqa.evalloop import stage_scores

    fake_sdk.scripts = [
        Script(program_turn("150.0", program=""), tool_calls=[("subtract", 200, 50)])
    ]
    _preds, _programs, captures = await _run(fake_sdk, ["what was the change?"])
    assert "preprocess" in captures[0]["sdk"]["stage_skips"]
    row = _rows(captures, [("what was the change?", "150", "subtract(200, 50)")]).iloc[
        0
    ]
    assert (
        stage_scores.first_fault(row.to_dict(), "revenue was 200 and 50")
        == "preprocess"
    )


async def test_no_tool_call_is_a_calculator_skip_and_a_calculator_fault(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    """Every upstream check passes; the answer is wrong; nothing was computed."""
    from convfinqa.evalloop import stage_scores

    fake_sdk.scripts = [Script(program_turn("149"))]  # no tool calls at all
    _preds, _programs, captures = await _run(fake_sdk, ["what was the change?"])
    sdk = captures[0]["sdk"]
    assert sdk["stage_skips"] == ["calculator"]
    assert sdk["tool_calls"] == 0
    assert captures[0]["calculator"]["trajectory"] == []
    row = _rows(captures, [("what was the change?", "150", "subtract(200, 50)")]).iloc[
        0
    ]
    assert (
        stage_scores.first_fault(row.to_dict(), "revenue was 200 and 50")
        == "calculator"
    )


async def test_an_answer_matching_no_tool_return_is_inline_arithmetic(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    fake_sdk.scripts = [Script(program_turn("151"), tool_calls=[("subtract", 200, 50)])]
    _preds, _programs, captures = await _run(fake_sdk, ["what was the change?"])
    assert captures[0]["sdk"]["inline_arithmetic"] is True
    assert captures[0]["sdk"]["stage_skips"] == []


async def test_a_tool_error_is_returned_to_the_model_not_raised(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    fake_sdk.scripts = [Script(program_turn("0"), tool_calls=[("divide", 1, 0)])]
    _preds, _programs, captures = await _run(fake_sdk, ["ratio?"])
    traj = captures[0]["calculator"]["trajectory"]
    assert traj[1]["event"] == "tool_return"
    assert traj[1]["result"].startswith("error:")


# ---------------------------------------------------------------------------
# Session control: budget, early stop, corrective retry
# ---------------------------------------------------------------------------


async def test_the_token_ceiling_marks_the_remaining_turns_as_errors(
    fake_sdk: FakeSdk,
    fake_document: str,
    sdk_prompt_module: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "sdk_total_tokens_limit", 150, raising=False)
    fake_sdk.scripts = [
        Script(number_turn("200"), usage={"input_tokens": 100, "output_tokens": 20}),
        Script(number_turn("50"), usage={"input_tokens": 100, "output_tokens": 20}),
        Script(number_turn("1")),
    ]
    preds, _programs, captures = await _run(fake_sdk, ["a?", "b?", "c?"])
    assert preds == ["200", "50", ""]
    assert "token budget exhausted" in captures[2]["error"]
    assert captures[2]["history_text"].count("\nA") == 2, (
        "the error turn still saw its history"
    )
    assert len(fake_sdk.prompts) == 2, "the third question was never sent"
    assert "sdk" not in captures[2]


async def test_stop_after_ends_the_session_early(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    fake_sdk.scripts = [Script(number_turn("7")), Script(number_turn("8"))]
    preds, programs, captures = await _run(
        fake_sdk, ["a?", "b?"], stop_after=lambda i, answer: answer == "7"
    )
    assert preds == ["7"] and programs == [""] and len(captures) == 1
    assert len(fake_sdk.prompts) == 1


async def test_an_invalid_reply_gets_one_correction_then_an_error_row(
    fake_sdk: FakeSdk,
    fake_document: str,
    sdk_prompt_module: str,
    spans: list[dict[str, Any]],
) -> None:
    """Two bad replies: the turn is an error, the history still advances."""
    fake_sdk.scripts = [
        Script({"turn_type": "number"}),  # no answer
        Script("not json at all"),
        Script(number_turn("50")),
    ]
    preds, _programs, captures = await _run(fake_sdk, ["a?", "b?"])
    assert preds == ["", "50"]
    assert "no valid reply after 2 attempts" in captures[0]["error"]
    assert "did not match the required output schema" in fake_sdk.prompts[1]
    assert captures[1]["history_text"] == f"Q1 [report={REPORT}]: a?\nA1: "
    llm = [s for s in spans if s.get("span_type") == "LLM"]
    assert [s["attributes"]["attempt"] for s in llm] == [1, 2, 1]
    assert "error" in llm[0]["attrs"] and "error" in llm[1]["attrs"]


async def test_a_corrected_reply_is_accepted(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    fake_sdk.scripts = [Script("garbage"), Script(number_turn("9"))]
    preds, _programs, captures = await _run(fake_sdk, ["a?"])
    assert preds == ["9"] and "error" not in captures[0]


# ---------------------------------------------------------------------------
# Refusals: a turn the CLI would not answer is not a wrong answer
# ---------------------------------------------------------------------------

SESSION_LIMIT = "You've hit your session limit · resets 5:40pm (Australia/Sydney)"


def test_rate_limit_markers_match_the_clis_refusals_and_nothing_else() -> None:
    """The classifier, alone. Every marker fires; ordinary prose does not."""
    from convfinqa.backends.agent_sdk import rate_limit_refusal

    for refusal in (
        SESSION_LIMIT,
        "YOU'VE HIT YOUR SESSION LIMIT",
        "Session limit reached — resets at 5pm",
        "429: rate limit exceeded, retry later",
        "You have reached your weekly usage limit",
        "Credit balance is too low. Add credits to continue.",
        "quota exceeded for this organization",
    ):
        assert rate_limit_refusal(refusal) == refusal.strip()

    for answer in (
        "the debt limit rose to 5",
        "there is no limit on the number of shares",
        "the lease resets annually",
        "",
        "   ",
    ):
        assert rate_limit_refusal(answer) is None
    assert rate_limit_refusal(None, 42, {"answer": "1"}) is None


async def test_a_session_limit_reply_is_a_refusal_that_aborts_the_conversation(
    fake_sdk: FakeSdk,
    fake_document: str,
    sdk_prompt_module: str,
    spans: list[dict[str, Any]],
) -> None:
    """One ask, no correction, no later turn: the session is spent."""
    from convfinqa.backends import agent_sdk

    fake_sdk.scripts = [
        Script(number_turn("200")),
        Script(SESSION_LIMIT),
        Script(number_turn("50")),  # never reached
    ]
    captures: list[dict[str, Any]] = []
    with pytest.raises(agent_sdk.SdkRateLimitError) as caught:
        await agent_sdk.run_conversation(
            REPORT,
            ["a?", "b?", "c?"],
            system_prompt=SDK_PROMPT_TEXT,
            captures=captures,
            version=SDK_VERSION,
        )

    exc = caught.value
    assert isinstance(exc, agent_sdk.SdkTurnError), "a refusal is still a turn error"
    assert exc.refusal == SESSION_LIMIT
    assert exc.turn_index == 1
    # The answers captured before the refusal survive; nothing after it exists.
    assert exc.preds == ["200"] and exc.programs == [""]
    assert len(fake_sdk.prompts) == 2, (
        "the refusal was not retried and q2 was never asked"
    )
    assert len(captures) == 2
    refused = captures[1]
    assert refused["rate_limited"] is True
    assert refused["error"] == f"{agent_sdk.RATE_LIMIT_ERROR_PREFIX}{SESSION_LIMIT}"
    assert "sdk" not in refused, "no usage: there was no reply"
    llm = [s for s in spans if s.get("span_type") == "LLM"]
    assert [s["attributes"]["attempt"] for s in llm] == [1, 1]
    assert llm[1]["attrs"]["rate_limited"] is True


async def test_a_credit_balance_refusal_is_classified_the_same_way(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    """The API-billing failure is the other half of the same class."""
    from convfinqa.backends import agent_sdk

    fake_sdk.scripts = [Script("Credit balance is too low"), Script(number_turn("1"))]
    with pytest.raises(agent_sdk.SdkRateLimitError, match="Credit balance"):
        await agent_sdk.run_conversation(
            REPORT,
            ["a?"],
            system_prompt=SDK_PROMPT_TEXT,
            version=SDK_VERSION,
        )
    assert len(fake_sdk.prompts) == 1, "a refusal never spends the corrective retry"


async def test_an_answer_that_mentions_a_limit_is_still_an_answer(
    fake_sdk: FakeSdk, fake_document: str, sdk_prompt_module: str
) -> None:
    """Prose about limits is not a refusal — including on the text path."""
    payload = number_turn("200")
    payload["reasoning"] = (
        "the credit limit line resets each year; the revenue row is what was asked"
    )
    fake_sdk.scripts = [Script(payload), Script(json.dumps(payload))]
    preds, _programs, captures = await _run(fake_sdk, ["a?", "b?"])
    assert preds == ["200", "200"]
    assert not any("rate_limited" in c for c in captures)


# ---------------------------------------------------------------------------
# Prompts: the sdk lineage, references, the registry
# ---------------------------------------------------------------------------


def test_load_sdk_round_trips_and_stays_out_of_the_bundle_lineage(
    sdk_prompt_module: str,
) -> None:
    import convfinqa.prompts as prompts_pkg

    assert prompts_pkg.load_sdk(SDK_VERSION) == SDK_PROMPT_TEXT
    assert prompts_pkg.is_sdk_version(SDK_VERSION)
    assert not prompts_pkg.is_sdk_version("v2")
    with pytest.raises(ValueError, match="sdk_vN"):
        prompts_pkg.load_sdk("v2")
    with pytest.raises(AttributeError, match="TRIAGE_PROMPT"):
        prompts_pkg.load(SDK_VERSION)
    assert SDK_VERSION not in prompts_pkg.latest_all()


def test_sdk_prompt_ref_resolves_and_refuses_a_changed_prompt(
    sdk_prompt_module: str,
) -> None:
    from convfinqa.evalloop import prompt_refs

    ref = prompt_refs.sdk_prompt_ref(SDK_VERSION, SDK_PROMPT_TEXT)
    assert ref["kind"] == "sdk_prompt" and ref["version"] == SDK_VERSION
    assert prompt_refs.resolve(ref) == SDK_PROMPT_TEXT

    stale = prompt_refs.sdk_prompt_ref(SDK_VERSION, SDK_PROMPT_TEXT + " edited")
    with pytest.raises(prompt_refs.UnresolvedRefError, match="has changed"):
        prompt_refs.resolve(stale)


def test_the_sdk_lineage_registers_once_and_composes_as_s1(
    sdk_prompt_module: str, registry_tmp: Path
) -> None:
    from convfinqa.tracking import prompt_ledger, registry

    assert prompt_ledger.resolve_sdk(SDK_VERSION)["seq"] == "s?"
    first = prompt_ledger.ensure_sdk(SDK_VERSION, source="evalloop", run_id="r1")
    again = prompt_ledger.ensure_sdk(SDK_VERSION)
    assert first == again
    assert prompt_ledger.sdk_composition_string(first) == "s1"
    doc = registry.load()
    assert doc.sdk_prompts and doc.sdk_prompts[0]["first_seen_in"] == SDK_VERSION
    assert doc.sdk_prompts[0]["parent"] is None
    # The four agents' lineages are untouched by it.
    assert "sdk" not in (doc.agent_prompts or {})

    # register() copes with an sdk version: one lineage, not four.
    entry = registry.register(SDK_VERSION, source="evalloop", run_id="r1")
    assert entry["bundle"]["composition"] == "s1"
    assert entry["bundle"]["v_sdk"].startswith("s1@")
    assert "v_triage" not in entry["bundle"]


def test_sdk_aliases_never_touch_the_pipelines_champion(
    sdk_prompt_module: str, registry_tmp: Path
) -> None:
    from convfinqa.tracking import registry

    champion_before = registry.champion()
    registry.register(SDK_VERSION, source="evalloop")
    doc = registry.set_alias(registry.SDK_CHAMPION, SDK_VERSION)
    assert doc.aliases["sdk_champion"] == SDK_VERSION
    assert registry.champion() == champion_before
    with pytest.raises(ValueError, match="different runtimes"):
        registry.set_alias(registry.CHAMPION, SDK_VERSION)
    with pytest.raises(ValueError, match="different runtimes"):
        registry.set_alias(registry.SDK_CHAMPION, "v2")
    with pytest.raises(ValueError, match="single-session"):
        registry.promote(SDK_VERSION, force=True)
    assert registry.champion() == champion_before


def test_pipeline_sdk_options_are_gated_by_demo_mode(demo_mode: None) -> None:
    from convfinqa.llm import SDK_ALLOWED_TOOLS, DemoModeError, pipeline_sdk_options

    with pytest.raises(DemoModeError):
        pipeline_sdk_options(
            system_prompt="s",
            mcp_server={},
            allowed_tools=list(SDK_ALLOWED_TOOLS),
            output_schema={},
            max_turns=1,
        )


def test_pipeline_sdk_options_refuse_a_widened_tool_set(api_key: None) -> None:
    from convfinqa.llm import SDK_ALLOWED_TOOLS, pipeline_sdk_options

    with pytest.raises(ValueError, match="exactly"):
        pipeline_sdk_options(
            system_prompt="s",
            mcp_server={},
            allowed_tools=[*SDK_ALLOWED_TOOLS, "Bash"],
            output_schema={},
            max_turns=1,
        )


def test_the_backend_imports_without_a_key(demo_mode: None) -> None:
    import importlib

    importlib.import_module("convfinqa.backends.agent_sdk")


# ---------------------------------------------------------------------------
# run_split(runtime="agent_sdk") end to end, with the recorder faked
# ---------------------------------------------------------------------------


async def test_run_split_drives_the_sdk_runtime_and_logs_its_metrics(
    fake_sdk: FakeSdk,
    fake_document: str,
    sdk_prompt_module: str,
    registry_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from convfinqa.data.schemas import ConvExample
    from convfinqa.evalloop import runner
    from convfinqa.tracking import mlflow_log, registry, tracing

    example = ConvExample(
        report_id=REPORT,
        questions=["what was revenue in 2020?", "and the change from 2019?"],
        gold_answers=["200", "150"],
        gold_programs=["200", "subtract(200, 50)"],
        gold_turn_types=["number", "program"],
        gold_conv_types=["Type I", "Type I"],
    )
    monkeypatch.setattr(runner, "split_report_ids", lambda *a, **k: [REPORT])
    monkeypatch.setattr(runner, "examples_for", lambda ids: [example])
    monkeypatch.setattr(runner, "PREDICTIONS_DIR", tmp_path / "preds")
    monkeypatch.setattr(tracing, "enable", lambda: False)

    logged: dict[str, Any] = {"params": {}, "tags": {}, "metrics": {}}

    class _Rec:
        run_id = "run-sdk-1"

        def metrics(self, values: dict[str, float]) -> None:
            logged["metrics"].update(values)

        def artifact(self, path: Any) -> None:
            logged["artifact"] = str(path)

        def dict_artifact(self, name: str, payload: Any) -> None:
            return None

        def tag(self, key: str, value: str) -> None:
            logged["tags"][key] = value

        def param(self, key: str, value: Any) -> None:
            logged["params"][key] = value

    @contextlib.contextmanager
    def fake_run(name: str, **kw: Any) -> Any:
        logged["name"] = name
        logged["params"] = kw.get("params") or {}
        logged["tags"] = kw.get("tags") or {}
        yield _Rec()

    monkeypatch.setattr(mlflow_log, "run", fake_run)

    fake_sdk.scripts = [
        Script(number_turn("200")),
        Script(program_turn("150.0"), tool_calls=[("subtract", 200, 50)]),
    ]
    summary = await runner.run_split("train", SDK_VERSION, runtime="agent_sdk")

    assert summary["runtime"] == "agent_sdk"
    assert summary["accuracy"] == 1.0
    assert logged["name"].startswith(f"sdk-evalloop-train1-{SDK_VERSION}·s1-")
    assert logged["params"]["runtime"] == "agent_sdk"
    assert logged["params"]["sdk_model"] == "claude-sonnet-5"
    assert logged["params"]["billing"] == "api"
    assert logged["tags"]["runtime"] == "agent_sdk"
    m = logged["metrics"]
    assert m["sdk_turns_mean"] == 1.5
    assert m["sdk_tool_calls_mean"] == 0.5
    assert m["sdk_stage_skips"] == 0.0
    assert m["sdk_inline_arithmetic"] == 0.0
    assert m["sdk_cost_usd"] == 0.02
    assert "acc_calculator_exec" in m
    # A pass that answered everything says so, and is not tagged incomplete.
    assert m["complete"] == 1.0 and m["n_unscored"] == 0.0
    assert summary["complete"] is True and summary["n_unscored"] == 0
    assert "incomplete" not in logged["tags"]
    assert "unscored_rows" not in logged["params"]

    df = pd.read_csv(summary["csv"])
    assert list(df.columns) == [
        *runner.COLUMNS,
        *__import__(
            "convfinqa.evalloop.stage_scores", fromlist=["ROW_COLUMNS"]
        ).ROW_COLUMNS,
    ]
    assert df["model_version_id"].tolist() == [SDK_VERSION] * 2
    entry = registry.find_version(registry.load(), SDK_VERSION)
    assert entry is not None and "run-sdk-1" in entry["runs"]


async def test_run_split_refuses_a_mismatched_version_and_runtime(
    sdk_prompt_module: str,
) -> None:
    from convfinqa.evalloop import runner

    with pytest.raises(ValueError, match="does not belong"):
        await runner.run_split("train", "v2", runtime="agent_sdk")
    with pytest.raises(ValueError, match="does not belong"):
        await runner.run_split("train", SDK_VERSION, runtime="pipeline")
    with pytest.raises(ValueError, match="unknown runtime"):
        await runner.run_split("train", "v2", runtime="dspy")
