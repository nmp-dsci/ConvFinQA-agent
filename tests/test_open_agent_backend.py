"""The open runtime (s25) against a fake DeepSeek endpoint.

Every test drives `open_agent.run_conversation` for real — Pydantic AI, the
DeepSeek profile, the OpenAI client — and swaps only the HTTP transport, so the
assertions read the JSON DeepSeek would actually receive. That is the contract
"lift and shift" makes: the system prompt, the turn messages and the six tools
leave this process exactly as the Claude runtime sends them, and DeepSeek's
thinking protocol (reasoning sent back, no forced tool) is honoured on the wire.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from convfinqa.backends import agent_sdk, open_agent
from convfinqa.backends.agent_sdk import SdkRateLimitError

REPORT = "Single_FAKE/2020/page_1.pdf-1"
PROMPT = "You answer questions about one financial report. Use the calculator."
CALCULATORS = {f"mcp__cfq__{n}" for n in agent_sdk._TOOL_FNS}


def number_turn(answer: str) -> dict[str, Any]:
    return {
        "turn_type": "number",
        "conv_type": "Type I",
        "answer": answer,
        "reasoning": "read it off the table",
    }


def program_turn(answer: str = "150.0") -> dict[str, Any]:
    return {
        "turn_type": "program",
        "conv_type": "Type I",
        "sub_questions": ["revenue in 2020", "revenue in 2019"],
        "program": "subtract(A, B)",
        "retrieved": [
            {"question": "revenue in 2020", "answer": "200", "source": "table"},
            {"question": "revenue in 2019", "answer": "50", "source": "table"},
        ],
        "answer": answer,
        "reasoning": "triage → preprocess → retrieve → calculate",
    }


def reply(
    *,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    content: str | None = None,
    reasoning: str = "let me think",
    status: int = 200,
) -> httpx.Response:
    """One chat completion in DeepSeek's shape, reasoning included."""
    if status != 200:
        return httpx.Response(status, json={"error": {"message": f"status {status}"}})
    tool_calls = [
        {
            "id": f"call_{i}_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls or [])
    ]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1_790_000_000,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_cache_hit_tokens": 800,
                "prompt_cache_miss_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 60},
            },
        },
    )


class FakeDeepSeek:
    """A scripted DeepSeek: pops one reply per request and keeps every body."""

    def __init__(self, replies: list[httpx.Response]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return self.replies.pop(0)


@pytest.fixture
def deepseek(monkeypatch: pytest.MonkeyPatch) -> FakeDeepSeek:
    import convfinqa.llm as llm
    from convfinqa.data import loader
    from convfinqa.data.schemas import Document

    fake = FakeDeepSeek([])
    monkeypatch.setattr(
        llm,
        "_open_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)),
    )
    llm.reset_provider()
    monkeypatch.setitem(
        loader._DOCS,
        REPORT,
        Document(
            pre_text="revenue was 200 in 2020 and 50 in 2019.",
            post_text="",
            table={"revenue": {"2020": 200, "2019": 50}},
        ),
    )
    return fake


def run(
    questions: list[str], **kw: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}
    preds, programs = asyncio.run(
        open_agent.run_conversation(
            REPORT,
            questions,
            system_prompt=PROMPT,
            model="deepseek-flash",
            effort="high",
            captures=captures,
            trajectories=trajectories,
            **kw,
        )
    )
    return preds, programs, captures, trajectories


def two_turn_script() -> list[httpx.Response]:
    return [
        reply(calls=[("mcp__cfq__subtract", {"a": 200, "b": 50})], reasoning="r1"),
        reply(calls=[("StructuredOutput", program_turn())], reasoning="r2"),
        reply(calls=[("StructuredOutput", number_turn("7"))], reasoning="r3"),
    ]


def test_the_model_receives_the_champions_contract_verbatim(
    deepseek: FakeDeepSeek,
) -> None:
    from convfinqa.data.loader import _DOCS

    deepseek.replies = two_turn_script()
    preds, programs, _caps, _traj = run(["the change?", "and the count?"])
    assert preds == ["150.0", "7"] and programs == ["subtract(A, B)", ""]

    first = deepseek.bodies[0]
    assert first["messages"][0] == {"role": "system", "content": PROMPT}
    assert first["messages"][1]["content"] == agent_sdk._first_message(
        REPORT, _DOCS[REPORT], "the change?"
    )
    tools = {t["function"]["name"]: t["function"] for t in first["tools"]}
    assert set(tools) == CALCULATORS | {open_agent.OUTPUT_TOOL_NAME}
    for name in agent_sdk._TOOL_FNS:
        sent = tools[f"mcp__cfq__{name}"]
        assert sent["description"] == open_agent.tool_description(name)
        # The harness's one addition: OpenAI-style schemas close the object.
        params = dict(sent["parameters"])
        assert params.pop("additionalProperties") is False
        assert params == agent_sdk._TOOL_SCHEMA
    later = deepseek.bodies[2]["messages"]
    assert later[-1] == {
        "role": "user",
        "content": agent_sdk._later_message("and the count?"),
    }


def test_deepseek_thinking_protocol_is_honoured_on_the_wire(
    deepseek: FakeDeepSeek,
) -> None:
    """Reasoning goes back on every earlier turn; no tool is ever forced."""
    deepseek.replies = two_turn_script()
    run(["the change?", "and the count?"])
    for body in deepseek.bodies:
        assert body.get("reasoning_effort") == "high"
        assert body.get("tool_choice") in (None, "auto")
    sent_back = [
        m.get("reasoning_content")
        for m in deepseek.bodies[2]["messages"]
        if m["role"] == "assistant"
    ]
    assert sent_back == ["r1", "r2"]


def test_the_capture_is_the_sdk_runtimes_shape(deepseek: FakeDeepSeek) -> None:
    deepseek.replies = two_turn_script()
    _preds, _progs, caps, trajectories = run(["the change?", "and the count?"])
    program = caps[0]
    assert program["calculator"]["trajectory"] == [
        {"event": "tool_call", "tool": "subtract", "args": {"a": 200, "b": 50}},
        {"event": "tool_return", "tool": "subtract", "result": "150.0"},
    ]
    assert program["sdk"]["inline_arithmetic"] is False
    assert program["sdk"]["served_models"] == ["deepseek-flash"]
    assert program["triage"]["sdk_attempts"] == 1
    assert program["triage"]["thinking"] == ["r1", "r2"]
    # Two calls, each 1,000 prompt tokens of which 800 were cache hits.
    assert program["sdk"]["cache_read_input_tokens"] == 1600
    assert program["sdk"]["input_tokens"] == 400
    assert program["sdk"]["cost_usd"] > 0
    assert caps[1]["preprocess"] is None and caps[1]["calculator"] is None
    assert list(trajectories) == [REPORT] and len(trajectories[REPORT]) >= 5


def test_a_rejected_reply_is_corrected_with_the_sdk_text(
    deepseek: FakeDeepSeek,
) -> None:
    """Two invalid replies exhaust the harness's retry; ours asks again."""
    bad = {"turn_type": "number", "conv_type": "Type I"}  # no answer
    deepseek.replies = [
        reply(calls=[("StructuredOutput", bad)]),
        reply(calls=[("StructuredOutput", bad)]),
        reply(calls=[("StructuredOutput", number_turn("200"))]),
    ]
    preds, _progs, caps, _traj = run(["the revenue?"])
    assert preds == ["200"]
    retry = deepseek.bodies[2]["messages"]
    # The rejected attempt is not in the history; the question and the
    # correction are, in one user message.
    assert [m["role"] for m in retry] == ["system", "user"]
    assert (
        "Your last reply did not match the required output schema"
        in retry[-1]["content"]
    )
    assert caps[0]["triage"]["sdk_attempts"] == 2


def test_a_failed_turn_stays_in_the_history_of_the_next(
    deepseek: FakeDeepSeek,
) -> None:
    bad = {"turn_type": "number", "conv_type": "Type I"}
    deepseek.replies = [
        *[reply(calls=[("StructuredOutput", bad)]) for _ in range(4)],
        reply(calls=[("StructuredOutput", number_turn("50"))]),
    ]
    preds, _progs, caps, _traj = run(["the revenue?", "and in 2019?"])
    assert preds == ["", "50"]
    assert "error" in caps[0] and not caps[0].get("answer")
    sent = deepseek.bodies[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "revenue was 200 in 2020" in sent[1]["content"]
    assert "the revenue?" in sent[1]["content"]
    assert "and in 2019?" in sent[3]["content"]


@pytest.mark.parametrize("status", [402, 429, 503])
def test_a_provider_that_will_not_answer_is_unscored_not_wrong(
    deepseek: FakeDeepSeek, status: int
) -> None:
    # The OpenAI client retries a 429 or 5xx itself, so the refusal repeats.
    deepseek.replies = [
        reply(calls=[("StructuredOutput", number_turn("200"))]),
        *[reply(status=status) for _ in range(5)],
    ]
    with pytest.raises(SdkRateLimitError) as caught:
        run(["the revenue?", "and in 2019?"])
    assert caught.value.turn_index == 1 and caught.value.preds == ["200"]


def test_the_token_ceiling_ends_the_walk(
    deepseek: FakeDeepSeek, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "sdk_total_tokens_limit", 100)
    deepseek.replies = [reply(calls=[("StructuredOutput", number_turn("200"))])]
    preds, _progs, caps, _traj = run(["the revenue?", "and in 2019?"])
    assert preds == ["200", ""]
    assert caps[1]["error"].startswith("token budget exhausted")
    assert len(deepseek.bodies) == 1


def test_unknown_models_and_efforts_are_refused() -> None:
    import convfinqa.llm as llm

    with pytest.raises(ValueError, match="unknown open model"):
        llm.open_agent_model("deepseek-chat", "high")
    with pytest.raises(ValueError, match="unknown open effort"):
        llm.open_agent_model("deepseek-flash", "xhigh")


def test_open_runs_are_named_and_recognised_as_their_own_runtime() -> None:
    import pandas as pd

    from convfinqa.evalloop.runner import (
        prior_runtime,
        prior_sdk_effort,
        prior_sdk_model,
    )
    from convfinqa.llm import open_model_slug

    name = "open-evalloop-train9-sdk_v1·s1-pyd-deepseek-flash-high-20261025_120000.csv"
    assert open_model_slug("pydantic_ai", "deepseek-flash") == "pyd-deepseek-flash"
    assert (
        prior_runtime(pd.DataFrame({"model_version_id": ["sdk_v1"]}), name)
        == "open_agent"
    )
    assert prior_sdk_model(name) == "pyd-deepseek-flash"
    assert prior_sdk_effort(name) == "high"


def test_open_flags_are_refused_on_another_runtime() -> None:
    from convfinqa.evalloop.runner import run_split

    with pytest.raises(ValueError, match="open_agent only"):
        asyncio.run(
            run_split(
                "test", "sdk_v1", runtime="agent_sdk", open_model="deepseek-flash"
            )
        )


# --- s29: the checked harness (H1–H3) and the student's sampling -----------------


def checked_run(questions: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    preds, _progs, caps, _traj = run(questions, harness="pydantic_ai_checked")
    return preds, caps


def sent_text(body: dict[str, Any]) -> str:
    return json.dumps(body["messages"])


def test_h1_an_extra_calculator_argument_is_a_tool_error_not_a_crash(
    deepseek: FakeDeepSeek,
) -> None:
    deepseek.replies = [
        reply(calls=[("mcp__cfq__add", {"a": 1, "b": 2, "c": 3})]),
        reply(calls=[("mcp__cfq__add", {"a": 1, "b": 2})]),
        reply(calls=[("mcp__cfq__add", {"a": 3, "b": 3})]),
        reply(
            calls=[
                (
                    "StructuredOutput",
                    {**program_turn("6.0"), "program": "add(add(A, B), C)"},
                )
            ]
        ),
    ]
    preds, caps = checked_run(["the total?"])
    assert preds == ["6.0"] and "error" not in caps[0]
    assert "error: add takes exactly two numbers" in sent_text(deepseek.bodies[1])


def test_the_plain_harness_still_fails_a_turn_on_an_extra_argument(
    deepseek: FakeDeepSeek,
) -> None:
    """The plain harness is llm_react_v1's and the teacher's: unchanged."""
    deepseek.replies = [reply(calls=[("mcp__cfq__add", {"a": 1, "b": 2, "c": 3})])]
    preds, _progs, caps, _traj = run(["the total?"])
    assert preds == [""] and "TypeError" in caps[0]["error"]


def test_h2_an_answer_that_is_no_calculator_result_is_sent_back_once(
    deepseek: FakeDeepSeek,
) -> None:
    deepseek.replies = [
        reply(calls=[("mcp__cfq__subtract", {"a": 200, "b": 50})]),
        reply(calls=[("StructuredOutput", program_turn("15.0"))]),
        reply(calls=[("StructuredOutput", program_turn("150.0"))]),
    ]
    preds, caps = checked_run(["the change?"])
    assert preds == ["150.0"]
    assert "is not any calculator result" in sent_text(deepseek.bodies[2])
    assert caps[0]["sdk"]["harness_retry"] == "H2"


def test_a_check_sends_a_question_back_once_then_accepts_the_reply(
    deepseek: FakeDeepSeek,
) -> None:
    deepseek.replies = [
        reply(calls=[("mcp__cfq__subtract", {"a": 200, "b": 50})]),
        reply(calls=[("StructuredOutput", program_turn("15.0"))]),
        reply(calls=[("StructuredOutput", program_turn("15.0"))]),
        reply(calls=[("StructuredOutput", number_turn("7"))]),
    ]
    preds, caps = checked_run(["the change?", "and the count?"])
    # never an answer lost to a check; and the next question starts afresh
    assert preds == ["15.0", "7"]
    assert caps[0]["sdk"]["harness_retry"] == "H2"
    assert caps[1]["sdk"]["harness_retry"] == ""


def test_h3_operands_out_of_the_programs_order_are_sent_back(
    deepseek: FakeDeepSeek,
) -> None:
    deepseek.replies = [
        reply(calls=[("mcp__cfq__subtract", {"a": 50, "b": 200})]),
        reply(calls=[("StructuredOutput", program_turn("-150.0"))]),
        reply(calls=[("mcp__cfq__subtract", {"a": 200, "b": 50})]),
        reply(calls=[("StructuredOutput", program_turn("150.0"))]),
    ]
    preds, caps = checked_run(["the change?"])
    assert preds == ["150.0"]
    assert "in the program's order" in sent_text(deepseek.bodies[2])
    assert caps[0]["sdk"]["harness_retry"] == "H3"


def test_the_plain_harness_sends_nothing_back(deepseek: FakeDeepSeek) -> None:
    deepseek.replies = [
        reply(calls=[("mcp__cfq__subtract", {"a": 200, "b": 50})]),
        reply(calls=[("StructuredOutput", program_turn("15.0"))]),
    ]
    preds, _progs, caps, _traj = run(["the change?"])
    assert preds == ["15.0"] and len(deepseek.bodies) == 2
    assert "harness_retry" not in caps[0]["sdk"]


def test_a_checked_or_resampled_student_pass_is_named_for_it() -> None:
    from convfinqa.evalloop.runner import prior_sdk_effort, prior_sdk_model
    from convfinqa.llm import open_model_slug

    model = "student-qwen3-4b-sft1-e2"
    assert open_model_slug("pydantic_ai", model, 0.7) == f"pyd-{model}"
    slug = open_model_slug("pydantic_ai_checked", model, 0.2)
    assert slug == f"pydchk-{model}-t0p2"
    # a student's effort is "none", which --resume-from used to read as model
    name = f"open-evalloop-train100-sdk_v1·s1-{slug}-none-20261026_010000.csv"
    assert prior_sdk_model(name) == slug and prior_sdk_effort(name) == "none"


def test_a_temperature_is_a_students_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    import convfinqa.llm as llm
    from convfinqa.config import settings

    with pytest.raises(ValueError, match="applies to a student"):
        llm.open_agent_model("deepseek-flash", "high", temperature=0.2)
    monkeypatch.setattr(settings, "student_url", "http://student.invalid")
    llm.reset_provider()
    model = llm.open_agent_model("student-qwen3-4b-sft1-e2", "none", temperature=0.2)
    assert model.settings is not None and model.settings["temperature"] == 0.2
    llm.reset_provider()


def test_the_server_serves_every_student_the_runner_names() -> None:
    from convfinqa.llm import STUDENT_MODELS
    from convfinqa.sft.cloud import modal_app

    served = {modal_app.BASE_NAME, *modal_app.ADAPTERS}
    assert served == set(STUDENT_MODELS)
    command = modal_app.vllm_command(modal_app.ADAPTERS)
    assert command[command.index("--max-loras") + 1] == str(len(modal_app.ADAPTERS))
