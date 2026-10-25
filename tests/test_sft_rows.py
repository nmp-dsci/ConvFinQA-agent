"""The s28 SFT builder: recorded llm_react_v1 histories → training rows.

The histories here have the shape `ModelMessagesTypeAdapter.dump_python(…,
mode="json")` writes to a pass's trajectories artifact, so each test drives
the real converter and gates on what a pass actually records.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from convfinqa.sft import build, rows
from convfinqa.sft.prompt import REASONING_OPEN, student_prompt

ANSWER = {
    "turn_type": "program",
    "conv_type": "Type I",
    "sub_questions": ["2016 taxes", "2015 taxes"],
    "program": "add(A, B)",
    "retrieved": [
        {"question": "2016 taxes", "answer": "10.4", "source": "post_text"},
        {"question": "2015 taxes", "answer": "80.6", "source": "post_text"},
    ],
    "answer": "91.0",
    "reasoning": "Triage: program. Retrieve both years, add.",
}


def user(text: str) -> dict[str, Any]:
    return {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": text}]}


def response(
    *, thinking: str = "", text: str = "", calls: list[tuple[str, dict[str, Any]]] = ()
) -> dict[str, Any]:  # type: ignore[assignment]
    parts: list[dict[str, Any]] = []
    if thinking:
        parts.append(
            {"part_kind": "thinking", "content": thinking, "id": "reasoning_content"}
        )
    if text:
        parts.append({"part_kind": "text", "content": text})
    for i, (name, args) in enumerate(calls):
        parts.append(
            {
                "part_kind": "tool-call",
                "tool_name": name,
                "args": json.dumps(args),
                "tool_call_id": f"c{i}{name}",
            }
        )
    return {"kind": "response", "parts": parts}


def returned(name: str, content: str, call: str) -> dict[str, Any]:
    return {
        "kind": "request",
        "parts": [
            {
                "part_kind": "tool-return",
                "tool_name": name,
                "content": content,
                "tool_call_id": call,
            }
        ],
    }


def refusal(tool_name: str | None = None) -> dict[str, Any]:
    return {
        "kind": "request",
        "parts": [
            {
                "part_kind": "retry-feedback",
                "tool_name": tool_name,
                "content": rows.REFUSAL,
            }
        ],
    }


def clean_turn() -> list[dict[str, Any]]:
    return [
        response(
            thinking="2016 is 10.4 and 2015 is 80.6; add them.",
            calls=[("mcp__cfq__add", {"a": 10.4, "b": 80.6})],
        ),
        returned("mcp__cfq__add", "91.0", "c0mcp__cfq__add"),
        response(calls=[("StructuredOutput", ANSWER)]),
        returned("StructuredOutput", "Final result processed.", "c0StructuredOutput"),
    ]


def detour_turn(resent: dict[str, Any]) -> list[dict[str, Any]]:
    """DeepSeek writes the answer as text, is refused, and re-sends it as a call."""
    return [
        response(
            thinking="2016 is 10.4 and 2015 is 80.6; add them.",
            calls=[("mcp__cfq__add", {"a": 10.4, "b": 80.6})],
        ),
        returned("mcp__cfq__add", "91.0", "c0mcp__cfq__add"),
        response(text=json.dumps(ANSWER, indent=2)),
        refusal(),
        response(
            thinking="The user asks for a tool call; I should call StructuredOutput.",
            calls=[("StructuredOutput", resent)],
        ),
        returned("StructuredOutput", "Final result processed.", "c0StructuredOutput"),
    ]


def test_a_clean_turn_keeps_the_reasoning_before_its_call() -> None:
    out, detours = rows.turn_messages(clean_turn())
    assert detours == 0
    assert [m["role"] for m in out] == ["assistant", "tool", "assistant", "tool"]
    assert (
        out[0]["content"]
        == f"{REASONING_OPEN}\n2016 is 10.4 and 2015 is 80.6; add them.\n</reasoning>"
    )
    assert out[0]["tool_calls"][0]["function"] == {
        "name": "mcp__cfq__add",
        "arguments": '{"a": 10.4, "b": 80.6}',
    }
    assert out[2]["content"] == ""


def test_a_text_answer_detour_collapses_to_the_call() -> None:
    reworded = {**ANSWER, "reasoning": "Re-sent: program, add the two years."}
    out, detours = rows.turn_messages(detour_turn(reworded))
    assert detours == 1
    assert [m["role"] for m in out] == ["assistant", "tool", "assistant", "tool"]
    final = out[2]
    # the refused text, the refusal and the reasoning about it are all gone
    assert final["content"] == ""
    assert "The user asks" not in json.dumps(out)
    assert json.loads(final["tool_calls"][0]["function"]["arguments"]) == reworded


def test_a_resent_answer_that_decides_differently_is_dropped() -> None:
    with pytest.raises(rows.RowRejectedError) as exc:
        rows.turn_messages(detour_turn({**ANSWER, "answer": "70.2"}))
    assert exc.value.rule == "N1"


def test_a_refused_tool_call_is_dropped() -> None:
    steps = [response(calls=[("mcp__cfq__add", {"a": "x"})]), refusal("mcp__cfq__add")]
    with pytest.raises(rows.RowRejectedError) as exc:
        rows.turn_messages(steps)
    assert exc.value.rule == "F5"


def test_a_row_carries_the_students_own_system_message_and_tools() -> None:
    row, meta = rows.to_row(
        "Single_X/2016/page_1.pdf-1",
        [user("doc + q0"), *clean_turn(), user("q1"), *clean_turn()],
    )
    assert meta == {
        "report_id": "Single_X/2016/page_1.pdf-1",
        "n_turns": 2,
        "n_detours": 0,
    }
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][0]["content"].strip() == student_prompt("sdk_v1").strip()
    names = [t["function"]["name"] for t in row["tools"]]
    assert names[-1] == "StructuredOutput" and {
        "mcp__cfq__add",
        "mcp__cfq__divide",
    } <= set(names)
    assert [m["role"] for m in row["messages"]].count("user") == 2


def _frame(correct: list[bool], **cols: list[Any]) -> pd.DataFrame:
    n = len(correct)
    data: dict[str, Any] = {
        "turn_index": list(range(n)),
        "correct": correct,
        "error": [None] * n,
    }
    data.update(cols)
    return pd.DataFrame(data)


def test_the_gates_charge_the_first_rule_a_conversation_fails() -> None:
    rid = "Single_X/2016/page_1.pdf-1"
    history = [user("doc + q0"), *clean_turn(), user("q1"), *clean_turn()]
    ok = dict(allowed={rid}, n_questions=2)
    build._check(rid, _frame([True, True]), history, **ok)  # passes every gate
    cases = [
        (dict(allowed=set(), n_questions=2), _frame([True, True]), "F2"),
        (ok, _frame([True, False]), "F1"),
        (dict(allowed={rid}, n_questions=3), _frame([True, True]), "F3"),
        (
            ok,
            _frame(
                [True, True],
                triage_turn_type_ok=[True, False],
                gold_program=["add(1, 2)"] * 2,
            ),
            "F4",
        ),
    ]
    for kwargs, frame, rule in cases:
        with pytest.raises(build._GateError) as exc:
            build._check(rid, frame, history, **kwargs)
        assert exc.value.rule == rule


def test_a_corrected_reask_fails_the_first_attempt_gate() -> None:
    rid = "Single_X/2016/page_1.pdf-1"
    history = [
        user("doc + q0"),
        *clean_turn(),
        user(f"q1\n\n{build.CORRECTION_MARK}: …"),
        *clean_turn(),
    ]
    with pytest.raises(build._GateError) as exc:
        build._check(rid, _frame([True, True]), history, allowed={rid}, n_questions=2)
    assert exc.value.rule == "F6"


def test_a_test_pass_is_never_training_data(tmp_path: Any) -> None:
    csv = (
        tmp_path
        / "open-evalloop-test100-sdk_v1·s1-pyd-deepseek-flash-high-20261025_120625.csv"
    )
    _frame([True]).assign(split="test", report_id="x").to_csv(csv, index=False)
    with pytest.raises(ValueError, match="training data"):
        build.build([csv], version="pytest")


def test_an_effort_must_fit_its_model() -> None:
    from convfinqa.llm import check_open_agent

    check_open_agent("student-qwen3-4b", "none")
    check_open_agent("deepseek-flash", "high")
    for model, effort in (("student-qwen3-4b", "high"), ("deepseek-flash", "none")):
        with pytest.raises(ValueError, match="does not fit"):
            check_open_agent(model, effort)


def test_the_student_is_never_forced_into_a_tool_call() -> None:
    """vLLM honours a forced call with guided decoding, which leaves no room for
    the `<reasoning>` text the student writes before each call."""
    import asyncio

    import httpx

    from convfinqa.backends import open_agent
    from convfinqa.llm import student_wire_model

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        call = {"name": "StructuredOutput", "arguments": json.dumps(ANSWER)}
        message = {
            "role": "assistant",
            "content": "<reasoning>\nadd them\n</reasoning>",
            "tool_calls": [{"id": "c0", "type": "function", "function": call}],
        }
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "student-qwen3-4b",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls"}
                ],
            },
        )

    agent = open_agent.build_agent(
        student_prompt(), student_wire_model(httpx.MockTransport(handler)), []
    )
    result = asyncio.run(agent.run("q0"))
    assert result.output.answer == "91.0"
    assert bodies[0].get("tool_choice", "auto") == "auto"
    assert bodies[0]["temperature"] == 0.7 and bodies[0]["top_k"] == 20


def test_a_literal_gold_program_does_not_fail_the_turn_type() -> None:
    """Gold labels `36%` a program turn; answering it as a number is right."""
    rid = "Single_X/2016/page_1.pdf-1"
    history = [user("doc + q0"), *clean_turn(), user("q1"), *clean_turn()]
    frame = _frame(
        [True, True],
        triage_turn_type_ok=[True, False],
        gold_program=["add(1, 2)", "36%"],
    )
    build._check(rid, frame, history, allowed={rid}, n_questions=2)


def test_a_program_answer_must_be_the_last_calculator_result() -> None:
    """Otherwise the model did the arithmetic itself, whatever the tools said."""
    rid = "Single_X/2016/page_1.pdf-1"
    in_head = [
        response(
            thinking="add them", calls=[("mcp__cfq__add", {"a": 10.4, "b": 80.6})]
        ),
        returned("mcp__cfq__add", "91.0", "c0mcp__cfq__add"),
        response(calls=[("StructuredOutput", {**ANSWER, "answer": "92.0"})]),
        returned("StructuredOutput", "Final result processed.", "c0StructuredOutput"),
    ]
    no_call = [
        response(calls=[("StructuredOutput", ANSWER)]),
        returned("StructuredOutput", "Final result processed.", "c0StructuredOutput"),
    ]
    for turn in (in_head, no_call):
        with pytest.raises(build._GateError) as exc:
            build._check(
                rid,
                _frame([True]),
                [user("doc + q0"), *turn],
                allowed={rid},
                n_questions=1,
            )
        assert exc.value.rule == "F4"


@pytest.mark.parametrize(
    ("answer", "result", "same"),
    [
        ("-7.5%", "-7.465929290934229", True),  # a percentage rounded to one decimal
        ("14.1%", "0.1413638598681775", True),  # a fraction written as a percentage
        ("91", "91.0", True),
        ("0.41", "0.4118", True),
        ("no", "False", True),  # `greater`'s result, in words
        ("0.42", "0.4118", False),  # wrongly rounded
        ("374", "22203.0", False),  # not the last result at all
    ],
)
def test_an_answer_is_its_result_at_its_own_precision(
    answer: str, result: str, same: bool
) -> None:
    assert build.answer_is_result(answer, result) is same


# --- s29: the dropped tool reply (N0), prefix rows (N1), the aimed draw (N2) -----


def merged_history() -> list[dict[str, Any]]:
    """Two turns as Pydantic AI often stores them: q0's StructuredOutput return
    shares one request with q1's prompt."""
    q0 = clean_turn()[:-1]  # without its own closing return
    closing_and_q1 = {
        "kind": "request",
        "parts": [
            {
                "part_kind": "tool-return",
                "tool_name": "StructuredOutput",
                "content": "Final result processed.",
                "tool_call_id": "c0StructuredOutput",
            },
            {"part_kind": "user-prompt", "content": "q1"},
        ],
    }
    return [user("doc + q0"), *q0, closing_and_q1, *clean_turn()]


def test_a_tool_return_folded_into_the_next_question_closes_its_own_turn() -> None:
    from convfinqa.sft import trajectories as tj

    turns = tj.turns(merged_history())
    assert [t["prompt"] for t in turns] == ["doc + q0", "q1"]
    closing = turns[0]["steps"][-1]["parts"]
    assert [p["part_kind"] for p in closing] == ["tool-return"]
    row, _ = rows.to_row("Single_X/2016/page_1.pdf-1", merged_history())
    roles = [m["role"] for m in row["messages"]]
    # every answer call is followed by its tool reply, as the student is served
    one_turn = ["user", "assistant", "tool", "assistant", "tool"]
    assert roles == ["system", *one_turn, *one_turn]


def test_a_prefix_keeps_its_last_tool_reply_and_nothing_after() -> None:
    from convfinqa.sft import trajectories as tj

    cut = tj.prefix(merged_history(), 1)
    assert len(tj.turns(cut)) == 1
    assert "q1" not in json.dumps(cut)
    assert cut[-1]["parts"][0]["content"] == "Final result processed."


def test_an_f1_drop_keeps_its_turns_before_the_first_wrong_one(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from convfinqa.sft import splits, tokens
    from convfinqa.sft import trajectories as tj

    late, early = "Single_X/2016/page_1.pdf-1", "Single_Y/2016/page_2.pdf-1"
    run = "open-evalloop-train2-sdk_v1·s1-pyd-deepseek-flash-high-20261026_000000"
    traj = tmp_path / f"{run}.jsonl.gz"
    with gzip.open(traj, "wt") as f:
        for rid in (late, early):
            f.write(json.dumps({"report_id": rid, "messages": merged_history()}) + "\n")
    csv = tmp_path / f"{run}.csv"
    pd.DataFrame(
        {
            "report_id": [late, late, early, early],
            "turn_index": [0, 1, 0, 1],
            "correct": [True, False, False, True],
            "error": [None] * 4,
            "split": ["train"] * 4,
        }
    ).to_csv(csv, index=False)
    monkeypatch.setattr(tj, "path_for", lambda name: traj)
    monkeypatch.setattr(
        splits,
        "load",
        lambda name="sft_v1": {"splits": {"train": [late, early], "val": []}},
    )
    monkeypatch.setattr(splits, "excluded_pages", lambda: set())
    monkeypatch.setattr(build, "_questions", lambda: {late: 2, early: 2})
    monkeypatch.setattr(build, "SFT_ROOT", tmp_path / "sft")
    monkeypatch.setattr(tokens, "count", lambda row: 100)
    monkeypatch.setattr(tokens, "count_text", lambda text: 10)

    without = build.build([csv], version="pytest")
    assert without["n_rows"] == 0
    stats = build.build([csv], version="pytest", prefix_rows=True)
    assert (stats["n_rows"], stats["n_prefix_rows"], stats["n_turns"]) == (1, 1, 1)
    out = tmp_path / "sft" / "pytest"
    (row,) = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert row["meta"]["prefix"] is True and row["meta"]["report_id"] == late
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    drops = {
        d["report_id"]: d
        for d in map(json.loads, (out / "drops.jsonl").read_text().splitlines())
    }
    assert drops[late]["rule"] == "F1" and drops[late]["prefix_turns"] == 1
    assert "prefix_turns" not in drops[early]  # wrong at q0: nothing precedes it


def _gold(rows_: list[tuple[str, int, str]]) -> pd.DataFrame:
    from convfinqa.sft import aim

    qa = pd.DataFrame(rows_, columns=["report_id", "q_order", "turn_program"])
    qa["bucket"] = qa["turn_program"].map(aim.bucket)
    qa["late"] = qa["q_order"] >= aim.LATE_FROM
    return qa


def test_the_aimed_draw_ranks_conversations_by_expected_misses(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.sft import aim

    assert aim.bucket("subtract(1, 2), divide(#0, 2)") == "divide"
    assert aim.bucket("206588") == "lookup"
    gold = _gold(
        [
            # val: lookups right, divides wrong, a late lookup wrong
            *[("V", i, "1") for i in range(4)],
            ("V", 4, "1"),
            ("W", 0, "divide(1, 2)"),
            ("W", 1, "divide(1, 2)"),
            # candidates
            ("long", 0, "1"),
            ("long", 1, "1"),
            ("long", 2, "1"),
            ("long", 3, "1"),
            ("long", 4, "divide(1, 2)"),
            ("div", 0, "divide(1, 2)"),
            ("look", 0, "1"),
            ("look2", 0, "1"),
        ]
    )
    monkeypatch.setattr(aim, "gold_turns", lambda: gold)
    val = tmp_path / "val.csv"
    pd.DataFrame(
        {
            "report_id": ["V"] * 5 + ["W"] * 2,
            "turn_index": [0, 1, 2, 3, 4, 0, 1],
            "correct": [True, True, True, True, False, False, False],
            "split": ["train"] * 7,
        }
    ).to_csv(val, index=False)
    rates = aim.miss_rates(val)
    assert rates["miss_by_bucket"]["divide"] > rates["miss_by_bucket"]["lookup"]
    assert rates["late_ratio"] > 1
    ranked = aim.top({"long", "div", "look", "look2"}, rates, 4)
    assert [r for r, _ in ranked] == ["long", "div", "look", "look2"]  # ties by id

    pd.read_csv(val).assign(split="test").to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match="train-side"):
        aim.miss_rates(tmp_path / "t.csv")
    pd.read_csv(val).assign(unscored=True).to_csv(tmp_path / "u.csv", index=False)
    with pytest.raises(ValueError, match="incomplete"):
        aim.miss_rates(tmp_path / "u.csv")


def test_a_call_sent_back_as_a_retried_tool_return_fails_f5() -> None:
    """Pydantic AI 2.x stores a refused call as a tool return, outcome "retried"."""
    sent_back = {
        "kind": "request",
        "parts": [
            {
                "part_kind": "tool-return",
                "tool_name": "StructuredOutput",
                "content": "1 validation error: answer is required",
                "tool_call_id": "c0StructuredOutput",
                "outcome": "retried",
            }
        ],
    }
    steps = [response(calls=[("StructuredOutput", {"turn_type": "number"})]), sent_back]
    with pytest.raises(rows.RowRejectedError) as exc:
        rows.turn_messages([*steps, *clean_turn()])
    assert exc.value.rule == "F5"
