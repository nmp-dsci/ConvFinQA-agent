"""s33 sampling: grading, pairing and the branch-sample-continue walk, offline.

The walk drives `open_agent._ask` and `build_agent` for real on Pydantic AI's
`FunctionModel`, so what is stored is what a served try would store; no GPU,
no network, no key.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from convfinqa.dpo import pairs, sample

# --- grading -----------------------------------------------------------------


def test_sign_check_rejects_a_wrong_sign_numeric_match() -> None:
    from convfinqa.evaluation.metrics import numeric_match

    assert numeric_match("-0.4", "0.4")  # the blind spot: both round to 0
    g = pairs.grade("-0.4", "0.4")
    assert g == {"numeric_ok": True, "sign_ok": False, "right": False}
    assert pairs.grade("40%", "0.4")["right"]
    assert pairs.grade("0", "-0.2")["sign_ok"]  # zero has no sign to disagree
    assert pairs.grade("yes", "yes")["right"]


def _t(i: int, answer: str, right: bool, **kw: Any) -> dict[str, Any]:
    return {"i": i, "answer": answer, "right": right, "attempts": 1, **kw}


def test_pairs_are_disjoint_capped_and_cover_distinct_mistakes() -> None:
    tries = [
        _t(0, "5", True),
        _t(1, "7", False),
        _t(2, "7", False),
        _t(3, "9", False),
        _t(4, "5", True),
        _t(5, "5", True),
    ]
    for seed in range(20):
        got = pairs.select(tries, random.Random(seed))
        assert len(got) == 2
        used = [i for p in got for i in p]
        assert len(used) == len(set(used))
        assert {tries[r]["answer"] for _, r in got} == {"7", "9"}


def test_unclean_and_dropped_tries_are_never_paired() -> None:
    tries = [
        _t(0, "5", True, harness_retry="H2"),
        _t(1, "6", False),
        {"i": 2, "error": "no valid reply"},
        _t(3, "5", True, attempts=2),
    ]
    assert pairs.select(tries, random.Random(0)) == []
    # but a right try with a retry is still history the conversation can go on
    assert pairs.keep(tries, [], random.Random(0)) in (0, 3)


def test_keep_prefers_an_unpaired_clean_right_try() -> None:
    tries = [_t(0, "5", True), _t(1, "6", False), _t(2, "5", True), _t(3, "5", True)]
    got = pairs.select(tries, random.Random(1))
    kept = pairs.keep(tries, got, random.Random(1))
    assert kept not in {i for p in got for i in p}
    assert pairs.keep([_t(0, "6", False)], [], random.Random(0)) is None


# --- the walk ------------------------------------------------------------------


REPORT = "Single_JKHY/2009/page_28.pdf-3"


def _fake_model(answers: dict[int, list[str]]) -> Any:
    """A model whose k-th question's tries answer `answers[k]` in turn."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    served: dict[int, int] = {}

    def k_of(messages: list[Any]) -> int:
        prompts = [
            p
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, UserPromptPart)
        ]
        return len(prompts) - 1

    def fn(messages: list[Any], info: AgentInfo) -> ModelResponse:
        k = k_of(messages)
        n = served.get(k, 0)
        served[k] = n + 1
        answer = answers[k][n % len(answers[k])]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="StructuredOutput",
                    args={
                        "turn_type": "number",
                        "conv_type": "Type I",
                        "answer": answer,
                        "reasoning": f"read {answer}",
                    },
                )
            ]
        )

    return FunctionModel(fn)


def _walk(answers: dict[int, list[str]], golds: list[str]) -> dict[str, Any]:
    model = _fake_model(answers)
    qs = [
        {
            "question_id": f"{REPORT}_q{k}",
            "question": f"q{k}?",
            "gold": g,
            "turn_type": "Number",
        }
        for k, g in enumerate(golds)
    ]
    return asyncio.run(
        sample.sample_conversation(
            REPORT,
            qs,
            system_prompt="You answer questions about one report.",
            make_model=lambda: model,
            model_name="fake",
            tries=4,
            max_turns=4,
        )
    )


def test_walk_pairs_continues_on_a_right_try_and_stops_when_none_is_right() -> None:
    rec = _walk(
        {0: ["10", "10", "12", "10"], 1: ["3", "3", "3", "3"], 2: ["8"]},
        ["10", "3", "1", "4"],
    )
    assert rec["stop"] == "no right try at q2"
    assert [len(q["tries"]) for q in rec["questions"]] == [4, 4, 4]
    q0, q1, q2 = rec["questions"]
    assert len(q0["pairs"]) == 1 and q0["tries"][q0["pairs"][0][1]]["answer"] == "12"
    assert q0["kept"] not in q0["pairs"][0]
    assert q1["pairs"] == [] and q1["history_from"] == q0["kept"]
    assert q2["kept"] is None and all(not t["right"] for t in q2["tries"])


def test_stored_tries_leave_out_the_report_and_the_system_prompt() -> None:
    rec = _walk({0: ["10", "11"]}, ["10"])
    q0 = rec["questions"][0]
    assert q0["prompt"] is None and len(q0["prompt_sha"]) == 16
    for t in q0["tries"]:
        first = t["messages"][0]
        assert first["kind"] == "request"
        assert not first.get("instructions")
        assert [
            p["content"] for p in first["parts"] if p["part_kind"] == "user-prompt"
        ] == [""]
        assert t["prompt_suffix"] == ""
        assert "_history" not in t
    assert "JKHY" not in str(q0["tries"])  # the document is not stored per try


def test_later_questions_store_their_short_prompt() -> None:
    rec = _walk({0: ["10"], 1: ["2", "5"]}, ["10", "2"])
    q1 = rec["questions"][1]
    assert q1["prompt"] and "q1?" in q1["prompt"]
    assert len(q1["pairs"]) == 2
    assert rec["stop"] == "completed"


def test_manifest_is_page_clean_and_disjoint() -> None:
    from convfinqa.dpo import manifest
    from convfinqa.sft import splits

    m = manifest.build()
    assert m["stats"]["conversations"] == 1489
    assert m["stats"]["questions"] == 4394
    train, ev = m["splits"]["train"], m["splits"]["eval"]
    assert not {splits.page_of(r) for r in train} & {splits.page_of(r) for r in ev}
    assert sorted(m["order"]) == sorted(train + ev)
    assert len(set(m["order"])) == len(m["order"])


# --- s34: the top-up ----------------------------------------------------------


SYSTEM = "You answer questions about one report."


def _questions(golds: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": f"{REPORT}_q{k}",
            "question": f"q{k}?",
            "gold": g,
            "turn_type": "Number",
        }
        for k, g in enumerate(golds)
    ]


def test_rebuilt_history_is_what_the_model_was_sent() -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter, UserPromptPart

    seen: dict[int, list[Any]] = {}
    model = _fake_model({0: ["10"], 1: ["2"], 2: ["7"]})
    inner = model.function

    def spy(messages: list[Any], info: Any) -> Any:
        k = (
            sum(
                isinstance(p, UserPromptPart)
                for m in messages
                for p in getattr(m, "parts", [])
            )
            - 1
        )
        seen.setdefault(k, list(messages))
        return inner(messages, info)  # type: ignore[misc]

    model.function = spy  # type: ignore[assignment]
    golds = ["10", "2", "7"]
    rec = asyncio.run(
        sample.sample_conversation(
            REPORT,
            _questions(golds),
            system_prompt=SYSTEM,
            make_model=lambda: model,
            model_name="fake",
            tries=2,
            max_turns=4,
        )
    )
    assert rec["stop"] == "completed"
    for k in (1, 2):
        rebuilt = sample.rebuild_history(REPORT, rec["questions"], k, SYSTEM)
        dump = ModelMessagesTypeAdapter.dump_python
        a, b = dump(rebuilt, mode="json"), dump(seen[k], mode="json")
        # the request being sent is the last one; the history is everything before
        # it, its closing tool return included, which Pydantic AI folds into it
        last = dict(
            b[-1], parts=[p for p in b[-1]["parts"] if p["part_kind"] != "user-prompt"]
        )

        def sent(ms: list[Any]) -> list[Any]:
            """What reaches the model: back-to-back requests merge, no stamps."""
            out: list[Any] = []
            for m in ms:
                parts = [
                    {x: v for x, v in p.items() if x != "timestamp"} for p in m["parts"]
                ]
                if out and m["kind"] == "request" and out[-1][0] == "request":
                    out[-1][2].extend(parts)
                else:
                    out.append((m["kind"], m.get("instructions"), parts))
            return out

        assert sent(a) == sent(b[:-1] + [last])


def test_topup_adds_tries_keeps_history_and_walks_past_an_old_stop() -> None:
    golds = ["10", "3", "4"]
    old_model = _fake_model({0: ["10", "11"], 1: ["9"], 2: ["4"]})
    old = asyncio.run(
        sample.sample_conversation(
            REPORT,
            _questions(golds),
            system_prompt=SYSTEM,
            make_model=lambda: old_model,
            model_name="fake",
            tries=2,
            max_turns=4,
        )
    )
    assert old["stop"] == "no right try at q1"
    old["tries"] = 2
    # the top-up's tries at q1 include a right one, so the walk reaches q2
    new_model = _fake_model({0: ["10", "12"], 1: ["3", "9"], 2: ["4"]})
    rec = asyncio.run(
        sample.topup_conversation(
            old,
            _questions(golds),
            system_prompt=SYSTEM,
            make_model=lambda: new_model,
            model_name="fake",
            tries=4,
            max_turns=4,
        )
    )
    q0, q1, q2 = rec["questions"]
    assert [t["i"] for t in q0["tries"]] == [0, 1, 2, 3]
    assert q0["tries"][:2] == old["questions"][0]["tries"]
    assert q0["kept"] == old["questions"][0]["kept"]
    assert all(t.get("round") == 2 for t in q0["tries"][2:])
    assert q1["kept"] in (2, 3) and q1["tries"][q1["kept"]]["right"]
    assert len(q2["tries"]) == 4 and rec["stop"] == "completed"
    assert rec["topped_up_from"] == 2


def test_capped_pairs_spread_over_mistakes_and_tries_before_repeating() -> None:
    tries = [_t(i, "5", True) for i in range(4)] + [
        _t(4, "7", False),
        _t(5, "7", False),
        _t(6, "9", False),
        _t(7, "5", True, harness_retry="H3"),
    ]
    for seed in range(20):
        got = pairs.select_capped(tries, random.Random(seed), cap=4)
        assert len(got) == 4 and len(set(got)) == 4
        assert 7 not in {c for c, _ in got}  # unclean never paired
        assert {tries[r]["answer"] for _, r in got[:2]} == {"7", "9"}
        assert len({c for c, _ in got}) == 4  # each right try used once
    assert len(pairs.select_capped(tries, random.Random(0), cap=99)) == 12


def test_strict_chosen_drops_an_integer_rounding_match() -> None:
    assert not pairs.close("0.0931", "0.0852")
    assert pairs.close("4.2%", "4%") and pairs.close("0.773", "0.8")
    assert pairs.close("18.5%", "0.185") and pairs.close("yes", "yes")
    tries = [_t(0, "0.0931", True), _t(1, "0.5", False)]
    assert (
        pairs.select_capped(tries, random.Random(0), 4, gold="0.0852", strict=True)
        == []
    )


def test_selection_rows_index_tries_and_are_reproducible() -> None:
    q = {
        "k": 0,
        "question_id": "R_q0",
        "gold": "5",
        "gold_turn_type": "Program",
        "tries": [_t(0, "5", True), _t(1, "7", False), _t(2, "5", True)],
    }
    records = {"R": {"questions": [q]}}
    rows = pairs.selection(records, {"R": "train"}, cap=4)
    assert rows == pairs.selection(records, {"R": "train"}, cap=4)
    assert len(rows) == 2
    assert {(r["chosen_i"], r["rejected_i"]) for r in rows} == {(0, 1), (2, 1)}
    assert all(r["kind"] == "program" and r["split"] == "train" for r in rows)
    assert all(r["rejected_answer"] == "7" for r in rows)


def test_records_read_the_same_gzipped(tmp_path: Path) -> None:
    import gzip

    from convfinqa.dpo import sample

    rec = {"report_id": "R", "questions": []}
    plain = tmp_path / "tries.jsonl"
    plain.write_text(json.dumps(rec) + "\n")
    assert sample.load_records(plain) == {"R": rec}
    with gzip.open(tmp_path / "tries.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rec) + "\n")
    plain.unlink()
    assert sample.load_records(plain) == {"R": rec}
    assert sample.done_ids(plain) == {"R"}


def _msg(role: str, **kw: Any) -> dict[str, Any]:
    return {"role": role, "content": kw.pop("content", ""), **kw}


def _final(answer: str) -> dict[str, Any]:
    args = json.dumps({"answer": answer})
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": "StructuredOutput", "arguments": args},
    }
    return _msg("assistant", tool_calls=[call])


def test_dpo_row_check_accepts_a_sound_row_and_names_faults() -> None:
    from convfinqa.dpo import render

    prompt = [_msg("system", content="sys"), _msg("user", content="q?")]
    row = {
        "prompt": prompt,
        "chosen": [_final("5")],
        "rejected": [_final("7")],
        "tools": [],
        "meta": {"chosen_answer": "5", "rejected_answer": "7"},
    }
    assert render.check(row) == []
    bad = {
        **row,
        "rejected": [_final("5")],
        "meta": {**row["meta"], "rejected_answer": "5"},
    }
    assert "chosen and rejected are the same" in render.check(bad)
    leak = {**row, "chosen": [_msg("user", content="x"), _final("5")]}
    assert any("does not open with an assistant" in p for p in render.check(leak))
    first, rest = render._split(prompt + [_final("5")])
    assert first == prompt and rest == [_final("5")]


def test_teacher_turn_is_shaped_like_a_student_try() -> None:
    """Question k's messages, opened by a blanked prompt, the previous turn's tail dropped."""
    from convfinqa.dpo.teacher import teacher_turn

    def req(*parts: dict[str, object], **extra: object) -> dict[str, object]:
        return {"kind": "request", "parts": list(parts), **extra}

    def resp(name: str) -> dict[str, object]:
        return {
            "kind": "response",
            "parts": [{"part_kind": "tool-call", "tool_name": name}],
        }

    ret = {"part_kind": "tool-return", "content": "Final result processed."}
    conv = [
        req({"part_kind": "user-prompt", "content": "q0"}, instructions="sys"),
        resp("StructuredOutput"),
        req(ret, {"part_kind": "user-prompt", "content": "q1"}),
        resp("divide"),
        req({"part_kind": "tool-return", "content": "2.0"}),
        resp("StructuredOutput"),
        req(ret, {"part_kind": "user-prompt", "content": "q2"}),
        resp("StructuredOutput"),
    ]
    turn = teacher_turn(conv, 1)
    assert turn[0]["parts"] == [{"part_kind": "user-prompt", "content": ""}]
    assert [m["kind"] for m in turn] == ["request", "response", "request", "response"]
    assert teacher_turn(conv, 0)[0]["instructions"] == "sys"
    assert len(teacher_turn(conv, 2)) == 2
    import pytest

    with pytest.raises(ValueError):
        teacher_turn(conv, 3)
