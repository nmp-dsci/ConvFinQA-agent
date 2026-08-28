"""Demo mode: the gate holds, and it holds *only* in demo mode.

Both directions matter. A gate that refuses everything is not a demo, and a gate
that can be routed around is not a gate — so these tests assert the refusal in
demo mode and the absence of any refusal outside it, against the same code.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.llm import DemoModeError, demo_mode_enabled, get_model, get_provider
from convfinqa.serving import app as api_app
from convfinqa.serving.demo_pack import replay
from convfinqa.serving.demo_pack.store import (
    DemoPack,
    PackedTurn,
    load_pack,
    similarity,
)
from convfinqa.serving.routes import chat as chat_routes


def _client() -> TestClient:
    return TestClient(api_app.create_app(eviction_interval_seconds=3600))


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_llm_is_blocked_in_demo_mode(demo_mode: None) -> None:
    """Every model constructor refuses before a provider is even built."""
    assert demo_mode_enabled() is True
    with pytest.raises(DemoModeError):
        get_provider()
    with pytest.raises(DemoModeError):
        get_model()


def test_dspy_path_is_blocked_too(demo_mode: None) -> None:
    """DSPy builds its own client, so it needs the same gate — and has it."""
    from convfinqa.llm import dspy_lm_kwargs

    with pytest.raises(DemoModeError):
        dspy_lm_kwargs()


def test_llm_functions_untouched_outside_demo() -> None:
    """Outside demo mode nothing refuses; the gate is not a blanket disable."""
    assert demo_mode_enabled() is False
    model = get_model()
    assert model is not None


def test_demo_error_carries_stable_code() -> None:
    """The frontend maps `code`, not prose, so the code must be stable."""
    assert DemoModeError.code == "not_available_demo"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _pack() -> DemoPack:
    return DemoPack(
        turns=[
            PackedTurn(
                report_id="r1",
                turn_index=0,
                question="what was the total revenue in 2009?",
                answer="1234.5",
                program="",
                gold_answer="1234.5",
                correct=True,
                events=[
                    {"event": "stage_start", "stage": "triage"},
                    {
                        "event": "stage_output",
                        "stage": "triage",
                        "output": {"turn_type": "number"},
                    },
                    {"event": "answer", "answer": "1234.5", "program": ""},
                ],
            )
        ]
    )


@pytest.mark.asyncio
async def test_replay_emits_the_recorded_events() -> None:
    """Replay yields the recorded stream and appends to history like the live path."""
    from convfinqa.data.schemas import ConversationHistory

    history = ConversationHistory()
    events = [
        event
        async for event in replay.replay_turn(
            "what was the total revenue in 2009?",
            "r1",
            history,
            pack=_pack(),
            pace=False,
        )
    ]
    assert [e["event"] for e in events] == ["stage_start", "stage_output", "answer"]
    assert len(history.pairs) == 1
    assert history.pairs[0].answer == "1234.5"


@pytest.mark.asyncio
async def test_replay_declines_rather_than_guessing() -> None:
    """An unrelated question is an honest miss, never the nearest recording.

    Confidently returning another report's number would be the worst failure this
    demo could have — the whole subject is numerical accuracy.
    """
    from convfinqa.data.schemas import ConversationHistory

    with pytest.raises(replay.NoRecordingError):
        async for _ in replay.replay_turn(
            "who is the chief executive officer?",
            "r1",
            ConversationHistory(),
            pack=_pack(),
            pace=False,
        ):
            pass


def test_fuzzy_match_tolerates_rewording() -> None:
    """A reworded question still resolves; an unrelated one does not."""
    pack = _pack()
    matched, score = pack.match("r1", "what was total revenue in 2009")
    assert matched is not None
    assert score > 0.6
    missed, _ = pack.match("r1", "how many employees were there")
    assert missed is None


def test_similarity_is_bounded() -> None:
    """Similarity stays in [0, 1] and is 1.0 only for identical text."""
    assert similarity("a b c", "a b c") == 1.0
    assert 0.0 <= similarity("total revenue", "revenue total") <= 1.0
    assert similarity("", "anything") == 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_healthz_reports_demo_mode(demo_mode: None) -> None:
    """The frontend configures itself from this field, so it must be right."""
    with _client() as client:
        body = client.get("/healthz").json()
    assert body["mode"] == "demo"


def test_read_only_routes_stay_live_in_demo(demo_mode: None) -> None:
    """Splits, answers and reports are genuinely live — that is the honesty."""
    with _client() as client:
        assert client.get("/eval/splits").status_code == 200
        assert client.get("/eval/runs").status_code == 200
        assert client.get("/reports?limit=5").status_code == 200


def test_admin_writes_refused_in_demo(demo_mode: None) -> None:
    """Promotion is refused without a token, and would be refused with one."""
    with _client() as client:
        response = client.post("/admin/registry/promote", json={"version": "v2"})
    assert response.status_code == 403


def test_committed_pack_is_loadable_and_nonempty() -> None:
    """The pack that ships is real: it parses and holds recorded turns."""
    pack = load_pack()
    assert pack.turns, "committed demo pack is empty — run `convfinqa-demo-pack`"
    for turn in pack.turns:
        assert turn.events, f"{turn.report_id} turn {turn.turn_index} has no events"
        assert turn.events[-1]["event"] == "answer"


def test_pack_events_match_the_live_vocabulary() -> None:
    """Recorded events use exactly the event names the live path emits."""
    allowed = {"stage_start", "stage_output", "tool_call", "tool_return", "answer"}
    for turn in load_pack().turns:
        for event in turn.events:
            assert event["event"] in allowed


@pytest.mark.asyncio
async def test_demo_ask_stream_replays_end_to_end(demo_mode: None) -> None:
    """A full turn over the real route in demo mode ends with an answer frame."""
    pack = load_pack()
    if not pack.turns:
        pytest.skip("no committed demo pack")
    turn = pack.turns[0]

    with _client() as client:
        session = client.post("/sessions", json={"report_id": turn.report_id}).json()
        with client.stream(
            "POST",
            f"/sessions/{session['session_id']}/ask/stream",
            json={"question": turn.question},
        ) as response:
            frames = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert frames, "no SSE frames received"
    assert any(f.get("event") == "answer" for f in frames)
    assert frames[-1]["event"] == "done"


def test_demo_questions_endpoint(demo_mode: None) -> None:
    """The chip rail lists what the report can actually answer."""
    pack = load_pack()
    if not pack.turns:
        pytest.skip("no committed demo pack")
    report_id = pack.report_ids[0]
    with _client() as client:
        body: list[dict[str, Any]] = client.get(
            f"/demo/questions?report_id={report_id}"
        ).json()
    assert body
    assert all("question" in entry for entry in body)


def test_demo_report_list_is_restricted_to_the_pack(demo_mode: None) -> None:
    """In demo mode the picker only offers reports the pack can answer."""
    pack = load_pack()
    if not pack.turns:
        pytest.skip("no committed demo pack")
    with _client() as client:
        listed = client.get("/reports?limit=500").json()
    assert set(listed) <= set(pack.report_ids)
    assert set(listed) <= set(chat_routes.REPORT_IDS)


# ---------------------------------------------------------------------------
# Keyless import
# ---------------------------------------------------------------------------


def test_every_module_imports_without_a_key(demo_mode: None) -> None:
    """No module may construct a model at import time.

    Regression test for a real deployment failure: `backends.dspy` built its
    DSPy LMs at module scope, so the keyless demo container returned 500 from
    `/eval/splits` — a read-only route that has no business touching a model —
    simply because reading a dataset fact imported that module.
    """
    import importlib

    for name in (
        "convfinqa.backends.dspy",
        "convfinqa.backends.pydantic",
        "convfinqa.diagnosis.agents",
        "convfinqa.optimization.gepa",
        "convfinqa.serving.evaldata",
        "convfinqa.pipeline.runner",
    ):
        importlib.import_module(name)


def test_split_is_readable_without_importing_an_optimizer(demo_mode: None) -> None:
    """The serving layer reads the split from the dataset, not from DSPy."""
    from convfinqa.serving import evaldata

    assert len(evaldata.optimizer_train_ids()) == 120
    assert len(evaldata.splits()["never_seen"]) == 80
