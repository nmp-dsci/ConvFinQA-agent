"""The agent_sdk runtime and the confidence judge in the serving path (s12).

No test reaches a model. `SdkSession` is replaced by a scripted fake that
returns a parsed `SdkTurnResult` and its capture, and `judge_turn` by a fake
that returns a verdict — so the event stream, the abstention contract, the
trace row and the session lifecycle are produced by the real code paths.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.backends.agent_sdk import SdkTurnResult, result_to_capture
from convfinqa.serving import app as api_app
from convfinqa.serving import sdk_turn
from convfinqa.serving.routes import chat as chat_routes

REPORT = chat_routes.REPORT_IDS[0] if chat_routes.REPORT_IDS else ""


def _result(answer: str, *, program: bool = True) -> SdkTurnResult:
    if program:
        return SdkTurnResult.model_validate(
            {
                "turn_type": "program",
                "conv_type": "Type I",
                "sub_questions": ["a", "b"],
                "program": "subtract(A, B)",
                "retrieved": [
                    {"question": "a", "answer": "200", "source": "table"},
                    {"question": "b", "answer": "50", "source": "table"},
                ],
                "answer": answer,
                "reasoning": "computed",
            }
        )
    return SdkTurnResult.model_validate(
        {
            "turn_type": "number",
            "conv_type": "Type I",
            "answer": answer,
            "retrieved": [{"question": "q", "answer": answer, "source": "table"}],
        }
    )


class FakeSdkSession:
    """Scripted answers per turn; records opens, asks and closes."""

    answers: list[str] = ["150", "42"]
    instances: list[FakeSdkSession] = []

    def __init__(self, report_id: str, **_: Any) -> None:
        self.report_id = report_id
        self.asked: list[str] = []
        self.closed = False
        FakeSdkSession.instances.append(self)

    async def ask(
        self, question: str, *, history_text: str
    ) -> tuple[SdkTurnResult, dict[str, Any], dict[str, Any]]:
        answer = self.answers[len(self.asked)]
        self.asked.append(question)
        result = _result(answer)
        trajectory = [
            {"event": "tool_call", "tool": "subtract", "args": {"a": 200, "b": 50}},
            {"event": "tool_return", "tool": "subtract", "result": answer},
        ]
        capture: dict[str, Any] = {"history_text": history_text}
        capture.update(
            result_to_capture(
                result,
                question=question,
                history_text=history_text,
                trajectory=trajectory,
                metrics={
                    "num_turns": 2,
                    "duration_ms": 900,
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            )
        )
        return result, capture, {}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def sdk_serving(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The agent_sdk runtime with a fake session and a scripted judge."""
    from convfinqa.config import settings
    from convfinqa.evalloop import judge
    from convfinqa.serving import sdk_session
    from convfinqa.tracking import registry

    FakeSdkSession.instances = []
    FakeSdkSession.answers = ["150", "42"]
    monkeypatch.setattr(sdk_session, "SdkSession", FakeSdkSession)
    monkeypatch.setattr(settings, "serving_runtime", "agent_sdk")
    monkeypatch.setattr(settings, "judge_enabled", True)
    # Left at the shipped default (`advisory`) on purpose: the tests that
    # assert withholding opt into `gate` themselves, so the ones that do not
    # are testing the policy that actually ships.
    monkeypatch.setattr(settings, "judge_mode", "advisory")
    monkeypatch.setattr(registry, "judge_champion", lambda path=None: "judge_j1")
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v1")

    verdicts: dict[str, Any] = {"band": "high", "p_correct": 0.97}
    seen: list[dict[str, Any]] = []

    async def fake_judge_turn(
        row: Any, *, version: str, **kw: Any
    ) -> tuple[Any, dict[str, Any]]:
        seen.append({"row": dict(row), "version": version})
        if verdicts.get("raise"):
            raise RuntimeError("the SDK returned no content at all")
        return (
            judge.JudgeVerdict(
                checks=judge.Checks(**dict.fromkeys(judge.CHECKS, "pass")),
                band=verdicts["band"],
                p_correct=verdicts["p_correct"],
                reason="scripted",
            ),
            {"duration_ms": 300, "usage": {"input_tokens": 50, "output_tokens": 8}},
        )

    monkeypatch.setattr(judge, "judge_turn", fake_judge_turn)
    return {"verdicts": verdicts, "seen": seen}


def _client() -> TestClient:
    return TestClient(
        api_app.create_app(session_ttl_seconds=1800, eviction_interval_seconds=3600)
    )


def _stream(client: TestClient, session_id: str, question: str) -> list[dict[str, Any]]:
    with client.stream(
        "POST", f"/sessions/{session_id}/ask/stream", json={"question": question}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    return [
        json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
    ]


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_a_high_band_turn_streams_the_stages_the_verdict_and_the_answer(
    sdk_serving: dict[str, Any],
) -> None:
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        events = _stream(client, sid, "what was the change?")
        names = [e["event"] for e in events]
        # The pipeline's vocabulary, in its order, then the judge, then the answer.
        assert names[:2] == ["stage_start", "stage_output"]
        assert [e["stage"] for e in events if e["event"] == "stage_start"] == [
            "triage", "preprocess", "retriever", "calculator", "judge",
        ]  # fmt: skip
        assert "tool_call" in names and "tool_return" in names
        verdict = next(e for e in events if e["event"] == "judge")
        assert verdict["band"] == "high" and verdict["version"] == "judge_j1"
        assert set(verdict["checks"]) == set(
            ("operand_in_source", "period_matches", "reference_resolved",
             "program_matches", "arithmetic_verified", "unit_and_scale")
        )  # fmt: skip
        answer = next(e for e in events if e["event"] == "answer")
        assert answer == {
            "event": "answer",
            "answer": "150",
            "program": "subtract(A, B)",
            "band": "high",
            "withheld": False,
        }
        assert events[-1]["event"] == "done"
        # The judge saw the capture as a row — the trail, never gold.
        row = sdk_serving["seen"][-1]["row"]
        assert row["pred_answer"] == "150" and "gold_answer" not in row
        assert json.loads(row["calculator_io"])["trajectory"][0]["tool"] == "subtract"
        # History shows the released answer; the session held the conversation.
        history = client.get(f"/sessions/{sid}").json()["history"]
        assert history[-1]["answer"] == "150"
        assert FakeSdkSession.instances[-1].asked == ["what was the change?"]


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_advisory_is_the_default_and_a_low_band_still_releases_the_answer(
    sdk_serving: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """s13: the shipped policy. A `low` band is a caveat, not an abstention.

    Gating was measured and did not earn its cost — it withheld four correct
    answers for every wrong one it stopped — so the default shows the answer
    with the band attached, and nothing about the turn's record changes.
    """
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "judge_mode", "advisory")
    sdk_serving["verdicts"].update({"band": "low", "p_correct": 0.3})
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        body = client.post(
            f"/sessions/{sid}/ask", json={"question": "what was the change?"}
        ).json()
        assert body["band"] == "low" and body["withheld"] is False
        assert body["answer"] == "150"
        assert body["history"][-1]["answer"] == "150"
        # The stage frames are not redacted either — there is nothing to hide.
        events = _stream(client, sid, "and doubled?")
        outputs = [e for e in events if e["event"] == "stage_output"]
        assert outputs and any(e.get("output") for e in outputs)
        answer = next(e for e in events if e["event"] == "answer")
        assert answer["answer"] == "42" and answer["withheld"] is False
        assert answer["band"] == "low"


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_a_low_band_turn_withholds_the_answer_but_records_it(
    sdk_serving: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`judge_mode="gate"`: kept and tested for a judge that earns it."""
    from convfinqa.config import settings
    from convfinqa.tracking import traces

    monkeypatch.setattr(settings, "judge_mode", "gate")

    store = traces.TraceStore(tmp_path / "traces.db")
    monkeypatch.setattr(traces, "get_store", lambda: store)
    sdk_serving["verdicts"].update({"band": "low", "p_correct": 0.3})
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        response = client.post(
            f"/sessions/{sid}/ask", json={"question": "what was the change?"}
        )
        body = response.json()
        assert body["withheld"] is True and body["band"] == "low"
        assert body["answer"] == ""
        assert body["history"][-1]["answer"] == ""
        # The trace keeps the withheld value and the band, so it stays measurable.
        row = store.get_turn(body["trace_id"])
        assert row is not None
        assert row["answer"] == "150" and row["judge_band"] == "low"
        assert row["judge_p"] == pytest.approx(0.3)
        assert row["runtime"] == "agent_sdk"
        assert row["capture"]["judge"]["answer"] == "150"
        assert store.list_turns()[0]["judge_band"] == "low"
        # A later turn still runs on the same session, with the history it saw.
        FakeSdkSession.answers = ["150", "42"]
        sdk_serving["verdicts"].update({"band": "high", "p_correct": 0.9})
        events = _stream(client, sid, "and doubled?")
        answer = next(e for e in events if e["event"] == "answer")
        assert answer["answer"] == "42" and answer["withheld"] is False
        assert FakeSdkSession.instances[-1].asked == [
            "what was the change?",
            "and doubled?",
        ]


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_a_low_band_stream_never_leaks_the_answer_before_the_verdict(
    sdk_serving: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `gate`, withholding has to hold on the streaming path too.

    The stage frames carry the answer and are built before the judge runs, so
    a stream that forwarded them verbatim released what the band refused.
    """
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "judge_mode", "gate")
    sdk_serving["verdicts"].update({"band": "low", "p_correct": 0.3})
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        events = _stream(client, sid, "what was the change?")
        # The real value never appears in any frame — not the content frames
        # emitted before the judge's verdict, not the judge frame, not the
        # answer frame.
        assert "150" not in json.dumps(events)
        stage_output = [e for e in events if e["event"] == "stage_output"]
        assert stage_output and all(e["output"] == {} for e in stage_output)
        tool_calls = [e for e in events if e["event"] in {"tool_call", "tool_return"}]
        assert tool_calls
        for frame in tool_calls:
            assert frame.get("args", {}) == {} and frame.get("result", "") == ""
        answer = next(e for e in events if e["event"] == "answer")
        assert answer["answer"] == "" and answer["withheld"] is True


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_a_failed_judge_fails_closed(
    sdk_serving: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge that cannot run bands `low` — under `gate`, that withholds.

    Failing closed is about the *band*, which is what the record keeps and
    what a stricter mode acts on; what a `low` band then does to the answer
    is `judge_mode`.
    """
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "judge_mode", "gate")
    sdk_serving["verdicts"]["raise"] = True
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        events = _stream(client, sid, "what was the change?")
        verdict = next(e for e in events if e["event"] == "judge")
        assert verdict["band"] == "low" and "no content" in verdict["error"]
        answer = next(e for e in events if e["event"] == "answer")
        assert answer["withheld"] is True and answer["answer"] == ""


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_no_judge_releases_every_answer(
    sdk_serving: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "judge_enabled", False)
    with _client() as client:
        assert client.get("/healthz").json()["judge_champion"] is None
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        events = _stream(client, sid, "what was the change?")
        assert not [e for e in events if e["event"] == "judge"]
        answer = next(e for e in events if e["event"] == "answer")
        assert answer["band"] is None and answer["answer"] == "150"
        assert not sdk_serving["seen"]


@pytest.mark.skipif(not REPORT, reason="no reports loaded")
def test_deleting_a_session_closes_its_client_and_healthz_names_the_runtime(
    sdk_serving: dict[str, Any],
) -> None:
    with _client() as client:
        health = client.get("/healthz").json()
        assert health["runtime"] == "agent_sdk"
        assert (
            health["sdk_champion"] == "sdk_v1"
            and health["judge_champion"] == "judge_j1"
        )
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        _stream(client, sid, "what was the change?")
        session = FakeSdkSession.instances[-1]
        assert not session.closed
        assert client.delete(f"/sessions/{sid}").status_code == 204
        assert session.closed
    # And shutdown closes what eviction left behind.
    with _client() as client:
        sid = client.post("/sessions", json={"report_id": REPORT}).json()["session_id"]
        _stream(client, sid, "what was the change?")
        session = FakeSdkSession.instances[-1]
    assert session.closed


def test_stage_frames_mirror_the_pipeline_vocabulary() -> None:
    result = _result("150")
    capture: dict[str, Any] = {"history_text": ""}
    capture.update(
        result_to_capture(
            result,
            question="q",
            history_text="",
            trajectory=[{"event": "tool_call", "tool": "subtract", "args": {}}],
            metrics={"num_turns": 1},
        )
    )
    frames = sdk_turn.stage_frames(capture)
    assert [f["event"] + ":" + f.get("stage", "") for f in frames] == [
        "stage_start:triage", "stage_output:triage",
        "stage_start:preprocess", "stage_output:preprocess",
        "stage_start:retriever", "stage_output:retriever",
        "stage_start:calculator", "tool_call:calculator", "stage_output:calculator",
    ]  # fmt: skip
    number: dict[str, Any] = {"history_text": ""}
    number.update(
        result_to_capture(
            _result("7", program=False),
            question="q",
            history_text="",
            trajectory=[],
            metrics={},
        )
    )
    assert [f["stage"] for f in sdk_turn.stage_frames(number) if f["event"] == "stage_start"] == [
        "triage", "retriever",
    ]  # fmt: skip
