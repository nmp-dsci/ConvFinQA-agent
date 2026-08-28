# ruff: noqa: D103

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from convfinqa.serving import app as api_app
from convfinqa.serving.routes import chat as chat_routes


def _stub_overrides(
    triage_args: dict[str, Any],
    retriever_args: dict[str, Any] | None = None,
    preprocess_args: dict[str, Any] | None = None,
    calc_args: dict[str, Any] | None = None,
) -> ExitStack:
    from convfinqa.backends.pydantic import default_agents

    agents = default_agents()
    stack = ExitStack()
    stack.enter_context(
        agents["triage"].override(model=TestModel(custom_output_args=triage_args))
    )
    if preprocess_args is not None:
        stack.enter_context(
            agents["preprocess"].override(
                model=TestModel(custom_output_args=preprocess_args)
            )
        )
    if retriever_args is not None:
        stack.enter_context(
            agents["retriever"].override(
                model=TestModel(custom_output_args=retriever_args)
            )
        )
    if calc_args is not None:
        stack.enter_context(
            agents["calculator"].override(
                model=TestModel(custom_output_args=calc_args, call_tools=[])
            )
        )
    return stack


def _client(
    *, ttl_seconds: int = 1800, eviction_interval_seconds: int = 3600
) -> TestClient:
    return TestClient(
        api_app.create_app(
            session_ttl_seconds=ttl_seconds,
            eviction_interval_seconds=eviction_interval_seconds,
        )
    )


def _stub_turn_events(monkeypatch: Any) -> None:
    """Replace the pipeline with a deterministic generator.

    Yields the same `answer` event the real `turn_events` does and appends to the
    conversation the same way, so the route under test exercises its real code
    path without a model.
    """
    from convfinqa.pipeline import runner as pipeline_runner

    async def fake_turn_events(
        question: str,
        report_id: str,
        conversation: Any,
        *,
        agents: Any = None,
        capture: Any = None,
    ) -> Any:
        answer = f"{question}:{len(conversation.pairs)}"
        conversation.append(question=question, answer=answer, report_id=report_id)
        yield {"event": "answer", "answer": answer, "program": ""}

    monkeypatch.setattr(pipeline_runner, "turn_events", fake_turn_events)


def test_healthz() -> None:
    """Health reports liveness plus the mode and bundle the frontend reads."""
    with _client() as client:
        body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["mode"] == "dev"
    assert body["bundle"]["prompts_version"]
    assert len(body["bundle_id"]) == 12


def test_reports_and_questions_endpoints() -> None:
    rid = chat_routes.REPORT_IDS[0]
    with _client() as client:
        reports = client.get("/reports?limit=5").json()
        assert rid in reports
        report = client.get(f"/reports/{rid}").json()
        assert report["report_id"] == rid
        questions = client.get(f"/reports/{rid}/questions").json()
        assert questions
        assert {"q_order", "question", "gold_answer"} <= set(questions[0])


def test_session_lifecycle_starts_empty() -> None:
    rid = chat_routes.REPORT_IDS[0]
    with _client() as client:
        created = client.post("/sessions", json={"report_id": rid}).json()
        assert created["report_id"] == rid
        assert created["n_turns"] == 0
        assert created["history"] == []

        fetched = client.get(f"/sessions/{created['session_id']}").json()
        assert fetched["n_turns"] == 0

        deleted = client.delete(f"/sessions/{created['session_id']}")
        assert deleted.status_code == 204
        assert client.get(f"/sessions/{created['session_id']}").status_code == 404


def test_ask_rejects_extra_report_id_field() -> None:
    rid = chat_routes.REPORT_IDS[0]
    with _client() as client:
        session = client.post("/sessions", json={"report_id": rid}).json()
        response = client.post(
            f"/sessions/{session['session_id']}/ask",
            json={"question": "q", "report_id": "other"},
        )
        assert response.status_code == 422


def test_ask_uses_run_turn_and_updates_history(monkeypatch: Any) -> None:
    rid = chat_routes.REPORT_IDS[0]
    calls: list[tuple[str, str, int]] = []

    from convfinqa.pipeline import runner as pipeline_runner

    async def fake_turn_events(
        question: str,
        report_id: str,
        conversation: Any,
        *,
        agents: Any = None,
        capture: Any = None,
    ) -> Any:
        calls.append((question, report_id, len(conversation.pairs)))
        conversation.append(question=question, answer="stubbed", report_id=report_id)
        yield {"event": "answer", "answer": "stubbed", "program": ""}

    monkeypatch.setattr(pipeline_runner, "turn_events", fake_turn_events)

    with _client() as client:
        session = client.post("/sessions", json={"report_id": rid}).json()
        response = client.post(
            f"/sessions/{session['session_id']}/ask",
            json={"question": "what now?"},
        )
        body = response.json()
        assert body["answer"] == "stubbed"
        assert body["turn_index"] == 0
        assert calls == [("what now?", rid, 0)]


def test_number_path_with_testmodel_overrides() -> None:
    rid = chat_routes.REPORT_IDS[0]
    triage_a = {"reasoning": "r", "turn_type": "number", "conv_type": "Type I"}
    retr_a = {"reasoning": "r", "answers": [{"question": "q", "answer": "42"}]}

    with _stub_overrides(triage_a, retriever_args=retr_a), _client() as client:
        session = client.post("/sessions", json={"report_id": rid}).json()
        response = client.post(
            f"/sessions/{session['session_id']}/ask",
            json={"question": "what is the value?"},
        )
        assert response.json()["answer"] == "42"


def test_program_path_with_testmodel_overrides() -> None:
    rid = chat_routes.REPORT_IDS[0]
    triage_a = {"reasoning": "r", "turn_type": "program", "conv_type": "Type I"}
    pp_a = {
        "reasoning": "r",
        "sub_questions": ["a?", "b?"],
        "program": "subtract(A, B)",
    }
    retr_a = {
        "reasoning": "r",
        "answers": [
            {"question": "a?", "answer": "10"},
            {"question": "b?", "answer": "3"},
        ],
    }
    calc_a = {"answer": "7"}

    with (
        _stub_overrides(
            triage_a,
            retriever_args=retr_a,
            preprocess_args=pp_a,
            calc_args=calc_a,
        ),
        _client() as client,
    ):
        session = client.post("/sessions", json={"report_id": rid}).json()
        response = client.post(
            f"/sessions/{session['session_id']}/ask",
            json={"question": "compute the change"},
        )
        assert response.json()["answer"] == "7"


def test_new_sessions_do_not_inherit_history(monkeypatch: Any) -> None:
    rid = chat_routes.REPORT_IDS[0]

    _stub_turn_events(monkeypatch)

    with _client() as client:
        first = client.post("/sessions", json={"report_id": rid}).json()
        client.post(f"/sessions/{first['session_id']}/ask", json={"question": "q1"})

        second = client.post("/sessions", json={"report_id": rid}).json()
        fetched = client.get(f"/sessions/{second['session_id']}").json()
        assert fetched["n_turns"] == 0
        assert fetched["history"] == []


def test_session_isolation_same_report(monkeypatch: Any) -> None:
    rid = chat_routes.REPORT_IDS[0]

    _stub_turn_events(monkeypatch)

    with _client() as client:
        a = client.post("/sessions", json={"report_id": rid}).json()
        b = client.post("/sessions", json={"report_id": rid}).json()
        ra = client.post(
            f"/sessions/{a['session_id']}/ask", json={"question": "x"}
        ).json()
        rb = client.post(
            f"/sessions/{b['session_id']}/ask", json={"question": "x"}
        ).json()
        assert ra["turn_index"] == 0
        assert rb["turn_index"] == 0
        assert client.get(f"/sessions/{a['session_id']}").json()["n_turns"] == 1
        assert client.get(f"/sessions/{b['session_id']}").json()["n_turns"] == 1


def test_ttl_eviction_cleans_session_and_lock() -> None:
    rid = chat_routes.REPORT_IDS[0]
    with _client(ttl_seconds=1) as client:
        session = client.post("/sessions", json={"report_id": rid}).json()
        store = client.app.state.session_store
        state = store.get(session["session_id"])
        state.updated_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        evicted = store.evict_expired()
        assert session["session_id"] in evicted
        assert session["session_id"] not in store.sessions
        assert session["session_id"] not in store.locks
