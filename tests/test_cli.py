# ruff: noqa: D103

from __future__ import annotations

import json
from typing import Any

import httpx
from typer.testing import CliRunner

from convfinqa.serving import cli


class _Prompt:
    def __init__(self, value: str | None) -> None:
        self.value = value

    def ask(self) -> str | None:
        return self.value


def _mock_client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(base_url="http://testserver", timeout=120.0, transport=handler)


def test_reports_command(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/reports"
        return httpx.Response(200, json=["r1", "r2"])

    monkeypatch.setattr(cli, "build_client", lambda base_url=cli.DEFAULT_BASE_URL: _mock_client(httpx.MockTransport(handler)))
    result = CliRunner().invoke(cli.cli_app, ["reports"])
    assert result.exit_code == 0
    assert "r1" in result.stdout
    assert "r2" in result.stdout


def test_ask_one_shot(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "abc123"})
        if request.url.path == "/sessions/abc123/ask":
            payload = json.loads(request.content.decode())
            assert payload["question"] == "what?"
            return httpx.Response(200, json={"answer": "42"})
        raise AssertionError(f"unexpected path {request.url.path}")

    monkeypatch.setattr(cli, "build_client", lambda base_url=cli.DEFAULT_BASE_URL: _mock_client(httpx.MockTransport(handler)))
    result = CliRunner().invoke(cli.cli_app, ["ask", "--report", "r1", "--question", "what?"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


def _sse(events: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events)).encode()


def test_interactive_default_flow_with_change_report(monkeypatch: Any) -> None:
    prompts = iter(
        [
            "r1",
            "Ask a question",
            "gold q1",
            "Ask a question",
            "gold q1",
            "Change report",
            "r2",
            "Ask a question",
            "gold q2",
            "Quit",
        ]
    )

    def next_prompt() -> str:
        return next(prompts)

    monkeypatch.setattr(cli.questionary, "autocomplete", lambda *args, **kwargs: _Prompt(next_prompt()))
    monkeypatch.setattr(cli.questionary, "select", lambda *args, **kwargs: _Prompt(next_prompt()))

    session_ids = iter(["sess-1", "sess-2"])
    asked: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reports":
            return httpx.Response(200, json=["r1", "r2"])
        if request.url.path == "/reports/r1/questions":
            return httpx.Response(
                200,
                json=[{"q_order": 1, "question": "gold q1", "gold_answer": "a1"}],
            )
        if request.url.path == "/reports/r2/questions":
            return httpx.Response(
                200,
                json=[{"q_order": 1, "question": "gold q2", "gold_answer": "a2"}],
            )
        if request.url.path == "/sessions":
            payload = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={"session_id": next(session_ids), "report_id": payload["report_id"]},
            )
        if request.url.path.endswith("/ask/stream"):
            payload = json.loads(request.content.decode())
            asked.append((request.url.path, payload["question"]))
            body = _sse(
                [
                    {"event": "stage_start", "stage": "triage"},
                    {
                        "event": "stage_output",
                        "stage": "triage",
                        "output": {"turn_type": "number", "conv_type": "Type I"},
                    },
                    {"event": "answer", "answer": f"ans:{payload['question']}"},
                    {"event": "done", "turn_index": 0},
                ]
            )
            return httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    monkeypatch.setattr(cli, "build_client", lambda base_url=cli.DEFAULT_BASE_URL: _mock_client(httpx.MockTransport(handler)))
    result = CliRunner().invoke(cli.cli_app, [])
    assert result.exit_code == 0
    assert "Session sess-1" in result.stdout
    assert "Session sess-2" in result.stdout
    assert asked == [
        ("/sessions/sess-1/ask/stream", "gold q1"),
        ("/sessions/sess-1/ask/stream", "gold q1"),
        ("/sessions/sess-2/ask/stream", "gold q2"),
    ]
    assert "ans:gold q1" in result.stdout
    assert "ans:gold q2" in result.stdout


def test_run_all_streams_every_gold_question(monkeypatch: Any) -> None:
    asked: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "sx", "report_id": "r1"})
        if request.url.path == "/reports/r1/questions":
            return httpx.Response(
                200,
                json=[
                    {"q_order": 1, "question": "q1", "gold_answer": "10"},
                    {"q_order": 2, "question": "q2", "gold_answer": "20"},
                ],
            )
        if request.url.path == "/sessions/sx/ask/stream":
            payload = json.loads(request.content.decode())
            asked.append((request.url.path, payload["question"]))
            answer = "10" if payload["question"] == "q1" else "20"
            body = _sse(
                [
                    {"event": "stage_start", "stage": "triage"},
                    {"event": "answer", "answer": answer},
                    {"event": "done", "turn_index": 0},
                ]
            )
            return httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    monkeypatch.setattr(cli, "build_client", lambda base_url=cli.DEFAULT_BASE_URL: _mock_client(httpx.MockTransport(handler)))
    result = CliRunner().invoke(cli.cli_app, ["run-all", "--report", "r1"])
    assert result.exit_code == 0
    assert asked == [
        ("/sessions/sx/ask/stream", "q1"),
        ("/sessions/sx/ask/stream", "q2"),
    ]
    assert "Total: 2/2" in result.stdout
