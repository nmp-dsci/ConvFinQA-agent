# ruff: noqa: D103
# mypy: ignore-errors

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import httpx
import questionary
import typer
import uvicorn

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
cli_app = typer.Typer(add_completion=False, invoke_without_command=True)


def build_client(
    base_url: str = DEFAULT_BASE_URL,
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=300.0, transport=transport)


def _abort_if_none(value: str | None) -> str:
    if value is None:
        raise typer.Abort()
    return value


def _get_json(client: httpx.Client, path: str) -> Any:
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def _post_json(client: httpx.Client, path: str, payload: dict[str, Any]) -> Any:
    response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _create_session(client: httpx.Client, report_id: str) -> dict[str, Any]:
    return _post_json(client, "/sessions", {"report_id": report_id})


def _pick_report(client: httpx.Client) -> str:
    report_ids = _get_json(client, "/reports?limit=500")
    report_id = questionary.autocomplete(
        "Report ID",
        choices=report_ids,
        match_middle=True,
        ignore_case=True,
        validate=lambda text: (
            True if text in report_ids else "Choose a valid report_id"
        ),
    ).ask()
    return _abort_if_none(report_id)


def _pick_question(questions: list[dict[str, Any]]) -> str:
    presets = [q["question"] for q in questions]
    question = questionary.autocomplete(
        "Question",
        choices=presets,
        match_middle=True,
        ignore_case=True,
    ).ask()
    return _abort_if_none(question)


def _print_questions(questions: list[dict[str, Any]]) -> None:
    typer.echo("Gold questions:")
    for item in questions:
        typer.echo(f"{item['q_order']}. {item['question']} -> {item['gold_answer']}")


def _fmt_compact(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_event(event: dict[str, Any]) -> None:
    name = event.get("event")
    stage = event.get("stage", "")
    if name == "stage_start":
        typer.secho(f"→ {stage}…", fg=typer.colors.CYAN)
    elif name == "stage_output":
        out = event.get("output") or {}
        if stage == "triage":
            typer.secho(
                f"  turn_type={out.get('turn_type')}  conv_type={out.get('conv_type')}",
                fg=typer.colors.BRIGHT_BLACK,
            )
        elif stage == "preprocess":
            sq = out.get("sub_questions") or []
            typer.secho(f"  sub_questions ({len(sq)}):", fg=typer.colors.BRIGHT_BLACK)
            for q in sq:
                typer.secho(f"    - {q}", fg=typer.colors.BRIGHT_BLACK)
            typer.secho(
                f"  program: {out.get('program', '')}", fg=typer.colors.BRIGHT_BLACK
            )
        elif stage == "retriever":
            for a in out.get("answers", []) or []:
                typer.secho(
                    f"  Q: {_fmt_compact(a.get('question'))}",
                    fg=typer.colors.BRIGHT_BLACK,
                )
                typer.secho(
                    f"  A: {_fmt_compact(a.get('answer'))}",
                    fg=typer.colors.BRIGHT_BLACK,
                )
        elif stage == "calculator":
            typer.secho(f"  answer={out.get('answer')}", fg=typer.colors.BRIGHT_BLACK)
        else:
            typer.secho(f"  {_fmt_compact(out)}", fg=typer.colors.BRIGHT_BLACK)
    elif name == "tool_call":
        typer.secho(
            f"  ⚙ {event.get('tool')}({_fmt_compact(event.get('args'))})",
            fg=typer.colors.YELLOW,
        )
    elif name == "tool_return":
        typer.secho(
            f"    = {_fmt_compact(event.get('result'))}",
            fg=typer.colors.YELLOW,
        )
    elif name == "answer":
        typer.secho(f"Answer: {event.get('answer')}", fg=typer.colors.GREEN, bold=True)
    elif name == "error":
        typer.secho(f"Error: {event.get('error')}", fg=typer.colors.RED, bold=True)


def _stream_ask(
    client: httpx.Client,
    session_id: str,
    question: str,
    *,
    gold: str | None = None,
) -> tuple[str | None, bool]:
    """POST /sessions/{id}/ask/stream and print SSE events.

    If `gold` is provided, the final answer line is colored green when it
    matches the gold and red when it doesn't.

    Returns (final_answer, ok). ok is False if the server emitted an error.
    """
    answer: str | None = None
    ok = True
    with client.stream(
        "POST",
        f"/sessions/{session_id}/ask/stream",
        json={"question": question},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            kind = payload.get("event")
            if kind == "answer" and gold is not None:
                pred = payload.get("answer")
                match = pred is not None and _loose_numeric_match(str(pred), gold)
                color = typer.colors.GREEN if match else typer.colors.RED
                marker = "✓" if match else "✗"
                typer.secho(
                    f"Answer: {pred}  {marker} (gold: {gold})",
                    fg=color,
                    bold=True,
                )
            else:
                _print_event(payload)
            if kind == "answer":
                answer = payload.get("answer")
            elif kind == "error":
                ok = False
            elif kind == "done":
                return answer, ok
    return answer, ok


def _interactive_loop(client: httpx.Client, initial_report: str | None = None) -> None:
    report_id = initial_report or _pick_report(client)
    session = _create_session(client, report_id)
    typer.echo(f"Session {session['session_id'][:8]} report={report_id}")

    while True:
        action = questionary.select(
            "What next?",
            choices=[
                "Ask a question",
                "Run all gold questions",
                "Change report",
                "Quit",
            ],
        ).ask()
        action = _abort_if_none(action)

        if action == "Quit":
            return

        if action == "Change report":
            report_id = _pick_report(client)
            session = _create_session(client, report_id)
            typer.echo(f"Session {session['session_id'][:8]} report={report_id}")
            continue

        if action == "Run all gold questions":
            _run_all_gold(client, session["session_id"], report_id)
            continue

        # Ask a question
        questions = _get_json(client, f"/reports/{quote(report_id, safe='')}/questions")
        _print_questions(questions)
        question = _pick_question(questions)
        _stream_ask(client, session["session_id"], question)


def _run_all_gold(client: httpx.Client, session_id: str, report_id: str) -> None:
    """Walk every gold question for this report through /ask/stream sequentially.

    Each question is asked in turn against the *same* session, so conversation
    history is threaded just like during evaluation.
    """
    questions = _get_json(client, f"/reports/{quote(report_id, safe='')}/questions")
    typer.secho(
        f"\nRunning {len(questions)} gold questions on session "
        f"{session_id[:8]} ({report_id})\n",
        fg=typer.colors.CYAN,
        bold=True,
    )
    rows: list[tuple[int, str, str, str | None]] = []
    for q in questions:
        typer.secho(
            f"\n=== Q{q['q_order']}: {q['question']}",
            fg=typer.colors.BRIGHT_WHITE,
            bold=True,
        )
        typer.secho(f"    gold: {q['gold_answer']}", fg=typer.colors.BRIGHT_BLACK)
        try:
            answer, _ = _stream_ask(
                client, session_id, q["question"], gold=q["gold_answer"]
            )
        except httpx.HTTPError as exc:
            typer.secho(f"  HTTP error: {exc}", fg=typer.colors.RED)
            answer = None
        rows.append((q["q_order"], q["question"], q["gold_answer"], answer))

    typer.secho("\n=== Summary ===", fg=typer.colors.CYAN, bold=True)
    for q_order, _question, gold, pred in rows:
        match = pred is not None and _loose_numeric_match(pred, gold)
        marker = "✓" if match else "✗"
        color = typer.colors.GREEN if match else typer.colors.RED
        typer.secho(
            f"{marker} Q{q_order}: pred={pred!r}  gold={gold!r}",
            fg=color,
        )
    correct = sum(
        1
        for _, _, gold, pred in rows
        if pred is not None and _loose_numeric_match(pred, gold)
    )
    typer.secho(
        f"\nTotal: {correct}/{len(rows)}",
        fg=typer.colors.CYAN,
        bold=True,
    )


def _loose_numeric_match(pred: str, gold: str) -> bool:
    def _clean(s: str) -> str:
        return s.strip().replace("$", "").replace(",", "").replace("%", "").strip()

    try:
        return round(float(_clean(str(pred)))) == round(float(_clean(str(gold))))
    except (ValueError, TypeError):
        return _clean(str(pred)).lower() == _clean(str(gold)).lower()


@cli_app.callback()
def main(
    ctx: typer.Context,
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
) -> None:
    if ctx.invoked_subcommand is None:
        with build_client(base_url) as client:
            _interactive_loop(client)


@cli_app.command()
def ask(
    report: str = typer.Option(None, "--report"),
    question: str = typer.Option(None, "--question"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
) -> None:
    with build_client(base_url) as client:
        if report and question:
            session = _create_session(client, report)
            answer = _post_json(
                client,
                f"/sessions/{session['session_id']}/ask",
                {"question": question},
            )
            typer.echo(answer["answer"])
            return
        _interactive_loop(client, initial_report=report)


@cli_app.command("run-all")
def run_all_cmd(
    report: str = typer.Option(None, "--report"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
) -> None:
    """Run every gold question for a report sequentially through the stream endpoint."""
    with build_client(base_url) as client:
        report_id = report or _pick_report(client)
        session = _create_session(client, report_id)
        typer.echo(f"Session {session['session_id'][:8]} report={report_id}")
        _run_all_gold(client, session["session_id"], report_id)


@cli_app.command("reports")
def reports_cmd(
    q: str = typer.Option("", "--q"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
) -> None:
    with build_client(base_url) as client:
        report_ids = _get_json(client, f"/reports?q={q}&limit=500")
    for report_id in report_ids:
        typer.echo(report_id)


@cli_app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8765, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    uvicorn.run("app:app", host=host, port=port, workers=1, reload=reload)


if __name__ == "__main__":
    cli_app()
