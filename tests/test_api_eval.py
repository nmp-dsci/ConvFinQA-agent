# ruff: noqa: D103

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from convfinqa.evaluation import api_runner as api_eval


def test_evaluate_api_writes_predictions_csv(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_eval, "analyze_predictions", lambda _: None)

    examples = [
        SimpleNamespace(
            report_id="r1",
            questions=["q1", "q2"],
            gold_answers=["1", "2"],
        )
    ]

    asks = iter(["1", "2"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "sess-1"})
        if request.url.path == "/sessions/sess-1/ask":
            return httpx.Response(200, json={"answer": next(asks)})
        if request.url.path == "/sessions/sess-1":
            return httpx.Response(204)
        raise AssertionError(f"unexpected path {request.url.path}")

    out = api_eval.evaluate_api(
        base_url="http://testserver",
        examples=examples,
        transport=httpx.MockTransport(handler),
    )

    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["report_id"] == "r1"
    assert rows[1]["pred_answer"] == "2"


def test_compare_api_outputs_compares_against_both_existing_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    qa_data = api_eval.pd.DataFrame(
        [
            {
                "report_id": "r1",
                "q_order": 1,
                "qa_split": False,
                "turn_type": "Number",
            }
        ]
    )
    monkeypatch.setattr(api_eval, "load_conv_examples_test", lambda: ([], qa_data))

    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    api_csv = run_dir / "api_predictions.csv"
    dspy_csv = run_dir / "predictions.csv"
    pyd_csv = run_dir / "pydantic_predictions.csv"
    rows = [
        {
            "report_id": "r1",
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        }
    ]
    for path in (api_csv, dspy_csv, pyd_csv):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    comparison_out = api_eval.compare_api_outputs(
        api_csv,
        run_dir=run_dir,
    )
    assert comparison_out is not None
    assert comparison_out.exists()


def test_compare_model_accuracies_writes_summary_table(tmp_path: Path) -> None:
    qa_data = api_eval.pd.DataFrame(
        [
            {
                "report_id": "r1",
                "q_order": 1,
                "qa_split": False,
                "turn_type": "Number",
            },
            {
                "report_id": "r1",
                "q_order": 2,
                "qa_split": True,
                "turn_type": "Program",
            },
        ]
    )
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = [
        {
            "report_id": "r1",
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        },
        {
            "report_id": "r1",
            "turn_index": 1,
            "question": "q2",
            "gold_answer": "2",
            "pred_answer": "0",
            "correct": False,
        },
    ]
    rows_api = [
        {
            "report_id": "r1",
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        },
        {
            "report_id": "r1",
            "turn_index": 1,
            "question": "q2",
            "gold_answer": "2",
            "pred_answer": "2",
            "correct": True,
        },
    ]
    for filename, payload in (
        ("predictions.csv", rows),
        ("pydantic_predictions.csv", rows),
        ("api_predictions.csv", rows_api),
    ):
        with (run_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)

    output = api_eval.compare_model_accuracies(run_dir=run_dir, qa_data=qa_data)
    assert output is not None

    comparison = api_eval.pd.read_csv(output)
    assert {"slice", "value", "dspy_acc", "pydantic_acc", "api_acc"} <= set(
        comparison.columns
    )
    assert "overall" in comparison["slice"].tolist()
    assert "turn_type" in comparison["slice"].tolist()
    assert "q_order" in comparison["slice"].tolist()
    q_order_2 = comparison[
        (comparison["slice"] == "q_order") & (comparison["value"] == "2")
    ]
    assert not q_order_2.empty
    assert q_order_2["api_acc"].item() == 1.0
