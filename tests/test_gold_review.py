"""`GET /eval/gold-review`: the diagnoses a human settles by hand, read from the ledger."""

# ruff: noqa: D103

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.evalloop import ledgers
from convfinqa.serving import app as api_app


def _get(path: str = "/eval/gold-review") -> Any:
    with TestClient(api_app.create_app(eviction_interval_seconds=3600)) as client:
        response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


def _write(ledger_dir: Path, lines: list[str]) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "diagnoses.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _row(**over: Any) -> str:
    base: dict[str, Any] = {
        "diagnosis_id": "d-1",
        "report_id": "Single_X/2010/page_1.pdf-1",
        "question_id": "Single_X/2010/page_1.pdf-1_q0",
        "turn_index": 0,
        "question": "what was revenue?",
        "gold_answer": "12",
        "gold_program": "",
        "pred_answer": "13",
        "derived_agent": "retriever",
        "gold_suspect": False,
        "adjudicated": False,
        "attribution_disputed": False,
    }
    base.update(over)
    return json.dumps(base)


def test_committed_ledger_counts_match_the_file(
    demo_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demo serves the committed ledger, and the counts are the file's own."""
    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(ledgers.LEDGER_DIR))
    rows = [
        json.loads(line)
        for line in (ledgers.LEDGER_DIR / "diagnoses.jsonl").read_text().splitlines()
        if line.strip()
    ]
    body = _get()
    summary = body["summary"]
    assert body["reason"] == ""
    assert body["source"] == "evaluation/diagnostics/evalloop/diagnoses.jsonl"
    assert summary["n_diagnoses"] == len(rows)
    assert summary["n_gold_suspect_flags"] == sum(
        bool(r.get("gold_suspect")) for r in rows
    )
    assert summary["n_derived_gold_suspect"] == sum(
        r.get("derived_agent") == "gold_suspect" for r in rows
    )
    assert summary["n_ambiguous"] == sum(
        r.get("derived_agent") == "ambiguous" for r in rows
    )
    assert summary["n_adjudicated"] == sum(bool(r.get("adjudicated")) for r in rows)
    assert summary["n_disputed"] == sum(
        bool(r.get("attribution_disputed")) for r in rows
    )
    assert summary["n_review_rows"] == len(body["cases"]) > 0
    assert summary["n_review_questions"] == len(
        {c["question_id"] for c in body["cases"]}
    )
    for case in body["cases"]:
        assert case["reasons"], case["diagnosis_id"]


def test_review_set_is_exactly_the_rows_a_human_should_see(
    demo_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / "review-ledger"
    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(ledger_dir))
    _write(
        ledger_dir,
        [
            _row(diagnosis_id="plain"),  # an ordinary agent fault: not for review
            _row(diagnosis_id="disputed-only", attribution_disputed=True),
            _row(diagnosis_id="flag", gold_suspect="true"),  # an old string spelling
            _row(diagnosis_id="rule", derived_agent="gold_suspect"),
            _row(
                diagnosis_id="amb",
                derived_agent="ambiguous",
                missing_gold_operands='["14.3", "11.8"]',
            ),
            _row(
                diagnosis_id="adj",
                adjudicated=True,
                adjudication_reason="the sub-question asked for it",
                attribution_disputed=True,
                turn_index=None,
                confidence=float("nan"),
            ),
            "{not json",  # a torn line is skipped, never a 500
        ],
    )
    body = _get()
    ids = [c["diagnosis_id"] for c in body["cases"]]
    assert ids == ["flag", "rule", "amb", "adj"]
    by_id = {c["diagnosis_id"]: c for c in body["cases"]}
    assert by_id["flag"]["reasons"] == ["gold_suspect_flag"]
    assert by_id["rule"]["reasons"] == ["derived_gold_suspect"]
    assert by_id["amb"]["reasons"] == ["ambiguous"]
    assert by_id["amb"]["missing_gold_operands"] == ["14.3", "11.8"]
    assert by_id["adj"]["reasons"] == ["adjudicated"]
    assert by_id["adj"]["turn_index"] is None
    assert by_id["adj"]["confidence"] is None
    assert by_id["adj"]["gold_answer"] == "12"  # gold is shown: this is dev evidence
    assert body["summary"] == {
        "n_diagnoses": 6,
        "n_gold_suspect_flags": 1,
        "n_derived_gold_suspect": 1,
        "n_ambiguous": 1,
        "n_adjudicated": 1,
        "n_disputed": 2,
        "n_review_rows": 4,
        "n_review_questions": 1,
    }


def test_missing_ledger_is_an_empty_payload_with_its_reason(
    demo_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(tmp_path / "nowhere"))
    body = _get()
    assert body["cases"] == []
    assert body["summary"]["n_diagnoses"] == 0
    assert "no diagnoses ledger" in body["reason"]
