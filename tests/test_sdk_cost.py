"""The price an sdk pass paid (s19, 2026-10-15).

The CLI's `total_cost_usd` is a session's running total, and the bundled CLI
prices a model it does not know on a fallback card. Both corrections live in
`tracking/cost.py`; the backfill derives the figure from a run's traces.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from convfinqa.evalloop import story
from convfinqa.tracking import cost, sdk_cost


def test_a_conversation_costs_its_last_running_total() -> None:
    # Two conversations; the CLI's figure only grows within each.
    totals = [[0.10, 0.25, 0.40], [0.05, None, 0.09]]
    assert cost.sdk_session_cost_usd(totals, "claude-sonnet-5") == pytest.approx(0.49)


def test_a_turn_with_no_reply_does_not_reset_the_total() -> None:
    assert cost.sdk_session_cost_usd([[0.2, None], [None]], None) == pytest.approx(0.2)


def test_a_model_the_cli_prices_is_taken_as_priced() -> None:
    assert cost.sdk_cli_cost_factor("claude-sonnet-5") == 1.0
    assert cost.sdk_cli_cost_factor("claude-haiku-4-5-20251001") == 1.0
    assert cost.sdk_cli_cost_factor(None) == 1.0


def test_haiku_5_5_is_repriced_from_the_fallback_card() -> None:
    # A recorded Haiku 5.5 turn (Double_KIM/2010 q0, 2026-10-15): the CLI said
    # $0.2168665 for 4 input, 1,572 output, 15,268 cache-written and 49,733
    # cache-read tokens. That is the fallback card to the micro-dollar ...
    fallback = (4 * 5 + 1572 * 25 + 15268 * 5 * 2 + 49733 * 5 * 0.1) / 1e6
    assert fallback == pytest.approx(0.2168665, abs=1e-6)
    # ... and Haiku 5.5's own card (≤100K prompt) is the same bill times 1/50.
    haiku = (4 * 0.10 + 1572 * 0.50 + 15268 * 0.10 * 2 + 49733 * 0.10 * 0.1) / 1e6
    factor = cost.sdk_cli_cost_factor("claude-haiku-5-5")
    assert factor == pytest.approx(0.02)
    assert cost.sdk_session_cost_usd([[0.2168665]], "claude-haiku-5-5") == (
        pytest.approx(haiku, abs=1e-6)
    )


def test_the_repricing_table_is_pinned_to_the_cli_it_was_measured_on() -> None:
    # The table is true of the CLI bundled with this SDK only. An upgrade whose
    # CLI knows Haiku 5.5 would be divided by 50 a second time: re-measure (the
    # CLI prints `unrecognized_model` for a model it does not price), then move
    # this pin.
    import claude_agent_sdk

    assert claude_agent_sdk.__version__ == "0.2.152"


def _archive(path: Path, run_id: str, conversations: list[list[float]]) -> None:
    db = sqlite3.connect(path)
    db.execute("create table trace_request_metadata (request_id, key, value)")
    db.execute("create table spans (trace_id, content)")
    for i, totals in enumerate(conversations):
        trace = f"tr-{i}"
        db.execute(
            "insert into trace_request_metadata values (?, 'mlflow.sourceRun', ?)",
            (trace, run_id),
        )
        db.execute(
            "insert into spans values (?, ?)",
            (trace, json.dumps({"attributes": {"report_id": '"r"'}})),
        )
        for value in totals:
            attributes = {"total_cost_usd": json.dumps(value)}
            db.execute(
                "insert into spans values (?, ?)",
                (trace, json.dumps({"attributes": attributes})),
            )
    db.commit()
    db.close()


def test_archive_totals_are_read_per_conversation(tmp_path: Path) -> None:
    db = tmp_path / "mlflow.db"
    _archive(db, "run-1", [[0.1, 0.3], [0.2]])
    assert sdk_cost.archive_running_totals(db, "run-1") == [[0.1, 0.3], [0.2]]
    assert sdk_cost.archive_running_totals(db, "other") is None
    assert sdk_cost.archive_running_totals(tmp_path / "missing.db", "run-1") is None


def test_backfill_writes_the_snapshot_and_skips_what_it_cannot_find(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "mlflow.db"
    _archive(db, "found", [[0.1, 0.3], [0.2]])
    runs = [
        {
            "run_id": run_id,
            "run_name": run_id,
            "params": {"runtime": "agent_sdk", "sdk_model": model},
            "metrics": {"sdk_cost_usd": 9.9},
            "tags": {},
        }
        for run_id, model in (("found", "claude-sonnet-5"), ("lost", None))
    ]
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"runs": runs}))
    monkeypatch.setattr(sdk_cost.snapshot, "MLFLOW_SNAPSHOT_PATH", path)
    monkeypatch.setattr(
        sdk_cost.snapshot, "read_snapshot", lambda: json.loads(path.read_text())
    )
    monkeypatch.setattr(sdk_cost.mlflow_log, "reachable", lambda: False)

    report = sdk_cost.backfill_sdk_cost(archive=db)

    assert report["backfilled"]["found"]["sdk_session_cost_usd"] == pytest.approx(0.5)
    assert report["skipped"] == ["lost"]
    written = {r["run_id"]: r for r in json.loads(path.read_text())["runs"]}
    assert written["found"]["metrics"]["sdk_session_cost_usd"] == pytest.approx(0.5)
    # The recorded figure stays, under its own name.
    assert written["found"]["metrics"]["sdk_cost_usd"] == 9.9
    assert "sdk_session_cost_usd" not in written["lost"]["metrics"]


def test_an_sdk_arm_is_priced_by_the_session_cost_alone() -> None:
    sdk = {"params": {"runtime": "agent_sdk"}, "metrics": {"sdk_cost_usd": 27.62}}
    assert story._arm(sdk)["cost"] is None
    sdk["metrics"]["sdk_session_cost_usd"] = 11.83
    assert story._arm(sdk)["cost"] == 11.83
    pipeline = {"params": {"runtime": "multi_agent"}, "metrics": {"cost_usd": 1.2}}
    assert story._arm(pipeline)["cost"] == 1.2
