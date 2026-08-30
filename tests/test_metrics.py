"""`/metrics/production`, and the error vocabulary it groups by.

The property under test throughout is that the three sources stay apart. A
replayed demo turn and a live serving turn are both "a turn the system answered"
and their latencies mean different things, so any aggregate that mixes them is
wrong however it is computed. The tests assert the separation directly rather
than asserting numbers that happen to come out right today.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.error_codes import ALL_CODES, ErrorCode, classify, normalise
from convfinqa.serving import app as api_app
from convfinqa.tracking import traces
from convfinqa.tracking.traces import TraceStore


def _client() -> TestClient:
    return TestClient(api_app.create_app(eviction_interval_seconds=3600))


def _capture(latency: float, tokens_in: int, tokens_out: int) -> dict[str, Any]:
    return {
        "triage": {
            "output": {"turn_type": "number"},
            "metrics": {
                "latency_ms": latency,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        }
    }


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TraceStore]:
    """A real trace store wired in as the process store, for route tests."""
    from convfinqa.config import settings

    monkeypatch.setattr(settings, "trace_capture_enabled", True, raising=False)
    monkeypatch.setattr(traces, "default_db_path", lambda: tmp_path / "metrics.db")
    traces.reset_store()
    live = traces.get_store()
    assert live is not None
    yield live
    traces.reset_store()


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


def test_every_code_is_in_the_published_list() -> None:
    """`ALL_CODES` is what a dashboard builds its x-axis from."""
    assert set(ALL_CODES) == {code.value for code in ErrorCode}


def test_demo_and_recording_errors_keep_their_own_codes() -> None:
    """The two refusals a visitor can actually trigger stay distinguishable."""
    from convfinqa.llm import DemoModeError
    from convfinqa.serving.demo_pack.replay import NoRecordingError

    assert classify(DemoModeError()) == "not_available_demo"
    assert classify(NoRecordingError("r1", "q", 0.1)) == "no_recording"


def test_timeout_beats_the_declared_code() -> None:
    """An abandoned call is a timeout, even though it is raised as unavailable."""
    import asyncio

    assert classify(asyncio.TimeoutError()) == "timeout"


def test_rate_limit_is_read_off_the_response() -> None:
    """A 429 is its own condition: it means wait, not that the model is down."""

    class _Response:
        status_code = 429

    class _RateLimitedError(Exception):
        response = _Response()

    assert classify(_RateLimitedError()) == "rate_limited"


def test_unclassifiable_errors_are_unknown_not_dropped() -> None:
    """An unrecognised failure still counts; the free text does the explaining."""
    assert classify(ValueError("something odd")) == "unknown"
    assert normalise("") == "unknown"
    assert normalise("some_legacy_code") == "unknown"
    assert normalise("timeout") == "timeout"


# ---------------------------------------------------------------------------
# The store side
# ---------------------------------------------------------------------------


def test_error_code_is_persisted_beside_the_free_text(tmp_path: Path) -> None:
    """Both, never one: the code groups, the message explains."""
    store = TraceStore(tmp_path / "t.db")
    trace_id = store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture={},
        error="provider returned 503",
        error_code="llm_unavailable",
    )
    record = store.get_turn(trace_id)
    assert record is not None
    assert record["error"] == "provider returned 503"
    assert record["error_code"] == "llm_unavailable"
    store.close()


def test_cost_is_computed_and_stored_per_turn(tmp_path: Path) -> None:
    """`tracking.cost` was computing this and nothing was reading it."""
    store = TraceStore(tmp_path / "t.db")
    trace_id = store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture=_capture(1000.0, 10_000, 2_000),
    )
    record = store.get_turn(trace_id)
    assert record is not None
    assert record["input_tokens"] == 10_000
    assert record["output_tokens"] == 2_000
    assert record["cost_usd"] == pytest.approx(
        (10_000 * 0.28 + 2_000 * 0.42) / 1_000_000
    )
    store.close()


def test_an_unmeasured_turn_stores_nulls_not_zeros(tmp_path: Path) -> None:
    """ "Nobody wrote it down" must not be stored as "it cost nothing".

    The committed prediction CSVs, and the demo pack built from them, predate
    per-stage metrics. Recording those turns as zeros would put a free, instant
    turn into every average — the one direction a wrong number looks like good
    news.
    """
    store = TraceStore(tmp_path / "t.db")
    trace_id = store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture={"triage": {"output": {"turn_type": "number"}}},
    )
    record = store.get_turn(trace_id)
    assert record is not None
    assert record["latency_ms"] is None
    assert record["total_tokens"] is None
    assert record["cost_usd"] is None
    store.close()


def test_unmeasured_turns_are_counted_but_not_averaged(store: TraceStore) -> None:
    """They are real turns; they are just not latency samples."""
    store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture={"triage": {"output": {}}},
        source="demo",
    )
    with _client() as client:
        group = client.get("/metrics/production").json()["sources"]["demo"]

    assert group["n_turns"] == 1
    assert group["latency_ms"]["n_measured"] == 0
    assert group["latency_ms"]["p50"] is None
    assert group["cost_usd"]["per_turn"] is None


def test_an_old_database_is_widened_rather_than_rejected(tmp_path: Path) -> None:
    """A dev machine's existing traces.db must survive the new columns."""
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE turns (trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
        "source TEXT NOT NULL, session_id TEXT, report_id TEXT NOT NULL, "
        "turn_index INTEGER NOT NULL, question TEXT NOT NULL, answer TEXT, "
        "program TEXT, gold_answer TEXT, correct INTEGER, bundle_id TEXT, "
        "bundle TEXT, latency_ms REAL, total_tokens INTEGER, error TEXT, "
        "capture TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    store = TraceStore(path)
    trace_id = store.record(
        report_id="r1", turn_index=0, question="q", capture=_capture(50.0, 100, 10)
    )
    assert store.get_turn(trace_id) is not None
    store.close()


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_empty_store_returns_zeros_not_a_500() -> None:
    """An empty trace store is a state, not a failure."""
    with _client() as client:
        response = client.get("/metrics/production")
    assert response.status_code == 200
    body = response.json()
    assert body["n_turns_total"] == 0
    for source in ("serving", "demo", "eval"):
        group = body["sources"][source]
        assert group["n_turns"] == 0
        # `None`, not 0.0 — zero is a real latency and "no sample" is not.
        assert group["latency_ms"]["p50"] is None
        assert group["accuracy"]["accuracy"] is None
        assert group["errors"]["n_errors"] == 0
        assert len(group["series"]) == 24


def test_sources_are_reported_separately_and_never_blended(store: TraceStore) -> None:
    """The honesty rule, asserted: a demo turn must not move a serving number."""
    store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture=_capture(30_000.0, 8_000, 400),
        source="serving",
        gold_answer="42",
        correct=True,
    )
    store.record(
        report_id="r1",
        turn_index=0,
        question="q",
        capture=_capture(120.0, 10, 5),
        source="demo",
        gold_answer="42",
        correct=False,
    )

    with _client() as client:
        body = client.get("/metrics/production").json()

    serving = body["sources"]["serving"]
    demo = body["sources"]["demo"]
    assert serving["n_turns"] == 1
    assert demo["n_turns"] == 1
    assert serving["latency_ms"]["p50"] == pytest.approx(30_000.0)
    assert demo["latency_ms"]["p50"] == pytest.approx(120.0)
    assert serving["accuracy"]["accuracy"] == 1.0
    assert demo["accuracy"]["accuracy"] == 0.0
    assert body["n_turns_total"] == 2


def test_errors_are_broken_down_by_code(store: TraceStore) -> None:
    """The point of the enum: an error tile that can explain itself."""
    for code, message in (
        ("llm_unavailable", "provider returned 503"),
        ("llm_unavailable", "provider returned 502"),
        ("no_recording", "nothing recorded for that"),
    ):
        store.record(
            report_id="r1",
            turn_index=0,
            question="q",
            capture={},
            source="serving",
            error=message,
            error_code=code,
        )

    with _client() as client:
        errors = client.get("/metrics/production").json()["sources"]["serving"][
            "errors"
        ]

    assert errors["n_errors"] == 3
    assert errors["by_code"]["llm_unavailable"] == 2
    assert errors["by_code"]["no_recording"] == 1
    assert errors["by_code"]["timeout"] == 0
    assert set(errors["by_code"]) == set(ALL_CODES)


def test_cost_and_tokens_surface_per_turn(store: TraceStore) -> None:
    """Cost per turn is the number that decides whether the shape is affordable."""
    for _ in range(2):
        store.record(
            report_id="r1",
            turn_index=0,
            question="q",
            capture=_capture(1_000.0, 10_000, 2_000),
            source="serving",
        )

    with _client() as client:
        group = client.get("/metrics/production").json()["sources"]["serving"]

    per_turn = (10_000 * 0.28 + 2_000 * 0.42) / 1_000_000
    assert group["cost_usd"]["per_turn"] == pytest.approx(per_turn, rel=1e-3)
    assert group["cost_usd"]["total"] == pytest.approx(2 * per_turn, rel=1e-3)
    assert group["tokens_per_turn"]["p50"] == pytest.approx(12_000.0)


def test_metrics_is_registered_in_demo_mode(demo_mode: None) -> None:
    """It is a read route, so the public demo serves it too — labelled `demo`."""
    with _client() as client:
        response = client.get("/metrics/production")
    assert response.status_code == 200
    assert "demo" in response.json()["sources"]


# ---------------------------------------------------------------------------
# Metrics reaching the committed artefacts
# ---------------------------------------------------------------------------


def test_eval_csv_columns_carry_the_stage_metrics() -> None:
    """A scored turn's `*_io` must keep `metrics`, not just input/output.

    Without this the demo — which is rebuilt from these CSVs — shows zeros for
    latency and cost, which read as facts rather than as missing data.
    """
    import json as _json

    from convfinqa.evaluation.runner import _capture_to_row_fields

    fields = _capture_to_row_fields(
        {
            "triage": {
                "input": {"question": "q"},
                "output": {"turn_type": "number"},
                "metrics": {"latency_ms": 900.0, "input_tokens": 5, "output_tokens": 2},
            }
        }
    )
    assert _json.loads(fields["triage_io"])["metrics"]["latency_ms"] == 900.0


def test_demo_pack_events_forward_the_recorded_metrics() -> None:
    """`events_from_row` is the other half of the runner's event contract."""
    import json as _json

    import pandas as pd

    from convfinqa.serving.demo_pack.cli import events_from_row

    row = pd.Series(
        {
            "triage_io": _json.dumps(
                {
                    "output": {"turn_type": "number"},
                    "metrics": {"latency_ms": 1200.0, "total_tokens": 40},
                }
            ),
            "preprocess_io": "",
            "retriever_io": _json.dumps(
                {"output": {"answers": []}, "metrics": {"latency_ms": 800.0}}
            ),
            "calculator_io": "",
            "pred_answer": "42",
            "pred_program": "",
        }
    )
    outputs = {
        e["stage"]: e.get("metrics")
        for e in events_from_row(row)
        if e["event"] == "stage_output"
    }
    assert outputs["triage"]["latency_ms"] == 1200.0
    assert outputs["retriever"]["latency_ms"] == 800.0
