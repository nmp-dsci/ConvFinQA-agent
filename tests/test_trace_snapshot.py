"""The committed trace export: verbatim, idempotent, and only for empty stores."""

# ruff: noqa: D103

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from convfinqa.config import settings
from convfinqa.tracking import trace_snapshot
from convfinqa.tracking.traces import TraceStore


def _capture() -> dict[str, Any]:
    return {
        "triage": {"output": {"turn_type": "Number"}, "metrics": {"latency_ms": 12.0}}
    }


def _store_with_history(path: Path) -> TraceStore:
    store = TraceStore(path)
    store.record(
        report_id="Double_MAR/2010/page_55.pdf",
        turn_index=0,
        question="what is the net change in cash from operations?",
        capture=_capture(),
        answer="1234",
        gold_answer="1234",
        correct=True,
        source="eval",
        run_id="run-a",
        split="test",
        bundle={"prompts_version": "sdk_v1"},
    )
    store.record(
        report_id="Double_MRO/2011/page_37.pdf",
        turn_index=1,
        question="what is the percent change?",
        capture=_capture(),
        answer="12%",
        source="serving",
        bundle={"prompts_version": "sdk_v1"},
    )
    # The demo's own replays are written by the deployment, so they are excluded
    # from the export rather than shipped back into the image.
    store.record(
        report_id="Double_OKE/2012/page_91.pdf",
        turn_index=0,
        question="replayed",
        capture={},
        source="demo",
        bundle={"prompts_version": "sdk_v1"},
    )
    return store


@pytest.fixture(autouse=True)
def _seeding_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trace_seed_from_snapshot", True, raising=False)


def test_export_then_seed_reproduces_the_store(tmp_path: Path) -> None:
    source = _store_with_history(tmp_path / "source.db")
    original = source.list_turns(limit=10)
    snapshot = tmp_path / "snap.jsonl.gz"
    written = trace_snapshot.export_snapshot(source, path=snapshot)
    source.close()

    assert written["n_turns"] == 2
    assert written["by_source"] == {"eval": 1, "serving": 1}

    target = TraceStore(tmp_path / "target.db")
    assert trace_snapshot.seed_if_empty(target, path=snapshot) == 2
    seeded = {row["trace_id"]: row for row in target.list_turns(limit=10)}
    for row in original:
        if row["source"] == "demo":
            assert row["trace_id"] not in seeded
            continue
        # Verbatim: the identity and the moment it was answered both survive.
        assert seeded[row["trace_id"]]["created_at"] == row["created_at"]
        assert seeded[row["trace_id"]]["answer"] == row["answer"]
    target.close()


def test_capture_survives_the_round_trip(tmp_path: Path) -> None:
    source = _store_with_history(tmp_path / "source.db")
    trace_id = source.list_turns(source="eval", limit=1)[0]["trace_id"]
    snapshot = tmp_path / "snap.jsonl.gz"
    trace_snapshot.export_snapshot(source, path=snapshot)
    source.close()

    target = TraceStore(tmp_path / "target.db")
    trace_snapshot.load_snapshot(target, path=snapshot)
    assert target.get_turn(trace_id)["capture"] == _capture()
    target.close()


def test_loading_twice_adds_nothing(tmp_path: Path) -> None:
    source = _store_with_history(tmp_path / "source.db")
    snapshot = tmp_path / "snap.jsonl.gz"
    trace_snapshot.export_snapshot(source, path=snapshot)
    source.close()

    target = TraceStore(tmp_path / "target.db")
    assert trace_snapshot.load_snapshot(target, path=snapshot) == 2
    assert trace_snapshot.load_snapshot(target, path=snapshot) == 0
    assert target.count() == 2
    target.close()


def test_a_store_with_its_own_history_is_never_seeded(tmp_path: Path) -> None:
    """A dev machine keeps what it measured; only a fresh store reads the file."""
    source = _store_with_history(tmp_path / "source.db")
    snapshot = tmp_path / "snap.jsonl.gz"
    trace_snapshot.export_snapshot(source, path=snapshot)
    source.close()

    target = _store_with_history(tmp_path / "target.db")
    assert trace_snapshot.seed_if_empty(target, path=snapshot) == 0
    assert target.count() == 3
    target.close()


def test_seeding_is_disabled_by_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store_with_history(tmp_path / "source.db")
    snapshot = tmp_path / "snap.jsonl.gz"
    trace_snapshot.export_snapshot(source, path=snapshot)
    source.close()

    monkeypatch.setattr(settings, "trace_seed_from_snapshot", False, raising=False)
    target = TraceStore(tmp_path / "target.db")
    assert trace_snapshot.seed_if_empty(target, path=snapshot) == 0
    target.close()


def test_a_missing_snapshot_is_not_an_error(tmp_path: Path) -> None:
    target = TraceStore(tmp_path / "target.db")
    assert trace_snapshot.seed_if_empty(target, path=tmp_path / "absent.gz") == 0
    target.close()


def test_unknown_columns_are_dropped_rather_than_refused(tmp_path: Path) -> None:
    """A snapshot from a later build loads into a store that predates it."""
    snapshot = tmp_path / "snap.jsonl.gz"
    row = {
        "trace_id": "abc123",
        "created_at": "2026-09-05T23:20:01+00:00",
        "source": "eval",
        "report_id": "Double_MAR/2010/page_55.pdf",
        "turn_index": 0,
        "question": "q",
        "capture": "{}",
        "a_column_from_the_future": "ignored",
    }
    with gzip.open(snapshot, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    target = TraceStore(tmp_path / "target.db")
    assert trace_snapshot.load_snapshot(target, path=snapshot) == 1
    assert target.get_turn("abc123")["question"] == "q"
    target.close()


def test_export_is_byte_stable_for_an_unchanged_store(tmp_path: Path) -> None:
    """An export with nothing new must not show up as a git diff."""
    source = _store_with_history(tmp_path / "source.db")
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    trace_snapshot.export_snapshot(source, path=first)
    trace_snapshot.export_snapshot(source, path=second)
    source.close()
    assert first.read_bytes() == second.read_bytes()


def test_committed_snapshot_covers_both_runtimes() -> None:
    """The shipped file is what makes prod's Traces page prod's, not a stub."""
    rows = list(trace_snapshot.iter_snapshot())
    assert len(rows) > 1000
    assert {r["source"] for r in rows} <= {"eval", "serving"}
    versions = {
        json.loads(r["bundle"] or "{}").get("prompts_version")
        for r in rows
        if r.get("bundle")
    }
    assert "sdk_v1" in versions
    assert any(v and not v.startswith("sdk_") for v in versions)
