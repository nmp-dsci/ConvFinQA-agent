"""Run-trace store: every turn the system answers, kept and browsable.

Logfire remains the deep external trace. This is the in-app view — the thing that
makes "why did it answer that" a question the product can answer about itself,
rather than one you answer by opening a vendor dashboard.

SQLite rather than JSONL: the trace viewer filters by report, by session, by
correctness and by bundle, and a scan-the-whole-file design stops being viable at
a few thousand turns. One file, no server, `check_same_thread=False` because
FastAPI serves from a threadpool, and a `WAL` journal so a read during a write
does not block.

What is stored per turn is exactly what `capture` already produced — the same
structure the eval CSVs carry — so a live turn and a scored turn are inspectable
through one code path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from convfinqa.config import TRACES_DIR, settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    trace_id      TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    source        TEXT NOT NULL,          -- 'serving' | 'eval' | 'demo'
    session_id    TEXT,
    report_id     TEXT NOT NULL,
    turn_index    INTEGER NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT,
    program       TEXT,
    gold_answer   TEXT,
    correct       INTEGER,                -- NULL when no gold exists
    bundle_id     TEXT,
    bundle        TEXT,                   -- JSON fingerprint
    latency_ms    REAL,
    total_tokens  INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    error         TEXT,
    error_code    TEXT,                   -- convfinqa.error_codes.ErrorCode
    capture       TEXT NOT NULL           -- JSON per-stage IO
);
CREATE INDEX IF NOT EXISTS idx_turns_report  ON turns(report_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_turns_bundle  ON turns(bundle_id);
"""

# Columns added after the table shipped. A dev machine already has a `traces.db`
# written by the original schema, and `CREATE TABLE IF NOT EXISTS` will not
# widen it — so widen it here rather than asking anyone to delete their history.
_ADDED_COLUMNS: dict[str, str] = {
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cost_usd": "REAL",
    "error_code": "TEXT",
}

_lock = threading.Lock()


def default_db_path() -> Path:
    """Path to the trace database."""
    return TRACES_DIR / "traces.db"


class TraceStore:
    """A SQLite-backed store of per-turn traces."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def _migrate(self) -> None:
        """Add any column this build knows about that the file predates."""
        existing = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(turns)").fetchall()
        }
        for column, sql_type in _ADDED_COLUMNS.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE turns ADD COLUMN {column} {sql_type}")

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with _lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def record(
        self,
        *,
        report_id: str,
        turn_index: int,
        question: str,
        capture: dict[str, Any],
        answer: str = "",
        program: str = "",
        source: str = "serving",
        session_id: str | None = None,
        gold_answer: str | None = None,
        correct: bool | None = None,
        bundle: dict[str, Any] | None = None,
        error: str = "",
        error_code: str = "",
    ) -> str:
        """Persist one turn; return its trace id.

        Never raises into the caller: a trace that fails to write must not fail
        the turn that produced it. The answer is the product; the trace is not.

        `error_code` is the stable classification (`convfinqa.error_codes`) and
        `error` the original message. Both are stored: the code is what a
        dashboard groups by, the message is what a human reads.
        """
        trace_id = uuid4().hex
        try:
            from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id
            from convfinqa.tracking.cost import turn_usage

            spec = bundle if bundle is not None else bundle_fingerprint()
            # A turn whose stages recorded nothing stores NULLs, not zeros: an
            # unmeasured turn must not be averaged in as a free, instant one.
            measured = has_stage_metrics(capture)
            latency, tokens = _rollup(capture) if measured else (None, None)
            usage = turn_usage(capture) if measured else {}
            with self._write() as conn:
                conn.execute(
                    """
                    INSERT INTO turns (
                        trace_id, created_at, source, session_id, report_id,
                        turn_index, question, answer, program, gold_answer,
                        correct, bundle_id, bundle, latency_ms, total_tokens,
                        input_tokens, output_tokens, cost_usd,
                        error, error_code, capture
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        trace_id,
                        datetime.now(timezone.utc).isoformat(),
                        source,
                        session_id,
                        report_id,
                        turn_index,
                        question,
                        answer,
                        program,
                        gold_answer,
                        None if correct is None else int(correct),
                        bundle_id(spec),
                        json.dumps(spec),
                        latency,
                        tokens,
                        int(usage["input_tokens"]) if usage else None,
                        int(usage["output_tokens"]) if usage else None,
                        usage["cost_usd"] if usage else None,
                        error,
                        error_code or "",
                        json.dumps(capture, default=str),
                    ),
                )
        except Exception:  # noqa: BLE001 — telemetry must never break serving
            return trace_id
        return trace_id

    def list_turns(
        self,
        *,
        report_id: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Trace summaries, newest first. Excludes the heavy `capture` blob."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("report_id", report_id),
            ("session_id", session_id),
            ("source", source),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, min(limit, 500)), max(0, offset)])
        rows = self._conn.execute(
            f"""
            SELECT trace_id, created_at, source, session_id, report_id, turn_index,
                   question, answer, program, gold_answer, correct, bundle_id,
                   latency_ms, total_tokens, cost_usd, error, error_code
            FROM turns {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def get_turn(self, trace_id: str) -> dict[str, Any] | None:
        """One trace with its full per-stage capture, or None."""
        row = self._conn.execute(
            "SELECT * FROM turns WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["capture"] = _loads(record.get("capture"), {})
        record["bundle"] = _loads(record.get("bundle"), {})
        return record

    def stats(self) -> dict[str, Any]:
        """Headline counts for the traces tab."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n_turns,
                   COUNT(DISTINCT report_id) AS n_reports,
                   AVG(latency_ms) AS avg_latency_ms,
                   SUM(COALESCE(total_tokens, 0)) AS total_tokens
            FROM turns
            """
        ).fetchone()
        return {
            "n_turns": int(row["n_turns"] or 0),
            "n_reports": int(row["n_reports"] or 0),
            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
            "total_tokens": int(row["total_tokens"] or 0),
        }

    def metric_rows(
        self, *, since: str | None = None, limit: int = 50_000
    ) -> list[dict[str, Any]]:
        """The narrow projection `/metrics/production` aggregates over.

        Deliberately not the whole row: `capture` and `bundle` are the two heavy
        columns and neither contributes to a headline number. Cost is read from
        the column rather than recomputed from the capture, so the aggregation
        stays a scan of scalars even when the store holds tens of thousands of
        turns.
        """
        clause = "WHERE created_at >= ?" if since else ""
        params: list[Any] = [since] if since else []
        params.append(max(1, limit))
        rows = self._conn.execute(
            f"""
            SELECT created_at, source, latency_ms, total_tokens, input_tokens,
                   output_tokens, cost_usd, correct, error, error_code
            FROM turns {clause}
            ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


def _loads(raw: Any, fallback: Any) -> Any:
    if not isinstance(raw, str) or not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def has_stage_metrics(capture: dict[str, Any]) -> bool:
    """True when at least one stage of this turn recorded any metrics.

    The difference between "this turn cost nothing" and "nobody wrote down what
    this turn cost" has to survive into the database, or a dashboard reads the
    second as the first. Artefacts recorded before per-stage metrics existed —
    the committed prediction CSVs, and the demo pack built from them — are
    exactly that second case, and they must land as NULL rather than as zero.
    """
    for stage in ("triage", "preprocess", "retriever", "calculator"):
        entry = capture.get(stage)
        if isinstance(entry, dict) and entry.get("metrics"):
            return True
    return False


def _rollup(capture: dict[str, Any]) -> tuple[float, int]:
    """Sum per-stage latency and tokens into turn-level totals."""
    latency = 0.0
    tokens = 0
    for stage in ("triage", "preprocess", "retriever", "calculator"):
        entry = capture.get(stage)
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        latency += float(metrics.get("latency_ms", 0.0) or 0.0)
        tokens += int(metrics.get("total_tokens", 0) or 0)
    return round(latency, 1), tokens


_store: TraceStore | None = None


def get_store() -> TraceStore | None:
    """The process-wide trace store, or None when trace capture is disabled."""
    global _store
    if not settings.trace_capture_enabled:
        return None
    if _store is None:
        _store = TraceStore()
    return _store


def reset_store() -> None:
    """Drop the cached store. For tests that point it at a temp directory."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
