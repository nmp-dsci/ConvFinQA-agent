"""Production metrics, grouped by where the turn came from.

One rule shapes this whole module: **`serving`, `demo` and `eval` are never
blended.** A recorded turn replayed in about four seconds did not take four
seconds — it took the thirty-odd the recording cost, and the replay is paced for
watchability. An eval turn ran at concurrency 8 on a warm cache. A serving turn
is the only one whose latency is a latency anyone would experience. Averaging
the three produces a number that is true of nothing, which is exactly the kind of
dashboard figure this project exists to argue against.

So the response is keyed by source, every source is always present (an absent
group would read as "no data" when it means "no turns yet"), and each group
carries its own count so a reader can see how much weight its p95 deserves.

Read-only, therefore registered in demo mode too: the public demo showing its own
`demo`-source numbers, correctly labelled, is the honest version of a metrics
page — not one that hides them.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query

from convfinqa.error_codes import ALL_CODES, normalise
from convfinqa.tracking.traces import get_store

router = APIRouter(prefix="/metrics")

#: The sources the trace store writes, in the order a reader should meet them.
SOURCES: tuple[str, ...] = ("serving", "demo", "eval")

#: Hours in the sparkline series.
SERIES_HOURS = 24


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. `None` for an empty sample, never 0.0.

    Zero is a real latency and "no turns" is not, so they must not share a
    representation — a tile that prints `0 ms` for an empty store is lying in
    the one direction that looks like good news.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return round(ordered[rank - 1], 1)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _numbers(rows: list[dict[str, Any]], column: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _hour_key(created_at: Any) -> str:
    return str(created_at)[:13]


def _series(rows: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """One bucket per hour for the last `SERIES_HOURS`, oldest first.

    Every hour is emitted, including the empty ones — a sparkline that silently
    drops idle hours compresses time and turns a quiet night into a cliff.
    """
    by_hour: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_hour.setdefault(_hour_key(row.get("created_at")), []).append(row)

    buckets: list[dict[str, Any]] = []
    for offset in range(SERIES_HOURS - 1, -1, -1):
        stamp = (now - timedelta(hours=offset)).replace(
            minute=0, second=0, microsecond=0
        )
        hour_rows = by_hour.get(stamp.isoformat()[:13], [])
        buckets.append(
            {
                "hour": stamp.isoformat(),
                "n_turns": len(hour_rows),
                "n_errors": sum(1 for r in hour_rows if r.get("error")),
                "p50_latency_ms": _percentile(_numbers(hour_rows, "latency_ms"), 50),
                "cost_usd": round(sum(_numbers(hour_rows, "cost_usd")), 6),
            }
        )
    return buckets


def _group(source: str, rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    """Aggregate one source's turns. Shape is identical whether or not it has any."""
    latencies = _numbers(rows, "latency_ms")
    tokens = _numbers(rows, "total_tokens")
    costs = _numbers(rows, "cost_usd")

    scored = [r for r in rows if r.get("correct") is not None]
    n_correct = sum(1 for r in scored if int(r["correct"] or 0) == 1)

    failed = [r for r in rows if r.get("error")]
    by_code = Counter(normalise(str(r.get("error_code") or "")) for r in failed)

    # `created_at` is ISO-8601 UTC, so a lexicographic compare is a time compare.
    cutoff = (now - timedelta(hours=SERIES_HOURS)).isoformat()
    recent = [r for r in rows if str(r.get("created_at") or "") >= cutoff]

    return {
        "source": source,
        "n_turns": len(rows),
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": _mean(latencies),
            "n_measured": len(latencies),
        },
        "tokens_per_turn": {
            "p50": _percentile(tokens, 50),
            "mean": _mean(tokens),
            "total": int(sum(tokens)),
            "n_measured": len(tokens),
        },
        "cost_usd": {
            "per_turn": round(sum(costs) / len(costs), 6) if costs else None,
            "total": round(sum(costs), 6),
            "n_measured": len(costs),
        },
        "accuracy": {
            "accuracy": round(n_correct / len(scored), 6) if scored else None,
            "n_correct": n_correct,
            "n_scored": len(scored),
        },
        "errors": {
            "n_errors": len(failed),
            "error_rate": round(len(failed) / len(rows), 6) if rows else None,
            "by_code": {code: by_code.get(code, 0) for code in ALL_CODES},
        },
        "series": _series(recent, now),
    }


@router.get("/production")
async def production_metrics(
    limit: int = Query(default=50_000, ge=1, le=200_000),
) -> dict[str, Any]:
    """Turn counts, latency, tokens, cost, accuracy and errors — per source.

    An empty (or disabled) trace store is a valid state, not a failure: every
    group is returned with zero counts and `None` where a statistic has no
    sample, so the frontend renders one layout in every case.
    """
    now = datetime.now(timezone.utc)
    store = get_store()
    rows = store.metric_rows(limit=limit) if store is not None else []

    by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    for row in rows:
        by_source.setdefault(str(row.get("source") or "unknown"), []).append(row)

    return {
        "generated_at": now.isoformat(),
        "window_hours": SERIES_HOURS,
        "n_turns_total": len(rows),
        "trace_capture_enabled": store is not None,
        # Never blended. The three populations answer different questions and a
        # combined figure would answer none of them.
        "sources": {
            source: _group(source, group, now)
            for source, group in sorted(
                by_source.items(),
                key=lambda item: (
                    SOURCES.index(item[0]) if item[0] in SOURCES else len(SOURCES)
                ),
            )
        },
    }
