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

**Every aggregate here is all-time**, over every turn the store holds. It always
was; what was wrong until 2026-09-08 was the label. The response advertised
`window_hours: 24` and the UI printed "last 24 h" above figures computed from the
whole store — so the demo, whose committed traces are days old, showed "no turns
in the last 24 h" beside a count of eight thousand. The only thing that was ever
windowed is the sparkline, and it is now windowed to *the data*: buckets end at
the newest turn of that source and widen from hours to days to weeks so the
series always spans the run history rather than an arbitrary yesterday.
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

#: How many buckets a sparkline draws, whatever the bucket turns out to be.
SERIES_BUCKETS = 24

#: Bucket widths, narrowest first. The first one that covers a source's whole
#: history in `SERIES_BUCKETS` steps wins, so a busy afternoon reads hour by hour
#: and a month of eval runs reads day by day, in the same number of bars.
BUCKET_UNITS: tuple[tuple[str, timedelta], ...] = (
    ("hour", timedelta(hours=1)),
    ("day", timedelta(days=1)),
    ("week", timedelta(weeks=1)),
)


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


def _parse(created_at: Any) -> datetime | None:
    """A stored `created_at` as an aware UTC datetime, or None if unreadable."""
    try:
        stamp = datetime.fromisoformat(str(created_at))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _floor(stamp: datetime, unit: str) -> datetime:
    """Truncate to the start of its bucket, so bars line up with clock time."""
    stamp = stamp.replace(minute=0, second=0, microsecond=0)
    if unit == "hour":
        return stamp
    return stamp.replace(hour=0)  # `day` and `week` both start at midnight UTC


def _bucket_unit(span: timedelta) -> tuple[str, timedelta]:
    """The narrowest bucket that fits `span` into `SERIES_BUCKETS` bars."""
    for unit, step in BUCKET_UNITS:
        if span <= step * SERIES_BUCKETS:
            return unit, step
    return BUCKET_UNITS[-1]


def _series(
    rows: list[dict[str, Any]], now: datetime
) -> tuple[list[dict[str, Any]], str, datetime | None, datetime | None]:
    """`SERIES_BUCKETS` buckets ending at this source's newest turn, oldest first.

    Returns the buckets plus the bucket unit and the first/last turn, because a
    reader cannot interpret a sparkline without knowing what one bar is worth —
    and until 2026-09-08 the label said "24 h" regardless.

    Every bucket is emitted, including the empty ones: a series that silently
    drops idle buckets compresses time and turns a quiet night into a cliff.
    Anchoring the right-hand edge on the newest *turn* rather than on `now` is
    what makes this work in the demo, where every committed trace is days old and
    a series ending at `now` would be twenty-four measured zeros.
    """
    stamps = [s for s in (_parse(r.get("created_at")) for r in rows) if s is not None]
    if not stamps:
        # Shape is identical for a source that has served nothing, so the
        # frontend renders one layout: empty hourly buckets ending now.
        unit, step = BUCKET_UNITS[0]
        end = _floor(now, unit)
        first = last = None
    else:
        first, last = min(stamps), max(stamps)
        unit, step = _bucket_unit(last - first)
        end = _floor(last, unit)

    start = end - step * (SERIES_BUCKETS - 1)
    binned: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        stamp = _parse(row.get("created_at"))
        if stamp is None:
            continue
        index = int((stamp - start) // step)
        if 0 <= index < SERIES_BUCKETS:
            binned.setdefault(index, []).append(row)

    buckets: list[dict[str, Any]] = []
    for index in range(SERIES_BUCKETS):
        bucket_rows = binned.get(index, [])
        buckets.append(
            {
                # `hour` is the historical name of this field and stays one, so
                # a stored payload keeps parsing; `bucket` beside the series
                # says what a step actually is.
                "hour": (start + step * index).isoformat(),
                "n_turns": len(bucket_rows),
                "n_errors": sum(1 for r in bucket_rows if r.get("error")),
                "p50_latency_ms": _percentile(_numbers(bucket_rows, "latency_ms"), 50),
                "cost_usd": round(sum(_numbers(bucket_rows, "cost_usd")), 6),
            }
        )
    return buckets, unit, first, last


def _group(source: str, rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    """Aggregate one source's turns. Shape is identical whether or not it has any."""
    latencies = _numbers(rows, "latency_ms")
    tokens = _numbers(rows, "total_tokens")
    costs = _numbers(rows, "cost_usd")

    scored = [r for r in rows if r.get("correct") is not None]
    n_correct = sum(1 for r in scored if int(r["correct"] or 0) == 1)

    failed = [r for r in rows if r.get("error")]
    by_code = Counter(normalise(str(r.get("error_code") or "")) for r in failed)

    series, bucket, first_turn, last_turn = _series(rows, now)

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
        # All-time, like every figure above it — the series is bucketed, not
        # windowed, and the caller needs the unit to label a bar.
        "series": series,
        "series_bucket": bucket,
        "first_turn_at": first_turn.isoformat() if first_turn else None,
        "last_turn_at": last_turn.isoformat() if last_turn else None,
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
        # Not a window. Every aggregate below covers every turn the store holds;
        # the field is kept so a client can say so out loud rather than assume.
        "window": "all-time",
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
