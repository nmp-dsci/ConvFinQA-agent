"""Token and cost accounting.

Cost-per-conversation is the number that decides whether a pipeline shape is
affordable at scale, and it is not derivable from accuracy. A four-agent
pipeline that gains two points for triple the tokens is a different proposition
from one that gains two points for free, and without this the two look
identical in the experiment table.

Prices are per million tokens, published DeepSeek rates. They are declared here
rather than fetched so a historical run stays reproducible: re-scoring a
year-old run must not silently reprice it.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

# USD per 1M tokens. Update deliberately, and treat a change as a new bundle —
# past runs keep the numbers they were computed with.
PRICING: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek-v4-pro": (0.55, 2.19),
}

DEFAULT_MODEL = "deepseek-v4-flash"

# The open runtime (s25) is priced on DeepSeek's current three-way card — a
# cache hit, a cache miss and output — because a session re-reads its whole
# context on every call and most of that is served from DeepSeek's cache, at a
# fiftieth of the miss price. Peak rates, USD per 1M tokens, published on
# api-docs.deepseek.com/quick_start/pricing (read 2026-10-25); off-peak is half.
DEEPSEEK_CARD: dict[str, tuple[float, float, float]] = {
    # model: (cache hit, cache miss, output)
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}


def deepseek_peak(at: datetime) -> bool:
    """DeepSeek's peak window: 01:00–04:00 and 06:00–10:00 UTC, Monday to Friday."""
    utc = at.astimezone(UTC)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


def deepseek_cost_usd(
    model: str,
    *,
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
    at: datetime,
) -> float:
    """What one open-runtime call cost, on the card in force at `at`.

    A student (s28) is served by vLLM on a Modal GPU, which bills by the second,
    not by the token: its calls cost nothing here, and the GPU time is costed
    from the server's own wall clock instead of being priced as DeepSeek.
    """
    if model.startswith("student-"):
        return 0.0
    hit, miss, out = DEEPSEEK_CARD.get(model, DEEPSEEK_CARD["deepseek-flash"])
    scale = 1.0 if deepseek_peak(at) else 0.5
    return (
        scale
        * (cache_hit_tokens * hit + cache_miss_tokens * miss + output_tokens * out)
        / 1_000_000
    )


def price_for(model: str) -> tuple[float, float]:
    """Input/output price per million tokens, falling back to the mini model."""
    return PRICING.get(model, PRICING[DEFAULT_MODEL])


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str = DEFAULT_MODEL,
) -> float:
    """Cost of one call in USD."""
    price_in, price_out = price_for(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def turn_usage(capture: dict[str, Any]) -> dict[str, float]:
    """Roll a turn's per-stage metrics into totals, including cost.

    Reads the `metrics` dict `pipeline.runner` already records per stage, so
    nothing new has to be measured — the numbers were being collected and
    thrown away.
    """
    input_tokens = 0
    output_tokens = 0
    latency_ms = 0.0
    stages = 0
    for stage in ("triage", "preprocess", "retriever", "calculator"):
        entry = capture.get(stage)
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        stages += 1
        input_tokens += int(metrics.get("input_tokens", 0) or 0)
        output_tokens += int(metrics.get("output_tokens", 0) or 0)
        latency_ms += float(metrics.get("latency_ms", 0.0) or 0.0)

    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(input_tokens + output_tokens),
        "latency_ms": round(latency_ms, 1),
        "n_stages": float(stages),
        "cost_usd": round(cost_usd(input_tokens, output_tokens), 6),
    }


def aggregate(captures: list[dict[str, Any]]) -> dict[str, float]:
    """Sum usage across many turns, and derive the per-turn average cost."""
    totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "total_tokens": 0.0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }
    for capture in captures:
        usage = turn_usage(capture)
        for key in totals:
            totals[key] += usage[key]
    n = max(1, len(captures))
    totals["n_turns"] = float(len(captures))
    totals["cost_usd_per_turn"] = round(totals["cost_usd"] / n, 6)
    totals["latency_ms_per_turn"] = round(totals["latency_ms"] / n, 1)
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return totals


# --- Agent SDK passes ---------------------------------------------------------
#
# An sdk pass is priced by the Claude CLI itself (`ResultMessage.total_cost_usd`),
# with two corrections, both found on 2026-10-15 (s19):
#
# 1. The figure is the *session's running total*, not the turn's, so a
#    conversation costs its last turn's figure. `sdk_cost_usd` summed every
#    turn's and overstated sdk_v1's test pass 2.3x: $27.62 recorded against
#    $11.83 spent. It stays on the runs that logged it, under its own name.
# 2. The CLI bundled with the SDK prices a model it does not know (it prints
#    `unrecognized_model`) on a fallback card: $5 / $25 per MTok, 1-hour cache
#    writes at 2x and reads at 0.1x, reproduced to the micro-dollar on a Haiku
#    5.5 turn. Every part of the bill scales with the input rate, so the real
#    figure is the CLI's times (real input rate / fallback input rate). That
#    holds while each prompt stays at or under 100K tokens, the edge of Haiku
#    5.5's first rate card. The table is only true of the bundled CLI it was
#    measured on: `tests/test_sdk_cost.py` pins the SDK version, so an upgrade
#    that teaches the CLI the model fails a test instead of dividing twice.
SDK_CLI_FALLBACK_INPUT_PER_MTOK = 5.0
SDK_UNPRICED_INPUT_PER_MTOK: dict[str, float] = {
    # model the bundled CLI does not price: its published input rate (≤100K prompt)
    "claude-haiku-5-5": 0.10,
}


def sdk_cli_cost_factor(model: str | None) -> float:
    """What the CLI's figure for `model` is multiplied by to be the price paid."""
    rate = SDK_UNPRICED_INPUT_PER_MTOK.get(model or "")
    return 1.0 if rate is None else rate / SDK_CLI_FALLBACK_INPUT_PER_MTOK


def sdk_session_cost_usd(
    running_totals: Iterable[Iterable[float | None]], model: str | None
) -> float:
    """What an sdk pass cost: each conversation's last running total, repriced.

    `running_totals` holds one sequence per conversation — the CLI's
    `total_cost_usd` after each turn, `None` for a turn with no reply. The
    largest is the session's total (the figure only grows).
    """
    spent = 0.0
    for totals in running_totals:
        seen = [float(t) for t in totals if t is not None]
        if seen:
            spent += max(seen)
    return round(spent * sdk_cli_cost_factor(model), 6)
