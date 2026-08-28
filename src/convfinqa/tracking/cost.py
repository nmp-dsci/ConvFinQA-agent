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

from typing import Any

# USD per 1M tokens. Update deliberately, and treat a change as a new bundle —
# past runs keep the numbers they were computed with.
PRICING: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek-v4-pro": (0.55, 2.19),
}

DEFAULT_MODEL = "deepseek-v4-flash"


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
