"""Validate the running FastAPI server against the held-out evaluation set."""

# ruff: noqa: D103, T201

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from convfinqa.config import PREDICTIONS_DIR, settings
from convfinqa.data.loader import load_conv_examples_test, training_data
from convfinqa.data.schemas import ConvExample
from convfinqa.evaluation import (
    flush_csv_atomic,
    load_cached_conversations,
    numeric_match,
)
from convfinqa.evaluation.joining import analyze_predictions, join_predictions

GEPA_NAME = settings.gepa_name or "gepa_real_20260502_005251"
EVAL_DIR = PREDICTIONS_DIR
VERSION = settings.prompts_version or "v2"


def _build_conv_examples(report_ids: list[str], qa_data: pd.DataFrame) -> list[ConvExample]:
    examples: list[ConvExample] = []
    for rid in report_ids:
        group = qa_data[qa_data["report_id"] == rid].sort_values("q_order")
        examples.append(
            ConvExample(
                report_id=rid,
                questions=group["conv_questions"].tolist(),
                gold_answers=group["conv_answers"].tolist(),
                gold_programs=group["turn_program"].fillna("").tolist(),
                gold_turn_types=group["turn_type"].tolist(),
                gold_conv_types=group["qa_split"]
                .map({True: "Type II", False: "Type I"})
                .tolist(),
            )
        )
    return examples


def _local_load_conv_examples_test() -> tuple[list[ConvExample], pd.DataFrame]:
    """Independent sample matching `dspy_agent.py` for cross-runner consistency."""
    qa_data = training_data()
    qa_data = qa_data.query('data_key=="train"')

    sampled_report_ids = (
        qa_data["report_id"].drop_duplicates().sample(n=200, random_state=42).tolist()
    )
    qa_data = qa_data[qa_data["report_id"].isin(sampled_report_ids)].reset_index(drop=True)
    return _build_conv_examples(sampled_report_ids, qa_data), qa_data


def _analyze_predictions_local(predictions_path: Path, qa_data: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    qa["conv_type"] = qa["qa_split"].map({True: "Type II", False: "Type I"})
    joined = preds.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "conv_type"]],
        on=["report_id", "turn_index"],
        how="left",
    )
    out_path = predictions_path.with_name(f"{predictions_path.stem}_joined.csv")
    joined.to_csv(out_path, index=False)

    print("\nAccuracy by turn_type:")
    print(joined.groupby("turn_type")["correct"].mean().to_string())
    print("\nAccuracy by conv_type:")
    print(joined.groupby("conv_type")["correct"].mean().to_string())
    print("\nAccuracy by q_order:")
    print(joined.groupby("q_order")["correct"].mean().to_string())
    print(f"\nWrote {out_path}")
    return joined


def compare_model_accuracies(
    *,
    run_dir: Path,
    qa_data: pd.DataFrame,
    version: str = VERSION,
) -> Path | None:
    """Build a cross-model accuracy comparison for one prompt version."""
    sources = {
        "dspy": run_dir / f"dspy_predictions_{version}.csv",
        "pydantic": run_dir / f"pydantic_predictions_{version}.csv",
        "api": run_dir / f"api_predictions_{version}.csv",
    }
    fallbacks = {
        "dspy": run_dir / "predictions.csv",
        "pydantic": run_dir / "pydantic_predictions.csv",
        "api": run_dir / "api_predictions.csv",
    }
    sources = {
        label: path if path.exists() else fallbacks[label]
        for label, path in sources.items()
    }
    available = {
        label: join_predictions(path, qa_data)
        for label, path in sources.items()
        if path.exists()
    }
    if not available:
        return None

    rows: list[dict[str, Any]] = []

    overall: dict[str, Any] = {"slice": "overall", "value": "overall"}
    for label, frame in available.items():
        overall[f"{label}_acc"] = frame["correct"].mean()
    rows.append(overall)

    all_turn_types = sorted(
        set().union(*(set(frame["turn_type"].dropna().unique()) for frame in available.values()))
    )
    for turn_type in all_turn_types:
        row: dict[str, Any] = {"slice": "turn_type", "value": turn_type}
        for label, frame in available.items():
            cut = frame[frame["turn_type"] == turn_type]
            row[f"{label}_acc"] = cut["correct"].mean() if not cut.empty else None
        rows.append(row)

    all_q_orders = sorted(
        set().union(*(set(frame["q_order"].dropna().unique()) for frame in available.values()))
    )
    for q_order in all_q_orders:
        row = {"slice": "q_order", "value": q_order}
        for label, frame in available.items():
            cut = frame[frame["q_order"] == q_order]
            row[f"{label}_acc"] = cut["correct"].mean() if not cut.empty else None
        rows.append(row)

    out = pd.DataFrame(rows)
    ordered_cols = [
        "slice",
        "value",
        *[f"{label}_acc" for label in ("dspy", "pydantic", "api") if label in available],
    ]
    out = out[ordered_cols]
    out_path = run_dir / f"model_accuracy_comparison_{version}.csv"
    out.to_csv(out_path, index=False)

    printable = out.copy()
    for col in [c for c in printable.columns if c.endswith("_acc")]:
        printable[col] = printable[col].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else ""
        )
    print("\nModel accuracy comparison:")
    print(printable.to_string(index=False))
    print(f"\nWrote {out_path}")
    return out_path


async def _evaluate_conversation(
    client: httpx.AsyncClient,
    ex: Any,
) -> tuple[str, list[list[Any]]]:
    rows: list[list[Any]] = []
    session = await client.post("/sessions", json={"report_id": ex.report_id})
    session.raise_for_status()
    session_id = session.json()["session_id"]
    try:
        for i, (question, gold) in enumerate(
            zip(ex.questions, ex.gold_answers, strict=True)
        ):
            response = await client.post(
                f"/sessions/{session_id}/ask",
                json={"question": question},
            )
            response.raise_for_status()
            pred = response.json()["answer"]
            rows.append(
                [
                    ex.report_id,
                    i,
                    question,
                    gold,
                    pred,
                    numeric_match(pred, gold),
                ]
            )
    finally:
        response = await client.delete(f"/sessions/{session_id}")
        response.raise_for_status()
    return ex.report_id, rows


_API_CSV_COLUMNS = [
    "report_id",
    "turn_index",
    "question",
    "gold_answer",
    "pred_answer",
    "correct",
]


async def _evaluate_api_async(
    *,
    base_url: str,
    timeout: float,
    examples: list[Any],
    transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None,
    max_concurrency: int,
    out_path: Path,
    initial_rows: list[list[Any]],
) -> list[list[Any]]:
    """Score `examples` concurrently, flushing the CSV after each conversation."""
    semaphore = asyncio.Semaphore(max_concurrency)
    write_lock = asyncio.Lock()
    all_rows: list[list[Any]] = list(initial_rows)
    total = len(examples)
    completed = 0

    flush_csv_atomic(out_path, all_rows, _API_CSV_COLUMNS)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        transport=transport,
    ) as client:
        health = await client.get("/healthz")
        health.raise_for_status()

        async def run_one(ex: Any) -> str:
            nonlocal completed
            async with semaphore:
                report_id, conv_rows = await _evaluate_conversation(client, ex)
            async with write_lock:
                all_rows.extend(conv_rows)
                flush_csv_atomic(out_path, all_rows, _API_CSV_COLUMNS)
                completed += 1
                width = 24
                filled = int(width * completed / total) if total else width
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"[api_eval] [{bar}] {completed}/{total} "
                    f"report_id={report_id} ({completed / total:.1%})"
                )
            return report_id

        await asyncio.gather(*(run_one(ex) for ex in examples))

    return all_rows


def _load_cached_rows(
    out_path: Path, examples: list[Any]
) -> tuple[list[list[Any]], set[str]]:
    """Return (cached_rows, cached_rids) for fully-scored conversations."""
    df, cached_rids = load_cached_conversations(out_path, examples)
    if df.empty or not cached_rids:
        return [], cached_rids
    kept = df[df["report_id"].isin(cached_rids)]
    return kept[_API_CSV_COLUMNS].values.tolist(), cached_rids


def evaluate_api(
    *,
    base_url: str = "http://127.0.0.1:8765",
    timeout: float = 120.0,
    examples: list[Any] | None = None,
    transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    max_concurrency: int = 8,
    reuse_existing: bool = True,
) -> Path:
    """Score the API on conv_examples_test with per-conversation caching."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"api_predictions_{VERSION}.csv"

    qa_data: pd.DataFrame | None = None
    if examples is None:
        examples, qa_data = load_conv_examples_test()

    cached_rows, cached_rids = (
        _load_cached_rows(out_path, examples) if reuse_existing else ([], set())
    )
    to_run = [ex for ex in examples if ex.report_id not in cached_rids]
    n_cached = len(examples) - len(to_run)

    if n_cached:
        print(
            f"\n[api {VERSION}] cache hit: {n_cached}/{len(examples)} conversations "
            f"({len(cached_rows)} questions) — skipping"
        )

    if to_run:
        print(
            f"[api {VERSION}] running {len(to_run)} conversations "
            f"(max_concurrency={max_concurrency})"
        )
        asyncio.run(
            _evaluate_api_async(
                base_url=base_url,
                timeout=timeout,
                examples=to_run,
                transport=transport,
                max_concurrency=max_concurrency,
                out_path=out_path,
                initial_rows=cached_rows,
            )
        )
    elif not out_path.exists():
        flush_csv_atomic(out_path, [], _API_CSV_COLUMNS)

    print(f"\nWrote {out_path}")
    if qa_data is not None:
        analyze_predictions(out_path)
    return out_path


def compare_api_outputs(
    api_csv: Path,
    *,
    run_dir: Path | None = None,
) -> Path | None:
    run_dir = run_dir or EVAL_DIR
    _, qa_data = load_conv_examples_test()
    return compare_model_accuracies(run_dir=run_dir, qa_data=qa_data)


# Backwards-compat alias.
_join_predictions = join_predictions
