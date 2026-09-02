"""The eval-loop runner: one pass = one split × one version = one MLflow run.

Every conversation of the chosen reports runs end to end through the
four-agent pipeline — the agent's *own* earlier answers are its history, so a
wrong turn poisons the turns below it exactly as it would in production. Every
turn is scored against gold, given a cascade flag, written to the predictions
CSV, and recorded as a trace row stamped with ``run_id`` / ``split`` /
``question_id`` / ``model_version_id``. The MLflow run wraps the whole pass,
so a pass cannot happen without being recorded.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import pandas as pd

from convfinqa.config import EVAL_ROOT
from convfinqa.evalloop.splits import load_manifest, split_report_ids
from convfinqa.tracking import tracing

PREDICTIONS_DIR = EVAL_ROOT / "predictions" / "evalloop"

COLUMNS = [
    "report_id",
    "turn_index",
    "question_id",
    "question",
    "gold_answer",
    "pred_answer",
    "correct",
    "cascade",
    "first_wrong_turn",
    "pred_program",
    "gold_program",
    "gold_turn_type",
    "gold_conv_type",
    "pred_turn_type",
    "pred_conv_type",
    "pred_sub_questions",
    "history_text",
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
    "error",
    "trace_id",
    "run_id",
    "split",
    "model_version_id",
]


def first_wrong_index(oks: list[bool]) -> int | None:
    """Index of the first wrong turn, or None when every turn passed."""
    for i, ok in enumerate(oks):
        if not ok:
            return i
    return None


def examples_for(report_ids: list[str]) -> list[Any]:
    """ConvExamples for arbitrary pool reports, built the loader's own way."""
    from convfinqa.data.loader import _build_conv_examples, training_data

    return _build_conv_examples(list(report_ids), training_data())


async def _run_conversations(
    examples: list[Any],
    agents: dict[str, Any],
    concurrency: int,
    trace_tags: dict[str, Any] | None = None,
) -> list[tuple[Any, list[str], list[str], list[dict[str, Any]], str]]:
    from convfinqa.pipeline.runner import ConversationRunner

    runner = ConversationRunner()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(
        ex: Any,
    ) -> tuple[Any, list[str], list[str], list[dict[str, Any]], str]:
        captures: list[dict[str, Any]] = []
        error = ""
        async with sem:
            # One trace per conversation: report at the root, a child span per
            # question under it, the autologged agent/LLM spans under those.
            with tracing.span(
                ex.report_id,
                attributes={
                    "report_id": ex.report_id,
                    "n_questions": len(ex.questions),
                },
                trace_tags=trace_tags,
            ):
                try:
                    preds, programs = await runner.run_conversation(
                        ex.report_id, ex.questions, agents=agents, captures=captures
                    )
                except Exception as e:  # noqa: BLE001 — one bad conversation must not sink the pass
                    print(f"  [error] {ex.report_id}: {e!r}")  # noqa: T201
                    preds, programs, error = [], [], repr(e)
        return ex, preds, programs, captures, error

    return list(await asyncio.gather(*(one(ex) for ex in examples)))


async def run_split(
    split: str,
    version: str,
    *,
    n_reports: int | None = None,
    n_questions: int | None = None,
    concurrency: int = 8,
    environment: str = "dev",
) -> dict[str, Any]:
    """Run one split × version pass; return a summary with the CSV and run id."""
    import convfinqa.prompts as prompts_pkg
    from convfinqa.backends.pydantic import make_agents
    from convfinqa.evaluation.metrics import numeric_match
    from convfinqa.evaluation.runner import _capture_to_row_fields
    from convfinqa.tracking import mlflow_log, registry
    from convfinqa.tracking.bundle import bundle_fingerprint
    from convfinqa.tracking.comparator import program_accuracy
    from convfinqa.tracking.traces import TraceStore

    manifest = load_manifest()
    report_ids = split_report_ids(split, n_reports=n_reports, n_questions=n_questions)
    examples = examples_for(report_ids)
    n_questions = sum(len(ex.questions) for ex in examples)
    agents = make_agents(prompts_pkg.load(version))
    from convfinqa.tracking import prompt_ledger

    composition = prompt_ledger.composition_string(
        prompt_ledger.ensure(version)  # register any new prompt hashes first
    )
    fingerprint = bundle_fingerprint(version=version)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"evalloop-{split}{len(report_ids)}-{version}"
        f"·{composition.replace('.', '')}-{stamp}"
    )

    print(  # noqa: T201
        f"[{run_name}] {len(examples)} conversations, {n_questions} questions, "
        f"concurrency {concurrency}"
    )
    # Autolog before any conversation runs; conversations run *inside* the
    # MLflow run so every trace links to it in the Traces tab.
    tracing.enable()
    trace_tags = {
        "model_version_id": version,
        "composition": composition,
        "split": split,
        "run_name": run_name,
        "environment": environment,
    }

    with mlflow_log.run(
        run_name,
        kind="evalloop",
        version=version,
        params={
            "split": split,
            "manifest": manifest["name"],
            "n_reports": len(report_ids),
            "n_questions": n_questions,
            "concurrency": concurrency,
            "environment": environment,
        },
        tags={"split": split, "environment": environment, "loop": "evalloop"},
    ) as rec:
        run_id = str(getattr(rec, "run_id", ""))
        t0 = time.perf_counter()
        results = await _run_conversations(
            examples, agents, concurrency, trace_tags=trace_tags
        )
        wall = time.perf_counter() - t0
        store = TraceStore()
        rows: list[dict[str, Any]] = []
        for ex, preds, programs, captures, error in results:
            oks = [
                numeric_match(preds[i], g) if i < len(preds) else False
                for i, g in enumerate(ex.gold_answers)
            ]
            first_wrong = first_wrong_index(oks)
            n = len(ex.questions)
            gold_programs = ex.gold_programs or [""] * n
            gold_turn_types = ex.gold_turn_types or [""] * n
            gold_conv_types = ex.gold_conv_types or [""] * n
            for i, question in enumerate(ex.questions):
                pred = preds[i] if i < len(preds) else None
                prog = programs[i] if i < len(programs) else ""
                cap = captures[i] if i < len(captures) else {}
                question_id = f"{ex.report_id}_q{i}"
                trace_id = store.record(
                    report_id=ex.report_id,
                    turn_index=i,
                    question=question,
                    capture=cap if isinstance(cap, dict) else {},
                    answer=str(pred or ""),
                    program=prog,
                    source="eval",
                    gold_answer=str(ex.gold_answers[i]),
                    correct=oks[i],
                    bundle=fingerprint,
                    error=error if i == n - 1 else "",
                    run_id=run_id,
                    split=split,
                    question_id=question_id,
                    model_version_id=version,
                )
                rows.append(
                    {
                        "report_id": ex.report_id,
                        "turn_index": i,
                        "question_id": question_id,
                        "question": question,
                        "gold_answer": ex.gold_answers[i],
                        "pred_answer": pred,
                        "correct": oks[i],
                        "cascade": first_wrong is not None and i > first_wrong,
                        "first_wrong_turn": first_wrong,
                        "pred_program": prog,
                        "gold_program": gold_programs[i],
                        "gold_turn_type": gold_turn_types[i],
                        "gold_conv_type": gold_conv_types[i],
                        **_capture_to_row_fields(cap if isinstance(cap, dict) else {}),
                        "trace_id": trace_id,
                        "run_id": run_id,
                        "split": split,
                        "model_version_id": version,
                    }
                )
        store.close()

        df = pd.DataFrame(rows, columns=COLUMNS)
        from convfinqa.evalloop import stage_scores

        df = stage_scores.score_rows(df)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = PREDICTIONS_DIR / f"{run_name}.csv"
        df.to_csv(csv_path, index=False)

        accuracy = float(df["correct"].mean()) if len(df) else 0.0
        n_cascade = int(df["cascade"].sum())
        metrics: dict[str, float] = {
            "accuracy": round(accuracy, 6),
            "n_questions": float(len(df)),
            "n_conversations": float(len(examples)),
            "n_wrong": float(int((~df["correct"]).sum())),
            "n_cascade": float(n_cascade),
            "wall_seconds": round(wall, 2),
            "questions_per_minute": round(len(df) / wall * 60, 2) if wall else 0.0,
        }
        metrics.update(program_accuracy(df))
        metrics.update(stage_scores.run_metrics(df))
        for column in ("gold_turn_type", "gold_conv_type"):
            for value, group in df.groupby(column):
                label = str(value).strip().replace(" ", "_")
                if label and label.lower() != "nan":
                    metrics[f"accuracy_{column}_{label}"] = round(
                        float(group["correct"].mean()), 6
                    )
        rec.metrics(metrics)
        rec.artifact(csv_path)

    registry.register(version, source="evalloop", run_id=run_id)

    summary = {
        "run_name": run_name,
        "run_id": run_id,
        "csv": str(csv_path),
        "split": split,
        "version": version,
        "n_reports": len(report_ids),
        "n_questions": len(df),
        "accuracy": round(accuracy, 6),
        "n_cascade": n_cascade,
        "wall_seconds": round(wall, 2),
    }
    print(  # noqa: T201
        f"[{run_name}] accuracy {accuracy:.1%} on {len(df)} questions "
        f"({n_cascade} cascade) in {wall:.0f}s → {csv_path}"
    )
    return summary
