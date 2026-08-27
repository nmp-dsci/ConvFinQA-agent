"""Pydantic-evals harness: build dataset, run conversations, write predictions CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from convfinqa.backends.pydantic import make_agents
from convfinqa.config import PREDICTIONS_DIR
from convfinqa.data.schemas import ConversationHistory
from convfinqa.evaluation import load_cached_conversations, numeric_match
from convfinqa.evaluation.joining import write_joined_predictions
from convfinqa.evaluation.reporting import print_accuracy_table, write_predictions_html
from convfinqa.pipeline.prompts_loader import GEPA_NAME
from convfinqa.pipeline.runner import ConversationRunner, run_turn

EVAL_DIR = PREDICTIONS_DIR


class ConvInput(BaseModel):
    """Input to one conversation case."""

    report_id: str
    questions: list[str]


class ConvOutput(BaseModel):
    """Output of one conversation case: per-turn predictions in q_order."""

    preds: list[str]
    programs: list[str]
    stage_captures: list[dict[str, Any]] = Field(default_factory=list)


class TurnAccuracy(Evaluator[ConvInput, ConvOutput, list[str]]):
    """Mean per-turn correctness over a conversation, scored against gold metadata."""

    def evaluate(
        self, ctx: EvaluatorContext[ConvInput, ConvOutput, list[str]]
    ) -> float:
        """Score this conversation: fraction of turns where pred matches gold."""
        gold = ctx.metadata or []
        preds = ctx.output.preds if ctx.output else []
        if not gold:
            return 0.0
        return sum(
            numeric_match(preds[i], g) if i < len(preds) else False
            for i, g in enumerate(gold)
        ) / len(gold)


def _build_dataset(examples: list[Any]) -> Dataset[ConvInput, ConvOutput, list[str]]:
    cases = [
        Case(
            name=ex.report_id,
            inputs=ConvInput(report_id=ex.report_id, questions=list(ex.questions)),
            metadata=list(ex.gold_answers),
        )
        for ex in examples
    ]
    return Dataset(
        name=f"convfinqa-{GEPA_NAME}",
        cases=cases,
        evaluators=[TurnAccuracy()],
    )


_RUNNER = ConversationRunner()


async def _conv_task(case: ConvInput) -> ConvOutput:
    captures: list[dict[str, Any]] = []
    error = ""
    try:
        preds, programs = await _RUNNER.run_conversation(
            case.report_id, case.questions, captures=captures
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [error] {case.report_id}: {e!r}")  # noqa: T201
        preds, programs = [], []
        error = repr(e)
    if error and captures:
        captures[-1]["error"] = error
    elif error:
        captures.append({"error": error})
    return ConvOutput(preds=preds, programs=programs, stage_captures=captures)


def make_task_fn(agents: dict[str, Agent]):
    """Return a pydantic-evals task function that uses the given agents."""

    async def _task(case: ConvInput) -> ConvOutput:
        captures: list[dict[str, Any]] = []
        error = ""
        try:
            conversation = ConversationHistory()
            preds: list[str] = []
            programs: list[str] = []
            for question in case.questions:
                cap: dict[str, Any] = {}
                answer, program = await run_turn(
                    question, case.report_id, conversation, agents=agents, capture=cap
                )
                preds.append(answer)
                programs.append(program)
                captures.append(cap)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {case.report_id}: {e!r}")  # noqa: T201
            preds, programs = [], []
            error = repr(e)
        if error and captures:
            captures[-1]["error"] = error
        elif error:
            captures.append({"error": error})
        return ConvOutput(preds=preds, programs=programs, stage_captures=captures)

    return _task


_make_task_fn = make_task_fn


PREDICTIONS_COLUMNS = [
    "report_id",
    "turn_index",
    "question",
    "gold_answer",
    "pred_answer",
    "correct",
    "pred_program",
    "gold_program",
    "pred_turn_type",
    "gold_turn_type",
    "pred_conv_type",
    "gold_conv_type",
    "pred_sub_questions",
    "history_text",
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
    "error",
]


def _capture_to_row_fields(cap: dict[str, Any]) -> dict[str, str]:
    triage = cap.get("triage") or {}
    preprocess = cap.get("preprocess") or {}
    retriever = cap.get("retriever") or {}
    calculator = cap.get("calculator") or {}
    triage_out = triage.get("output", {}) if isinstance(triage, dict) else {}
    preprocess_out = preprocess.get("output", {}) if isinstance(preprocess, dict) else {}
    return {
        "pred_turn_type": str(triage_out.get("turn_type", "")),
        "pred_conv_type": str(triage_out.get("conv_type", "")),
        "pred_sub_questions": json.dumps(
            preprocess_out.get("sub_questions", []), default=str
        ),
        "history_text": str(cap.get("history_text", "")),
        "triage_io": json.dumps(triage, default=str) if triage else "",
        "preprocess_io": json.dumps(preprocess, default=str) if preprocess else "",
        "retriever_io": json.dumps(retriever, default=str) if retriever else "",
        "calculator_io": json.dumps(calculator, default=str) if calculator else "",
        "error": str(cap.get("error", "")),
    }


def _write_predictions_csv(
    report: Any,
    examples: list[Any],
    *,
    output_name: str = "pydantic_predictions.csv",
) -> Path:
    """Write a predictions CSV from the pydantic-evals report."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / output_name

    by_rid = {c.name: c for c in report.cases}

    n_correct = 0
    n_total = 0
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTIONS_COLUMNS)
        w.writeheader()
        for ex in examples:
            case = by_rid.get(ex.report_id)
            preds = case.output.preds if (case and case.output) else []
            programs = case.output.programs if (case and case.output) else []
            captures = case.output.stage_captures if (case and case.output) else []
            n = len(ex.questions)
            gold_programs = ex.gold_programs if ex.gold_programs else [""] * n
            gold_turn_types = ex.gold_turn_types if ex.gold_turn_types else [""] * n
            gold_conv_types = ex.gold_conv_types if ex.gold_conv_types else [""] * n
            for i, (q, g, gp, gtt, gct) in enumerate(
                zip(ex.questions, ex.gold_answers, gold_programs, gold_turn_types, gold_conv_types, strict=False)
            ):
                p = preds[i] if i < len(preds) else None
                prog = programs[i] if i < len(programs) else ""
                cap = captures[i] if i < len(captures) else {}
                ok = numeric_match(p, g) if p is not None else False
                row = {
                    "report_id": ex.report_id,
                    "turn_index": i,
                    "question": q,
                    "gold_answer": g,
                    "pred_answer": p,
                    "correct": ok,
                    "pred_program": prog,
                    "gold_program": gp,
                    "gold_turn_type": gtt,
                    "gold_conv_type": gct,
                    **_capture_to_row_fields(cap if isinstance(cap, dict) else {}),
                }
                w.writerow(row)
                n_correct += int(ok)
                n_total += 1

    overall = n_correct / n_total if n_total else 0.0
    print(f"\nOverall turn accuracy: {overall:.1%}  ({n_correct}/{n_total})")  # noqa: T201
    print(f"Wrote {out_path}")  # noqa: T201
    return out_path


def get_predictions_path() -> Path:
    """Return the canonical predictions artifact path."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR / "pydantic_predictions.csv"


async def evaluate(examples: list[Any], max_concurrency: int = 8) -> Path:
    """Run the runner over every conversation, write predictions CSV + HTML."""
    dataset = _build_dataset(examples)
    report = await dataset.evaluate(
        _conv_task, max_concurrency=max_concurrency, progress=True
    )
    report.print(
        include_input=False,
        include_output=False,
        include_metadata=False,
        include_durations=False,
    )
    csv_path = _write_predictions_csv(report, examples)
    write_predictions_html(csv_path)
    return csv_path


_REQUIRED_PRED_COLUMNS = {
    "pred_program",
    "gold_program",
    "gold_turn_type",
    "gold_conv_type",
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
}


async def evaluate_cached(
    examples: list[Any],
    *,
    max_concurrency: int = 8,
    reuse_existing: bool = True,
) -> Path:
    """Evaluate unless a cached predictions artifact already exists and is current."""
    out_path = get_predictions_path()
    html_path = out_path.with_suffix(".html")
    if reuse_existing and out_path.exists():
        existing = pd.read_csv(out_path)
        missing = _REQUIRED_PRED_COLUMNS - set(existing.columns)
        if not missing:
            print(f"Reusing existing {out_path}")  # noqa: T201
            if not html_path.exists():
                write_predictions_html(out_path)
            return out_path
        print(  # noqa: T201
            f"Required columns missing from {out_path} ({sorted(missing)}) — "
            "re-running evaluation…"
        )
    return await evaluate(examples, max_concurrency=max_concurrency)


async def evaluate_version(
    examples: list[Any],
    version: str,
    *,
    reuse: bool = True,
    max_concurrency: int = 8,
) -> Path:
    """Run evaluation for one prompt version with per-conversation caching."""
    import convfinqa.prompts as _pkg

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"pydantic_predictions_{version}.csv"
    csv_path = EVAL_DIR / csv_name

    if reuse:
        cached_df, cached_rids = load_cached_conversations(
            csv_path, examples, required_columns=_REQUIRED_PRED_COLUMNS
        )
    else:
        cached_df, cached_rids = pd.DataFrame(), set()

    to_run = [ex for ex in examples if ex.report_id not in cached_rids]
    n_cached = len(examples) - len(to_run)

    if n_cached:
        cached_q = int(cached_df["report_id"].isin(cached_rids).sum())
        print(  # noqa: T201
            f"\n[{version}] cache hit: {n_cached}/{len(examples)} conversations "
            f"({cached_q} questions) — skipping"
        )

    new_df = pd.DataFrame()
    if to_run:
        print(  # noqa: T201
            f"\n{'=' * 60}\nEvaluating {version} — {len(to_run)} conversations\n{'=' * 60}"
        )
        agents = make_agents(_pkg.load(version))
        dataset = _build_dataset(to_run)
        report = await dataset.evaluate(
            make_task_fn(agents), max_concurrency=max_concurrency, progress=True
        )
        report.print(
            include_input=False,
            include_output=False,
            include_metadata=False,
            include_durations=False,
        )
        _write_predictions_csv(report, to_run, output_name=csv_name)
        new_df = pd.read_csv(csv_path)

    if not cached_df.empty and cached_rids:
        cached_subset = cached_df[cached_df["report_id"].isin(cached_rids)]
        combined = pd.concat([cached_subset, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        total = len(combined)
        correct = int(combined["correct"].astype(str).str.lower().isin({"true", "1"}).sum())
        print(  # noqa: T201
            f"\n[{version}] combined accuracy: {correct / total:.1%}  ({correct}/{total} questions)"
        )

    write_predictions_html(
        csv_path,
        output_path=EVAL_DIR / f"pydantic_predictions_{version}.html",
    )
    write_joined_predictions(csv_path)
    return csv_path


_evaluate_version = evaluate_version


async def run_all_versions(reuse: bool = True) -> dict[str, Path]:
    """Evaluate every available prompt version; print accuracy table.

    Auto-discovers all `v\\d+(_\\d+)?` modules in `convfinqa.prompts` (v1, v2,
    v3_1, v3_2, …). When `settings.prompts_version` is set, that version is
    the focus — it is forced into the iteration set even if it wasn't in the
    auto-discovered list — but prior auto-discovered versions are kept so the
    comparison table still shows accuracy deltas. To evaluate a single version
    in isolation, set `PROMPTS_VERSION` AND remove the prior `.py` files (or
    rename them outside the regex).
    """
    import convfinqa.prompts as _prompts_pkg
    from convfinqa.config import settings
    from convfinqa.data.loader import load_conv_examples_test

    examples, _ = load_conv_examples_test()
    discovered = _prompts_pkg.latest_all()  # already sorted by (major, minor)
    pinned = settings.prompts_version
    if pinned and pinned not in discovered:
        # Pinned name didn't match the auto-discovery regex (e.g. tagged variant
        # like "v3_2_alt"). Insert it at the end so it still gets evaluated.
        versions = list(discovered) + [pinned]
    else:
        versions = list(discovered)
    paths: dict[str, Path] = {}
    for version in versions:
        paths[version] = await evaluate_version(examples, version, reuse=reuse)
    print_accuracy_table(paths)
    return paths


from convfinqa.evaluation.joining import (  # noqa: E402, F401
    compare_prediction_runs,
    compare_runs,
)
