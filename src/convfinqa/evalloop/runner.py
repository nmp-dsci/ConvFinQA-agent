"""The eval-loop runner: one pass = one split × one version = one MLflow run.

Every conversation of the chosen reports runs end to end through the
four-agent pipeline — the agent's *own* earlier answers are its history, so a
wrong turn poisons the turns below it exactly as it would in production. Every
turn is scored against gold, given a cascade flag, written to the predictions
CSV, and recorded as a trace row stamped with ``run_id`` / ``split`` /
``question_id`` / ``model_version_id``. The MLflow run wraps the whole pass,
so a pass cannot happen without being recorded.

Two runtimes walk a conversation: the four-agent pipeline (``runtime="pipeline"``,
a ``vN`` bundle) and the single-session Agent SDK runtime
(``runtime="agent_sdk"``, an ``sdk_vN`` prompt). They differ only in the
callable that produces predictions and captures; scoring, the CSV, the trace
rows and the metric panel are the same code, which is what makes their runs
comparable.

**A turn that was never answered is not a wrong answer.** When the Claude CLI
refuses (session limit, rate limit, no credit) the SDK runtime aborts that
conversation, and every turn from the refusal onwards is written `unscored`,
excluded from the accuracy numerator *and* denominator, tagged `incomplete` on
the MLflow run and refused by both gates. A pass that ends this way is
completed — never re-run from scratch, never stitched mid-conversation — with
``--resume-from``, which copies whole answered conversations through verbatim
and runs only the ones that are missing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from convfinqa.backends.agent_sdk import RATE_LIMIT_ERROR_PREFIX
from convfinqa.config import EVAL_ROOT
from convfinqa.evalloop.splits import draw_train, load_manifest, split_report_ids
from convfinqa.tracking import tracing

PREDICTIONS_DIR = EVAL_ROOT / "predictions" / "evalloop"

RUNTIMES = ("pipeline", "agent_sdk")

# One conversation, any runtime: (report_id, questions, captures=, stop_after=)
# -> (preds, programs). Both runtimes are bound to this shape before the pass
# starts, so `_run_conversations` never knows which one it is driving.
ConversationFn = Callable[..., Awaitable[tuple[list[str], list[str]]]]


class ConvOutcome(NamedTuple):
    """What one conversation produced, whatever ended it.

    `refusal` is the verbatim CLI refusal when the conversation was rate
    limited and empty otherwise; when it is set, `preds` covers only the turns
    that were actually answered and everything after them was never attempted.
    """

    example: Any
    preds: list[str]
    programs: list[str]
    captures: list[dict[str, Any]]
    error: str
    refusal: str = ""


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
    # Appended, trailing, 2026-09-05 — a CSV written before them still loads:
    # readers treat a missing `unscored` as all-False and a missing
    # `resumed_from_run_id` as "".
    #
    # `unscored`: the turn was never answered (the CLI refused, or the refusal
    # ended the conversation before this turn was asked). Excluded from
    # accuracy on both sides of the fraction; both gates refuse a CSV that has
    # any.
    "unscored",
    # `resumed_from_run_id`: non-empty on a row copied through by
    # `--resume-from`, naming the run that actually produced it. `run_id` and
    # `trace_id` stay the original ones, so a copied row's provenance is the
    # truth rather than the run that assembled the file.
    "resumed_from_run_id",
]

BOOL_COLUMNS = ("correct", "cascade", "unscored")


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


def pipeline_conversation_fn(agents: dict[str, Any]) -> ConversationFn:
    """`ConversationRunner.run_conversation` with the four agents bound."""
    from convfinqa.pipeline.runner import ConversationRunner

    runner = ConversationRunner()

    async def run(
        report_id: str, questions: list[str], **kw: Any
    ) -> tuple[list[str], list[str]]:
        return await runner.run_conversation(report_id, questions, agents=agents, **kw)

    return run


def sdk_conversation_fn(system_prompt: str, version: str) -> ConversationFn:
    """`backends.agent_sdk.run_conversation` with the session prompt bound."""
    from convfinqa.backends import agent_sdk

    async def run(
        report_id: str, questions: list[str], **kw: Any
    ) -> tuple[list[str], list[str]]:
        return await agent_sdk.run_conversation(
            report_id, questions, system_prompt=system_prompt, version=version, **kw
        )

    return run


async def _run_conversations(
    examples: list[Any],
    run_conversation: ConversationFn,
    concurrency: int,
    trace_tags: dict[str, Any] | None = None,
    stop_at_first_wrong: bool = False,
) -> list[ConvOutcome]:
    from convfinqa.backends.agent_sdk import SdkRateLimitError
    from convfinqa.evaluation.metrics import numeric_match

    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(ex: Any) -> ConvOutcome:
        captures: list[dict[str, Any]] = []
        error = ""
        refusal = ""
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
                stop_after = None
                if stop_at_first_wrong:

                    def stop_after(i: int, answer: str, _ex: Any = ex) -> bool:
                        return not numeric_match(answer, _ex.gold_answers[i])

                try:
                    preds, programs = await run_conversation(
                        ex.report_id,
                        ex.questions,
                        captures=captures,
                        stop_after=stop_after,
                    )
                except SdkRateLimitError as e:
                    # Not a failure of the runtime: the account cannot spend.
                    # The turns already answered are kept as themselves; the
                    # rest of this conversation was never attempted.
                    print(  # noqa: T201
                        f"  [rate limited] {ex.report_id} at q{e.turn_index}: "
                        f"{e.refusal}"
                    )
                    preds, programs, refusal = (
                        list(e.preds),
                        list(e.programs),
                        e.refusal,
                    )
                except Exception as e:  # noqa: BLE001 — one bad conversation must not sink the pass
                    print(f"  [error] {ex.report_id}: {e!r}")  # noqa: T201
                    preds, programs, error = [], [], repr(e)
        return ConvOutcome(ex, preds, programs, captures, error, refusal)

    return list(await asyncio.gather(*(one(ex) for ex in examples)))


# --- Resuming a pass that was cut short ---------------------------------------


def _flags(df: pd.DataFrame, column: str) -> pd.Series:
    """One column as booleans; a column that is not there is all-False.

    Trailing columns arrive over time, and for `unscored` "the column is
    absent" and "no row is unscored" are the same claim — which is what lets
    every CSV committed before it keep loading.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].astype(str).str.lower().isin({"true", "1"})


def load_prior_csv(path: Path | str) -> pd.DataFrame:
    """A prior pass's predictions CSV, with the boolean columns normalised.

    Not `gate.load_run_csv`: that one *refuses* a frame with unscored rows,
    which is exactly the frame a resume exists to read.
    """
    df = pd.read_csv(path)
    for column in BOOL_COLUMNS:
        df[column] = _flags(df, column)
    if "resumed_from_run_id" not in df.columns:
        df["resumed_from_run_id"] = ""
    # An empty cell reads back as NaN, and `str(nan)` is the string "nan",
    # which pandas then reads back as NaN again — a provenance column has to be
    # text or empty, never the float.
    df["resumed_from_run_id"] = df["resumed_from_run_id"].fillna("").astype(str)
    df["turn_index"] = df["turn_index"].astype(int)
    return df


def prior_runtime(df: pd.DataFrame, path: Path | str) -> str:
    """Which runtime wrote a predictions CSV.

    There is no `runtime` column (the committed CSVs predate the second
    runtime), so it is read from the run-name prefix the runner wrote the file
    under, and otherwise from the version's own lineage — `sdk_vN` cannot run
    anywhere but the session runtime.
    """
    import convfinqa.prompts as prompts_pkg

    if "runtime" in df.columns:
        named = sorted({str(v) for v in df["runtime"].dropna() if str(v).strip()})
        if len(named) == 1:
            return named[0]
    if Path(path).name.startswith("sdk-evalloop"):
        return "agent_sdk"
    versions = {str(v) for v in df.get("model_version_id", pd.Series(dtype=str))}
    if any(prompts_pkg.is_sdk_version(v) for v in versions):
        return "agent_sdk"
    return "pipeline"


def reusable_conversations(
    prior: pd.DataFrame, examples: list[Any]
) -> dict[str, pd.DataFrame]:
    """The prior rows of every conversation that can be copied through.

    Whole or nothing. A conversation qualifies only when the prior pass
    answered *every* one of its questions exactly once with no unscored and no
    rate-limited row. A half-finished conversation is re-run from turn 0: the
    premise of the session runtime is that a conversation is one session whose
    later turns depend on its earlier ones, so half a session's rows are not a
    valid prefix of a fresh session's.
    """
    expected = {str(ex.report_id): len(ex.questions) for ex in examples}
    out: dict[str, pd.DataFrame] = {}
    for report_id, group in prior.groupby("report_id"):
        n = expected.get(str(report_id))
        if n is None:
            continue
        rows = group.sort_values("turn_index")
        if list(rows["turn_index"].astype(int)) != list(range(n)):
            continue
        if bool(_flags(rows, "unscored").any()):
            continue
        errors = rows.get("error", pd.Series(dtype=str)).astype(str)
        if bool(errors.str.startswith(RATE_LIMIT_ERROR_PREFIX).any()):
            continue
        out[str(report_id)] = rows
    return out


def _reused_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """One conversation's prior rows, copied verbatim into the new frame.

    `run_id` and `trace_id` are the ones that produced the answer, not the run
    that assembled the file — a copied row must not claim to have been produced
    by a pass that never asked the question. `resumed_from_run_id` is what says
    it was copied, and a row copied twice keeps naming its original run.
    """

    def text(value: Any) -> str:
        """NaN is not a run id: an empty cell reads back as a float."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value)

    out: list[dict[str, Any]] = []
    for record in rows.to_dict("records"):
        row = {column: record.get(column) for column in COLUMNS}
        row["unscored"] = False
        row["resumed_from_run_id"] = text(record.get("resumed_from_run_id")) or text(
            record.get("run_id")
        )
        out.append(row)
    return out


def check_resume(
    prior: pd.DataFrame,
    path: Path | str,
    *,
    split: str,
    version: str,
    runtime: str,
    report_ids: list[str],
    train_seed: int | None,
) -> None:
    """Refuse a resume that would silently change the question set.

    Four ways it can: a different split, a different version, a different
    runtime, or a report set the current pass does not cover. `--train-seed`
    draws its own reports, so the check there is that every prior report is in
    *this* draw — a different draw is a different question set, and stitching
    the two would produce a CSV that is not any split.
    """
    name = Path(path).name
    prior_splits = sorted({str(v) for v in prior["split"].dropna()})
    if prior_splits != [split]:
        raise ValueError(
            f"--resume-from {name} is a {'/'.join(prior_splits) or 'unknown'} "
            f"run; this pass is {split!r}. A resume completes one pass, it does "
            "not merge two."
        )
    prior_versions = sorted({str(v) for v in prior["model_version_id"].dropna()})
    if prior_versions != [version]:
        raise ValueError(
            f"--resume-from {name} ran version "
            f"{'/'.join(prior_versions) or 'unknown'}; this pass is {version!r}. "
            "Rows from another prompt are not this prompt's evidence."
        )
    was = prior_runtime(prior, path)
    if was != runtime:
        raise ValueError(
            f"--resume-from {name} ran runtime {was!r}; this pass is "
            f"{runtime!r}. The two arms answer the same questions differently, "
            "which is the whole point of comparing them."
        )
    unknown = sorted({str(v) for v in prior["report_id"].dropna()} - set(report_ids))
    if unknown and train_seed is not None:
        raise ValueError(
            f"--resume-from {name} cannot be combined with --train-seed "
            f"{train_seed}: {len(unknown)} of its reports are not in that draw, "
            "so it is a different question set. Resume without --train-seed."
        )
    if unknown:
        sample = ", ".join(unknown[:3]) + ("…" if len(unknown) > 3 else "")
        raise ValueError(
            f"--resume-from {name} covers {len(unknown)} reports this pass does "
            f"not ({sample}): the pass must be a superset of the run it resumes."
        )


def sdk_run_metrics(captures: list[dict[str, Any]]) -> dict[str, float]:
    """Run-level metrics from the ``capture["sdk"]`` dicts of an sdk pass.

    Means are over the turns that produced a reply; counts and sums are over
    the whole pass. `sdk_cache_read_share` is the fraction of input tokens
    served from the prompt cache — in a session the report is sent once and
    read back on every later turn, so this is what the single-session design
    is supposed to buy.
    """
    rows = [
        c["sdk"]
        for c in captures
        if isinstance(c, dict) and isinstance(c.get("sdk"), dict)
    ]
    out: dict[str, float] = {
        "sdk_turns_answered": float(len(rows)),
        "sdk_stage_skips": float(sum(len(r.get("stage_skips") or []) for r in rows)),
        "sdk_inline_arithmetic": float(
            sum(1 for r in rows if r.get("inline_arithmetic"))
        ),
        "sdk_cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6),
    }
    if rows:
        turns = [float(r["num_turns"]) for r in rows if r.get("num_turns") is not None]
        if turns:
            out["sdk_turns_mean"] = round(sum(turns) / len(turns), 4)
        out["sdk_tool_calls_mean"] = round(
            sum(float(r.get("tool_calls") or 0) for r in rows) / len(rows), 4
        )
    fresh = sum(int(r.get("input_tokens") or 0) for r in rows)
    cached = sum(int(r.get("cache_read_input_tokens") or 0) for r in rows)
    if fresh + cached:
        out["sdk_cache_read_share"] = round(cached / (fresh + cached), 6)
    return out


async def run_split(
    split: str,
    version: str,
    *,
    n_reports: int | None = None,
    n_questions: int | None = None,
    concurrency: int = 8,
    environment: str = "dev",
    train_seed: int | None = None,
    stop_at_first_wrong: bool = False,
    campaign: str | None = None,
    label: str | None = None,
    runtime: str = "pipeline",
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run one split × version pass; return a summary with the CSV and run id.

    `train_seed` replaces the manifest's train list with a fresh stratified draw
    from ``pool − gate`` — resampling train every cycle is what stops the teacher
    from overfitting to one set of conversations, and the seed plus the drawn ids
    are logged so any cycle can be recreated exactly. It is refused for any other
    split: the gate must never move.

    `stop_at_first_wrong` ends each conversation at its first wrong answer. Only
    signal-bearing on train, and refused on the gate, where the comparison is
    paired per question.

    `runtime` picks who walks the conversations: ``pipeline`` (a ``vN`` bundle,
    four agents) or ``agent_sdk`` (an ``sdk_vN`` prompt, one Claude session).
    Version names and runtimes are checked against each other, because a
    bundle cannot run in a session and a session prompt cannot build agents.

    `resume_from` completes a pass that was cut short — by a rate limit, or by
    anything else that left rows unscored. Conversations the prior CSV answered
    *whole* are copied through verbatim (their own `run_id` and `trace_id`
    kept, `resumed_from_run_id` set); every other conversation of the split is
    run again from turn 0. Split, version, runtime and the report set are
    checked against the prior CSV first, because a resume that quietly changed
    the question set would produce a file that is not any split's evidence.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evaluation.metrics import numeric_match
    from convfinqa.evaluation.runner import _capture_to_row_fields
    from convfinqa.tracking import mlflow_log, registry
    from convfinqa.tracking.bundle import bundle_fingerprint
    from convfinqa.tracking.comparator import program_accuracy
    from convfinqa.tracking.traces import TraceStore

    manifest = load_manifest()
    if train_seed is not None and split != "train":
        raise ValueError(
            f"--train-seed draws a fresh train split; refusing on {split!r}, "
            "whose whole value is that it does not move"
        )
    if stop_at_first_wrong and split != "train":
        raise ValueError(
            f"--stop-at-first-wrong is refused on {split!r}: the gate compares "
            "question by question, and a run that stops early leaves the turns "
            "it skipped with no counterpart"
        )
    draw: dict[str, Any] | None = None
    if train_seed is not None:
        report_ids, draw = draw_train(
            seed=train_seed, n_reports=n_reports or len(manifest["splits"]["train"])
        )
    else:
        report_ids = split_report_ids(
            split, n_reports=n_reports, n_questions=n_questions
        )
    if runtime not in RUNTIMES:
        raise ValueError(f"unknown runtime {runtime!r}; expected one of {RUNTIMES}")
    if prompts_pkg.is_sdk_version(version) != (runtime == "agent_sdk"):
        raise ValueError(
            f"version {version!r} does not belong to runtime {runtime!r}: "
            "sdk_vN prompts run under --runtime agent_sdk, vN bundles under pipeline"
        )
    examples = examples_for(report_ids)
    n_questions = sum(len(ex.questions) for ex in examples)

    reused: dict[str, pd.DataFrame] = {}
    if resume_from is not None:
        prior = load_prior_csv(resume_from)
        check_resume(
            prior,
            resume_from,
            split=split,
            version=version,
            runtime=runtime,
            report_ids=report_ids,
            train_seed=train_seed,
        )
        reused = reusable_conversations(prior, examples)
    to_run = [ex for ex in examples if str(ex.report_id) not in reused]
    n_reused_questions = sum(len(rows) for rows in reused.values())

    from convfinqa.tracking import prompt_ledger

    runtime_params: dict[str, Any] = {"runtime": runtime}
    if runtime == "agent_sdk":
        from convfinqa.config import settings
        from convfinqa.llm import sdk_model_name

        system_prompt = prompts_pkg.load_sdk(version)
        composition = prompt_ledger.sdk_composition_string(
            prompt_ledger.ensure_sdk(version)  # register the prompt hash first
        )
        run_conversation = sdk_conversation_fn(system_prompt, version)
        runtime_params.update(
            {
                "sdk_model": sdk_model_name(),
                "billing": settings.sdk_billing,
                "max_turns": settings.sdk_max_turns,
                "sdk_total_tokens_limit": settings.sdk_total_tokens_limit,
            }
        )
        prefix = "sdk-evalloop"
    else:
        from convfinqa.backends.pydantic import make_agents

        composition = prompt_ledger.composition_string(
            prompt_ledger.ensure(version)  # register any new prompt hashes first
        )
        run_conversation = pipeline_conversation_fn(
            make_agents(prompts_pkg.load(version))
        )
        prefix = "evalloop"
    fingerprint = bundle_fingerprint(version=version)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{prefix}-{split}{len(report_ids)}-{version}"
        f"·{composition.replace('.', '')}-{stamp}"
    )

    print(  # noqa: T201
        f"[{run_name}] {len(examples)} conversations, {n_questions} questions, "
        f"concurrency {concurrency}"
    )
    if resume_from is not None:
        print(  # noqa: T201
            f"[{run_name}] resuming {Path(resume_from).name}: "
            f"{len(reused)} conversations ({n_reused_questions} questions) reused "
            f"whole, {len(to_run)} run again from turn 0"
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
        "runtime": runtime,
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
            "stop_at_first_wrong": stop_at_first_wrong,
            **runtime_params,
            **({"train_draw_seed": train_seed} if train_seed is not None else {}),
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
            **(
                {
                    "resumed_from": Path(resume_from).name,
                    "n_reused_conversations": len(reused),
                }
                if resume_from is not None
                else {}
            ),
        },
        tags={
            "split": split,
            "environment": environment,
            "loop": "evalloop",
            "runtime": runtime,
            **({"campaign": campaign} if campaign else {}),
        },
    ) as rec:
        if draw is not None:
            rec.dict_artifact("train_draw.json", {**draw, "report_ids": report_ids})
        run_id = str(getattr(rec, "run_id", ""))
        t0 = time.perf_counter()
        results = await _run_conversations(
            to_run,
            run_conversation,
            concurrency,
            trace_tags=trace_tags,
            stop_at_first_wrong=stop_at_first_wrong,
        )
        wall = time.perf_counter() - t0
        store = TraceStore()
        rows_by_report: dict[str, list[dict[str, Any]]] = {}
        skipped = 0
        n_rate_limited = 0
        for ex, preds, programs, captures, error, refusal in results:
            oks = [
                numeric_match(preds[i], g) if i < len(preds) else False
                for i, g in enumerate(ex.gold_answers)
            ]
            # An early-stopped conversation has no predictions for the turns it
            # never attempted. Writing them as wrong would be a lie about a turn
            # that was not run, so the rows stop where the run stopped and the
            # skipped count is reported separately.
            #
            # A refused conversation is the other shape of the same truth: the
            # rows are written (the gate needs to see that the question exists
            # and was not answered) but marked `unscored`, and `answered` is
            # where the refusal landed, so nothing past it is scored.
            answered = len(preds) if refusal else len(ex.questions)
            n = (
                len(preds)
                if stop_at_first_wrong and preds and not refusal
                else len(ex.questions)
            )
            skipped += len(ex.questions) - n
            first_wrong = first_wrong_index(oks[: min(answered, n)])
            gold_programs = ex.gold_programs or [""] * n
            gold_turn_types = ex.gold_turn_types or [""] * n
            gold_conv_types = ex.gold_conv_types or [""] * n
            conv_rows: list[dict[str, Any]] = []
            for i, question in enumerate(ex.questions[:n]):
                unscored = bool(refusal) and i >= answered
                pred = "" if unscored else (preds[i] if i < len(preds) else None)
                prog = "" if unscored else (programs[i] if i < len(programs) else "")
                cap = captures[i] if i < len(captures) else {}
                cap = cap if isinstance(cap, dict) else {}
                fields = _capture_to_row_fields(cap)
                if unscored:
                    if cap.get("rate_limited"):
                        n_rate_limited += 1
                    if not str(fields.get("error") or "").startswith(
                        RATE_LIMIT_ERROR_PREFIX
                    ):
                        fields["error"] = (
                            f"{RATE_LIMIT_ERROR_PREFIX}{refusal} — turn not "
                            f"attempted: the session was refused at q{answered}"
                        )
                question_id = f"{ex.report_id}_q{i}"
                trace_id = store.record(
                    report_id=ex.report_id,
                    turn_index=i,
                    question=question,
                    capture=cap,
                    answer=str(pred or ""),
                    program=prog,
                    source="eval",
                    gold_answer=str(ex.gold_answers[i]),
                    correct=bool(oks[i]) and not unscored,
                    bundle=fingerprint,
                    error=str(fields.get("error") or "")
                    or (error if i == n - 1 else ""),
                    run_id=run_id,
                    split=split,
                    question_id=question_id,
                    model_version_id=version,
                )
                conv_rows.append(
                    {
                        "report_id": ex.report_id,
                        "turn_index": i,
                        "question_id": question_id,
                        "question": question,
                        "gold_answer": ex.gold_answers[i],
                        "pred_answer": pred,
                        # An unscored row's `correct` is False only because the
                        # column is a boolean; `unscored` is what readers must
                        # honour, and the gates refuse a frame that has any.
                        "correct": bool(oks[i]) and not unscored,
                        "cascade": (
                            not unscored and first_wrong is not None and i > first_wrong
                        ),
                        "first_wrong_turn": first_wrong,
                        "pred_program": prog,
                        "gold_program": gold_programs[i],
                        "gold_turn_type": gold_turn_types[i],
                        "gold_conv_type": gold_conv_types[i],
                        **fields,
                        "trace_id": trace_id,
                        "run_id": run_id,
                        "split": split,
                        "model_version_id": version,
                        "unscored": unscored,
                        "resumed_from_run_id": "",
                    }
                )
            rows_by_report[str(ex.report_id)] = conv_rows
        store.close()

        # Report/turn order, whatever produced each conversation: the file must
        # read the same whether a row was answered in this pass or copied in.
        rows: list[dict[str, Any]] = []
        for ex in examples:
            report_id = str(ex.report_id)
            if report_id in reused:
                rows.extend(_reused_rows(reused[report_id]))
            else:
                rows.extend(rows_by_report.get(report_id, []))

        df = pd.DataFrame(rows, columns=COLUMNS)
        for column in BOOL_COLUMNS:
            df[column] = df[column].fillna(False).astype(bool)
        from convfinqa.evalloop import stage_scores

        # The whole frame, reused rows included: the panel is computed by this
        # run's scorer over every row it publishes, never inherited.
        df = stage_scores.score_rows(df)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = PREDICTIONS_DIR / f"{run_name}.csv"
        df.to_csv(csv_path, index=False)

        # Accuracy is over the turns that were actually answered. An unscored
        # turn is absent from both the numerator and the denominator — counting
        # a refusal as a wrong answer is how a live pass reported 44.4% for
        # "half the turns were never asked".
        scored = df[~df["unscored"]].copy()
        n_unscored = int(df["unscored"].sum())
        complete = n_unscored == 0
        accuracy = float(scored["correct"].mean()) if len(scored) else 0.0
        n_cascade = int(scored["cascade"].sum())
        metrics: dict[str, float] = {
            "accuracy": round(accuracy, 6),
            "n_questions": float(len(df)),
            "n_scored": float(len(scored)),
            "n_unscored": float(n_unscored),
            "n_rate_limited": float(n_rate_limited),
            "complete": 1.0 if complete else 0.0,
            "n_conversations": float(len(examples)),
            "n_wrong": float(int((~scored["correct"]).sum())),
            "n_cascade": float(n_cascade),
            "n_turns_skipped": float(skipped),
            "wall_seconds": round(wall, 2),
            "questions_per_minute": round(len(df) / wall * 60, 2) if wall else 0.0,
            **(
                {"n_reused_questions": float(n_reused_questions)}
                if resume_from is not None
                else {}
            ),
        }
        metrics.update(program_accuracy(scored))
        metrics.update(stage_scores.run_metrics(scored))
        if runtime == "agent_sdk":
            metrics.update(
                sdk_run_metrics([c for out in results for c in out.captures])
            )
        for column in ("gold_turn_type", "gold_conv_type"):
            for value, group in scored.groupby(column):
                label = str(value).strip().replace(" ", "_")
                if label and label.lower() != "nan":
                    metrics[f"accuracy_{column}_{label}"] = round(
                        float(group["correct"].mean()), 6
                    )
        rec.metrics(metrics)
        rec.artifact(csv_path)
        if not complete:
            # A partial pass must be impossible to mistake for a complete one,
            # in the store as well as on the terminal.
            rec.tag("incomplete", "true")
            rec.param("unscored_rows", n_unscored)

    registry.register(version, source="evalloop", run_id=run_id)

    summary = {
        "run_name": run_name,
        "run_id": run_id,
        "csv": str(csv_path),
        "split": split,
        "version": version,
        "runtime": runtime,
        "n_reports": len(report_ids),
        "n_questions": len(df),
        "n_scored": int(len(scored)),
        "accuracy": round(accuracy, 6),
        "n_cascade": n_cascade,
        "n_turns_skipped": skipped,
        # Read these two before the accuracy. `complete=False` means the
        # accuracy describes a subset of the split, and the pass cannot be
        # gated until `--resume-from` finishes it.
        "complete": complete,
        "n_unscored": n_unscored,
        "n_rate_limited": n_rate_limited,
        "wall_seconds": round(wall, 2),
        **({"train_draw": draw} if draw else {}),
        **(
            {
                "resumed_from": str(resume_from),
                "n_reused_conversations": len(reused),
                "n_reused_questions": n_reused_questions,
            }
            if resume_from is not None
            else {}
        ),
    }
    print(  # noqa: T201
        f"[{run_name}] accuracy {accuracy:.1%} on {len(scored)} scored questions "
        f"({n_cascade} cascade) in {wall:.0f}s → {csv_path}"
    )
    if not complete:
        print(  # noqa: T201
            f"[{run_name}] INCOMPLETE: {n_unscored} of {len(df)} turns were never "
            f"answered ({n_rate_limited} refused outright); the accuracy above is "
            f"over the {len(scored)} scored turns only and this pass CANNOT be "
            f"gated. Finish it with:\n"
            f"  convfinqa-evalloop run --split {split} --version {version} "
            f"--runtime {runtime} --resume-from {csv_path}"
        )
    return summary
