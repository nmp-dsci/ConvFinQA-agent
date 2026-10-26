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
from convfinqa.tracking import cost, tracing

PREDICTIONS_DIR = EVAL_ROOT / "predictions" / "evalloop"

RUNTIMES = ("pipeline", "agent_sdk", "open_agent")
# The runtimes that answer a conversation in one session from an `sdk_vN`
# prompt: Claude's harness, and the same contract on an open model (s25).
SESSION_RUNTIMES = ("agent_sdk", "open_agent")

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


def sdk_conversation_fn(
    system_prompt: str,
    version: str,
    model: str | None = None,
    effort: str | None = None,
) -> ConversationFn:
    """`backends.agent_sdk.run_conversation` with the prompt, model and effort bound.

    `model` is the one thing a model-swap pass changes: the prompt, the tools,
    the split and the scoring are identical, so two passes that differ only
    here are a paired comparison of the models. `effort` is bound beside it
    because it is part of what the agent is, and two passes at different
    efforts are not a comparison of the models.
    """
    from convfinqa.backends import agent_sdk

    async def run(
        report_id: str, questions: list[str], **kw: Any
    ) -> tuple[list[str], list[str]]:
        return await agent_sdk.run_conversation(
            report_id,
            questions,
            system_prompt=system_prompt,
            version=version,
            model=model,
            effort=effort,
            **kw,
        )

    return run


class OpenConversationFn:
    """`backends.open_agent.run_conversation` with its agent bound.

    A class rather than a closure so the pass can collect what every
    conversation's history held — `trajectories`, report id to the full
    message list — and log it once as the run's trajectory artifact.
    """

    def __init__(
        self,
        system_prompt: str,
        version: str,
        model: str,
        effort: str,
        harness: str = "pydantic_ai",
        temperature: float | None = None,
    ):
        self.system_prompt = system_prompt
        self.version = version
        self.model = model
        self.effort = effort
        self.harness = harness
        self.temperature = temperature
        self.trajectories: dict[str, Any] = {}

    async def __call__(
        self, report_id: str, questions: list[str], **kw: Any
    ) -> tuple[list[str], list[str]]:
        """Walk one conversation; its history lands in `self.trajectories`."""
        from convfinqa.backends import open_agent

        return await open_agent.run_conversation(
            report_id,
            questions,
            system_prompt=self.system_prompt,
            version=self.version,
            model=self.model,
            effort=self.effort,
            trajectories=self.trajectories,
            harness=self.harness,
            temperature=self.temperature,
            **kw,
        )


def _log_trajectories(rec: Any, run_name: str, trajectories: dict[str, Any]) -> None:
    """Log each conversation's full message history as one gzipped JSONL artifact.

    The lossless record a later fine-tune trains on: reasoning, tool calls,
    tool results and replies, in Pydantic AI's own message schema. It goes to
    the run, not to git — it is dev data, like the traces.
    """
    import gzip
    import json
    import tempfile

    if not trajectories:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trajectories.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for report_id, messages in trajectories.items():
                fh.write(
                    json.dumps(
                        {
                            "run_name": run_name,
                            "report_id": report_id,
                            "messages": messages,
                        }
                    )
                    + "\n"
                )
        rec.artifact(path)


def _name_slug_parts(path: Path | str) -> list[str]:
    """The tokens between the composition and the stamp of an sdk run name."""
    stem = Path(path).stem
    if "·" not in stem:
        return []
    tail = stem.rsplit("·", 1)[1]  # s1-haiku-5-5-high-20261015_101500
    parts = tail.split("-")
    if len(parts) < 3:
        return []
    return parts[1:-1]


def _name_efforts() -> set[str]:
    """Every effort a session run name can end its slug on: the SDK's and the
    open runtime's (a student's is ``none``, which no SDK effort is)."""
    from convfinqa.llm import OPEN_EFFORTS, SDK_EFFORTS

    return set(SDK_EFFORTS) | set(OPEN_EFFORTS)


def prior_sdk_model(path: Path | str) -> str | None:
    """The model slug an sdk run name carries, or None for a pre-slug name.

    `sdk-evalloop-test100-sdk_v1·s1-haiku-4-5-20260906_101500` → `haiku-4-5`,
    and `…·s1-haiku-5-5-high-20261015_101500` → `haiku-5-5` (the trailing
    effort is not part of the model). A name from before the slug existed
    (`…·s1-20260905_174627`) returns None: those runs all ran the default
    model, and their MLflow `sdk_model` param is the record.
    """
    middle = _name_slug_parts(path)
    if middle and middle[-1] in _name_efforts():
        middle = middle[:-1]
    return "-".join(middle) if middle else None


def prior_sdk_effort(path: Path | str) -> str | None:
    """The effort an sdk run name carries, or None for a name from before it did."""
    middle = _name_slug_parts(path)
    return middle[-1] if middle and middle[-1] in _name_efforts() else None


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
    if Path(path).name.startswith("open-evalloop"):
        return "open_agent"
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
    sdk_model_slug: str | None = None,
    sdk_effort: str | None = None,
) -> None:
    """Refuse a resume that would silently change the question set.

    Four ways it can: a different split, a different version, a different
    runtime, or a report set the current pass does not cover. `--train-seed`
    draws its own reports, so the check there is that every prior report is in
    *this* draw — a different draw is a different question set, and stitching
    the two would produce a CSV that is not any split. For the sdk runtime a
    fifth: a different model, read from the prior run name's slug — reused
    conversations answered by one model and fresh ones by another would be no
    model's evidence. A pre-slug name is accepted as the default model. A
    sixth, the same way: a different effort, read from the name's trailing
    effort token; a name from before that token existed is accepted.
    """
    name = Path(path).name
    if sdk_effort is not None:
        was_effort = prior_sdk_effort(path)
        if was_effort is not None and was_effort != sdk_effort:
            raise ValueError(
                f"--resume-from {name} ran at effort {was_effort!r}; this pass is "
                f"{sdk_effort!r}. A pass is one agent's evidence; resume with the "
                "same --sdk-effort or start a fresh pass."
            )
    if sdk_model_slug is not None:
        was_model = prior_sdk_model(path)
        if was_model is not None and was_model != sdk_model_slug:
            raise ValueError(
                f"--resume-from {name} ran model {was_model!r}; this pass is "
                f"{sdk_model_slug!r}. A pass is one model's evidence; resume "
                "with the same --sdk-model or start a fresh pass."
            )
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


def listed_ids(
    path: str | Path, manifest: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """The train-side ids listed in `path`, in file order; refuses any sealed id.

    A conversation in the manifest's test or holdout split is refused outright:
    a listed run is never allowed to answer the gate split or the sealed set.
    """
    import hashlib

    raw = Path(path).read_bytes()
    listed = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not listed:
        raise ValueError(f"--report-ids-file {path} lists no report ids")
    splits = manifest.get("splits") or {}
    sealed = {str(r) for s in ("test", "holdout") for r in splits.get(s) or []}
    bad = sorted(set(listed) & sealed)
    if bad:
        raise ValueError(
            f"--report-ids-file {Path(path).name} lists {len(bad)} test or holdout "
            f"ids ({', '.join(bad[:3])}): a listed run never answers a sealed split"
        )
    return listed, {
        "n_reports": len(listed),
        "report_ids_file": Path(path).name,
        "report_ids_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def narrow_draw(
    report_ids: list[str], draw: dict[str, Any], path: str | Path
) -> tuple[list[str], dict[str, Any]]:
    """Keep the drawn ids listed in `path`, in draw order; refuse any extra id."""
    import hashlib

    raw = Path(path).read_bytes()
    listed = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not listed:
        raise ValueError(f"--report-ids-file {path} lists no report ids")
    drawn = set(report_ids)
    extra = sorted(set(listed) - drawn)
    if extra:
        sample = ", ".join(extra[:3]) + ("…" if len(extra) > 3 else "")
        raise ValueError(
            f"--report-ids-file {Path(path).name} lists {len(extra)} ids that are "
            f"not in seed {draw.get('draw_seed')}'s draw ({sample}): the file "
            "narrows a draw, it never adds to one"
        )
    keep = set(listed)
    kept = [r for r in report_ids if r in keep]
    return kept, {
        **draw,
        "n_reports_drawn": len(report_ids),
        "n_reports": len(kept),
        "report_ids_file": Path(path).name,
        "report_ids_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


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
        # The CLI's figure is the session's running total, so this sum
        # overstates a pass; `cost.sdk_session_cost_usd` is the price paid.
        "sdk_cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6),
    }
    checked = [r for r in rows if "harness_retry" in r]
    if checked:  # the checked open harness (s29): questions a check sent back
        for check in ("H2", "H3"):
            out[f"harness_retries_{check.lower()}"] = float(
                sum(1 for r in checked if r["harness_retry"] == check)
            )
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
    sdk_model: str | None = None,
    sdk_effort: str | None = None,
    report_ids_file: str | Path | None = None,
    open_harness: str | None = None,
    open_model: str | None = None,
    open_effort: str | None = None,
    student_temperature: float | None = None,
) -> dict[str, Any]:
    """Run one split × version pass; return a summary with the CSV and run id.

    `sdk_model` pins the model the sdk runtime runs on for this pass (default
    `llm.sdk_model_name()`), so the same prompt can be scored on two models
    with nothing else changed. The model's slug goes into the run name and the
    CSV name, and the full id is the `sdk_model` param and tag on the run. It
    is refused on the pipeline runtime, whose four agents take their model
    from `llm.py` and not from a pass. `sdk_effort` is the same for effort
    (default `llm.sdk_effort_name()`, the champion's): the run name carries it
    after the model slug, and it is the `sdk_effort` param and tag.

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
    ``open_agent`` (s25) runs an ``sdk_vN`` prompt unchanged on an open model:
    `open_harness`, `open_model` and `open_effort` pick the agent, are refused
    on any other runtime, and name the run ``open-evalloop-…·s1-pyd-<model>-
    <effort>-<stamp>``.

    `resume_from` completes a pass that was cut short — by a rate limit, or by
    anything else that left rows unscored. Conversations the prior CSV answered
    *whole* are copied through verbatim (their own `run_id` and `trace_id`
    kept, `resumed_from_run_id` set); every other conversation of the split is
    run again from turn 0. Split, version, runtime and the report set are
    checked against the prior CSV first, because a resume that quietly changed
    the question set would produce a file that is not any split's evidence.

    `report_ids_file` narrows a `train_seed` draw to the ids listed in it (one
    per line), keeping the draw's order. Every listed id must be in the draw:
    the file filters a draw, it never adds to one, so the pass stays that
    seed's evidence. The SFT data build uses it to skip the drawn conversations
    that share a report page with test or the sealed holdout. The file's name,
    sha256 and the kept count are logged with the draw.
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
    if report_ids_file is not None and train_seed is None:
        if split != "train":
            raise ValueError(
                f"--report-ids-file lists train-side conversations; refusing on {split!r}"
            )
        # s28: an explicit list from the train side (sft_v1's val set, cut by
        # page from the pool, is no seeded draw). Never a gate-split id.
        report_ids, draw = listed_ids(report_ids_file, manifest)
    elif train_seed is not None:
        report_ids, draw = draw_train(
            seed=train_seed, n_reports=n_reports or len(manifest["splits"]["train"])
        )
        if report_ids_file is not None:
            report_ids, draw = narrow_draw(report_ids, draw, report_ids_file)
    else:
        report_ids = split_report_ids(
            split, n_reports=n_reports, n_questions=n_questions
        )
    if runtime not in RUNTIMES:
        raise ValueError(f"unknown runtime {runtime!r}; expected one of {RUNTIMES}")
    if prompts_pkg.is_sdk_version(version) != (runtime in SESSION_RUNTIMES):
        raise ValueError(
            f"version {version!r} does not belong to runtime {runtime!r}: "
            "sdk_vN prompts run under --runtime agent_sdk or open_agent, vN "
            "bundles under pipeline"
        )
    if runtime != "open_agent" and (
        open_harness or open_model or open_effort or student_temperature is not None
    ):
        raise ValueError(
            "--open-harness, --open-model and --open-effort apply to "
            "--runtime open_agent only"
        )
    if sdk_model is not None and runtime != "agent_sdk":
        raise ValueError(
            f"--sdk-model {sdk_model!r} applies to --runtime agent_sdk only: the "
            "pipeline's agents take their model from llm.py, not from a pass"
        )
    if sdk_effort is not None and runtime != "agent_sdk":
        raise ValueError(
            f"--sdk-effort {sdk_effort!r} applies to --runtime agent_sdk only"
        )
    model_slug: str | None = None
    if runtime == "agent_sdk":
        from convfinqa.llm import (
            SDK_EFFORTS,
            sdk_effort_name,
            sdk_model_name,
            sdk_model_slug,
        )

        sdk_model = sdk_model or sdk_model_name()
        sdk_effort = sdk_effort or sdk_effort_name()
        if sdk_effort not in SDK_EFFORTS:
            raise ValueError(
                f"unknown effort {sdk_effort!r}; expected one of {SDK_EFFORTS}"
            )
        model_slug = sdk_model_slug(sdk_model)
    if runtime == "open_agent":
        from convfinqa.llm import (
            OPEN_EFFORTS,
            OPEN_HARNESSES,
            OPEN_MODELS,
            STUDENT_MODELS,
            check_open_agent,
            open_model_slug,
        )

        open_harness = open_harness or OPEN_HARNESSES[0]
        open_model = open_model or OPEN_MODELS[0]
        open_effort = open_effort or (
            "none" if open_model in STUDENT_MODELS else "high"
        )
        for value, allowed, flag in (
            (open_harness, OPEN_HARNESSES, "--open-harness"),
            (open_model, OPEN_MODELS, "--open-model"),
            (open_effort, OPEN_EFFORTS, "--open-effort"),
        ):
            if value not in allowed:
                raise ValueError(f"{flag} {value!r}; expected one of {allowed}")
        check_open_agent(open_model, open_effort)
        if student_temperature is not None and open_model not in STUDENT_MODELS:
            raise ValueError(
                f"--student-temperature applies to a student model, not {open_model!r}"
            )
        model_slug = open_model_slug(open_harness, open_model, student_temperature)
    # The effort the run name carries, whichever session runtime ran.
    name_effort = sdk_effort if runtime == "agent_sdk" else open_effort
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
            sdk_model_slug=model_slug,
            sdk_effort=name_effort if runtime in SESSION_RUNTIMES else None,
        )
        reused = reusable_conversations(prior, examples)
    to_run = [ex for ex in examples if str(ex.report_id) not in reused]
    n_reused_questions = sum(len(rows) for rows in reused.values())

    from convfinqa.tracking import prompt_ledger

    runtime_params: dict[str, Any] = {"runtime": runtime}
    runtime_tags: dict[str, str] = {}
    if runtime == "agent_sdk":
        from convfinqa.config import settings

        system_prompt = prompts_pkg.load_sdk(version)
        composition = prompt_ledger.sdk_composition_string(
            prompt_ledger.ensure_sdk(version)  # register the prompt hash first
        )
        run_conversation = sdk_conversation_fn(
            system_prompt, version, sdk_model, sdk_effort
        )
        runtime_params.update(
            {
                "sdk_model": sdk_model,
                "sdk_effort": sdk_effort,
                "billing": settings.sdk_billing,
                "max_turns": settings.sdk_max_turns,
                "sdk_total_tokens_limit": settings.sdk_total_tokens_limit,
            }
        )
        # A tag as well as a param: the model is a search key (every run of
        # one model, across versions), and MLflow's run search filters on tags.
        runtime_tags["sdk_model"] = str(sdk_model)
        runtime_tags["sdk_effort"] = str(sdk_effort)
        # The slug sits between the composition and the stamp, so a listing
        # sorted by name groups a prompt's runs and reads the model off each.
        prefix = "sdk-evalloop"
    elif runtime == "open_agent":
        from importlib.metadata import version as package_version

        from convfinqa.config import settings
        from convfinqa.llm import (
            OPEN_BILLING,
            STUDENT_BILLING,
            STUDENT_MODELS,
            STUDENT_SAMPLING,
        )

        student = open_model in STUDENT_MODELS
        if student:
            # s28's D5: the sdk prompt verbatim plus the reasoning paragraph,
            # for the base model and the tuned one alike.
            from convfinqa.sft.prompt import student_prompt
            from convfinqa.tracking.prompt_ledger import prompt_hash

            system_prompt = student_prompt(version)
            runtime_params["student_prompt"] = (
                f"student_v1@{prompt_hash(system_prompt)}"
            )
        else:
            system_prompt = prompts_pkg.load_sdk(version)
        composition = prompt_ledger.sdk_composition_string(
            prompt_ledger.ensure_sdk(version)
        )
        run_conversation = OpenConversationFn(
            system_prompt,
            version,
            str(open_model),
            str(open_effort),
            harness=str(open_harness),
            temperature=student_temperature,
        )
        runtime_params.update(
            {
                "open_harness": open_harness,
                "open_harness_version": package_version("pydantic-ai-slim"),
                "open_model": open_model,
                "open_effort": open_effort,
                "student_temperature": student_temperature
                if student_temperature is not None
                else (STUDENT_SAMPLING["temperature"] if student else None),
                "billing": STUDENT_BILLING if student else OPEN_BILLING,
                "max_turns": settings.sdk_max_turns,
                "sdk_total_tokens_limit": settings.sdk_total_tokens_limit,
            }
        )
        runtime_tags["open_model"] = str(open_model)
        runtime_tags["open_effort"] = str(open_effort)
        runtime_tags["open_harness"] = str(open_harness)
        prefix = "open-evalloop"
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
        f"·{composition.replace('.', '')}"
        f"{'-' + model_slug if model_slug else ''}"
        f"{'-' + str(name_effort) if model_slug else ''}-{stamp}"
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
    # Preflight the central server before spending a single model call: every
    # MLflow write below degrades to a no-op when the store is away, which is
    # right for serving and wrong here — an eval with no record is money burnt.
    # Only a server URI is probed; the archived local store needs no health.
    if mlflow_log.is_remote():
        mlflow_log.assert_reachable()
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
        **runtime_tags,
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
            **runtime_tags,
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
        if runtime == "open_agent":
            captures_all = [c for out in results for c in out.captures]
            metrics.update(sdk_run_metrics(captures_all))
            sdks = [c.get("sdk") or {} for c in captures_all]
            # Priced per call on DeepSeek's card in force when it ran.
            metrics["open_cost_usd"] = round(
                sum(float(s.get("cost_usd") or 0.0) for s in sdks), 6
            )
            metrics["reasoning_tokens"] = float(
                sum(int(s.get("reasoning_tokens") or 0) for s in sdks)
            )
            served = sorted({m for s in sdks for m in s.get("served_models") or []})
            rec.param("served_models", ",".join(served) or "none")
            if isinstance(run_conversation, OpenConversationFn):
                _log_trajectories(rec, run_name, run_conversation.trajectories)
        if runtime == "agent_sdk":
            metrics.update(
                sdk_run_metrics([c for out in results for c in out.captures])
            )
            # The price paid: `sdk_cost_usd` above sums the CLI's running
            # totals and is kept only so new runs read like old ones.
            metrics["sdk_session_cost_usd"] = cost.sdk_session_cost_usd(
                (
                    [(c.get("sdk") or {}).get("cost_usd") for c in out.captures]
                    for out in results
                ),
                sdk_model,
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
