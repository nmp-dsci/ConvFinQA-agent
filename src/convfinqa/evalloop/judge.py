"""The confidence judge (s12): answer when a second model can verify the trace.

The Agent SDK runtime answers 90% of the gate correctly and every answer with
the same conviction. This module builds the piece that makes it shippable — a
cheap second model that reads a finished turn's evidence (the question, the
document, the session's sub-questions, retrieved cells with their sources, the
program, the calculator trajectory and the answer) and returns a band:
``high`` releases the answer, ``low`` withholds it. The judge never sees gold.

How it is trained, in the two moves that produced ``sdk_v1``:

1. **Diagnose, with gold.** A teacher goes through every case of the optimise
   split — correct and incorrect — comparing gold answer and program against the
   agent's answer, program and trace, and writes what the trace got wrong or
   what it got right, which of six named checks a reader could have run, and a
   rule a judge *without* gold could apply. ``judge_diagnoses.jsonl`` is that
   record, append-only.
2. **Distil, without gold.** One teacher call reads every diagnosis and writes
   the judge's system prompt, ``prompts/judge_jN.py``. The distil prompt forbids
   any reference to gold and `validate_judge_prompt` refuses a draft that names
   a gold field, so the judge cannot be trained to pattern-match an answer it
   will not have.

Then **score** (Haiku runs the prompt over a split at natural prevalence),
**gate** (a candidate judge replaces the champion only if it keeps the high
band inside the error target with more coverage and no fewer failures caught)
and, once, **test** on the gate split with the frozen champion.

Three splits, and why they are separate: *optimise* is what the teacher sees
and is balanced 50/50 (every attributable negative plus as many hard positives)
because the teacher's unit is a miss; *calibrate* picks the operating point and
gates versions, and stays at natural prevalence because coverage and precision
are prevalence-dependent; *test* is the runtime's own gate split, touched once.
The three are cut by conversation, never by turn.

Everything here that reads a model goes through `evalloop.sdk.run_structured`
— the one span-opening chokepoint — with the judge's own model, so a judge
call is traced exactly as a teacher call is.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from convfinqa.config import EVAL_ROOT, REPO_ROOT
from convfinqa.evalloop import ledgers, prompt_refs
from convfinqa.tracking import tracing

RUNTIME = "agent_sdk"
JUDGE_EXPERIMENT = "convfinqa-judge"
JUDGE_DIR = EVAL_ROOT / "judge"
JUDGE_DIR_ENV = "CONVFINQA_JUDGE_DIR"
PROMPTS_DIR = REPO_ROOT / "src" / "convfinqa" / "prompts"
SPLITS = ("optimise", "calibrate", "test")
DEFAULT_ERROR_TARGET = 0.01

#: The six claims a reader can test against the document and the trace, and
#: the vocabulary the diagnoses, the judge prompt and the verdict all share.
CHECKS: tuple[str, ...] = (
    "operand_in_source",
    "period_matches",
    "reference_resolved",
    "program_matches",
    "arithmetic_verified",
    "unit_and_scale",
)
CHECK_DESCRIPTIONS: dict[str, str] = {
    "operand_in_source": (
        "every retrieved value is actually in the cell or sentence the trace cites, "
        "as the number it claims (sign and magnitude)"
    ),
    "period_matches": (
        "the period (year, quarter, column) of each operand is the one the question "
        "names, including relative periods like 'the prior year'"
    ),
    "reference_resolved": (
        "'that', 'this change', 'the sum', 'it' and the like bind to the right prior "
        "turn's answer, and a prior answer reused as an operand is the right one"
    ),
    "program_matches": (
        "the operation(s) are what the question asks for — difference vs ratio, "
        "percentage change vs share, operand order, growth vs decline"
    ),
    "arithmetic_verified": (
        "the calculator calls compute that program from those operands and the "
        "answer is the result of the last call, not a number produced in prose"
    ),
    "unit_and_scale": (
        "percent vs decimal, millions vs thousands, sign, and rounding match how "
        "the question and the document frame the quantity"
    ),
}
Verdict = Literal["pass", "fail", "cannot_tell"]

#: Keys that must never reach the judge. `judge_payload` asserts on them; a
#: test pins the assertion.
GOLD_KEYS: frozenset[str] = frozenset(
    {
        "gold_answer",
        "gold_program",
        "gold_turn_type",
        "gold_conv_type",
        "correct",
        "derived_attribution",
        "derived_checks",
        "missing_gold_operands",
        "prior_gold_answers",
        "adjudication",
        "first_wrong_turn",
        "cascade",
    }
)
#: Words a distilled judge prompt may not contain: a rule phrased in terms of
#: the gold answer is a rule the judge cannot run.
FORBIDDEN_PROMPT_WORDS: tuple[str, ...] = ("gold", "ground truth", "ground-truth")

JUDGE_HEADINGS: tuple[str, ...] = (
    "1. Role",
    "2. What you are given",
    "3. The six checks",
    "4. Patterns that mean the answer is wrong",
    "5. Patterns that mean the answer is right",
    "6. Deciding the band",
    "7. Output contract",
)
JUDGE_MIN_PROMPT_CHARS = 800


# ── Schemas ───────────────────────────────────────────────────────────────


class Checks(BaseModel):
    """One verdict per named check. `cannot_tell` is an honest third value."""

    operand_in_source: Verdict
    period_matches: Verdict
    reference_resolved: Verdict
    program_matches: Verdict
    arithmetic_verified: Verdict
    unit_and_scale: Verdict

    def as_dict(self) -> dict[str, str]:
        """Check name → verdict, in `CHECKS` order."""
        return {c: getattr(self, c) for c in CHECKS}


class JudgeVerdict(BaseModel):
    """What the judge returns for one turn. The band is the contract."""

    checks: Checks
    band: Literal["high", "low"] = Field(
        description="high = the answer can be released; low = withhold it"
    )
    p_correct: float = Field(
        ge=0.0, le=1.0, description="Probability the answer is correct"
    )
    reason: str = Field(description="One line a reviewer can act on")


class JudgeDiagnosis(BaseModel):
    """The teacher's reading of one case, with gold in hand."""

    checks: Checks = Field(
        description="How each check reads for this trace, judged WITH gold"
    )
    what_happened: str = Field(
        description=(
            "2-4 sentences: for a wrong answer, the first mistake and how it produced "
            "the wrong number; for a right answer, what the trace did that made it "
            "right"
        )
    )
    evidence: str = Field(
        description="Quoted from the document, the history or the trace"
    )
    detectable_without_gold: bool = Field(
        description=(
            "Could a reader with the document and the trace, but NOT the gold "
            "answer, have known this answer was right/wrong?"
        )
    )
    judge_rule: str = Field(
        description=(
            "ONE imperative, general rule for a judge that has no gold answer, "
            "phrased as a test on the trace — not about this company or year"
        )
    )
    gold_suspect: bool = Field(
        description="True if the gold answer itself looks wrong or ambiguous"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class JudgePromptDraft(BaseModel):
    """The distil agent's reply: a judge system prompt."""

    prompt: str = Field(description="The complete judge system prompt")
    sections: list[str] = Field(description="The headings, in order")
    notes: str = Field(description="Judgement calls made while distilling")


# ── Prompts ───────────────────────────────────────────────────────────────

_CHECKS_TEXT = "\n".join(f"- `{c}`: {d}" for c, d in CHECK_DESCRIPTIONS.items())

JUDGE_DIAGNOSE_PROMPT = f"""You diagnose ONE turn of a single-session financial Q&A agent,
knowing the outcome. The agent answers a conversation about one financial
report in one session; for every turn it reports a trail: the turn type it
chose (number or program), the sub-questions and symbolic program it planned,
the values it retrieved with their sources, and its answer. Arithmetic must go
through six calculator tools, and every tool call is in the calculator
trajectory you are shown.

You are told whether the answer was CORRECT or INCORRECT, and you are given the
gold answer and gold program. Your reader is a JUDGE that will NOT have gold:
it will see only the question, the history, the document and the trail, and it
must decide whether to release the answer. Your job is to turn this one case
into something that judge can use.

For an INCORRECT answer: find the first mistake — the wrong cell, the wrong
period, the misresolved reference, the wrong operation, the skipped or
inline arithmetic, the unit or scale slip — and say what in the trail REVEALS
it to a reader with the document. If nothing in the trail reveals it (the trail
is internally consistent and the document genuinely supports the wrong reading),
say so with detectable_without_gold=false.

For a CORRECT answer: say what the trail did right — the cited cell matches
the question's row and period, the program is the question's operation, the
tool calls compute it — and what would have made this a wrong answer, so the
judge learns what a right trail looks like and does not over-abstain on
unusual-looking but correct ones.

Judge each of these six checks for this trace, as a reader with gold would:
{_CHECKS_TEXT}

`evidence` must quote the document, the history or the trail. `judge_rule` is
ONE imperative, general rule the judge can apply without gold — a test on the
trail, not on the answer; if the judge's own verdict is provided (a second
round), explain where its reasoning went wrong and sharpen the rule against
that. If the gold answer itself looks wrong, say so with gold_suspect=true and
lower your confidence."""


JUDGE_DISTIL_PROMPT = f"""You are writing the system prompt for a confidence JUDGE: a
model that reads one finished turn of a single-session financial Q&A agent and
decides whether the answer can be released to a user or must be withheld.

The judge sees the question, the conversation history, the report (table and
text), and the agent's trail — turn type, sub-questions, symbolic program,
retrieved values with their cited sources, the calculator trajectory, the
answer and the agent's reasoning. It does NOT see any reference answer, and
never will. Its output is six check verdicts, a band (high = release, low =
withhold), a probability that the answer is correct, and a one-line reason.

You are given diagnoses: for many turns, a teacher who DID know the outcome
wrote what the trail got wrong or got right, how each check read, and a rule
the judge could apply without a reference answer. Distil those into the
prompt: the general tests, the failure patterns, the patterns that mark a
right answer, and how to weigh them into a band. Cite no company, year or
value from any case.

The six checks, in this order, with these exact names:
{_CHECKS_TEXT}

Hard constraints:
- Use EXACTLY the seven headings given, in that order, as markdown `## N. Name`
  lines. Nothing else may be a `## ` heading.
- The judge has no answer key and a rule phrased against one is a rule it cannot
  run. So the words "gold", "ground truth" and "ground-truth" must not appear
  ANYWHERE in the prompt — not even in a sentence telling the judge it has none
  (a validator refuses the draft on the substring alone). Say "you have no
  answer key" if you need to say it.
- Name all six checks in the checks section, and require the judge to actually
  re-read the cited cell and re-derive the operation from the question.
- The band rule must be explicit: which check failures force `low`, how
  `cannot_tell` is weighed, and what a `high` requires. The operating target is
  that the high band is at least 99% correct; when in doubt, withhold.
- Name every key of the output schema in the Output contract section.
- Aim for a prompt a careful analyst could follow: dense, imperative, ordered.

When a previous judge prompt is provided, you are REVISING it: keep what
worked, change what the new diagnoses (the previous judge's own misses) show
was wrong, and keep the seven headings.

Return JSON matching the schema: the prompt, the headings you used, and your
notes."""


# ── Location ──────────────────────────────────────────────────────────────


def judge_dir() -> Path:
    """Where the judge's committed record lives: the env override, else `JUDGE_DIR`."""
    override = os.environ.get(JUDGE_DIR_ENV)
    return Path(override) if override else JUDGE_DIR


def dataset_path() -> Path:
    """The split manifest: which question ids are in which split, and from where."""
    return judge_dir() / "dataset.json"


def diagnoses_path() -> Path:
    """The append-only diagnosis record."""
    return judge_dir() / "judge_diagnoses.jsonl"


def gates_path() -> Path:
    """The append-only gate record."""
    return judge_dir() / "judge_gates.jsonl"


def scores_dir() -> Path:
    """One CSV per (judge version × split) scoring pass."""
    return judge_dir() / "scores"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _append_jsonl(target: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows, fsync, never rewrite — the ledgers' discipline."""
    if not rows:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(target: Path) -> list[dict[str, Any]]:
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_diagnoses(version: str | None = None) -> list[dict[str, Any]]:
    """Every diagnosis row, optionally only those written against one judge version."""
    rows = _read_jsonl(diagnoses_path())
    if version is not None:
        rows = [r for r in rows if r.get("judge_version") == version]
    return rows


def load_gates() -> list[dict[str, Any]]:
    """Every judge gate verdict, oldest first."""
    return _read_jsonl(gates_path())


# ── The dataset ───────────────────────────────────────────────────────────


def _read_run(csv_path: Path | str) -> pd.DataFrame:
    """A run CSV with `correct` as bool and the per-agent checks present."""
    from convfinqa.evalloop import stage_scores

    df = pd.read_csv(csv_path)
    df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
    if "unscored" in df.columns:
        unscored = df["unscored"].astype(str).str.lower().isin({"true", "1"})
        if unscored.any():
            raise ValueError(
                f"{Path(csv_path).name} carries {int(unscored.sum())} unscored rows — "
                "an incomplete pass cannot label a judge"
            )
    stage_scores._ensure_scored(df)
    return df


def _hard_positive(row: pd.Series) -> bool:
    """A correct turn whose trace looks like a failure — the positives worth learning."""
    from convfinqa.evalloop.sdk_teacher import sdk_flags

    flags = sdk_flags(row)
    return bool(
        flags["stage_skips"]
        or flags["inline_arithmetic"]
        or (
            str(row.get("pred_turn_type", "")).lower() == "program"
            and int(flags["tool_calls"]) == 0
        )
    )


def _balanced_optimise(
    df: pd.DataFrame, *, seed: int
) -> tuple[list[str], dict[str, Any]]:
    """Every attributable negative plus an equal number of positives, hard ones first.

    The positive fill is stratified on the negatives' turn-type mix so the set
    teaches the judge the same shape of question it will be wrong about, and
    the remainder is drawn with the logged seed so the cut is reproducible.
    """
    from convfinqa.evalloop import stage_scores

    negatives = df[~df["correct"]].copy()
    positives = df[df["correct"]].copy()
    attribution = (
        stage_scores.attribute_frame(negatives)
        if len(negatives)
        else pd.Series(dtype=str)
    )
    non_agent = (
        attribution.isin(stage_scores.NON_AGENT)
        if len(negatives)
        else pd.Series(dtype=bool)
    )
    excluded = negatives[non_agent] if len(negatives) else negatives
    negatives = negatives[~non_agent] if len(negatives) else negatives

    hard_mask = (
        positives.apply(_hard_positive, axis=1)
        if len(positives)
        else pd.Series(dtype=bool)
    )
    hard = positives[hard_mask] if len(positives) else positives
    rest = positives[~hard_mask] if len(positives) else positives

    n_target = len(negatives)
    chosen: list[str] = list(hard["question_id"])[:n_target]
    need = n_target - len(chosen)
    rng = random.Random(seed)
    if need > 0 and len(rest):
        # Stratify the fill on the negatives' turn-type mix.
        mix = negatives["gold_turn_type"].value_counts(normalize=True).to_dict()
        pool = {t: list(rest[rest["gold_turn_type"] == t]["question_id"]) for t in mix}
        for ids in pool.values():
            rng.shuffle(ids)
        quotas = {t: int(round(need * share)) for t, share in mix.items()}
        for t, q in quotas.items():
            chosen.extend(pool.get(t, [])[:q])
        leftover = need - (len(chosen) - min(len(hard), n_target))
        if leftover > 0:
            spare = [q for ids in pool.values() for q in ids if q not in set(chosen)]
            rng.shuffle(spare)
            chosen.extend(spare[:leftover])
    ids = list(negatives["question_id"]) + chosen
    stats = {
        "n_negatives": int(len(negatives)),
        "n_positives": int(len(chosen)),
        "n_hard_positives": int(min(len(hard), n_target)),
        "n_excluded_non_agent": int(len(excluded)),
        "excluded_question_ids": list(excluded["question_id"]),
    }
    return ids, stats


def build_dataset(
    *,
    optimise_csv: Path | str,
    calibrate_csv: Path | str,
    test_csv: Path | str,
    seed: int = 2026,
    name: str = "judge_v1",
    path: Path | None = None,
) -> dict[str, Any]:
    """Cut the three splits from three committed run CSVs and write the manifest.

    The optimise and calibrate CSVs are two train draws of the same prompt on
    the same model; the four conversations they share go to optimise only, so
    no conversation is on both sides. The test CSV is the gate split's pass.
    The manifest carries the attribution rule id: the NON_AGENT exclusion
    depends on it, and a rule change means a rebuild, as it does for the
    diagnoses ledger.
    """
    from convfinqa.evalloop import stage_scores

    opt = _read_run(optimise_csv)
    cal = _read_run(calibrate_csv)
    tst = _read_run(test_csv)
    for frame, label in ((opt, "optimise"), (cal, "calibrate"), (tst, "test")):
        if "question_id" not in frame.columns:
            raise ValueError(f"{label} CSV has no question_id column")

    shared = set(opt["report_id"]) & set(cal["report_id"])
    cal = cal[~cal["report_id"].isin(shared)]
    gate_overlap = (set(opt["report_id"]) | set(cal["report_id"])) & set(
        tst["report_id"]
    )
    if gate_overlap:
        raise ValueError(
            f"{len(gate_overlap)} conversation(s) of the test CSV also appear in a "
            "training draw — the judge's test must be untouched"
        )

    optimise_ids, opt_stats = _balanced_optimise(opt, seed=seed)
    manifest = {
        "name": name,
        "built_at": _now(),
        "seed": seed,
        "runtime": RUNTIME,
        "runtime_version": str(opt["model_version_id"].iloc[0])
        if "model_version_id" in opt
        else "",
        "attribution_rule": stage_scores.attribution_rule_id(),
        "sources": {
            "optimise": str(optimise_csv),
            "calibrate": str(calibrate_csv),
            "test": str(test_csv),
        },
        "splits": {
            "optimise": optimise_ids,
            "calibrate": list(cal["question_id"]),
            "test": list(tst["question_id"]),
        },
        "shared_reports_to_optimise": sorted(shared),
        "optimise_balance": opt_stats,
        "stats": {
            "optimise": {
                "n": len(optimise_ids),
                "n_wrong": opt_stats["n_negatives"],
                "n_reports": int(
                    opt[opt["question_id"].isin(optimise_ids)]["report_id"].nunique()
                ),
            },
            "calibrate": {
                "n": int(len(cal)),
                "n_wrong": int((~cal["correct"]).sum()),
                "n_reports": int(cal["report_id"].nunique()),
            },
            "test": {
                "n": int(len(tst)),
                "n_wrong": int((~tst["correct"]).sum()),
                "n_reports": int(tst["report_id"].nunique()),
            },
        },
    }
    target = path or dataset_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    """The committed manifest."""
    target = path or dataset_path()
    if not target.exists():
        raise FileNotFoundError(
            f"no judge dataset at {target} — build one with `convfinqa-evalloop judge-dataset`"
        )
    loaded: dict[str, Any] = json.loads(target.read_text())
    return loaded


def split_frame(manifest: dict[str, Any], split: str) -> pd.DataFrame:
    """The rows of one split, read from its source CSV, in manifest order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {SPLITS}")
    df = _read_run(manifest["sources"][split])
    wanted = list(manifest["splits"][split])
    df = df[df["question_id"].isin(wanted)].copy()
    order = {q: i for i, q in enumerate(wanted)}
    df["_order"] = df["question_id"].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ── Payloads ──────────────────────────────────────────────────────────────


def row_from_capture(
    capture: dict[str, Any],
    *,
    report_id: str,
    turn_index: int,
    question: str,
    answer: str,
    program: str,
) -> dict[str, Any]:
    """A served turn's capture as the row shape the payload builders read.

    The eval CSV row and the live capture are the same information; this is
    the bridge so serving and scoring build byte-identical judge inputs.
    """
    from convfinqa.evaluation.runner import _capture_to_row_fields

    fields = _capture_to_row_fields(capture)
    sdk = capture.get("sdk")
    return {
        "report_id": report_id,
        "turn_index": turn_index,
        "question_id": f"{report_id}_q{turn_index}",
        "question": question,
        "pred_answer": answer,
        "pred_program": program,
        **fields,
        "sdk_io": json.dumps(sdk, default=str) if isinstance(sdk, dict) else "",
    }


def judge_payload(row: Any, doc: Any = None) -> dict[str, Any]:
    """Everything the judge sees about one turn — and nothing gold.

    Built from the same readers the teacher uses so the two never disagree
    about what the trail says; then checked against `GOLD_KEYS` so a future
    field added to the case payload cannot leak in by accident.
    """
    from convfinqa.evalloop import stage_scores
    from convfinqa.evalloop.sdk_teacher import (
        _report_for,
        retrieved_with_sources,
        sdk_flags,
    )

    get = ledgers._get
    payload = {
        "report_id": get(row, "report_id", ""),
        "report": doc if doc is not None else _report_for(str(get(row, "report_id"))),
        "conversation_history": get(row, "history_text") or "(no prior turns)",
        "question": get(row, "question", ""),
        "trail": {
            "turn_type": get(row, "pred_turn_type", ""),
            "conv_type": get(row, "pred_conv_type", ""),
            "sub_questions": stage_scores.planned_sub_questions(row),
            "program": get(row, "pred_program") or "",
            "retrieved": retrieved_with_sources(row),
            "answer": get(row, "pred_answer", ""),
            "reasoning": _trail_reasoning(row),
        },
        "calculator_trajectory": ledgers._calc_trajectory(row),
        "sdk_flags": sdk_flags(row),
    }
    leaked = GOLD_KEYS & set(payload) | GOLD_KEYS & set(payload["trail"])
    if leaked:
        raise AssertionError(f"gold reached the judge payload: {sorted(leaked)}")
    return payload


def _trail_reasoning(row: Any) -> str:
    from convfinqa.evalloop.sdk_teacher import _io

    for stage in ("triage", "calculator", "preprocess"):
        text = _io(row, stage).get("reasoning")
        if isinstance(text, str) and text.strip():
            return text
    return ""


def judge_prompt_text(payload: dict[str, Any]) -> str:
    """The exact user prompt a judge call sends."""
    return json.dumps(payload, default=str)


def diagnose_payload(
    row: Any, *, judge_verdict: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The teacher's view: the case with gold, its outcome, and the judge's verdict if any."""
    from convfinqa.evalloop.sdk_teacher import sdk_case_payload

    payload = sdk_case_payload(row)
    payload["outcome"] = (
        "CORRECT" if bool(ledgers._get(row, "correct")) else "INCORRECT"
    )
    payload["checks_to_judge"] = dict(CHECK_DESCRIPTIONS)
    if judge_verdict is not None:
        payload["judge_verdict"] = judge_verdict
    return payload


def diagnose_prompt_text(payload: dict[str, Any]) -> str:
    """The exact user prompt a judge-diagnosis call sends."""
    return json.dumps(payload, default=str)


# ── The judge call ────────────────────────────────────────────────────────


def judge_prompt_of(version: str) -> str:
    """The system prompt of judge `version`."""
    import convfinqa.prompts as prompts_pkg

    return prompts_pkg.load_judge(version)


async def judge_turn(
    row: Any,
    *,
    version: str,
    system_prompt: str | None = None,
    doc: Any = None,
    model: str | None = None,
) -> tuple[JudgeVerdict, dict[str, Any]]:
    """Judge one turn with prompt `version`; returns the verdict and the usage."""
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.llm import judge_model_name

    prompt = system_prompt if system_prompt is not None else judge_prompt_of(version)
    payload = judge_payload(row, doc)
    return await run_structured(
        judge_prompt_text(payload),
        schema=JudgeVerdict,
        system_prompt=prompt,
        max_turns=3,
        refs={
            "system_prompt": prompt_refs.judge_prompt_ref(version, prompt),
            "user_prompt": {
                "kind": "judge_case",
                "question_id": str(ledgers._get(row, "question_id", "")),
                "sha": prompt_refs.sha(judge_prompt_text(payload)),
            },
        },
        model=model or judge_model_name(),
    )


def verdict_record(verdict: JudgeVerdict, usage: dict[str, Any]) -> dict[str, Any]:
    """The verdict as it is stored on a capture, a scores row or an event."""
    tokens = usage.get("usage") or {}
    return {
        "band": verdict.band,
        "p_correct": round(float(verdict.p_correct), 4),
        "reason": verdict.reason,
        "checks": verdict.checks.as_dict(),
        "metrics": {
            "latency_ms": usage.get("duration_ms"),
            "input_tokens": tokens.get("input_tokens"),
            "output_tokens": tokens.get("output_tokens"),
            "cache_read_input_tokens": tokens.get("cache_read_input_tokens"),
            "cost_usd": usage.get("total_cost_usd"),
        },
    }


# ── Metrics ───────────────────────────────────────────────────────────────


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUROC (ties averaged); None without both classes."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    ranked = sorted(((s, i) for i, s in enumerate(scores)), key=lambda t: t[0])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _ece(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> float | None:
    if not scores:
        return None
    total = 0.0
    n = len(scores)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (s, y)
            for s, y in zip(scores, labels, strict=True)
            if (lo <= s < hi) or (b == bins - 1 and s == 1.0)
        ]
        if not members:
            continue
        conf = sum(s for s, _ in members) / len(members)
        acc = sum(1 for _, y in members if y) / len(members)
        total += len(members) / n * abs(conf - acc)
    return total


def wilson_upper(errors: int, n: int, z: float = 1.6449) -> float | None:
    """One-sided 95% upper bound on a rate; the rule of three at zero errors."""
    if n <= 0:
        return None
    p = errors / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / denom)


def risk_coverage(
    scores: Sequence[float], labels: Sequence[bool]
) -> list[dict[str, float]]:
    """The curve: for each threshold on p_correct, coverage and error among the answered."""
    if not scores:
        return []
    order = sorted(set(scores), reverse=True)
    n = len(scores)
    curve: list[dict[str, float]] = []
    for tau in order:
        answered = [(s, y) for s, y in zip(scores, labels, strict=True) if s >= tau]
        errors = sum(1 for _, y in answered if not y)
        curve.append(
            {
                "threshold": float(tau),
                "coverage": len(answered) / n,
                "error": errors / len(answered) if answered else 0.0,
                "n_answered": float(len(answered)),
            }
        )
    return curve


def selective_metrics(
    df: pd.DataFrame, *, error_target: float = DEFAULT_ERROR_TARGET
) -> dict[str, Any]:
    """Everything a judge run reports, from a scores frame.

    A scores frame has one row per judged turn with `correct`, `band`,
    `p_correct` and `report_id`. Band-based numbers are the contract (the band
    is what serving acts on); the threshold-based ones read the curve the
    recorded `p_correct` draws, for the case where the binary band proves too
    coarse.
    """
    n = int(len(df))
    if n == 0:
        return {"n": 0}
    correct = df["correct"].astype(bool)
    high = df["band"].astype(str) == "high"
    n_wrong = int((~correct).sum())
    n_high = int(high.sum())
    n_high_wrong = int((high & ~correct).sum())
    n_low_correct = int((~high & correct).sum())
    scores = [float(x) for x in df["p_correct"]]
    labels = [bool(x) for x in correct]
    curve = risk_coverage(scores, labels)
    at_target = [pt for pt in curve if pt["error"] <= error_target]
    best = max(at_target, key=lambda pt: pt["coverage"]) if at_target else None
    return {
        "n": n,
        "n_wrong": n_wrong,
        "n_reports": int(df["report_id"].nunique()) if "report_id" in df else None,
        "accuracy": float(correct.mean()),
        "error_target": error_target,
        # band-based (the contract)
        "coverage": n_high / n,
        "n_high": n_high,
        "n_high_wrong": n_high_wrong,
        "high_band_accuracy": (n_high - n_high_wrong) / n_high if n_high else None,
        "high_band_error": n_high_wrong / n_high if n_high else None,
        "high_band_error_upper95": wilson_upper(n_high_wrong, n_high),
        "failure_capture": (n_wrong - n_high_wrong) / n_wrong if n_wrong else None,
        "n_failures_caught": n_wrong - n_high_wrong,
        "n_failures_missed": n_high_wrong,
        "false_alarm_rate": n_low_correct / max(1, n - n_wrong),
        "n_false_alarms": n_low_correct,
        "meets_target": bool(n_high and n_high_wrong / n_high <= error_target),
        # score-based (the curve)
        "auroc": _auroc(scores, labels),
        "brier": sum((s - float(y)) ** 2 for s, y in zip(scores, labels, strict=True))
        / n,
        "ece": _ece(scores, labels),
        "coverage_at_target": best["coverage"] if best else 0.0,
        "threshold_at_target": best["threshold"] if best else None,
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    stat: str,
    *,
    n_boot: int = 1000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """A 95% CI on one `selective_metrics` statistic, resampling conversations."""
    if "report_id" not in df.columns or not len(df):
        return None, None
    groups = [g for _, g in df.groupby("report_id")]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sample = pd.concat([groups[rng.randrange(len(groups))] for _ in groups])
        v = selective_metrics(sample).get(stat)
        if isinstance(v, int | float):
            values.append(float(v))
    if not values:
        return None, None
    values.sort()
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


# ── Diagnose ──────────────────────────────────────────────────────────────


DIAGNOSIS_COLUMNS: tuple[str, ...] = (
    "diagnosis_id",
    "diagnosed_at",
    "runtime",
    "runtime_version",
    "judge_version",
    "round",
    "split",
    "dataset",
    "diagnosis_run_id",
    "diagnoser_model",
    "report_id",
    "question_id",
    "turn_index",
    "outcome",
    "question",
    "gold_answer",
    "pred_answer",
    "pred_program",
    "derived_attribution",
    "judge_band",
    "judge_p_correct",
    "judge_reason",
    "checks",
    "what_happened",
    "evidence",
    "detectable_without_gold",
    "judge_rule",
    "gold_suspect",
    "confidence",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)


def _misses(scores: pd.DataFrame) -> pd.DataFrame:
    """The judge's misses: wrong answers banded high, right answers banded low."""
    correct = scores["correct"].astype(bool)
    high = scores["band"].astype(str) == "high"
    return scores[(high & ~correct) | (~high & correct)].copy()


async def diagnose_split(
    *,
    split: str = "optimise",
    judge_scores: Path | str | None = None,
    judge_version: str = "",
    concurrency: int = 8,
    label: str | None = None,
    experiment: str = JUDGE_EXPERIMENT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Diagnose every case of `split` — or, given a scores CSV, the judge's misses on it.

    Round 1 (no scores): every optimise case, correct and incorrect, so the
    distil pass sees both what a wrong trail looks like and what a right one
    looks like. Round 2 (scores given): only the cases the judge got wrong, with
    the judge's own verdict in the prompt, so the teacher can say where its
    reasoning failed.
    """
    from convfinqa.evalloop import stage_scores, teacher
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.llm import teacher_model_name
    from convfinqa.tracking import mlflow_log

    manifest = load_dataset(manifest_path)
    frame = split_frame(manifest, split)
    verdicts: dict[str, dict[str, Any]] = {}
    round_n = 1
    if judge_scores is not None:
        scores = pd.read_csv(judge_scores)
        misses = _misses(scores)
        verdicts = {
            str(r.question_id): {
                "band": r.band,
                "p_correct": float(r.p_correct),
                "reason": str(r.reason),
                "checks": json.loads(str(r.checks))
                if isinstance(r.checks, str)
                else {},
            }
            for r in misses.itertuples()
        }
        frame = frame[frame["question_id"].isin(set(verdicts))].copy()
        round_n = 2
    attribution = (
        stage_scores.attribute_frame(frame) if len(frame) else pd.Series(dtype=str)
    )
    model = teacher_model_name()
    stamp = _stamp()
    run_name = f"judge-diagnose-{split}-r{round_n}{'-' + judge_version if judge_version else ''}-{stamp}"
    tracing.enable()

    with mlflow_log.run(
        run_name,
        kind="judge_diagnose",
        version=manifest.get("runtime_version") or None,
        params={
            "runtime": RUNTIME,
            "dataset": manifest["name"],
            "split": split,
            "round": round_n,
            "judge_version": judge_version,
            "judge_scores": str(judge_scores or ""),
            "n_cases": int(len(frame)),
            "concurrency": concurrency,
            "teacher_model": model,
            **({"experiment_label": label} if label else {}),
        },
        tags={"loop": "judge", "stage": "diagnose", "runtime": RUNTIME},
        experiment=experiment,
        actor_model=model,
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        sem = asyncio.Semaphore(max(1, concurrency))
        rows = list(frame.iterrows())

        async def one(
            order: int, row: pd.Series
        ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
            qid = str(row.question_id)
            payload = diagnose_payload(row, judge_verdict=verdicts.get(qid))
            payload["derived_attribution"] = str(attribution.iloc[order])
            text = diagnose_prompt_text(payload)
            async with sem:
                with tracing.span(
                    f"judge-diagnose {row.report_id} q{int(row.turn_index)}",
                    span_type="AGENT",
                    attributes={
                        "runtime": RUNTIME,
                        "report_id": row.report_id,
                        "turn_index": int(row.turn_index),
                        "outcome": payload["outcome"],
                        "round": round_n,
                    },
                    trace_tags={"stage": "judge-diagnose", "runtime": RUNTIME},
                ) as span:
                    try:
                        output, usage = await run_structured(
                            text,
                            schema=JudgeDiagnosis,
                            system_prompt=JUDGE_DIAGNOSE_PROMPT,
                            max_turns=4,
                            refs={
                                "system_prompt": prompt_refs.teacher_prompt_ref(
                                    "JUDGE_DIAGNOSE_PROMPT", JUDGE_DIAGNOSE_PROMPT
                                ),
                                "user_prompt": {
                                    "kind": "judge_diagnose_case",
                                    "dataset": manifest["name"],
                                    "split": split,
                                    "question_id": qid,
                                    "sha": prompt_refs.sha(text),
                                },
                            },
                            model=model,
                        )
                    except Exception as exc:  # noqa: BLE001 — one bad case must not sink the pass
                        return order, None, {"question_id": qid, "error": repr(exc)}, {}
                    span.set(
                        what_happened=output.what_happened,
                        judge_rule=output.judge_rule,
                        detectable_without_gold=output.detectable_without_gold,
                        confidence=float(output.confidence),
                    )
            verdict = verdicts.get(qid) or {}
            tokens = usage.get("usage") or {}
            record = {
                "diagnosis_id": ledgers.new_id("jd"),
                "diagnosed_at": _now(),
                "runtime": RUNTIME,
                "runtime_version": manifest.get("runtime_version", ""),
                "judge_version": judge_version,
                "round": round_n,
                "split": split,
                "dataset": manifest["name"],
                "diagnosis_run_id": str(rec.run_id),
                "diagnoser_model": model,
                "report_id": str(row.report_id),
                "question_id": qid,
                "turn_index": int(row.turn_index),
                "outcome": payload["outcome"],
                "question": str(row.question),
                "gold_answer": str(row.gold_answer),
                "pred_answer": str(row.pred_answer),
                "pred_program": str(row.get("pred_program", "") or ""),
                "derived_attribution": payload["derived_attribution"],
                "judge_band": verdict.get("band", ""),
                "judge_p_correct": verdict.get("p_correct"),
                "judge_reason": verdict.get("reason", ""),
                "checks": json.dumps(output.checks.as_dict()),
                "what_happened": output.what_happened,
                "evidence": output.evidence,
                "detectable_without_gold": bool(output.detectable_without_gold),
                "judge_rule": output.judge_rule,
                "gold_suspect": bool(output.gold_suspect),
                "confidence": float(output.confidence),
                "input_tokens": int(tokens.get("input_tokens") or 0),
                "output_tokens": int(tokens.get("output_tokens") or 0),
                "cost_usd": float(usage.get("total_cost_usd") or 0.0),
            }
            return order, record, None, usage

        settled = await asyncio.gather(*(one(i, r) for i, (_, r) in enumerate(rows)))
        diagnoses: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        for _order, record, failure, usage in sorted(settled, key=lambda r: r[0]):
            if failure is not None:
                failures.append(failure)
                print(f"  [skip] {failure['question_id']}: {failure['error']}")  # noqa: T201
                continue
            assert record is not None
            teacher._accumulate_usage(usage_total, usage)
            diagnoses.append(record)
            print(  # noqa: T201
                f"  [{record['report_id']} q{record['turn_index']}] {record['outcome']}"
                f" · detectable={record['detectable_without_gold']}"
                f" · {record['judge_rule'][:80]}"
            )
        _append_jsonl(diagnoses_path(), diagnoses)
        rec.text_artifact(
            "judge_diagnoses.jsonl", "".join(json.dumps(d) + "\n" for d in diagnoses)
        )
        n_wrong = sum(1 for d in diagnoses if d["outcome"] == "INCORRECT")
        rec.metrics(
            {
                "n_diagnosed": float(len(diagnoses)),
                "n_diagnose_failures": float(len(failures)),
                "n_incorrect": float(n_wrong),
                "n_correct": float(len(diagnoses) - n_wrong),
                "n_undetectable": float(
                    sum(1 for d in diagnoses if not d["detectable_without_gold"])
                ),
                "n_gold_suspect": float(sum(1 for d in diagnoses if d["gold_suspect"])),
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        if failures and len(failures) > len(diagnoses):
            raise SystemExit(
                f"{len(failures)} of {len(failures) + len(diagnoses)} cases failed to "
                "diagnose — that is a broken diagnoser, not a flaky call"
            )
        return {
            "run_id": rec.run_id,
            "run_name": run_name,
            "split": split,
            "round": round_n,
            "n_cases": int(len(frame)),
            "n_diagnosed": len(diagnoses),
            "n_failures": len(failures),
            "failures": failures,
            "n_incorrect": n_wrong,
            "usage": usage_total,
        }


# ── Distil ────────────────────────────────────────────────────────────────


def _compact_diagnosis(d: dict[str, Any]) -> dict[str, Any]:
    """What the distil prompt needs from one diagnosis row."""
    checks = d.get("checks")
    if isinstance(checks, str):
        try:
            checks = json.loads(checks)
        except json.JSONDecodeError:
            checks = {}
    return {
        "outcome": d.get("outcome"),
        "turn_type": d.get("pred_turn_type") or "",
        "checks": checks,
        "what_happened": d.get("what_happened"),
        "evidence": str(d.get("evidence", ""))[:600],
        "detectable_without_gold": d.get("detectable_without_gold"),
        "judge_rule": d.get("judge_rule"),
        "gold_suspect": d.get("gold_suspect"),
        **(
            {
                "previous_judge": {
                    "band": d.get("judge_band"),
                    "p_correct": d.get("judge_p_correct"),
                    "reason": d.get("judge_reason"),
                }
            }
            if d.get("judge_band")
            else {}
        ),
    }


def distil_prompt_text(
    diagnoses: Sequence[dict[str, Any]], *, base_prompt: str | None = None
) -> str:
    """The exact user prompt a distil call sends."""
    return json.dumps(
        {
            "n_diagnoses": len(diagnoses),
            "diagnoses": [_compact_diagnosis(d) for d in diagnoses],
            "checks": dict(CHECK_DESCRIPTIONS),
            "output_contract_schema": JudgeVerdict.model_json_schema(),
            "headings": list(JUDGE_HEADINGS),
            "error_target": DEFAULT_ERROR_TARGET,
            **({"previous_prompt": base_prompt} if base_prompt else {}),
        },
        default=str,
    )


_HEADING_RE = re.compile(r"^## .+$", re.MULTILINE)
_ANY_LEVEL_RE = re.compile(r"^#{1,4} (?=\d\. )", re.MULTILINE)


def normalise_headings(text: str) -> str:
    """Put every numbered section heading at level two.

    The distil agent sometimes writes ``# 1. Role`` where the contract says
    ``## 1. Role``. The level is formatting, not content, and refusing a
    17k-character prompt for it would cost a teacher call to fix a character —
    so the level is normalised here, mechanically, before the draft is judged
    on what it says. Nothing else about the text changes.
    """
    return _ANY_LEVEL_RE.sub("## ", text)


def validate_judge_prompt(text: str) -> list[str]:
    """Why a drafted judge prompt cannot be used, or an empty list."""
    problems: list[str] = []
    if len(text) < JUDGE_MIN_PROMPT_CHARS:
        problems.append(
            f"prompt is {len(text)} chars; minimum {JUDGE_MIN_PROMPT_CHARS}"
        )
    found = [line.strip() for line in _HEADING_RE.findall(text)]
    expected = [f"## {h}" for h in JUDGE_HEADINGS]
    if found != expected:
        problems.append(f"headings are {found}; expected exactly {expected}")
    lowered = text.lower()
    for word in FORBIDDEN_PROMPT_WORDS:
        if word in lowered:
            problems.append(
                f"prompt mentions {word!r}: the judge has no reference answer, so a "
                "rule phrased against one is a rule it cannot run"
            )
    for check in CHECKS:
        if check not in text:
            problems.append(f"prompt never names the check {check!r}")
    for key in ("checks", "band", "p_correct", "reason"):
        if key not in text:
            problems.append(f"prompt never names the output key {key!r}")
    return problems


def _write_judge_module(version: str, *, prompt: str, header: str) -> Path:
    """Write `prompts/<version>.py` exporting `JUDGE_PROMPT`. Refuses to overwrite."""
    import convfinqa.prompts as prompts_pkg

    if not prompts_pkg.is_judge_version(version):
        raise SystemExit(f"{version!r} is not a judge_jN version name")
    literal = prompt.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if literal.endswith('"'):
        literal += "\\n"
    body = f'''"""Generated by convfinqa.evalloop.judge — do not hand-edit.

{header}
"""

__all__ = ["{prompts_pkg.JUDGE_VAR}"]

{prompts_pkg.JUDGE_VAR} = """{literal}"""
'''
    path = PROMPTS_DIR / f"{version}.py"
    if path.exists():
        raise SystemExit(f"{path} already exists — pick a new version name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    import importlib

    importlib.invalidate_caches()
    return path


def next_judge_version() -> str:
    """`judge_j1` when none exists, else one past the highest."""
    import convfinqa.prompts as prompts_pkg

    versions = prompts_pkg.judge_versions()
    n = int(versions[-1].removeprefix("judge_j")) + 1 if versions else 1
    return f"judge_j{n}"


async def distil_judge(
    *,
    new_version: str,
    base_version: str | None = None,
    diagnoses: Sequence[dict[str, Any]] | None = None,
    experiment: str = JUDGE_EXPERIMENT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write `prompts/<new_version>.py` from the diagnoses; register it in the lineage.

    With `base_version` the call is a *revision*: the previous prompt goes in
    beside the new diagnoses (the judge's own misses, round 2). Without one it
    is the root of the lineage. Either way the draft is validated before it is
    written, and the first judge becomes `judge_champion` by default.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.llm import teacher_model_name
    from convfinqa.tracking import mlflow_log, prompt_ledger, registry

    if not prompts_pkg.is_judge_version(new_version):
        raise SystemExit(f"{new_version!r} is not a judge_jN version name")
    if (PROMPTS_DIR / f"{new_version}.py").exists():
        raise SystemExit(
            f"prompts/{new_version}.py already exists — a distillation is never redone"
        )
    manifest = load_dataset(manifest_path)
    rows = list(diagnoses) if diagnoses is not None else load_diagnoses()
    if not rows:
        raise SystemExit("no diagnoses to distil from — run `judge-diagnose` first")
    base_prompt = prompts_pkg.load_judge(base_version) if base_version else None
    user_prompt = distil_prompt_text(rows, base_prompt=base_prompt)
    model = teacher_model_name()
    tracing.enable()

    with mlflow_log.run(
        f"judge-distil-{new_version}{'-from-' + base_version if base_version else ''}",
        kind="judge_distil",
        version=manifest.get("runtime_version") or None,
        params={
            "runtime": RUNTIME,
            "dataset": manifest["name"],
            "new_version": new_version,
            "base_version": base_version or "",
            "n_diagnoses": len(rows),
            "teacher_model": model,
        },
        tags={"loop": "judge", "stage": "distil", "runtime": RUNTIME},
        experiment=experiment,
        actor_model=model,
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        rec.text_artifact("distil_prompt.txt", user_prompt)
        with tracing.span(
            f"judge-distil {new_version}",
            span_type="AGENT",
            attributes={"new_version": new_version, "base_version": base_version or ""},
            trace_tags={"stage": "judge-distil", "runtime": RUNTIME},
        ) as span:
            draft, usage = await run_structured(
                user_prompt,
                schema=JudgePromptDraft,
                system_prompt=JUDGE_DISTIL_PROMPT,
                max_turns=6,
                refs={
                    "system_prompt": prompt_refs.teacher_prompt_ref(
                        "JUDGE_DISTIL_PROMPT", JUDGE_DISTIL_PROMPT
                    ),
                    "user_prompt": prompt_refs.run_artifact_ref(
                        "distil_prompt.txt", user_prompt, run_id=rec.run_id
                    ),
                    **(
                        {
                            "base_prompt": prompt_refs.judge_prompt_ref(
                                base_version, base_prompt
                            )
                        }
                        if base_version and base_prompt
                        else {}
                    ),
                },
                model=model,
            )
            span.set(
                prompt_chars=len(draft.prompt),
                sections=list(draft.sections),
                notes=draft.notes,
            )
        prompt = normalise_headings(draft.prompt.rstrip("\n") + "\n")
        problems = validate_judge_prompt(prompt)
        rec.dict_artifact(
            "draft.json",
            {"sections": draft.sections, "notes": draft.notes, "problems": problems},
        )
        if problems:
            rec.text_artifact("rejected_prompt.txt", prompt)
            raise SystemExit(
                "the distilled judge prompt failed its contract and was not written:\n  - "
                + "\n  - ".join(problems)
            )
        module_path = _write_judge_module(
            new_version,
            prompt=prompt,
            header=(
                f"{'Revised from ' + base_version + ' against' if base_version else 'Distilled from'} "
                f"{len(rows)} judge diagnoses by the teacher agent ({model}); MLflow run "
                f"{rec.run_id}. Written by `convfinqa-evalloop judge-distil`, never by hand."
            ),
        )
        rec.text_artifact("judge_prompt.txt", prompt)
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        from convfinqa.evalloop import teacher

        teacher._accumulate_usage(usage_total, usage)
        rec.metrics(
            {
                "prompt_chars": float(len(prompt)),
                "n_diagnoses": float(len(rows)),
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        entry = prompt_ledger.ensure_judge(
            new_version,
            source="judge_distil" if not base_version else "judge_revise",
            run_id=rec.run_id,
        )
        registry.register(
            new_version,
            source="judge_distil",
            run_id=rec.run_id,
            notes=f"confidence judge · {entry['seq']} · calibrated to {manifest.get('runtime_version', '')}",
        )
        outcome = None
        if registry.judge_champion() is None:
            outcome = registry.promote_judge(
                new_version,
                runtime_version=str(manifest.get("runtime_version", "")),
                actor="evalloop-judge-distil",
            )
        return {
            "run_id": rec.run_id,
            "module_path": str(module_path),
            "new_version": new_version,
            "base_version": base_version,
            "seq": entry["seq"],
            "prompt_chars": len(prompt),
            "n_diagnoses": len(rows),
            "usage": usage_total,
            "promoted": bool(outcome and outcome.promoted),
        }


# ── Score ─────────────────────────────────────────────────────────────────


SCORE_COLUMNS: tuple[str, ...] = (
    "question_id",
    "report_id",
    "turn_index",
    "gold_turn_type",
    "correct",
    "pred_answer",
    "gold_answer",
    "band",
    "p_correct",
    "reason",
    "checks",
    "error",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)


async def score_split(
    *,
    version: str,
    split: str,
    concurrency: int = 8,
    error_target: float = DEFAULT_ERROR_TARGET,
    experiment: str = JUDGE_EXPERIMENT,
    manifest_path: Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Run judge `version` over every turn of `split`; write the scores CSV and the run.

    A judge call that fails is recorded as a `low` band with the error in the
    row — the guard-rail fails closed, and the row says so — never as a
    silently answered turn.
    """
    from convfinqa.evalloop import teacher
    from convfinqa.llm import judge_model_name, sdk_model_name
    from convfinqa.tracking import mlflow_log, prompt_ledger

    manifest = load_dataset(manifest_path)
    frame = split_frame(manifest, split)
    system_prompt = judge_prompt_of(version)
    seq = prompt_ledger.resolve_judge(version)["seq"]
    model = judge_model_name()
    stamp = _stamp()
    run_name = f"judge-score-{split}{len(frame)}-{version}·{seq}-{stamp}"
    tracing.enable()

    with mlflow_log.run(
        run_name,
        kind="judge_score",
        version=version,
        params={
            "runtime": RUNTIME,
            "runtime_version": manifest.get("runtime_version", ""),
            "runtime_model": sdk_model_name(),
            "judge_version": version,
            "judge_seq": seq,
            "judge_model": model,
            "dataset": manifest["name"],
            "split": split,
            "n_turns": int(len(frame)),
            "error_target": error_target,
            "concurrency": concurrency,
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "judge",
            "stage": "score",
            "runtime": RUNTIME,
            "judge_model": model,
            "judge_version": version,
            "split": split,
        },
        experiment=experiment,
        actor_model=model,
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        sem = asyncio.Semaphore(max(1, concurrency))
        rows = list(frame.iterrows())

        async def one(
            order: int, row: pd.Series
        ) -> tuple[int, dict[str, Any], dict[str, Any]]:
            base = {
                "question_id": str(row.question_id),
                "report_id": str(row.report_id),
                "turn_index": int(row.turn_index),
                "gold_turn_type": str(row.get("gold_turn_type", "") or ""),
                "correct": bool(row.correct),
                "pred_answer": str(row.pred_answer),
                "gold_answer": str(row.gold_answer),
            }
            async with sem:
                with tracing.span(
                    f"judge {row.report_id} q{int(row.turn_index)}",
                    span_type="AGENT",
                    attributes={
                        "runtime": RUNTIME,
                        "report_id": row.report_id,
                        "turn_index": int(row.turn_index),
                        "judge_version": version,
                    },
                    trace_tags={
                        "stage": "judge-score",
                        "runtime": RUNTIME,
                        "judge_version": version,
                        "split": split,
                        "run_name": run_name,
                    },
                ) as span:
                    try:
                        verdict, usage = await judge_turn(
                            row,
                            version=version,
                            system_prompt=system_prompt,
                            model=model,
                        )
                    except Exception as exc:  # noqa: BLE001 — a failed judge fails closed
                        span.set(error=repr(exc), band="low")
                        return (
                            order,
                            {
                                **base,
                                "band": "low",
                                "p_correct": 0.0,
                                "reason": "judge unavailable",
                                "checks": "{}",
                                "error": repr(exc),
                                "latency_ms": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "cost_usd": None,
                            },
                            {},
                        )
                    record = verdict_record(verdict, usage)
                    span.set(
                        band=record["band"],
                        p_correct=record["p_correct"],
                        correct=bool(row.correct),
                        miss=(record["band"] == "high") != bool(row.correct),
                    )
            m = record["metrics"]
            return (
                order,
                {
                    **base,
                    "band": record["band"],
                    "p_correct": record["p_correct"],
                    "reason": record["reason"],
                    "checks": json.dumps(record["checks"]),
                    "error": "",
                    "latency_ms": m.get("latency_ms"),
                    "input_tokens": m.get("input_tokens"),
                    "output_tokens": m.get("output_tokens"),
                    "cost_usd": m.get("cost_usd"),
                },
                usage,
            )

        settled = await asyncio.gather(*(one(i, r) for i, (_, r) in enumerate(rows)))
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        out_rows: list[dict[str, Any]] = []
        for _order, record, usage in sorted(settled, key=lambda r: r[0]):
            out_rows.append(record)
            teacher._accumulate_usage(usage_total, usage)
            mark = "" if (record["band"] == "high") == record["correct"] else "  MISS"
            print(  # noqa: T201
                f"  [{record['report_id']} q{record['turn_index']}] "
                f"{'✓' if record['correct'] else '✗'} → {record['band']} "
                f"(p={record['p_correct']:.2f}){mark}"
            )
        scores = pd.DataFrame(out_rows, columns=list(SCORE_COLUMNS))
        scores_dir().mkdir(parents=True, exist_ok=True)
        csv_path = scores_dir() / f"{run_name}.csv"
        scores.to_csv(csv_path, index=False)
        rec.artifact(csv_path)

        metrics = selective_metrics(scores, error_target=error_target)
        lo, hi = cluster_bootstrap(scores, "high_band_accuracy")
        cap_lo, cap_hi = cluster_bootstrap(scores, "failure_capture")
        metrics["high_band_accuracy_ci"] = [lo, hi]
        metrics["failure_capture_ci"] = [cap_lo, cap_hi]
        metrics["n_judge_errors"] = int((scores["error"].astype(str) != "").sum())
        rec.dict_artifact("metrics.json", metrics)
        rec.dict_artifact(
            "risk_coverage.json",
            {
                "curve": risk_coverage(
                    [float(x) for x in scores["p_correct"]],
                    [bool(x) for x in scores["correct"]],
                )
            },
        )
        rec.metrics(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, int | float)
                and not isinstance(v, bool)
                and v is not None
            }
        )
        rec.metrics({"meets_target": 1.0 if metrics.get("meets_target") else 0.0})
        rec.metrics({f"judge_{k}": v for k, v in usage_total.items()})
        print(  # noqa: T201
            f"\n{run_name}: coverage {metrics['coverage']:.1%} · high band "
            f"{(metrics['high_band_accuracy'] or 0):.2%} correct ({metrics['n_high_wrong']} wrong of "
            f"{metrics['n_high']}) · failures caught {(metrics['failure_capture'] or 0):.1%} "
            f"({metrics['n_failures_caught']}/{metrics['n_wrong']}) · auroc "
            f"{metrics['auroc'] if metrics['auroc'] is None else round(metrics['auroc'], 3)}"
        )
        return {
            "run_id": rec.run_id,
            "run_name": run_name,
            "csv": str(csv_path),
            "version": version,
            "split": split,
            "metrics": metrics,
            "usage": usage_total,
        }


# ── Gate ──────────────────────────────────────────────────────────────────


def gate_judges(
    baseline_scores: Path | str,
    candidate_scores: Path | str,
    *,
    baseline_version: str,
    candidate_version: str,
    error_target: float = DEFAULT_ERROR_TARGET,
) -> dict[str, Any]:
    """Compare two judges on the same split; say whether the candidate replaces the champion.

    The rule: the candidate's high band is inside the error target, its
    coverage is higher, and it catches no fewer failures. Deterministic on
    purpose — the numbers are prevalence-dependent and the split is the same,
    so the paired band flips are the whole story and are recorded beside the
    verdict.
    """
    base = pd.read_csv(baseline_scores)
    cand = pd.read_csv(candidate_scores)
    if set(base["question_id"]) != set(cand["question_id"]):
        raise ValueError("the two scores files do not cover the same questions")
    bm = selective_metrics(base, error_target=error_target)
    cm = selective_metrics(cand, error_target=error_target)
    joined = base.merge(cand, on="question_id", suffixes=("_b", "_c"))
    flips = {
        "released": [
            q
            for q in joined[
                (joined.band_b == "low") & (joined.band_c == "high")
            ].question_id
        ],
        "withheld": [
            q
            for q in joined[
                (joined.band_b == "high") & (joined.band_c == "low")
            ].question_id
        ],
    }
    reasons: list[str] = []
    if not cm["meets_target"]:
        reasons.append(
            f"candidate high band error {cm['high_band_error']:.2%} exceeds the {error_target:.0%} target"
        )
    if cm["coverage"] <= bm["coverage"] and bm["meets_target"]:
        reasons.append(
            f"coverage did not rise ({bm['coverage']:.1%} → {cm['coverage']:.1%})"
        )
    if (cm["failure_capture"] or 0) < (bm["failure_capture"] or 0):
        reasons.append(
            f"failure capture fell ({(bm['failure_capture'] or 0):.1%} → {(cm['failure_capture'] or 0):.1%})"
        )
    promotable = not reasons
    reason = (
        f"high band {cm['high_band_accuracy']:.2%} correct at {cm['coverage']:.1%} coverage, "
        f"failures caught {(cm['failure_capture'] or 0):.1%}"
        + (
            f" (was {bm['coverage']:.1%} coverage, {(bm['failure_capture'] or 0):.1%} caught)"
        )
        if promotable
        else "; ".join(reasons)
    )
    return {
        "gate_id": ledgers.new_id("jg"),
        "gated_at": _now(),
        "runtime": RUNTIME,
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "baseline_scores": str(baseline_scores),
        "candidate_scores": str(candidate_scores),
        "error_target": error_target,
        "n_paired": int(len(joined)),
        "baseline": bm,
        "candidate": cm,
        "flips": flips,
        "n_released": len(flips["released"]),
        "n_withheld": len(flips["withheld"]),
        "promotable": promotable,
        "reason": reason,
    }


def record_gate(
    verdict: dict[str, Any], *, promoted: bool, champion_after: str
) -> dict[str, Any]:
    """Append the verdict to the judge gates record."""
    row = {**verdict, "promoted": promoted, "champion_after": champion_after}
    _append_jsonl(gates_path(), [row])
    return row


# ── Summary for the story ─────────────────────────────────────────────────


def latest_scores(version: str, split: str) -> Path | None:
    """The newest scores CSV for one (version, split), or None."""
    target = scores_dir()
    if not target.exists():
        return None
    matches = sorted(target.glob(f"judge-score-{split}*-{version}·*.csv"))
    return matches[-1] if matches else None


def summary(*, error_target: float = DEFAULT_ERROR_TARGET) -> dict[str, Any]:
    """Everything the story and the campaigns API need, from committed files only."""
    import convfinqa.prompts as prompts_pkg
    from convfinqa.tracking import registry

    try:
        manifest = load_dataset()
    except FileNotFoundError:
        return {"dataset": None, "champion": None, "versions": [], "gates": []}
    versions: list[dict[str, Any]] = []
    for version in prompts_pkg.judge_versions():
        entry: dict[str, Any] = {"version": version, "splits": {}}
        for split in SPLITS:
            path = latest_scores(version, split)
            if path is None:
                continue
            frame = pd.read_csv(path)
            m = selective_metrics(frame, error_target=error_target)
            lo, hi = cluster_bootstrap(frame, "high_band_accuracy", n_boot=400)
            m["high_band_accuracy_ci"] = [lo, hi]
            entry["splits"][split] = {
                "scores_csv": str(path.relative_to(REPO_ROOT)),
                **m,
            }
        versions.append(entry)
    diagnoses = load_diagnoses()
    return {
        "dataset": {k: v for k, v in manifest.items() if k != "splits"},
        "champion": registry.judge_champion(),
        "runtime_version": manifest.get("runtime_version"),
        "error_target": error_target,
        "versions": versions,
        "gates": load_gates(),
        "n_diagnoses": len(diagnoses),
        "n_diagnoses_by_round": {
            str(r): sum(1 for d in diagnoses if d.get("round") == r) for r in (1, 2)
        },
    }


__all__ = (
    "CHECKS",
    "JudgeVerdict",
    "JudgeDiagnosis",
    "build_dataset",
    "load_dataset",
    "split_frame",
    "judge_payload",
    "judge_turn",
    "verdict_record",
    "row_from_capture",
    "selective_metrics",
    "risk_coverage",
    "diagnose_split",
    "distil_judge",
    "validate_judge_prompt",
    "score_split",
    "gate_judges",
    "record_gate",
    "summary",
)
