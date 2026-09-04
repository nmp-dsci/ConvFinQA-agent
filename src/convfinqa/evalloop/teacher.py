"""The teacher: diagnose first-wrong questions, target ONE subagent, rewrite its prompt.

Per cycle the loop is: **diagnose → pick target → rewrite → challenge → gate**.

- *Diagnose*: for every report's FIRST wrong question (later wrongs are cascade,
  not signal), the gold program and gold answer determine which stage first
  diverged — deterministically, for free. That attribution is handed to a
  teacher agent, which explains the failure and may dissent; a dissent is
  recorded as ``attribution_disputed`` and the derived reading is what targets.
- *Pick target*: the subagent with the most derived first-faults. One experiment
  changes ONE subagent, so a champion move is attributable to a specific prompt.
- *Rewrite*: a prompt writer receives that agent's whole current prompt, the
  failures against it, and the ledger of every previous rewrite of the same
  agent **with its gate outcome**, and returns a complete replacement — free to
  reorder, compress, or start over. Its output contract is validated before the
  module is written.
- *Gate*: net positive AND one-sided cluster-corrected McNemar p < 0.05. The
  target agent's own metric is reported beside the verdict as evidence, never as
  a second route to promotion.

Both agents run on the **Claude Agent SDK** (Opus 5, subscription) rather than
in-process pydantic-ai, which is what gives the writer read-only MLflow tools;
the four pipeline agents stay on DeepSeek, because they run every turn and their
cost has to stay measurable per question.

Teacher runs log to their own MLflow experiment (default
``convfinqa-optimization``) with full tracing. Diagnoses, prompts, diffs,
rationales and verdicts are all artifacts on those runs — which is also the
memory: the next cycle reads them back rather than starting blind.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from convfinqa.config import EVAL_ROOT, REPO_ROOT
from convfinqa.tracking import tracing

DIAGNOSTICS_DIR = EVAL_ROOT / "diagnostics" / "evalloop"
OPTIMIZATION_EXPERIMENT = "convfinqa-optimization"
AGENTS = ("triage", "preprocess", "retriever", "calculator")


class Diagnosis(BaseModel):
    """One first-wrong question, attributed and explained.

    ``failed_agent`` is the teacher's own reading. The *gold-derived*
    attribution is computed before the call and given to it; when the two
    disagree the case is recorded as ``attribution_disputed`` and the derived
    one is what drives targeting. The teacher is the better explainer; the gold
    program is the better judge of which stage first diverged from it.
    """

    failed_agent: Literal["triage", "preprocess", "retriever", "calculator"]
    failure_mode: str = Field(
        description="Short label, e.g. 'wrong-base-for-percentage'"
    )
    what_went_wrong: str = Field(
        description="2-4 sentences: the first mistake and how it produced the wrong answer"
    )
    evidence: str = Field(
        description="The specific captured output that shows the mistake"
    )
    attribution_reason: str = Field(
        description=(
            "One or two sentences saying why THIS agent and not the one before "
            "it in the pipeline. If you disagree with derived_attribution, this "
            "is where you justify the disagreement against the gold program."
        )
    )
    proposed_rule: str = Field(
        description="ONE imperative prompt rule for the failed agent that would have prevented this"
    )
    gold_suspect: bool = Field(
        description="True if the gold answer itself looks wrong or ambiguous"
    )
    confidence: float = Field(ge=0.0, le=1.0)


TEACHER_PROMPT = """You are the teacher for a four-stage financial Q&A pipeline
(triage -> preprocess -> retriever -> calculator). You are shown ONE question the
pipeline answered wrongly — always the FIRST wrong turn of its conversation, so
the mistake originated here, not upstream — with every stage's captured input and
output, the gold answer, and the gold reasoning program.

You are given a `derived_attribution` field: which stage first diverged from the
gold program, computed deterministically from gold rather than judged. Treat it
as the default answer. Set `failed_agent` to something else ONLY when you can
point at captured evidence that the derived check misread the case — an
equivalent program shaped differently, an operand that came from history, a gold
answer that is itself wrong. Say so in `what_went_wrong` when you do.

For reference, the stages own these mistakes:
- triage: wrong turn_type (number vs program) or conv_type.
- preprocess: mis-resolved references to conversation history, wrong
  sub-questions, wrong operation chosen.
- retriever: right thing looked up, wrong value returned (wrong row, column,
  year, sign, or scale).
- calculator: right values, wrong computation, wrong base for a percentage,
  sign or rounding errors, wrong final formatting.

A downstream agent that faithfully consumed an upstream mistake did not fail.
Compare against the gold program step by step to locate the divergence.

Every attribution must carry its own justification in `attribution_reason`:
one or two sentences on why this agent and not the one before it. An
attribution with no stated reason cannot be reviewed, and the disagreements
between your reading and the derived one are the cases most worth reviewing.

Use one of these failure modes when it fits (frozen taxonomy, 2026-09-02 —
open-coded from the first battle-test cycles); only when none fits, use
"new:<your-label>" so the gap is visible:
- triage/wrong-turn-type          (number vs program misclassified)
- triage/wrong-conv-type
- preprocess/wrong-operation      (computed what the document reports directly,
                                   or picked the wrong op for the question)
- preprocess/misresolved-reference (history reference resolved to wrong turn)
- preprocess/wrong-output-format  (ratio vs percentage vs absolute form)
- retriever/wrong-period          (right metric, wrong year/column; estimate
                                   used instead of the actual for the asked year)
- retriever/wrong-value           (wrong row or cell, invented adjustment,
                                   sign or scale wrong at lookup)
- calculator/wrong-scale          (thousands/millions/percent scaling)
- calculator/wrong-format         (right number, wrong final form)
- calculator/wrong-computation    (right operands, wrong math or sign)

Then propose ONE targeted rule for the failed agent: imperative, general (not
about this one company), and additive — it must not contradict the agent's
existing instructions. If prior diagnoses are provided, do not repeat a rule
that was already proposed; sharpen or extend instead.

If the gold answer itself looks wrong, still attribute the divergence, set
gold_suspect=true, and lower your confidence."""


PROMPT_WRITER_PROMPT = """You maintain the system prompt of ONE subagent in a
four-stage financial Q&A pipeline (triage -> preprocess -> retriever ->
calculator). You are given that agent's current prompt, the failures diagnosed
against it in the run that just ran, EVERY OTHER failure ever recorded against
this exact prompt text, and the history of every previous rewrite of this same
agent with what the gate said about each — including, for the rewrites that
were rejected, the individual questions they broke.

Return a COMPLETE REPLACEMENT prompt. You may reorder, restructure, compress,
delete, or rewrite from scratch — you are not appending to what is there. A
prompt whose structure is the problem cannot be fixed by adding another rule to
the bottom of it, which is what the previous version of this system could only
ever do.

Hard constraints, all of them load-bearing:
- Preserve the agent's OUTPUT CONTRACT exactly. Every field name, format
  instruction, and DSL operation the current prompt requires must still be
  required. The pipeline parses this agent's output; a rewrite that drops a
  required field breaks every turn, not just the failing ones.
- Stay general. Never mention a specific company, year, or value from the
  failures. You are writing instructions, not patching cases.
- Change ONE agent. You are only shown one; do not write instructions that
  presuppose changes to another stage.
- Read the attempt history before writing. If a change was already REJECTED,
  do not propose it again unless you can say what is different this time.
- Weigh what past rewrites BROKE as heavily as what they fixed. Nearly every
  rejection here was net-positive-but-not-significant: the fixes were real and
  the collateral damage cancelled them out. The questions listed under BROKE
  are ones this agent used to get right. Your rewrite has to keep getting them
  right, so prefer a narrow rule with a stated precondition over a broad
  instruction that changes behaviour on turns that were never failing.
- `failures_this_run` is a sample; `failures_same_prompt` is the accumulated
  record of what this exact prompt gets wrong across every run that used it.
  Treat a mode that recurs across both as the real target and a mode appearing
  once as possible noise.

You have read-only tools for the record: `search_attempts` (rewrite history and
outcomes), `get_prompt` (any past version's prompt for any agent), and
`get_failures` (the diagnosed cases behind a fault count). Use them when the
baked context is not enough; you are not required to.

Return JSON matching the schema: the full prompt, a one-paragraph rationale, and
a short summary of what you changed."""


class PromptRewrite(BaseModel):
    """A complete replacement prompt for one subagent, with its reasoning."""

    prompt: str = Field(
        description="The complete replacement system prompt for the target agent"
    )
    rationale: str = Field(
        description=(
            "One paragraph: what you judged to be wrong with the current prompt "
            "and why this rewrite should fix it. Becomes the caption on the "
            "promotion record."
        )
    )
    summary_of_changes: str = Field(
        description="One or two sentences naming what actually changed"
    )


def first_wrong_cases(csv_path: Path | str) -> pd.DataFrame:
    """First wrong question per report — the only rows that carry fresh signal.

    Scored on the way through, so every case carries its gold-derived checks
    whether or not the CSV was written by a runner that computed them.
    """
    from convfinqa.evalloop import stage_scores

    df = pd.read_csv(csv_path)
    df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
    if "triage_turn_type_ok" not in df.columns:
        stage_scores.score_rows(df)
    return df[df.turn_index == df.first_wrong_turn].copy()


def _stage_io(row: pd.Series, stage: str) -> Any:
    raw = row.get(f"{stage}_io")
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(loaded, dict):
            loaded.pop("metrics", None)  # latency/tokens — noise for diagnosis
        return loaded
    return None


def case_payload(row: pd.Series) -> dict[str, Any]:
    """Everything the teacher needs about one case, and nothing else."""
    from convfinqa.evalloop import stage_scores

    return {
        "derived_attribution": stage_scores.attribute(row),
        "derived_checks": {
            "triage_turn_type_ok": row.get("triage_turn_type_ok"),
            "preprocess_skeleton_ok": row.get("preprocess_skeleton_ok"),
            "retriever_operand_recall": row.get("retriever_operand_recall"),
            "calculator_ok": row.get("calculator_ok"),
        },
        "report_id": row.report_id,
        "question": row.question,
        "conversation_history": row.get("history_text") or "(no prior turns)",
        "gold_answer": row.gold_answer,
        "gold_program": row.get("gold_program") or "(number selection — no program)",
        "pipeline_answer": row.pred_answer,
        "pipeline_program": row.get("pred_program") or "",
        "stages": {s: _stage_io(row, s) for s in AGENTS},
    }


async def _diagnose_case(
    payload: dict[str, Any], memory_text: str
) -> tuple[Diagnosis, dict[str, Any]]:
    """One diagnosis, on the Agent SDK, validated against the same schema."""
    from convfinqa.evalloop.sdk import run_structured

    return await run_structured(
        json.dumps(payload, default=str) + memory_text,
        schema=Diagnosis,
        system_prompt=TEACHER_PROMPT,
        max_turns=4,
    )


def prior_diagnoses(experiment: str, limit_runs: int = 5) -> list[dict[str, Any]]:
    """Diagnoses from previous teacher runs, read back from MLflow.

    This is the loop's memory: what was already attributed and proposed feeds
    the next teacher run so it extends instead of repeating.
    """
    try:
        from mlflow.tracking import MlflowClient

        from convfinqa.tracking import mlflow_log

        mlflow_log._mlflow()
        client = MlflowClient(tracking_uri=mlflow_log.tracking_uri())
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return []
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.kind = 'diagnose'",
            order_by=["attributes.start_time DESC"],
            max_results=limit_runs,
        )
        out: list[dict[str, Any]] = []
        for r in runs:
            try:
                local = client.download_artifacts(r.info.run_id, "diagnoses.jsonl")
                for line in Path(local).read_text().splitlines():
                    d = json.loads(line)
                    out.append(
                        {
                            "run_name": r.data.tags.get("mlflow.runName", ""),
                            "version": r.data.params.get("prompts_version", ""),
                            "report_id": d.get("report_id"),
                            "failed_agent": d.get("failed_agent"),
                            "failure_mode": d.get("failure_mode"),
                            "proposed_rule": d.get("proposed_rule"),
                        }
                    )
            except Exception:  # noqa: BLE001 — one unreadable run must not block diagnosis
                continue
        return out
    except Exception:  # noqa: BLE001
        return []


async def diagnose_run(
    csv_path: Path | str,
    version: str,
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Diagnose every first-wrong case of one eval run; return the summary.

    Cases run concurrently under a semaphore, matching the eval runner. They are
    independent — no diagnosis reads another's result — so the serial version
    only ever converted that independence into wall clock.
    """
    from convfinqa.tracking import mlflow_log

    cases = first_wrong_cases(csv_path)
    memory = prior_diagnoses(experiment)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"diagnose-{version}-{stamp}"
    tracing.enable()

    memory_text = ""
    if memory:
        lines = [
            f"- [{m['version']}] {m['failed_agent']}/{m['failure_mode']}: {m['proposed_rule']}"
            for m in memory[:40]
        ]
        memory_text = "\n\nPrior diagnoses (do not repeat these rules):\n" + "\n".join(
            lines
        )

    diagnoses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
    with mlflow_log.run(
        run_name,
        kind="diagnose",
        version=version,
        params={
            "source_csv": str(csv_path),
            "n_cases": len(cases),
            "n_prior_diagnoses": len(memory),
            "concurrency": concurrency,
        },
        tags={"loop": "evalloop", "stage": "diagnose"},
        experiment=experiment,
        # The fingerprint describes the bundle being diagnosed, not the agent
        # doing the diagnosing — so `lm_max` (the configured optimiser model)
        # names something this run never calls.
        actor_model=teacher_model(),
        omit_fingerprint=("lm_max",),
    ) as rec:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def one(
            order: int, row: pd.Series
        ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
            """Diagnose one case. Never raises — a bad case is counted, not fatal."""
            payload = case_payload(row)
            derived = str(payload["derived_attribution"])
            async with sem:
                with tracing.span(
                    f"diagnose {row.report_id} q{int(row.turn_index)}",
                    span_type="AGENT",
                    attributes={
                        "report_id": row.report_id,
                        "turn_index": int(row.turn_index),
                        "question": str(row.question),
                        "gold_answer": str(row.gold_answer),
                        "pipeline_answer": str(row.pred_answer),
                        # What gold says failed, before the teacher is asked.
                        "derived_attribution": derived,
                    },
                    trace_tags={
                        "model_version_id": version,
                        "run_name": run_name,
                        "stage": "diagnose",
                    },
                ) as trace_span:
                    try:
                        output, usage = await _diagnose_case(payload, memory_text)
                    except Exception as exc:  # noqa: BLE001 — one bad case must not sink the pass
                        # The runner already takes this position for
                        # conversations, and a diagnosis pass is worth more:
                        # fifty calls and the whole cycle downstream of them. A
                        # case that cannot be diagnosed is counted and skipped,
                        # so the target is picked from the cases that *did*
                        # work rather than from nothing at all.
                        return (
                            order,
                            None,
                            {"report_id": row.report_id, "error": repr(exc)},
                            {},
                        )
                # The span opened knowing only what gold derived. Now that the
                # teacher has answered, put its verdict *and its reasoning* on
                # the same span — an attribution with no stated reason is not
                # reviewable, and reviewing the disputes is the whole point of
                # recording them.
                trace_span.set(
                    failed_agent=output.failed_agent,
                    failure_mode=output.failure_mode,
                    reason=output.attribution_reason,
                    what_went_wrong=output.what_went_wrong,
                    evidence=output.evidence,
                    proposed_rule=output.proposed_rule,
                    confidence=float(output.confidence),
                    gold_suspect=bool(output.gold_suspect),
                    attribution_disputed=output.failed_agent != derived,
                )
            d = output.model_dump()
            d.update(
                report_id=row.report_id,
                question_id=row.get("question_id", ""),
                turn_index=int(row.turn_index),
                version=version,
                derived_agent=derived,
                attribution_disputed=d["failed_agent"] != derived,
            )
            return order, d, None, usage

        # The cases are independent by construction — one diagnosis never reads
        # another's result — so running them serially bought nothing and cost
        # the whole pass: fifty calls at ~27s each is over twenty minutes of
        # wall clock, which was the single largest term in a cycle.
        settled = await asyncio.gather(
            *(one(i, row) for i, (_, row) in enumerate(cases.iterrows()))
        )
        # Reassembled in case order, so the JSONL artifact and the printed log
        # do not depend on which call happened to return first.
        for _, d, failure, usage in sorted(settled, key=lambda r: r[0]):
            if failure is not None:
                failures.append(failure)
                print(f"  [skip] {failure['report_id']}: {failure['error']}")  # noqa: T201
                continue
            assert d is not None
            _accumulate_usage(usage_total, usage)
            diagnoses.append(d)
            mark = " DISPUTED" if d["attribution_disputed"] else ""
            print(  # noqa: T201
                f"  [{d['report_id']} q{d['turn_index']}] gold->{d['derived_agent']}"
                f" teacher->{d['failed_agent']}{mark}"
                f" · {d['failure_mode']} (conf {d['confidence']:.2f})"
            )

        # Targeting runs off the *derived* attribution, not the teacher's: the
        # gold program is the better judge of which stage first diverged, and
        # the two agree on only about half of cases. The teacher's reading is
        # kept beside it as evidence, and every disagreement is counted.
        counts = {
            a: sum(1 for d in diagnoses if d["derived_agent"] == a) for a in AGENTS
        }
        teacher_counts = {
            a: sum(1 for d in diagnoses if d["failed_agent"] == a) for a in AGENTS
        }
        n_disputed = sum(1 for d in diagnoses if d["attribution_disputed"])
        target = max(counts, key=lambda a: counts[a]) if diagnoses else None
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DIAGNOSTICS_DIR / f"diagnoses_{version}_{stamp}.jsonl"
        out_path.write_text("".join(json.dumps(d) + "\n" for d in diagnoses))
        rec.artifact(out_path)
        # Also under the canonical artifact name prior_diagnoses() reads back.
        rec.dict_artifact("summary.json", {"counts": counts, "target": target})
        _log_jsonl_artifact(rec, diagnoses)
        rec.metrics(
            {
                "n_diagnosed": float(len(diagnoses)),
                "n_diagnose_failures": float(len(failures)),
                "n_attribution_disputed": float(n_disputed),
                "attribution_agreement": round(1 - n_disputed / len(diagnoses), 4)
                if diagnoses
                else 0.0,
                **{f"faults_{a}": float(counts[a]) for a in AGENTS},
                **{f"teacher_faults_{a}": float(teacher_counts[a]) for a in AGENTS},
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        if failures and len(failures) > len(diagnoses):
            raise SystemExit(
                f"{len(failures)} of {len(failures) + len(diagnoses)} cases failed "
                "to diagnose — that is not a flaky call, it is a broken teacher, "
                "and targeting off the remainder would be picking from noise"
            )
        summary = {
            "run_name": run_name,
            "run_id": rec.run_id,
            "version": version,
            "n_cases": len(diagnoses),
            "n_failures": len(failures),
            "failures": failures,
            "counts": counts,
            "teacher_counts": teacher_counts,
            "n_attribution_disputed": n_disputed,
            "target": target,
            "diagnoses_path": str(out_path),
            "gold_suspects": [d["report_id"] for d in diagnoses if d["gold_suspect"]],
            "teacher_usage": usage_total,
        }
    return summary


def teacher_model() -> str:
    """The model the teacher runs on — recorded on every teacher run."""
    from convfinqa.llm import LM_TEACHER_MODEL

    return LM_TEACHER_MODEL


def _accumulate_usage(total: dict[str, float], usage: dict[str, Any]) -> None:
    """Fold one SDK call's usage into a run total.

    Teacher usage was previously recorded nowhere at all. It is the campaign's
    real throughput limit — roughly fifty calls a cycle, five cycles a campaign —
    so it goes on the run beside the pipeline's cost, never inside it: one is
    dollars per question, the other is subscription consumption, and adding them
    would make the economics read better than they are.
    """
    raw = usage.get("usage") or {}
    if isinstance(raw, dict):
        total["input_tokens"] += float(raw.get("input_tokens", 0) or 0)
        total["output_tokens"] += float(raw.get("output_tokens", 0) or 0)
    total["cost_usd"] += float(usage.get("total_cost_usd") or 0.0)


def _log_jsonl_artifact(rec: Any, diagnoses: list[dict[str, Any]]) -> None:
    """Store diagnoses.jsonl on the run under the exact name the memory reads."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "diagnoses.jsonl"
        p.write_text("".join(json.dumps(d) + "\n" for d in diagnoses))
        rec.artifact(p)


async def propose_version(
    diagnoses_path: Path | str,
    *,
    base_version: str,
    new_version: str,
    target: str | None = None,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    campaign: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Write a generated prompts module changing ONE agent, and register it.

    The writer gets three things the M2 version did not: the whole current
    prompt to replace rather than append to, the ledger of every previous
    rewrite of this same agent with its gate outcome, and read-only tools to
    dig further into the record. What it returns is a complete prompt plus the
    reasoning behind it, both logged — the rationale becomes the caption on the
    promotion record, which is what makes a champion move explicable later.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop import ledger, tools
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.tracking import mlflow_log, registry

    diagnoses = [
        json.loads(line)
        for line in Path(diagnoses_path).read_text().splitlines()
        if line.strip()
    ]

    # Derived attribution is what targets; the teacher's own reading is evidence.
    def _agent_of(d: dict[str, Any]) -> str:
        return str(d.get("derived_agent") or d["failed_agent"])

    counts = {a: sum(1 for d in diagnoses if _agent_of(d) == a) for a in AGENTS}
    target = target or max(counts, key=lambda a: counts[a])
    targeted = [d for d in diagnoses if _agent_of(d) == target]
    if not targeted:
        raise SystemExit(f"no diagnoses attribute a fault to {target!r}")

    base_prompts = prompts_pkg.load(base_version)
    history = ledger.ledger_text(target)
    n_prior = len(ledger.attempts(target_agent=target, limit=50))
    # Every failure ever recorded against the exact prompt text being replaced,
    # not just the ones this cycle's run happened to sample. Cases already in
    # `targeted` are dropped so the writer does not read one turn twice and
    # mistake a duplicate for a recurring mode.
    seen_cases = {
        (str(d.get("report_id")), int(d.get("turn_index", -1))) for d in targeted
    }
    prior_failures = [
        d
        for d in ledger.diagnoses_for_agent(target, base_version, experiment=experiment)
        if (str(d.get("report_id")), int(d.get("turn_index", -1))) not in seen_cases
    ]
    tracing.enable()
    with mlflow_log.run(
        f"propose-{new_version}-{target}",
        kind="propose",
        version=base_version,
        params={
            "target_agent": target,
            "new_version": new_version,
            "n_diagnoses": len(targeted),
            "n_prior_attempts": n_prior,
            "n_failures_same_prompt": len(prior_failures),
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "propose",
            "target_agent": target,
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        actor_model=teacher_model(),
        omit_fingerprint=("lm_max",),
    ) as rec:
        with tracing.span(
            f"propose {new_version} ({target})",
            span_type="AGENT",
            attributes={
                "target_agent": target,
                "base_version": base_version,
                "new_version": new_version,
                "n_failures_this_run": len(targeted),
                "n_failures_same_prompt": len(prior_failures),
                "n_prior_attempts": n_prior,
            },
            trace_tags={"stage": "propose", "target_agent": target},
        ) as propose_span:
            output, usage = await run_structured(
                json.dumps(
                    {
                        "target_agent": target,
                        "current_prompt": base_prompts[target],
                        "failures_same_prompt": prior_failures,
                        "failures_this_run": [
                            {
                                "failure_mode": d["failure_mode"],
                                "what_went_wrong": d["what_went_wrong"],
                                "proposed_rule": d["proposed_rule"],
                                "gold_suspect": d.get("gold_suspect"),
                            }
                            for d in targeted
                        ],
                    },
                    default=str,
                )
                + history,
                schema=PromptRewrite,
                system_prompt=PROMPT_WRITER_PROMPT,
                mcp_servers={"loop": tools.loop_server()},
                allowed_tools=tools.ALLOWED_TOOLS,
                max_turns=20,
            )
            propose_span.set(
                summary_of_changes=output.summary_of_changes,
                rationale=output.rationale,
                prompt_chars_before=len(base_prompts[target]),
                prompt_chars_after=len(output.prompt),
            )

        problems = validate_prompt(target, base_prompts[target], output.prompt)
        if problems:
            raise SystemExit(
                "the rewrite failed its output contract and was not written:\n  - "
                + "\n  - ".join(problems)
            )

        module_path = _write_version_module(
            new_version, base_version=base_version, target=target, prompt=output.prompt
        )
        diff = prompt_diff(base_prompts[target], output.prompt, target=target)
        rec.dict_artifact(
            "proposal.json",
            {
                "target": target,
                "base_version": base_version,
                "new_version": new_version,
                "prompt": output.prompt,
                "rationale": output.rationale,
                "summary_of_changes": output.summary_of_changes,
                "module": str(module_path),
                "tools_used": usage.get("tools_used", []),
            },
        )
        rec.dict_artifact("prompt_diff.json", {"target": target, "diff": diff})
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        _accumulate_usage(usage_total, usage)
        rec.metrics(
            {
                "prompt_chars_before": float(len(base_prompts[target])),
                "prompt_chars_after": float(len(output.prompt)),
                "n_prior_attempts": float(n_prior),
                "n_tool_calls": float(len(usage.get("tools_used", []))),
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        from convfinqa.tracking import prompt_ledger

        comp = prompt_ledger.ensure(new_version, source="teacher", run_id=rec.run_id)
        registry.register(
            new_version,
            source="evalloop-teacher",
            run_id=rec.run_id,
            notes=(
                f"targeted challenger: only {target} rewritten (parent "
                f"{base_version}). {output.summary_of_changes}"
            ),
            extra={
                "parent": base_version,
                "target_agent": target,
                "rationale": output.rationale,
                "changed_agents": prompt_ledger.changed_agents(
                    base_version, new_version
                ),
                "composition": prompt_ledger.composition_string(comp),
                **({"campaign": campaign} if campaign else {}),
            },
        )
    return {
        "new_version": new_version,
        "target": target,
        "rationale": output.rationale,
        "summary_of_changes": output.summary_of_changes,
        "n_prior_attempts": n_prior,
        "tools_used": usage.get("tools_used", []),
        "chars": {
            "before": len(base_prompts[target]),
            "after": len(output.prompt),
        },
        "module": str(module_path),
        "propose_run_id": rec.run_id,
    }


# The tokens each agent's output contract depends on. A rewrite is free to say
# anything it likes about *how* to do the job, and nothing at all about the shape
# of what it returns — the pipeline parses that, so dropping one of these breaks
# every turn rather than only the failing ones. Checked before the module is
# written, so a bad rewrite costs nothing.
CONTRACT_TOKENS: dict[str, tuple[str, ...]] = {
    "triage": ("turn_type", "conv_type"),
    "preprocess": ("sub_question",),
    "retriever": ("answer",),
    "calculator": ("add", "subtract", "multiply", "divide"),
}

MIN_PROMPT_CHARS = 200


def validate_prompt(agent: str, before: str, after: str) -> list[str]:
    """Reasons this rewrite must not be written to disk. Empty means it may."""
    problems: list[str] = []
    if len(after.strip()) < MIN_PROMPT_CHARS:
        problems.append(
            f"the rewrite is {len(after.strip())} characters — under the "
            f"{MIN_PROMPT_CHARS}-character floor, which is what a collapsed "
            "prompt looks like"
        )
    lowered = after.lower()
    for token in CONTRACT_TOKENS.get(agent, ()):
        if token.lower() in before.lower() and token.lower() not in lowered:
            problems.append(
                f"the current prompt requires {token!r} and the rewrite does "
                "not mention it — that is the agent's output contract"
            )
    return problems


def prompt_diff(before: str, after: str, *, target: str) -> str:
    """Unified diff of one agent's prompt, for the promotion record."""
    import difflib

    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{target}/before",
            tofile=f"{target}/after",
            n=3,
        )
    )


_AGENT_VARS = {
    "triage": "TRIAGE_PROMPT",
    "preprocess": "PREPROCESS_PROMPT",
    "retriever": "RETRIEVER_PROMPT",
    "calculator": "CALCULATOR_PROMPT",
}


def _write_version_module(
    new_version: str, *, base_version: str, target: str, prompt: str
) -> Path:
    """Generated module: three prompts imported unchanged, one replaced outright.

    The three untouched prompts are *imported* rather than copied, so the diff
    between consecutive champions is exactly one agent's prompt — which is what
    lets a story page attribute a move to a specific change instead of asserting
    it.
    """
    var = _AGENT_VARS[target]
    others = ",\n    ".join(v for k, v in _AGENT_VARS.items() if k != target)
    # The prompt goes into a triple-quoted literal, so the two sequences that
    # could end it early are neutralised. Readability is the point of the
    # triple quote — a repr would be safe too and unreadable, and these modules
    # are meant to be read when a promotion is questioned.
    literal = prompt.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if literal.endswith('"'):
        literal += "\\n"
    body = f'''"""Generated by convfinqa.evalloop.teacher — do not hand-edit.

Targeted challenger for {base_version}: only the {target} prompt changes, and it
is replaced rather than extended. Regenerate via `convfinqa-evalloop propose`.
"""

from convfinqa.prompts.{base_version} import (
    {others},
)

__all__ = [
    "TRIAGE_PROMPT",
    "PREPROCESS_PROMPT",
    "RETRIEVER_PROMPT",
    "CALCULATOR_PROMPT",
]

{var} = """{literal}"""
'''
    path = REPO_ROOT / "src" / "convfinqa" / "prompts" / f"{new_version}.py"
    if path.exists():
        raise SystemExit(f"{path} already exists — pick a new version name")
    path.write_text(body)
    return path


def gate_targeted(
    baseline_csv: Path | str,
    candidate_csv: Path | str,
    *,
    target_agent: str,
    baseline_version: str,
    candidate_version: str,
    baseline_diagnoses: Path | str | None = None,
    candidate_diagnoses: Path | str | None = None,
) -> tuple[dict[str, Any], Any]:
    """The campaign's promotion rule, with the target agent's evidence beside it.

    One rule decides: **net positive on the shared questions AND one-sided
    cluster-corrected McNemar p < 0.05**. The per-agent metric no longer offers a
    second path to promotion — under M2 it did, and that is how three challengers
    were promoted on evidence whose confidence interval contained zero. It is
    reported here because it answers a different and still-useful question: *did
    the change do what it was supposed to do to the agent it targeted?* A
    challenger that moves its agent's metric but fails the gate is a real finding
    about a sample too small to see it, not a promotion.
    """
    from convfinqa.evalloop import stage_scores
    from convfinqa.evalloop.gate import gate_reason, gate_runs, load_run_csv

    result, stats = gate_runs(
        baseline_csv,
        candidate_csv,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
    )
    metric_name = stage_scores.TARGET_METRIC[target_agent]
    base_panel = stage_scores.run_metrics(load_run_csv(baseline_csv))
    cand_panel = stage_scores.run_metrics(load_run_csv(candidate_csv))
    metric_before = base_panel.get(metric_name)
    metric_after = cand_panel.get(metric_name)

    def _faults(path: Path | str) -> int:
        rows = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        return sum(
            1
            for d in rows
            if str(d.get("derived_agent") or d["failed_agent"]) == target_agent
        )

    base_faults = _faults(baseline_diagnoses) if baseline_diagnoses else None
    cand_faults = _faults(candidate_diagnoses) if candidate_diagnoses else None

    if metric_before is not None and metric_after is not None:
        target_moved = metric_after > metric_before
        target_evidence = f"{metric_name} {metric_before:.3f} → {metric_after:.3f}"
    elif base_faults is not None and cand_faults is not None:
        target_moved = cand_faults < base_faults
        target_evidence = f"first-faults {base_faults} → {cand_faults} (attribution)"
    else:
        target_moved = False
        target_evidence = f"no evidence available for {metric_name}"

    verdict = {
        "target_agent": target_agent,
        "target_metric": metric_name,
        "target_metric_before": metric_before,
        "target_metric_after": metric_after,
        "target_metric_delta": (
            round(metric_after - metric_before, 6)
            if metric_before is not None and metric_after is not None
            else None
        ),
        "baseline_target_faults": base_faults,
        "candidate_target_faults": cand_faults,
        "target_moved": target_moved,
        "target_evidence": target_evidence,
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "overall_delta": stats["accuracy_delta"],
        "evidence_split": stats["evidence_split"],
        "promotable": stats["promotable"],
        "cluster_p_one_sided": stats["cluster_p_one_sided"],
        "agent_panel_baseline": base_panel,
        "agent_panel_candidate": cand_panel,
        "comparison": stats,
        "reason": f"{gate_reason(stats)} — target: {target_evidence}",
    }
    return verdict, result


def log_gate_verdict(
    verdict: dict[str, Any],
    *,
    comparison: Any = None,
    campaign: str | None = None,
    label: str | None = None,
    experiment: str = OPTIMIZATION_EXPERIMENT,
) -> str:
    """Record one gate decision as an MLflow run, so the ledger can read it back.

    Without this the loop had no memory of outcomes at all — proposals were
    logged and verdicts were printed to a terminal. A rejected idea could
    therefore be proposed again next cycle, indefinitely. This is the run the
    prompt writer's ledger joins its proposals against.

    `comparison` is the `ComparisonResult`, and passing it writes `flips.json`:
    the individual questions the candidate fixed and, more to the point, the
    ones it broke. Counts alone tell the next writer that its predecessor broke
    eighteen questions; only the questions themselves tell it *which* eighteen,
    which is the difference between "be more careful" and a specific rule that
    stops a specific kind of collateral damage.
    """
    from convfinqa.tracking import mlflow_log

    stats = verdict["comparison"]
    with mlflow_log.run(
        f"gate-{verdict['candidate_version']}-vs-{verdict['baseline_version']}",
        kind="gate",
        version=verdict["candidate_version"],
        params={
            "baseline_version": verdict["baseline_version"],
            "candidate_version": verdict["candidate_version"],
            "target_agent": verdict["target_agent"],
            "evidence_split": verdict["evidence_split"],
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "gate",
            "promoted": "true" if verdict["promotable"] else "false",
            "target_agent": verdict["target_agent"],
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        # A gate verdict is pure arithmetic over two committed CSVs — it makes
        # no model calls of any kind, so neither model belongs on it.
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        rec.dict_artifact(
            "verdict.json",
            {
                "promoted": bool(verdict["promotable"]),
                "reason": verdict["reason"],
                **{
                    k: stats[k]
                    for k in (
                        "accuracy_delta",
                        "cluster_p_one_sided",
                        "n_compared",
                        "fail_to_pass",
                        "pass_to_fail",
                        "delta_ci_lo",
                        "delta_ci_hi",
                        "delta_p_positive",
                    )
                },
            },
        )
        if comparison is not None:
            rec.dict_artifact(
                "flips.json",
                {
                    "broken": [f.as_dict() for f in comparison.regressions],
                    "fixed": [f.as_dict() for f in comparison.improvements],
                },
            )
        rec.metrics(
            {
                "accuracy_delta": float(stats["accuracy_delta"]),
                "cluster_p_one_sided": float(stats["cluster_p_one_sided"]),
                "n_compared": float(stats["n_compared"]),
                "fail_to_pass": float(stats["fail_to_pass"]),
                "pass_to_fail": float(stats["pass_to_fail"]),
                "delta_ci_lo": float(stats["delta_ci_lo"]),
                "delta_ci_hi": float(stats["delta_ci_hi"]),
                "promoted": 1.0 if verdict["promotable"] else 0.0,
                **(
                    {"target_metric_delta": float(verdict["target_metric_delta"])}
                    if verdict.get("target_metric_delta") is not None
                    else {}
                ),
            }
        )
        return str(rec.run_id)
