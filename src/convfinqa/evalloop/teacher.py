"""The teacher (M2): diagnose first-wrong questions, target ONE subagent, propose a fix.

Per eval run, the loop is: **diagnose → pick target → propose → challenge → gate**.

- *Diagnose*: for every report's FIRST wrong question (later wrongs are cascade,
  not signal), a teacher LLM reads the four stage captures plus the gold answer
  and gold program, names the subagent that made the first mistake, and proposes
  one targeted prompt rule for that agent.
- *Pick target*: the subagent with the most attributed first-faults — one
  optimisation changes ONE subagent, so a challenger is attributable.
- *Propose*: merge that agent's proposed rules into a rules block and write a
  generated prompts module (base version's other three prompts imported
  unchanged).
- *Gate* (targeted): the challenger promotes when the target agent's first-fault
  count drops on the same reports AND overall paired accuracy does not regress.

Teacher runs log to their own MLflow experiment (default
``convfinqa-optimization``) with full tracing, and every diagnosis is written
both to ``evaluation/diagnostics/evalloop/`` and onto the run as an artifact —
which is also the memory: later teacher runs read prior diagnoses back from
MLflow and see what was already tried.
"""

from __future__ import annotations

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
    """One first-wrong question, attributed and explained."""

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

Attribute the FIRST mistake to exactly one subagent:
- triage: wrong turn_type (number vs program) or conv_type.
- preprocess: mis-resolved references to conversation history, wrong
  sub-questions, wrong operation chosen.
- retriever: right thing looked up, wrong value returned (wrong row, column,
  year, sign, or scale).
- calculator: right values, wrong computation, wrong base for a percentage,
  sign or rounding errors, wrong final formatting.

A downstream agent that faithfully consumed an upstream mistake did not fail.
Compare against the gold program step by step to locate the divergence.

Then propose ONE targeted rule for the failed agent: imperative, general (not
about this one company), and additive — it must not contradict the agent's
existing instructions. If prior diagnoses are provided, do not repeat a rule
that was already proposed; sharpen or extend instead.

If the gold answer itself looks wrong, still attribute the divergence, set
gold_suspect=true, and lower your confidence."""


PROMPT_WRITER_PROMPT = """You maintain the system prompt of one subagent in a
financial Q&A pipeline. You get the agent's current prompt and a list of
diagnosed failures with proposed rules. Merge the proposals into at most five
crisp, imperative rules. Do not repeat anything the current prompt already
says; drop proposals it already covers. Rules must be general — never mention
specific companies, years, or values from the failures. Return only the rules,
one per line, no numbering commentary."""


class RulesBlock(BaseModel):
    """The merged targeted rules for one agent."""

    rules: list[str] = Field(description="At most 5 imperative prompt rules")


def first_wrong_cases(csv_path: Path | str) -> pd.DataFrame:
    """First wrong question per report — the only rows that carry fresh signal."""
    df = pd.read_csv(csv_path)
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
    return {
        "report_id": row.report_id,
        "question": row.question,
        "conversation_history": row.get("history_text") or "(no prior turns)",
        "gold_answer": row.gold_answer,
        "gold_program": row.get("gold_program") or "(number selection — no program)",
        "pipeline_answer": row.pred_answer,
        "pipeline_program": row.get("pred_program") or "",
        "stages": {s: _stage_io(row, s) for s in AGENTS},
    }


def _teacher_agent() -> Any:
    from pydantic_ai import Agent

    from convfinqa.backends.pydantic import lm_max

    return Agent(
        lm_max(), output_type=Diagnosis, instructions=TEACHER_PROMPT, name="teacher"
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
) -> dict[str, Any]:
    """Diagnose every first-wrong case of one eval run; return the summary."""
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

    agent = _teacher_agent()
    diagnoses: list[dict[str, Any]] = []
    with mlflow_log.run(
        run_name,
        kind="diagnose",
        version=version,
        params={
            "source_csv": str(csv_path),
            "n_cases": len(cases),
            "n_prior_diagnoses": len(memory),
        },
        tags={"loop": "evalloop", "stage": "diagnose"},
        experiment=experiment,
    ) as rec:
        for _, row in cases.iterrows():
            payload = case_payload(row)
            with tracing.span(
                f"diagnose {row.report_id} q{int(row.turn_index)}",
                attributes={
                    "report_id": row.report_id,
                    "turn_index": int(row.turn_index),
                },
                trace_tags={
                    "model_version_id": version,
                    "run_name": run_name,
                    "stage": "diagnose",
                },
            ):
                result = await agent.run(json.dumps(payload, default=str) + memory_text)
            d = result.output.model_dump()
            d.update(
                report_id=row.report_id,
                question_id=row.get("question_id", ""),
                turn_index=int(row.turn_index),
                version=version,
            )
            diagnoses.append(d)
            print(  # noqa: T201
                f"  [{row.report_id} q{int(row.turn_index)}] -> {d['failed_agent']}"
                f" · {d['failure_mode']} (conf {d['confidence']:.2f})"
            )

        counts = {
            a: sum(1 for d in diagnoses if d["failed_agent"] == a) for a in AGENTS
        }
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
                **{f"faults_{a}": float(counts[a]) for a in AGENTS},
            }
        )
        summary = {
            "run_name": run_name,
            "run_id": rec.run_id,
            "version": version,
            "n_cases": len(diagnoses),
            "counts": counts,
            "target": target,
            "diagnoses_path": str(out_path),
            "gold_suspects": [d["report_id"] for d in diagnoses if d["gold_suspect"]],
        }
    return summary


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
) -> dict[str, Any]:
    """Write a generated prompts module changing ONE agent, and register it."""
    from pydantic_ai import Agent

    import convfinqa.prompts as prompts_pkg
    from convfinqa.backends.pydantic import lm_max
    from convfinqa.tracking import mlflow_log, registry

    diagnoses = [
        json.loads(line)
        for line in Path(diagnoses_path).read_text().splitlines()
        if line.strip()
    ]
    counts = {a: sum(1 for d in diagnoses if d["failed_agent"] == a) for a in AGENTS}
    target = target or max(counts, key=lambda a: counts[a])
    targeted = [d for d in diagnoses if d["failed_agent"] == target]
    if not targeted:
        raise SystemExit(f"no diagnoses attribute a fault to {target!r}")

    base_prompts = prompts_pkg.load(base_version)
    writer = Agent(
        lm_max(),
        output_type=RulesBlock,
        instructions=PROMPT_WRITER_PROMPT,
        name="prompt_writer",
    )
    tracing.enable()
    with mlflow_log.run(
        f"propose-{new_version}-{target}",
        kind="propose",
        version=base_version,
        params={
            "target_agent": target,
            "new_version": new_version,
            "n_diagnoses": len(targeted),
        },
        tags={"loop": "evalloop", "stage": "propose"},
        experiment=experiment,
    ) as rec:
        with tracing.span(
            f"propose {new_version} ({target})",
            trace_tags={"stage": "propose", "target_agent": target},
        ):
            result = await writer.run(
                json.dumps(
                    {
                        "current_prompt": base_prompts[target],
                        "failures": [
                            {
                                "failure_mode": d["failure_mode"],
                                "what_went_wrong": d["what_went_wrong"],
                                "proposed_rule": d["proposed_rule"],
                            }
                            for d in targeted
                        ],
                    }
                )
            )
        rules = result.output.rules
        module_path = _write_version_module(
            new_version, base_version=base_version, target=target, rules=rules
        )
        rec.dict_artifact(
            "proposal.json",
            {"target": target, "rules": rules, "module": str(module_path)},
        )
        from convfinqa.tracking import prompt_ledger

        comp = prompt_ledger.ensure(new_version, source="teacher", run_id=rec.run_id)
        registry.register(
            new_version,
            source="evalloop-teacher",
            run_id=rec.run_id,
            notes=(
                f"targeted challenger: only {target} changed (parent {base_version}); "
                f"hypothesis: fixes {counts[target]} of {len(diagnoses)} diagnosed first-faults"
            ),
            extra={
                "parent": base_version,
                "changed_agents": prompt_ledger.changed_agents(
                    base_version, new_version
                ),
                "composition": prompt_ledger.composition_string(comp),
            },
        )
    return {
        "new_version": new_version,
        "target": target,
        "rules": rules,
        "module": str(module_path),
    }


_AGENT_VARS = {
    "triage": "TRIAGE_PROMPT",
    "preprocess": "PREPROCESS_PROMPT",
    "retriever": "RETRIEVER_PROMPT",
    "calculator": "CALCULATOR_PROMPT",
}


def _write_version_module(
    new_version: str, *, base_version: str, target: str, rules: list[str]
) -> Path:
    """Generated module: three prompts imported unchanged, one extended."""
    var = _AGENT_VARS[target]
    others = ",\n    ".join(v for k, v in _AGENT_VARS.items() if k != target)
    rules_block = "\n".join(f"- {r}" for r in rules)
    body = f'''"""Generated by convfinqa.evalloop.teacher — do not hand-edit.

Targeted challenger for {base_version}: only the {target} prompt changes.
Regenerate via `convfinqa-evalloop propose`.
"""

from convfinqa.prompts.{base_version} import (
    {others},
)
from convfinqa.prompts.{base_version} import {var} as _BASE

__all__ = [
    "TRIAGE_PROMPT",
    "PREPROCESS_PROMPT",
    "RETRIEVER_PROMPT",
    "CALCULATOR_PROMPT",
]

{var} = (
    _BASE
    + """

## Targeted rules ({new_version}, teacher-diagnosed)
{rules_block}
"""
)
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
    """M2's promotion rule: the targeted subagent must improve, overall must not regress.

    "Improve" is judged on the target's *deterministic* per-agent metric
    (`stage_scores.TARGET_METRIC`) — derived from the dataset's own gold — with
    teacher first-fault counts as secondary evidence when diagnoses are given.
    Overall net-positive (the M1 rule) still promotes on its own strength; the
    targeted rule exists so a real subagent fix is not thrown away because the
    small shared set happened to tie overall.
    """
    from convfinqa.evalloop import stage_scores
    from convfinqa.evalloop.gate import gate_runs, load_run_csv

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
        return sum(
            1
            for line in Path(path).read_text().splitlines()
            if line.strip() and json.loads(line)["failed_agent"] == target_agent
        )

    base_faults = _faults(baseline_diagnoses) if baseline_diagnoses else None
    cand_faults = _faults(candidate_diagnoses) if candidate_diagnoses else None

    if metric_before is not None and metric_after is not None:
        target_improved = metric_after > metric_before
        target_evidence = f"{metric_name} {metric_before:.3f} → {metric_after:.3f}"
    elif base_faults is not None and cand_faults is not None:
        target_improved = cand_faults < base_faults
        target_evidence = f"first-faults {base_faults} → {cand_faults} (attribution)"
    else:
        raise SystemExit(
            f"no evidence for {target_agent}: metric {metric_name} unavailable "
            "and no diagnoses supplied"
        )

    overall_ok = stats["accuracy_delta"] >= 0
    verdict = {
        "target_agent": target_agent,
        "target_metric": metric_name,
        "target_metric_before": metric_before,
        "target_metric_after": metric_after,
        "baseline_target_faults": base_faults,
        "candidate_target_faults": cand_faults,
        "target_improved": target_improved,
        "overall_delta": stats["accuracy_delta"],
        "overall_not_regressed": overall_ok,
        "evidence_split": stats["evidence_split"],
        "promotable_targeted": bool(target_improved and overall_ok),
        "promotable_overall": bool(result.promotable),
        "agent_panel_baseline": base_panel,
        "agent_panel_candidate": cand_panel,
        "comparison": stats,
        "reason": (
            f"targeted (M2): {target_agent} {target_evidence} on the shared "
            f"{stats['evidence_split']} reports; overall Δ "
            f"{stats['accuracy_delta'] * 100:+.2f}pp "
            f"({stats['fail_to_pass']} fixed vs {stats['pass_to_fail']} broken); "
            f"McNemar p={stats['mcnemar_p']}"
        ),
    }
    return verdict, result
