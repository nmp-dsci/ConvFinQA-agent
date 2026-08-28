"""Per-case loop driver: Diagnose → (Route+Fix → Verify) × retry_n."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import settings
from convfinqa.diagnosis.agents import propose_fix, route_case
from convfinqa.diagnosis.models import (
    AgentName,
    CaseResult,
    FailureReason,
    FixAttempt,
    FixPayload,
    FixProposal,
    RouterDiagnosis,
    RouterPayload,
    StageIO,
)
from convfinqa.diagnosis.rules_store import (
    append_attempt,
    append_rule,
    read_attempts,
)
from convfinqa.diagnosis.verify import verify_patch
from convfinqa.prompts import load as load_prompts

log = logging.getLogger("convfinqa.diagnosis")

_SPEC_AGENTS: set[AgentName] = {"triage", "preprocess", "retriever", "calculator"}


def load_case_cache(
    case_log_path: Path,
) -> dict[tuple[str, int], CaseResult]:
    """Read existing case_results JSONL into a (report_id, turn_index) → CaseResult map.

    The full `CaseResult` is preserved so all three step caches (diagnose,
    propose, verify) can hit independently inside `run_case`. Missing file,
    empty file, or malformed lines are silently skipped — the cache is
    advisory, not authoritative.
    """
    cache: dict[tuple[str, int], CaseResult] = {}
    if not case_log_path.exists():
        return cache
    try:
        text = case_log_path.read_text()
    except OSError:
        return cache
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            case = CaseResult.model_validate_json(raw)
        except Exception:
            continue
        cache[(case.report_id, int(case.turn_index))] = case
    return cache


def load_diagnose_cache(
    case_log_path: Path,
) -> dict[tuple[str, int], RouterDiagnosis]:
    """Back-compat shim — returns only the RouterDiagnosis subset of load_case_cache."""
    return {
        k: c.router_diagnosis
        for k, c in load_case_cache(case_log_path).items()
        if c.router_diagnosis is not None
    }


def _assemble_current_prompts(base_version: str = "v2") -> dict[str, str]:
    """Return live prompts: v2 base + already-passing rules from each store."""
    from convfinqa.diagnosis.assembler import assemble_prompts
    from convfinqa.diagnosis.rules_store import all_rules

    base = load_prompts(base_version)
    rules_by_agent = all_rules()
    merged = assemble_prompts(base, rules_by_agent)
    return {str(agent): text for agent, text in merged.items()}


def _stage_io_from_router_payload(
    payload: RouterPayload, agent: AgentName
) -> StageIO | None:
    return getattr(payload, f"{agent}_io", None)


def _build_fix_payload(
    payload: RouterPayload,
    *,
    router_diagnosis: RouterDiagnosis,
    current_prompts: dict[str, str],
    prior_attempts: list[FixAttempt],
) -> FixPayload:
    agent: AgentName = router_diagnosis.failed_agent  # type: ignore[assignment]
    failed_io = _stage_io_from_router_payload(payload, agent)
    upstream: dict[str, StageIO | None] = {}
    for upstream_agent in ("triage", "preprocess", "retriever", "calculator"):
        if upstream_agent == agent:
            break
        upstream[upstream_agent] = _stage_io_from_router_payload(
            payload, upstream_agent
        )  # type: ignore[arg-type]
    return FixPayload(
        report_id=payload.report_id,
        turn_index=payload.turn_index,
        question=payload.question,
        history_text=payload.history_text,
        gold_answer=payload.gold_answer,
        pred_answer=payload.pred_answer,
        gold_program=payload.gold_program,
        router_diagnosis=router_diagnosis,
        failed_agent_io=failed_io,
        upstream_ios=upstream,
        current_prompt=current_prompts[agent],
        prior_rule_attempts=read_attempts(agent),
        prior_attempts=list(prior_attempts),
    )


async def run_case(
    payload: RouterPayload,
    *,
    diagnose_only: bool = False,
    propose_only: bool = False,
    base_version: str = "v2",
    cached_case: CaseResult | None = None,
    use_diagnose_cache: bool = True,
    use_propose_cache: bool = True,
    use_verify_cache: bool = True,
) -> CaseResult:
    """Run the per-case flow.

    - Default: Step 1 — Diagnose → Step 2 — Propose → Step 3 — Verify (attempt loop).
    - `diagnose_only=True`: Step 1 only.
    - `propose_only=True`: Step 1 + Step 2 (skip Step 3). No rule writes.

    Step caches (see spec §Step Caches): if `cached_case` is provided, each step
    independently consults its cache before falling through to the live call.
    `use_diagnose_cache` / `use_propose_cache` / `use_verify_cache` toggle each
    cache (`True` = honour cache, `False` = re-run live regardless of contents).
    """
    if diagnose_only and propose_only:
        raise ValueError("diagnose_only and propose_only are mutually exclusive")
    cached_diagnosis: RouterDiagnosis | None = (
        cached_case.router_diagnosis
        if (cached_case is not None and use_diagnose_cache)
        else None
    )
    log.info(
        "[case] report=%s turn=%s gold=%s pred=%s",
        payload.report_id,
        payload.turn_index,
        payload.gold_answer,
        payload.pred_answer,
    )

    # Step 1 — Diagnose (once per case; skipped on cache hit)
    current_prompts = _assemble_current_prompts(base_version)
    # Refresh the four current prompts in the router payload to reflect any
    # rules already promoted by earlier cases in this run.
    payload = payload.model_copy(
        update={
            "current_triage_prompt": current_prompts["triage"],
            "current_preprocess_prompt": current_prompts["preprocess"],
            "current_retriever_prompt": current_prompts["retriever"],
            "current_calculator_prompt": current_prompts["calculator"],
        }
    )
    if cached_diagnosis is not None:
        diagnosis = cached_diagnosis
        log.info(
            "[%s] diagnosis: cached (mode=%s conf=%.2f)",
            diagnosis.failed_agent,
            diagnosis.failure_mode,
            diagnosis.confidence,
        )
    else:
        diagnosis = await route_case(payload)
        log.info(
            "[%s] diagnosis: mode=%s conf=%.2f",
            diagnosis.failed_agent,
            diagnosis.failure_mode,
            diagnosis.confidence,
        )

    result = CaseResult(
        report_id=payload.report_id,
        turn_index=payload.turn_index,
        question=payload.question,
        gold_answer=payload.gold_answer,
        original_pred_answer=payload.pred_answer,
        gold_turn_type=payload.gold_turn_type,
        gold_program=payload.gold_program,
        router_diagnosis=diagnosis,
    )

    if diagnose_only:
        result.attempts.append(
            FixAttempt(
                iteration=1,
                failed_agent=diagnosis.failed_agent,
                patch_applied="",
                full_prompt="",
                correct=False,
            )
        )
        return result

    if (
        diagnosis.failed_agent == "ambiguous"
        or diagnosis.failed_agent not in _SPEC_AGENTS
    ):
        log.info("[ambiguous] no specialist routing; case unresolved")
        result.attempts.append(
            FixAttempt(
                iteration=1,
                failed_agent="ambiguous",
                patch_applied="",
                full_prompt="",
                correct=False,
                failure_reason="ambiguous_followup",
            )
        )
        return result

    agent: AgentName = diagnosis.failed_agent  # type: ignore[assignment]
    prior_attempts: list[FixAttempt] = []
    max_attempts = max(1, min(int(settings.retry_n), 3))

    for attempt_idx in range(1, max_attempts + 1):
        # Step 2 + Step 3 cache lookup — cached attempt for this attempt index.
        cached_attempt: FixAttempt | None = None
        if cached_case is not None and len(cached_case.attempts) >= attempt_idx:
            cached_attempt = cached_case.attempts[attempt_idx - 1]

        # Step 2 — Propose (cache hit if cached_attempt.patch_applied is set).
        propose_cache_hit = (
            use_propose_cache
            and cached_attempt is not None
            and bool(cached_attempt.patch_applied.strip())
        )
        if propose_cache_hit and cached_attempt is not None:
            log.info(
                "[%s] attempt %d/%d — propose: cached (rule=%r)",
                agent,
                attempt_idx,
                max_attempts,
                cached_attempt.patch_applied[:80],
            )
            fix = FixProposal(
                rule=cached_attempt.patch_applied,
                fix_type=cached_attempt.fix_type or "add_rule",
                confidence=cached_attempt.fix_confidence
                if cached_attempt.fix_confidence is not None
                else 0.5,
                rationale="(restored from step-cache)",
            )
        else:
            log.info(
                "[%s] attempt %d/%d — propose_fix", agent, attempt_idx, max_attempts
            )
            fix_payload = _build_fix_payload(
                payload,
                router_diagnosis=diagnosis,
                current_prompts=current_prompts,
                prior_attempts=prior_attempts,
            )
            fix = await propose_fix(agent, fix_payload)
        if not fix.rule.strip():
            log.info("[%s] empty rule — unresolved", agent)
            placeholder = FixAttempt(
                iteration=attempt_idx,
                failed_agent=agent,
                patch_applied="",
                full_prompt=current_prompts[agent],
                correct=False,
                failure_reason="ambiguous_followup",
            )
            result.attempts.append(placeholder)
            break

        if fix.rule.strip() in {a.patch_applied.strip() for a in prior_attempts}:
            log.info("[%s] duplicate patch — terminate", agent)
            append_attempt(
                agent,
                fix.rule,
                fix.fix_type,
                fix.confidence,
                payload.report_id,
                payload.turn_index,
                "failed",
                first_failing_turn=None,
                failure_reason="duplicate_patch",
            )
            placeholder = FixAttempt(
                iteration=attempt_idx,
                failed_agent=agent,
                patch_applied=fix.rule,
                full_prompt=current_prompts[agent],
                correct=False,
                failure_reason="duplicate_patch",
            )
            result.attempts.append(placeholder)
            break

        if propose_only:
            log.info(
                "[%s] propose-only — skipping Step 3 (verify). rule=%r conf=%.2f",
                agent,
                fix.rule[:80],
                fix.confidence,
            )
            placeholder = FixAttempt(
                iteration=attempt_idx,
                failed_agent=agent,
                patch_applied=fix.rule,
                full_prompt=current_prompts[agent],
                correct=False,
                # verify_result left as None → CSV/HTML render Group C cells as empty.
            )
            result.attempts.append(placeholder)
            break

        # Step 3 — Verify (cache hit if cached_attempt has verify_result + matching rule).
        verify_cache_hit = (
            use_verify_cache
            and cached_attempt is not None
            and cached_attempt.verify_result is not None
            and cached_attempt.patch_applied.strip() == fix.rule.strip()
        )
        if verify_cache_hit and cached_attempt is not None:
            log.info(
                "[%s] verify: cached (result=%s first_failing_turn=%s)",
                agent,
                cached_attempt.verify_result,
                cached_attempt.first_failing_turn,
            )
            # Round-trip the cached FixAttempt with the current iteration index
            # (case-cache may have a different iteration field if retry_n changed).
            attempt = cached_attempt.model_copy(
                update={
                    "iteration": attempt_idx,
                    "fix_type": fix.fix_type,
                    "fix_confidence": fix.confidence,
                }
            )
        else:
            log.info(
                "[%s] verify (replay 0..%d) — rule=%r conf=%.2f",
                agent,
                payload.turn_index,
                fix.rule[:80],
                fix.confidence,
            )
            attempt = await verify_patch(
                agent,
                fix.rule,
                iteration=attempt_idx,
                report_id=payload.report_id,
                failed_turn_index=payload.turn_index,
                current_prompts=current_prompts,
            )
        prior_attempts.append(attempt)
        result.attempts.append(attempt)

        failure_reason: FailureReason | None = attempt.failure_reason
        promoted_rule_id: str | None = None
        if attempt.correct:
            promoted_rule_id = append_rule(
                agent,
                fix.rule,
                fix.fix_type,
                fix.confidence,
                payload.report_id,
                payload.turn_index,
            )
            log.info(
                "[%s] PASSED — rule %s promoted to %s",
                agent,
                promoted_rule_id,
                f"rules_{agent}_{settings.variant}.jsonl",
            )
        else:
            log.info(
                "[%s] FAILED — first_failing_turn=%s reason=%s",
                agent,
                attempt.first_failing_turn,
                attempt.failure_reason,
            )

        append_attempt(
            agent,
            fix.rule,
            fix.fix_type,
            fix.confidence,
            payload.report_id,
            payload.turn_index,
            "passed" if attempt.correct else "failed",
            first_failing_turn=attempt.first_failing_turn,
            failure_reason=failure_reason,
            promoted_rule_id=promoted_rule_id,
        )

        if attempt.correct:
            result.resolved = True
            result.winning_iteration = attempt_idx
            result.final_patch = fix.rule
            # Refresh current_prompts so subsequent cases see the new rule.
            current_prompts = _assemble_current_prompts(base_version)
            break

    return result


async def run_harness(
    payloads: list[RouterPayload],
    *,
    diagnose_only: bool = False,
    propose_only: bool = False,
    base_version: str = "v2",
    case_log_path: Path | None = None,
    disable_cache: bool = False,
    use_propose_cache: bool = True,
    use_verify_cache: bool = True,
) -> list[CaseResult]:
    """Sequential per-case loop. One case at a time.

    Loads the step caches from `case_log_path` (if it exists) before truncating
    the file for the fresh run. The three step caches (diagnose, propose, verify)
    can each be disabled independently:
    - `disable_cache=True` disables Step 1 (diagnose) cache.
    - `use_propose_cache=False` disables Step 2 (propose) cache.
    - `use_verify_cache=False` disables Step 3 (verify) cache.
    See spec §Step Caches.
    """
    if diagnose_only and propose_only:
        raise ValueError("diagnose_only and propose_only are mutually exclusive")
    results: list[CaseResult] = []
    case_cache: dict[tuple[str, int], CaseResult] = {}
    if case_log_path is not None:
        case_log_path.parent.mkdir(parents=True, exist_ok=True)
        case_cache = load_case_cache(case_log_path)
        if case_cache:
            log.info("[step-cache] loaded %d cached cases", len(case_cache))
        # Truncate to start fresh (cache is held in memory for the rest of the run).
        case_log_path.write_text("")
    use_diagnose_cache = not disable_cache
    hits = {"diagnose": 0, "propose": 0, "verify": 0}
    skipped: list[tuple[str, int, str]] = []
    for i, payload in enumerate(payloads, start=1):
        log.info("--- case %d/%d ---", i, len(payloads))
        cached = case_cache.get((payload.report_id, int(payload.turn_index)))
        if cached is not None:
            if use_diagnose_cache and cached.router_diagnosis is not None:
                hits["diagnose"] += 1
            for a in cached.attempts:
                if use_propose_cache and a.patch_applied.strip():
                    hits["propose"] += 1
                if use_verify_cache and a.verify_result is not None:
                    hits["verify"] += 1
        try:
            result = await run_case(
                payload,
                diagnose_only=diagnose_only,
                propose_only=propose_only,
                base_version=base_version,
                cached_case=cached,
                use_diagnose_cache=use_diagnose_cache,
                use_propose_cache=use_propose_cache,
                use_verify_cache=use_verify_cache,
            )
        except Exception as exc:
            # Transient API failures (504s, rate limits) on one case must not
            # take down the whole run and wipe the step cache. Log loudly,
            # preserve any prior cached result so it survives the next run,
            # and continue with the next case.
            log.error(
                "[case %d/%d] FAILED — report=%s turn=%s: %s: %s",
                i,
                len(payloads),
                payload.report_id,
                payload.turn_index,
                type(exc).__name__,
                exc,
            )
            skipped.append(
                (payload.report_id, int(payload.turn_index), type(exc).__name__)
            )
            if cached is not None and case_log_path is not None:
                with case_log_path.open("a") as f:
                    f.write(cached.model_dump_json() + "\n")
            continue
        results.append(result)
        if case_log_path is not None:
            with case_log_path.open("a") as f:
                f.write(result.model_dump_json() + "\n")
    if case_cache:
        log.info(
            "[step-cache] hits diagnose=%d propose=%d verify=%d (cache size=%d cases)",
            hits["diagnose"],
            hits["propose"],
            hits["verify"],
            len(case_cache),
        )
    if skipped:
        log.warning(
            "[run-harness] %d case(s) skipped due to errors — rerun to retry: %s",
            len(skipped),
            ", ".join(f"{rid}#{ti} ({err})" for rid, ti, err in skipped),
        )
    return results


def join_full_df_columns(
    results: list[CaseResult], full_df: pd.DataFrame
) -> pd.DataFrame:
    """Join Group A columns (from the input CSV) onto the case results."""
    keys = [(r.report_id, r.turn_index) for r in results]
    keys_set = set(keys)
    sub = full_df[
        full_df.apply(
            lambda r: (r["report_id"], int(r["turn_index"])) in keys_set, axis=1
        )
    ].copy()
    return sub


def case_results_to_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    """Flatten CaseResults to one dict per (case, attempt) for the CSV/HTML."""
    rows: list[dict[str, Any]] = []
    for r in results:
        attempts = r.attempts or [
            FixAttempt(
                iteration=1,
                failed_agent=r.router_diagnosis.failed_agent
                if r.router_diagnosis
                else "ambiguous",
                patch_applied="",
                full_prompt="",
                correct=False,
            )
        ]
        for a in attempts:
            diag = r.router_diagnosis
            # When verify did not run (diagnose-only / propose-fix / ambiguous /
            # empty-rule placeholder), emit all Group C harness columns as empty
            # strings so the HTML renders them as `—`. See spec §Run Modes.
            verified = a.verify_result is not None
            rows.append(
                {
                    "report_id": r.report_id,
                    "turn_index": r.turn_index,
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "original_pred_answer": r.original_pred_answer,
                    "gold_turn_type": r.gold_turn_type,
                    "gold_program": r.gold_program,
                    "attempt_id": a.iteration,
                    "failed_agent": diag.failed_agent if diag else "",
                    "failure_mode": diag.failure_mode if diag else "",
                    "failure_explanation": diag.failure_explanation if diag else "",
                    "supporting_evidence": json.dumps(diag.supporting_evidence)
                    if diag
                    else "",
                    "confidence": diag.confidence if diag else "",
                    "system_prompt_fix": a.patch_applied,
                    "fix_type": "",
                    "harness_correct": a.correct if verified else "",
                    "harness_first_failing_turn": (
                        ""
                        if (not verified or a.first_failing_turn is None)
                        else a.first_failing_turn
                    ),
                    "harness_turn_results": (
                        json.dumps([tr.model_dump() for tr in a.turn_results])
                        if verified
                        else ""
                    ),
                    "harness_pred_answer": a.pred_answer if verified else "",
                    "harness_triage_io": (
                        a.triage_io.model_dump_json()
                        if (verified and a.triage_io)
                        else ""
                    ),
                    "harness_preprocess_io": (
                        a.preprocess_io.model_dump_json()
                        if (verified and a.preprocess_io)
                        else ""
                    ),
                    "harness_retriever_io": (
                        a.retriever_io.model_dump_json()
                        if (verified and a.retriever_io)
                        else ""
                    ),
                    "harness_calculator_io": (
                        a.calculator_io.model_dump_json()
                        if (verified and a.calculator_io)
                        else ""
                    ),
                    "verify_result": a.verify_result or "",
                    "failure_reason": a.failure_reason or "",
                    "resolved": r.resolved,
                }
            )
    return rows
