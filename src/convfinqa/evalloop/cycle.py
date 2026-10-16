"""One experiment, end to end: train → diagnose → rewrite → gate → decide.

This is the command the campaign is actually driven with. Everything it does
was previously five commands typed in order with the outputs copied between
them, which worked exactly as long as nobody mistyped a version or forgot to
pass the gate CSV — and left no record tying the five runs together.

The sequence, and why it is this sequence:

1. **Train pass** on a freshly drawn split (`pool − gate`), stopping each
   conversation at its first wrong answer. Fresh because a fixed train split
   would be overfitted within a few cycles; early-stopped because everything
   after the first wrong turn is cascade, and cascade is the teacher's noise.
2. **Diagnose** every first-wrong turn. Attribution comes from gold; the teacher
   explains and may dissent.
3. **Pick the target** — most derived faults, subject to the campaign's rotation
   rule.
4. **Rewrite** that one agent's prompt, with its own attempt history in front of
   it, and write the challenger module.
5. **Gate pass** — baseline and challenger both on the *fixed* gate split, every
   question, no early stopping.
6. **Decide** on the campaign rule and record the verdict, promoted or not.

Every step stamps `campaign` / `experiment_label` / `target_agent`, so the whole
experiment is one query away afterwards.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def next_version(base: str) -> str:
    """The next unused `vN` name after `base`, so a cycle never has to be told one."""
    import re

    from convfinqa.config import REPO_ROOT

    prompts_dir = REPO_ROOT / "src" / "convfinqa" / "prompts"
    sdk = re.match(r"sdk_v(\d+)$", base)
    if sdk:
        # The single-session lineage: `sdk_vN` → `sdk_v(N+1)`, skipping any
        # module already on disk. Its own namespace, so it never collides with
        # — and is never handed out next to — a pipeline `vN`.
        used_sdk = {
            int(m.group(1))
            for path in prompts_dir.glob("sdk_v*.py")
            if (m := re.match(r"sdk_v(\d+)$", path.stem))
        }
        n = int(sdk.group(1)) + 1
        while n in used_sdk:
            n += 1
        return f"sdk_v{n}"
    # Compare on the numeric prefix, not the whole stem: `v3_1` exists, so `v3`
    # is taken even though no file is named that. Handing out `v3` next to
    # `v3_1` would make two different bundles read like variants of each other
    # in every run name, chart label and prompt lineage that mentions them.
    used = set()
    for path in prompts_dir.glob("v*.py"):
        found = re.match(r"v(\d+)", path.stem)
        if found:
            used.add(int(found.group(1)))
    match = re.match(r"v(\d+)", base)
    n = int(match.group(1)) if match else 1
    while n in used:
        n += 1
    return f"v{n}"


async def run_cycle(
    *,
    campaign: str,
    baseline_version: str | None = None,
    new_version: str | None = None,
    target: str | None = None,
    train_reports: int = 100,
    train_seed: int | None = None,
    concurrency: int = 8,
    promote: bool = True,
    baseline_gate_csv: str | None = None,
    runtime: str = "pipeline",
    sdk_model: str | None = None,
    sdk_effort: str | None = None,
    train_csvs: list[str] | None = None,
    rescore_seeds: list[int] | None = None,
    diagnose_concurrency: int | None = None,
    smoke_reports: int = 3,
) -> dict[str, Any]:
    """Run one full experiment. Returns everything it did, in order.

    `runtime` picks the arm: ``pipeline`` (below, unchanged) or ``agent_sdk``
    (`run_sdk_cycle`) — the same six steps, with the differences §05 of the
    s10 plan lists: the target is a failure class, the rewrite may touch
    several areas of the one prompt, the gate judges overall accuracy, and
    the promotion moves `sdk_champion`.
    """
    if runtime == "agent_sdk":
        return await run_sdk_cycle(
            campaign=campaign,
            baseline_version=baseline_version,
            new_version=new_version,
            target=target,
            train_reports=train_reports,
            train_seed=train_seed,
            concurrency=concurrency,
            promote=promote,
            baseline_gate_csv=baseline_gate_csv,
            sdk_model=sdk_model,
            sdk_effort=sdk_effort,
            train_csvs=train_csvs,
            rescore_seeds=rescore_seeds,
            diagnose_concurrency=diagnose_concurrency,
            smoke_reports=smoke_reports,
        )
    if sdk_model or sdk_effort or train_csvs or rescore_seeds:
        raise SystemExit(
            "--sdk-model, --sdk-effort, --train-csv and --rescore-seed apply to "
            "--runtime agent_sdk only"
        )
    if runtime != "pipeline":
        raise SystemExit(f"unknown runtime {runtime!r}; expected pipeline or agent_sdk")
    from convfinqa.evalloop import campaign as camp
    from convfinqa.evalloop import ledger, teacher
    from convfinqa.evalloop.runner import run_split
    from convfinqa.tracking import registry

    past = camp.history(campaign, runtime="pipeline")
    camp.check_capacity(campaign, past)
    label = f"{campaign}-e{len(past) + 1:02d}"
    baseline = baseline_version or registry.pipeline_champion()
    if not baseline:
        raise SystemExit("no pipeline champion registered — nothing to challenge")
    if registry.is_sdk_version(baseline):
        raise SystemExit(
            f"{baseline!r} is a single-session prompt — run it with "
            "`cycle --runtime agent_sdk`"
        )
    seed = train_seed if train_seed is not None else 2026 + len(past)
    steps: dict[str, Any] = {"campaign": campaign, "experiment": label}
    print(f"\n=== {label}: challenging {baseline} ===")  # noqa: T201

    # 1 — train pass, fresh draw, early stop
    train = await run_split(
        "train",
        baseline,
        n_reports=train_reports,
        concurrency=concurrency,
        train_seed=seed,
        stop_at_first_wrong=True,
        campaign=campaign,
        label=label,
    )
    steps["train_run"] = train

    # 2 — diagnose
    diagnosis = await teacher.diagnose_run(
        train["csv"], baseline, concurrency=concurrency
    )
    steps["diagnosis"] = diagnosis
    if not diagnosis["n_cases"]:
        raise SystemExit("the train pass produced no failures to learn from")

    # 3 — pick the target under the campaign's rotation rule, on the pooled
    # evidence rather than this one draw. Train is resampled every cycle, so a
    # single draw ranks the agents with about fifty cases split four ways; the
    # accumulated draws of the *same* prompt are the better-powered version of
    # the same question.
    pooled = ledger.merge_draw(
        ledger.fault_history(baseline, exclude_run_id=diagnosis.get("run_id")),
        diagnosis["counts"],
        baseline,
    )
    chosen, why = camp.pick_target(
        diagnosis["counts"], past, requested=target, pooled=pooled
    )
    steps["target"] = {"agent": chosen, "why": why, "pooled_faults": pooled}
    print(f"  target: {chosen} — {why}")  # noqa: T201

    # 4 — rewrite that one agent
    challenger = new_version or next_version(baseline)
    proposal = await teacher.propose_version(
        diagnosis["diagnoses_path"],
        base_version=baseline,
        new_version=challenger,
        target=chosen,
        campaign=campaign,
        label=label,
        pooled=pooled,
    )
    steps["proposal"] = proposal

    # 5 — gate passes: both arms on the fixed split, every question
    if baseline_gate_csv:
        base_csv = baseline_gate_csv
        steps["baseline_gate_run"] = {"csv": base_csv, "reused": True}
    else:
        base_run = await run_split(
            "test",
            baseline,
            concurrency=concurrency,
            campaign=campaign,
            label=label,
        )
        base_csv = base_run["csv"]
        steps["baseline_gate_run"] = base_run
    cand_run = await run_split(
        "test", challenger, concurrency=concurrency, campaign=campaign, label=label
    )
    steps["candidate_gate_run"] = cand_run

    # 6 — decide, and record the decision whichever way it goes
    verdict, comparison = teacher.gate_targeted(
        base_csv,
        cand_run["csv"],
        target_agent=chosen,
        baseline_version=baseline,
        candidate_version=challenger,
    )
    # The gates ledger row is written here, before the promotion below is
    # applied, so it is told what the champion is about to be rather than
    # reading the registry too early. A rejection extends the target's streak.
    will_promote = bool(
        promote and verdict["promotable"] and verdict["evidence_split"] == "test"
    )
    streak = 0
    for past_exp in reversed([e for e in past if e["target_agent"] == chosen]):
        if past_exp["promoted"]:
            break
        streak += 1
    verdict["gate_run_id"] = teacher.log_gate_verdict(
        verdict,
        comparison=comparison,
        campaign=campaign,
        label=label,
        rewrite_id=proposal.get("rewrite_id"),
        consecutive_rejections=0 if verdict["promotable"] else streak + 1,
        champion_after=challenger if will_promote else baseline,
    )
    steps["verdict"] = verdict
    print(f"  {verdict['reason']}")  # noqa: T201

    if promote and verdict["promotable"]:
        if verdict["evidence_split"] != "test":
            raise SystemExit(
                "promotion evidence must come from the gate split — this "
                f"comparison ran on {verdict['evidence_split']!r}"
            )
        outcome = registry.promote(
            challenger,
            comparison=comparison,
            actor="evalloop-cycle",
            force=True,
            reason=f"{verdict['reason']} | {proposal['summary_of_changes']}",
        )
        steps["promotion"] = outcome.as_dict()
        print(f"  PROMOTED {challenger} (was {baseline})")  # noqa: T201
    else:
        steps["promotion"] = {"promoted": False, "champion_retained": baseline}
        print(f"  rejected — {baseline} retained as champion")  # noqa: T201

    steps["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out_dir = train["csv"]
    steps["record"] = str(out_dir)
    print(json.dumps(camp.summarise(campaign), indent=2, default=str))  # noqa: T201
    return steps


def _sdk_teacher() -> Any:
    """`evalloop.sdk_teacher`, imported when a cycle needs it and not before."""
    import importlib

    return importlib.import_module("convfinqa.evalloop.sdk_teacher")


async def run_sdk_cycle(
    *,
    campaign: str,
    baseline_version: str | None = None,
    new_version: str | None = None,
    target: str | None = None,
    train_reports: int = 100,
    train_seed: int | None = None,
    concurrency: int = 8,
    promote: bool = True,
    baseline_gate_csv: str | None = None,
    sdk_model: str | None = None,
    sdk_effort: str | None = None,
    train_csvs: list[str] | None = None,
    rescore_seeds: list[int] | None = None,
    diagnose_concurrency: int | None = None,
    smoke_reports: int = 3,
) -> dict[str, Any]:
    """One experiment on the single-session arm: draw → diagnose → rank → rewrite → gate → decide.

    The order is the pipeline's; what each step reads and writes is the SDK
    arm's. The baseline is the champion's prompt (or the newest `sdk_vN`
    before any is promoted); the diagnosis agent files each first-wrong case
    under a failure class; the classes are ranked on the pooled Wilson bound
    over every draw of this prompt **on this agent** (never a pipeline draw,
    never another model's); the rewrite is one tagged edit per class it
    addresses — or exactly one, once the lineage has been rejected twice in a
    row; promotion is only ever on test evidence.

    Every pass runs on `sdk_model` · `sdk_effort` (default: the champion
    record's). Two cases (s21):

    - **The champion's own agent** — a prompt challenger. The house gate
      (`gate_overall`) judges it against the champion's committed evidence
      (`champion_agent.evidence_run`), which is reused, never re-run; a pass
      promotes the prompt (`promote_sdk`) and keeps the model.
    - **Another model or effort** — a challenger agent. Its train failures are
      that agent's (`train_csvs` reuses committed draws of it, checked for
      version, model, effort and completeness), the writer is told its target,
      a `smoke_reports` pass checks the rewrite still emits the output
      contract, the challenger is scored on test and re-scored on the
      champion's train seeds (`rescore_seeds`), and `gate_challenger` judges it
      under s19's rule against the champion's committed runs. A pass promotes
      prompt, model and effort together (`promote_challenger`).

    The gate row is written after the promotion was tried, so `promoted` is
    what the registry did and `gate_passed` what the rule said.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop import campaign as camp

    # Everything the gate reads is imported here, at the start: a cycle runs
    # for the better part of an hour, and a module first imported at step 6
    # would be whatever is on disk by then, not what the cycle started with.
    from convfinqa.evalloop import gate, sdk_gate, story  # noqa: F401
    from convfinqa.evalloop.runner import PREDICTIONS_DIR, run_split
    from convfinqa.llm import sdk_effort_name, sdk_model_name
    from convfinqa.tracking import registry

    sdk_teacher = _sdk_teacher()
    past = camp.history(campaign, runtime="agent_sdk")
    camp.check_capacity(campaign, past, runtime="agent_sdk")
    label = f"{campaign}-e{len(past) + 1:02d}"
    champion = registry.champion_agent() or {}
    model = sdk_model or sdk_model_name()
    effort = sdk_effort or sdk_effort_name()
    challenger_mode = bool(champion) and (model, effort) != (
        champion.get("model"),
        champion.get("effort"),
    )
    baseline = baseline_version or registry.sdk_champion()
    if not baseline:
        try:
            baseline = prompts_pkg.latest_sdk()
        except RuntimeError:
            raise SystemExit(
                "no sdk_vN prompt exists yet — distil one first with "
                "`convfinqa-evalloop sdk-distil --source-version v8 "
                "--new-version sdk_v1`"
            ) from None
    if not prompts_pkg.is_sdk_version(baseline):
        raise SystemExit(
            f"{baseline!r} is not an sdk_vN prompt — `--runtime agent_sdk` "
            "challenges the single-session lineage only"
        )
    evidence_csv = (
        str(PREDICTIONS_DIR / f"{champion['evidence_run']}.csv")
        if champion.get("evidence_run")
        else None
    )
    if (
        baseline_gate_csv
        and challenger_mode
        and (Path(baseline_gate_csv).stem != champion.get("evidence_run"))
    ):
        raise SystemExit(
            "a challenger agent is paired against the champion's committed "
            f"evidence ({champion.get('evidence_run')}); nothing else may stand "
            "in for it"
        )
    seed = train_seed if train_seed is not None else 2026 + len(past)
    single_area = camp.single_area_mode(past)
    steps: dict[str, Any] = {
        "campaign": campaign,
        "experiment": label,
        "runtime": "agent_sdk",
        "sdk_model": model,
        "sdk_effort": effort,
        "challenger_mode": challenger_mode,
        "single_area_mode": single_area,
    }
    print(  # noqa: T201
        f"\n=== {label} [agent_sdk · {model} · {effort}]: challenging {baseline}"
        + (" as a challenger agent ===" if challenger_mode else " ===")
    )

    # 1 — this agent's train failures: committed draws, or a fresh early-stopped one
    if train_csvs:
        steps["train_runs"] = [
            _checked_train_csv(csv, version=baseline, model=model, effort=effort)
            for csv in train_csvs
        ]
    else:
        train = await run_split(
            "train",
            baseline,
            n_reports=train_reports,
            concurrency=concurrency,
            train_seed=seed,
            stop_at_first_wrong=True,
            campaign=campaign,
            label=label,
            runtime="agent_sdk",
            sdk_model=model,
            sdk_effort=effort,
        )
        _require_complete(train, "the train draw")
        steps["train_runs"] = [train]
    steps["train_run"] = steps["train_runs"][0]

    # 2 — diagnose: one ledger row per first-wrong case, filed under a class
    diagnoses = [
        await sdk_teacher.diagnose_run(
            run["csv"],
            baseline,
            concurrency=diagnose_concurrency or concurrency,
            campaign=campaign,
            label=label,
            sdk_model=model,
            sdk_effort=effort,
        )
        for run in steps["train_runs"]
    ]
    steps["diagnosis"] = diagnoses
    if not sum(int(d["n_cases"]) for d in diagnoses):
        raise SystemExit("the train pass produced no failures to learn from")

    # 3 — rank the failure classes on the pooled evidence for this prompt on
    # this agent; the ranking is the writer's evidence as well as the target's
    ranking = sdk_teacher.rank_classes(baseline, sdk_model=model, sdk_effort=effort)
    chosen, why = camp.pick_target_class(ranking, requested=target)
    steps["target"] = {"failure_class": chosen, "why": why, "ranking": ranking}
    print(f"  target class: {chosen} — {why}")  # noqa: T201

    # 4 — rewrite the one prompt, for this agent: several tagged edits, or one
    # after two consecutive rejections
    challenger = new_version or next_version(baseline)
    paths = [d["diagnoses_path"] for d in diagnoses]
    proposal = await sdk_teacher.propose_version(
        paths[0] if len(paths) == 1 else paths,
        base_version=baseline,
        new_version=challenger,
        campaign=campaign,
        label=label,
        pooled=ranking,
        max_areas=1 if single_area else None,
        target_model=model,
        target_effort=effort,
    )
    steps["proposal"] = proposal

    # 5 — score the challenger; the champion is never run
    if challenger_mode:
        if smoke_reports:
            smoke = await run_split(
                "train",
                challenger,
                n_reports=smoke_reports,
                concurrency=concurrency,
                campaign=campaign,
                label=label,
                runtime="agent_sdk",
                sdk_model=model,
                sdk_effort=effort,
            )
            steps["smoke_run"] = smoke
            _require_contract(smoke)
        base_csv = str(evidence_csv)
        steps["baseline_gate_run"] = {"csv": base_csv, "reused": True}
    elif baseline_gate_csv:
        base_csv = baseline_gate_csv
        steps["baseline_gate_run"] = {"csv": base_csv, "reused": True}
    elif evidence_csv and baseline == champion.get("version"):
        base_csv = evidence_csv
        steps["baseline_gate_run"] = {"csv": base_csv, "reused": True}
    else:
        base_run = await run_split(
            "test",
            baseline,
            concurrency=concurrency,
            campaign=campaign,
            label=label,
            runtime="agent_sdk",
            sdk_model=model,
            sdk_effort=effort,
        )
        _require_complete(base_run, "the baseline gate pass")
        base_csv = base_run["csv"]
        steps["baseline_gate_run"] = base_run
    cand_run = await run_split(
        "test",
        challenger,
        concurrency=concurrency,
        campaign=campaign,
        label=label,
        runtime="agent_sdk",
        sdk_model=model,
        sdk_effort=effort,
    )
    steps["candidate_gate_run"] = cand_run
    _require_complete(cand_run, "the challenger's test pass")
    rescored: dict[str, str] = {}
    if challenger_mode:
        records = sdk_gate._records()
        seeds = (
            rescore_seeds
            if rescore_seeds is not None
            else _champion_seeds(records, champion)
        )
        steps["train_rescore_runs"] = []
        for rescore_seed in seeds:
            run = await run_split(
                "train",
                challenger,
                n_reports=train_reports,
                concurrency=concurrency,
                train_seed=int(rescore_seed),
                stop_at_first_wrong=True,
                campaign=campaign,
                label=label,
                runtime="agent_sdk",
                sdk_model=model,
                sdk_effort=effort,
            )
            steps["train_rescore_runs"].append(run)
            _require_complete(run, f"the train re-score on seed {rescore_seed}")
            rescored[str(rescore_seed)] = run["csv"]

    # 6 — decide, try the promotion, then record the verdict either way
    if challenger_mode:
        verdict, comparison = sdk_gate.gate_challenger(
            cand_run["csv"],
            candidate_version=challenger,
            sdk_model=model,
            sdk_effort=effort,
            candidate_train=rescored,
            target_class=chosen,
            base_agent_csv=_latest_test_csv(records, baseline, model, effort),
            eval_records=records,
        )
    else:
        verdict, comparison = sdk_gate.gate_overall(
            base_csv,
            cand_run["csv"],
            baseline_version=baseline,
            candidate_version=challenger,
            target_class=chosen,
        )
        verdict["baseline_agent"] = verdict["candidate_agent"] = (model, effort)
    steps["verdict"] = verdict
    print(f"  {verdict['reason']}")  # noqa: T201
    if promote and verdict["promotable"] and verdict["evidence_split"] != "test":
        raise SystemExit(
            "promotion evidence must come from the gate split — this "
            f"comparison ran on {verdict['evidence_split']!r}"
        )
    if promote and verdict["promotable"]:
        if challenger_mode:
            outcome = registry.promote_challenger(
                challenger,
                model=model,
                effort=effort,
                verdict=verdict["challenger_verdict"],
                reason=f"{verdict['reason']} | {_edit_summary(proposal)}",
            )
        else:
            outcome = registry.promote_sdk(
                challenger,
                comparison=comparison,
                evidence_split=str(verdict["evidence_split"]),
                evidence_csv=cand_run["csv"],
                reason=f"{verdict['reason']} | {_edit_summary(proposal)}",
            )
        steps["promotion"] = outcome.as_dict()
        verb = "PROMOTED" if outcome.promoted else "NOT PROMOTED"
        print(f"  {verb} {challenger}: {outcome.reason}")  # noqa: T201
    else:
        steps["promotion"] = {"promoted": False, "champion_retained": baseline}
        print(f"  rejected — {registry.sdk_champion()} retained as champion")  # noqa: T201
    promoted = bool(steps["promotion"].get("promoted"))
    streak = camp.consecutive_rejections(past)
    verdict["gate_run_id"] = sdk_gate.log_gate_verdict(
        verdict,
        comparison=comparison,
        campaign=campaign,
        label=label,
        rewrite_id=proposal.get("rewrite_id"),
        consecutive_rejections=0 if verdict["promotable"] else streak + 1,
        champion_after=challenger
        if promoted
        else (registry.sdk_champion() or baseline),
        promoted=promoted,
        extra_artifacts=(
            {
                "challenger_pair.json": verdict.get("challenger_pair"),
                **(
                    {"vs_base_agent.json": verdict["vs_base_agent"]}
                    if verdict.get("vs_base_agent")
                    else {}
                ),
            }
            if challenger_mode
            else None
        ),
    )

    steps["finished_at"] = datetime.now().isoformat(timespec="seconds")
    steps["record"] = str(steps["train_run"]["csv"])
    print(json.dumps(camp.summarise(campaign), indent=2, default=str))  # noqa: T201
    return steps


def _require_complete(run: dict[str, Any], what: str) -> None:
    """Stop the cycle on a pass with unscored turns: it cannot be gated."""
    if run.get("complete", True):
        return
    raise SystemExit(
        f"{what} is incomplete ({run.get('n_unscored')} turns unscored, "
        f"{run.get('n_rate_limited')} rate-limited): finish it with `run "
        f"--resume-from {run.get('csv')}` (same flags), then judge the "
        "challenger with `gate-challenger`. The cycle does not switch billing."
    )


def _require_contract(run: dict[str, Any]) -> None:
    """Stop before scoring when the rewrite no longer gets a structured answer."""
    import pandas as pd

    _require_complete(run, "the smoke pass")
    frame = pd.read_csv(run["csv"])
    errors = frame["error"].dropna().astype(str) if "error" in frame else []
    broken = [e for e in errors if e and not e.startswith("token budget exhausted")]
    if broken:
        raise SystemExit(
            f"the smoke pass broke the output contract on {len(broken)} turn(s) "
            f"— first: {broken[0][:300]}"
        )


def _checked_train_csv(
    csv: str, *, version: str, model: str, effort: str
) -> dict[str, Any]:
    """A committed train draw, refused unless it is this agent's complete early-stopped draw."""
    from convfinqa.evalloop import sdk_gate
    from convfinqa.tracking import registry
    from convfinqa.tracking.comparator import load_evalloop_csv

    path = Path(csv)
    if not path.exists():
        raise SystemExit(f"{csv} does not exist")
    problems = registry._ran_as(
        path.stem, {"version": version, "model": model, "effort": effort}
    )
    if problems:
        raise SystemExit(
            f"{path.name} is not a draw of this agent: {'; '.join(problems)}"
        )
    try:
        frame = load_evalloop_csv(path, split="train")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"{path.name}: {exc}") from exc
    return {
        "csv": str(path),
        "run_name": path.stem,
        "reused": True,
        "train_seed": sdk_gate.train_seed_of(path),
        "n_questions": int(len(frame)),
    }


def _champion_seeds(
    records: list[dict[str, Any]], champion: dict[str, Any]
) -> list[int]:
    """The seeds the champion has early-stopped train draws on: what a challenger pairs with."""
    from convfinqa.evalloop import story

    draws = story._train_runs_by_agent(
        [
            r
            for r in records
            if (r.get("params") or {}).get("runtime") == "agent_sdk"
            and (r.get("params") or {}).get("prompts_version")
            == champion.get("version")
        ],
        str(champion.get("model")),
    ).get((str(champion.get("model")), str(champion.get("effort"))), {})
    return sorted(int(s) for s in draws)


def _latest_test_csv(
    records: list[dict[str, Any]], version: str, model: str, effort: str
) -> str | None:
    """The latest complete test pass of `version` on this agent, if one is committed."""
    from convfinqa.evalloop import story
    from convfinqa.evalloop.runner import PREDICTIONS_DIR

    hits = [
        r
        for r in records
        if (r.get("params") or {}).get("runtime") == "agent_sdk"
        and (r.get("params") or {}).get("split") == "test"
        and (r.get("params") or {}).get("prompts_version") == version
        and str(r.get("run_name") or "").startswith("sdk-evalloop-test100-")
        and story._agent_key(r, model) == (model, effort)
        and (r.get("metrics") or {}).get("complete") != 0.0
    ]
    latest = story._latest(hits)
    if latest is None:
        return None
    path = PREDICTIONS_DIR / f"{latest['run_name']}.csv"
    return str(path) if path.exists() else None


def _edit_summary(proposal: dict[str, Any]) -> str:
    """One line naming the classes a multi-area rewrite addressed."""
    edits = proposal.get("edits") or []
    classes = [str(e.get("failure_class") or e.get("target") or "?") for e in edits]
    if classes:
        return f"{len(edits)} edit(s): {', '.join(classes)}"
    return str(proposal.get("summary_of_changes") or "rewrite")
