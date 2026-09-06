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
    baseline = baseline_version or registry.champion()
    if not baseline:
        raise SystemExit("no champion registered — nothing to challenge")
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
) -> dict[str, Any]:
    """One experiment on the single-session arm: draw → diagnose → rank → rewrite → gate → decide.

    The order is the pipeline's; what each step reads and writes is the SDK
    arm's. The baseline is `sdk_champion` (or the newest `sdk_vN` before any
    is promoted); the diagnosis agent files each first-wrong case under a
    failure class; the classes are ranked on the pooled Wilson bound over every
    draw of this prompt (never a pipeline draw); the rewrite is one tagged edit
    per class it addresses — or exactly one, once the lineage has been rejected
    twice in a row; the gate is overall accuracy on the fixed split; promotion
    moves `sdk_champion` and only ever on test evidence.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop import campaign as camp
    from convfinqa.evalloop import sdk_gate
    from convfinqa.evalloop.runner import run_split
    from convfinqa.tracking import registry

    sdk_teacher = _sdk_teacher()
    past = camp.history(campaign, runtime="agent_sdk")
    camp.check_capacity(campaign, past, runtime="agent_sdk")
    label = f"{campaign}-e{len(past) + 1:02d}"
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
    seed = train_seed if train_seed is not None else 2026 + len(past)
    single_area = camp.single_area_mode(past)
    steps: dict[str, Any] = {
        "campaign": campaign,
        "experiment": label,
        "runtime": "agent_sdk",
        "single_area_mode": single_area,
    }
    print(f"\n=== {label} [agent_sdk]: challenging {baseline} ===")  # noqa: T201

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
        runtime="agent_sdk",
    )
    steps["train_run"] = train

    # 2 — diagnose: one ledger row per first-wrong case, filed under a class
    diagnosis = await sdk_teacher.diagnose_run(
        train["csv"],
        baseline,
        concurrency=concurrency,
        campaign=campaign,
        label=label,
    )
    steps["diagnosis"] = diagnosis
    if not diagnosis["n_cases"]:
        raise SystemExit("the train pass produced no failures to learn from")

    # 3 — rank the failure classes on the pooled evidence for this prompt.
    # SDK draws pool with SDK draws of the same prompt hash and with nothing
    # else; the ranking is the writer's evidence as well as the target's.
    ranking = sdk_teacher.rank_classes(baseline)
    chosen, why = camp.pick_target_class(ranking, requested=target)
    steps["target"] = {"failure_class": chosen, "why": why, "ranking": ranking}
    print(f"  target class: {chosen} — {why}")  # noqa: T201

    # 4 — rewrite the one prompt: several tagged edits, or one after two
    # consecutive rejections
    challenger = new_version or next_version(baseline)
    proposal = await sdk_teacher.propose_version(
        diagnosis["diagnoses_path"],
        base_version=baseline,
        new_version=challenger,
        campaign=campaign,
        label=label,
        pooled=ranking,
        max_areas=1 if single_area else None,
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
            runtime="agent_sdk",
        )
        base_csv = base_run["csv"]
        steps["baseline_gate_run"] = base_run
    cand_run = await run_split(
        "test",
        challenger,
        concurrency=concurrency,
        campaign=campaign,
        label=label,
        runtime="agent_sdk",
    )
    steps["candidate_gate_run"] = cand_run

    # 6 — decide on overall accuracy, record the verdict either way
    verdict, comparison = sdk_gate.gate_overall(
        base_csv,
        cand_run["csv"],
        baseline_version=baseline,
        candidate_version=challenger,
        target_class=chosen,
    )
    will_promote = bool(
        promote and verdict["promotable"] and verdict["evidence_split"] == "test"
    )
    streak = camp.consecutive_rejections(past)
    verdict["gate_run_id"] = sdk_gate.log_gate_verdict(
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
        outcome = registry.promote_sdk(
            challenger,
            comparison=comparison,
            evidence_split=str(verdict["evidence_split"]),
            reason=f"{verdict['reason']} | {_edit_summary(proposal)}",
        )
        steps["promotion"] = outcome.as_dict()
        print(f"  PROMOTED {challenger} to sdk_champion (was {baseline})")  # noqa: T201
    else:
        steps["promotion"] = {"promoted": False, "champion_retained": baseline}
        print(f"  rejected — {baseline} retained as sdk_champion")  # noqa: T201

    steps["finished_at"] = datetime.now().isoformat(timespec="seconds")
    steps["record"] = str(train["csv"])
    print(json.dumps(camp.summarise(campaign), indent=2, default=str))  # noqa: T201
    return steps


def _edit_summary(proposal: dict[str, Any]) -> str:
    """One line naming the classes a multi-area rewrite addressed."""
    edits = proposal.get("edits") or []
    classes = [str(e.get("failure_class") or e.get("target") or "?") for e in edits]
    if classes:
        return f"{len(edits)} edit(s): {', '.join(classes)}"
    return str(proposal.get("summary_of_changes") or "rewrite")
