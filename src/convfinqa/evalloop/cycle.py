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
) -> dict[str, Any]:
    """Run one full experiment. Returns everything it did, in order."""
    from convfinqa.evalloop import campaign as camp
    from convfinqa.evalloop import teacher
    from convfinqa.evalloop.runner import run_split
    from convfinqa.tracking import registry

    past = camp.history(campaign)
    camp.check_capacity(campaign, past)
    label = f"{campaign}-e{len(past) + 1:02d}"
    baseline = baseline_version or registry.champion()
    if not baseline:
        raise SystemExit("no champion registered — nothing to challenge")
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
    diagnosis = await teacher.diagnose_run(train["csv"], baseline)
    steps["diagnosis"] = diagnosis
    if not diagnosis["n_cases"]:
        raise SystemExit("the train pass produced no failures to learn from")

    # 3 — pick the target under the campaign's rotation rule
    chosen, why = camp.pick_target(diagnosis["counts"], past, requested=target)
    steps["target"] = {"agent": chosen, "why": why}
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
    verdict["gate_run_id"] = teacher.log_gate_verdict(
        verdict, campaign=campaign, label=label
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
