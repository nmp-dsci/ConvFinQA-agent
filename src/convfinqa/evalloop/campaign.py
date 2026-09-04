"""Campaigns: a bounded, named series of experiments against one fixed gate split.

Optimisation happens periodically rather than continuously, so the unit of work
is not "an experiment" but a **campaign** — up to five experiments against one
gate split, reviewed as a whole, after which the next campaign starts
deliberately with a new name and possibly a new split.

Two caps are enforced here rather than left to discipline, because both failure
modes are ones a person running the loop will not notice happening:

- **Five experiments per campaign.** Every challenger measured against the same
  gate split spends a little of that split's unseen-ness. Twenty challengers
  against 349 questions will eventually promote noise; five will not, and the
  cap makes the review a scheduled event instead of a thing that never happens.
- **Two consecutive rejections rotate the target.** Without it the loop gets
  stuck: the agent with the most faults keeps being chosen, keeps being
  rewritten, and keeps failing the gate, for as long as the campaign lasts.

State lives in MLflow — the runs *are* the campaign record — so nothing has to
be kept in sync with a separate file.
"""

from __future__ import annotations

from typing import Any

from convfinqa.evalloop.teacher import AGENTS, OPTIMIZATION_EXPERIMENT

MAX_EXPERIMENTS = 5
MAX_CONSECUTIVE_REJECTIONS = 2


def history(
    campaign: str, *, experiment: str = OPTIMIZATION_EXPERIMENT
) -> list[dict[str, Any]]:
    """Every gated experiment in one campaign, oldest first."""
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
            filter_string=f"tags.kind = 'gate' and tags.campaign = '{campaign}'",
            order_by=["attributes.start_time ASC"],
            max_results=100,
        )
    except Exception:  # noqa: BLE001 — no store means no history, not a crash
        return []
    return [
        {
            "run_id": r.info.run_id,
            "at": r.info.start_time,
            "candidate_version": r.data.params.get("candidate_version", ""),
            "baseline_version": r.data.params.get("baseline_version", ""),
            "target_agent": r.data.params.get("target_agent", ""),
            "label": r.data.params.get("experiment_label", ""),
            "promoted": r.data.tags.get("promoted") == "true",
            "accuracy_delta": r.data.metrics.get("accuracy_delta"),
            "cluster_p_one_sided": r.data.metrics.get("cluster_p_one_sided"),
        }
        for r in runs
    ]


def blocked_agents(past: list[dict[str, Any]]) -> set[str]:
    """Agents whose last two experiments both failed the gate — rotate off them."""
    out: set[str] = set()
    for agent in AGENTS:
        theirs = [e for e in past if e["target_agent"] == agent]
        recent = theirs[-MAX_CONSECUTIVE_REJECTIONS:]
        if len(recent) == MAX_CONSECUTIVE_REJECTIONS and not any(
            e["promoted"] for e in recent
        ):
            out.add(agent)
    return out


def check_capacity(campaign: str, past: list[dict[str, Any]]) -> None:
    """Refuse a sixth experiment. Raises with what to do instead."""
    if len(past) >= MAX_EXPERIMENTS:
        raise SystemExit(
            f"campaign {campaign!r} already holds {len(past)} experiments — the "
            f"cap is {MAX_EXPERIMENTS}. Review it as a whole "
            "(`convfinqa-evalloop campaign-status`), then start the next one "
            "with a new --campaign name."
        )


def pick_target(
    counts: dict[str, int],
    past: list[dict[str, Any]],
    *,
    requested: str | None = None,
    pooled: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """The agent this experiment will change, and why it was chosen.

    `pooled` is the accumulated first-fault evidence from every train draw that
    ran this agent's current prompt (see `ledger.fault_history`). When it is
    given the ranking uses the pooled fault **rate** instead of this one draw's
    counts, because one draw is ~50 cases split four ways and the top two agents
    routinely sit a couple of cases apart — inside the noise. Three v2 draws put
    preprocess at 18, 26 and 14 against retriever's 15, 15 and 16, so which
    agent "has the most faults" depended on which reports were drawn.

    An explicit `--target` still honours the rotation rule: being able to
    override the cap by naming the blocked agent would make the cap advisory,
    which is the same as not having it.
    """
    blocked = blocked_agents(past)

    def _weight(agent: str) -> float:
        if pooled:
            return float(pooled.get(agent, {}).get("rate", 0.0))
        return float(counts.get(agent, 0))

    def _evidence(agent: str) -> int:
        if pooled:
            return int(pooled.get(agent, {}).get("faults", 0))
        return int(counts.get(agent, 0))

    ranked = sorted(AGENTS, key=lambda a: (-_weight(a), a))
    if requested:
        if requested in blocked:
            raise SystemExit(
                f"{requested!r} failed its last {MAX_CONSECUTIVE_REJECTIONS} "
                "experiments in this campaign — rotate to another agent, or "
                "start a new campaign if you have a genuinely new idea for it"
            )
        return requested, "named on the command line"
    for agent in ranked:
        if agent in blocked:
            continue
        if not _evidence(agent):
            continue
        if pooled:
            ev = pooled.get(agent, {})
            note = (
                f"highest pooled first-fault rate "
                f"({ev.get('faults', 0)}/{ev.get('cases', 0)} = "
                f"{_weight(agent):.1%} across {ev.get('n_runs', 0)} train "
                f"draw(s) of this prompt; {counts.get(agent, 0)} in this draw)"
            )
        else:
            note = f"most derived first-faults ({counts[agent]})"
        # Only claim the rotation changed the outcome when it actually did —
        # a blocked agent that ranked *below* the pick was never in contention,
        # and saying otherwise would credit the cap for a choice it did not make.
        outranked = sorted(b for b in blocked if _weight(b) > _weight(agent))
        if outranked:
            note += f"; rotated past {', '.join(outranked)}"
        return agent, note
    raise SystemExit(
        "every agent with diagnosed faults has failed twice in a row in this "
        f"campaign ({', '.join(sorted(blocked)) or 'none'}) — the campaign has "
        "nothing left to try and should be reviewed"
    )


def summarise(campaign: str) -> dict[str, Any]:
    """Campaign state: experiments used, what promoted, what is blocked."""
    past = history(campaign)
    return {
        "campaign": campaign,
        "experiments": past,
        "n_experiments": len(past),
        "n_remaining": max(0, MAX_EXPERIMENTS - len(past)),
        "n_promoted": sum(1 for e in past if e["promoted"]),
        "blocked_agents": sorted(blocked_agents(past)),
        "complete": len(past) >= MAX_EXPERIMENTS,
    }
