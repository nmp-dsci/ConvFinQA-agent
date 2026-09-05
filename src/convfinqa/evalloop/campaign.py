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

State is the gates ledger (`evalloop/ledgers.py`, one line per verdict, joined
to the rewrite it judged); MLflow's ``kind=gate`` runs are the fallback for
campaigns recorded before the ledger existed.

Two runtimes run campaigns under the same caps, and differ in what a target is:

- **pipeline** (``runtime="pipeline"``, ledger runtime ``multi_agent``) — the
  target is one of four subagents, and two consecutive rejections *block* it.
- **agent_sdk** — there is one prompt, so the target is a **failure class**
  (a taxonomy label the diagnosis agent files cases under), and two consecutive
  rejections do not block anything: they switch the lineage to *single-area
  mode*, one tagged edit per cycle for the rest of the campaign (see
  `single_area_mode`). SDK draws never pool with pipeline draws — the ranking
  comes from `sdk_teacher.rank_classes`, keyed on the sdk prompt hash.
"""

from __future__ import annotations

from typing import Any

from convfinqa.evalloop.teacher import AGENTS, OPTIMIZATION_EXPERIMENT

MAX_EXPERIMENTS = 5
# The SDK arm is capped lower than the pipeline (owner's instruction,
# 2026-09-05). Two reasons it is a different number rather than an oversight:
# a cycle costs a train draw plus a full gate pass of subscription time instead
# of a couple of dollars of DeepSeek, and `sdk_v1` landed at 90.5% with 33 wrong
# turns left, some of them suspect gold — so the headroom a fifth experiment
# would search barely exists. Per runtime, because the cap bounds optimisation
# work against one fixed gate split and the two arms spend it at different rates.
MAX_EXPERIMENTS_BY_RUNTIME = {"pipeline": 5, "agent_sdk": 2}
MAX_CONSECUTIVE_REJECTIONS = 2


#: Ledger runtime names → the runtime names the CLI and runner use.
_LEDGER_RUNTIME = {"multi_agent": "pipeline", "agent_sdk": "agent_sdk"}


def _epoch_ms(stamp: Any) -> int | None:
    """An ISO ``gated_at`` as epoch milliseconds, matching MLflow's ``start_time``."""
    if stamp is None or stamp == "":
        return None
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(str(stamp)).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _history_from_ledger(campaign: str) -> list[dict[str, Any]] | None:
    """`history` off the gates ledger; None when it holds nothing for `campaign`.

    The target is not a gates column — it belongs to the rewrite the gate
    judged — so the rewrites ledger is joined on ``rewrite_id``: `target` for
    a pipeline rewrite (the subagent), `failure_class` for an SDK one.
    """
    from convfinqa.evalloop import ledgers

    try:
        gates = ledgers.load("gates", campaign=campaign)
        rewrites = ledgers.load("rewrites")
    except Exception:  # noqa: BLE001 — an unreadable ledger falls back to the store
        return None
    if gates.empty:
        return None
    target_of: dict[str, tuple[str, str]] = {}
    for r in rewrites.itertuples():
        rid = str(r.rewrite_id)
        if rid and rid not in target_of:
            target_of[rid] = (str(r.target or ""), str(r.failure_class or ""))
    out: list[dict[str, Any]] = []
    for g in gates.sort_values("gated_at").itertuples():
        runtime = _LEDGER_RUNTIME.get(str(g.runtime), str(g.runtime))
        agent, failure_class = target_of.get(str(g.rewrite_id), ("", ""))
        target = failure_class if runtime == "agent_sdk" else agent
        exp_n = g.experiment_n
        label = (
            f"{campaign}-e{int(exp_n):02d}"
            if exp_n is not None and str(exp_n) != "nan"
            else ""
        )
        out.append(
            {
                "run_id": str(g.gate_run_id or ""),
                "gate_id": str(g.gate_id or ""),
                "at": _epoch_ms(g.gated_at),
                "runtime": runtime,
                "candidate_version": str(g.candidate_version or ""),
                "baseline_version": str(g.baseline_version or ""),
                "target_agent": target,
                "target_class": failure_class if runtime == "agent_sdk" else "",
                "label": label,
                "promoted": bool(g.promoted),
                "accuracy_delta": (
                    None
                    if g.delta_pp is None or str(g.delta_pp) == "nan"
                    else float(g.delta_pp) / 100.0
                ),
                "cluster_p_one_sided": (
                    None
                    if g.p_value is None or str(g.p_value) == "nan"
                    else float(g.p_value)
                ),
            }
        )
    return out


def history(
    campaign: str,
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    runtime: str | None = None,
) -> list[dict[str, Any]]:
    """Every gated experiment in one campaign, oldest first.

    The gates ledger is read first; MLflow's ``kind=gate`` runs are the
    fallback for campaigns that predate it. `runtime` filters to one arm —
    a campaign is one arm's, but the name is only a convention, and a cycle
    must not count the other arm's verdicts against its own cap.
    """
    rows = _history_from_ledger(campaign)
    if rows is None:
        rows = _history_from_mlflow(campaign, experiment=experiment)
    if runtime is not None:
        rows = [r for r in rows if r.get("runtime", "pipeline") == runtime]
    return rows


def _history_from_mlflow(
    campaign: str, *, experiment: str = OPTIMIZATION_EXPERIMENT
) -> list[dict[str, Any]]:
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
    out: list[dict[str, Any]] = []
    for r in runs:
        runtime = r.data.tags.get("runtime") or r.data.params.get("runtime")
        runtime = "agent_sdk" if runtime == "agent_sdk" else "pipeline"
        target_class = r.data.params.get("target_class", "")
        out.append(
            {
                "run_id": r.info.run_id,
                "at": r.info.start_time,
                "runtime": runtime,
                "candidate_version": r.data.params.get("candidate_version", ""),
                "baseline_version": r.data.params.get("baseline_version", ""),
                "target_agent": (
                    target_class
                    if runtime == "agent_sdk"
                    else r.data.params.get("target_agent", "")
                ),
                "target_class": target_class if runtime == "agent_sdk" else "",
                "label": r.data.params.get("experiment_label", ""),
                "promoted": r.data.tags.get("promoted") == "true",
                "accuracy_delta": r.data.metrics.get("accuracy_delta"),
                "cluster_p_one_sided": r.data.metrics.get("cluster_p_one_sided"),
            }
        )
    return out


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


def single_area_mode(past: list[dict[str, Any]]) -> bool:
    """SDK arm: has the lineage failed the gate twice in a row in this campaign?

    The pipeline's answer to two consecutive rejections is to rotate off the
    agent. The SDK arm has one prompt, so there is nothing to rotate to;
    instead the rewrite drops from "one tagged edit per failure class the
    teacher chooses to address" to **one area per cycle** for the rest of the
    campaign, so the next rejection can at least be read against one edit.
    Once entered it is not left — a later promotion does not restore
    multi-area rewrites, because the campaign is the unit of review and the
    reviewer should see one regime per campaign.
    """
    streak = 0
    for exp in past:
        streak = 0 if exp.get("promoted") else streak + 1
        if streak >= MAX_CONSECUTIVE_REJECTIONS:
            return True
    return False


def consecutive_rejections(past: list[dict[str, Any]]) -> int:
    """Rejections since the lineage's last promotion in this campaign."""
    streak = 0
    for exp in reversed(past):
        if exp.get("promoted"):
            break
        streak += 1
    return streak


def max_experiments(runtime: str | None = None) -> int:
    """The experiment cap for `runtime`; the pipeline's 5 when unknown."""
    return MAX_EXPERIMENTS_BY_RUNTIME.get(runtime or "pipeline", MAX_EXPERIMENTS)


def check_capacity(
    campaign: str, past: list[dict[str, Any]], *, runtime: str | None = None
) -> None:
    """Refuse one experiment past the cap. Raises with what to do instead.

    `runtime` selects the cap: the SDK arm's is lower (see
    `MAX_EXPERIMENTS_BY_RUNTIME`). Omitted, it is inferred from the campaign's
    own history so an existing caller keeps its behaviour.
    """
    chosen = runtime or runtime_of(past)
    cap = max_experiments(chosen)
    if len(past) >= cap:
        raise SystemExit(
            f"campaign {campaign!r} already holds {len(past)} experiments — the "
            f"cap for the {chosen!r} runtime is {cap}. Review it as a whole "
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
            # The Wilson lower bound, not the rate — see `ledger._score`. The
            # agents do not carry equal evidence, so a point estimate lets a
            # single noisy draw outrank a well-measured rival.
            entry = pooled.get(agent, {})
            return float(entry.get("score", entry.get("rate", 0.0)))
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
                f"{float(ev.get('rate', 0.0)):.1%}, Wilson lower bound "
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


def pick_target_class(
    ranking: dict[str, dict[str, Any]],
    *,
    requested: str | None = None,
) -> tuple[str, str]:
    """SDK arm: the failure class this experiment will address, and why.

    `ranking` is `sdk_teacher.rank_classes` — per label, the pooled faults,
    the Wilson lower bound and a rank. Top of the ranking wins, on the bound
    rather than the count for the reason `ledger._score` gives: the classes do
    not carry equal evidence. Nothing is ever blocked here: two rejections
    switch the *lineage* to single-area mode instead (`single_area_mode`).
    """
    if requested:
        if requested not in ranking:
            return requested, "named on the command line (no pooled evidence)"
        return requested, "named on the command line"

    def _bound(label: str) -> float:
        return float(ranking[label].get("wilson_lower") or 0.0)

    ranked = sorted(
        (label for label, ev in ranking.items() if int(ev.get("faults") or 0) > 0),
        key=lambda label: (
            int(ranking[label].get("rank") or 10**6),
            -_bound(label),
            label,
        ),
    )
    if not ranked:
        raise SystemExit(
            "no failure class carries any diagnosed fault — nothing to rewrite for"
        )
    top = ranked[0]
    ev = ranking[top]
    return top, (
        f"highest-ranked failure class ({int(ev.get('faults') or 0)}/"
        f"{int(ev.get('n') or 0)} pooled first-wrong cases, Wilson lower bound "
        f"{_bound(top):.1%}; stages {', '.join(ev.get('stages') or []) or '—'})"
    )


def runtime_of(past: list[dict[str, Any]]) -> str:
    """The arm a campaign belongs to, read off its experiments (default pipeline)."""
    runtimes = {str(e.get("runtime") or "pipeline") for e in past}
    return "agent_sdk" if runtimes == {"agent_sdk"} else "pipeline"


def summarise(campaign: str) -> dict[str, Any]:
    """Campaign state: experiments used, what promoted, what is blocked.

    For an SDK campaign `blocked_agents` is always empty (nothing rotates),
    `single_area_mode` says whether the lineage has dropped to one edit per
    cycle, and each experiment's target is a failure class.
    """
    past = history(campaign)
    runtime = runtime_of(past)
    return {
        "campaign": campaign,
        "runtime": runtime,
        "experiments": past,
        "n_experiments": len(past),
        "n_remaining": max(0, max_experiments(runtime) - len(past)),
        "n_promoted": sum(1 for e in past if e["promoted"]),
        "blocked_agents": (
            [] if runtime == "agent_sdk" else sorted(blocked_agents(past))
        ),
        "single_area_mode": (
            single_area_mode(past) if runtime == "agent_sdk" else False
        ),
        "targets": [
            {
                "label": e.get("label", ""),
                "target": e.get("target_agent", ""),
                "kind": "failure_class" if runtime == "agent_sdk" else "subagent",
                "promoted": e.get("promoted", False),
            }
            for e in past
        ],
        "complete": len(past) >= max_experiments(runtime),
    }
