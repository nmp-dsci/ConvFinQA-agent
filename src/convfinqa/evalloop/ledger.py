"""The prompt writer's memory: what was tried on each agent, and how it went.

The M2 teacher had memory of a kind — the last five runs' *diagnoses* — but it
was memory without feedback. It could see what had been proposed and never
whether any of it worked, so a rejected idea could be proposed again, sharpened,
and rejected again, for as many cycles as the campaign lasted.

This joins the two halves that were never joined: every ``kind=propose`` run
carries the prompt it wrote and the agent it targeted; every ``kind=gate`` run
carries the verdict for a candidate version. Keyed on that version, one query
produces, per target agent, the attempt history a writer needs before it writes
again — *this is what I changed, this is what happened*.

It also answers the question the bundle version cannot: *what has this agent's
current prompt already been shown to get wrong?* A bundle version is four
prompts, so `v2` and `v8` share one preprocess prompt but not one retriever
prompt, and diagnoses filed under either bundle bear on preprocess equally.
`diagnoses_for_agent` keys on the per-agent prompt **hash** instead, gathering
every failure recorded against the exact text the writer is about to replace.

Everything here is read-only and best-effort: a tracking store that is down
degrades the writer to the memoryless behaviour it had before, never blocks it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPTIMIZATION_EXPERIMENT = "convfinqa-optimization"


def _client() -> Any:
    from mlflow.tracking import MlflowClient

    from convfinqa.tracking import mlflow_log

    mlflow_log._mlflow()
    return MlflowClient(tracking_uri=mlflow_log.tracking_uri())


def _runs(client: Any, experiment: str, kind: str, limit: int = 200) -> list[Any]:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    return list(
        client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.kind = '{kind}'",
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    )


def _artifact_json(client: Any, run_id: str, name: str) -> Any:
    try:
        return json.loads(Path(client.download_artifacts(run_id, name)).read_text())
    except Exception:  # noqa: BLE001 — an unreadable artifact is missing memory, not an error
        return None


def attempts(
    *,
    target_agent: str | None = None,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Past prompt rewrites with their gate outcomes, newest first.

    `target_agent` narrows to one agent's lineage, which is what the writer
    wants: the history that bears on the prompt in front of it.

    Read from the rewrites and gates ledgers first (`evalloop.ledgers`); the
    MLflow search is the fallback for a store that predates them.
    """
    from_ledger = _attempts_from_ledger(target_agent=target_agent, limit=limit)
    if from_ledger is not None:
        return from_ledger
    try:
        client = _client()
        proposals = _runs(client, experiment, "propose")
        verdicts = _runs(client, experiment, "gate")
    except Exception:  # noqa: BLE001
        return []

    by_version: dict[str, dict[str, Any]] = {}
    for run in verdicts:
        version = run.data.params.get("candidate_version", "")
        if version and version not in by_version:
            record = _artifact_json(client, run.info.run_id, "verdict.json")
            if record is not None:
                flips = _artifact_json(client, run.info.run_id, "flips.json") or {}
                record = {**record, "flips": flips}
            by_version[version] = record or {
                "promoted": run.data.tags.get("promoted") == "true",
                "reason": run.data.tags.get("reason", ""),
                **{
                    k: v
                    for k, v in run.data.metrics.items()
                    if k in {"accuracy_delta", "cluster_p_one_sided", "n_compared"}
                },
            }

    out: list[dict[str, Any]] = []
    for run in proposals:
        agent = run.data.params.get("target_agent", "")
        if target_agent and agent != target_agent:
            continue
        version = run.data.params.get("new_version", "")
        proposal = _artifact_json(client, run.info.run_id, "proposal.json") or {}
        outcome = by_version.get(version)
        out.append(
            {
                "version": version,
                "base_version": run.data.params.get("prompts_version", ""),
                "target_agent": agent,
                "at": run.info.start_time,
                "rationale": proposal.get("rationale", ""),
                "summary_of_changes": proposal.get("summary_of_changes", ""),
                "prompt": proposal.get("prompt", ""),
                "outcome": "promoted"
                if (outcome or {}).get("promoted")
                else ("rejected" if outcome else "not yet gated"),
                "verdict": (outcome or {}).get("reason", ""),
                "accuracy_delta": (outcome or {}).get("accuracy_delta"),
                "cluster_p_one_sided": (outcome or {}).get("cluster_p_one_sided"),
                "fixed": (outcome or {}).get("fail_to_pass"),
                "broken": (outcome or {}).get("pass_to_fail"),
                "broken_cases": ((outcome or {}).get("flips") or {}).get("broken", []),
            }
        )
        if len(out) >= limit:
            break
    return out


def _attempts_from_ledger(
    *, target_agent: str | None, limit: int
) -> list[dict[str, Any]] | None:
    """`attempts` read off the flat ledgers; None when they hold no rewrites.

    Same shape as the MLflow path. The gates ledger carries flip *counts* by
    class, not the flips themselves, so the broken questions — which the
    writer's prompt names — are still read from the gate run's ``flips.json``
    when the store is reachable, and are empty when it is not.
    """
    from convfinqa.evalloop import ledgers

    try:
        rewrites = ledgers.load("rewrites", runtime="multi_agent")
        gates = ledgers.load("gates", runtime="multi_agent")
    except Exception:  # noqa: BLE001 — an unreadable ledger falls back to the store
        return None
    if rewrites.empty:
        return None
    by_rewrite: dict[str, Any] = {}
    by_version: dict[str, Any] = {}
    for g in gates.itertuples():
        if g.rewrite_id:
            by_rewrite[str(g.rewrite_id)] = g
        by_version.setdefault(str(g.candidate_version), g)

    client: Any = None
    out: list[dict[str, Any]] = []
    ordered = rewrites.sort_values("proposed_at", ascending=False)
    for r in ordered.itertuples():
        if target_agent and str(r.target) != target_agent:
            continue
        g = by_rewrite.get(str(r.rewrite_id)) or by_version.get(str(r.new_version))
        prompt = ""
        try:
            import convfinqa.prompts as prompts_pkg

            prompt = prompts_pkg.load(str(r.new_version))[str(r.target)]
        except Exception:  # noqa: BLE001 — a rewrite never written has no module
            prompt = ""
        broken_cases: list[dict[str, Any]] = []
        if g is not None and g.gate_run_id:
            try:
                client = client or _client()
                flips = _artifact_json(client, str(g.gate_run_id), "flips.json") or {}
                broken_cases = list(flips.get("broken", []))
            except Exception:  # noqa: BLE001
                broken_cases = []
        delta = None if g is None or g.delta_pp is None else float(g.delta_pp) / 100.0
        out.append(
            {
                "version": str(r.new_version),
                "base_version": str(r.base_version),
                "target_agent": str(r.target),
                "at": str(r.proposed_at),
                "rationale": str(r.rationale or ""),
                "summary_of_changes": str(r.edit_text or ""),
                "prompt": prompt,
                "outcome": "promoted"
                if g is not None and bool(g.promoted)
                else ("rejected" if g is not None else "not yet gated"),
                "verdict": str(g.reason) if g is not None else "",
                "accuracy_delta": delta,
                "cluster_p_one_sided": None
                if g is None or g.p_value is None
                else float(g.p_value),
                "fixed": None if g is None else int(g.fixed),
                "broken": None if g is None else int(g.broken),
                "broken_cases": broken_cases,
            }
        )
        if len(out) >= limit:
            break
    return out


def ledger_text(target_agent: str, limit: int = 12, broken_examples: int = 4) -> str:
    """The attempt history as prose for the writer's prompt.

    Deliberately blunt about outcomes: a rejected attempt is labelled REJECTED
    with its delta and p, because the point of showing it is to stop the writer
    re-proposing it. An empty history says so explicitly rather than being
    omitted, so the writer can tell "nothing tried yet" from "history withheld".
    """
    rows = attempts(target_agent=target_agent, limit=limit)
    if not rows:
        return (
            f"\n\n## Prior attempts on {target_agent}\n"
            "None. This is the first rewrite of this agent's prompt in the "
            "recorded history.\n"
        )
    lines = [f"\n\n## Prior attempts on {target_agent} (newest first)"]
    for r in rows:
        head = f"- {r['version']} — {r['outcome'].upper()}"
        if r.get("accuracy_delta") is not None:
            head += f" (Δ {float(r['accuracy_delta']) * 100:+.2f}pp"
            if r.get("cluster_p_one_sided") is not None:
                head += f", p={float(r['cluster_p_one_sided']):.3f}"
            head += ")"
        lines.append(head)
        if r.get("fixed") is not None or r.get("broken") is not None:
            lines.append(
                f"  it fixed {int(r.get('fixed') or 0)} questions "
                f"and broke {int(r.get('broken') or 0)}"
            )
        if r.get("summary_of_changes"):
            lines.append(f"  changed: {r['summary_of_changes']}")
        if r.get("rationale"):
            lines.append(f"  reasoning: {r['rationale'][:400]}")
        # The counts say a rewrite cost eighteen questions; these say which,
        # which is the only form of that fact a writer can act on.
        for case in (r.get("broken_cases") or [])[:broken_examples]:
            lines.append(
                f"  BROKE {case.get('report_id', '?')} q{case.get('q_order', '?')}: "
                f"{str(case.get('question', ''))[:160]}"
                f" | gold {case.get('gold_answer')}"
                f" | before {case.get('baseline_answer')}"
                f" -> after {case.get('candidate_answer')}"
            )
    lines.append(
        "\nDo not re-propose a change that was already REJECTED unless you can "
        "say what is different this time. Where a past attempt broke questions, "
        "your rewrite must not break them the same way — a change that fixes as "
        "much as it breaks is a rejection, and that is how most of these were "
        "lost."
    )
    return "\n".join(lines)


def _agent_prompt_hash(version: str, agent: str) -> str | None:
    """The hash of one agent's prompt inside a bundle version, or None."""
    try:
        from convfinqa.tracking import prompt_ledger

        return prompt_ledger.resolve(version)[agent]["hash"]
    except Exception:  # noqa: BLE001 — an unresolvable version is no memory, not an error
        return None


def diagnoses_for_agent(
    agent: str,
    version: str,
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    limit_runs: int = 25,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Every diagnosis filed against `agent` while it ran `version`'s prompt.

    Scoped by the agent's prompt **hash**, not the bundle version it happened to
    be diagnosed under: a bundle is four prompts, so failures recorded under two
    different bundle versions bear on this agent identically whenever the two
    share its text. That is the population the writer actually wants — the full
    record of what this exact prompt gets wrong, rather than the slice of it
    that one run happened to sample.

    Attribution is the gold-derived `derived_agent`, matching what targets an
    experiment; a case the teacher reassigned elsewhere is not this agent's.
    """
    want = _agent_prompt_hash(version, agent)
    if want is None:
        return []
    from_ledger = _diagnoses_from_ledger(agent, want, limit=limit)
    if from_ledger is not None:
        return from_ledger
    try:
        client = _client()
        runs = _runs(client, experiment, "diagnose", limit=limit_runs)
    except Exception:  # noqa: BLE001
        return []

    seen_hash: dict[str, str | None] = {}
    out: list[dict[str, Any]] = []
    for run in runs:
        run_version = run.data.params.get("prompts_version", "")
        if not run_version:
            continue
        if run_version not in seen_hash:
            seen_hash[run_version] = _agent_prompt_hash(run_version, agent)
        if seen_hash[run_version] != want:
            continue
        try:
            local = client.download_artifacts(run.info.run_id, "diagnoses.jsonl")
            rows = [
                json.loads(line)
                for line in Path(local).read_text().splitlines()
                if line.strip()
            ]
        except Exception:  # noqa: BLE001 — one unreadable run is not a failure
            continue
        for d in rows:
            if str(d.get("derived_agent") or d.get("failed_agent")) != agent:
                continue
            out.append(
                {
                    "report_id": d.get("report_id"),
                    "turn_index": d.get("turn_index"),
                    "version": run_version,
                    "failure_mode": d.get("failure_mode"),
                    "what_went_wrong": d.get("what_went_wrong"),
                    "attribution_reason": d.get("attribution_reason", ""),
                    "proposed_rule": d.get("proposed_rule"),
                    "gold_suspect": d.get("gold_suspect"),
                }
            )
            if len(out) >= limit:
                return out
    return out


def _diagnoses_from_ledger(
    agent: str, want: str, *, limit: int
) -> list[dict[str, Any]] | None:
    """`diagnoses_for_agent` off the diagnoses ledger; None when it is empty.

    Each ledger row records the hash of the prompt its *attributed* agent was
    running, so the filter is one equality rather than a resolve per run.
    """
    from convfinqa.evalloop import ledgers

    try:
        frame = ledgers.load("diagnoses", runtime="multi_agent")
    except Exception:  # noqa: BLE001
        return None
    if frame.empty:
        return None
    hit = frame[(frame["derived_agent"] == agent) & (frame["prompt_hash"] == want)]
    hit = hit.sort_values("diagnosed_at", ascending=False)
    return [
        {
            "report_id": r.report_id,
            "turn_index": r.turn_index,
            "version": r.version,
            "failure_mode": r.label,
            "what_went_wrong": r.what_went_wrong,
            "attribution_reason": r.attribution_reason or "",
            "proposed_rule": r.fix_hint,
            "gold_suspect": bool(r.gold_suspect),
        }
        for r in hit.head(limit).itertuples()
    ]


def _fault_history_from_ledger(
    want: dict[str, str | None], *, exclude_run_id: str | None
) -> dict[str, dict[str, Any]] | None:
    """`fault_history` off the diagnoses ledger; None when it is empty.

    One diagnose pass is one group — keyed on its run id, or on
    (version, diagnosed_at) for backfilled rows that predate run ids — and the
    denominator is that pass's *attributed* cases, exactly as the MLflow path
    counts `n_attributed`.
    """
    from convfinqa.evalloop import ledgers
    from convfinqa.evalloop.teacher import AGENTS

    try:
        frame = ledgers.load("diagnoses", runtime="multi_agent")
    except Exception:  # noqa: BLE001
        return None
    if frame.empty:
        return None
    out: dict[str, dict[str, Any]] = {
        a: {"faults": 0, "cases": 0, "n_runs": 0, "versions": []} for a in AGENTS
    }
    seen_hash: dict[str, dict[str, str | None]] = {}
    key = frame["diagnosis_run_id"].where(
        frame["diagnosis_run_id"].astype(bool),
        frame["version"].astype(str) + "@" + frame["diagnosed_at"].astype(str),
    )
    for group_key, rows in frame.groupby(key, sort=True):
        if exclude_run_id and str(group_key) == exclude_run_id:
            continue
        version = str(rows["version"].iloc[0])
        attributed = rows[rows["derived_agent"].isin(AGENTS)]
        n = len(attributed)
        if not version or not n:
            continue
        if version not in seen_hash:
            seen_hash[version] = {a: _agent_prompt_hash(version, a) for a in AGENTS}
        for agent in AGENTS:
            if want[agent] is None or seen_hash[version][agent] != want[agent]:
                continue
            out[agent]["faults"] += int((attributed["derived_agent"] == agent).sum())
            out[agent]["cases"] += n
            out[agent]["n_runs"] += 1
            out[agent]["versions"].append(version)
    for agent in AGENTS:
        _score(out[agent])
    return out


def _run_csv_for(version: str, split: str, directory: Path) -> Path | None:
    """The committed predictions CSV for one version's run on one split."""
    matches = sorted(directory.glob(f"evalloop-{split}*-{version}·*.csv"))
    return matches[-1] if matches else None


def backfill_flips(
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    predictions_dir: Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Attach `flips.json` to gate runs recorded before the gate wrote one.

    Gate verdicts logged before this existed carry `fail_to_pass`/`pass_to_fail`
    counts and nothing else, so the prompt writer can read that a past attempt
    broke twenty-three questions but never which — the half of that fact it can
    act on. The flips are recoverable: both arms' predictions CSVs are committed,
    and the comparison is pure arithmetic over them.

    Each recomputation is **checked against the recorded verdict** before
    anything is written. If the recomputed counts disagree, the CSVs are not the
    ones that produced the verdict and the run is skipped rather than annotated
    with plausible-looking flips from the wrong comparison — a wrong record here
    would be read as history by every future writer.
    """
    from convfinqa.evalloop.gate import load_run_csv
    from convfinqa.tracking.comparator import compare_frames

    directory = predictions_dir or (
        Path(__file__).resolve().parents[3] / "evaluation" / "predictions" / "evalloop"
    )
    client = _client()
    out: list[dict[str, Any]] = []
    for run in _runs(client, experiment, "gate"):
        run_id = run.info.run_id
        name = run.data.tags.get("mlflow.runName", run_id)
        params = run.data.params
        base_v = params.get("baseline_version", "")
        cand_v = params.get("candidate_version", "")
        split = params.get("evidence_split", "test")
        existing = {a.path for a in client.list_artifacts(run_id)}
        if "flips.json" in existing:
            out.append({"run": name, "status": "already present"})
            continue

        verdict = _artifact_json(client, run_id, "verdict.json") or {}
        base_csv = _run_csv_for(base_v, split, directory)
        cand_csv = _run_csv_for(cand_v, split, directory)
        if base_csv is None or cand_csv is None:
            out.append(
                {
                    "run": name,
                    "status": "skipped — predictions CSV not found",
                    "baseline_csv": str(base_csv),
                    "candidate_csv": str(cand_csv),
                }
            )
            continue

        result = compare_frames(
            load_run_csv(base_csv),
            load_run_csv(cand_csv),
            baseline_version=base_v,
            candidate_version=cand_v,
        )
        fixed, broken = len(result.improvements), len(result.regressions)
        want_fixed = verdict.get("fail_to_pass")
        want_broken = verdict.get("pass_to_fail")
        if (want_fixed, want_broken) != (fixed, broken):
            out.append(
                {
                    "run": name,
                    "status": "skipped — recomputation disagrees with the verdict",
                    "recorded": {"fixed": want_fixed, "broken": want_broken},
                    "recomputed": {"fixed": fixed, "broken": broken},
                }
            )
            continue

        payload = {
            "broken": [f.as_dict() for f in result.regressions],
            "fixed": [f.as_dict() for f in result.improvements],
            "backfilled_from": {
                "baseline_csv": base_csv.name,
                "candidate_csv": cand_csv.name,
            },
        }
        if not dry_run:
            client.log_dict(run_id, payload, "flips.json")
        out.append(
            {
                "run": name,
                "status": "would write" if dry_run else "written",
                "fixed": fixed,
                "broken": broken,
            }
        )
    return out


def fault_history(
    base_version: str,
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    exclude_run_id: str | None = None,
    limit_runs: int = 40,
) -> dict[str, dict[str, Any]]:
    """Pooled first-fault evidence per agent, over runs sharing its prompt.

    Train is resampled every cycle, so one draw's fault counts are a sample of
    about fifty cases split four ways — and the top two agents routinely sit
    within a couple of cases of each other, which is inside the noise of a
    single draw. Three v2 draws ranked preprocess 18, 26 and 14 against
    retriever 15, 15 and 16: the same prompts, and a different winner depending
    on which reports were drawn.

    Pooling is per agent and keyed on **that agent's prompt hash**, so each
    agent accumulates only the draws in which it was running the text it is
    still running. That makes the contributing run sets different lengths — an
    agent rewritten last cycle has one draw, one untouched since v2 has four —
    which is why this reports a `rate` (faults per diagnosed case) rather than a
    count. Ranking on raw totals would systematically favour whichever agent had
    gone longest without a rewrite, which is precisely backwards: the agent we
    know least about is the one just changed.
    """
    from convfinqa.evalloop.teacher import AGENTS

    want = {a: _agent_prompt_hash(base_version, a) for a in AGENTS}
    from_ledger = _fault_history_from_ledger(want, exclude_run_id=exclude_run_id)
    if from_ledger is not None:
        return from_ledger
    try:
        client = _client()
        runs = _runs(client, experiment, "diagnose", limit=limit_runs)
    except Exception:  # noqa: BLE001
        runs = []

    out: dict[str, dict[str, Any]] = {
        a: {"faults": 0, "cases": 0, "n_runs": 0, "versions": []} for a in AGENTS
    }
    seen_hash: dict[str, dict[str, str | None]] = {}
    for run in runs:
        if exclude_run_id and run.info.run_id == exclude_run_id:
            continue
        version = run.data.params.get("prompts_version", "")
        metrics = run.data.metrics
        # An aborted pass logs no fault metrics; it is absence of evidence.
        if not version or not any(f"faults_{a}" in metrics for a in AGENTS):
            continue
        # The denominator is *attributed* cases, not diagnosed ones. Since
        # 2026-09-04 attribution may return a verdict that names no agent
        # (`gold_suspect`, `ambiguous`, `unscorable`); those are evidence about
        # the dataset or about our own instrumentation, not about any prompt, so
        # counting them below the line would deflate every agent's fault rate
        # equally and make the Wilson bound read as though we had more evidence
        # than we do. Older runs logged no `n_attributed`, and for them the sum
        # of the fault counts *is* the attributed total — so the fallback is
        # exact rather than approximate, and old and new runs pool correctly.
        n = int(
            metrics.get("n_attributed")
            or sum(metrics.get(f"faults_{a}", 0.0) for a in AGENTS)
        )
        if not n:
            continue
        if version not in seen_hash:
            seen_hash[version] = {a: _agent_prompt_hash(version, a) for a in AGENTS}
        for agent in AGENTS:
            if want[agent] is None or seen_hash[version][agent] != want[agent]:
                continue
            out[agent]["faults"] += int(metrics.get(f"faults_{agent}", 0.0))
            out[agent]["cases"] += n
            out[agent]["n_runs"] += 1
            out[agent]["versions"].append(version)
    for agent in AGENTS:
        _score(out[agent])
    return out


#: The four subagents, for validating a verdict read back from an artifact.
AGENT_NAMES = ("triage", "preprocess", "retriever", "calculator")

Z_95 = 1.959963984540054


def _score(entry: dict[str, Any]) -> None:
    """Set `rate` and the `score` targeting ranks on, in place.

    `score` is the **lower bound of the Wilson interval** on the fault rate, not
    the rate itself. Ranking on the raw rate re-creates the very problem pooling
    exists to solve, because the agents do not carry equal evidence: c02-e01
    picked the retriever at 18/45 = 40.0% over preprocess at 71/195 = 36.4%, and
    the retriever's 40% was one draw of a prompt rewritten the cycle before while
    preprocess's was four draws. A point estimate lets the noisier side win on
    noise.

    The Wilson bound is the standard fix and needs no tuning: it penalises a
    small sample in proportion to how small it is, so a freshly rewritten agent
    has to look *clearly* worse — not marginally worse — before the loop spends
    an experiment on it. It stays strictly positive whenever any fault is
    recorded, so a new agent is never unreachable, which was the other failure.
    """
    faults, cases = int(entry.get("faults", 0)), int(entry.get("cases", 0))
    entry["rate"] = (faults / cases) if cases else 0.0
    if not cases or not faults:
        entry["score"] = 0.0
        return
    p_hat, z = faults / cases, Z_95
    denom = 1.0 + z * z / cases
    centre = p_hat + z * z / (2 * cases)
    margin = z * ((p_hat * (1 - p_hat) / cases) + z * z / (4 * cases * cases)) ** 0.5
    entry["score"] = max(0.0, (centre - margin) / denom)


def merge_draw(
    pooled: dict[str, dict[str, Any]], counts: dict[str, int], version: str
) -> dict[str, dict[str, Any]]:
    """Fold the run just diagnosed into the pooled evidence.

    Added explicitly rather than read back, so the current draw counts exactly
    once no matter how promptly the tracking store makes the new run visible.
    """
    from convfinqa.evalloop.teacher import AGENTS

    n = sum(counts.get(a, 0) for a in AGENTS)
    out = {
        a: dict(pooled.get(a, {"faults": 0, "cases": 0, "n_runs": 0, "versions": []}))
        for a in AGENTS
    }
    for agent in AGENTS:
        out[agent]["faults"] = int(out[agent].get("faults", 0)) + int(
            counts.get(agent, 0)
        )
        out[agent]["cases"] = int(out[agent].get("cases", 0)) + n
        out[agent]["n_runs"] = int(out[agent].get("n_runs", 0)) + 1
        out[agent]["versions"] = [*out[agent].get("versions", []), version]
        _score(out[agent])
    return out


def _adjudications(client: Any, run_id: str) -> dict[tuple[str, int], str]:
    """Adjudicated verdicts recorded on a diagnose run, by (report, turn).

    Read back from the run's own `diagnoses.jsonl`, which records `adjudicated`
    and the resulting `derived_agent` per case. Absent or unreadable, the caller
    simply keeps `ambiguous` — a missing artifact must not invent a verdict.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        local = _Path(client.download_artifacts(run_id, "diagnoses.jsonl"))
        rows = [
            _json.loads(line) for line in local.read_text().splitlines() if line.strip()
        ]
    except Exception:  # noqa: BLE001 — no artifact is a normal, older run
        return {}
    return {
        (str(r.get("report_id")), int(r.get("turn_index", 0))): str(r["derived_agent"])
        for r in rows
        if r.get("adjudicated") and r.get("derived_agent") in AGENT_NAMES
    }


def backfill_attribution(
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Recompute every past diagnose run's fault counts under the current rule.

    Attribution was rewritten on 2026-09-04 and moved 37.4% of 554 first-wrong
    cases. `fault_history` pools draws across cycles and ranks them on a Wilson
    bound, so a store holding counts from *both* rules is not a larger sample of
    one measurement — it is two measurements averaged, and the pooling that was
    built to remove noise would be adding a bias instead.

    Runs already carrying the current rule's fingerprint are skipped; a run
    scored by an older rule is recomputed rather than reported as done. `force`
    recomputes regardless. See `stage_scores.attribution_rule_id`.

    Recomputation, not migration: each run names the CSV it diagnosed in
    `source_csv`, that CSV is committed, and attribution is a pure function of
    it. A run whose CSV is missing is **skipped, not estimated** — the same
    position `backfill_flips` takes, for the same reason. Nothing else on the
    run is touched: the diagnoses artifact still records what the teacher said
    at the time, which is history and not ours to rewrite.
    """
    import pandas as pd

    from convfinqa.evalloop import stage_scores
    from convfinqa.evalloop.teacher import AGENTS

    client = _client()
    out: list[dict[str, Any]] = []
    for run in _runs(client, experiment, "diagnose"):
        run_id = run.info.run_id
        name = run.data.tags.get("mlflow.runName", run_id)
        source = run.data.params.get("source_csv", "")
        path = Path(source)
        if not source or not path.exists():
            out.append({"run": name, "status": "skipped — source CSV not found"})
            continue
        rule = stage_scores.attribution_rule_id()
        if not force and run.data.tags.get("attribution_rule") == rule:
            out.append({"run": name, "status": "already on the current rule"})
            continue

        df = pd.read_csv(path)
        df["correct"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
        stage_scores.score_rows(df)
        first_wrong = df[df.turn_index == df.first_wrong_turn]
        docs = stage_scores.report_documents()
        # Adjudications the run already paid a model call for. Recomputing
        # without them would push those cases back into `ambiguous` and *lose*
        # information — a backfill that leaves a run worse informed than it
        # found it is not a backfill.
        settled = _adjudications(client, run_id)
        verdicts = []
        for row in stage_scores.with_prior_gold(first_wrong):
            verdict = stage_scores.attribute(
                row, docs.get(str(row.get("report_id")), "")
            )
            if verdict == "ambiguous":
                verdict = settled.get(
                    (str(row.get("report_id")), int(row.get("turn_index", 0))),
                    verdict,
                )
            verdicts.append(verdict)
        counts = {a: verdicts.count(a) for a in AGENTS}
        unattributed = {v: verdicts.count(v) for v in stage_scores.NON_AGENT}
        n_attributed = sum(counts.values())
        before = {a: int(run.data.metrics.get(f"faults_{a}", 0.0)) for a in AGENTS}
        record = {
            "run": name,
            "run_id": run_id,
            "n_cases": len(verdicts),
            "before": before,
            "after": counts,
            "unattributed": unattributed,
            "n_attributed": n_attributed,
            "status": "would rewrite" if dry_run else "rewritten",
        }
        # An `ambiguous` case with no recorded adjudication stays unresolved:
        # settling it needs a model call, and a backfill over the whole store is
        # not the place to spend that. Those cases stay out of the counts
        # entirely, which understates every agent equally rather than guessing.
        if not dry_run:
            for agent, value in counts.items():
                client.log_metric(run_id, f"faults_{agent}", float(value))
            for verdict, value in unattributed.items():
                client.log_metric(run_id, f"unattributed_{verdict}", float(value))
            client.log_metric(run_id, "n_attributed", float(n_attributed))
            # The fingerprint of the rule that produced these counts, so a later
            # backfill can tell "already done" from "done by an older rule".
            client.set_tag(run_id, "attribution_rule", rule)
        out.append(record)
    return out
