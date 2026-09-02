"""The release gate (M3): the holdout opens once, for the champion, on purpose.

Promotion runs on the test split; the holdout exists for the moment a champion
is about to ship. Opening it consumes its unseen-ness for every version that
existed at the time, so:

- only the current champion can be released — the holdout confirms a decision
  already made on test evidence, it does not make the decision;
- every opening is appended to the registry history (``holdout_opened``), so a
  later version cannot quietly claim the holdout as unseen;
- the CLI demands an explicit acknowledgement flag before running anything.

On a pass the ``released`` alias moves to the candidate; on a fail it stays
where it was and the failed opening is still recorded — a burned holdout that
looks unburned is the worst outcome this module exists to prevent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


async def run_release(
    *,
    baseline: str | None = None,
    n_reports: int | None = None,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Open the holdout for the current champion vs the last released version."""
    from convfinqa.evalloop.gate import gate_runs
    from convfinqa.evalloop.runner import run_split
    from convfinqa.tracking import registry

    doc = registry.load()
    candidate = doc.aliases.get("champion")
    if not candidate:
        raise SystemExit("no champion to release")
    baseline = baseline or doc.aliases.get("released")
    if baseline == candidate:
        raise SystemExit(f"{candidate} is already the released version")

    prior = [e for e in doc.history if e.get("event") == "holdout_opened"]
    if any(e.get("candidate") == candidate for e in prior):
        raise SystemExit(
            f"the holdout was already opened for {candidate} — it cannot be "
            "reused as unseen evidence for the same version"
        )

    cand_summary = await run_split(
        "holdout", candidate, n_reports=n_reports, concurrency=concurrency
    )
    verdict: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_run": cand_summary["run_name"],
        "candidate_accuracy": cand_summary["accuracy"],
    }
    passed = True
    if baseline:
        base_summary = await run_split(
            "holdout", baseline, n_reports=n_reports, concurrency=concurrency
        )
        result, stats = gate_runs(
            base_summary["csv"],
            cand_summary["csv"],
            baseline_version=baseline,
            candidate_version=candidate,
        )
        passed = stats["accuracy_delta"] >= 0  # confirmatory: must not regress
        verdict.update(
            baseline_run=base_summary["run_name"],
            baseline_accuracy=base_summary["accuracy"],
            comparison=stats,
        )

    verdict["passed"] = passed
    doc = registry.load()  # reload: the runs registered themselves meanwhile
    doc.history.append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": "holdout_opened",
            "candidate": candidate,
            "baseline": baseline,
            "passed": passed,
            "verdict": {k: v for k, v in verdict.items() if k != "passed"},
        }
    )
    if passed:
        doc.aliases["released"] = candidate
    registry.save(doc)
    return verdict
