"""`convfinqa-demo-pack` — build the recorded pack the demo replays.

The pack is reconstructed from a committed predictions CSV, not from fresh model
calls. Every row of that CSV already carries the full per-stage IO the live path
produces (`triage_io`, `preprocess_io`, `retriever_io`, `calculator_io`), because
both paths fill the same `capture` dict. So the recording already happened — this
only re-renders it into the event stream the frontend consumes.

Curation is by design, not convenience: the default selection favours reports
that show the system doing something worth watching — number turns, multi-step
programs, and cross-turn reference resolution — rather than whichever reports
happen to sort first. The result is a PR-reviewable JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.serving.demo_pack.store import DemoPack, PackedTurn, pack_path
from convfinqa.tracking.bundle import bundle_fingerprint
from convfinqa.tracking.comparator import load_predictions

DEFAULT_N_REPORTS = 8


def _loads(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def events_from_row(row: pd.Series) -> list[dict[str, Any]]:
    """Rebuild the SSE event stream for one recorded turn.

    Mirrors `pipeline.runner.turn_events` exactly — same event names, same order,
    same payload keys. If that vocabulary ever changes, this function is the
    other half that has to change with it, which is why the shape is spelled out
    here rather than inferred.
    """
    triage = _loads(row.get("triage_io"))
    preprocess = _loads(row.get("preprocess_io"))
    retriever = _loads(row.get("retriever_io"))
    calculator = _loads(row.get("calculator_io"))

    events: list[dict[str, Any]] = []

    events.append({"event": "stage_start", "stage": "triage"})
    if triage.get("output"):
        events.append(
            {
                "event": "stage_output",
                "stage": "triage",
                "output": triage["output"],
                "metrics": triage.get("metrics", {}),
            }
        )

    if preprocess.get("output"):
        events.append({"event": "stage_start", "stage": "preprocess"})
        events.append(
            {
                "event": "stage_output",
                "stage": "preprocess",
                "output": preprocess["output"],
                "metrics": preprocess.get("metrics", {}),
            }
        )

    events.append({"event": "stage_start", "stage": "retriever"})
    if retriever.get("output"):
        events.append(
            {
                "event": "stage_output",
                "stage": "retriever",
                "output": retriever["output"],
                "metrics": retriever.get("metrics", {}),
            }
        )

    if calculator.get("output"):
        events.append({"event": "stage_start", "stage": "calculator"})
        for step in calculator.get("trajectory", []) or []:
            if isinstance(step, dict) and step.get("event") in {
                "tool_call",
                "tool_return",
            }:
                events.append({**step, "stage": "calculator"})
        events.append(
            {
                "event": "stage_output",
                "stage": "calculator",
                "output": calculator["output"],
                "metrics": calculator.get("metrics", {}),
            }
        )

    answer = "" if pd.isna(row.get("pred_answer")) else str(row.get("pred_answer", ""))
    program = (
        "" if pd.isna(row.get("pred_program")) else str(row.get("pred_program", ""))
    )
    events.append({"event": "answer", "answer": answer, "program": program})
    return events


def _showcase_score(group: pd.DataFrame) -> tuple[int, int, int]:
    """Rank a conversation by how much of the system it puts on display.

    Ordered by: turns answered correctly (a demo should show the thing working),
    then how many turns use the full program path, then conversation length.
    """
    correct = int(group["correct"].sum())
    program_turns = int(
        (
            group.get("gold_turn_type", pd.Series(dtype=str)).astype(str) == "Program"
        ).sum()
    )
    return (correct, program_turns, len(group))


def select_reports(df: pd.DataFrame, n: int) -> list[str]:
    """Pick the `n` most demonstrative conversations."""
    ranked = sorted(
        df.groupby("report_id"),
        key=lambda item: _showcase_score(item[1]),
        reverse=True,
    )
    return [report_id for report_id, _ in ranked[:n]]


def build_pack(
    version: str, *, report_ids: list[str] | None = None, n: int
) -> DemoPack:
    """Assemble a pack from the committed predictions for `version`."""
    df = load_predictions(version)
    if "triage_io" not in df.columns:
        raise ValueError(
            f"{version} predictions carry no per-stage IO columns, so no pack can "
            "be built from them. Re-run the evaluation to regenerate the CSV."
        )
    chosen = report_ids or select_reports(df, n)
    turns: list[PackedTurn] = []
    for report_id in chosen:
        group = df[df["report_id"] == report_id].sort_values("turn_index")
        for _, row in group.iterrows():
            turns.append(
                PackedTurn(
                    report_id=str(row["report_id"]),
                    turn_index=int(row["turn_index"]),
                    question=str(row["question"]),
                    answer=(
                        "" if pd.isna(row["pred_answer"]) else str(row["pred_answer"])
                    ),
                    program=str(row.get("pred_program", "") or ""),
                    gold_answer=str(row["gold_answer"]),
                    correct=bool(row["correct"]),
                    events=events_from_row(row),
                )
            )
    return DemoPack(
        turns=turns,
        built_at=datetime.now(timezone.utc).isoformat(),
        bundle=bundle_fingerprint(version=version),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the demo-pack CLI."""
    parser = argparse.ArgumentParser(
        prog="convfinqa-demo-pack",
        description="Build the recorded conversation pack the demo replays.",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Prompt version to record from (default: the registered champion).",
    )
    parser.add_argument(
        "--reports",
        default="",
        help="Comma-separated report ids to include (default: auto-curated).",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N_REPORTS)
    parser.add_argument("--out", default="", help="Output path (default: pack.json).")
    args = parser.parse_args(argv)

    version = args.version
    if not version:
        from convfinqa.tracking import registry

        version = registry.champion() or "v2"

    report_ids = [r.strip() for r in args.reports.split(",") if r.strip()] or None
    pack = build_pack(version, report_ids=report_ids, n=args.n)

    out = Path(args.out) if args.out else pack_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack.as_dict(), indent=2) + "\n")

    n_correct = sum(1 for t in pack.turns if t.correct)
    print(
        f"wrote {out}\n"
        f"  version   : {version}\n"
        f"  reports   : {len(pack.report_ids)}\n"
        f"  turns     : {len(pack.turns)} ({n_correct} correct)\n"
        f"  size      : {out.stat().st_size / 1024:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
