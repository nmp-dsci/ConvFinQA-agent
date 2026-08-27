"""Build unresolved_cases_<variant>.json from CaseResults."""

from __future__ import annotations

import json
from pathlib import Path

from convfinqa.diagnosis.models import CaseResult


def build_unresolved_cases(
    results: list[CaseResult], unresolved_path: Path
) -> Path:
    unresolved: list[dict] = []
    for r in results:
        if r.resolved:
            continue
        unresolved.append(
            {
                "report_id": r.report_id,
                "turn_index": r.turn_index,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "original_pred_answer": r.original_pred_answer,
                "router_diagnosis": (
                    r.router_diagnosis.model_dump() if r.router_diagnosis else None
                ),
                "attempts": [a.model_dump(exclude={"full_prompt"}) for a in r.attempts],
            }
        )
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_path.write_text(json.dumps(unresolved, indent=2, default=str))
    return unresolved_path
