"""The production-readiness scorecard: a hand-maintained record, checked in CI.

`evaluation/readiness.json` scores the repo against the portfolio's nine
rubric dimensions (R1–R9). It is written by hand, because the judgements it
records — "partial", "designed" — are not derivable from the tracking store.
What *is* checkable is whether the record still points at real things: every
proof path must exist, every app route must be one the router serves, and the
headline score must agree with the rows it summarises. `problems()` reports
where it does not, and `story_check` runs it on every pull request so the
scorecard cannot quietly describe a repo that has moved on.

Nothing here builds a model; the route that serves this file stays live in
the demo container.
"""

from __future__ import annotations

import json
from typing import Any

from convfinqa.config import EVAL_ROOT, REPO_ROOT

READINESS_PATH = EVAL_ROOT / "readiness.json"

STATUSES = ("shipped", "partial", "designed", "na")
"""The status vocabulary; the file's `statuses` object describes each one."""

EXPECTED_REFS = tuple(f"R{i}" for i in range(1, 10))
"""The rubric's nine dimensions, in the order the page shows them."""

KNOWN_ROUTES = (
    "/",
    "/chat",
    "/admin",
    "/admin/evaluations",
    "/admin/dataset",
    "/admin/campaigns",
    "/admin/runtimes",
    "/admin/experiments",
    "/admin/traces",
    "/admin/research",
    "/admin/system",
    "/admin/readiness",
)
"""Every route the frontend router serves. An `app` link outside this list
would send a reader to a blank page."""


def load_readiness() -> dict[str, Any]:
    """The committed scorecard, as written."""
    data: dict[str, Any] = json.loads(READINESS_PATH.read_text())
    return data


def problems(data: dict[str, Any] | None = None) -> list[str]:
    """Everything in the scorecard that no longer holds. Empty means current.

    Checks, in order: the rows are exactly R1..R9 in order; every status is in
    the vocabulary; a row that is not `shipped` explains itself in `how`; every
    proof path exists in the repo; every app link is a route the frontend
    serves; the headline score counts the shipped rows.
    """
    if data is None:
        if not READINESS_PATH.exists():
            return [f"{READINESS_PATH} is missing"]
        data = load_readiness()

    out: list[str] = []
    rows: list[dict[str, Any]] = list(data.get("rows") or [])

    refs = [str(r.get("ref", "")) for r in rows]
    if tuple(refs) != EXPECTED_REFS:
        out.append(
            f"rows must be exactly {', '.join(EXPECTED_REFS)} in order; got "
            f"{', '.join(refs) or 'nothing'}"
        )

    for row in rows:
        ref = str(row.get("ref", "?"))
        status = str(row.get("status", ""))
        if status not in STATUSES:
            out.append(f"{ref}: status {status!r} is not one of {', '.join(STATUSES)}")
        if status != "shipped" and not str(row.get("how", "")).strip():
            out.append(
                f"{ref}: status {status!r} needs a `how` that says what is missing"
            )
        for path in row.get("proof") or []:
            if not (REPO_ROOT / str(path)).exists():
                out.append(f"{ref}: proof path {path!r} does not exist in the repo")
        for route in row.get("app") or []:
            if route not in KNOWN_ROUTES:
                out.append(f"{ref}: app route {route!r} is not a route the app serves")

    n_shipped = sum(1 for r in rows if r.get("status") == "shipped")
    expected_score = f"{n_shipped} / {len(rows)}"
    score = str(data.get("score", ""))
    if score != expected_score:
        out.append(
            f"score {score!r} does not match the rows — {n_shipped} of "
            f"{len(rows)} are shipped, so it should read {expected_score!r}"
        )
    return out
