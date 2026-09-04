"""Prompt references: what a trace stores instead of a prompt.

A teacher call's prompt is large and almost entirely redundant. The system
prompt is a constant repeated across every call of a run; the writer's user
prompt carries a whole subagent prompt, every failure against it, and the
attempt ledger. Storing that on ~50 spans per run put megabytes of duplicated
text in the tracking store and still, at a 20k cap, recorded a *truncated*
prompt — the worst of both, since a truncated prompt is neither cheap nor
faithful.

So a span stores a **reference** and the store holds one copy:

- ``teacher_prompt`` — the teacher's or writer's own system prompt, a module
  constant. Named, hashed, resolved from the code.
- ``agent_prompt`` — a pipeline subagent's prompt, which the prompt ledger
  already identifies exactly as ``p2@4bc21f75``. Resolved from the committed
  bundle module.
- ``run_artifact`` — text logged once on the run: the diagnosis memory block
  (identical for every case in a pass) and the writer's fully assembled user
  prompt (one per cycle). Resolved by downloading it.
- ``diagnose_case`` — a row of a committed predictions CSV, identified by
  report id and turn index; the payload is rebuilt by the same function that
  built it originally.

Every ref carries the sha256 of the text it stands for, so a resolution can be
*checked* rather than assumed. That matters because two of these resolve against
code rather than data: if `TEACHER_PROMPT` has been edited since the run, the
hash will not match and `resolve` says so instead of returning today's prompt as
though it were the one that ran. The run's ``code_sha`` param is what to check
out in that case.
"""

from __future__ import annotations

import hashlib
from typing import Any


def sha(text: str) -> str:
    """The first 12 hex of the sha256 — the same shortening the ledger uses."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def teacher_prompt_ref(name: str, text: str) -> dict[str, Any]:
    """A reference to one of this module's own system prompts, by name."""
    return {"kind": "teacher_prompt", "name": name, "sha": sha(text)}


def agent_prompt_ref(agent: str, version: str, text: str) -> dict[str, Any]:
    """A reference to a pipeline subagent's prompt inside a bundle version."""
    from convfinqa.tracking import prompt_ledger

    try:
        seq = prompt_ledger.resolve(version)[agent]["seq"]
    except Exception:  # noqa: BLE001 — a ref is never load-bearing
        seq = ""
    return {
        "kind": "agent_prompt",
        "agent": agent,
        "version": version,
        "seq": seq,
        "sha": sha(text),
    }


def run_artifact_ref(name: str, text: str, *, run_id: str = "") -> dict[str, Any]:
    """A reference to text logged once on the run rather than on every span."""
    return {"kind": "run_artifact", "name": name, "run_id": run_id, "sha": sha(text)}


def diagnose_case_ref(
    csv_path: str,
    report_id: str,
    turn_index: int,
    *,
    memory: str = "",
    text: str = "",
) -> dict[str, Any]:
    """A reference to one diagnosed case: a CSV row plus the pass's memory block.

    `text` is the prompt this stands for. Pass it: without a `sha` there is
    nothing for `resolve` to check against, so a rebuild that has silently
    drifted — a changed `case_payload`, an edited CSV — comes back looking like
    the text that ran.
    """
    ref: dict[str, Any] = {
        "kind": "diagnose_case",
        "csv": str(csv_path),
        "report_id": report_id,
        "turn_index": int(turn_index),
        "memory_artifact": memory,
    }
    if text:
        ref["sha"] = sha(text)
    return ref


def adjudicate_case_ref(
    csv_path: str, report_id: str, turn_index: int, *, text: str = ""
) -> dict[str, Any]:
    """A reference to one adjudicated case — a *different* prompt from its diagnosis.

    The adjudicator is handed a deliberately narrow payload: the question, the
    sub-questions, what came back, and the missing numbers, with the gold answer
    and the pipeline's answer withheld because it is settling one fact rather
    than attributing blame. A `diagnose_case` ref cannot stand for it — that
    kind rebuilds through `teacher.case_payload` and would hand back the full
    diagnosis payload as though it were the text the adjudicator saw.
    """
    ref: dict[str, Any] = {
        "kind": "adjudicate_case",
        "csv": str(csv_path),
        "report_id": report_id,
        "turn_index": int(turn_index),
    }
    if text:
        ref["sha"] = sha(text)
    return ref


class UnresolvedRefError(RuntimeError):
    """The reference could not be turned back into the text it stands for."""


def resolve(ref: dict[str, Any], *, run_id: str = "") -> str:
    """Reconstruct the text a reference stands for.

    Raises `UnresolvedRefError` rather than returning an approximation. A prompt that is
    *nearly* the one that ran is worse than an honest failure: the whole point of
    reading it back is to know what the model actually saw.
    """
    kind = ref.get("kind")
    if kind == "teacher_prompt":
        from convfinqa.evalloop import teacher

        text = getattr(teacher, str(ref.get("name")), None)
        if not isinstance(text, str):
            raise UnresolvedRefError(
                f"no teacher prompt named {ref.get('name')!r} in this code"
            )
    elif kind == "agent_prompt":
        import convfinqa.prompts as prompts_pkg

        try:
            text = prompts_pkg.load(str(ref["version"]))[str(ref["agent"])]
        except Exception as exc:  # noqa: BLE001
            raise UnresolvedRefError(
                f"cannot load {ref.get('version')}: {exc}"
            ) from exc
    elif kind == "run_artifact":
        text = _download_text(str(ref.get("name")), ref.get("run_id") or run_id)
    elif kind == "diagnose_case":
        text = _rebuild_case(ref, run_id=run_id)
    elif kind == "adjudicate_case":
        text = _rebuild_adjudication(ref)
    else:
        raise UnresolvedRefError(f"unknown reference kind {kind!r}")

    want = ref.get("sha")
    if want and sha(text) != want:
        raise UnresolvedRefError(
            f"{kind} {ref.get('name') or ref.get('agent') or ''} has changed since "
            f"the run: recorded {want}, current {sha(text)}. Check out the run's "
            "code_sha to read the text that actually ran."
        )
    return text


def _download_text(name: str, run_id: str) -> str:
    from pathlib import Path

    from mlflow.tracking import MlflowClient

    from convfinqa.tracking import mlflow_log

    if not run_id:
        raise UnresolvedRefError(f"artifact {name!r} needs the run it was logged on")
    mlflow_log._mlflow()
    client = MlflowClient(tracking_uri=mlflow_log.tracking_uri())
    try:
        return Path(client.download_artifacts(run_id, name)).read_text()
    except Exception as exc:  # noqa: BLE001
        raise UnresolvedRefError(
            f"cannot read artifact {name!r} on {run_id}: {exc}"
        ) from exc


def _case_row(ref: dict[str, Any]) -> Any:
    """The committed CSV row a case reference points at, prepared as it was.

    Goes through `teacher.first_wrong_cases` rather than a bare `read_csv`
    because that is what the call site did, and the difference is not cosmetic:
    it derives `prior_gold_answers`, which is not a column in the CSV and which
    both payloads depend on. Reading the file directly rebuilds a *different*
    prompt — caught by the sha guard, which is the whole reason to carry one.
    """
    from pathlib import Path

    from convfinqa.evalloop import teacher

    csv = Path(str(ref["csv"]))
    if not csv.exists():
        raise UnresolvedRefError(f"predictions CSV is gone: {csv}")
    # `iterrows`, not `.iloc[]`, because that is how both call sites walk the
    # frame — and it is not an equivalent choice: `iterrows` coerces a row to a
    # single dtype, so an integer column comes back as `4.0` where `.iloc[]`
    # keeps `4`. The rebuilt prompt then differs from the one that was sent by
    # exactly that much, which the sha guard reports as a changed prompt.
    for _index, row in teacher.first_wrong_cases(csv).iterrows():
        if str(row["report_id"]) == str(ref["report_id"]) and int(
            row["turn_index"]
        ) == int(ref["turn_index"]):
            return row
    raise UnresolvedRefError(
        f"{ref['report_id']} q{ref['turn_index']} is not in {csv.name}"
    )


def _rebuild_case(ref: dict[str, Any], *, run_id: str) -> str:
    """Rebuild a diagnosis user prompt from its CSV row and the pass's memory."""
    from convfinqa.evalloop import teacher

    memory = ""
    if ref.get("memory_artifact"):
        memory = _download_text(str(ref["memory_artifact"]), run_id)
    return teacher.diagnose_prompt_text(teacher.case_payload(_case_row(ref)), memory)


def _rebuild_adjudication(ref: dict[str, Any]) -> str:
    """Rebuild an adjudication user prompt from its CSV row.

    Shares `_case_row` with the diagnosis rebuild but goes through
    `teacher.adjudication_prompt_text`, which is what keeps the two prompts
    distinct on the way back out as well as on the way in.
    """
    from convfinqa.evalloop import teacher

    return teacher.adjudication_prompt_text(_case_row(ref))
