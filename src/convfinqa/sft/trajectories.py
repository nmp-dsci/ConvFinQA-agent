"""llm_react_v1's message histories (s28): fetch, store, read, split into turns.

An open-agent pass logs one artifact, ``trajectories.jsonl.gz``: one line per
conversation holding Pydantic AI's whole message history — the system prompt,
the user messages, DeepSeek's reasoning as thinking parts, every tool call and
result, and the final ``StructuredOutput`` call. That file, not the MLflow
trace (trimmed, prompts by reference), is the training source.

It lives in the tracking server's artifact store, so `fetch` copies it into
``evaluation/sft/trajectories/<run name>.jsonl.gz`` beside the run's CSV, where
it is committed like every other evaluation artifact. It uses one plain GET:
MLflow's own client downloads in chunks, and on the central server those
chunks fail and retry until they give up.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

TRAJ_DIR = EVAL_ROOT / "sft" / "trajectories"
ARTIFACT = "trajectories.jsonl.gz"


def path_for(run_name: str) -> Path:
    """Where a pass's trajectories live in the repo."""
    return TRAJ_DIR / f"{run_name}.jsonl.gz"


def _run_of(run_name: str) -> Any:
    from mlflow import MlflowClient

    from convfinqa.config import settings

    # the central server, named outright: a bare client reads a local ./mlruns
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    runs = client.search_runs(
        [e.experiment_id for e in client.search_experiments()],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=2,
    )
    if len(runs) != 1:
        raise LookupError(
            f"{len(runs)} MLflow runs are named {run_name!r}; expected one"
        )
    return runs[0]


def fetch(run_name: str, *, force: bool = False) -> Path:
    """Copy a pass's trajectories artifact into the repo; return its path."""
    import httpx

    from convfinqa.config import settings

    out = path_for(run_name)
    if out.exists() and not force:
        return out
    run = _run_of(run_name)
    uri = str(run.info.artifact_uri)
    prefix = "mlflow-artifacts:/"
    if not uri.startswith(prefix):
        raise ValueError(f"{run_name}: artifact uri {uri!r} is not on the proxy")
    base = settings.mlflow_tracking_uri.rstrip("/")
    url = f"{base}/api/2.0/mlflow-artifacts/artifacts/{uri[len(prefix) :]}/{ARTIFACT}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with httpx.stream("GET", url, timeout=300) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    with gzip.open(tmp, "rt") as f:  # refuse a truncated or non-gzip body
        n = sum(1 for _ in f)
    if n == 0:
        tmp.unlink()
        raise ValueError(f"{run_name}: the trajectories artifact is empty")
    tmp.replace(out)
    return out


def load(path: Path) -> dict[str, list[dict[str, Any]]]:
    """{report_id: messages} from one trajectories file."""
    out: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["report_id"])] = list(row["messages"])
    return out


def parts(message: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """A message's parts of one kind (`thinking`, `tool-call`, …)."""
    return [p for p in message.get("parts") or [] if p.get("part_kind") == kind]


def turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A conversation's messages cut at each question.

    A turn is ``{"prompt": the user message text, "steps": [every message
    after it, up to the next question]}``. The opening request also carries
    the system prompt as ``instructions``.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        prompts = (
            parts(message, "user-prompt") if message.get("kind") == "request" else []
        )
        if prompts:
            out.append({"prompt": str(prompts[0].get("content") or ""), "steps": []})
        elif out:
            out[-1]["steps"].append(message)
    return out


def responses(turn: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The model's responses within one turn, in order."""
    yield from (m for m in turn["steps"] if m.get("kind") == "response")


def tool_args(part: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments as a dict, however the history stored them."""
    args = part.get("args")
    if isinstance(args, str):
        loaded = json.loads(args) if args.strip() else {}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(args or {})
