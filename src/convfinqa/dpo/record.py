"""The sampling pass as an MLflow run (s33 P3's record), from the committed files only.

One run per tries file in the project's `convfinqa` experiment, kind
`dpo_sample`: the sampler header as params, the stats, coverage and rendered
row card as metrics, and the small JSON files as artifacts. The tries
themselves stay in git (gzipped); the run points at them by path and sha256.

    uv run python -m convfinqa.dpo.record evaluation/dpo/v1/tries8.jsonl
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from convfinqa.dpo.sample import resolve


def _flat(prefix: str, d: dict[str, Any], out: dict[str, float]) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}".replace(":", "_").replace("/", "_of_").replace(" ", "_")
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            _flat(f"{key}.", v, out)


def log(tries: Path) -> str:
    """Log one sampling pass; return the run id."""
    from convfinqa.tracking import mlflow_log

    mlflow_log.assert_reachable()
    mlflow = mlflow_log._mlflow()
    d = tries.parent
    stem = tries.name.removesuffix(".gz").removesuffix(".jsonl")
    runs = [
        json.loads(line)
        for line in (d / f"{stem}_runs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    head = runs[-1]
    stats = json.loads((d / f"stats_{stem}.json").read_text())
    coverage = json.loads((d / f"coverage_{stem}.json").read_text())
    cards = sorted(d.glob(f"rows_{stem}_*_card.json"))
    path = resolve(tries)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()
    metrics: dict[str, float] = {}
    _flat("", {k: stats[k] for k in ("overall", "program", "number")}, metrics)
    _flat("coverage.", {k: coverage[k] for k in ("conversations", "caps")}, metrics)
    metrics["wall_s"] = float(sum(float(r.get("wall_s") or 0) for r in runs))
    for card in cards:
        _flat(
            f"{card.stem.removesuffix('_card')}.", json.loads(card.read_text()), metrics
        )
    with mlflow.start_run(run_name=f"dpo-sample-{stem}") as run:
        mlflow.set_tags(
            {
                "kind": "dpo_sample",
                "project": "convfinqa",
                "git_sha": git_sha,
                "env": "local",
                "billing": "modal",
                "manifest": head["manifest"],
                "dirty_tree": str(bool(dirty)).lower(),
            }
        )
        mlflow.log_params(
            {
                "model": head["model"],
                "harness": head["harness"],
                "tries": head["tries"],
                "system_prompt": head["system_prompt"],
                "sampler": json.dumps(head["sampler"]),
                "topup_from": head.get("topup_from") or "",
                "tries_file": str(path),
                "tries_sha256": sha,
            }
        )
        mlflow.log_metrics(metrics)
        for f in [d / f"stats_{stem}.json", d / f"coverage_{stem}.json", *cards]:
            mlflow.log_artifact(str(f))
        return str(run.info.run_id)


if __name__ == "__main__":
    sys.stdout.write(log(Path(sys.argv[1])) + "\n")
