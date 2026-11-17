"""The pilot's dataset: a seeded draw of rendered train pairs, plus every eval pair.

Written to `evaluation/dpo/v1/<name>/` as `train.jsonl.gz`, `eval.jsonl.gz` and a
`stats.json` holding each file's sha256, which the trainer checks before it
loads the model; `upload` puts the folder on Modal's data volume.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT
from convfinqa.dpo.sample import open_text

ROOT = EVAL_ROOT / "dpo" / "v1"


def write(rows_file: Path, *, n: int, seed: int, name: str) -> dict[str, Any]:
    """Draw `n` train rows (seeded) and keep every eval row; return the stats."""
    with open_text(rows_file) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    train = [r for r in rows if r["meta"]["split"] == "train"]
    evals = [r for r in rows if r["meta"]["split"] == "eval"]
    picked = sorted(random.Random(seed).sample(range(len(train)), min(n, len(train))))
    train = [train[i] for i in picked]
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "name": name,
        "source": str(rows_file),
        "seed": seed,
        "n_train": len(train),
        "n_eval": len(evals),
        "train_questions": len({r["meta"]["question_id"] for r in train}),
        "train_kind": dict(Counter(r["meta"]["kind"] for r in train)),
        "train_position": dict(Counter(f"q{r['meta']['k']}" for r in train)),
    }
    for split, part in (("train", train), ("eval", evals)):
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        raw = gzip.compress(body.encode(), mtime=0)
        (out / f"{split}.jsonl.gz").write_bytes(raw)
        stats[f"{split}_sha256"] = hashlib.sha256(raw).hexdigest()
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def upload(name: str) -> int:
    """Put the dataset folder on Modal's data volume."""
    from convfinqa.sft.cloud.modal_app import DATA_VOLUME

    for f in ("train.jsonl.gz", "eval.jsonl.gz", "stats.json"):
        code = subprocess.call(
            [sys.executable, "-m", "modal", "volume", "put", "--force", DATA_VOLUME,
             str(ROOT / name / f), f"/{name}/{f}"]
        )  # fmt: skip
        if code:
            return code
    return 0
