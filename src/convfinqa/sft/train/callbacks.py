"""Every logged training step to the run folder's metrics.jsonl (from tau-telecom-sft).

The run folder is the record. The job runs on Modal, which cannot reach the
central MLflow server, so nothing is logged there from here; `convfinqa-sft
sync` brings the folder home.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

STEP_KEYS = ("loss", "grad_norm", "learning_rate", "epoch")


def _mem_gb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(float(torch.cuda.max_memory_allocated()) / 2**30, 2)
    return None


class RunFolder(TrainerCallback):  # type: ignore[misc]
    """Append one line per logged step; on a resume, drop the steps that will rerun."""

    def __init__(self, run_dir: Path, resume_step: int = 0) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.t0 = self.last_t = time.time()
        self.last_tokens = 0
        self.tokens = 0
        self.last_loss: float | None = None
        self.peak_mem_gb: float | None = None
        if self.path.exists():
            kept = [
                line
                for line in self.path.read_text().splitlines()
                if line.strip() and json.loads(line)["step"] <= resume_step
            ]
            self.path.write_text("".join(f"{line}\n" for line in kept))

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Start the clock at the first step, not at model load."""
        self.t0 = self.last_t = time.time()
        self.last_tokens = int(state.num_input_tokens_seen or 0)

    def on_log(
        self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
    ) -> None:
        """One line per logged step, with throughput and peak memory."""
        logs = dict(logs or {})
        if "loss" not in logs:
            return
        now = time.time()
        tokens = int(logs.get("num_tokens") or state.num_input_tokens_seen or 0)
        rate = (
            (tokens - self.last_tokens) / max(now - self.last_t, 1e-6)
            if tokens
            else None
        )
        mem = _mem_gb()
        if mem is not None:
            self.peak_mem_gb = max(mem, self.peak_mem_gb or 0.0)
        row = {
            "step": int(state.global_step),
            **{k: logs.get(k) for k in STEP_KEYS},
            "tokens": tokens or None,
            "tokens_per_s": round(rate, 1) if rate else None,
            "mem_gb": mem,
            "elapsed_s": round(now - self.t0, 1),
        }
        self.last_t, self.last_tokens, self.tokens = now, tokens, tokens
        self.last_loss = logs.get("loss")
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
