"""LoRA SFT in TRL on the s28 dataset, the loss on the agent's messages only.

tau-telecom-sft's trainer (`train/sft.py`), trimmed to what runs on Modal:

- Before the model loads, ``train.jsonl`` is checked against the sha256 in its
  ``stats.json``, so the job trains on the dataset that was built and reviewed.
- Before the first step, every row the trainer tokenized is read back and its
  loss mask checked (`mask`): one trained span per agent message, holding that
  message's reasoning, calls and end token, and nothing else. One failed row
  stops the job before it costs anything (tau's check caught 273).
- Every run writes its folder: run.json, metrics.jsonl, mask.json, a
  checkpoint per epoch, and the final adapter with its sha256. A job given a
  run id that already holds a checkpoint resumes from the newest complete one,
  because Modal re-runs a preempted input with the same arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from . import mask as mask_mod


def load_config(path: Path) -> dict[str, Any]:
    """A training config, refused when a required section is missing."""
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    for key in ("label", "base_model", "revision", "dataset", "lora", "sft"):
        if key not in cfg:
            raise ValueError(f"{path}: missing {key!r}")
    return cfg


def new_run_id(label: str, max_steps: int | None) -> str:
    """`<UTC stamp>_train_<label>`, with `-stepN` for a measured step."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_train_{label}" + (f"-step{max_steps}" if max_steps else "")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as indented JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def check_data(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows, once train.jsonl is the file its stats.json names."""
    stats: dict[str, Any] = json.loads((data_dir / "stats.json").read_text())
    raw = (data_dir / "train.jsonl").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != stats["train_sha256"]:
        raise RuntimeError(
            f"{data_dir}/train.jsonl has sha256 {sha[:12]}, not stats.json's "
            f"{stats['train_sha256'][:12]}; upload the built dataset again"
        )
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return rows, stats


def newest_checkpoint(run_dir: Path) -> Path | None:
    """The newest checkpoint written in full (trainer_state.json is written last)."""
    done = [
        p
        for p in (run_dir / "checkpoints").glob("checkpoint-*")
        if (p / "trainer_state.json").exists()
    ]
    return max(done, key=lambda p: int(p.name.rsplit("-", 1)[1]), default=None)


def _sha256_dir(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        data = f.read_bytes()
        h.update(f.relative_to(path).as_posix().encode())
        h.update(data)
        size += len(data)
    return h.hexdigest(), size


def train(
    config_path: Path,
    *,
    data_root: Path,
    runs_root: Path,
    run_id: str,
    code_sha: str,
    max_steps: int | None = None,
    longest: int | None = None,
) -> Path:
    """Train one adapter; return its run folder."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .callbacks import RunFolder

    cfg = load_config(config_path)
    rows, stats = check_data(data_root / cfg["dataset"])
    note = f"all {len(rows)} rows"
    if longest:
        rows = sorted(rows, key=lambda r: -int(r["meta"]["tokens"]))[:longest]
        note = f"the {longest} longest rows, for a measured step"
    run_dir = runs_root / run_id
    resume = newest_checkpoint(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sft_kwargs = dict(cfg["sft"])
    if max_steps:
        sft_kwargs["max_steps"] = max_steps
        sft_kwargs["save_strategy"] = "no"
    if resume is not None:
        meta: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
        meta["status"] = "running"
        meta["resumed_from"] = [*meta.get("resumed_from", []), resume.name]
    else:
        meta = {
            "run_id": run_id,
            "kind": "train",
            "label": cfg["label"],
            "purpose": "measured step" if max_steps else "result",
            "status": "running",
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_sha": code_sha,
            "config": {**cfg, "sft": sft_kwargs},
            "base_model": cfg["base_model"],
            "revision": cfg["revision"],
            "dataset": cfg["dataset"],
            "train_sha256": stats["train_sha256"],
            "system_sha256": stats.get("system_sha256"),
            "tools_sha256": stats.get("tools_sha256"),
            "n_rows": len(rows),
            "rows_selected": note,
            "row_ids": [r["meta"]["report_id"] for r in rows],
            "instance": f"Modal {torch.cuda.get_device_name(0)} "
            f"({os.environ.get('MODAL_REGION', '?')})"
            if torch.cuda.is_available()
            else "cpu",
            "resumed_from": [],
        }
    write_json(run_dir / "run.json", meta)
    print(
        f"run {run_id}: {note}" + (f", resuming from {resume.name}" if resume else "")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], dtype=torch.bfloat16
    )
    dataset = Dataset.from_list(
        [{"messages": r["messages"], "tools": json.dumps(r["tools"])} for r in rows]
    )
    args = SFTConfig(
        output_dir=str(run_dir / "checkpoints"), report_to="none", **sft_kwargs
    )
    step = int(resume.name.rsplit("-", 1)[1]) if resume is not None else 0
    folder = RunFolder(run_dir, resume_step=step)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
        callbacks=[folder],
    )

    # The trainer's own tokenized rows, read back: their labels are exactly what
    # the loss sees (-100 is ignored), so the mask is checked on every row.
    if len(trainer.train_dataset) != len(rows):
        raise RuntimeError(f"TRL kept {len(trainer.train_dataset)} of {len(rows)} rows")
    failed: list[dict[str, Any]] = []
    first: dict[str, Any] = {}
    for i, row in enumerate(rows):
        ex = trainer.train_dataset[i]
        trained = [int(label != -100) for label in ex["labels"]]
        segs = mask_mod.segments(tokenizer, ex["input_ids"], trained)
        chk = mask_mod.check(row, segs)
        if i == 0:
            first = {
                "segs": segs,
                "chk": chk,
                "n": len(ex["input_ids"]),
                "t": sum(trained),
            }
        if not chk["ok"]:
            failed.append(
                {"report_id": row["meta"]["report_id"], "problems": chk["problems"][:3]}
            )
    write_json(
        run_dir / "mask.json",
        {
            "row": rows[0]["meta"]["report_id"],
            "n_tokens": first["n"],
            "n_trained": first["t"],
            "segments": first["segs"],
            "check": first["chk"],
            "rows_checked": len(rows),
            "rows_failed": failed,
        },
    )
    meta["mask_check"] = {
        "rows_checked": len(rows),
        "rows_failed": len(failed),
        "ok": not failed,
    }
    print(f"every row's mask: {len(rows) - len(failed)} of {len(rows)} ok")
    if failed:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        raise RuntimeError(
            f"loss-mask check failed on {len(failed)} of {len(rows)} rows, the first "
            f"{failed[0]['report_id']}: {failed[0]['problems']}"
        )
    write_json(run_dir / "run.json", meta)

    t0 = time.time()
    try:
        out = trainer.train(
            resume_from_checkpoint=str(resume) if resume is not None else None
        )
    except BaseException:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        raise
    adapter = run_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    sha, size = _sha256_dir(adapter)
    epochs = {
        str(
            round(float(json.loads((p / "trainer_state.json").read_text())["epoch"]))
        ): p.name
        for p in sorted((run_dir / "checkpoints").glob("checkpoint-*"))
        if (p / "trainer_state.json").exists()
    }
    meta.update(
        {
            "status": "finished",
            "finished": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": round(time.time() - t0, 1),
            "steps": int(out.global_step),
            "epochs": float(trainer.state.epoch or 0),
            "final_loss": folder.last_loss,
            "train_loss": float(out.training_loss),
            "peak_mem_gb": folder.peak_mem_gb,
            "tokens_trained": folder.tokens,
            "checkpoints_by_epoch": epochs,
            "adapter": {"path": "adapter", "sha256": sha, "bytes": size},
        }
    )
    write_json(run_dir / "run.json", meta)
    print(f"finished {run_id}: {meta['steps']} steps, final loss {meta['final_loss']}")
    return run_dir
