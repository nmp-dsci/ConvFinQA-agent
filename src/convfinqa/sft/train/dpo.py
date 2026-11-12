"""LoRA DPO on the s35 pairs, sft2-e1 as policy start and frozen reference (s33 P4).

A plain PyTorch loop rather than TRL's DPOTrainer, because the completion here
is a whole agent turn: the student's messages *and* the calculator's returns
between them. Only the student's tokens are scored; a tool return is what the
environment said, and scoring it would push the policy on text it never chose.

- **Data.** `train.jsonl.gz` and `eval.jsonl.gz` (rows from `dpo.render`) are
  checked against the sha256s in `stats.json` before the model loads.
- **Masks.** Each side is rendered with the student's own chat template; the
  student's spans are found by rendering one message at a time (the same cut
  `dpo.render.completion_spans` made locally) and mapped to tokens by offset.
  Every row's mask is checked with `mask.check` before the first step: one
  trained span per assistant message, holding its text and calls, nothing of
  a tool return, the prompt or a header. One failure stops the job.
- **Reference.** The policy starts as sft2-e1's adapter, so the reference is
  the same weights: its log-probs are computed once, before training, with
  dropout off, and cached in the run folder.
- **Step 0.** The policy's log-probs on the first rows, recomputed with the
  trainable adapter before any update, must equal the reference's (|Δ| below
  `STEP0_TOL` nats per row): proof the reference is the policy's start.
- **Loss.** DPO's sigmoid loss with `beta`, plus `nll_alpha` × the chosen's
  length-normalised NLL (Pang et al.'s RPO term), which keeps the chosen likely
  while the margin grows.
- **Record.** run.json, metrics.jsonl (per step: loss, dpo, nll, reward
  accuracy, margin, chosen/rejected Δ log-prob), mask.json, eval.jsonl (the
  eval pairs at step 0 and at each checkpoint), and an adapter per checkpoint
  with a `trainer_state.json`, which is what the student server waits for.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from . import mask as mask_mod

HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>"
STEP0_TOL = 0.05  # nats per row; bf16 recomputation of the same weights


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as indented JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    """A DPO config, refused when a required section is missing."""
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    for key in ("label", "base_model", "revision", "dataset", "init_adapter", "dpo"):
        if key not in cfg:
            raise ValueError(f"{path}: missing {key!r}")
    return cfg


def check_data(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The train and eval rows, once each file is the one `stats.json` names."""
    stats: dict[str, Any] = json.loads((data_dir / "stats.json").read_text())
    out = []
    for name in ("train", "eval"):
        raw = (data_dir / f"{name}.jsonl.gz").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != stats[f"{name}_sha256"]:
            raise RuntimeError(f"{name}.jsonl.gz has sha256 {sha[:12]}, not stats'")
        text = gzip.decompress(raw).decode()
        out.append([json.loads(line) for line in text.splitlines() if line.strip()])
    return out[0], out[1], stats


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: Any) -> str:
    text: str = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    return text


def side_tokens(tokenizer: Any, row: dict[str, Any], side: str) -> dict[str, Any]:
    """One side tokenized: input ids, the student-token mask, where the completion starts."""
    prompt, comp, tools = row["prompt"], row[side], row["tools"]
    base = _render(tokenizer, prompt, tools)
    spans: list[tuple[int, int]] = []
    done = base
    n = len(comp)
    for i in range(n):
        if i + 1 < n and comp[i]["role"] == comp[i + 1]["role"] == "tool":
            continue
        upto = _render(tokenizer, prompt + comp[: i + 1], tools)
        if not upto.startswith(done):
            raise ValueError("the template does not render the completion as a suffix")
        if comp[i]["role"] == "assistant":
            piece = upto[len(done) :]
            if not piece.startswith(HEADER):
                raise ValueError("an assistant piece does not open with its header")
            end = len(done) + piece.rstrip("\n").rfind(END) + len(END)
            spans.append((len(done) + len(HEADER), end))
        done = upto
    full = done
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    mask = [0] * len(ids)
    prompt_chars = len(base)
    first = None
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if first is None and a >= prompt_chars:
            first = t
        for s, e in spans:
            if a >= s and b <= e and b > a:
                mask[t] = 1
            elif a < s < b or a < e < b:
                raise ValueError("a token straddles a student span's edge")
    if first is None:
        raise ValueError("no completion tokens")
    return {"input_ids": ids, "mask": mask, "start": first}


def check_mask(
    tokenizer: Any, row: dict[str, Any], tok: dict[str, Any], side: str
) -> dict[str, Any]:
    """`mask.check` on one side: the trained spans are exactly its assistant messages."""
    segs = mask_mod.segments(tokenizer, tok["input_ids"], tok["mask"])
    return mask_mod.check({"messages": row[side]}, segs)


def seq_logp(model: Any, tok: dict[str, Any], device: Any) -> tuple[Any, int]:
    """Summed log-prob of the student's tokens on one side, and how many there are."""
    import torch

    ids = torch.tensor([tok["input_ids"]], device=device)
    start = tok["start"]
    keep = len(tok["input_ids"]) - start + 1
    out = model(input_ids=ids, use_cache=False, logits_to_keep=keep)
    logits = out.logits[0, :-1].float()  # predicts tokens start .. end
    targets = ids[0, start:]
    m = torch.tensor(tok["mask"][start:], device=device, dtype=torch.float32)
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
    return (logp * m).sum(), int(m.sum().item())


def _no_dropout(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _evaluate(
    model: Any,
    toks: list[dict[str, Any]],
    refs: list[dict[str, float]],
    beta: float,
    device: Any,
) -> dict[str, Any]:
    import torch

    margins, dc, dr = [], [], []
    with torch.no_grad():
        for t, r in zip(toks, refs, strict=True):
            c, _ = seq_logp(model, t["chosen"], device)
            j, _ = seq_logp(model, t["rejected"], device)
            dc.append(float(c) - r["chosen"])
            dr.append(float(j) - r["rejected"])
            margins.append(beta * (dc[-1] - dr[-1]))
    n = max(1, len(margins))
    return {
        "reward_accuracy": round(sum(m > 0 for m in margins) / n, 4),
        "margin": round(sum(margins) / n, 4),
        "chosen_dlogp": round(sum(dc) / n, 3),
        "rejected_dlogp": round(sum(dr) / n, 3),
        "n": len(margins),
    }


def train(
    config_path: Path,
    *,
    data_root: Path,
    runs_root: Path,
    run_id: str,
    code_sha: str,
    max_steps: int | None = None,
) -> Path:
    """Train one DPO adapter; return its run folder."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(config_path)
    d = cfg["dpo"]
    train_rows, eval_rows, stats = check_data(data_root / cfg["dataset"])
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if max_steps:
        train_rows = train_rows[: max_steps * int(d["gradient_accumulation_steps"])]
        eval_rows = eval_rows[:8]
    meta: dict[str, Any] = {
        "run_id": run_id,
        "kind": "train_dpo",
        "label": cfg["label"],
        "purpose": "measured step" if max_steps else "result",
        "status": "running",
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": code_sha,
        "config": cfg,
        "dataset": cfg["dataset"],
        "train_sha256": stats["train_sha256"],
        "eval_sha256": stats["eval_sha256"],
        "init_adapter": cfg["init_adapter"],
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "instance": f"Modal {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "cpu",
    }
    write_json(run_dir / "run.json", meta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["revision"]
    )
    toks: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    failed: list[dict[str, Any]] = []
    first: dict[str, Any] = {}
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        for row in rows:
            pair = {s: side_tokens(tokenizer, row, s) for s in ("chosen", "rejected")}
            for s in ("chosen", "rejected"):
                chk = check_mask(tokenizer, row, pair[s], s)
                if not first:
                    first = {
                        "row": row["meta"]["question_id"],
                        "side": s,
                        "segments": mask_mod.segments(
                            tokenizer, pair[s]["input_ids"], pair[s]["mask"]
                        )[-12:],
                        "check": chk,
                    }
                if not chk["ok"]:
                    failed.append(
                        {
                            "row": row["meta"]["question_id"],
                            "side": s,
                            "problems": chk["problems"][:3],
                        }
                    )
            toks[split].append(pair)
    n_checked = 2 * (len(train_rows) + len(eval_rows))
    write_json(
        run_dir / "mask.json", {**first, "sides_checked": n_checked, "failed": failed}
    )
    meta["mask_check"] = {
        "sides_checked": n_checked,
        "failed": len(failed),
        "ok": not failed,
    }
    write_json(run_dir / "run.json", meta)
    print(f"every side's mask: {n_checked - len(failed)} of {n_checked} ok")
    if failed:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        raise RuntimeError(f"mask check failed on {len(failed)} sides: {failed[0]}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    init = runs_root / cfg["init_adapter"]
    model = PeftModel.from_pretrained(model, str(init), is_trainable=True)
    _no_dropout(model)
    model.config.use_cache = False

    # the reference: sft2-e1 as it loads, once, cached so a rerun skips it
    ref_path = run_dir / "reference_logps.json"
    if ref_path.exists():
        refs = json.loads(ref_path.read_text())
    else:
        t0 = time.time()
        model.eval()
        refs = {"train": [], "eval": []}
        with torch.no_grad():
            for split in ("train", "eval"):
                for pair in toks[split]:
                    c, nc = seq_logp(model, pair["chosen"], device)
                    r, nr = seq_logp(model, pair["rejected"], device)
                    refs[split].append(
                        {
                            "chosen": float(c),
                            "rejected": float(r),
                            "n_chosen": nc,
                            "n_rejected": nr,
                        }
                    )
        refs["seconds"] = round(time.time() - t0, 1)
        write_json(ref_path, refs)
    meta["reference_seconds"] = refs.get("seconds")
    print(
        f"reference log-probs: {len(refs['train'])} train, {len(refs['eval'])} eval pairs"
    )

    # step 0: the trainable policy, before any update, is the reference
    model.train()
    _no_dropout(model)
    step0 = []
    with torch.no_grad():
        for pair, ref in list(zip(toks["train"], refs["train"], strict=True))[:4]:
            c, _ = seq_logp(model, pair["chosen"], device)
            step0.append(abs(float(c) - ref["chosen"]))
    meta["step0_max_abs_dlogp"] = round(max(step0), 4)
    write_json(run_dir / "run.json", meta)
    print(f"step 0: max |policy - reference| = {max(step0):.4f} nats")
    if max(step0) > STEP0_TOL:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        raise RuntimeError(
            f"step 0: policy and reference differ by {max(step0):.4f} nats"
        )

    beta, alpha = float(d["beta"]), float(d["nll_alpha"])
    accum = int(d["gradient_accumulation_steps"])
    epochs = int(d.get("num_train_epochs", 1))
    order = list(range(len(toks["train"])))
    steps_per_epoch = len(order) // accum
    total = steps_per_epoch * epochs
    saves = sorted({max(1, round(total * f)) for f in d.get("save_fractions", [1.0])})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(d["learning_rate"]), weight_decay=0.0)
    warm = max(1, round(total * float(d.get("warmup_ratio", 0.05))))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        frac = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * frac))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(int(d.get("seed", 0)))
    metrics_path = run_dir / "metrics.jsonl"
    eval_path = run_dir / "eval.jsonl"
    ev = _evaluate(model, toks["eval"], refs["eval"], beta, device)
    with eval_path.open("w") as fh:
        fh.write(json.dumps({"step": 0, **ev}) + "\n")
    print(f"eval at step 0: {ev}")

    t0 = time.time()
    step = 0
    checkpoints: dict[str, str] = {}
    for epoch in range(epochs):
        rng.shuffle(order)
        for s in range(steps_per_epoch):
            batch = order[s * accum : (s + 1) * accum]
            agg = {
                "loss": 0.0,
                "dpo": 0.0,
                "nll": 0.0,
                "acc": 0.0,
                "margin": 0.0,
                "dc": 0.0,
                "dr": 0.0,
            }
            for i in batch:
                pair, ref = toks["train"][i], refs["train"][i]
                c, nc = seq_logp(model, pair["chosen"], device)
                r, _ = seq_logp(model, pair["rejected"], device)
                dc = c - ref["chosen"]
                dr = r - ref["rejected"]
                z = beta * (dc - dr)
                dpo = -torch.nn.functional.logsigmoid(z)
                nll = -c / max(1, nc)
                loss = (dpo + alpha * nll) / len(batch)
                loss.backward()
                agg["loss"] += float(loss)
                agg["dpo"] += float(dpo) / len(batch)
                agg["nll"] += float(nll) / len(batch)
                agg["acc"] += float(z > 0) / len(batch)
                agg["margin"] += float(z) / len(batch)
                agg["dc"] += float(dc) / len(batch)
                agg["dr"] += float(dr) / len(batch)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    params, float(d.get("max_grad_norm", 1.0))
                )
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            row = {
                "step": step,
                "epoch": round(epoch + (s + 1) / steps_per_epoch, 3),
                **{k: round(v, 5) for k, v in agg.items()},
                "grad_norm": round(grad_norm, 4),
                "learning_rate": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None,
            }
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row))
            if step in saves or (max_steps and step >= max_steps):
                ck = run_dir / "checkpoints" / f"checkpoint-{step}"
                model.save_pretrained(str(ck))
                ev = _evaluate(model, toks["eval"], refs["eval"], beta, device)
                with eval_path.open("a") as fh:
                    fh.write(json.dumps({"step": step, **ev}) + "\n")
                print(f"eval at step {step}: {ev}")
                write_json(
                    ck / "trainer_state.json",
                    {"global_step": step, "epoch": row["epoch"], "eval": ev},
                )
                checkpoints[f"{row['epoch']:.2f}"] = ck.name
                model.train()
                _no_dropout(model)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    meta.update(
        {
            "status": "finished",
            "finished": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": round(time.time() - t0, 1),
            "steps": step,
            "seconds_per_pair": round((time.time() - t0) / max(1, step * accum), 2),
            "checkpoints_by_epoch": checkpoints,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else None,
            "pid": os.getpid(),
        }
    )
    write_json(run_dir / "run.json", meta)
    print(f"finished {run_id}: {step} steps")
    return run_dir
