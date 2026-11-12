"""The fine-tuning record (s32): every student pass and training run, as one file.

`evaluation/fine_tuning.json` is what Optimise › Fine-tuning draws. It is built
from committed files only, so it rebuilds byte for byte on any checkout, and
`tests/test_fine_tuning.py` holds the committed copy to a rebuild:

- the student's scoring passes and the teacher's (llm_react_v1) test pass,
  `evaluation/predictions/evalloop/open-evalloop-*.csv`;
- each training run's record, `runs/sft/<run id>/run.json` and `metrics.jsonl`;
- each dataset's card, `evaluation/sft/<version>/stats.json`;
- `evaluation/splits/sft_v1.json`, which says which conversations are val and
  which are test.

A pass is placed by its report ids, not its name. The val passes ran as
``--split train`` (val is cut from the pool's train side), so their names and
MLflow params say "train100"; only the ids say val. A pass with an unscored
turn is listed as incomplete and is never drawn or compared.

A round's best checkpoint is the one with the highest val accuracy in its
sweep (its checkpoints scored on one harness at the default temperature), which
is how sft1 e2 and sft2 e1 were picked in s28 and s29. Every comparison is
paired on the same questions with the gate's own statistics
(`tracking.comparator`): the one-sided cluster-corrected McNemar and a
conversation-bootstrap interval.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import EVAL_ROOT
from convfinqa.llm import STUDENT_SAMPLING

REPO = EVAL_ROOT.parent
PROGRESS_PATH = EVAL_ROOT / "fine_tuning.json"
PASSES_DIR = EVAL_ROOT / "predictions" / "evalloop"
RUNS_DIR = REPO / "runs" / "sft"
# s33: a DPO round shares the runs folder but has its own data and record; it is
# not on the SFT progression yet, so neither its rounds nor its inputs are read
SFT_KINDS = ("train",)


def _sft_runs(name: str) -> list[Path]:
    """`runs/sft/*/<name>` for the SFT training runs only (no DPO rounds)."""
    return [
        p
        for p in sorted(RUNS_DIR.glob(f"*/{name}"))
        if json.loads((p.parent / "run.json").read_text()).get("kind", "train")
        in SFT_KINDS
    ]


DATA_DIR = EVAL_ROOT / "sft"
MANIFEST_PATH = EVAL_ROOT / "splits" / "sft_v1.json"

BASE = "student-qwen3-4b"
TEACHER = ("deepseek-flash", "high")
TEACHER_NAME = "llm_react_v1"
DEFAULT_TEMPERATURE = float(STUDENT_SAMPLING["temperature"])
HARNESSES = {
    "pydantic_ai": {
        "label": "plain",
        "about": "llm_react_v1's harness and the teacher's: Pydantic AI with the "
        "champion's tools and output schema.",
    },
    "pydantic_ai_checked": {
        "label": "checked",
        "about": "The student's harness from s29 (H1–H3): a reply goes back for "
        "another try when its answer is not one of its own calculator results "
        "or does not follow from its own program.",
    },
}
_SLUG_HARNESS = {"pyd": "pydantic_ai", "pydchk": "pydantic_ai_checked"}
_NAME = re.compile(
    r"^open-evalloop-[a-z]+\d+-(?P<version>[^·]+)·[^-]+-(?P<slug>pydchk|pyd)-"
    r"(?P<model>.+?)(?:-t(?P<temp>[0-9p]+))?-(?P<effort>none|low|high|max)-"
    r"(?P<stamp>\d{8}_\d{6})$"
)
_STUDENT = re.compile(
    rf"^{re.escape(BASE)}(?:-(?P<round>sft\d+)-e(?P<epoch>[0-9p]+))?$"
)
# a checkpoint's served name rounds its epoch (sft2's 0.51 is served as e0p5)
_EPOCH_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------


def _number(slug: str) -> float:
    return float(slug.replace("p", "."))


def _truthy(column: pd.Series) -> pd.Series:
    return column.astype(str).str.lower().eq("true")


def _slice(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    k = int(frame["correct"].sum())
    return {"n": n, "correct": k, "accuracy": round(k / n, 6) if n else None}


def _splits() -> dict[str, set[str]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {name: set(manifest["splits"][name]) for name in ("val", "test")}


def _frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "unscored" not in frame.columns:
        frame["unscored"] = False
    frame["unscored"] = _truthy(frame["unscored"])
    frame["correct"] = _truthy(frame["correct"])
    return frame


def read_pass(path: Path, splits: dict[str, set[str]]) -> dict[str, Any] | None:
    """One committed pass as a record, or None when it is not a fine-tuning pass."""
    m = _NAME.match(path.stem)
    if not m:
        return None
    model, effort = m["model"], m["effort"]
    student = _STUDENT.match(model)
    if student:
        role = "checkpoint" if student["round"] else "base"
    elif (model, effort) == TEACHER:
        role = "teacher"
    else:
        return None
    frame = _frame(path)
    ids = set(frame["report_id"])
    split = next((name for name, members in splits.items() if ids == members), None)
    record: dict[str, Any] = {
        "id": path.stem,
        "csv": str(path.relative_to(REPO)),
        "run_id": next(iter(frame["run_id"].dropna().astype(str)), None),
        "date": f"{m['stamp'][:4]}-{m['stamp'][4:6]}-{m['stamp'][6:8]}",
        "split": split,
        "role": role,
        "model": model,
        "effort": effort,
        "harness": _SLUG_HARNESS[m["slug"]],
        "temperature": _number(m["temp"]) if m["temp"] else None,
        "round": student["round"] if student else None,
        "epoch": _number(student["epoch"]) if student and student["epoch"] else None,
        "n_conversations": len(ids),
        "n_unscored": int(frame["unscored"].sum()),
    }
    if role == "teacher":
        record["temperature"] = None
    elif record["temperature"] is None:
        record["temperature"] = DEFAULT_TEMPERATURE
    scored = frame[~frame["unscored"]]
    turn = scored["gold_turn_type"].astype(str).str.lower()
    record |= {
        "complete": record["n_unscored"] == 0 and len(scored) > 0,
        **_slice(scored),
        "program": _slice(scored[turn == "program"]),
        "number": _slice(scored[turn == "number"]),
    }
    return record


def _scored(record: dict[str, Any]) -> pd.DataFrame:
    frame = _frame(REPO / record["csv"])
    return frame[~frame["unscored"]]


def paired(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """`cand` against `base` on their shared questions, with the gate's statistics."""
    from convfinqa.tracking.comparator import cluster_bootstrap_ci, compare_frames

    a, b = _scored(base), _scored(cand)
    result = compare_frames(
        a, b, baseline_version=base["id"], candidate_version=cand["id"]
    )
    ci = cluster_bootstrap_ci(a, b)
    return {
        "from": base["id"],
        "to": cand["id"],
        "n": result.n_compared,
        "delta_pp": round(
            100 * (result.candidate_accuracy - result.baseline_accuracy), 2
        ),
        "fixed": len(result.improvements),
        "broken": len(result.regressions),
        "p_one_sided": round(result.cluster_p_one_sided, 4),
        "ci_pp": [round(100 * ci["lo"], 2), round(100 * ci["hi"], 2)],
        "same_harness": base["harness"] == cand["harness"],
    }


# ---------------------------------------------------------------------------
# training runs
# ---------------------------------------------------------------------------


def _loss_curve(run_dir: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "loss" in row:
            out.append(
                {
                    "step": int(row["step"]),
                    "epoch": round(float(row["epoch"]), 4),
                    "loss": round(float(row["loss"]), 4),
                    "learning_rate": float(f"{float(row['learning_rate']):.3g}"),
                }
            )
    return out


def _checkpoints(meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for epoch, name in meta["checkpoints_by_epoch"].items():
        out.append({"epoch": float(epoch), "step": int(name.split("-")[-1])})
    return sorted(out, key=lambda c: c["epoch"])


def _key(epoch: float) -> str:
    return "e" + f"{epoch:g}".replace(".", "p")


def _find(
    passes: list[dict[str, Any]],
    *,
    model: str,
    split: str,
    harness: str | None = None,
) -> dict[str, Any] | None:
    """The complete pass of `model` on `split` at the default temperature, newest first."""
    hits = [
        p
        for p in passes
        if p["complete"]
        and p["model"] == model
        and p["split"] == split
        and p["temperature"] in (None, DEFAULT_TEMPERATURE)
        and (harness is None or p["harness"] == harness)
    ]
    return max(hits, key=lambda p: p["id"]) if hits else None


def _round(
    run_dir: Path,
    meta: dict[str, Any],
    passes: list[dict[str, Any]],
    parents: dict[str, Any],
) -> dict[str, Any]:
    label = meta["label"]
    cfg = meta["config"]
    stats = json.loads((DATA_DIR / meta["dataset"] / "stats.json").read_text())
    mine = [p for p in passes if p["complete"] and p["round"] == label]

    # the sweep: the harness that scored the most checkpoints on val by default
    by_harness: dict[str, set[float]] = {}
    for p in mine:
        if p["split"] == "val" and p["temperature"] == DEFAULT_TEMPERATURE:
            by_harness.setdefault(p["harness"], set()).add(p["epoch"])
    harness = (
        max(by_harness, key=lambda h: (len(by_harness[h]), h)) if by_harness else None
    )

    if meta.get("init_adapter"):
        parent_run, _, parent_ckpt = meta["init_adapter"].partition("/checkpoints/")
        parent = parents[parent_run]
        step = int(parent_ckpt.split("-")[-1])
        epoch = next(c["epoch"] for c in parent["checkpoints"] if c["step"] == step)
        start = {
            "label": f"{parent['key']} {_key(epoch)}",
            "round": parent["key"],
            "model": f"{BASE}-{parent['key']}-{_key(epoch)}",
        }
    else:
        start = {"label": "Base", "round": None, "model": BASE}
    start_val = _find(
        passes, model=start["model"], split="val", harness=harness
    ) or _find(passes, model=start["model"], split="val")
    start["val_pass"] = start_val["id"] if start_val else None

    checkpoints = []
    for c in _checkpoints(meta):
        key = _key(round(c["epoch"] * 2) / 2)
        val = next(
            (
                p
                for p in mine
                if p["split"] == "val"
                and p["harness"] == harness
                and p["temperature"] == DEFAULT_TEMPERATURE
                and abs(p["epoch"] - c["epoch"]) < _EPOCH_TOLERANCE
            ),
            None,
        )
        test = next(
            (
                p
                for p in sorted(mine, key=lambda p: p["id"], reverse=True)
                if p["split"] == "test"
                and p["harness"] == harness
                and abs(p["epoch"] - c["epoch"]) < _EPOCH_TOLERANCE
            ),
            None,
        )
        checkpoints.append(
            {
                "key": key,
                "epoch": c["epoch"],
                "step": c["step"],
                "model": f"{BASE}-{label}-{key}",
                "val_pass": val["id"] if val else None,
                "test_pass": test["id"] if test else None,
                "vs_start": paired(start_val, val) if start_val and val else None,
            }
        )
    accuracy = {p["id"]: p["accuracy"] for p in mine}
    scored = [c for c in checkpoints if c["val_pass"]]
    best = (
        max(scored, key=lambda c: (accuracy[c["val_pass"]], -c["epoch"]))
        if scored
        else None
    )
    in_sweep = {c["val_pass"] for c in checkpoints} | {
        c["test_pass"] for c in checkpoints
    }
    rescores = [
        p["id"] for p in sorted(mine, key=lambda p: p["id"]) if p["id"] not in in_sweep
    ]

    lora = cfg["lora"]
    sft = cfg["sft"]
    return {
        "key": label,
        "label": f"Round {label.removeprefix('sft')}",
        "run_id": meta["run_id"],
        "record": str((run_dir / "run.json").relative_to(REPO)),
        "started": meta["created"],
        "finished": meta["finished"],
        "instance": meta["instance"],
        "code_sha": meta["code_sha"],
        "start": start,
        "dataset": {
            "version": meta["dataset"],
            "card": str((DATA_DIR / meta["dataset"] / "card.md").relative_to(REPO)),
            "manifest": stats.get("manifest", "sft_v1"),
            "n_rows": stats["n_rows"],
            "n_prefix_rows": stats.get("n_prefix_rows", 0),
            "n_turns": stats["n_turns"],
            "n_type2": stats["n_type2"],
            "n_dropped": stats["n_dropped"],
            "tokens_trained": meta["tokens_trained"],
        },
        "settings": {
            "learning_rate": sft["learning_rate"],
            "schedule": sft["lr_scheduler_type"],
            "epochs": sft["num_train_epochs"],
            "effective_batch": sft["per_device_train_batch_size"]
            * sft["gradient_accumulation_steps"],
            "max_length": sft["max_length"],
            "lora_r": lora["r"],
            "lora_alpha": lora["lora_alpha"],
            "lora_dropout": lora["lora_dropout"],
            "target_modules": lora["target_modules"],
            "save": sft["save_strategy"]
            + (
                f" every {sft['save_steps']} of the run"
                if sft.get("save_steps")
                else ""
            ),
        },
        "steps": meta["steps"],
        "train_loss": round(meta["train_loss"], 4),
        "final_loss": round(meta["final_loss"], 4),
        "duration_s": round(meta["duration_s"]),
        "cost_usd": meta["cost_usd"],
        "cost_basis": meta["cost_basis"],
        "loss": _loss_curve(run_dir),
        "sweep_harness": harness,
        "checkpoints": checkpoints,
        "best": best["key"] if best else None,
        "rescores": rescores,
    }


def _rounds(
    passes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rounds: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    parents: dict[str, Any] = {}
    metas = []
    for run_json in _sft_runs("run.json"):
        meta = json.loads(run_json.read_text())
        if meta.get("purpose") != "result" or meta.get("status") != "finished":
            skipped.append(
                {
                    "run_id": meta["run_id"],
                    "reason": f"purpose {meta.get('purpose')!r}, status "
                    f"{meta.get('status')!r}: not a training result",
                }
            )
            continue
        metas.append((run_json.parent, meta))
    for run_dir, meta in sorted(metas, key=lambda x: x[1]["created"]):
        r = _round(run_dir, meta, passes, parents)
        parents[meta["run_id"]] = r
        rounds.append(r)
    return rounds, skipped


# ---------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------


def _progression(
    rounds: list[dict[str, Any]], passes: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    """Base, then each round's best, in order; a harness change is a stage of its own.

    sft2 was scored on the checked harness and sft1 on the plain one, so the
    step from one to the other is two changes. sft1's best was re-scored on the
    checked harness, and that pass stands between them: the step into it is the
    harness alone, the step out of it is the training alone.
    """
    first = rounds[0]["sweep_harness"] if rounds else None
    stages: list[dict[str, Any]] = [
        {
            "key": "base",
            "label": "Base",
            "kind": "base",
            "model": BASE,
            "harness": first,
            "pass": (_find(passes, model=BASE, split=split, harness=first) or {}).get(
                "id"
            ),
        }
    ]
    for r in rounds:
        harness = r["sweep_harness"]
        prev = stages[-1]
        if harness != prev["harness"]:
            stages.append(
                {
                    "key": f"{prev['key']}-{HARNESSES[harness]['label']}",
                    "label": f"{prev['label']}, re-scored on the {HARNESSES[harness]['label']} harness",
                    "kind": "harness",
                    "model": prev["model"],
                    "harness": harness,
                    "pass": (
                        _find(passes, model=prev["model"], split=split, harness=harness)
                        or {}
                    ).get("id"),
                }
            )
        model = f"{BASE}-{r['key']}-{r['best']}" if r["best"] else None
        stages.append(
            {
                "key": r["key"],
                "label": f"{r['label']} best, {r['best']}" if r["best"] else r["label"],
                "kind": "round",
                "round": r["key"],
                "model": model,
                "harness": harness,
                "pass": (
                    (
                        _find(passes, model=model, split=split, harness=harness) or {}
                    ).get("id")
                    if model
                    else None
                ),
            }
        )
    by_id = {p["id"]: p for p in passes}
    last: dict[str, Any] | None = None
    for stage in stages:
        here = by_id.get(stage["pass"] or "")
        stage["vs_prev"] = paired(last, here) if last and here else None
        if here:
            last = here
    return stages


def _inputs(passes: list[dict[str, Any]]) -> list[dict[str, str]]:
    paths = [MANIFEST_PATH, *_sft_runs("run.json")]
    paths += _sft_runs("metrics.jsonl")
    paths += sorted(DATA_DIR.glob("v*/stats.json"))
    paths += [REPO / p["csv"] for p in passes]
    return [
        {
            "path": str(p.relative_to(REPO)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        }
        for p in paths
    ]


def build() -> dict[str, Any]:
    """The whole record, from committed files only."""
    splits = _splits()
    passes: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in sorted(PASSES_DIR.glob("open-evalloop-*.csv")):
        record = read_pass(path, splits)
        if record is None:
            continue
        if record["role"] == "teacher" and record["split"] != "test":
            continue  # the teacher's collection passes, not its score
        if record["split"] is None:
            excluded.append(
                {
                    "id": record["id"],
                    "reason": "its conversations are neither val nor test (a smoke pass)",
                }
            )
        elif not record["complete"]:
            excluded.append(
                {
                    "id": record["id"],
                    "reason": f"{record['n_unscored']} of its turns were never scored: an incomplete pass",
                }
            )
        passes.append(record)
    usable = [p for p in passes if p["split"] is not None]
    rounds, skipped_runs = _rounds(usable)

    teacher = _find(usable, model=TEACHER[0], split="test")
    progression = {
        split: _progression(rounds, usable, split) for split in ("val", "test")
    }
    by_id = {p["id"]: p for p in usable}
    final = next(
        (by_id[s["pass"]] for s in reversed(progression["test"]) if s["pass"]), None
    )
    base_test = by_id.get(progression["test"][0]["pass"] or "")
    reference = None
    if teacher:
        reference = {
            "name": TEACHER_NAME,
            "model": TEACHER[0],
            "effort": TEACHER[1],
            "pass": teacher["id"],
            "vs_final": paired(final, teacher) if final else None,
            "gap_closed": (
                round(
                    (final["accuracy"] - base_test["accuracy"])
                    / (teacher["accuracy"] - base_test["accuracy"]),
                    4,
                )
                if final and base_test and teacher["accuracy"] != base_test["accuracy"]
                else None
            ),
        }
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        "schema": 1,
        "student": "Qwen/Qwen3-4B-Instruct-2507",
        "splits": {
            "val": {
                "manifest": manifest["name"],
                "n_conversations": len(splits["val"]),
                "role": "picks each round's checkpoint; never trained on",
            },
            "test": {
                "manifest": manifest["gate_manifest"],
                "n_conversations": len(splits["test"]),
                "role": "the student's holdout (test 100): opened only for a picked "
                "checkpoint, on the owner's word",
            },
        },
        "harnesses": HARNESSES,
        "default_temperature": DEFAULT_TEMPERATURE,
        "rounds": rounds,
        "progression": progression,
        "teacher": reference,
        "passes": passes,
        "excluded": excluded,
        "skipped_runs": skipped_runs,
        "inputs": _inputs(passes),
    }


def render(record: dict[str, Any]) -> str:
    """The record as committed: one-space indent, a trailing newline."""
    return json.dumps(record, indent=1, ensure_ascii=False) + "\n"


def write(path: Path = PROGRESS_PATH) -> Path:
    """Rebuild `evaluation/fine_tuning.json`."""
    path.write_text(render(build()))
    return path


def load(path: Path = PROGRESS_PATH) -> dict[str, Any]:
    """The committed record."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ---------------------------------------------------------------------------
# the cases, for the drill-down
# ---------------------------------------------------------------------------


def questions(
    pass_ids: list[str], record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every question of the given passes, side by side, with gold.

    Only passes the record lists as complete fine-tuning passes are opened, and
    they must share a split: the ids name files, so the record is the allow-list.
    """
    record = record if record is not None else load()
    known = {p["id"]: p for p in record["passes"] if p["complete"] and p["split"]}
    unknown = [i for i in pass_ids if i not in known]
    if unknown:
        raise KeyError(f"not a complete fine-tuning pass: {', '.join(unknown)}")
    if not pass_ids:
        raise KeyError("name at least one pass")
    picked = [known[i] for i in pass_ids]
    split = picked[0]["split"]
    if any(p["split"] != split for p in picked):
        raise ValueError("passes on different splits are not the same questions")

    frames = [_scored(p).set_index(["report_id", "turn_index"]) for p in picked]
    lead = frames[0]
    rows = []
    for (report_id, turn_index), row in lead.iterrows():
        answers: list[dict[str, Any] | None] = []
        for frame in frames:
            hit = (
                frame.loc[(report_id, turn_index)]
                if (report_id, turn_index) in frame.index
                else None
            )
            if hit is None:
                answers.append(None)
                continue
            error = str(hit.get("error") if pd.notna(hit.get("error")) else "")
            answers.append(
                {
                    "answer": str(hit["pred_answer"])
                    if pd.notna(hit["pred_answer"])
                    else "",
                    "correct": bool(hit["correct"]),
                    **({"error": error[:240]} if error else {}),
                }
            )
        rows.append(
            {
                "report_id": str(report_id),
                "turn_index": int(turn_index),
                "question": str(row["question"]),
                "gold_answer": str(row["gold_answer"]),
                "gold_program": str(row["gold_program"])
                if pd.notna(row.get("gold_program"))
                else "",
                "turn_type": str(row["gold_turn_type"]).lower(),
                "answers": answers,
            }
        )
    return {"split": split, "passes": pass_ids, "rows": rows}
