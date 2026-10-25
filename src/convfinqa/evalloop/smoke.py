"""The open-agent smoke (s25): 20 train conversations the champion already answered.

The owner's rule for the open runtime is that nothing new is run for Claude:
the yardstick is the champion's committed early-stopped train draws (seeds 2027
and 2028). So the smoke is a subset of those draws, and a smoke pass is the
open agent run on `--train-seed N --report-ids-file smoke_v1_seedN.txt` — the
existing machinery, which refuses an id outside the draw.

`pick` chooses the 20 by a seeded search: stratified draws from the draws'
page-clean conversations (none on a report page that test or the sealed holdout
uses), keeping only sets that cover every case below, and taking the one
closest to the draws' own mix and to the champion's record on them.
`compare` reads a smoke pass beside the champion's records, at conversation and
question level, the way `story._train_pair_from_csvs` compares train draws.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.config import EVAL_ROOT

# The champion's committed train draws (s19): sdk_v1 · Sonnet 5 · high,
# stopping at the first wrong turn. They are the smoke's only baseline.
CHAMPION_TRAIN_RUNS: dict[int, str] = {
    2027: "sdk-evalloop-train100-sdk_v1·s1-20260905_180924",
    2028: "sdk-evalloop-train100-sdk_v1·s1-20260905_184211",
}
SMOKE_NAME = "smoke_v1"
SMOKE_SEED = 2033
SMOKE_SIZE = 20
SMOKE_TRIES = 200_000
OPERATIONS = ("add", "subtract", "multiply", "divide", "greater")

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
_OPERAND = re.compile(r"(?<![#\w])-?\d+\.?\d*")
_OP = re.compile(r"(add|subtract|multiply|divide|exp|greater)\(")


def _predictions_dir() -> Path:
    from convfinqa.evalloop.runner import PREDICTIONS_DIR

    return PREDICTIONS_DIR


def manifest_path(name: str = SMOKE_NAME) -> Path:
    """The committed smoke manifest."""
    return EVAL_ROOT / "splits" / f"{name}.json"


def ids_path(seed: int, name: str = SMOKE_NAME) -> Path:
    """The `--report-ids-file` for one seed's share of the smoke."""
    return EVAL_ROOT / "splits" / f"{name}_seed{seed}.txt"


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[,$%]", "", x).lstrip("0") or "0" for x in _NUMBER.findall(text)}


def conversation_features(record: dict[str, Any]) -> dict[str, Any]:
    """What a conversation asks of an agent, read from the dataset alone.

    `source` is a heuristic — where a gold operand appears, in the table, the
    text or both — good enough to stratify on, not to score with.
    """
    dialogue, features = record["dialogue"], record["features"]
    programs = list(dialogue["turn_program"])
    answers = [str(a) for a in dialogue["conv_answers"]]
    steps = [p.count("(") for p in programs]
    doc = record["doc"]
    in_table = _numbers(json.dumps(doc["table"]))
    in_text = _numbers(doc["pre_text"] + " " + doc["post_text"])
    sources: set[str] = set()
    for program in programs:
        for token in _OPERAND.findall(program):
            value = token.lstrip("-").lstrip("0") or "0"
            if value == "100":
                continue
            if value in in_table and value not in in_text:
                sources.add("table")
            elif value in in_text and value not in in_table:
                sources.add("text")
    source = (
        "both"
        if sources == {"table", "text"}
        else (next(iter(sources)) if sources else "table")
    )
    return {
        "type2": bool(features["has_type2_question"]),
        "n_turns": len(programs),
        "n_program": sum(s > 0 for s in steps),
        "max_steps": max(steps),
        "ops": "+".join(sorted(set(_OP.findall(" ".join(programs))))),
        "pct_answer": any("%" in a for a in answers),
        "yes_no": any(a.strip().lower() in ("yes", "no") for a in answers),
        "neg_answer": any(a.strip().startswith("-") for a in answers),
        "dup_cols": bool(features["has_duplicate_columns"]),
        "non_numeric": bool(features["has_non_numeric_values"]),
        "source": source,
    }


def outcomes(csv: Path | str) -> dict[str, dict[str, Any]]:
    """Per conversation: right to the end, turns answered, right, right before error."""
    frame = pd.read_csv(csv)
    out: dict[str, dict[str, Any]] = {}
    for report_id, rows in frame.groupby("report_id", sort=False):
        ok = rows.sort_values("turn_index")["correct"].astype(str).str.lower() == "true"
        out[str(report_id)] = {
            "full": bool(ok.all()),
            "rows": len(ok),
            "right": int(ok.sum()),
            "right_before_error": int(ok.cumprod().sum()),
        }
    return out


def champion_records() -> pd.DataFrame:
    """The champion's per-conversation record on both draws, in draw order."""
    rows = []
    for seed, run in CHAMPION_TRAIN_RUNS.items():
        for report_id, rec in outcomes(_predictions_dir() / f"{run}.csv").items():
            rows.append({"seed": seed, "report_id": report_id, **rec})
    return pd.DataFrame(rows)


def _profile(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "type2": frame.type2.mean(),
        "pct": frame.pct_answer.mean(),
        "neg": frame.neg_answer.mean(),
        "nonnum": frame.non_numeric.mean(),
        "turns": frame.n_turns.mean() / 8,
        "steps3": (frame.max_steps >= 3).mean(),
        "steps1": (frame.max_steps == 1).mean(),
        "text": (frame.source != "table").mean(),
        "prog": frame.n_program.sum() / frame.n_turns.sum(),
    }


def _covers(frame: pd.DataFrame) -> bool:
    ops = set("+".join(frame.ops.fillna("")).split("+"))
    return all(
        [
            set(OPERATIONS) <= ops,
            (frame.max_steps == 1).any(),
            (frame.max_steps >= 3).sum() >= 2,
            (frame.n_turns <= 2).any(),
            (frame.n_turns >= 6).any(),
            set(frame.source) >= {"table", "text", "both"},
            frame.dup_cols.any(),
            frame.non_numeric.sum() >= 2,
            frame.neg_answer.sum() >= 3,
            frame.yes_no.any(),
            set(frame.seed) == set(CHAMPION_TRAIN_RUNS),
        ]
    )


def candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every draw conversation with its features, the page-clean pool).

    A conversation in both draws keeps its first draw's record, so each
    appears once and its smoke run is paired with the draw it came from.
    """
    from convfinqa.data.loader import load_raw_dataset
    from convfinqa.sft.splits import excluded_pages, page_of

    raw = load_raw_dataset()
    records = {r["id"]: r for part in raw.values() for r in part}
    champ = champion_records().drop_duplicates("report_id")
    feats = pd.DataFrame(
        [
            {"report_id": rid, **conversation_features(records[rid])}
            for rid in champ.report_id
        ]
    )
    every = feats.merge(champ, on="report_id")
    bad = excluded_pages()
    pool = every[~every.report_id.map(page_of).isin(bad)]
    return every, pool


def pick(
    *, seed: int = SMOKE_SEED, size: int = SMOKE_SIZE, tries: int = SMOKE_TRIES
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The smoke set and how it was chosen. Deterministic for a seed."""
    every, pool = candidates()
    records = champion_records()
    target = _profile(every)
    target_full = float(records.full.mean())
    target_acc = float(records.right.sum() / records.rows.sum())
    rng = random.Random(seed)
    hybrid, single = pool[pool.type2], pool[~pool.type2]
    n_hybrid = round(size * target["type2"])
    best: pd.DataFrame | None = None
    best_score = float("inf")
    for _ in range(tries):
        sample = pd.concat(
            [
                hybrid.sample(n_hybrid, random_state=rng.randrange(2**31)),
                single.sample(size - n_hybrid, random_state=rng.randrange(2**31)),
            ]
        )
        if not _covers(sample):
            continue
        prof = _profile(sample)
        score = (
            sum(abs(prof[k] - target[k]) for k in target)
            + 3 * abs(sample.full.mean() - target_full)
            + 3 * abs(sample.right.sum() / sample.rows.sum() - target_acc)
        )
        if score < best_score:
            best, best_score = sample, score
    if best is None:
        raise RuntimeError(f"no set of {size} covers every case in {tries} tries")
    chosen = best.sort_values(["type2", "n_turns"]).reset_index(drop=True)
    record = {
        "seed": seed,
        "tries": tries,
        "score": round(best_score, 6),
        "pool": len(pool),
        "population": len(every),
        "champion_full": [int(chosen.full.sum()), size],
        "champion_turns_right": [int(chosen.right.sum()), int(chosen.rows.sum())],
        "population_full": [int(records.full.sum()), len(records)],
        "population_turns_right": [int(records.right.sum()), int(records.rows.sum())],
    }
    return chosen, record


def write(name: str = SMOKE_NAME, *, force: bool = False, **kw: Any) -> Path:
    """Write the manifest and one `--report-ids-file` per seed."""
    path = manifest_path(name)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; it is what every smoke pass ran on")
    chosen, record = pick(**kw)
    by_seed = {
        int(seed): list(group.report_id)
        for seed, group in chosen.groupby("seed", sort=True)
    }
    path.write_text(
        json.dumps(
            {
                "name": name,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "baseline_runs": {str(k): v for k, v in CHAMPION_TRAIN_RUNS.items()},
                "selection": record,
                "ids_by_seed": {str(k): v for k, v in by_seed.items()},
                "conversations": chosen.to_dict(orient="records"),
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    for seed, ids in by_seed.items():
        ids_path(seed, name).write_text("\n".join(ids) + "\n")
    return path


def _arm(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    picked = [rows[i] for i in ids]
    return {
        "conversations": len(picked),
        "right_to_end": sum(r["full"] for r in picked),
        "answered_turns": sum(r["rows"] for r in picked),
        "answered_right": sum(r["right"] for r in picked),
        "mean_right_before_error": round(
            sum(r["right_before_error"] for r in picked) / max(1, len(picked)), 4
        ),
    }


def _early_stopped(
    rows: dict[str, dict[str, Any]], csv: Path | str
) -> dict[str, dict[str, Any]]:
    """A full run read as if it had stopped at its first wrong turn.

    Turns before the first error are the same in either mode — a later turn
    cannot change an earlier one — so truncation gives exactly the early-stop
    record the champion's runs hold.
    """
    frame = pd.read_csv(csv)
    out: dict[str, dict[str, Any]] = {}
    for report_id, group in frame.groupby("report_id", sort=False):
        ok = list(
            group.sort_values("turn_index")["correct"].astype(str).str.lower() == "true"
        )
        stop = ok.index(False) + 1 if False in ok else len(ok)
        kept = ok[:stop]
        out[str(report_id)] = {
            "full": all(kept) and stop == len(ok),
            "rows": len(kept),
            "right": sum(kept),
            "right_before_error": rows[str(report_id)]["right_before_error"],
        }
    return out


def compare(
    candidate_csvs: Mapping[int, Path | str], name: str = SMOKE_NAME
) -> dict[str, Any]:
    """A smoke pass beside the champion's records on the same conversations.

    Conversation level is the repo's train comparison: right to the end, paired
    per conversation, exact one-sided McNemar. Question level is given twice:
    in the champion's own mode (turns up to the first wrong one), which is the
    only like-for-like number, and over every turn, which only the open agent
    has — the champion was never run past a wrong turn on train.
    """
    from convfinqa.tracking.comparator import mcnemar_exact_p_one_sided

    manifest = json.loads(manifest_path(name).read_text())
    ids_by_seed = {int(k): v for k, v in manifest["ids_by_seed"].items()}
    champ_rows: dict[str, dict[str, Any]] = {}
    cand_full: dict[str, dict[str, Any]] = {}
    cand_stop: dict[str, dict[str, Any]] = {}
    for seed, ids in ids_by_seed.items():
        base = outcomes(_predictions_dir() / f"{CHAMPION_TRAIN_RUNS[seed]}.csv")
        champ_rows.update({i: base[i] for i in ids})
        full = outcomes(candidate_csvs[seed])
        missing = sorted(set(ids) - set(full))
        if missing:
            raise ValueError(f"seed {seed}: the candidate pass lacks {missing}")
        cand_full.update({i: full[i] for i in ids})
        cand_stop.update(
            {
                i: v
                for i, v in _early_stopped(full, candidate_csvs[seed]).items()
                if i in ids
            }
        )
    ids = [i for seed in sorted(ids_by_seed) for i in ids_by_seed[seed]]
    fixed = sum(1 for i in ids if cand_stop[i]["full"] and not champ_rows[i]["full"])
    broken = sum(1 for i in ids if champ_rows[i]["full"] and not cand_stop[i]["full"])
    per_conversation = [
        {
            "report_id": i,
            "seed": next(s for s, v in ids_by_seed.items() if i in v),
            "champion": champ_rows[i],
            "candidate_early_stop": cand_stop[i],
            "candidate_full": cand_full[i],
        }
        for i in ids
    ]
    return {
        "name": name,
        "champion": _arm(champ_rows, ids),
        "candidate_early_stop": _arm(cand_stop, ids),
        "candidate_full": _arm(cand_full, ids),
        "fixed": fixed,
        "broken": broken,
        "p_better": round(mcnemar_exact_p_one_sided(broken, fixed), 6),
        "p_worse": round(mcnemar_exact_p_one_sided(fixed, broken), 6),
        "per_conversation": per_conversation,
    }
