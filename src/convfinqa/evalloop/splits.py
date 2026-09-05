"""Deterministic grouped splits for the eval loop, materialised as a manifest.

The committed manifest — not the seed — is the source of truth: it holds the
actual ``report_id`` lists, so recreating a split never depends on re-running
this code. The seed is provenance. (The old machinery is the cautionary tale:
two 60/40 splits both seeded 42 agree on only 78 of 120 conversations, because
the code around the seed changed.)

Allocation is by conversation, never by question — a question's siblings share
the report, the history and usually the error — and deterministically
stratified on ``has_type2_question``: each stratum is shuffled once with the
manifest seed and dealt round-robin across the three splits, against a
per-split budget proportional to the stratum's share of the pool — so every
split carries the pool's own Type II mix. The pool is the 2,777 train
conversations neither GEPA nor s7 ever touched.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import EVAL_ROOT

SPLITS_DIR = EVAL_ROOT / "splits"
DEFAULT_MANIFEST_NAME = "eval_loop_v1"
DEFAULT_MANIFEST_PATH = SPLITS_DIR / f"{DEFAULT_MANIFEST_NAME}.json"
SPLIT_NAMES = ("train", "test", "holdout")


def manifest_path(name: str | None = None) -> Path:
    """Resolve which manifest to read: explicit name, ``EVAL_MANIFEST``, or default.

    A campaign runs against one manifest for its whole life, so the selector is
    an environment variable rather than a flag threaded through every call site
    — set it once for the session and every run, gate and diagnosis agrees on
    what "the gate split" means.
    """
    chosen = name or os.environ.get("EVAL_MANIFEST") or DEFAULT_MANIFEST_NAME
    if chosen.endswith(".json"):
        return Path(chosen)
    return SPLITS_DIR / f"{chosen}.json"


def _excluded_report_ids() -> set[str]:
    """The 260 conversations the old optimisers saw — never in any new split.

    ``loader.qa_data`` is already filtered to exactly that set: the 200 sampled
    conversations plus the 60 additional test ones.
    """
    from convfinqa.data.loader import qa_data

    return set(qa_data["report_id"])


def build_manifest(
    *, target_questions: int = 200, seed: int = 2026, name: str = "eval_loop_v1"
) -> dict[str, Any]:
    """Allocate the pool into train/test/holdout; return the manifest dict."""
    from convfinqa.data.loader import training_data
    from convfinqa.tracking.bundle import dataset_hash

    qa = training_data()
    excluded = _excluded_report_ids()
    per_report = (
        qa.groupby("report_id")
        .agg(
            n_questions=("question_id", "size"),
            type2=("has_type2_question", "first"),
        )
        .reset_index()
    )
    per_report = per_report[~per_report["report_id"].isin(excluded)]

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    n_by_report = dict(
        zip(per_report["report_id"], per_report["n_questions"], strict=True)
    )

    # Every split carries the pool's own stratum mix: each stratum gets a
    # per-split question budget proportional to its share of the pool, and is
    # dealt round-robin until each split's budget for it is met. Without the
    # budgets, whichever stratum is dealt first fills every split alone.
    total_q = int(per_report["n_questions"].sum())
    type2_q = int(per_report.loc[per_report["type2"], "n_questions"].sum())
    share2 = type2_q / total_q
    budgets = {True: target_questions * share2, False: target_questions * (1 - share2)}
    stratum_counts: dict[str, dict[bool, int]] = {
        s: {True: 0, False: 0} for s in SPLIT_NAMES
    }

    for stratum in (True, False):
        ids = sorted(per_report.loc[per_report["type2"] == bool(stratum), "report_id"])
        rng.shuffle(ids)
        deal = 0
        for rid in ids:
            open_splits = [
                s for s in SPLIT_NAMES if stratum_counts[s][stratum] < budgets[stratum]
            ]
            if not open_splits:
                break
            chosen = open_splits[deal % len(open_splits)]
            splits[chosen].append(rid)
            stratum_counts[chosen][stratum] += int(n_by_report[rid])
            deal += 1

    counts = {
        s: stratum_counts[s][True] + stratum_counts[s][False] for s in SPLIT_NAMES
    }

    all_ids = [rid for ids in splits.values() for rid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split allocation produced overlapping report ids")
    if set(all_ids) & excluded:
        raise ValueError("split allocation leaked an excluded report id")
    short = [s for s in SPLIT_NAMES if counts[s] < target_questions]
    if short:
        raise ValueError(f"splits below the question target: {short}")

    type2_ids = set(per_report.loc[per_report["type2"], "report_id"])
    stats = {
        s: {
            "n_reports": len(ids),
            "n_questions": counts[s],
            "type2_share": round(sum(1 for r in ids if r in type2_ids) / len(ids), 4),
        }
        for s, ids in splits.items()
    }

    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_hash": dataset_hash(),
        "excluded": {
            "n_report_ids": len(excluded),
            "reason": "seen by GEPA (optimizer_split) and s7 (v2 failures)",
        },
        "stratify": ["has_type2_question"],
        "target_questions": target_questions,
        "stats": stats,
        "splits": splits,
        "opened": [],
    }


def write_manifest(
    manifest: dict[str, Any], path: Path | None = None, *, force: bool = False
) -> Path:
    """Write the manifest; refuse to overwrite an existing one unless forced."""
    path = path or manifest_path()
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists — the committed manifest is the source of "
            "truth for every run scored against it. Pass --force only to "
            "rebuild a manifest nothing has used yet."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the committed manifest."""
    path = path or manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No split manifest at {path}. Run `convfinqa-evalloop make-splits`."
        )
    manifest: dict[str, Any] = json.loads(path.read_text())
    return manifest


def split_report_ids(
    split: str,
    *,
    n_reports: int | None = None,
    n_questions: int | None = None,
    path: Path | None = None,
) -> list[str]:
    """The report ids of one split, in manifest order, optionally truncated.

    `n_reports` truncates by conversation count. `n_questions` truncates by
    cumulative question count instead — a per-run budget, not a resize of the
    committed manifest — by walking manifest order and stopping once the
    budget is met. Pass at most one; both leave the manifest's own train/test/
    holdout pools (and any evidence already recorded against them) untouched.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if n_reports and n_questions:
        raise ValueError("pass at most one of n_reports, n_questions")
    ids = list(load_manifest(path)["splits"][split])
    if n_reports:
        return ids[:n_reports]
    if n_questions:
        from convfinqa.data.loader import training_data

        counts = training_data().groupby("report_id")["question_id"].size().to_dict()
        out: list[str] = []
        total = 0
        for report_id in ids:
            if total >= n_questions:
                break
            out.append(report_id)
            total += counts.get(report_id, 0)
        return out
    return ids


def _pool_frame() -> Any:
    """Per-report pool: everything neither GEPA nor s7 saw, with its stratum."""
    from convfinqa.data.loader import training_data

    qa = training_data()
    per_report = (
        qa.groupby("report_id")
        .agg(n_questions=("question_id", "size"), type2=("has_type2_question", "first"))
        .reset_index()
    )
    return per_report[~per_report["report_id"].isin(_excluded_report_ids())]


def _stratified_draw(
    candidates: Any, n: int, rng: random.Random, already: dict[bool, int]
) -> list[str]:
    """Draw `n` reports keeping the pool's own Type II mix.

    `already` is the stratum composition of whatever the split holds when
    extending a parent manifest, so the *finished* split matches the pool rather
    than the increment doing so.
    """
    share2 = float(candidates.loc[candidates["type2"], "n_questions"].sum()) / max(
        1.0, float(candidates["n_questions"].sum())
    )
    total = sum(already.values()) + n
    want = {True: round(total * share2), False: 0}
    want[False] = total - want[True]
    need = {k: max(0, want[k] - already.get(k, 0)) for k in (True, False)}
    # Rounding can over-ask by one; the loop below stops at `n` regardless.
    out: list[str] = []
    for stratum in (True, False):
        ids = sorted(candidates.loc[candidates["type2"] == stratum, "report_id"])
        rng.shuffle(ids)
        out.extend(ids[: need[stratum]])
    rng.shuffle(out)
    return out[:n]


def build_report_manifest(
    *,
    name: str,
    train_reports: int,
    test_reports: int,
    extend: str | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Allocate train/gate splits by *report count*, optionally extending a parent.

    Two properties this guarantees, both asserted before it returns:

    - **Superset.** Every report of the parent's train stays in train, and every
      report of its test stays in test. Evidence already recorded against the
      parent therefore remains evidence about the same questions — extending a
      split does not invalidate the runs that came before it.
    - **Disjoint.** ``train ∩ test = ∅``, and neither touches the 260 reports the
      old optimisers saw. A gate report reaching train would mean the teacher
      tunes prompts on the very questions the gate scores.

    The holdout is deliberately left **unallocated**: during a campaign it is the
    untouched remainder of the pool, from which a sealed split can be cut later.
    """
    from convfinqa.tracking.bundle import dataset_hash

    per_report = _pool_frame()
    type2_by_id = dict(zip(per_report["report_id"], per_report["type2"], strict=True))
    n_by_report = dict(
        zip(per_report["report_id"], per_report["n_questions"], strict=True)
    )

    parent: dict[str, Any] | None = None
    splits: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    if extend:
        parent = load_manifest(manifest_path(extend))
        splits["train"] = list(parent["splits"]["train"])
        splits["test"] = list(parent["splits"]["test"])
        # The parent's holdout goes back to the reserve rather than being
        # inherited: a campaign that never opens it should not carry a
        # pre-committed one around, and re-cutting it later is free.

    rng = random.Random(seed)
    taken = set(splits["train"]) | set(splits["test"])
    if extend and parent is not None:
        taken |= set(parent["splits"].get("holdout", []))

    for split, target in (("train", train_reports), ("test", test_reports)):
        have = splits[split]
        if len(have) > target:
            raise ValueError(
                f"{split} already holds {len(have)} reports in {extend!r}; "
                f"the superset property forbids shrinking it to {target}"
            )
        candidates = per_report[~per_report["report_id"].isin(taken)]
        already = {
            True: sum(1 for r in have if type2_by_id.get(r)),
            False: sum(1 for r in have if not type2_by_id.get(r)),
        }
        drawn = _stratified_draw(candidates, target - len(have), rng, already)
        splits[split] = have + drawn
        taken |= set(drawn)

    train_set, test_set = set(splits["train"]), set(splits["test"])
    if train_set & test_set:
        raise ValueError("train and test overlap — the gate would be tuned against")
    if (train_set | test_set) & _excluded_report_ids():
        raise ValueError("split allocation leaked a report the old optimisers saw")
    if parent is not None:
        for split in ("train", "test"):
            missing = set(parent["splits"][split]) - set(splits[split])
            if missing:
                raise ValueError(
                    f"{split} is not a superset of {extend}: dropped {sorted(missing)}"
                )

    stats = {
        s: {
            "n_reports": len(ids),
            "n_questions": int(sum(n_by_report[r] for r in ids)),
            "type2_share": round(
                sum(1 for r in ids if type2_by_id.get(r)) / len(ids), 4
            )
            if ids
            else 0.0,
        }
        for s, ids in splits.items()
    }
    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_hash": dataset_hash(),
        "extends": extend,
        "allocation": "report_count",
        "excluded": {
            "n_report_ids": len(_excluded_report_ids()),
            "reason": "seen by GEPA (optimizer_split) and s7 (v2 failures)",
        },
        "stratify": ["has_type2_question"],
        "targets": {"train": train_reports, "test": test_reports, "holdout": 0},
        "holdout_note": (
            "unallocated by design — during a campaign the holdout is the "
            "untouched remainder of the pool, cut only for a confirmatory run"
        ),
        "stats": stats,
        "splits": splits,
        "opened": [],
    }


def reserve_report_ids(path: Path | None = None) -> list[str]:
    """Pool reports no split of this manifest claims — where a holdout is cut from."""
    manifest = load_manifest(path)
    claimed = {r for ids in manifest["splits"].values() for r in ids}
    return sorted(set(_pool_frame()["report_id"]) - claimed)


def draw_train(
    *, seed: int, n_reports: int, path: Path | None = None
) -> tuple[list[str], dict[str, Any]]:
    """A fresh train draw from ``pool − gate``, with the provenance to recreate it.

    Resampling train every cycle is what stops the teacher from overfitting to
    one set of 100 conversations — but the *gate* split must never move, so the
    draw excludes it explicitly rather than trusting that it will not collide.
    Returns the ids and the provenance dict that gets logged with the run.
    """
    manifest = load_manifest(path)
    gate = set(manifest["splits"]["test"])
    per_report = _pool_frame()
    candidates = per_report[~per_report["report_id"].isin(gate)]
    rng = random.Random(seed)
    ids = _stratified_draw(candidates, n_reports, rng, {True: 0, False: 0})
    if set(ids) & gate:
        raise ValueError("train draw collided with the gate split")
    return ids, {
        "manifest": manifest["name"],
        "draw_seed": seed,
        "n_reports": len(ids),
        "excluded_gate_reports": len(gate),
        "pool_size": int(len(candidates)),
    }
