"""The fine-tuning record (s32): current, placed by report ids, and served as committed."""

# ruff: noqa: D103

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from convfinqa.serving import app as api_app
from convfinqa.sft import progress


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    return progress.build()


def _pass(record: dict[str, Any], pass_id: str | None) -> dict[str, Any]:
    return next(p for p in record["passes"] if p["id"] == pass_id)


def test_the_committed_record_is_a_rebuild_of_its_inputs(
    record: dict[str, Any],
) -> None:
    committed = progress.PROGRESS_PATH.read_text()
    assert committed == progress.render(record), (
        "evaluation/fine_tuning.json is stale: run `uv run convfinqa-sft progress`"
    )


def test_val_passes_are_placed_by_their_ids_not_their_names(
    record: dict[str, Any],
) -> None:
    val = [p for p in record["passes"] if p["split"] == "val"]
    assert val, "no val pass placed"
    # they ran as --split train, so every one is named train100
    assert all("-train100-" in p["id"] for p in val)
    assert all(
        p["n_conversations"] == record["splits"]["val"]["n_conversations"] for p in val
    )


def test_each_round_picks_the_checkpoint_val_ranks_highest(
    record: dict[str, Any],
) -> None:
    rounds = {r["key"]: r for r in record["rounds"]}
    assert list(rounds) == ["sft1", "sft2", "dpo1", "dpo1full"]
    for r in rounds.values():
        scored = [c for c in r["checkpoints"] if c["val_pass"]]
        top = max(scored, key=lambda c: _pass(record, c["val_pass"])["accuracy"])
        assert r["best"] == top["key"]
    assert rounds["sft1"]["best"] == "e2"
    assert rounds["sft2"]["best"] == "e1"
    # round 2 starts where round 1's pick left off, on round 2's harness
    start = rounds["sft2"]["start"]
    assert start["model"] == "student-qwen3-4b-sft1-e2"
    assert (
        _pass(record, start["val_pass"])["harness"] == rounds["sft2"]["sweep_harness"]
    )


def test_the_test_progression_steps_one_change_at_a_time(
    record: dict[str, Any],
) -> None:
    stages = record["progression"]["test"]
    # DPO v1 is on the line but not opened on test: the owner's word opens it
    assert stages[-1]["key"] == "dpo1full" and stages[-1]["pass"] is None
    assert stages[-1]["vs_prev"] is None
    stages = stages[:-1]
    assert [s["kind"] for s in stages] == ["base", "round", "harness", "round"]
    accuracy = [round(_pass(record, s["pass"])["accuracy"], 3) for s in stages]
    assert accuracy == [0.562, 0.774, 0.802, 0.814]
    # each step pairs with the one before it on the same questions
    assert stages[0]["vs_prev"] is None
    assert [s["vs_prev"]["same_harness"] for s in stages[1:]] == [True, False, True]
    assert all(s["vs_prev"]["n"] == _pass(record, s["pass"])["n"] for s in stages[1:])


def test_an_incomplete_pass_is_excluded_and_never_drawn(record: dict[str, Any]) -> None:
    incomplete = [p for p in record["passes"] if not p["complete"] and p["split"]]
    assert incomplete, "the aborted sft1 e2 test pass should be on record"
    drawn = {s["pass"] for stages in record["progression"].values() for s in stages}
    drawn |= {c["val_pass"] for r in record["rounds"] for c in r["checkpoints"]}
    drawn |= {c["test_pass"] for r in record["rounds"] for c in r["checkpoints"]}
    for p in incomplete:
        assert p["id"] not in drawn
        assert p["id"] in {e["id"] for e in record["excluded"]}


def test_the_teacher_is_the_reference_not_a_stage(record: dict[str, Any]) -> None:
    teacher = record["teacher"]
    assert teacher["name"] == "llm_react_v1"
    assert _pass(record, teacher["pass"])["split"] == "test"
    assert all(s["model"] != "deepseek-flash" for s in record["progression"]["test"])
    assert 0 < teacher["gap_closed"] < 1


def test_questions_line_passes_up_and_refuse_strangers(record: dict[str, Any]) -> None:
    r2 = next(r for r in record["rounds"] if r["key"] == "sft2")
    ids = [r2["start"]["val_pass"], *[c["val_pass"] for c in r2["checkpoints"]]]
    out = progress.questions(ids, record)
    assert out["split"] == "val"
    assert len(out["rows"]) == _pass(record, ids[0])["n"]
    for i, pass_id in enumerate(ids):
        right = sum(1 for row in out["rows"] if row["answers"][i]["correct"])
        assert right == _pass(record, pass_id)["correct"]

    with pytest.raises(KeyError):
        progress.questions(["evaluation/../../etc/passwd"], record)
    test_id = record["progression"]["test"][0]["pass"]
    with pytest.raises(ValueError):
        progress.questions([ids[0], test_id], record)


def test_the_routes_serve_the_record_and_its_cases(demo_mode: None) -> None:
    committed = progress.load()
    base = committed["progression"]["test"][0]["pass"]
    with TestClient(api_app.create_app(eviction_interval_seconds=3600)) as client:
        body = client.get("/eval/fine-tuning")
        cases = client.get("/eval/fine-tuning/questions", params={"pass": [base]})
        stranger = client.get("/eval/fine-tuning/questions", params={"pass": ["nope"]})
    assert body.status_code == 200, body.text
    assert body.json() == committed
    assert cases.status_code == 200, cases.text
    assert cases.json()["split"] == "test"
    assert stranger.status_code == 404


def test_dpo_rounds_continue_sft2_and_the_pilot_stays_off_the_line(
    record: dict[str, Any],
) -> None:
    rounds = {r["key"]: r for r in record["rounds"]}
    full, pilot = rounds["dpo1full"], rounds["dpo1"]
    assert full["method"] == pilot["method"] == "dpo"
    assert pilot["pilot"] and not full["pilot"]
    assert full["start"]["model"] == "student-qwen3-4b-sft2-e1"
    assert full["best"] == "e1"
    assert full["dataset"]["n_rows"] == 2306
    # the held-out pairs are read at each saved step
    assert [c["pair_eval"]["step"] for c in full["checkpoints"]] == [36, 72, 108, 144]
    # on val, DPO v1 steps from sft2's pick, paired on the same questions
    val = {s["key"]: s for s in record["progression"]["val"]}
    assert "dpo1" not in val
    step = val["dpo1full"]["vs_prev"]
    assert "sft2-e1" in step["from"] and step["same_harness"]
    assert step["delta_pp"] == 6.3 and step["p_one_sided"] < 0.05


def test_dpo_v2_is_listed_as_built_not_trained(record: dict[str, Any]) -> None:
    planned = {p["key"]: p for p in record["planned"]}
    assert planned["dpo_v2"]["status"] == "dataset built, not trained"
    assert planned["dpo_v2"]["n_pairs"] == 2678


def test_the_other_agents_are_their_newest_complete_test_passes(
    record: dict[str, Any],
) -> None:
    agents = record["agents"]
    assert [a["champion"] for a in agents].count(True) == 1
    assert agents[0]["champion"] and agents[0]["accuracy"] > 0.9
    keys = [(a["runtime"], a["version"], a["model"], a["effort"]) for a in agents]
    assert len(keys) == len(set(keys))
