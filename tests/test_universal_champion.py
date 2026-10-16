"""The universal champion and the challenger-agent machinery (s19).

Pins: the champion is one agent (prompt, runtime, model, effort) and the
pipeline lineage keeps its own alias; a challenger that shares the champion's
prompt is promoted only through `promote_agent` on a passing test-split
verdict; effort is a recorded property of every sdk pass; and the model-swap
verdict reads the rule fixed before any challenger number existed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Registry: the migration and the three promotion paths
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A registry shaped as it was before 2026-10-15: two separate champions."""
    from convfinqa.tracking import registry

    target = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", target)
    for version in ("v2", "v8", "sdk_v1", "sdk_v2"):
        registry.register(version, source="test")
    doc = registry.load()
    doc.aliases = {"champion": "v8", "sdk_champion": "sdk_v1"}
    registry.save(doc)
    return target


def _migrate() -> None:
    from convfinqa.tracking import registry

    registry.migrate_to_universal_champion(
        model="claude-sonnet-5", effort="high", reason="test"
    )


SONNET_RUN = "sdk-evalloop-test100-sdk_v1·s1-20260905_174627"
HAIKU_RUN = "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_175310"


@pytest.fixture
def evidence_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty predictions tree: no legacy CSVs, and evalloop CSVs written per test."""
    from convfinqa.tracking import comparator

    root = tmp_path / "predictions"
    (root / "evalloop").mkdir(parents=True)
    monkeypatch.setattr(comparator, "PREDICTIONS_DIR", root)
    monkeypatch.setattr(comparator, "EVALLOOP_PREDICTIONS_DIR", root / "evalloop")
    return root / "evalloop"


def _pass(
    directory: Path,
    run: str,
    outcomes: list[bool],
    *,
    split: str = "test",
    unscored: bool = False,
) -> str:
    """A committed pass whose `correct` column agrees with re-scoring."""
    pd.DataFrame(
        [
            {
                "report_id": f"r{i // 3}",
                "turn_index": i % 3,
                "question": f"q{i}",
                "gold_answer": "1",
                "pred_answer": "1" if ok else "2",
                "correct": ok,
                "split": split,
                "unscored": unscored and i == 0,
            }
            for i, ok in enumerate(outcomes)
        ]
    ).to_csv(directory / f"{run}.csv", index=False)
    return run


class _Passing:
    promotable_significant = True

    def reason(self) -> str:
        return "passes"

    def as_dict(self) -> dict[str, Any]:
        return {"ok": True}


def test_migration_makes_the_sdk_champion_the_one_champion(
    legacy_registry: Path,
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    doc = json.loads(legacy_registry.read_text())
    assert doc["aliases"] == {"champion": "sdk_v1", "pipeline_champion": "v8"}
    assert doc["champion_agent"] == {
        "version": "sdk_v1",
        "runtime": "agent_sdk",
        "model": "claude-sonnet-5",
        "effort": "high",
    }
    assert registry.champion() == registry.sdk_champion() == "sdk_v1"
    assert registry.pipeline_champion() == "v8"
    event = doc["history"][-1]
    assert event["event"] == "universal_champion"
    assert event["previous_champion"] == "v8"
    # Idempotent: a second run appends nothing.
    _migrate()
    assert len(json.loads(legacy_registry.read_text())["history"]) == len(
        doc["history"]
    )


def test_a_legacy_registry_still_reads_both_champions(legacy_registry: Path) -> None:
    from convfinqa.tracking import registry

    assert registry.champion_agent() is None
    assert registry.sdk_champion() == "sdk_v1"
    assert registry.pipeline_champion() == "v8"


def test_a_pipeline_promotion_never_dethrones_the_champion(
    legacy_registry: Path,
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    outcome = registry.promote("v2", force=True, reason="test rollback")
    assert outcome.promoted
    assert registry.pipeline_champion() == "v2"
    assert registry.champion() == "sdk_v1"
    assert registry.load().history[-1]["alias"] == registry.PIPELINE_CHAMPION


def test_a_prompt_promotion_keeps_the_champions_model_and_effort(
    legacy_registry: Path,
) -> None:
    from convfinqa.tracking import registry

    _migrate()

    class Passing:
        promotable_significant = True

        def reason(self) -> str:
            return "passes"

        def as_dict(self) -> dict[str, Any]:
            return {"ok": True}

    outcome = registry.promote_sdk("sdk_v2", comparison=Passing())  # type: ignore[arg-type]
    assert outcome.promoted
    assert registry.champion() == "sdk_v2"
    assert registry.champion_agent() == {
        "version": "sdk_v2",
        "runtime": "agent_sdk",
        "model": "claude-sonnet-5",
        "effort": "high",
    }
    assert registry.pipeline_champion() == "v8"


def _verdict(**over: Any) -> dict[str, Any]:
    base = {
        "outcome": "matches",
        "reason": "within margin",
        "version": "sdk_v1",
        "candidate_model": "claude-haiku-5-5",
        "candidate_effort": "high",
        "candidate_run": HAIKU_RUN,
    }
    return {**base, **over}


def test_promote_agent_refuses_a_verdict_for_another_configuration(
    legacy_registry: Path,
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    before = legacy_registry.read_text()
    bad = [
        _verdict(candidate_model="claude-haiku-4-5-20251001"),
        _verdict(candidate_effort="low"),
        _verdict(version="sdk_v2"),
        _verdict(candidate_run=None),
        {"outcome": "matches", "reason": "no identity"},
    ]
    for verdict in bad:
        out = registry.promote_agent(
            model="claude-haiku-5-5", effort="high", verdict=verdict
        )
        assert not out.promoted and "not measured" in out.reason
        assert legacy_registry.read_text() == before
    assert registry.champion_agent()["model"] == "claude-sonnet-5"  # type: ignore[index]


def test_promote_agent_moves_model_and_effort_on_a_passing_verdict(
    legacy_registry: Path, evidence_dir: Path
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    _pass(evidence_dir, HAIKU_RUN, [True] * 7 + [False])
    outcome = registry.promote_agent(
        model="claude-haiku-5-5",
        effort="high",
        verdict=_verdict(),
    )
    assert outcome.promoted
    assert registry.champion() == "sdk_v1"
    # The floor moves with the agent: the challenger's own test pass.
    assert registry.champion_agent() == {
        "version": "sdk_v1",
        "runtime": "agent_sdk",
        "model": "claude-haiku-5-5",
        "effort": "high",
        "evidence_run": HAIKU_RUN,
        "accuracy": 0.875,
    }
    event = registry.load().history[-1]
    assert event["event"] == "promote_agent"
    assert event["champion_agent_before"]["model"] == "claude-sonnet-5"
    assert event["champion_agent_after"]["model"] == "claude-haiku-5-5"


def test_promote_agent_refuses_what_it_should(legacy_registry: Path) -> None:
    from convfinqa.tracking import registry

    verdict = _verdict(model="m")
    with pytest.raises(ValueError, match="no champion_agent"):
        registry.promote_agent(model="m", effort="high", verdict=verdict)
    _migrate()
    with pytest.raises(ValueError, match="unseen test split"):
        registry.promote_agent(
            model="m", effort="high", verdict=verdict, evidence_split="train"
        )
    with pytest.raises(ValueError, match="unknown effort"):
        registry.promote_agent(model="m", effort="turbo", verdict=verdict)
    refused = registry.promote_agent(
        model="claude-haiku-5-5",
        effort="high",
        verdict={"outcome": "falls_short", "reason": "CI below margin"},
    )
    assert not refused.promoted and "falls_short" in refused.reason
    assert registry.champion_agent()["model"] == "claude-sonnet-5"  # type: ignore[index]


# ---------------------------------------------------------------------------
# The champion's floor: what the CI eval-regression gate holds it to
# ---------------------------------------------------------------------------


def _gate(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    from convfinqa.tracking import gate

    code = gate.main()
    return code, capsys.readouterr().out


def test_the_gate_fails_a_champion_with_no_recorded_floor(
    legacy_registry: Path, evidence_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The regression: sdk_v1's version entry has no metrics, and the gate read
    # that as a 0.00% floor and passed, with the champion's CSV right there.
    _migrate()
    _pass(evidence_dir, SONNET_RUN, [True] * 9 + [False])
    code, out = _gate(capsys)
    assert code == 1
    assert "champion sdk_v1: no recorded floor" in out
    assert "eval-gate FAILED" in out


def test_the_gate_holds_the_champion_to_the_pass_that_measured_it(
    legacy_registry: Path, evidence_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    _pass(evidence_dir, SONNET_RUN, [True] * 9 + [False])
    # A later pass of the same prompt and model, which a "latest CSV" lookup
    # would pick up instead of the champion's evidence.
    _pass(evidence_dir, "sdk-evalloop-test100-sdk_v1·s1-20261101_000000", [True, False])
    record = registry.record_champion_evidence(SONNET_RUN)
    assert record["accuracy"] == 0.9 and record["evidence_run"] == SONNET_RUN
    assert registry.load().history[-1]["event"] == "champion_evidence"

    code, out = _gate(capsys)
    assert code == 0
    assert "champion sdk_v1: 90.00% (registered 90.00%, floor 89.50%)" in out

    # A floor the evidence no longer meets fails.
    doc = registry.load()
    doc.champion_agent = {**doc.champion_agent, "accuracy": 0.99}  # type: ignore[dict-item]
    registry.save(doc)
    code, out = _gate(capsys)
    assert code == 1
    assert "below its registered floor" in out


@pytest.mark.parametrize(
    ("run", "kwargs", "why"),
    [
        (HAIKU_RUN, {}, "ran haiku-5-5, not claude-sonnet-5"),
        ("sdk-evalloop-test100-sdk_v2·s2-20260905_182812", {}, "not a sdk_v1 run"),
        (
            "sdk-evalloop-test100-sdk_v1·s1-sonnet-5-low-20261101_000000",
            {},
            "effort low",
        ),
        (SONNET_RUN, {"split": "train"}, "outside the 'test' split"),
        (SONNET_RUN, {"unscored": True}, "unscored rows"),
    ],
)
def test_champion_evidence_refuses_a_pass_of_another_agent(
    legacy_registry: Path,
    evidence_dir: Path,
    run: str,
    kwargs: dict[str, Any],
    why: str,
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    before = legacy_registry.read_text()
    _pass(evidence_dir, run, [True, True, False], **kwargs)
    with pytest.raises(ValueError, match=why):
        registry.record_champion_evidence(run)
    assert legacy_registry.read_text() == before


def test_a_prompt_promotion_takes_its_own_floor_or_none(
    legacy_registry: Path, evidence_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    registry.record_champion_evidence(_pass(evidence_dir, SONNET_RUN, [True] * 4))
    sdk_v2 = _pass(
        evidence_dir,
        "sdk-evalloop-test100-sdk_v2·s2-20261101_000000",
        [True] * 3 + [False],
    )
    # Evidence of another agent is a caller bug, refused before anything is written.
    with pytest.raises(ValueError, match="not sdk_v2's gate pass"):
        registry.promote_sdk(
            "sdk_v2",
            comparison=_Passing(),  # type: ignore[arg-type]
            evidence_csv=evidence_dir / f"{SONNET_RUN}.csv",
        )
    assert registry.champion() == "sdk_v1"

    outcome = registry.promote_sdk(
        "sdk_v2",
        comparison=_Passing(),  # type: ignore[arg-type]
        evidence_csv=evidence_dir / f"{sdk_v2}.csv",
    )
    assert outcome.promoted
    agent = registry.champion_agent() or {}
    assert (agent["version"], agent["evidence_run"], agent["accuracy"]) == (
        "sdk_v2",
        sdk_v2,
        0.75,
    )
    assert _gate(capsys)[0] == 0

    # Promoted with no evidence: sdk_v1's floor measured another prompt, so it
    # is dropped, and the gate fails until the new champion's is recorded.
    registry.register("sdk_v3", source="test")
    registry.promote_sdk("sdk_v3", comparison=_Passing())  # type: ignore[arg-type]
    assert not {"accuracy", "evidence_run"} & set(registry.champion_agent() or {})
    code, out = _gate(capsys)
    assert code == 1 and "no recorded floor" in out


def test_the_champion_alias_takes_either_runtime_but_never_a_judge(
    legacy_registry: Path,
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    with pytest.raises(ValueError, match="different runtimes"):
        registry.set_alias(registry.PIPELINE_CHAMPION, "sdk_v2")
    registry.set_alias(registry.CHAMPION, "v8")
    assert registry.champion() == "v8"


# ---------------------------------------------------------------------------
# llm: model and effort resolution
# ---------------------------------------------------------------------------


def test_model_and_effort_come_from_the_record_unless_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa import llm
    from convfinqa.config import settings
    from convfinqa.tracking import registry

    monkeypatch.setattr(settings, "sdk_model", None, raising=False)
    monkeypatch.setattr(settings, "sdk_effort", None, raising=False)
    monkeypatch.setattr(
        registry,
        "champion_agent",
        lambda path=None: {"model": "claude-haiku-5-5", "effort": "high"},
    )
    assert llm.sdk_model_name() == "claude-haiku-5-5"
    assert llm.sdk_effort_name() == "high"
    monkeypatch.setattr(settings, "sdk_effort", "medium", raising=False)
    assert llm.sdk_effort_name() == "medium"
    monkeypatch.setattr(registry, "champion_agent", lambda path=None: None)
    monkeypatch.setattr(settings, "sdk_effort", None, raising=False)
    assert llm.sdk_model_name() == llm.LM_SDK_MODEL
    assert llm.sdk_effort_name() == llm.LM_SDK_EFFORT


def test_a_runs_effort_is_dated_against_the_low_pin() -> None:
    from datetime import datetime

    from convfinqa.llm import SDK_EFFORT_PINNED_LOW_AT, sdk_effort_of_run

    pin = datetime.fromisoformat(SDK_EFFORT_PINNED_LOW_AT).timestamp() * 1000
    assert sdk_effort_of_run({}, pin - 1) == "high"
    assert sdk_effort_of_run({}, pin + 1) == "low"
    assert sdk_effort_of_run({"sdk_effort": "xhigh"}, pin - 1) == "xhigh"


def test_pipeline_sdk_options_carry_the_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.config import settings
    from convfinqa.llm import SDK_ALLOWED_TOOLS, pipeline_sdk_options

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("SDK_ALLOW_API_BILLING", "1")
    monkeypatch.setattr(settings, "sdk_billing", "api", raising=False)
    kwargs: dict[str, Any] = {
        "system_prompt": "s",
        "mcp_server": {},
        "allowed_tools": list(SDK_ALLOWED_TOOLS),
        "output_schema": {},
        "max_turns": 1,
    }
    assert pipeline_sdk_options(**kwargs, effort="high").effort == "high"
    with pytest.raises(ValueError, match="unknown effort"):
        pipeline_sdk_options(**kwargs, effort="turbo")


# ---------------------------------------------------------------------------
# runner: the run name carries the effort, and a resume cannot change it
# ---------------------------------------------------------------------------


def test_run_names_split_model_and_effort() -> None:
    from convfinqa.evalloop.runner import prior_sdk_effort, prior_sdk_model

    name = "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_101500.csv"
    assert prior_sdk_model(name) == "haiku-5-5"
    assert prior_sdk_effort(name) == "high"
    old = "sdk-evalloop-test100-sdk_v1·s1-haiku-4-5-20260906_090517.csv"
    assert prior_sdk_model(old) == "haiku-4-5"
    assert prior_sdk_effort(old) is None


def test_a_resume_refuses_a_different_effort() -> None:
    from convfinqa.evalloop import runner

    prior = pd.DataFrame(
        {
            "split": ["test"],
            "model_version_id": ["sdk_v1"],
            "report_id": ["r0"],
            "runtime": ["agent_sdk"],
        }
    )
    path = "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-low-20261015_101500.csv"
    kwargs: dict[str, Any] = {
        "split": "test",
        "version": "sdk_v1",
        "runtime": "agent_sdk",
        "report_ids": ["r0"],
        "train_seed": None,
        "sdk_model_slug": "haiku-5-5",
    }
    with pytest.raises(ValueError, match="effort"):
        runner.check_resume(prior, path, sdk_effort="high", **kwargs)


# ---------------------------------------------------------------------------
# story: the train pairing and the verdict
# ---------------------------------------------------------------------------


def _early_stopped(outcomes: dict[str, list[bool]]) -> pd.DataFrame:
    """A train CSV the way `--stop-at-first-wrong` writes it: rows end at the first miss."""
    rows = []
    for report_id, oks in outcomes.items():
        for turn, ok in enumerate(oks):
            rows.append(
                {
                    "report_id": report_id,
                    "turn_index": turn,
                    "correct": ok,
                    "split": "train",
                    "model_version_id": "sdk_v1",
                }
            )
            if not ok:
                break
    return pd.DataFrame(rows)


def test_train_pairs_are_per_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import runner, story

    monkeypatch.setattr(runner, "PREDICTIONS_DIR", tmp_path)
    base = {"a": [True, True], "b": [True, False], "c": [False], "d": [True, True]}
    cand = {"a": [True, True], "b": [True, True], "c": [False], "d": [True, False]}
    _early_stopped(base).to_csv(tmp_path / "base.csv", index=False)
    _early_stopped(cand).to_csv(tmp_path / "cand.csv", index=False)
    pair = story._train_pair_from_csvs("base", "cand")
    assert pair["n_conversations"] == 4
    assert (pair["baseline_full"], pair["candidate_full"]) == (2, 2)
    assert (pair["fixed"], pair["broken"]) == (1, 1)


def test_train_pair_means_count_turns_before_the_first_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from convfinqa.evalloop import runner, story

    monkeypatch.setattr(runner, "PREDICTIONS_DIR", tmp_path)
    base = {"a": [True, True], "b": [True, False], "c": [False], "d": [True, True]}
    _early_stopped(base).to_csv(tmp_path / "base.csv", index=False)
    _early_stopped(base).to_csv(tmp_path / "same.csv", index=False)
    pair = story._train_pair_from_csvs("base", "same")
    assert pair["baseline_mean_right_before_error"] == 1.25
    assert (pair["fixed"], pair["broken"]) == (0, 0)
    # A different report set is a different question set: not paired.
    _early_stopped({"a": [True]}).to_csv(tmp_path / "other.csv", index=False)
    assert story._train_pair_from_csvs("base", "other") == {}


def _record(
    name: str, *, split: str, model: str, effort: str, seed: str | None = None
) -> dict[str, Any]:
    params = {
        "split": split,
        "runtime": "agent_sdk",
        "prompts_version": "sdk_v1",
        "sdk_model": model,
        "sdk_effort": effort,
        "stop_at_first_wrong": "True" if seed else "False",
    }
    if seed:
        params["train_draw_seed"] = seed
    return {"run_name": name, "start_time": 1, "params": params, "metrics": {}}


def test_train_runs_pair_only_on_seed_and_early_stop() -> None:
    from convfinqa.evalloop import story

    records = [
        _record("s27", split="train", model="sonnet", effort="high", seed="2027"),
        _record("h27", split="train", model="haiku", effort="high", seed="2027"),
        _record("h28", split="train", model="haiku", effort="high", seed="2028"),
        _record("hfull", split="train", model="haiku", effort="high"),
    ]
    runs = story._train_runs_by_agent(records, "sonnet")
    assert set(runs[("haiku", "high")]) == {"2027", "2028"}
    pairs = story._train_pairs(
        runs[("sonnet", "high")], runs[("haiku", "high")], read_csvs=False
    )
    assert [p["seed"] for p in pairs["by_seed"]] == ["2027"]


def test_agents_are_keyed_on_model_and_effort() -> None:
    from convfinqa.evalloop import story

    records = [
        _record(
            "sdk-evalloop-test100-a",
            split="test",
            model="claude-sonnet-5",
            effort="high",
        ),
        _record(
            "sdk-evalloop-test100-b",
            split="test",
            model="claude-haiku-5-5",
            effort="high",
        ),
        _record(
            "sdk-evalloop-test100-c",
            split="test",
            model="claude-haiku-5-5",
            effort="low",
        ),
    ]
    out = story.sdk_model_comparison(
        records,
        sdk_champion="sdk_v1",
        reference_model="claude-sonnet-5",
        reference_effort="high",
        read_csvs=False,
    )
    assert [(m["model"], m["effort"]) for m in out["models"]][0] == (
        "claude-sonnet-5",
        "high",
    )
    assert len(out["models"]) == 3 and len(out["pairs"]) == 2
    assert all(p["verdict"]["outcome"] is None for p in out["pairs"])


def _pair(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "sdk_v1",
        "candidate_model": "claude-haiku-5-5",
        "candidate_effort": "high",
        "n_compared": 349,
        "delta_pp": -1.0,
        "ci": [-0.025, 0.005],
        "p_better": 0.8,
        "p_worse": 0.2,
        "by_turn_type": {"program": {"cluster_p_one_sided": 0.6}},
        "train": {"pooled": {"n_conversations": 200, "p_worse": 0.4}},
        "ceiling_errors": {"baseline": 0, "candidate": 0},
    }
    base.update(over)
    return base


def test_the_verdict_reads_the_s19_rule() -> None:
    from convfinqa.evalloop.story import MODEL_SWAP_MARGIN_PP, model_swap_verdict

    assert MODEL_SWAP_MARGIN_PP == 3.0
    assert model_swap_verdict(_pair())["outcome"] == "matches"
    identity = model_swap_verdict(_pair())
    assert (
        identity["version"],
        identity["candidate_model"],
        identity["candidate_effort"],
    ) == ("sdk_v1", "claude-haiku-5-5", "high")
    beats = _pair(delta_pp=4.0, ci=[0.01, 0.07], p_better=0.01, p_worse=0.99)
    assert model_swap_verdict(beats)["outcome"] == "beats"
    # Each "matches" condition can sink it on its own.
    assert model_swap_verdict(_pair(ci=[-0.031, 0.0]))["outcome"] == "falls_short"
    assert model_swap_verdict(_pair(p_worse=0.04))["outcome"] == "falls_short"
    program_worse = _pair(by_turn_type={"program": {"cluster_p_one_sided": 0.97}})
    assert model_swap_verdict(program_worse)["outcome"] == "falls_short"
    ceiling = _pair(ceiling_errors={"baseline": 0, "candidate": 3})
    assert model_swap_verdict(ceiling)["outcome"] == "falls_short"
    # Train is a veto, never a substitute — for "beats" as much as "matches".
    vetoed = {"pooled": {"n_conversations": 200, "p_worse": 0.01}}
    assert model_swap_verdict(_pair(train=vetoed))["outcome"] == "falls_short"
    assert model_swap_verdict({**beats, "train": vetoed})["outcome"] == "falls_short"
    no_train = model_swap_verdict(_pair(train={"pooled": {"n_draws": 0}}))
    assert (
        no_train["outcome"] == "matches" and "no train evidence" in no_train["reason"]
    )
    # No complete test pass, no verdict.
    assert model_swap_verdict(_pair(n_compared=None))["outcome"] is None


def test_the_cross_runtime_arm_stays_the_run_its_gate_measured() -> None:
    from convfinqa.evalloop import story

    gated = _record(
        "sdk-evalloop-test100-gated",
        split="test",
        model="claude-sonnet-5",
        effort="high",
    )
    gated["run_id"] = "g1"
    later = _record(
        "sdk-evalloop-test100-later",
        split="test",
        model="claude-haiku-5-5",
        effort="high",
    )
    later["run_id"] = "l1"
    later["start_time"] = 99
    out = story.runtime_comparison(
        [gated, later],
        [
            {
                "baseline_version": "v8",
                "split": "test",
                "gated_at": "2026-09-05",
                "candidate_eval_run_id": "g1",
            }
        ],
        champion="v8",
        sdk_champion="sdk_v1",
        reference_model="claude-haiku-5-5",
    )
    assert out["agent_sdk"]["run_name"] == "sdk-evalloop-test100-gated"


# ---------------------------------------------------------------------------
# judge and story_check
# ---------------------------------------------------------------------------


def test_the_judge_reads_its_runtime_model_from_the_dataset() -> None:
    from convfinqa.evalloop.judge import runtime_model_of
    from convfinqa.llm import LM_SDK_MODEL

    assert runtime_model_of({"runtime_model": "claude-haiku-5-5"}) == "claude-haiku-5-5"
    pre_slug = {"sources": {"t": "sdk-evalloop-test100-sdk_v1·s1-20260905_174627.csv"}}
    assert runtime_model_of(pre_slug) == LM_SDK_MODEL
    slugged = {
        "sources": {"t": "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_1.csv"}
    }
    assert runtime_model_of(slugged) == "claude-haiku-5-5"
    mixed = {
        "sources": {
            "a": "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_1.csv",
            "b": "sdk-evalloop-test100-sdk_v1·s1-sonnet-5-high-20261015_2.csv",
        }
    }
    assert runtime_model_of(mixed) == "claude-haiku-5-5,claude-sonnet-5"


def test_story_check_catches_a_champion_whose_model_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from convfinqa.evalloop import story_check
    from convfinqa.tracking import registry

    record = {"version": "sdk_v1", "model": "claude-sonnet-5", "effort": "high"}
    story_path = tmp_path / "story.json"
    story_path.write_text(
        json.dumps(
            {"champion": "sdk_v1", "sdk_champion": "sdk_v1", "champion_agent": record}
        )
    )
    monkeypatch.setattr(story_check, "STORY_PATH", story_path)
    monkeypatch.setattr(registry, "champion", lambda path=None: "sdk_v1")
    monkeypatch.setattr(
        registry,
        "champion_agent",
        lambda path=None: {**record, "model": "claude-haiku-5-5"},
    )
    assert any(
        "champion_agent" in p or "records the champion" in p
        for p in story_check.problems()
    )


# ---------------------------------------------------------------------------
# The record survives the move to the central tracking server
# ---------------------------------------------------------------------------


def test_a_snapshot_rebuild_keeps_runs_the_store_no_longer_holds() -> None:
    from convfinqa.tracking.snapshot import merge_runs

    live = [
        {"run_id": "new", "run_name": "haiku"},
        {"run_id": "both", "run_name": "live"},
    ]
    committed = [
        {"run_id": "old", "run_name": "sonnet"},
        {"run_id": "both", "run_name": "stale"},
    ]
    merged = merge_runs(live, committed)
    assert [r["run_id"] for r in merged] == ["new", "both", "old"]
    assert next(r for r in merged if r["run_id"] == "both")["run_name"] == "live"


def test_the_story_sees_committed_runs_the_store_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.evalloop import story
    from convfinqa.tracking import snapshot

    monkeypatch.setattr(
        snapshot,
        "read_snapshot",
        lambda path=None: {
            "runs": [
                {
                    "run_id": "old",
                    "kind": "evalloop",
                    "run_name": "sonnet",
                    "params": {},
                },
                {"run_id": "g", "kind": "gate", "run_name": "gate", "params": {}},
            ]
        },
    )
    merged = story._with_snapshot_runs([{"run_id": "new", "run_name": "haiku"}])
    assert [r["run_id"] for r in merged] == ["new", "old"]


# ---------------------------------------------------------------------------
# promote_challenger (s21): a prompt written for another model moves as one
# ---------------------------------------------------------------------------

SDK_V3_HAIKU_RUN = "sdk-evalloop-test100-sdk_v3·s3-haiku-5-5-high-20261015_205249"


def _challenger(**over: Any) -> dict[str, Any]:
    return _verdict(**{"version": "sdk_v3", "candidate_run": SDK_V3_HAIKU_RUN, **over})


def test_promote_challenger_moves_prompt_model_and_effort_together(
    legacy_registry: Path, evidence_dir: Path
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    registry.register("sdk_v3", source="test")
    _pass(evidence_dir, SDK_V3_HAIKU_RUN, [True] * 9 + [False])
    outcome = registry.promote_challenger(
        "sdk_v3", model="claude-haiku-5-5", effort="high", verdict=_challenger()
    )
    assert outcome.promoted, outcome.reason
    assert registry.champion() == "sdk_v3"
    assert registry.champion_agent() == {
        "version": "sdk_v3",
        "runtime": "agent_sdk",
        "model": "claude-haiku-5-5",
        "effort": "high",
        "evidence_run": SDK_V3_HAIKU_RUN,
        "accuracy": 0.9,
    }
    event = registry.load().history[-1]
    assert event["event"] == "promote_challenger"
    assert event["champion_agent_before"]["version"] == "sdk_v1"
    # The pipeline lineage is untouched.
    assert registry.pipeline_champion() == "v8"


def test_promote_challenger_refuses_what_it_should(
    legacy_registry: Path, evidence_dir: Path
) -> None:
    from convfinqa.tracking import registry

    _migrate()
    registry.register("sdk_v3", source="test")
    _pass(evidence_dir, SDK_V3_HAIKU_RUN, [True] * 9 + [False])
    before = legacy_registry.read_text()
    refused = [
        _challenger(outcome="falls_short"),
        _challenger(outcome=None),
        _challenger(candidate_model="claude-haiku-4-5-20251001"),
        # sdk_v1's Haiku pass is not sdk_v3's evidence.
        _challenger(candidate_run=HAIKU_RUN),
        # A pass named for another model cannot floor a Haiku champion.
        _challenger(candidate_run=SONNET_RUN),
        _challenger(candidate_run=None),
    ]
    for verdict in refused:
        out = registry.promote_challenger(
            "sdk_v3", model="claude-haiku-5-5", effort="high", verdict=verdict
        )
        assert not out.promoted
        assert legacy_registry.read_text() == before
    with pytest.raises(ValueError, match="test split"):
        registry.promote_challenger(
            "sdk_v3",
            model="claude-haiku-5-5",
            effort="high",
            verdict=_challenger(),
            evidence_split="train",
        )
    with pytest.raises(ValueError, match="unregistered"):
        registry.promote_challenger(
            "sdk_v9", model="claude-haiku-5-5", effort="high", verdict=_challenger()
        )
    with pytest.raises(ValueError, match="single-session"):
        registry.promote_challenger(
            "v8", model="claude-haiku-5-5", effort="high", verdict=_challenger()
        )
