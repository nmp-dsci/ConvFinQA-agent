"""The s10 Agent SDK arm of the campaign loop: cycle, campaign, CLI, story, registry.

`evalloop.sdk_teacher` is built concurrently and is imported lazily by the
code under test, so every test here stubs it through `cycle._sdk_teacher` /
`cli._sdk_teacher` rather than importing the module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ── next_version ────────────────────────────────────────────────────────


def test_next_version_walks_the_sdk_lineage_separately() -> None:
    from convfinqa.evalloop.cycle import next_version

    assert next_version("sdk_v3") == "sdk_v4"
    assert next_version("sdk_v11") == "sdk_v12"
    # ...and the pipeline lineage is untouched by it.
    assert next_version("v2").startswith("v") and not next_version("v2").startswith(
        "sdk"
    )


# ── campaign ────────────────────────────────────────────────────────────


def _exp(promoted: bool, target: str = "reference_resolution") -> dict[str, Any]:
    return {
        "target_agent": target,
        "target_class": target,
        "promoted": promoted,
        "runtime": "agent_sdk",
        "label": "s01-e0x",
    }


def test_single_area_mode_after_two_consecutive_rejections() -> None:
    from convfinqa.evalloop import campaign

    assert campaign.single_area_mode([]) is False
    assert campaign.single_area_mode([_exp(False)]) is False
    assert campaign.single_area_mode([_exp(False), _exp(True)]) is False
    assert campaign.single_area_mode([_exp(False), _exp(False)]) is True
    # Entered for the rest of the campaign — a later promotion does not leave it.
    assert campaign.single_area_mode([_exp(False), _exp(False), _exp(True)]) is True
    # A rejection on either side of a promotion is not two in a row.
    assert campaign.single_area_mode([_exp(False), _exp(True), _exp(False)]) is False
    assert campaign.consecutive_rejections([_exp(True), _exp(False), _exp(False)]) == 2
    # Nothing is ever blocked on the SDK arm: failure classes are not agents.
    assert campaign.blocked_agents([_exp(False), _exp(False)]) == set()


def test_pick_target_class_ranks_on_the_pooled_bound_and_honours_a_request() -> None:
    from convfinqa.evalloop import campaign

    ranking = {
        "reference_resolution": {
            "faults": 12,
            "n": 40,
            "wilson_lower": 0.18,
            "rank": 1,
            "stages": ["preprocess"],
            "diagnosis_ids": [],
        },
        "unit_scale": {
            "faults": 3,
            "n": 40,
            "wilson_lower": 0.02,
            "rank": 2,
            "stages": ["retriever"],
            "diagnosis_ids": [],
        },
        "never_seen": {"faults": 0, "n": 40, "wilson_lower": 0.0, "rank": 3},
    }
    label, why = campaign.pick_target_class(ranking)
    assert label == "reference_resolution" and "12/40" in why
    assert (
        campaign.pick_target_class(ranking, requested="unit_scale")[0] == "unit_scale"
    )
    with pytest.raises(SystemExit):
        campaign.pick_target_class({"x": {"faults": 0, "n": 5}})


def test_history_reads_the_gates_ledger_before_mlflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ledger is the record; MLflow search is the fallback for older campaigns."""
    from convfinqa.evalloop import campaign, ledgers

    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(tmp_path / "ledgers"))

    def _boom(*a: Any, **k: Any) -> list[dict[str, Any]]:
        raise AssertionError("MLflow must not be searched when the ledger answers")

    monkeypatch.setattr(campaign, "_history_from_mlflow", _boom)

    ledgers.append(
        "rewrites",
        [
            ledgers.rewrite_row(
                target="system_prompt",
                failure_class="reference_resolution",
                base_version="sdk_v1",
                new_version="sdk_v2",
                prompt_before="a",
                prompt_after="b",
                diff="",
                rationale="r",
                change_kind="rule",
                runtime="agent_sdk",
                campaign="s01",
                label="s01-e01",
                rewrite_id="rw-1",
            )
        ],
    )
    stats = {
        "evidence_split": "test",
        "n_compared": 349,
        "baseline_accuracy": 0.7,
        "candidate_accuracy": 0.72,
        "accuracy_delta": 0.02,
        "fail_to_pass": 10,
        "pass_to_fail": 3,
        "cluster_p_one_sided": 0.03,
        "delta_ci_lo": 0.001,
        "delta_ci_hi": 0.04,
    }
    ledgers.append(
        "gates",
        [
            ledgers.gate_row(
                stats,
                baseline_version="sdk_v1",
                candidate_version="sdk_v2",
                promoted=False,
                reason="REJECT",
                runtime="agent_sdk",
                campaign="s01",
                label="s01-e01",
                rewrite_id="rw-1",
                gate_run_id="g-run-1",
                champion_after="sdk_v1",
                gated_at="2026-09-05T10:00:00+00:00",
            ),
            ledgers.gate_row(
                stats,
                baseline_version="v8",
                candidate_version="v13",
                promoted=True,
                reason="PROMOTE",
                runtime="multi_agent",
                campaign="c04",
                label="c04-e01",
                gate_run_id="g-run-2",
                champion_after="v13",
            ),
        ],
    )
    past = campaign.history("s01")
    assert len(past) == 1
    row = past[0]
    assert row["runtime"] == "agent_sdk"
    assert row["target_agent"] == "reference_resolution"  # the failure class
    assert row["target_class"] == "reference_resolution"
    assert row["label"] == "s01-e01" and row["run_id"] == "g-run-1"
    assert row["promoted"] is False
    assert row["accuracy_delta"] == pytest.approx(0.02)
    assert row["cluster_p_one_sided"] == pytest.approx(0.03)
    assert isinstance(row["at"], int)
    # The other arm's verdict is not this campaign's, and a runtime filter
    # keeps a mislabelled campaign name from counting against the wrong cap.
    assert campaign.history("c04")[0]["runtime"] == "pipeline"
    assert campaign.history("s01", runtime="pipeline") == []

    summary = campaign.summarise("s01")
    assert summary["runtime"] == "agent_sdk"
    assert summary["blocked_agents"] == [] and summary["single_area_mode"] is False
    assert summary["targets"][0] == {
        "label": "s01-e01",
        "target": "reference_resolution",
        "kind": "failure_class",
        "promoted": False,
    }
    assert campaign.summarise("c04")["targets"][0]["kind"] == "subagent"


def test_history_falls_back_to_mlflow_when_the_ledger_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from convfinqa.evalloop import campaign, ledgers

    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setattr(
        campaign,
        "_history_from_mlflow",
        lambda c, **k: [{"runtime": "pipeline", "promoted": True, "target_agent": "x"}],
    )
    assert campaign.history("c01")[0]["target_agent"] == "x"


# ── registry.promote_sdk ────────────────────────────────────────────────


def _registry(tmp_path: Path) -> Path:
    from convfinqa.tracking import registry

    path = tmp_path / "registry.json"
    registry.save(
        registry.RegistryDoc(
            versions=[{"version": v} for v in ("v8", "sdk_v1", "sdk_v2")],
            aliases={"champion": "v8"},
            history=[],
        ),
        path,
    )
    return path


def _comparison(ok: bool) -> Any:
    return SimpleNamespace(
        promotable_significant=ok,
        reason=lambda: "ok" if ok else "not significant",
        as_dict=lambda: {"promotable_significant": ok},
    )


def test_promote_sdk_moves_only_sdk_champion(tmp_path: Path) -> None:
    from convfinqa.tracking import registry

    path = _registry(tmp_path)
    # First sdk version: default promotion, no comparison needed.
    first = registry.promote_sdk("sdk_v1", path=path)
    assert first.promoted and first.previous_champion is None
    doc = registry.load(path)
    assert doc.aliases == {"champion": "v8", "sdk_champion": "sdk_v1"}
    assert doc.history[-1]["event"] == "promote_sdk"
    assert doc.history[-1]["alias"] == "sdk_champion"
    assert doc.history[-1]["actor"] == "evalloop-cycle-sdk"

    # Moving it again needs a passing comparison.
    with pytest.raises(ValueError, match="needs a passing gate comparison"):
        registry.promote_sdk("sdk_v2", path=path)
    refused = registry.promote_sdk("sdk_v2", comparison=_comparison(False), path=path)
    assert not refused.promoted and "refused" in refused.reason
    assert registry.load(path).aliases["sdk_champion"] == "sdk_v1"

    moved = registry.promote_sdk("sdk_v2", comparison=_comparison(True), path=path)
    assert moved.promoted and moved.previous_champion == "sdk_v1"
    doc = registry.load(path)
    assert doc.aliases["sdk_champion"] == "sdk_v2"
    assert doc.aliases["champion"] == "v8"  # never touched
    assert registry.sdk_champion(path) == "sdk_v2"


def test_promote_sdk_refuses_bundles_and_train_evidence(tmp_path: Path) -> None:
    from convfinqa.tracking import registry

    path = _registry(tmp_path)
    with pytest.raises(ValueError, match="not a single-session prompt"):
        registry.promote_sdk("v8", path=path)
    with pytest.raises(ValueError, match="unseen test split"):
        registry.promote_sdk(
            "sdk_v1", comparison=_comparison(True), evidence_split="train", path=path
        )
    with pytest.raises(ValueError, match="unregistered"):
        registry.promote_sdk("sdk_v9", path=path)
    # And the pipeline's promote still refuses the sdk lineage.
    with pytest.raises(ValueError, match="cannot be an sdk version"):
        registry.promote("sdk_v1", path=path)
    assert "sdk_champion" not in registry.load(path).aliases


# ── CLI ─────────────────────────────────────────────────────────────────


def test_cli_parses_the_runtime_flags_and_new_subcommands() -> None:
    from convfinqa.evalloop import cli

    ap = cli.build_parser()
    a = ap.parse_args(["cycle", "--campaign", "s01", "--runtime", "agent_sdk"])
    assert a.runtime == "agent_sdk" and a.campaign == "s01"
    assert ap.parse_args(["cycle", "--campaign", "c01"]).runtime == "pipeline"
    a = ap.parse_args(
        ["diagnose", "--csv", "x.csv", "--version", "sdk_v1", "--runtime", "agent_sdk"]
    )
    assert a.runtime == "agent_sdk"
    a = ap.parse_args(
        [
            "propose",
            "--diagnoses",
            "d.jsonl",
            "--base-version",
            "sdk_v1",
            "--new-version",
            "sdk_v2",
            "--runtime",
            "agent_sdk",
            "--max-areas",
            "1",
        ]
    )
    assert a.max_areas == 1
    a = ap.parse_args(
        [
            "gate-targeted",
            "--runtime",
            "agent_sdk",
            "--target-class",
            "reference_resolution",
            "--baseline-csv",
            "a.csv",
            "--candidate-csv",
            "b.csv",
            "--baseline-version",
            "sdk_v1",
            "--candidate-version",
            "sdk_v2",
            "--promote",
        ]
    )
    assert a.target_class == "reference_resolution" and a.target_agent is None
    a = ap.parse_args(["sdk-distil"])
    assert (a.source_version, a.new_version) == ("v8", "sdk_v1")
    a = ap.parse_args(["backfill-ledgers", "--no-mlflow"])
    assert a.no_mlflow is True
    assert ap.parse_args(["ledger-trace", "--question-id", "R1_q0"]).question_id
    assert ap.parse_args(["ledger-trace", "--edit-id", "e-1"]).edit_id == "e-1"
    with pytest.raises(SystemExit):
        ap.parse_args(["ledger-trace"])
    with pytest.raises(SystemExit):
        ap.parse_args(["cycle", "--campaign", "s01", "--runtime", "nope"])


def test_cli_refuses_a_version_from_the_other_runtime() -> None:
    from convfinqa.evalloop import cli

    ap = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.check_runtime(ap, "pipeline", "sdk_v1")
    with pytest.raises(SystemExit):
        cli.check_runtime(ap, "agent_sdk", "v8")
    cli.check_runtime(ap, "agent_sdk", "sdk_v1", None)
    cli.check_runtime(ap, "pipeline", "v8", "v3_1")


def test_backfill_ledgers_cli_passes_no_mlflow_through(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from convfinqa.evalloop import cli, ledgers

    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> dict[str, int]:
        seen.update(kwargs)
        return {"diagnoses": 3, "gates": 1}

    monkeypatch.setattr(ledgers, "backfill_ledgers", fake)
    monkeypatch.setattr(
        "sys.argv", ["convfinqa-evalloop", "backfill-ledgers", "--no-mlflow"]
    )
    cli.main()
    assert seen["use_mlflow"] is False
    assert json.loads(capsys.readouterr().out)["diagnoses"] == 3


# ── story ───────────────────────────────────────────────────────────────


def _eval(
    run_name: str, version: str, runtime: str, start: int, **metrics: float
) -> dict:
    return {
        "run_name": run_name,
        "start_time": start,
        "params": {"split": "test", "prompts_version": version, "runtime": runtime},
        "metrics": {
            "accuracy": 0.7,
            "acc_triage_turn_type": 0.9,
            "wall_seconds": 600.0,
            **metrics,
        },
    }


def test_runtime_comparison_is_all_none_without_an_sdk_run() -> None:
    from convfinqa.evalloop import story

    out = story.runtime_comparison(
        [_eval("evalloop-test100-v8·t3-1", "v8", "pipeline", 1)],
        [],
        champion="v8",
        sdk_champion=None,
    )
    assert out["pipeline"]["version"] == "v8"
    assert out["pipeline"]["accuracy"] == 0.7 and out["pipeline"]["cost"] is None
    assert all(v is None for v in out["agent_sdk"].values())
    assert out["gate"]["delta_pp"] is None and out["gate"]["ci"] == [None, None]

    empty = story.runtime_comparison([], [], champion=None, sdk_champion=None)
    assert all(v is None for v in empty["pipeline"].values())


def test_runtime_comparison_reads_the_latest_sdk_run_and_gate_row() -> None:
    from convfinqa.evalloop import story

    evals = [
        _eval("evalloop-test100-v8·t3-1", "v8", "pipeline", 1),
        _eval("evalloop-test100-v8·t3-2", "v8", "pipeline", 5, accuracy=0.71),
        _eval(
            "sdk-evalloop-test100-sdk_v1-1", "sdk_v1", "agent_sdk", 2, sdk_cost_usd=3.5
        ),
        _eval("sdk-evalloop-test100-sdk_v2-1", "sdk_v2", "agent_sdk", 9, accuracy=0.74),
        _eval("sdk-evalloop-train100-sdk_v2-1", "sdk_v2", "agent_sdk", 10),
    ]
    evals[-1]["params"]["split"] = "train"
    gates = [
        {
            "gated_at": "2026-09-05T09:00:00+00:00",
            "runtime": "agent_sdk",
            "baseline_version": "v8",
            "candidate_version": "sdk_v1",
            "split": "test",
            "delta_pp": -1.0,
            "p_value": 0.6,
            "ci_low": -0.03,
            "ci_high": 0.01,
            "fixed": 4,
            "broken": 8,
            "promoted": False,
            "gate_id": "g-1",
        },
        {
            "gated_at": "2026-09-05T12:00:00+00:00",
            "runtime": "agent_sdk",
            "baseline_version": "v8",
            "candidate_version": "sdk_v2",
            "split": "test",
            "delta_pp": 2.5,
            "p_value": 0.04,
            "ci_low": 0.002,
            "ci_high": 0.05,
            "fixed": 15,
            "broken": 6,
            "promoted": True,
            "gate_id": "g-2",
        },
        {  # an sdk-vs-sdk gate is a campaign verdict, not the cross-runtime one
            "gated_at": "2026-09-05T13:00:00+00:00",
            "runtime": "agent_sdk",
            "baseline_version": "sdk_v1",
            "candidate_version": "sdk_v2",
            "split": "test",
            "delta_pp": 9.0,
        },
    ]
    out = story.runtime_comparison(evals, gates, champion="v8", sdk_champion="sdk_v1")
    assert out["pipeline"]["accuracy"] == 0.71  # latest champion test100 run
    assert out["agent_sdk"]["version"] == "sdk_v1"  # the sdk champion's run wins
    assert out["agent_sdk"]["cost"] == 3.5 and out["agent_sdk"]["wall"] == 600.0
    assert out["agent_sdk"]["panel"]["triage"] == 0.9
    assert out["gate"] == {
        "delta_pp": 2.5,
        "p_value": 0.04,
        "ci": [0.002, 0.05],
        "fixed": 15,
        "broken": 6,
        "candidate_version": "sdk_v2",
        "promoted": True,
        "gate_id": "g-2",
        "by_turn_type": None,
    }
    # Without an sdk champion, the latest sdk test100 run is the arm.
    out = story.runtime_comparison(evals, gates, champion="v8", sdk_champion=None)
    assert out["agent_sdk"]["version"] == "sdk_v2"


def _story(**overrides: Any) -> dict[str, Any]:
    base = {
        "generated_at": "2026-09-05T00:00:00+00:00",
        "champion": "v8",
        "sdk_champion": None,
        "rule": "net positive AND one-sided clustered McNemar p < 0.05",
        "split": {"name": "eval_loop_v2", "gate_reports": 100, "gate_questions": 349},
        "campaigns": [],
        "lineage": [],
        "champion_track": [],
        "sdk_campaigns": [],
        "runtime_comparison": {
            "pipeline": dict.fromkeys(story_arm_keys()),
            "agent_sdk": dict.fromkeys(story_arm_keys()),
            "gate": {"delta_pp": None, "p_value": None, "ci": [None, None]},
        },
    }
    base.update(overrides)
    return base


def story_arm_keys() -> tuple[str, ...]:
    return ("version", "run_name", "accuracy", "panel", "cost", "wall")


def test_sdk_page_renders_empty_and_populated_records() -> None:
    from convfinqa.evalloop.story_page import render_sdk_page

    html = render_sdk_page(_story())
    assert "<!doctype html>" in html and "not yet run" in html
    assert "No cross-runtime gate yet" in html
    assert '"n_sdk_experiments": 0' in html and 'href="index.html"' in html

    populated = _story(
        sdk_champion="sdk_v2",
        runtime_comparison={
            "pipeline": {
                "version": "v8",
                "run_name": "evalloop-test100-v8",
                "accuracy": 0.71,
                "panel": {"triage": 0.9},
                "cost": None,
                "wall": 360,
            },
            "agent_sdk": {
                "version": "sdk_v2",
                "run_name": "sdk-evalloop-test100-sdk_v2",
                "accuracy": 0.735,
                "panel": {"triage": 0.92},
                "cost": 4.25,
                "wall": 900,
            },
            "gate": {
                "delta_pp": 2.5,
                "p_value": 0.04,
                "ci": [0.002, 0.05],
                "fixed": 15,
                "broken": 6,
                "candidate_version": "sdk_v2",
            },
        },
        sdk_campaigns=[
            {
                "name": "s01",
                "runtime": "agent_sdk",
                "experiments": [
                    {
                        "label": "s01-e01",
                        "target_agent": "reference_resolution",
                        "target_class": "reference_resolution",
                        "baseline_version": "sdk_v1",
                        "candidate_version": "sdk_v2",
                        "promoted": True,
                        "accuracy_delta": 0.025,
                        "cluster_p_one_sided": 0.04,
                        "delta_ci": [0.002, 0.05],
                        "n_compared": 349,
                        "fixed": 15,
                        "broken": 6,
                        "panel_baseline": {"preprocess": 0.8},
                        "panel_candidate": {"preprocess": 0.85},
                        "edits": [
                            {
                                "failure_class": "reference_resolution",
                                "change_kind": "rule",
                                "n_diagnoses": 7,
                                "rationale": "<b>bold</b> claim",
                            }
                        ],
                        "diff": "@@ -1 +1 @@\n-old\n+new\n",
                    }
                ],
            }
        ],
    )
    html = render_sdk_page(populated)
    assert "+2.50pp" in html and "$4.25" in html and "73.5%" in html
    assert "reference_resolution" in html and 'class="add"' in html
    assert "&lt;b&gt;bold" in html  # escaped
    assert '"sdk_champion": "sdk_v2"' in html and '"n_sdk_experiments": 1' in html


def test_index_page_links_to_the_sdk_page() -> None:
    from convfinqa.evalloop.story_page import render_page

    assert 'href="agent-sdk.html"' in render_page(_story())


def test_story_check_covers_the_sdk_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from convfinqa.evalloop import story, story_check
    from convfinqa.evalloop.story_page import render_page, render_sdk_page
    from convfinqa.tracking import registry

    docs = tmp_path / "docs"
    docs.mkdir()
    story_path = tmp_path / "story.json"
    monkeypatch.setattr(story_check, "STORY_PATH", story_path)
    monkeypatch.setattr(story_check, "DOCS_DIR", docs)
    monkeypatch.setattr(story, "STORY_PATH", story_path)
    monkeypatch.setattr(registry, "champion", lambda path=None: "v8")
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v1")

    data = _story(sdk_champion="sdk_v1")
    text = json.dumps(data, indent=1) + "\n"
    story_path.write_text(text)
    (docs / "story.json").write_text(text)
    (docs / "index.html").write_text(render_page(data))
    # Missing SDK page is staleness…
    assert any("agent-sdk.html" in p for p in story_check.problems())
    # …and a rendered one is accepted.
    (docs / "agent-sdk.html").write_text(render_sdk_page(data))
    assert story_check.problems() == []
    # A registry that moved sdk_champion after the build is caught.
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v2")
    found = story_check.problems()
    assert any("sdk_champion" in p for p in found)


# ── run_cycle(runtime="agent_sdk") end to end ───────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("past_rejections", [0, 2])
async def test_sdk_cycle_runs_the_steps_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, past_rejections: int
) -> None:
    """draw → diagnose → rank → propose → gate → decide, and max_areas after
    two consecutive rejections."""
    from convfinqa.evalloop import campaign, cycle, sdk_gate
    from convfinqa.evalloop import runner as runner_mod
    from convfinqa.tracking import registry

    calls: list[str] = []
    seen: dict[str, Any] = {}
    past = [_exp(False) for _ in range(past_rejections)]
    monkeypatch.setattr(campaign, "history", lambda c, **k: list(past))
    if past_rejections >= campaign.max_experiments("agent_sdk"):
        # Under the SDK cap of 2 a campaign is over by the time it has two
        # rejections, so `single_area_mode` is only reachable if the cap is
        # raised. Raise it here: what this case pins is the wiring that passes
        # `max_areas=1`, not the reachability of the mode under today's cap.
        monkeypatch.setattr(
            campaign, "MAX_EXPERIMENTS_BY_RUNTIME", {"pipeline": 5, "agent_sdk": 5}
        )
    monkeypatch.setattr(campaign, "summarise", lambda c: {"campaign": c})
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v1")

    async def run_split(split: str, version: str, **kw: Any) -> dict[str, Any]:
        calls.append(f"run:{split}:{version}")
        assert kw["runtime"] == "agent_sdk"
        if split == "train":
            assert kw["stop_at_first_wrong"] is True and kw["train_seed"] is not None
            seen["train_seed"] = kw["train_seed"]
        else:
            assert "stop_at_first_wrong" not in kw
        return {"csv": str(tmp_path / f"{split}-{version}.csv"), "run_id": "eval-1"}

    monkeypatch.setattr(runner_mod, "run_split", run_split)

    ranking = {
        "reference_resolution": {"faults": 9, "n": 30, "wilson_lower": 0.16, "rank": 1},
        "unit_scale": {"faults": 2, "n": 30, "wilson_lower": 0.01, "rank": 2},
    }

    async def diagnose_run(csv: str, version: str, **kw: Any) -> dict[str, Any]:
        calls.append("diagnose")
        assert kw["campaign"] == "s01" and kw["label"] == f"s01-e{len(past) + 1:02d}"
        return {
            "run_id": "diag-1",
            "diagnoses_path": str(tmp_path / "d.jsonl"),
            "n_cases": 11,
            "counts": {"preprocess": 9, "retriever": 2},
            "labels": {"reference_resolution": 9, "unit_scale": 2},
        }

    def rank_classes(version: str, **kw: Any) -> dict[str, Any]:
        calls.append("rank")
        assert version == "sdk_v1"
        return ranking

    async def propose_version(path: str, **kw: Any) -> dict[str, Any]:
        calls.append("propose")
        seen["propose"] = kw
        assert kw["base_version"] == "sdk_v1" and kw["new_version"].startswith("sdk_v")
        assert kw["pooled"] is ranking
        return {
            "rewrite_id": "rw-9",
            "new_version": kw["new_version"],
            "edits": [{"failure_class": "reference_resolution"}],
        }

    monkeypatch.setattr(
        cycle,
        "_sdk_teacher",
        lambda: SimpleNamespace(
            diagnose_run=diagnose_run,
            rank_classes=rank_classes,
            propose_version=propose_version,
        ),
    )

    comparison = _comparison(True)

    def gate_overall(base_csv: str, cand_csv: str, **kw: Any) -> tuple[dict, Any]:
        calls.append("gate")
        assert kw["target_class"] == "reference_resolution"
        return (
            {
                "runtime": "agent_sdk",
                "target_class": kw["target_class"],
                "baseline_version": kw["baseline_version"],
                "candidate_version": kw["candidate_version"],
                "evidence_split": "test",
                "promotable": True,
                "reason": "PROMOTE",
                "comparison": {},
            },
            comparison,
        )

    def log_gate_verdict(verdict: dict, **kw: Any) -> str:
        calls.append("log")
        seen["log"] = kw
        return "gate-run-1"

    monkeypatch.setattr(sdk_gate, "gate_overall", gate_overall)
    monkeypatch.setattr(sdk_gate, "log_gate_verdict", log_gate_verdict)

    def promote_sdk(version: str, **kw: Any) -> Any:
        calls.append("promote")
        seen["promote"] = kw
        return SimpleNamespace(as_dict=lambda: {"promoted": True, "version": version})

    monkeypatch.setattr(registry, "promote_sdk", promote_sdk)

    steps = await cycle.run_cycle(campaign="s01", runtime="agent_sdk", concurrency=2)

    challenger = steps["proposal"]["new_version"]
    assert calls == [
        "run:train:sdk_v1",
        "diagnose",
        "rank",
        "propose",
        "run:test:sdk_v1",
        f"run:test:{challenger}",
        "gate",
        "log",
        "promote",
    ]
    assert steps["runtime"] == "agent_sdk"
    assert steps["target"]["failure_class"] == "reference_resolution"
    assert seen["train_seed"] == 2026 + past_rejections
    assert seen["log"]["rewrite_id"] == "rw-9"
    assert seen["log"]["champion_after"] == challenger
    assert seen["log"]["consecutive_rejections"] == 0
    assert seen["promote"]["comparison"] is comparison
    assert seen["promote"]["evidence_split"] == "test"
    assert steps["single_area_mode"] is (past_rejections == 2)
    # The one guardrail the rejections buy: one area per cycle from now on.
    assert seen["propose"]["max_areas"] == (1 if past_rejections == 2 else None)


@pytest.mark.asyncio
async def test_sdk_cycle_reuses_the_baseline_gate_csv_and_records_a_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from convfinqa.evalloop import campaign, cycle, sdk_gate
    from convfinqa.evalloop import runner as runner_mod
    from convfinqa.tracking import registry

    # One prior rejection: under the SDK cap of 2 this is the last slot, and it
    # keeps the streak this test asserts on (that rejection plus this one).
    past = [_exp(False)]
    monkeypatch.setattr(campaign, "history", lambda c, **k: list(past))
    monkeypatch.setattr(campaign, "summarise", lambda c: {})
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: None)
    monkeypatch.setattr("convfinqa.prompts.latest_sdk", lambda: "sdk_v1")
    splits: list[str] = []

    async def run_split(split: str, version: str, **kw: Any) -> dict[str, Any]:
        splits.append(f"{split}:{version}")
        return {"csv": f"{split}.csv"}

    monkeypatch.setattr(runner_mod, "run_split", run_split)

    async def diagnose_run(*a: Any, **k: Any) -> dict[str, Any]:
        return {"run_id": "d", "diagnoses_path": "d.jsonl", "n_cases": 3, "counts": {}}

    async def propose_version(*a: Any, **k: Any) -> dict[str, Any]:
        return {"rewrite_id": "rw-2", "new_version": k["new_version"], "edits": []}

    monkeypatch.setattr(
        cycle,
        "_sdk_teacher",
        lambda: SimpleNamespace(
            diagnose_run=diagnose_run,
            rank_classes=lambda v, **k: {
                "unit_scale": {"faults": 3, "n": 3, "rank": 1}
            },
            propose_version=propose_version,
        ),
    )
    monkeypatch.setattr(
        sdk_gate,
        "gate_overall",
        lambda b, c, **kw: (
            {
                "baseline_version": kw["baseline_version"],
                "candidate_version": kw["candidate_version"],
                "evidence_split": "test",
                "promotable": False,
                "reason": "REJECT",
                "comparison": {},
                "target_class": kw["target_class"],
            },
            None,
        ),
    )
    logged: dict[str, Any] = {}
    monkeypatch.setattr(
        sdk_gate, "log_gate_verdict", lambda v, **kw: logged.update(kw) or "g"
    )

    def _no_promote(*a: Any, **k: Any) -> None:
        raise AssertionError("a rejection must not reach promote_sdk")

    monkeypatch.setattr(registry, "promote_sdk", _no_promote)

    steps = await cycle.run_cycle(
        campaign="s01", runtime="agent_sdk", baseline_gate_csv="reused.csv"
    )
    assert splits == ["train:sdk_v1", f"test:{steps['proposal']['new_version']}"]
    assert steps["baseline_gate_run"] == {"csv": "reused.csv", "reused": True}
    assert steps["promotion"] == {"promoted": False, "champion_retained": "sdk_v1"}
    assert logged["consecutive_rejections"] == 2  # one before, plus this one
    assert logged["champion_after"] == "sdk_v1"


@pytest.mark.asyncio
async def test_pipeline_cycle_refuses_an_sdk_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from convfinqa.evalloop import campaign, cycle

    monkeypatch.setattr(campaign, "history", lambda c, **k: [])
    with pytest.raises(SystemExit, match="agent_sdk"):
        await cycle.run_cycle(campaign="c09", baseline_version="sdk_v1")
    with pytest.raises(SystemExit, match="not an sdk_vN"):
        await cycle.run_cycle(
            campaign="s09", baseline_version="v8", runtime="agent_sdk"
        )


# ── sdk_gate ────────────────────────────────────────────────────────────


class _FakeRun:
    info = SimpleNamespace(run_id="sdk-gate-r1")

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class _FakeMlflow:
    def start_run(self, run_name: str = "") -> _FakeRun:
        return _FakeRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        return None

    def log_params(self, params: dict[str, str]) -> None:
        return None

    def get_experiment_by_name(self, name: str) -> object:
        return object()

    def set_experiment(self, name: str) -> None:
        return None


def test_sdk_gate_judges_overall_accuracy_and_stamps_the_ledger_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skipping `teacher.log_gate_verdict` is deliberate: its ledger row says
    `multi_agent`, and a gates row is append-only."""
    import pandas as pd

    from convfinqa.evalloop import ledgers, sdk_gate, stage_scores
    from convfinqa.tracking import mlflow_log, registry

    monkeypatch.setenv(ledgers.LEDGER_DIR_ENV, str(tmp_path / "ledgers"))
    monkeypatch.setattr(mlflow_log, "_mlflow", lambda: _FakeMlflow())
    monkeypatch.setattr(stage_scores, "report_documents", lambda: {})
    monkeypatch.setattr(sdk_gate, "sdk_prompt_hash", lambda v: f"{v}-hash")
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v1")

    def _row(rid: str, ok: bool) -> dict[str, Any]:
        return {
            "report_id": rid, "turn_index": 0, "question": "q", "gold_answer": "132",
            "pred_answer": "132" if ok else "0", "correct": ok,
            "gold_turn_type": "Program", "pred_turn_type": "program",
            "gold_program": "subtract(243, 111)", "pred_program": "subtract(A, B)",
            "split": "test", "run_id": "eval-" + rid,
            "pred_sub_questions": json.dumps(["x", "y"]),
            "retriever_io": json.dumps({"output": {"answers": [
                {"question": "q", "answer": v} for v in ("243", "111")]}}),
        }  # fmt: skip

    base = pd.DataFrame([_row("a", False), _row("b", True), _row("c", True)])
    cand = pd.DataFrame([_row("a", True), _row("b", True), _row("c", True)])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    base.to_csv(a, index=False)
    cand.to_csv(b, index=False)
    verdict, comparison = sdk_gate.gate_overall(
        a, b, baseline_version="sdk_v1", candidate_version="sdk_v2",
        target_class="reference_resolution",
    )  # fmt: skip
    assert verdict["runtime"] == "agent_sdk" and verdict["target_agent"] == ""
    assert verdict["target_class"] == "reference_resolution"
    assert verdict["target_metric"] == "accuracy" and verdict["target_moved"]
    assert "retriever_operand_recall" in verdict["agent_panel_candidate"]
    assert verdict["promotable"] == verdict["comparison"]["promotable"]

    run_id = sdk_gate.log_gate_verdict(
        verdict, comparison=comparison, campaign="s01", label="s01-e02",
        rewrite_id="rw-7", consecutive_rejections=0, champion_after="sdk_v2",
    )  # fmt: skip
    assert run_id == "sdk-gate-r1"
    table = ledgers.load("gates")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["runtime"] == "agent_sdk"
    assert (row["campaign"], row["experiment_n"], row["rewrite_id"]) == (
        "s01",
        2,
        "rw-7",
    )
    assert (row["baseline_hash"], row["candidate_hash"]) == (
        "sdk_v1-hash",
        "sdk_v2-hash",
    )
    assert row["champion_after"] == "sdk_v2" and row["gate_run_id"] == "sdk-gate-r1"
    assert (row["fixed"], row["broken"]) == (1, 0)
    assert ledgers.load("gates", runtime="multi_agent").empty


def test_sdk_experiment_cap_is_two_and_pipeline_keeps_five() -> None:
    """The cap is per runtime: 2 for the SDK arm, 5 for the pipeline.

    Pinned because it is the one thing standing between a campaign and an
    unbounded spend of subscription time against a gate split that has already
    been read many times.
    """
    from convfinqa.evalloop import campaign

    assert campaign.max_experiments("agent_sdk") == 2
    assert campaign.max_experiments("pipeline") == 5
    assert campaign.max_experiments(None) == campaign.MAX_EXPERIMENTS

    sdk_past = [
        {"label": f"s01-e{i:02d}", "runtime": "agent_sdk", "promoted": False}
        for i in range(2)
    ]
    with pytest.raises(SystemExit, match="cap for the 'agent_sdk' runtime is 2"):
        campaign.check_capacity("s01", sdk_past, runtime="agent_sdk")
    campaign.check_capacity("s01", sdk_past[:1], runtime="agent_sdk")

    # The same history length is still fine for the pipeline.
    pipeline_past = [
        {"label": f"c01-e{i:02d}", "runtime": "multi_agent", "promoted": False}
        for i in range(2)
    ]
    campaign.check_capacity("c01", pipeline_past, runtime="pipeline")


def test_sdk_capacity_cap_is_inferred_from_history_when_runtime_omitted() -> None:
    """An existing caller that passes no runtime still gets the right cap."""
    from convfinqa.evalloop import campaign

    sdk_past = [
        {"label": f"s01-e{i:02d}", "runtime": "agent_sdk", "promoted": False}
        for i in range(2)
    ]
    with pytest.raises(SystemExit, match="is 2"):
        campaign.check_capacity("s01", sdk_past)


def test_turn_type_gate_splits_a_paired_comparison(tmp_path: Path) -> None:
    """The per-slice verdict is computed from the committed CSVs, not the aggregate.

    Pinned because the aggregate delta averages two populations that behave
    differently: a reader given only the headline attributes the gain to both.
    """
    import pandas as pd

    from convfinqa.evalloop import story

    def csv(path: Path, correct: list[bool]) -> Path:
        pd.DataFrame(
            {
                "report_id": ["r1", "r1", "r2", "r2"],
                "turn_index": [0, 1, 0, 1],
                "gold_turn_type": ["Number", "Program", "Number", "Program"],
                "correct": correct,
            }
        ).to_csv(path, index=False)
        return path

    base = csv(tmp_path / "b.csv", [True, False, True, False])
    cand = csv(tmp_path / "c.csv", [True, True, True, True])
    out = story.turn_type_gate(base, cand)
    assert out is not None
    assert out["number"]["delta_pp"] == 0.0
    assert out["number"]["fixed"] == 0 and out["number"]["broken"] == 0
    assert out["program"]["fixed"] == 2 and out["program"]["broken"] == 0
    assert out["program"]["delta_pp"] == 100.0
    assert out["program"]["n"] == 2
    # A missing CSV yields None, so the page says "not yet run" rather than zero.
    assert story.turn_type_gate(tmp_path / "nope.csv", cand) is None


def test_arm_carries_the_turn_type_split_and_absence_stays_none() -> None:
    """Each arm reports number/program accuracy; an unrun arm reports neither."""
    from convfinqa.evalloop import story

    record = {
        "run_name": "sdk-evalloop-test100-sdk_v1",
        "params": {"prompts_version": "sdk_v1", "runtime": "agent_sdk"},
        "metrics": {
            "accuracy": 0.9,
            "accuracy_gold_turn_type_Number": 0.95,
            "accuracy_gold_turn_type_Program": 0.88,
        },
    }
    arm = story._arm(record)
    assert arm["by_turn_type"] == {"number": 0.95, "program": 0.88}
    assert story._arm(None)["by_turn_type"] is None
