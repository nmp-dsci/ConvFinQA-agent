"""The whole record on the gate split (s21): every test pass against the champion of its day.

Built from plain records, gate rows and registry history — no tracking server,
no CSVs. The timeline is what the Optimise › Story tab and the Runs ledger draw.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from convfinqa.evalloop import story

T0 = datetime(2026, 9, 3, 15, 0, tzinfo=UTC)


def _ms(minutes: int) -> int:
    return int((T0 + timedelta(minutes=minutes)).timestamp() * 1000)


def _iso(minutes: int) -> str:
    return (T0 + timedelta(minutes=minutes)).isoformat()


def _run(
    name: str,
    minutes: int,
    version: str,
    accuracy: float | None,
    *,
    split: str = "test",
    run_id: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    **metrics: Any,
) -> dict[str, Any]:
    params: dict[str, Any] = {"split": split, "prompts_version": version}
    if version.startswith("sdk_"):
        params["runtime"] = "agent_sdk"
        params["sdk_model"] = model or "claude-sonnet-5"
        if effort:
            params["sdk_effort"] = effort
    else:
        params["lm_mini"] = "deepseek-v4-flash"
    return {
        "run_name": name,
        "run_id": run_id or name,
        "start_time": _ms(minutes),
        "params": params,
        "metrics": {
            **({"accuracy": accuracy} if accuracy is not None else {}),
            **metrics,
        },
    }


V2 = "evalloop-test100-v2·t2p2r2c2-20260904_032533"
V8 = "evalloop-test100-v8·t2p2r5c2-20260904_052739"
CUT = "sdk-evalloop-test100-sdk_v1·s1-20260905_132411"
S1 = "sdk-evalloop-test100-sdk_v1·s1-20260905_174627"
S2 = "sdk-evalloop-test100-sdk_v2·s2-20260905_182812"
HAIKU = "sdk-evalloop-test100-sdk_v1·s1-haiku-5-5-high-20261015_175310"

RECORDS = [
    _run(V2, 0, "v2", 0.770774),
    _run(V8, 120, "v8", 0.816619),
    _run(CUT, 600, "sdk_v1", 0.444126),
    _run(S1, 860, "sdk_v1", 0.905444),
    _run(S2, 900, "sdk_v2", 0.876791),
    _run(HAIKU, 60000, "sdk_v1", 0.873926, model="claude-haiku-5-5", effort="high"),
    # Still running: no accuracy yet, so no point and no row.
    _run("sdk-evalloop-test100-sdk_v3·s3-haiku-5-5-high-20261015_205249", 60100,
         "sdk_v3", None, model="claude-haiku-5-5", effort="high"),
    _run("sdk-evalloop-train100-sdk_v1·s1-20260905_180924", 870, "sdk_v1", 0.885,
         split="train"),
]  # fmt: skip

GATES = [
    {"runtime": "multi_agent", "campaign": "c01", "experiment_n": 3,
     "baseline_eval_run_id": V2, "candidate_eval_run_id": V8,
     "baseline_version": "v2", "candidate_version": "v8", "promoted": True,
     "delta_pp": 4.5845, "p_value": 0.040428, "baseline_acc": 0.770774,
     "candidate_acc": 0.816619},
    {"runtime": "agent_sdk", "campaign": "s01", "experiment_n": None,
     "baseline_eval_run_id": V8, "candidate_eval_run_id": S1,
     "baseline_version": "v8", "candidate_version": "sdk_v1", "promoted": True,
     "delta_pp": 8.8825, "p_value": 0.000286, "baseline_acc": 0.816619,
     "candidate_acc": 0.905444},
    {"runtime": "agent_sdk", "campaign": "s01", "experiment_n": 2,
     "baseline_eval_run_id": S1, "candidate_eval_run_id": S2,
     "baseline_version": "sdk_v1", "candidate_version": "sdk_v2", "promoted": False,
     "delta_pp": -2.8653, "p_value": 0.917, "baseline_acc": 0.905444,
     "candidate_acc": 0.876791, "panel_baseline": '{"acc_triage_turn_type": 0.97}'},
]  # fmt: skip

HISTORY = [
    {"event": "promote", "version": "v2", "at": _iso(-60)},
    # v8's promotion is written after its own pass ran.
    {"event": "promote", "version": "v8", "at": _iso(150)},
    # sdk_v1's is written at 08:01 UTC, after sdk_v2's pass at 06:28: placed by
    # the event, sdk_v2 would be drawn against v8.
    {"event": "promote_sdk", "alias": "sdk_champion", "version": "sdk_v1", "at": _iso(990)},
    {"event": "promote_judge", "version": "judge_j1", "at": _iso(1000)},
    {"event": "universal_champion", "version": "sdk_v1", "at": _iso(59000),
     "champion_agent_after": {"version": "sdk_v1", "model": "claude-sonnet-5",
                              "effort": "high"}},
    # A pipeline promotion after the agent leads does not move the line.
    {"event": "promote", "alias": "pipeline_champion", "version": "v9", "at": _iso(59500)},
]  # fmt: skip

CHAMPION = {
    "version": "sdk_v1",
    "model": "claude-sonnet-5",
    "effort": "high",
    "evidence_run": S1,
    "accuracy": 0.905444,
}

CHALLENGERS = {
    "pairs": [
        {"candidate_run": HAIKU, "baseline_version": "sdk_v1",
         "baseline_model": "claude-sonnet-5", "delta_pp": -3.1519,
         "p_worse": 0.0429, "p_better": 0.9571,
         "verdict": {"outcome": "falls_short", "reason": "falls_short: Δ -3.15pp"}},
    ]
}  # fmt: skip


@pytest.fixture(autouse=True)
def _sonnet_is_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(story, "_champion_record", lambda: dict(CHAMPION))


def _timeline() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps = story.champion_steps(HISTORY, RECORDS, CHAMPION)
    rows = story.test_timeline(RECORDS, GATES, challengers=CHALLENGERS, steps=steps)
    return steps, rows


def test_the_step_line_moves_only_when_the_agent_changes() -> None:
    steps, _ = _timeline()
    assert [s["version"] for s in steps] == ["v2", "v8", "sdk_v1", "sdk_v1"]
    assert [s["event"] for s in steps][-1] == "universal_champion"
    v8, s1 = steps[1], steps[2]
    # Anchored at the pass that measured it, not at the promotion's timestamp.
    assert v8["effective_at_ms"] == _ms(120) and v8["run_name"] == V8
    assert s1["effective_at_ms"] == _ms(860) and s1["run_name"] == S1
    assert s1["accuracy"] == pytest.approx(0.905444)
    assert (s1["model"], s1["effort"]) == ("claude-sonnet-5", "high")


def test_every_test_pass_is_a_row_against_the_champion_of_its_day() -> None:
    _, rows = _timeline()
    names = [r["run_name"] for r in rows]
    assert names == [V2, V8, CUT, S1, S2, HAIKU]  # oldest first; no train, no running
    by = {r["run_name"]: r for r in rows}

    assert by[V2]["outcome"] == "champion" and by[V2]["vs"] is None
    assert by[V8]["outcome"] == "promoted" and by[V8]["label"] == "c01-e03"
    # A pass that became champion is set against the one it beat.
    assert by[V8]["vs"]["champion_of_the_day"] == "v2"
    assert by[V8]["vs"]["paired"] is True and by[V8]["vs"]["delta_pp"] == 4.5845
    assert by[S1]["label"] == "s01 cross-runtime"
    assert by[S1]["vs"]["champion_of_the_day"] == "v8"
    # Ran before sdk_v1's promotion was written, after its pass: sdk_v1's day.
    assert by[S2]["outcome"] == "rejected"
    assert by[S2]["vs"]["champion_of_the_day"] == "sdk_v1"

    haiku = by[HAIKU]
    assert haiku["outcome"] == "falls_short"
    assert (haiku["model"], haiku["effort"]) == ("claude-haiku-5-5", "high")
    assert haiku["vs"]["delta_pp"] == -3.1519 and haiku["vs"]["p_value"] == 0.0429
    assert haiku["vs"]["p_kind"] == "one-sided toward worse"
    assert haiku["date"] == "2026-10-15"
    assert by[V2]["model"] == "deepseek-v4-flash" and by[V2]["effort"] is None


def test_the_pass_that_ran_out_of_subscription_is_never_compared() -> None:
    _, rows = _timeline()
    cut = next(r for r in rows if r["run_name"] == CUT)
    assert cut["complete"] is False and cut["outcome"] == "incomplete"
    assert cut["vs"] is None and "session limit" in cut["reason"]


def test_an_sdk_versions_accuracy_is_its_pass_on_the_champions_agent() -> None:
    """The live bug: keyed on the version, sdk_v1 read as the newest pass of it —
    87.39%, a Haiku run — beside the +8.88pp its Sonnet pass measured."""
    by_version = story._gate_runs_by_version(RECORDS)
    assert by_version["sdk_v1"]["accuracy"] == pytest.approx(0.905444)

    carried = [
        {"baseline_version": "sdk_v1", "candidate_version": "sdk_v2",
         "accuracy_baseline": 0.873926, "accuracy_candidate": 0.876791},
    ]  # fmt: skip
    fixed = story._with_gate_accuracies(carried, GATES)
    assert fixed[0]["accuracy_baseline"] == pytest.approx(0.905444)
    assert fixed[0]["panel_baseline"]["triage"] == pytest.approx(0.97)


def test_train_passes_stay_off_the_test_axis() -> None:
    train = story.train_passes(RECORDS)
    assert [t["run_name"] for t in train] == [
        "sdk-evalloop-train100-sdk_v1·s1-20260905_180924"
    ]
    assert train[0]["date"] == "2026-09-05"


def test_story_check_fails_a_page_that_hides_the_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import json

    from convfinqa.evalloop import story_check
    from convfinqa.evalloop.story_page import render_page, render_sdk_page
    from convfinqa.tracking import registry

    docs = tmp_path / "docs"
    docs.mkdir()
    story_path = tmp_path / "story.json"
    monkeypatch.setattr(story_check, "STORY_PATH", story_path)
    monkeypatch.setattr(story_check, "DOCS_DIR", docs)
    monkeypatch.setattr(registry, "champion", lambda path=None: "sdk_v1")
    monkeypatch.setattr(registry, "sdk_champion", lambda path=None: "sdk_v1")
    monkeypatch.setattr(registry, "judge_champion", lambda path=None: None)
    monkeypatch.setattr(registry, "champion_agent", lambda path=None: None)
    steps, rows = _timeline()
    data = {
        "champion": "sdk_v1",
        "sdk_champion": "sdk_v1",
        "campaigns": [],
        "sdk_campaigns": [],
        "test_timeline": rows,
        "champion_steps": steps,
    }
    text = json.dumps(data, indent=1) + "\n"
    story_path.write_text(text)
    (docs / "story.json").write_text(text)
    (docs / "agent-sdk.html").write_text(render_sdk_page(data))
    page = render_page(data)
    (docs / "index.html").write_text(page)
    assert not [p for p in story_check.problems() if "test passes" in p]
    assert "Every test pass on the gate split" in page
    assert "incomplete — not compared" in page and "falls short" in page

    # A page built before the timeline existed is caught.
    (docs / "index.html").write_text(render_page({**data, "test_timeline": []}))
    assert any("6 test passes" in p for p in story_check.problems())


def test_a_gated_challenger_reads_as_its_verdict_not_the_house_gate() -> None:
    """A challenger cycle (s21) writes a gates row too; the row's rule was
    s19's, so the pass falls short — it was never "rejected" by the house gate."""
    v3 = "sdk-evalloop-test100-sdk_v3·s3-haiku-5-5-high-20261015_205249"
    records = [*RECORDS[:6], _run(v3, 60100, "sdk_v3", 0.856734,
                                  model="claude-haiku-5-5", effort="high")]  # fmt: skip
    gates = [
        *GATES,
        {"runtime": "agent_sdk", "campaign": "h01", "experiment_n": 1,
         "baseline_eval_run_id": S1, "candidate_eval_run_id": v3,
         "baseline_version": "sdk_v1", "candidate_version": "sdk_v3",
         "promoted": False, "gate_passed": False, "verdict_rule": "model_swap",
         "verdict_outcome": "falls_short", "delta_pp": -4.8711, "p_value": 0.989},
    ]  # fmt: skip
    challengers = {
        "pairs": [
            *CHALLENGERS["pairs"],
            {"candidate_run": v3, "baseline_version": "sdk_v1",
             "baseline_model": "claude-sonnet-5", "delta_pp": -4.8711,
             "p_worse": 0.0109, "p_better": 0.9891,
             "verdict": {"outcome": "falls_short", "reason": "falls_short"}},
        ]
    }  # fmt: skip
    steps = story.champion_steps(HISTORY, records, CHAMPION)
    rows = story.test_timeline(records, gates, challengers=challengers, steps=steps)
    row = next(r for r in rows if r["run_name"] == v3)
    assert row["outcome"] == "falls_short" and row["label"] == "h01-e01"
    assert row["vs"]["p_value"] == 0.0109
    assert row["vs"]["p_kind"] == "one-sided toward worse"
