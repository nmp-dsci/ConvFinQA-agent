"""CLI for the eval loop: splits, runs, the gate, and the M2 teacher.

    convfinqa-evalloop make-splits
    convfinqa-evalloop run --split train --version v2 --n-reports 10
    convfinqa-evalloop run --split train --version v2 --n-questions 50
    convfinqa-evalloop run --split test --version sdk_v1 --runtime agent_sdk \
        --resume-from <partial.csv>          # finish a rate-limited pass
    convfinqa-evalloop gate --baseline-csv A.csv --candidate-csv B.csv \
        --baseline-version v2 --candidate-version v3_1 --promote
    convfinqa-evalloop diagnose --csv <run.csv> --version v3_1
    convfinqa-evalloop propose --diagnoses <d.jsonl> --base-version v3_1 --new-version v4
    convfinqa-evalloop gate-targeted --target-agent triage ...

The single-session arm (s10) runs the same commands with ``--runtime agent_sdk``
and ``sdk_vN`` versions; ``gate-targeted`` then takes ``--target-class``:

    convfinqa-evalloop sdk-distil --source-version v8 --new-version sdk_v1
    convfinqa-evalloop cycle --campaign s01 --runtime agent_sdk
    convfinqa-evalloop diagnose --csv <run.csv> --version sdk_v1 --runtime agent_sdk
    convfinqa-evalloop propose --diagnoses <d.jsonl> --base-version sdk_v1 \
        --new-version sdk_v2 --runtime agent_sdk
    convfinqa-evalloop gate-targeted --runtime agent_sdk --target-class <label> ...
    convfinqa-evalloop backfill-ledgers [--no-mlflow]
    convfinqa-evalloop backfill-sdk-cost [--dry-run]
    convfinqa-evalloop ledger-trace --question-id <id> | --edit-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

RUNTIMES = ("pipeline", "agent_sdk")
RUNTIME_HELP = (
    "Which arm: the four-agent pipeline (vN bundles) or the single-session "
    "Agent SDK runtime (sdk_vN prompts). A version from the other arm is refused."
)


def _add_runtime(
    parser: argparse.ArgumentParser, help_text: str = RUNTIME_HELP
) -> None:
    parser.add_argument(
        "--runtime", default="pipeline", choices=RUNTIMES, help=help_text
    )


def _add_sdk_agent(parser: argparse.ArgumentParser, what: str) -> None:
    """--sdk-model / --sdk-effort: the agent an agent_sdk step is about."""
    parser.add_argument(
        "--sdk-model",
        default=None,
        metavar="MODEL",
        help=f"agent_sdk only: the model {what} (default: the champion record's).",
    )
    parser.add_argument(
        "--sdk-effort",
        default=None,
        choices=("low", "medium", "high", "xhigh", "max"),
        help=f"agent_sdk only: the effort {what} (default: the champion record's).",
    )


def check_runtime(
    ap: argparse.ArgumentParser, runtime: str, *versions: str | None
) -> None:
    """Fail fast when a version name and `--runtime` disagree.

    `sdk_vN` only runs under `agent_sdk`, a `vN` bundle only under `pipeline`.
    Letting the mismatch through would either build four agents from a prompt
    that has none, or hand a session a bundle it cannot read — both far into a
    paid run before anything notices.
    """
    import convfinqa.prompts as prompts_pkg

    for version in versions:
        if not version:
            continue
        session = runtime in ("agent_sdk", "open_agent")
        if prompts_pkg.is_sdk_version(version) != session:
            wanted = "agent_sdk" if prompts_pkg.is_sdk_version(version) else "pipeline"
            ap.error(
                f"version {version!r} belongs to --runtime {wanted}, not "
                f"{runtime!r}: sdk_vN prompts run under agent_sdk (or open_agent), "
                "vN bundles under pipeline"
            )


def _sdk_teacher() -> Any:
    """`evalloop.sdk_teacher`, imported only by the commands that need it."""
    import importlib

    return importlib.import_module("convfinqa.evalloop.sdk_teacher")


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI surface, separable from `main` so tests can parse it."""
    ap = argparse.ArgumentParser(
        prog="convfinqa-evalloop",
        description="The eval loop: splits → run → trace → score → gate.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-splits", help="Build and commit the split manifest.")
    mk.add_argument(
        "--force", action="store_true", help="Overwrite an existing manifest."
    )
    mk.add_argument("--target-questions", type=int, default=200)
    mk.add_argument("--seed", type=int, default=2026)
    mk.add_argument("--name", default=None, help="Manifest name, e.g. eval_loop_v2.")
    mk.add_argument(
        "--extend",
        default=None,
        help="Parent manifest to extend; the new splits are supersets of its.",
    )
    mk.add_argument(
        "--train-reports",
        type=int,
        default=None,
        help="Allocate train by report count (report-count mode).",
    )
    mk.add_argument(
        "--test-reports",
        type=int,
        default=None,
        help="Allocate the gate split by report count (report-count mode).",
    )

    rn = sub.add_parser("run", help="Run one split × version pass (an MLflow run).")
    rn.add_argument("--split", default="train", choices=("train", "test", "holdout"))
    rn.add_argument("--version", required=True, help="Prompt version, e.g. v2.")
    rn.add_argument(
        "--n-reports", type=int, default=None, help="Truncate by report count."
    )
    rn.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help=(
            "Truncate by cumulative question count instead of report count "
            "(walks the split in manifest order until the budget is met). "
            "Mutually exclusive with --n-reports."
        ),
    )
    rn.add_argument("--concurrency", type=int, default=8)
    rn.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Draw a fresh train split from pool-minus-gate with this seed.",
    )
    rn.add_argument(
        "--report-ids-file",
        default=None,
        metavar="FILE",
        help=(
            "With --train-seed: run only the drawn ids listed in FILE (one per "
            "line), in draw order. Every listed id must be in the draw."
        ),
    )
    rn.add_argument(
        "--stop-at-first-wrong",
        action="store_true",
        help="End each conversation at its first wrong answer (train only).",
    )
    rn.add_argument("--campaign", default=None)
    rn.add_argument("--label", default=None, help="Experiment label, e.g. c01-e02.")
    rn.add_argument(
        "--resume-from",
        default=None,
        metavar="CSV",
        help=(
            "Finish a pass that was cut short (a rate limit, say). "
            "Conversations the prior CSV answered whole are copied through "
            "verbatim; every other conversation of the split is run again from "
            "turn 0. Split, version, runtime and reports must match."
        ),
    )
    rn.add_argument(
        "--runtime",
        default="pipeline",
        choices=("pipeline", "agent_sdk", "open_agent"),
        help=(
            "Who walks the conversations: the four-agent pipeline (a vN bundle), "
            "one Claude Agent SDK session per conversation, or (s25) the same "
            "sdk_vN contract on an open model in an open harness. "
            "--version sdk_vN requires agent_sdk or open_agent."
        ),
    )
    from convfinqa.llm import OPEN_EFFORTS, OPEN_HARNESSES, OPEN_MODELS

    rn.add_argument(
        "--open-harness",
        default=None,
        choices=OPEN_HARNESSES,
        help="open_agent only: the harness (default: pydantic_ai).",
    )
    rn.add_argument(
        "--open-model",
        default=None,
        choices=OPEN_MODELS,
        help=(
            "open_agent only: the DeepSeek model (default: deepseek-flash). "
            "Goes into the run name as pyd-<model>."
        ),
    )
    rn.add_argument(
        "--open-effort",
        default=None,
        choices=OPEN_EFFORTS,
        help=(
            "open_agent only: DeepSeek's reasoning effort (default: high). Part "
            "of what the agent is, so it goes into the run name."
        ),
    )
    rn.add_argument(
        "--sdk-model",
        default=None,
        metavar="MODEL",
        help=(
            "Model for --runtime agent_sdk, e.g. claude-haiku-4-5-20251001 "
            "(default: SDK_MODEL, else llm.LM_SDK_MODEL). The prompt, tools and "
            "split stay the same, so two passes differing only here compare the "
            "models. Its slug goes into the run name; the id is the sdk_model "
            "param and tag."
        ),
    )
    rn.add_argument(
        "--sdk-effort",
        default=None,
        choices=("low", "medium", "high", "xhigh", "max"),
        help=(
            "Effort for --runtime agent_sdk (default: SDK_EFFORT, else the "
            "champion record's). Part of what the agent is, so it goes into the "
            "run name after the model slug and is the sdk_effort param and tag."
        ),
    )

    mk = sub.add_parser(
        "make-smoke",
        help=(
            "s25: pick the open-agent smoke set from the champion's train draws "
            "and write its manifest and one --report-ids-file per seed."
        ),
    )
    mk.add_argument("--name", default="smoke_v1")
    mk.add_argument("--tries", type=int, default=200_000)
    mk.add_argument(
        "--force", action="store_true", help="Overwrite an existing manifest."
    )
    sc = sub.add_parser(
        "smoke-compare",
        help="s25: a smoke pass beside the champion's records, per question and conversation.",
    )
    sc.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="SEED=CSV",
        help="One smoke pass per seed, e.g. 2027=evaluation/predictions/evalloop/<run>.csv",
    )
    sc.add_argument("--name", default="smoke_v1")
    sc.add_argument("--out", default=None, help="Also write the comparison here.")

    gt = sub.add_parser("gate", help="Paired comparison of two run CSVs.")
    gt.add_argument("--baseline-csv", required=True)
    gt.add_argument("--candidate-csv", required=True)
    gt.add_argument("--baseline-version", required=True)
    gt.add_argument("--candidate-version", required=True)
    gt.add_argument(
        "--promote",
        action="store_true",
        help="Promote the winner to champion via the registry.",
    )

    dg = sub.add_parser(
        "diagnose", help="Teacher: attribute each first-wrong question to a subagent."
    )
    dg.add_argument("--csv", required=True, help="Eval-loop predictions CSV.")
    dg.add_argument("--version", required=True, help="The version that produced it.")
    dg.add_argument("--experiment", default=None, help="MLflow experiment override.")
    dg.add_argument("--concurrency", type=int, default=8)
    dg.add_argument("--campaign", default=None)
    dg.add_argument("--label", default=None, help="Experiment label, e.g. s01-e02.")
    _add_runtime(dg)
    _add_sdk_agent(dg, "the CSV was answered on (default: the pass's own record)")

    pr = sub.add_parser(
        "propose", help="Write a challenger changing ONE subagent's prompt."
    )
    pr.add_argument(
        "--diagnoses",
        required=True,
        nargs="+",
        help="diagnoses_*.jsonl from diagnose (agent_sdk: one per train draw).",
    )
    pr.add_argument("--base-version", required=True)
    pr.add_argument("--new-version", required=True)
    pr.add_argument("--target", default=None, help="Subagent; default = most faults.")
    pr.add_argument("--campaign", default=None)
    pr.add_argument("--label", default=None)
    pr.add_argument(
        "--max-areas",
        type=int,
        default=None,
        help="agent_sdk only: cap the number of areas one rewrite may edit.",
    )
    _add_runtime(pr)
    _add_sdk_agent(pr, "the rewrite is written for")

    gtt = sub.add_parser(
        "gate-targeted",
        help="M2 gate: target subagent improved AND overall not regressed.",
    )
    gtt.add_argument("--baseline-csv", required=True)
    gtt.add_argument("--candidate-csv", required=True)
    gtt.add_argument("--baseline-version", required=True)
    gtt.add_argument("--candidate-version", required=True)
    gtt.add_argument(
        "--target-agent", default=None, help="pipeline: the subagent rewritten."
    )
    gtt.add_argument(
        "--target-class",
        default=None,
        help="agent_sdk: the failure class the rewrite addressed.",
    )
    _add_runtime(gtt)
    gtt.add_argument(
        "--baseline-diagnoses", default=None, help="Optional attribution evidence."
    )
    gtt.add_argument("--candidate-diagnoses", default=None)
    gtt.add_argument(
        "--promote",
        action="store_true",
        help="Promote the challenger when the targeted rule passes.",
    )
    gtt.add_argument(
        "--campaign", default=None, help="Count this verdict against a campaign."
    )
    gtt.add_argument("--label", default=None, help="Experiment label, e.g. s01-e02.")
    gtt.add_argument(
        "--rewrite-id", default=None, help="The rewrite this gate judges (rw-…)."
    )

    gch = sub.add_parser(
        "gate-challenger",
        help=(
            "agent_sdk: judge a challenger agent (another model or effort) "
            "against the champion's committed runs under the s19 rule."
        ),
    )
    gch.add_argument("--candidate-csv", required=True, help="Its test100 pass.")
    gch.add_argument("--candidate-version", required=True)
    gch.add_argument(
        "--candidate-train-csv",
        action="append",
        default=[],
        help="An early-stopped train draw of it on a champion seed (repeatable).",
    )
    _add_sdk_agent(gch, "the challenger ran on")
    gch.add_argument("--target-class", default=None)
    gch.add_argument(
        "--base-agent-csv",
        default=None,
        help="Its previous prompt's test pass on the same agent (information only).",
    )
    gch.add_argument("--campaign", default=None)
    gch.add_argument("--label", default=None)
    gch.add_argument("--rewrite-id", default=None)
    gch.add_argument(
        "--promote",
        action="store_true",
        help="Promote prompt, model and effort together when the rule passes.",
    )

    kp = sub.add_parser(
        "kappa",
        help="Teacher-vs-human agreement: build a labelling sheet, or score one.",
    )
    kp.add_argument("--make", action="store_true", help="Build a labelling sheet.")
    kp.add_argument(
        "--diagnoses", nargs="+", default=None, help="diagnoses_*.jsonl files."
    )
    kp.add_argument("--out", default=None)
    kp.add_argument("--n", type=int, default=30)
    kp.add_argument("--labels", default=None, help="A filled sheet to score.")

    rl = sub.add_parser(
        "release",
        help="Open the sealed holdout ONCE for the current champion (M3 gate).",
    )
    rl.add_argument("--baseline", default=None, help="Default: the released alias.")
    rl.add_argument("--n-reports", type=int, default=None)
    rl.add_argument("--concurrency", type=int, default=8)
    rl.add_argument(
        "--i-know-this-opens-the-holdout",
        action="store_true",
        dest="acknowledged",
        help="Required: opening the holdout consumes its unseen-ness.",
    )

    sub.add_parser(
        "backfill-prompts",
        help="Seed per-agent prompt lineages from the committed bundle modules.",
    )

    sp = sub.add_parser(
        "show-prompt",
        help="Reconstruct the prompts of a traced Agent SDK call from its refs.",
    )
    sp.add_argument("--trace", required=True, help="MLflow trace id (tr-…).")
    sp.add_argument(
        "--span",
        default=None,
        help="Only this span name; default is every agent_sdk span in the trace.",
    )

    bf = sub.add_parser(
        "backfill-flips",
        help="Attach flips.json to gate runs recorded before the gate wrote one.",
    )
    bf.add_argument("--experiment", default=None, help="MLflow experiment override.")
    bf.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute and check against each verdict, but write nothing.",
    )

    bc = sub.add_parser(
        "backfill-sdk-cost",
        help=(
            "Add sdk_session_cost_usd (the price paid) to sdk passes that only "
            "carry sdk_cost_usd, from their traces."
        ),
    )
    bc.add_argument(
        "--archive",
        default=None,
        help="The archived pre-2026-09-21 store (default .mlflow/store/mlflow.db).",
    )
    bc.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report, but write neither the snapshot nor a metric.",
    )

    ba = sub.add_parser(
        "backfill-attribution",
        help="Recompute past diagnose runs' fault counts under the current rule.",
    )
    ba.add_argument("--experiment", default=None, help="MLflow experiment override.")
    ba.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any metric.",
    )
    ba.add_argument(
        "--force",
        action="store_true",
        help="Recompute even runs already scored by the current rule.",
    )

    cy = sub.add_parser(
        "cycle",
        help="One full experiment: train -> diagnose -> rewrite -> gate -> decide.",
    )
    cy.add_argument("--campaign", required=True, help="Campaign name, e.g. c01.")
    cy.add_argument("--baseline-version", default=None, help="Default: the champion.")
    cy.add_argument("--new-version", default=None, help="Default: the next free vN.")
    cy.add_argument(
        "--target", default=None, help="Force a subagent (rotation still applies)."
    )
    cy.add_argument("--train-reports", type=int, default=100)
    cy.add_argument("--train-seed", type=int, default=None)
    cy.add_argument("--concurrency", type=int, default=8)
    cy.add_argument(
        "--baseline-gate-csv",
        default=None,
        help="Reuse a baseline gate run instead of re-running it.",
    )
    cy.add_argument(
        "--no-promote",
        action="store_true",
        help="Run the gate and record the verdict without moving the champion.",
    )
    _add_sdk_agent(
        cy,
        "every pass of the cycle runs on; another model or effort than the "
        "champion's makes it a challenger agent, judged against the champion's "
        "committed runs",
    )
    cy.add_argument(
        "--train-csv",
        action="append",
        default=None,
        help=(
            "agent_sdk: reuse a committed early-stopped train draw of this agent "
            "instead of drawing one (repeatable); refused unless its name, model, "
            "effort and completeness match."
        ),
    )
    cy.add_argument(
        "--rescore-seed",
        type=int,
        action="append",
        default=None,
        help=(
            "agent_sdk challenger: train seeds to re-score the rewrite on "
            "(default: every seed the champion has a draw on)."
        ),
    )
    cy.add_argument(
        "--diagnose-concurrency",
        type=int,
        default=None,
        help="Concurrency for diagnosis alone (default: --concurrency).",
    )
    cy.add_argument(
        "--smoke-reports",
        type=int,
        default=3,
        help="agent_sdk challenger: reports in the output-contract smoke pass (0 skips).",
    )
    _add_runtime(
        cy,
        RUNTIME_HELP
        + " Under agent_sdk the baseline defaults to sdk_champion, --target is a"
        " failure class, and a promotion moves sdk_champion.",
    )

    cs = sub.add_parser("campaign-status", help="Experiments used, promoted, blocked.")
    cs.add_argument("--campaign", required=True)

    st = sub.add_parser("story", help="Build the campaign story JSON and HTML page.")
    st.add_argument("--campaign", nargs="+", default=None, help="Campaigns to include.")
    st.add_argument("--out", default=None, help="Output directory for the page.")

    mp = sub.add_parser(
        "mirror-prompts",
        help="Mirror a bundle's four prompts into MLflow's prompt registry.",
    )
    mp.add_argument("--version", required=True)

    sd = sub.add_parser(
        "sdk-distil",
        help="Write the first single-session prompt (sdk_v1) from a bundle's four.",
    )
    sd.add_argument("--source-version", default="v8", help="Bundle to distil from.")
    sd.add_argument("--new-version", default="sdk_v1", help="sdk_vN module to write.")
    sd.add_argument("--experiment", default=None, help="MLflow experiment override.")

    # ── The confidence judge (s12) ────────────────────────────────────────
    jd = sub.add_parser(
        "judge-dataset",
        help="Cut the judge's optimise/calibrate/test splits from three run CSVs.",
    )
    jd.add_argument(
        "--optimise-csv", required=True, help="A train draw of the sdk champion."
    )
    jd.add_argument(
        "--calibrate-csv", required=True, help="A second, disjoint train draw."
    )
    jd.add_argument(
        "--test-csv", required=True, help="The gate-split pass of the sdk champion."
    )
    jd.add_argument("--seed", type=int, default=2026)
    jd.add_argument("--name", default="judge_v1")

    jdg = sub.add_parser(
        "judge-diagnose",
        help="Teacher (with gold): what each optimise case got wrong or right, for the judge.",
    )
    jdg.add_argument("--split", default="optimise", choices=("optimise", "calibrate"))
    jdg.add_argument(
        "--judge-scores",
        default=None,
        help="Round 2: a scores CSV; only the judge's misses are diagnosed, verdict attached.",
    )
    jdg.add_argument(
        "--judge-version", default="", help="The judge the scores came from."
    )
    jdg.add_argument("--concurrency", type=int, default=8)
    jdg.add_argument("--label", default=None)
    jdg.add_argument("--experiment", default=None)

    jds = sub.add_parser(
        "judge-distil",
        help="Write prompts/judge_jN.py from the diagnoses (revise with --base-version).",
    )
    jds.add_argument(
        "--new-version", default=None, help="judge_jN; default: next in line."
    )
    jds.add_argument(
        "--base-version",
        default=None,
        help="Revise this judge instead of starting fresh.",
    )
    jds.add_argument(
        "--round",
        type=int,
        default=None,
        help="Use only diagnoses from this round (default: every diagnosis on record).",
    )
    jds.add_argument("--experiment", default=None)

    jsc = sub.add_parser(
        "judge-score", help="Run a judge over one split; write the scores CSV."
    )
    jsc.add_argument("--version", required=True, help="judge_jN")
    jsc.add_argument(
        "--split", required=True, choices=("optimise", "calibrate", "test")
    )
    jsc.add_argument("--concurrency", type=int, default=8)
    jsc.add_argument("--error-target", type=float, default=0.01)
    jsc.add_argument("--label", default=None)
    jsc.add_argument("--experiment", default=None)

    jg = sub.add_parser(
        "judge-gate",
        help="Compare two judges' scores on the calibrate split; optionally promote.",
    )
    jg.add_argument("--baseline-scores", required=True)
    jg.add_argument("--candidate-scores", required=True)
    jg.add_argument("--baseline-version", required=True)
    jg.add_argument("--candidate-version", required=True)
    jg.add_argument("--error-target", type=float, default=0.01)
    jg.add_argument(
        "--promote", action="store_true", help="Move judge_champion on a pass."
    )

    bl = sub.add_parser(
        "backfill-ledgers",
        help="Seed diagnoses/rewrites/gates ledgers from the per-run files and MLflow.",
    )
    bl.add_argument(
        "--no-mlflow",
        action="store_true",
        help="File-derived rows only; do not read propose/gate runs from MLflow.",
    )
    bl.add_argument("--diagnostics-dir", default=None)
    bl.add_argument("--predictions-dir", default=None)

    lt = sub.add_parser(
        "ledger-trace",
        help="Follow one question, or one edit, through the three ledgers.",
    )
    lt_which = lt.add_mutually_exclusive_group(required=True)
    lt_which.add_argument("--question-id", default=None)
    lt_which.add_argument("--edit-id", default=None)

    return ap


def main() -> None:
    """Entry point for ``convfinqa-evalloop``."""
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "make-splits":
        from convfinqa.evalloop import splits as splits_mod

        if args.train_reports or args.test_reports:
            if not (args.train_reports and args.test_reports and args.name):
                ap.error(
                    "report-count mode needs --name, --train-reports and --test-reports"
                )
            manifest = splits_mod.build_report_manifest(
                name=args.name,
                train_reports=args.train_reports,
                test_reports=args.test_reports,
                extend=args.extend,
                seed=args.seed,
            )
        else:
            manifest = splits_mod.build_manifest(
                target_questions=args.target_questions, seed=args.seed
            )
        out_path = splits_mod.manifest_path(args.name) if args.name else None
        path = splits_mod.write_manifest(manifest, out_path, force=args.force)
        print(f"wrote {path}")  # noqa: T201
        print(json.dumps(manifest["stats"], indent=2))  # noqa: T201

    elif args.cmd == "run":
        if args.split == "holdout":
            ap.error(
                "the holdout is sealed — it opens once per release via the "
                "M3 gate, not from here"
            )
        from convfinqa.evalloop.runner import run_split

        if args.n_reports and args.n_questions:
            ap.error("pass at most one of --n-reports, --n-questions")
        check_runtime(ap, args.runtime, args.version)
        if args.sdk_model and args.runtime != "agent_sdk":
            ap.error("--sdk-model applies to --runtime agent_sdk only")
        if args.sdk_effort and args.runtime != "agent_sdk":
            ap.error("--sdk-effort applies to --runtime agent_sdk only")
        if args.runtime != "open_agent" and (
            args.open_harness or args.open_model or args.open_effort
        ):
            ap.error(
                "--open-harness/--open-model/--open-effort need --runtime open_agent"
            )
        summary = asyncio.run(
            run_split(
                args.split,
                args.version,
                train_seed=args.train_seed,
                stop_at_first_wrong=args.stop_at_first_wrong,
                campaign=args.campaign,
                label=args.label,
                n_reports=args.n_reports,
                n_questions=args.n_questions,
                concurrency=args.concurrency,
                runtime=args.runtime,
                resume_from=args.resume_from,
                sdk_model=args.sdk_model,
                sdk_effort=args.sdk_effort,
                report_ids_file=args.report_ids_file,
                open_harness=args.open_harness,
                open_model=args.open_model,
                open_effort=args.open_effort,
            )
        )
        print(json.dumps(summary, indent=2))  # noqa: T201

    elif args.cmd == "make-smoke":
        from convfinqa.evalloop import smoke

        path = smoke.write(args.name, force=args.force, tries=args.tries)
        print(path.read_text())  # noqa: T201

    elif args.cmd == "smoke-compare":
        from pathlib import Path

        from convfinqa.evalloop import smoke

        csvs: dict[int, Path] = {}
        for item in args.csv:
            seed, _, path_text = item.partition("=")
            if not path_text:
                ap.error(f"--csv {item!r}: expected SEED=CSV")
            csvs[int(seed)] = Path(path_text)
        comparison = smoke.compare(csvs, args.name)
        text = json.dumps(comparison, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n")
        print(text)  # noqa: T201

    elif args.cmd == "gate":
        from convfinqa.evalloop.gate import gate_runs, promote_winner

        result, stats = gate_runs(
            args.baseline_csv,
            args.candidate_csv,
            baseline_version=args.baseline_version,
            candidate_version=args.candidate_version,
        )
        from convfinqa.evalloop.gate import gate_reason

        print(json.dumps(stats, indent=2))  # noqa: T201
        print(gate_reason(stats))  # noqa: T201
        for flip in result.regressions:
            print(f"  regression: {flip.report_id} q{flip.q_order}")  # noqa: T201
        if args.promote:
            if stats.get("evidence_split") != "test":
                ap.error(
                    "promotion evidence must come from the unseen test split — "
                    f"this comparison ran on {stats.get('evidence_split')!r}. "
                    "Train runs optimise; test runs promote."
                )
            promotion = promote_winner(result, stats)
            print(json.dumps(promotion, indent=2, default=str))  # noqa: T201

    elif args.cmd == "diagnose":
        check_runtime(ap, args.runtime, args.version)
        kwargs = {"experiment": args.experiment} if args.experiment else {}
        if args.runtime == "agent_sdk":
            summary = asyncio.run(
                _sdk_teacher().diagnose_run(
                    args.csv,
                    args.version,
                    concurrency=args.concurrency,
                    campaign=args.campaign,
                    label=args.label,
                    sdk_model=args.sdk_model,
                    sdk_effort=args.sdk_effort,
                    **kwargs,
                )
            )
        else:
            if args.sdk_model or args.sdk_effort:
                ap.error("--sdk-model/--sdk-effort apply to --runtime agent_sdk only")
            from convfinqa.evalloop import teacher

            summary = asyncio.run(
                teacher.diagnose_run(
                    args.csv, args.version, concurrency=args.concurrency, **kwargs
                )
            )
        print(json.dumps(summary, indent=2, default=str))  # noqa: T201

    elif args.cmd == "propose":
        check_runtime(ap, args.runtime, args.base_version, args.new_version)
        if args.runtime == "agent_sdk":
            sdk_teacher = _sdk_teacher()
            model, effort = sdk_teacher.default_agent()
            model, effort = args.sdk_model or model, args.sdk_effort or effort
            out = asyncio.run(
                sdk_teacher.propose_version(
                    args.diagnoses,
                    base_version=args.base_version,
                    new_version=args.new_version,
                    campaign=args.campaign,
                    label=args.label,
                    pooled=sdk_teacher.rank_classes(
                        args.base_version, sdk_model=model, sdk_effort=effort
                    ),
                    max_areas=args.max_areas,
                    target_model=model,
                    target_effort=effort,
                )
            )
        else:
            if args.max_areas is not None:
                ap.error("--max-areas applies to --runtime agent_sdk only")
            if args.sdk_model or args.sdk_effort:
                ap.error("--sdk-model/--sdk-effort apply to --runtime agent_sdk only")
            if len(args.diagnoses) != 1:
                ap.error("--runtime pipeline takes one --diagnoses file")
            from convfinqa.evalloop import teacher

            out = asyncio.run(
                teacher.propose_version(
                    args.diagnoses[0],
                    base_version=args.base_version,
                    new_version=args.new_version,
                    target=args.target,
                    campaign=args.campaign,
                    label=args.label,
                )
            )
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "gate-targeted":
        check_runtime(ap, args.runtime, args.baseline_version, args.candidate_version)
        if args.runtime == "agent_sdk":
            if not args.target_class or args.target_agent:
                ap.error(
                    "--runtime agent_sdk judges overall accuracy and takes "
                    "--target-class (the failure class rewritten), not --target-agent"
                )
            from convfinqa.evalloop import sdk_gate
            from convfinqa.tracking import registry

            verdict, comparison = sdk_gate.gate_overall(
                args.baseline_csv,
                args.candidate_csv,
                baseline_version=args.baseline_version,
                candidate_version=args.candidate_version,
                target_class=args.target_class,
            )
            if args.promote and verdict["evidence_split"] != "test":
                ap.error(
                    "promotion evidence must come from the unseen test split — "
                    f"this comparison ran on {verdict['evidence_split']!r}. "
                    "Train runs optimise; test runs promote."
                )
            promoted = False
            if args.promote and verdict["promotable"]:
                outcome = registry.promote_sdk(
                    args.candidate_version,
                    comparison=comparison,
                    evidence_split=str(verdict["evidence_split"]),
                    evidence_csv=args.candidate_csv,
                    actor="evalloop-teacher-sdk",
                    reason=verdict["reason"],
                )
                promoted = outcome.promoted
                print(  # noqa: T201
                    json.dumps(
                        {"promoted_via": "sdk overall rule", **outcome.as_dict()},
                        indent=2,
                        default=str,
                    )
                )
            elif args.promote:
                print("gate rule failed — challenger NOT promoted")  # noqa: T201
            # Written after the promotion was tried, so the row says what the
            # registry did; --campaign counts it against that campaign's cap.
            verdict["gate_run_id"] = sdk_gate.log_gate_verdict(
                verdict,
                comparison=comparison,
                campaign=args.campaign,
                label=args.label,
                rewrite_id=args.rewrite_id,
                promoted=promoted,
            )
            print(json.dumps(verdict, indent=2, default=str))  # noqa: T201
        else:
            if not args.target_agent or args.target_class:
                ap.error(
                    "--runtime pipeline takes --target-agent (the subagent "
                    "rewritten), not --target-class"
                )
            from convfinqa.evalloop import teacher

            verdict, comparison = teacher.gate_targeted(
                args.baseline_csv,
                args.candidate_csv,
                target_agent=args.target_agent,
                baseline_version=args.baseline_version,
                candidate_version=args.candidate_version,
                baseline_diagnoses=args.baseline_diagnoses,
                candidate_diagnoses=args.candidate_diagnoses,
            )
            verdict["gate_run_id"] = teacher.log_gate_verdict(
                verdict, comparison=comparison
            )
            print(json.dumps(verdict, indent=2))  # noqa: T201
            if args.promote and verdict["evidence_split"] != "test":
                ap.error(
                    "promotion evidence must come from the unseen test split — "
                    f"this comparison ran on {verdict['evidence_split']!r}. "
                    "Train runs optimise; test runs promote."
                )
            if args.promote and verdict["promotable"]:
                from convfinqa.tracking import registry

                outcome = registry.promote(
                    args.candidate_version,
                    comparison=comparison,
                    actor="evalloop-teacher",
                    force=True,
                    reason=verdict["reason"],
                )
                print(  # noqa: T201
                    json.dumps(
                        {"promoted_via": "targeted rule", **outcome.as_dict()},
                        indent=2,
                        default=str,
                    )
                )
            elif args.promote:
                print("gate rule failed — challenger NOT promoted")  # noqa: T201

    elif args.cmd == "gate-challenger":
        from convfinqa.evalloop import sdk_gate
        from convfinqa.tracking import registry

        champion = registry.champion_agent() or {}
        model = args.sdk_model or str(champion.get("model") or "")
        effort = args.sdk_effort or str(champion.get("effort") or "")
        train: dict[str, str] = {}
        for csv in args.candidate_train_csv:
            seed = sdk_gate.train_seed_of(csv)
            if seed is None:
                ap.error(f"{csv}: no train_draw_seed on its run record")
            train[str(seed)] = csv
        verdict, comparison = sdk_gate.gate_challenger(
            args.candidate_csv,
            candidate_version=args.candidate_version,
            sdk_model=model,
            sdk_effort=effort,
            candidate_train=train,
            target_class=args.target_class,
            base_agent_csv=args.base_agent_csv,
        )
        promoted = False
        if args.promote and verdict["promotable"]:
            outcome = registry.promote_challenger(
                args.candidate_version,
                model=model,
                effort=effort,
                verdict=verdict["challenger_verdict"],
                actor="evalloop-gate-challenger",
                reason=verdict["reason"],
            )
            promoted = outcome.promoted
            print(json.dumps(outcome.as_dict(), indent=2, default=str))  # noqa: T201
        verdict["gate_run_id"] = sdk_gate.log_gate_verdict(
            verdict,
            comparison=comparison,
            campaign=args.campaign,
            label=args.label,
            rewrite_id=args.rewrite_id,
            promoted=promoted,
            extra_artifacts={
                "challenger_pair.json": verdict["challenger_pair"],
                **(
                    {"vs_base_agent.json": verdict["vs_base_agent"]}
                    if verdict.get("vs_base_agent")
                    else {}
                ),
            },
        )
        print(json.dumps(verdict, indent=2, default=str))  # noqa: T201

    elif args.cmd == "kappa":
        from convfinqa.evalloop import kappa

        if args.make:
            if not args.diagnoses:
                ap.error("--make needs --diagnoses <file.jsonl> [...]")
            sheet_path = kappa.make_sheet(args.diagnoses, out_path=args.out, n=args.n)
            print(f"labelling sheet: {sheet_path}")  # noqa: T201
            print("fill human_agent (triage|preprocess|retriever|calculator|gold),")  # noqa: T201
            print("then score with: convfinqa-evalloop kappa --labels", sheet_path)  # noqa: T201
        elif args.labels:
            print(json.dumps(kappa.score_sheet(args.labels), indent=2))  # noqa: T201
        else:
            ap.error("pass --make (build a sheet) or --labels (score one)")

    elif args.cmd == "release":
        if not args.acknowledged:
            ap.error(
                "the holdout opens once per release and stays opened for every "
                "version that exists today — pass --i-know-this-opens-the-holdout "
                "to proceed"
            )
        from convfinqa.evalloop.release import run_release

        verdict = asyncio.run(
            run_release(
                baseline=args.baseline,
                n_reports=args.n_reports,
                concurrency=args.concurrency,
            )
        )
        print(json.dumps(verdict, indent=2))  # noqa: T201

    elif args.cmd == "cycle":
        from convfinqa.evalloop.cycle import run_cycle

        check_runtime(ap, args.runtime, args.baseline_version, args.new_version)
        steps = asyncio.run(
            run_cycle(
                campaign=args.campaign,
                baseline_version=args.baseline_version,
                new_version=args.new_version,
                target=args.target,
                train_reports=args.train_reports,
                train_seed=args.train_seed,
                concurrency=args.concurrency,
                promote=not args.no_promote,
                baseline_gate_csv=args.baseline_gate_csv,
                runtime=args.runtime,
                sdk_model=args.sdk_model,
                sdk_effort=args.sdk_effort,
                train_csvs=args.train_csv,
                rescore_seeds=args.rescore_seed,
                diagnose_concurrency=args.diagnose_concurrency,
                smoke_reports=args.smoke_reports,
            )
        )
        print(json.dumps(steps, indent=2, default=str))  # noqa: T201

    elif args.cmd == "campaign-status":
        from convfinqa.evalloop import campaign as camp

        print(json.dumps(camp.summarise(args.campaign), indent=2, default=str))  # noqa: T201

    elif args.cmd == "story":
        from convfinqa.evalloop import story

        out = story.build(campaigns=args.campaign, out_dir=args.out)
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "show-prompt":
        import mlflow

        from convfinqa.evalloop import prompt_refs
        from convfinqa.tracking import mlflow_log

        mlflow_log._mlflow()
        trace = mlflow.get_trace(args.trace)
        if trace is None:
            ap.error(f"no trace {args.trace!r} in {mlflow_log.tracking_uri()}")
        run_id = (trace.info.tags or {}).get("mlflow.sourceRun", "")
        shown = 0
        for span in trace.data.spans:
            if args.span and span.name != args.span:
                continue
            refs = (span.inputs or {}).get("refs") or {}
            if not refs:
                continue
            shown += 1
            print(f"\n{'=' * 70}\n{span.name}\n{'=' * 70}")  # noqa: T201
            for slot in ("system_prompt", "target_prompt", "user_prompt"):
                ref = refs.get(slot)
                if not ref:
                    continue
                print(f"\n--- {slot} ({ref.get('kind')}) ---")  # noqa: T201
                try:
                    print(prompt_refs.resolve(ref, run_id=run_id))  # noqa: T201
                except prompt_refs.UnresolvedRefError as exc:
                    print(f"[unresolved] {exc}")  # noqa: T201
        if not shown:
            print(  # noqa: T201
                "no spans in this trace carry prompt refs — it predates them, or "
                "the call was not made through evalloop.sdk.run_structured"
            )

    elif args.cmd == "backfill-flips":
        from convfinqa.evalloop import ledger

        kwargs = {"experiment": args.experiment} if args.experiment else {}
        print(  # noqa: T201
            json.dumps(ledger.backfill_flips(dry_run=args.dry_run, **kwargs), indent=2)
        )

    elif args.cmd == "backfill-sdk-cost":
        from pathlib import Path

        from convfinqa.tracking import sdk_cost

        archive = Path(args.archive) if args.archive else sdk_cost.ARCHIVE_DB
        print(  # noqa: T201
            json.dumps(
                sdk_cost.backfill_sdk_cost(archive=archive, dry_run=args.dry_run),
                indent=2,
            )
        )

    elif args.cmd == "backfill-attribution":
        from convfinqa.evalloop import ledger

        kwargs = {"experiment": args.experiment} if args.experiment else {}
        print(  # noqa: T201
            json.dumps(
                ledger.backfill_attribution(
                    dry_run=args.dry_run, force=args.force, **kwargs
                ),
                indent=2,
            )
        )

    elif args.cmd == "backfill-prompts":
        from convfinqa.tracking import prompt_ledger

        print(json.dumps(prompt_ledger.backfill(), indent=2))  # noqa: T201

    elif args.cmd == "mirror-prompts":
        from convfinqa.tracking import prompt_ledger

        print(  # noqa: T201
            json.dumps(prompt_ledger.mirror_to_mlflow(args.version), indent=2)
        )

    elif args.cmd == "sdk-distil":
        import convfinqa.prompts as prompts_pkg

        if prompts_pkg.is_sdk_version(args.source_version):
            ap.error("--source-version is a pipeline bundle (vN) to distil from")
        if not prompts_pkg.is_sdk_version(args.new_version):
            ap.error("--new-version must be an sdk_vN name")
        kwargs = {"experiment": args.experiment} if args.experiment else {}
        out = asyncio.run(
            _sdk_teacher().distil_prompt(
                source_version=args.source_version,
                new_version=args.new_version,
                **kwargs,
            )
        )
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "judge-dataset":
        from convfinqa.evalloop import judge

        manifest = judge.build_dataset(
            optimise_csv=args.optimise_csv,
            calibrate_csv=args.calibrate_csv,
            test_csv=args.test_csv,
            seed=args.seed,
            name=args.name,
        )
        print(  # noqa: T201
            json.dumps(
                {k: v for k, v in manifest.items() if k != "splits"},
                indent=2,
                default=str,
            )
        )

    elif args.cmd == "judge-diagnose":
        from convfinqa.evalloop import judge

        kwargs = {"experiment": args.experiment} if args.experiment else {}
        out = asyncio.run(
            judge.diagnose_split(
                split=args.split,
                judge_scores=args.judge_scores,
                judge_version=args.judge_version,
                concurrency=args.concurrency,
                label=args.label,
                **kwargs,
            )
        )
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "judge-distil":
        from convfinqa.evalloop import judge

        rows = judge.load_diagnoses()
        if args.round is not None:
            rows = [r for r in rows if r.get("round") == args.round]
        kwargs = {"experiment": args.experiment} if args.experiment else {}
        out = asyncio.run(
            judge.distil_judge(
                new_version=args.new_version or judge.next_judge_version(),
                base_version=args.base_version,
                diagnoses=rows,
                **kwargs,
            )
        )
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "judge-score":
        from convfinqa.evalloop import judge

        kwargs = {"experiment": args.experiment} if args.experiment else {}
        out = asyncio.run(
            judge.score_split(
                version=args.version,
                split=args.split,
                concurrency=args.concurrency,
                error_target=args.error_target,
                label=args.label,
                **kwargs,
            )
        )
        print(json.dumps(out, indent=2, default=str))  # noqa: T201

    elif args.cmd == "judge-gate":
        from convfinqa.evalloop import judge
        from convfinqa.tracking import registry

        try:
            verdict = judge.gate_judges(
                args.baseline_scores,
                args.candidate_scores,
                baseline_version=args.baseline_version,
                candidate_version=args.candidate_version,
                error_target=args.error_target,
            )
        except judge.IncompleteJudgeScoresError as exc:
            raise SystemExit(f"cannot gate: {exc}") from exc
        promoted = False
        if args.promote and verdict["promotable"]:
            outcome = registry.promote_judge(
                args.candidate_version,
                verdict=verdict,
                runtime_version=str(judge.load_dataset().get("runtime_version", "")),
                evidence_split=str(verdict["evidence_split"]),
                actor="evalloop-judge-gate",
            )
            promoted = outcome.promoted
        row = judge.record_gate(
            verdict,
            promoted=promoted,
            champion_after=registry.judge_champion() or "",
        )
        print(json.dumps(row, indent=2, default=str))  # noqa: T201
        if args.promote and not verdict["promotable"]:
            print("judge rule failed — candidate NOT promoted")  # noqa: T201

    elif args.cmd == "backfill-ledgers":
        from pathlib import Path

        from convfinqa.evalloop import ledgers

        counts = ledgers.backfill_ledgers(
            diagnostics_dir=Path(args.diagnostics_dir)
            if args.diagnostics_dir
            else None,
            predictions_dir=Path(args.predictions_dir)
            if args.predictions_dir
            else None,
            use_mlflow=not args.no_mlflow,
        )
        print(json.dumps(counts, indent=2))  # noqa: T201

    elif args.cmd == "ledger-trace":
        from convfinqa.evalloop import ledgers

        joined = ledgers.trace(question_id=args.question_id, edit_id=args.edit_id)
        for name in ("diagnoses", "rewrites", "gates"):
            frame = joined[name]
            print(f"\n== {name} ({len(frame)} row{'s' if len(frame) != 1 else ''}) ==")  # noqa: T201
            for record in frame.to_dict(orient="records"):
                print(json.dumps(record, indent=2, default=str))  # noqa: T201


if __name__ == "__main__":
    main()
