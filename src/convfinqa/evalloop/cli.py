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
        if prompts_pkg.is_sdk_version(version) != (runtime == "agent_sdk"):
            wanted = "agent_sdk" if prompts_pkg.is_sdk_version(version) else "pipeline"
            ap.error(
                f"version {version!r} belongs to --runtime {wanted}, not "
                f"{runtime!r}: sdk_vN prompts run under agent_sdk, vN bundles "
                "under pipeline"
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
        choices=("pipeline", "agent_sdk"),
        help=(
            "Who walks the conversations: the four-agent pipeline (a vN bundle) "
            "or one Claude Agent SDK session per conversation. "
            "--version sdk_vN requires --runtime agent_sdk."
        ),
    )

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

    pr = sub.add_parser(
        "propose", help="Write a challenger changing ONE subagent's prompt."
    )
    pr.add_argument(
        "--diagnoses", required=True, help="diagnoses_*.jsonl from diagnose."
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
            )
        )
        print(json.dumps(summary, indent=2))  # noqa: T201

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
                    **kwargs,
                )
            )
        else:
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
            out = asyncio.run(
                sdk_teacher.propose_version(
                    args.diagnoses,
                    base_version=args.base_version,
                    new_version=args.new_version,
                    campaign=args.campaign,
                    label=args.label,
                    pooled=sdk_teacher.rank_classes(args.base_version),
                    max_areas=args.max_areas,
                )
            )
        else:
            if args.max_areas is not None:
                ap.error("--max-areas applies to --runtime agent_sdk only")
            from convfinqa.evalloop import teacher

            out = asyncio.run(
                teacher.propose_version(
                    args.diagnoses,
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

            verdict, comparison = sdk_gate.gate_overall(
                args.baseline_csv,
                args.candidate_csv,
                baseline_version=args.baseline_version,
                candidate_version=args.candidate_version,
                target_class=args.target_class,
            )
            verdict["gate_run_id"] = sdk_gate.log_gate_verdict(
                verdict, comparison=comparison
            )
            print(json.dumps(verdict, indent=2, default=str))  # noqa: T201
            if args.promote and verdict["evidence_split"] != "test":
                ap.error(
                    "promotion evidence must come from the unseen test split — "
                    f"this comparison ran on {verdict['evidence_split']!r}. "
                    "Train runs optimise; test runs promote."
                )
            if args.promote and verdict["promotable"]:
                from convfinqa.tracking import registry

                outcome = registry.promote_sdk(
                    args.candidate_version,
                    comparison=comparison,
                    evidence_split=str(verdict["evidence_split"]),
                    actor="evalloop-teacher-sdk",
                    reason=verdict["reason"],
                )
                print(  # noqa: T201
                    json.dumps(
                        {"promoted_via": "sdk overall rule", **outcome.as_dict()},
                        indent=2,
                        default=str,
                    )
                )
            elif args.promote:
                print("gate rule failed — challenger NOT promoted")  # noqa: T201
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
