"""CLI for the eval loop: splits, runs, the gate, and the M2 teacher.

    convfinqa-evalloop make-splits
    convfinqa-evalloop run --split train --version v2 --n-reports 10
    convfinqa-evalloop gate --baseline-csv A.csv --candidate-csv B.csv \
        --baseline-version v2 --candidate-version v3_1 --promote
    convfinqa-evalloop diagnose --csv <run.csv> --version v3_1
    convfinqa-evalloop propose --diagnoses <d.jsonl> --base-version v3_1 --new-version v4
    convfinqa-evalloop gate-targeted --target-agent triage ...
"""

from __future__ import annotations

import argparse
import asyncio
import json


def main() -> None:
    """Entry point for ``convfinqa-evalloop``."""
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

    rn = sub.add_parser("run", help="Run one split × version pass (an MLflow run).")
    rn.add_argument("--split", default="train", choices=("train", "test", "holdout"))
    rn.add_argument("--version", required=True, help="Prompt version, e.g. v2.")
    rn.add_argument("--n-reports", type=int, default=None)
    rn.add_argument("--concurrency", type=int, default=8)

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

    pr = sub.add_parser(
        "propose", help="Write a challenger changing ONE subagent's prompt."
    )
    pr.add_argument(
        "--diagnoses", required=True, help="diagnoses_*.jsonl from diagnose."
    )
    pr.add_argument("--base-version", required=True)
    pr.add_argument("--new-version", required=True)
    pr.add_argument("--target", default=None, help="Subagent; default = most faults.")

    gtt = sub.add_parser(
        "gate-targeted",
        help="M2 gate: target subagent improved AND overall not regressed.",
    )
    gtt.add_argument("--baseline-csv", required=True)
    gtt.add_argument("--candidate-csv", required=True)
    gtt.add_argument("--baseline-version", required=True)
    gtt.add_argument("--candidate-version", required=True)
    gtt.add_argument("--target-agent", required=True)
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

    mp = sub.add_parser(
        "mirror-prompts",
        help="Mirror a bundle's four prompts into MLflow's prompt registry.",
    )
    mp.add_argument("--version", required=True)

    args = ap.parse_args()

    if args.cmd == "make-splits":
        from convfinqa.evalloop.splits import build_manifest, write_manifest

        manifest = build_manifest(
            target_questions=args.target_questions, seed=args.seed
        )
        path = write_manifest(manifest, force=args.force)
        print(f"wrote {path}")  # noqa: T201
        print(json.dumps(manifest["stats"], indent=2))  # noqa: T201

    elif args.cmd == "run":
        if args.split == "holdout":
            ap.error(
                "the holdout is sealed — it opens once per release via the "
                "M3 gate, not from here"
            )
        from convfinqa.evalloop.runner import run_split

        summary = asyncio.run(
            run_split(
                args.split,
                args.version,
                n_reports=args.n_reports,
                concurrency=args.concurrency,
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
        print(json.dumps(stats, indent=2))  # noqa: T201
        print(result.reason())  # noqa: T201
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
        from convfinqa.evalloop import teacher

        kwargs = {"experiment": args.experiment} if args.experiment else {}
        summary = asyncio.run(teacher.diagnose_run(args.csv, args.version, **kwargs))
        print(json.dumps(summary, indent=2))  # noqa: T201

    elif args.cmd == "propose":
        from convfinqa.evalloop import teacher

        out = asyncio.run(
            teacher.propose_version(
                args.diagnoses,
                base_version=args.base_version,
                new_version=args.new_version,
                target=args.target,
            )
        )
        print(json.dumps(out, indent=2))  # noqa: T201

    elif args.cmd == "gate-targeted":
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
        print(json.dumps(verdict, indent=2))  # noqa: T201
        if args.promote and verdict["evidence_split"] != "test":
            ap.error(
                "promotion evidence must come from the unseen test split — "
                f"this comparison ran on {verdict['evidence_split']!r}. "
                "Train runs optimise; test runs promote."
            )
        if args.promote and verdict["promotable_targeted"]:
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
            print("targeted rule failed — challenger NOT promoted")  # noqa: T201

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

    elif args.cmd == "backfill-prompts":
        from convfinqa.tracking import prompt_ledger

        print(json.dumps(prompt_ledger.backfill(), indent=2))  # noqa: T201

    elif args.cmd == "mirror-prompts":
        from convfinqa.tracking import prompt_ledger

        print(  # noqa: T201
            json.dumps(prompt_ledger.mirror_to_mlflow(args.version), indent=2)
        )


if __name__ == "__main__":
    main()
