"""`convfinqa-mlflow` — register, compare, promote, backfill, export.

Everything the promotion lifecycle needs from a terminal. The same operations the
admin API exposes, so the UI and the CLI cannot drift into disagreeing about what
promotion means.
"""

from __future__ import annotations

import argparse
import json
import sys

from convfinqa.config import settings
from convfinqa.tracking import backfill as backfill_mod
from convfinqa.tracking import mlflow_log, registry, snapshot
from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id
from convfinqa.tracking.comparator import available_versions, compare


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convfinqa-mlflow",
        description="Bundle registry, experiment tracking, and promotion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show tracking config, aliases and versions.")
    sub.add_parser("bundle", help="Print the current bundle fingerprint.")
    sub.add_parser("versions", help="List versions with committed predictions.")
    sub.add_parser("snapshot", help="Export the committed mlflow_snapshot.json.")

    p_backfill = sub.add_parser(
        "backfill", help="Reconstruct history from committed CSVs and GEPA runs."
    )
    p_backfill.add_argument(
        "--champion", default="", help="Version to seed as champion (default: best)."
    )

    p_register = sub.add_parser("register", help="Register a bundle version.")
    p_register.add_argument("version")
    p_register.add_argument(
        "--source", default="manual", choices=["manual", "gepa", "s7"]
    )
    p_register.add_argument("--notes", default="")

    p_compare = sub.add_parser(
        "compare", help="Compare two versions question by question."
    )
    p_compare.add_argument("baseline")
    p_compare.add_argument("candidate")

    p_promote = sub.add_parser("promote", help="Promote a version to champion.")
    p_promote.add_argument("version")
    p_promote.add_argument(
        "--force",
        action="store_true",
        help="Bypass the comparator. Recorded as forced in the history.",
    )

    p_challenger = sub.add_parser("challenger", help="Point the challenger alias.")
    p_challenger.add_argument("version")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the tracking CLI."""
    args = _make_parser().parse_args(argv)

    if args.command == "status":
        _print(
            {
                "tracking": mlflow_log.env_summary(),
                "registry": registry.summary(),
                "settings": {
                    "demo_mode": settings.demo_mode,
                    "prompts_version": settings.prompts_version,
                },
            }
        )
        return 0

    if args.command == "bundle":
        fingerprint = bundle_fingerprint()
        _print({"bundle_id": bundle_id(fingerprint), **fingerprint})
        return 0

    if args.command == "versions":
        _print(available_versions())
        return 0

    if args.command == "snapshot":
        path = snapshot.write_snapshot()
        print(f"wrote {path}")
        return 0

    if args.command == "backfill":
        result = backfill_mod.backfill(champion=args.champion or None)
        _print(result)
        return 0

    if args.command == "register":
        entry = registry.record_evaluation(args.version, source=args.source)
        _print(entry)
        return 0

    if args.command == "compare":
        result = compare(args.baseline, args.candidate)
        _print(result.as_dict())
        return 0 if result.promotable else 1

    if args.command == "promote":
        outcome = registry.promote(args.version, force=args.force, actor="cli")
        _print(outcome.as_dict())
        return 0 if outcome.promoted else 1

    if args.command == "challenger":
        registry.set_alias(registry.CHALLENGER, args.version)
        _print(registry.summary()["aliases"])
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
