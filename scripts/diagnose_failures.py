"""CLI entry point for the s7 diagnosis harness (Diagnose → Route+Fix → Verify)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from convfinqa.config import settings
from convfinqa.diagnosis.aggregator import build_unresolved_cases
from convfinqa.diagnosis.assembler import assemble_v3_opt
from convfinqa.diagnosis.harness import run_harness
from convfinqa.diagnosis.loader import load_first_wrong_cases
from convfinqa.diagnosis.results_html import write_diagnostic_html
from convfinqa.diagnosis.results_writer import write_diagnostic_csv
from convfinqa.diagnosis.rules_store import reset_rules

DEFAULT_INPUT = Path("evaluation/pydantic_predictions_v2.csv")
DEFAULT_OUT_DIR = Path("evaluation")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diagnose_failures",
        description=(
            "Per-case Diagnose → Route+Fix → Verify loop over first-wrong cases."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Predictions CSV to read failing cases from.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for diagnostic_results, case_results, unresolved_cases.",
    )
    p.add_argument(
        "--version",
        default="v2",
        help="Base prompt version. Defaults to v2.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to the first N first-wrong cases.",
    )
    p.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run Step 1 only — router classification, no fix, no verify.",
    )
    p.add_argument(
        "--stage",
        choices=["all", "assemble", "regression"],
        default="all",
        help="all (default): run the per-case loop. assemble/regression: short-circuit.",
    )
    p.add_argument(
        "--reset-rules",
        action="store_true",
        help="Truncate rules_<agent>_v3_opt.jsonl AND rule_attempts_<agent>_v3_opt.jsonl for all agents.",
    )
    p.add_argument(
        "--retry-n",
        type=int,
        default=None,
        help="Total attempts cap per case (1..3). Default = settings.retry_n.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reserved for resume semantics — currently no-op (case_results is always overwritten).",
    )
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the post-loop regression subprocess (local dev only).",
    )
    p.add_argument(
        "--no-diagnose-cache",
        action="store_true",
        help=(
            "Ignore any existing case_results_v3_opt.jsonl and re-call the router "
            "for every case. Default reuses cached diagnoses keyed by (report_id, turn_index)."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG logging.",
    )
    return p


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _run_assemble() -> None:
    path = assemble_v3_opt()
    print(f"Assembled {path}")  # noqa: T201


def _run_regression(out_dir: Path) -> None:
    # Lightweight stub — subprocess invocation to eval-api is out of scope for
    # the initial live-end-to-end run. Stub prints a notice so operators see
    # what would happen.
    print(  # noqa: T201
        f"[regression] Skipped — implement subprocess to convfinqa-eval-api "
        f"--version v3_opt writing to {out_dir}/pydantic_predictions_v3_opt.csv."
    )


async def _amain(args: argparse.Namespace) -> int:
    if args.retry_n is not None:
        if not (1 <= args.retry_n <= 3):
            print("--retry-n must be 1..3", file=sys.stderr)  # noqa: T201
            return 2
        # Override the setting for this run only.
        settings.retry_n = args.retry_n

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "assemble":
        _run_assemble()
        return 0
    if args.stage == "regression":
        _run_regression(out_dir)
        return 0

    if args.reset_rules:
        reset_rules()
        print("[reset-rules] cleared all rules + rule_attempts stores")  # noqa: T201

    payloads, full_df = load_first_wrong_cases(
        args.input, version=args.version, limit=args.limit
    )
    print(  # noqa: T201
        f"[loader] {len(payloads)} first-wrong case(s) from {args.input} "
        f"(full df: {len(full_df)} rows)"
    )
    if not payloads:
        print("No failing cases — nothing to do.")  # noqa: T201
        return 0

    case_log_path = out_dir / "case_results_v3_opt.jsonl"
    results = await run_harness(
        payloads,
        diagnose_only=args.diagnose_only,
        base_version=args.version,
        case_log_path=case_log_path,
        disable_cache=args.no_diagnose_cache,
    )

    csv_path = out_dir / "diagnostic_results_v3_opt.csv"
    write_diagnostic_csv(results, full_df, output_path=csv_path)
    html_path = out_dir / "diagnostic_results_v3_opt.html"
    write_diagnostic_html(csv_path, output_path=html_path)
    unresolved_path = out_dir / "unresolved_cases_v3_opt.json"
    build_unresolved_cases(results, unresolved_path)

    if not args.diagnose_only:
        _run_assemble()
        if not args.skip_regression:
            _run_regression(out_dir)

    resolved = sum(1 for r in results if r.resolved)
    per_iteration: dict[int, int] = {}
    for r in results:
        if r.resolved and r.winning_iteration is not None:
            per_iteration[r.winning_iteration] = per_iteration.get(r.winning_iteration, 0) + 1

    print("")  # noqa: T201
    print("=== summary ===")  # noqa: T201
    print(f"cases processed: {len(results)}")  # noqa: T201
    print(f"resolved: {resolved}")  # noqa: T201
    print(f"unresolved: {len(results) - resolved}")  # noqa: T201
    print(f"per-iteration resolution: {dict(sorted(per_iteration.items()))}")  # noqa: T201
    print("artefacts:")  # noqa: T201
    print(f"  {csv_path}")  # noqa: T201
    print(f"  {html_path}")  # noqa: T201
    print(f"  {case_log_path}")  # noqa: T201
    print(f"  {unresolved_path}")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
