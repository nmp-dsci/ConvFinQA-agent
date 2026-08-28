"""CLI for the s7 diagnosis harness (Diagnose → Route+Fix → Verify).

Importable orchestration: ``main(argv)`` parses args and runs the per-case loop.
Invoked via the thin ``scripts/diagnose_failures.py`` shim or
``python -m convfinqa.diagnosis.cli``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from convfinqa.config import DIAGNOSTICS_DIR, PREDICTIONS_DIR, settings
from convfinqa.diagnosis.aggregator import build_unresolved_cases
from convfinqa.diagnosis.assembler import assemble_variant
from convfinqa.diagnosis.harness import run_harness
from convfinqa.diagnosis.loader import load_first_wrong_cases
from convfinqa.diagnosis.results_html import write_diagnostic_html
from convfinqa.diagnosis.results_writer import write_diagnostic_csv
from convfinqa.diagnosis.rules_store import reset_rules

# Diagnostic outputs land in evaluation/diagnostics/; the input predictions
# CSV is read from evaluation/predictions/ (where convfinqa-eval writes it).
DEFAULT_OUT_DIR = DIAGNOSTICS_DIR


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
        default=None,
        help=(
            "Predictions CSV to read failing cases from. Defaults to "
            "evaluation/predictions/pydantic_predictions_<prompts_version>.csv so each "
            "iteration optimises the failures of the baseline being improved "
            "(v3_2 reads v3_1's predictions, not v2's). The corresponding "
            "predictions CSV must exist; run `PROMPTS_VERSION=<prompts_version> "
            "uv run convfinqa-eval` first if it doesn't."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for diagnostic_results, case_results, unresolved_cases.",
    )
    p.add_argument(
        "--prompts-version",
        default=None,  # resolved in _amain from settings.prompts_version or "v2"
        help=(
            "INPUT prompts version — the baseline the optimisation runs on top of. "
            "Loaded via prompts.load(prompts_version). Defaults to settings.prompts_version "
            "(from PROMPTS_VERSION env) or 'v2'. Same name as convfinqa-eval's "
            "PROMPTS_VERSION so the whole app shares one version identifier. "
            "To chain v3_2 on top of v3_1, pass --prompts-version v3_1 --variant v3_2."
        ),
    )
    p.add_argument(
        "--variant",
        default=None,
        help=(
            "OUTPUT variant name. Controls the suffix used in every artifact "
            "(rules_<agent>_<variant>.jsonl, case_results_<variant>.jsonl, "
            "diagnostic_results_<variant>.{csv,html}) AND the generated prompts "
            "module name (src/convfinqa/prompts/<variant>.py). "
            f"Defaults to settings.variant ({settings.variant!r}). "
            "Pass e.g. --variant v3_2 to start a new iteration without overwriting v3_1."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to the first N first-wrong cases.",
    )
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run Step 1 (Diagnose) only — router classification, no propose, no verify.",
    )
    mode_group.add_argument(
        "--propose-fix",
        action="store_true",
        help=(
            "Run Step 1 (Diagnose) + Step 2 (Propose) — skip Step 3 (Verify). "
            "No rule store writes, no variant-module regeneration. Diagnose cache reused "
            "same as full mode."
        ),
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
        help=(
            "Truncate rules_<agent>_<variant>.jsonl AND "
            "rule_attempts_<agent>_<variant>.jsonl for all agents (variant from --variant)."
        ),
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
            "Ignore Step 1 (Diagnose) cache; re-call the router for every case. "
            "Default reuses cached diagnoses keyed by (report_id, turn_index)."
        ),
    )
    p.add_argument(
        "--no-propose-cache",
        action="store_true",
        help=(
            "Ignore Step 2 (Propose) cache; re-call the specialist Propose LLM "
            "for every attempt. Default reuses cached patch_applied + fix_type/confidence."
        ),
    )
    p.add_argument(
        "--no-verify-cache",
        action="store_true",
        help=(
            "Ignore Step 3 (Verify) cache; re-run verify replays for every "
            "attempt. Default reuses cached FixAttempt turn_results/IOs/correct."
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


def _run_assemble(base_version: str = "v2") -> None:
    path = assemble_variant(base_version=base_version)
    print(f"Assembled {path}")  # noqa: T201


def _run_regression(out_dir: Path) -> None:
    # Lightweight stub — subprocess invocation to eval-api is out of scope for
    # the initial live-end-to-end run. Stub prints a notice so operators see
    # what would happen.
    variant = settings.variant
    print(  # noqa: T201
        f"[regression] Skipped — implement subprocess to convfinqa-eval-api "
        f"--version {variant} writing to "
        f"{PREDICTIONS_DIR}/pydantic_predictions_{variant}.csv."
    )


async def _amain(args: argparse.Namespace) -> int:
    if args.retry_n is not None:
        if not (1 <= args.retry_n <= 3):
            print("--retry-n must be 1..3", file=sys.stderr)  # noqa: T201
            return 2
        # Override the setting for this run only.
        settings.retry_n = args.retry_n

    # Override the variant for this run if --variant was passed. This is read
    # by every downstream module (rules_store, assembler, results_html) via
    # `settings.variant`, so it must be set BEFORE any of them is invoked.
    if args.variant is not None:
        settings.variant = args.variant
    variant = settings.variant

    # Resolve --prompts-version: explicit flag wins; otherwise fall back to the
    # PROMPTS_VERSION env (already on settings.prompts_version) and finally to
    # "v2". This keeps the harness symmetric with convfinqa-eval, which also
    # reads PROMPTS_VERSION from settings.
    prompts_version: str = args.prompts_version or settings.prompts_version or "v2"

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve --input: defaults to predictions of the baseline being improved.
    # Each iteration optimises the failures of its own --prompts-version, so
    # v3_2 (which improves v3_1) must read pydantic_predictions_v3_1.csv —
    # NOT v2's predictions. The corresponding eval CSV must exist on disk.
    input_path: Path = (
        args.input
        if args.input is not None
        else PREDICTIONS_DIR / f"pydantic_predictions_{prompts_version}.csv"
    )
    if not input_path.exists():
        print(  # noqa: T201
            f"[error] predictions CSV not found: {input_path}\n"
            f"        Run `PROMPTS_VERSION={prompts_version} uv run convfinqa-eval` "
            f"to produce it, or pass --input <path>.",
            file=sys.stderr,
        )
        return 2

    if args.stage == "assemble":
        _run_assemble(base_version=prompts_version)
        return 0
    if args.stage == "regression":
        _run_regression(out_dir)
        return 0

    if args.reset_rules:
        reset_rules()
        print(  # noqa: T201
            f"[reset-rules] cleared rules_<agent>_{variant}.jsonl + "
            f"rule_attempts_<agent>_{variant}.jsonl for all agents"
        )

    payloads, full_df = load_first_wrong_cases(
        input_path, version=prompts_version, limit=args.limit
    )
    print(  # noqa: T201
        f"[loader] {len(payloads)} first-wrong case(s) from {input_path} "
        f"(full df: {len(full_df)} rows). prompts_version={prompts_version} variant={variant}"
    )
    if not payloads:
        print("No failing cases — nothing to do.")  # noqa: T201
        return 0

    case_log_path = out_dir / f"case_results_{variant}.jsonl"
    results = await run_harness(
        payloads,
        diagnose_only=args.diagnose_only,
        propose_only=args.propose_fix,
        base_version=prompts_version,
        case_log_path=case_log_path,
        disable_cache=args.no_diagnose_cache,
        use_propose_cache=not args.no_propose_cache,
        use_verify_cache=not args.no_verify_cache,
    )

    csv_path = out_dir / f"diagnostic_results_{variant}.csv"
    write_diagnostic_csv(results, full_df, output_path=csv_path)
    html_path = out_dir / f"diagnostic_results_{variant}.html"
    write_diagnostic_html(
        csv_path,
        output_path=html_path,
        prompts_version=prompts_version,
        variant=variant,
    )
    unresolved_path = out_dir / f"unresolved_cases_{variant}.json"
    build_unresolved_cases(results, unresolved_path)

    # Short-circuit modes (--diagnose-only / --propose-fix) never promote rules,
    # so the post-loop assemble + regression must NOT run — that would silently
    # regenerate <variant>.py from a stale rules store. Only full mode emits them.
    full_mode = not (args.diagnose_only or args.propose_fix)
    prompts_module_path: Path | None = None
    if full_mode:
        # Capture the freshly-assembled prompts module path so the summary can
        # tell the operator what was written.
        from convfinqa.diagnosis.assembler import assemble_variant

        prompts_module_path = assemble_variant(
            base_version=prompts_version, variant=variant
        )
        print(f"Assembled {prompts_module_path}")  # noqa: T201
        if not args.skip_regression:
            _run_regression(out_dir)

    resolved = sum(1 for r in results if r.resolved)
    per_iteration: dict[int, int] = {}
    for r in results:
        if r.resolved and r.winning_iteration is not None:
            per_iteration[r.winning_iteration] = (
                per_iteration.get(r.winning_iteration, 0) + 1
            )

    print("")  # noqa: T201
    print("=== summary ===")  # noqa: T201
    print(f"variant: {variant}  (prompts_version: {prompts_version})")  # noqa: T201
    print(f"cases processed: {len(results)}")  # noqa: T201
    print(f"resolved: {resolved}")  # noqa: T201
    print(f"unresolved: {len(results) - resolved}")  # noqa: T201
    print(f"per-iteration resolution: {dict(sorted(per_iteration.items()))}")  # noqa: T201
    print("artefacts:")  # noqa: T201
    print(f"  {csv_path}")  # noqa: T201
    print(f"  {html_path}")  # noqa: T201
    print(f"  {case_log_path}")  # noqa: T201
    print(f"  {unresolved_path}")  # noqa: T201
    if prompts_module_path is not None:
        print(f"  {prompts_module_path}  ← loadable via prompts.load({variant!r})")  # noqa: T201
        print("")  # noqa: T201
        print(f"Next: PROMPTS_VERSION={variant} uv run convfinqa-eval")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the s7 diagnose/route/fix/verify harness."""
    args = _make_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
