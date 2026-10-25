"""`convfinqa-sft`: the s28 data, training and serving commands.

    convfinqa-sft fetch --csv <pass>.csv          # its trajectories into evaluation/sft/
    convfinqa-sft build --csv <pass>.csv          # gate, render, write evaluation/sft/v1/
    convfinqa-sft splits --val-csv <e2 val>.csv   # write sft_v2: sft_v1 + the aimed draw
    convfinqa-sft build --csv <a>.csv --csv <b>.csv --version v2 --manifest sft_v2 --prefix-rows
    convfinqa-sft upload                          # the dataset onto Modal's data volume
    convfinqa-sft serve                           # deploy the vLLM student server on Modal
    convfinqa-sft train [--max-steps N --longest K]  # LoRA on Modal (spawned, detached)
    convfinqa-sft sync --run <run id>             # the run folder home, into runs/sft/

Scoring the student is the ordinary eval loop: ``convfinqa-evalloop run
--runtime open_agent --open-model student-qwen3-4b[-sft1] --open-effort none``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from convfinqa.config import EVAL_ROOT

ROOT = EVAL_ROOT.parent
MODAL_APP = "src/convfinqa/sft/cloud/modal_app.py"


def _modal(*args: str) -> int:
    """`modal …` from this repo's environment, output streamed."""
    return subprocess.call([sys.executable, "-m", "modal", *args], cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    """Run one `convfinqa-sft` command; return its exit code."""
    parser = argparse.ArgumentParser(
        prog="convfinqa-sft", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "fetch", help="copy a pass's trajectories artifact into the repo"
    )
    p.add_argument("--csv", type=Path, action="append", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("build", help="gate the passes and write the dataset")
    p.add_argument("--csv", type=Path, action="append", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--manifest", default="sft_v1", choices=("sft_v1", "sft_v2"))
    p.add_argument(
        "--prefix-rows",
        action="store_true",
        help="also keep an F1 drop's turns before its first wrong answer (s29 N1)",
    )

    p = sub.add_parser(
        "splits", help="write sft_v2: sft_v1's train plus the aimed draw (s29 N2)"
    )
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("upload", help="put the dataset on Modal's data volume")
    p.add_argument("--version", default="v1")

    sub.add_parser("serve", help="deploy the student server (base + adapters) on Modal")

    p = sub.add_parser("train", help="LoRA on one Modal L4, detached")
    p.add_argument("--config", default="sft_v1.yaml")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--longest", type=int, default=0)

    p = sub.add_parser("sync", help="copy a training run's folder home")
    p.add_argument("--run", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        from convfinqa.sft import trajectories

        for csv in args.csv:
            print(trajectories.fetch(Path(csv).stem, force=args.force))
        return 0
    if args.cmd == "build":
        from convfinqa.sft import build

        stats = build.build(
            [Path(c) for c in args.csv],
            version=args.version,
            manifest_name=args.manifest,
            prefix_rows=args.prefix_rows,
        )
        print(
            json.dumps(
                {
                    k: stats[k]
                    for k in (
                        "n_rows",
                        "n_prefix_rows",
                        "n_turns",
                        "n_dropped",
                        "drops_by_rule",
                        "tokens",
                        "train_sha256",
                    )
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "splits":
        from convfinqa.sft import splits

        path = splits.write_manifest_v2(args.val_csv, force=args.force)
        manifest = splits.load("sft_v2")
        print(path, json.dumps(manifest["stats"]), splits.COLLECT_V2_PATH)
        return 0
    if args.cmd == "upload":
        from convfinqa.sft.cloud.modal_app import DATA_VOLUME

        folder = EVAL_ROOT / "sft" / args.version
        for name in ("train.jsonl", "stats.json"):
            code = _modal(
                "volume",
                "put",
                "--force",
                DATA_VOLUME,
                str(folder / name),
                f"/{args.version}/{name}",
            )
            if code:
                return code
        return 0
    if args.cmd == "serve":
        return _modal("deploy", MODAL_APP)
    if args.cmd == "train":
        return _modal(
            "run",
            "--detach",
            f"{MODAL_APP}::train_job",
            "--config",
            args.config,
            "--max-steps",
            str(args.max_steps),
            "--longest",
            str(args.longest),
        )
    if args.cmd == "sync":
        from convfinqa.sft.cloud.modal_app import RUNS_VOLUME

        dest = ROOT / "runs" / "sft"
        dest.mkdir(parents=True, exist_ok=True)
        return _modal(
            "volume", "get", "--force", RUNS_VOLUME, f"/{args.run}", str(dest)
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
