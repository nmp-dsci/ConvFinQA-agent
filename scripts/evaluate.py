from __future__ import annotations

import argparse

# ruff: noqa: D103
import asyncio
import os

from convfinqa.evaluation.runner import run_all_versions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ConvFinQA prompt versions.")
    parser.parse_args()
    reuse = os.environ.get("REUSE_CACHE", "1") != "0"
    asyncio.run(run_all_versions(reuse=reuse))


if __name__ == "__main__":
    main()
