#!/usr/bin/env python
"""Thin shim for the s7 diagnosis CLI.

The implementation lives in ``convfinqa.diagnosis.cli`` so the orchestration is
importable and testable from the package. This script preserves the documented
``uv run python scripts/diagnose_failures.py ...`` invocation.
"""

from __future__ import annotations

from convfinqa.diagnosis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
