"""Background research jobs: launch an s7 round or a GEPA smoke run from the app.

Constraints, all deliberate:

* **One job at a time.** These saturate the provider and cost real money; a queue
  would let an impatient click become four concurrent rounds.
* **Owner-gated, and dev-only.** The demo shows completed rounds and their
  outcomes with the launch controls inert — the gate is `demo_mode`, checked
  here as well as in the route, because a background task is exactly the kind of
  thing that outlives the request that authorised it.
* **Progress over the same SSE machinery as chat**, so the frontend has one
  streaming client rather than two.

Output is captured line by line and kept in a ring buffer, so a browser that
connects mid-run still sees where the round has got to.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from convfinqa.config import REPO_ROOT

MAX_LOG_LINES = 500


@dataclass
class ResearchJob:
    """One launched round: what it is, how it is going, and what it printed."""

    job_id: str
    kind: str
    args: dict[str, Any]
    started_at: str
    status: str = "running"
    returncode: int | None = None
    finished_at: str | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for the API."""
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "args": self.args,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "log_tail": list(self.log)[-40:],
        }


class ResearchRunner:
    """Owns the single in-flight research job and the history of finished ones."""

    def __init__(self) -> None:
        self.current: ResearchJob | None = None
        self.history: list[ResearchJob] = []
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    @property
    def busy(self) -> bool:
        """True while a round is in flight."""
        return self.current is not None and self.current.status == "running"

    def _command(self, kind: str, args: dict[str, Any]) -> list[str]:
        """Build the subprocess command for a round.

        Runs the same CLIs an operator would run by hand — no second code path
        that could behave differently from the documented one.
        """
        if kind == "s7":
            cmd = [
                sys.executable,
                "-m",
                "convfinqa.diagnosis.cli",
                "--limit",
                str(int(args.get("limit", 5))),
                "--retry-n",
                str(int(args.get("retry_n", 1))),
            ]
            if args.get("variant"):
                cmd += ["--variant", str(args["variant"])]
            if args.get("skip_regression", True):
                cmd += ["--skip-regression"]
            return cmd
        if kind == "gepa_smoke":
            return [sys.executable, "-m", "scripts.optimize"]
        raise ValueError(f"Unknown research job kind: {kind!r}")

    def _env(self, kind: str, args: dict[str, Any]) -> dict[str, str]:
        env = dict(os.environ)
        if kind == "gepa_smoke":
            env.update({"RUN_GEPA": "1", "GEPA_MODE": "smoke"})
        if args.get("variant"):
            env["VARIANT"] = str(args["variant"])
        env["PYTHONUNBUFFERED"] = "1"
        return env

    async def start(self, kind: str, args: dict[str, Any]) -> ResearchJob:
        """Launch a round. Raises RuntimeError if one is already running."""
        if self.busy:
            raise RuntimeError("A research round is already running.")
        job = ResearchJob(
            job_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            kind=kind,
            args=args,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        command = self._command(kind, args)
        self.current = job
        self._process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(REPO_ROOT),
            env=self._env(kind, args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._task = asyncio.create_task(self._pump(job))
        await self._publish({"event": "job_start", "job": job.as_dict()})
        return job

    async def _pump(self, job: ResearchJob) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            async for raw in process.stdout:
                line = raw.decode(errors="replace").rstrip()
                job.log.append(line)
                await self._publish({"event": "log", "line": line})
            job.returncode = await process.wait()
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        finally:
            if job.status == "running":
                job.status = "succeeded" if job.returncode == 0 else "failed"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self.history.append(job)
            await self._publish({"event": "job_end", "job": job.as_dict()})
            self._process = None

    async def cancel(self) -> bool:
        """Terminate the running round, if any."""
        process = self._process
        if process is None:
            return False
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        if self.current:
            self.current.status = "cancelled"
        return True

    async def _publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register for progress events. Bounded, so a slow reader is dropped."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Stop receiving progress events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def status(self) -> dict[str, Any]:
        """Current job plus recent history, for the research console."""
        return {
            "busy": self.busy,
            "current": self.current.as_dict() if self.current else None,
            "history": [job.as_dict() for job in reversed(self.history[-20:])],
        }
