"""The committed export of the trace store — what the demo image browses.

`.traces/traces.db` is dev state: gitignored, 57 MB, and appended to by every
eval pass. The demo container therefore booted with an empty store, and its
Traces page showed only the handful of turns that deployment had replayed
itself, while the dev app showed 8,425 scored turns across every version and
both runtimes. Same build, same page, two different stories — which is the one
thing a demo of a system is not allowed to be.

This is the answer `mlflow_snapshot.json` already gives for the tracking store,
applied to this one: export to a committed artifact, and seed an empty store
from it. Two decisions worth stating.

*Rows are copied verbatim* — `trace_id` and `created_at` included. Re-recording
them through `TraceStore.record` would stamp every historical turn with the
image's build time, which would put eight thousand eval turns inside
`/metrics/production`'s 24-hour window and report a batch evaluation as a
traffic spike. A replayed turn keeps the moment it was actually answered.

*gzipped JSONL, not a copy of the database.* 6 MB instead of 57, a text artifact
rather than a binary one, and no coupling to the SQLite build that wrote it —
the loader maps by column name onto whatever schema the reading store has, so a
snapshot taken before a column existed still loads, and one taken after it was
added still loads into a store that predates it.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from convfinqa.config import EVAL_ROOT

if TYPE_CHECKING:  # pragma: no cover - import cycle, typing only
    from convfinqa.tracking.traces import TraceStore

SNAPSHOT_PATH = EVAL_ROOT / "traces_snapshot.jsonl.gz"

# The demo replays the pack rather than answering, so its own turns are written
# by the deployment itself and must never be shipped back into the image: a
# snapshot that carried them would double them on the next export.
EXPORTABLE_SOURCES = ("eval", "serving")


def export_snapshot(
    store: TraceStore,
    *,
    path: Path | None = None,
    sources: tuple[str, ...] = EXPORTABLE_SOURCES,
    limit_per_source: int | None = None,
) -> dict[str, Any]:
    """Write `store`'s turns to the committed snapshot; return what was written.

    Ordered oldest-first so the file appends rather than reshuffles when new
    runs are added, which keeps the git diff proportional to what changed.
    `limit_per_source` keeps the newest N of each source when it is set.
    """
    dest = path or SNAPSHOT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for source in sources:
        selected = store.export_rows(source=source, limit=limit_per_source)
        counts[source] = len(selected)
        rows.extend(selected)
    rows.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("trace_id"))))

    # `mtime=0` so an unchanged store exports byte-identical bytes: without it
    # every export is a git diff even when no turn was added.
    with gzip.GzipFile(filename="", fileobj=dest.open("wb"), mode="wb", mtime=0) as fh:
        for row in rows:
            fh.write(
                f"{json.dumps(row, separators=(',', ':'), default=str)}\n".encode()
            )
    return {
        "path": str(dest),
        "n_turns": len(rows),
        "by_source": counts,
        "bytes": dest.stat().st_size,
    }


def iter_snapshot(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield each turn in the snapshot, or nothing when there is no snapshot."""
    src = path or SNAPSHOT_PATH
    if not src.exists():
        return
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_snapshot(store: TraceStore, *, path: Path | None = None) -> int:
    """Insert the snapshot's turns into `store`; return how many were new.

    Idempotent: `trace_id` is the primary key and every insert is `OR IGNORE`,
    so loading twice is a no-op rather than a duplicate history.
    """
    return store.import_rows(iter_snapshot(path))


def seed_if_empty(store: TraceStore, *, path: Path | None = None) -> int:
    """Load the snapshot into a store that has no turns of its own.

    The guard is emptiness, not the deployment mode: a dev machine that has run
    evaluations has its own history and must keep it, and a fresh checkout —
    like the container — gets the committed one. Never raises; a trace store
    that cannot seed is a store with fewer rows, not a broken deployment.
    """
    from convfinqa.config import settings

    if not settings.trace_seed_from_snapshot:
        return 0
    try:
        if store.count() > 0:
            return 0
        return load_snapshot(store, path=path)
    except (OSError, sqlite3.Error, ValueError):
        return 0
