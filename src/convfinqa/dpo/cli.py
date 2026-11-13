"""`convfinqa-dpo`: the s33 sampling pass.

convfinqa-dpo manifest                      # cut dpo_v1 from the fresh pool
STUDENT_URL=… convfinqa-dpo warm            # wait for the student server
STUDENT_URL=… convfinqa-dpo sample --upto 300 --concurrency 8
STUDENT_URL=… convfinqa-dpo sample --upto 1489 --tries 8 --out evaluation/dpo/v1/tries8.jsonl \
    --topup-from evaluation/dpo/v1/tries.jsonl   # s34: everything at 8, the 4s topped up
convfinqa-dpo stats [--tries-file …]        # -> stats.json beside the tries file
convfinqa-dpo pairs --tries-file evaluation/dpo/v1/tries8.jsonl   # capped pairs, cap 4 strict
convfinqa-dpo render --tries-file evaluation/dpo/v1/tries8.jsonl \
    --pairs-file evaluation/dpo/v1/pairs_tries8_cap4_strict.jsonl   # -> rows_… + card
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from convfinqa.dpo.sample import DEFAULT_MODEL, TRIES, TRIES_PATH


def warm(timeout_s: float = 1200.0) -> float:
    """Block until the student server lists its models; return seconds waited."""
    import httpx

    from convfinqa.config import settings

    if not settings.student_url:
        raise SystemExit("STUDENT_URL is not set")
    headers = {}
    if settings.modal_proxy_token_id and settings.modal_proxy_token_secret:
        headers = {
            "Modal-Key": settings.modal_proxy_token_id.get_secret_value(),
            "Modal-Secret": settings.modal_proxy_token_secret.get_secret_value(),
        }
    url = f"{settings.student_url.rstrip('/')}/v1/models"
    t0 = time.monotonic()
    while True:
        try:
            r = httpx.get(url, headers=headers, timeout=60)
            if r.status_code == 200:
                names = [m["id"] for m in r.json().get("data", [])]
                print(f"ready after {time.monotonic() - t0:.0f}s: {names}")
                return time.monotonic() - t0
            print(f"  {r.status_code}; waiting")
        except httpx.HTTPError as exc:
            print(f"  {type(exc).__name__}; waiting")
        if time.monotonic() - t0 > timeout_s:
            raise SystemExit("the student server did not come up")
        time.sleep(15)


def main(argv: list[str] | None = None) -> int:
    """The `convfinqa-dpo` entry point."""
    ap = argparse.ArgumentParser(prog="convfinqa-dpo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest", help="write evaluation/splits/dpo_v1.json")
    p.add_argument("--force", action="store_true")
    sub.add_parser("warm", help="wait until the student server answers")
    p = sub.add_parser(
        "sample", help="sample tries until --upto conversations are on record"
    )
    p.add_argument("--upto", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--tries", type=int, default=TRIES)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=TRIES_PATH)
    p.add_argument("--topup-from", type=Path, default=None)
    p = sub.add_parser("stats", help="write stats.json beside a tries file")
    p.add_argument("--tries-file", type=Path, default=TRIES_PATH)
    p = sub.add_parser("pairs", help="write the capped pair selection")
    p.add_argument("--tries-file", type=Path, default=TRIES_PATH)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--plain", action="store_true", help="no strict chosen check")
    p = sub.add_parser("render", help="render a pair selection into DPO rows")
    p.add_argument("--tries-file", type=Path, default=TRIES_PATH)
    p.add_argument("--pairs-file", type=Path, required=True)
    p = sub.add_parser(
        "pilot-data", help="draw the pilot's train pairs, write and upload"
    )
    p.add_argument("--rows-file", type=Path, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=2034)
    p.add_argument("--name", default="dpo_v1_pilot")
    p.add_argument("--no-upload", action="store_true")
    p = sub.add_parser(
        "teacher-pairs",
        help="render the teacher's right turns on the 0/8 questions as DPO rows (s37)",
    )
    p.add_argument("--teacher-run", action="append", required=True, metavar="RUN")
    p.add_argument("--tries-file", type=Path, default=TRIES_PATH)
    p.add_argument(
        "--targets", type=Path, default=TRIES_PATH.parent / "all_wrong_tries8.json"
    )
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("builder-sets", help="cut the builder's dev/sealed sets (s39)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser(
        "builder-build", help="gold-hinted teacher traces for a set (s39)"
    )
    p.add_argument("--set", required=True, choices=("dev", "sealed"))
    p.add_argument("--tries", type=int, default=2)
    p.add_argument("--hint", default="h1")
    p.add_argument("--concurrency", type=int, default=16)
    p = sub.add_parser("judge-run", help="2AFC judge over a built set (s39)")
    p.add_argument("--set", required=True, choices=("dev", "sealed"))
    p.add_argument("--hint", default="h1")
    p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("judge-report", help="score a judged set (s39)")
    p.add_argument("--set", required=True, choices=("dev", "sealed"))
    p.add_argument("--hint", default="h1")
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        from convfinqa.dpo import manifest

        path = manifest.write(force=args.force)
        print(path, json.dumps(manifest.load()["stats"]))
    elif args.cmd == "warm":
        warm()
    elif args.cmd == "sample":
        from convfinqa.dpo import sample

        warm()
        asyncio.run(
            sample.run(
                upto=args.upto,
                concurrency=args.concurrency,
                model_name=args.model,
                tries=args.tries,
                out=args.out,
                runs_log=args.out.parent / f"{args.out.stem}_runs.jsonl",
                topup_from=args.topup_from,
            )
        )
    elif args.cmd == "stats":
        from convfinqa.dpo import stats

        s = stats.write(tries_path=args.tries_file)
        print(
            json.dumps(
                {k: s[k] for k in ("conversations", "reached_share", "overall")},
                indent=1,
            )
        )
    elif args.cmd == "pairs":
        from collections import Counter

        from convfinqa.dpo import manifest, pairs
        from convfinqa.dpo.sample import load_records

        m = manifest.load()
        split_of = {r: s for s, ids in m["splits"].items() for r in ids}
        rows = pairs.selection(
            load_records(args.tries_file),
            split_of,
            args.cap,
            strict=not args.plain,
        )
        tag = f"cap{args.cap}{'' if args.plain else '_strict'}"
        dest = args.tries_file.parent / f"pairs_{args.tries_file.stem}_{tag}.jsonl"
        with dest.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        counts = Counter(f"{r['split']}:{r['kind']}" for r in rows)
        print(dest, len(rows), json.dumps(dict(sorted(counts.items()))))
    elif args.cmd == "render":
        from convfinqa.dpo import render

        stem = args.pairs_file.stem.removeprefix("pairs_")
        out = args.pairs_file.parent / f"rows_{stem}.jsonl"
        summary = render.render_all(args.tries_file, args.pairs_file, out)
        (out.parent / f"rows_{stem}_card.json").write_text(
            json.dumps(summary, indent=1) + "\n"
        )
        print(out, json.dumps(summary, indent=1))
    elif args.cmd == "teacher-pairs":
        from convfinqa.config import EVAL_ROOT
        from convfinqa.dpo import manifest, teacher
        from convfinqa.sft import trajectories as tj

        m = manifest.load()
        split_of = {r: s for s, ids in m["splits"].items() for r in ids}
        passes = [
            (EVAL_ROOT / "predictions" / "evalloop" / f"{run}.csv", tj.fetch(run))
            for run in args.teacher_run
        ]
        card = teacher.build(
            passes, args.targets, args.tries_file, split_of, args.out, cap=args.cap
        )
        args.out.with_name(args.out.name.split(".")[0] + "_card.json").write_text(
            json.dumps(card, indent=1) + "\n"
        )
        print(args.out, json.dumps(card, indent=1))
    elif args.cmd == "builder-sets":
        from convfinqa.dpo import builder

        print(builder.write_sets(force=args.force))
    elif args.cmd == "builder-build":
        import asyncio

        from convfinqa.dpo import builder

        builder.prepare()
        print(
            asyncio.run(
                builder.build_set(
                    args.set,
                    tries=args.tries,
                    concurrency=args.concurrency,
                    hint=args.hint,
                )
            )
        )
    elif args.cmd == "judge-run":
        import asyncio

        from convfinqa.dpo import builder, judge

        builder.prepare()
        print(
            asyncio.run(
                judge.judge_set(args.set, concurrency=args.concurrency, hint=args.hint)
            )
        )
    elif args.cmd == "judge-report":
        from convfinqa.dpo import builder, judge

        builder.prepare()
        rep = judge.report(args.set, args.hint)
        (builder.ROOT / f"{args.set}_report{builder.tag(args.hint)}.json").write_text(
            json.dumps(rep, indent=1) + "\n"
        )
        print(json.dumps(rep, indent=1))
    elif args.cmd == "pilot-data":
        from convfinqa.dpo import pilot

        pstats = pilot.write(args.rows_file, n=args.n, seed=args.seed, name=args.name)
        print(json.dumps(pstats, indent=1))
        if not args.no_upload:
            return pilot.upload(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
