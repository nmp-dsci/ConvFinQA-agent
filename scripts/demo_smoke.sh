#!/usr/bin/env bash
# Smoke-test the live demo URL. Asserts the properties that make this
# deployment what it claims to be: it is up, it is in demo mode, it serves the
# committed evidence, and it refuses to call a model. A deploy failing any of
# these fails the workflow.
#
#   ./scripts/demo_smoke.sh https://xyz.ap-southeast-2.awsapprunner.com
set -euo pipefail

BASE="${1:?usage: demo_smoke.sh <base-url>}"
BASE="${BASE%/}"

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

# 1. Health says demo.
health="$(curl -fsS --max-time 20 "$BASE/healthz")" || fail "health unreachable"
echo "$health" | grep -q '"mode":"demo"' || fail "mode is not demo: $health"

# 2. A champion is registered — without one the registry view is empty and the
#    promotion story has nothing to show.
echo "$health" | grep -q '"champion":"' || fail "no champion registered: $health"

# 2b. Deploy binding (M3): what the container serves IS the champion. The
#     bundle's prompts_version and the registry champion are two fields
#     describing the same deployment — if they disagree, the image was built
#     from a stale registry or the resolver drifted (it has, twice).
echo "$health" | python3 -c '
import json, sys
body = json.load(sys.stdin)
champion = body.get("champion")
served = (body.get("bundle") or {}).get("prompts_version")
if champion and served and champion != served:
    sys.exit(f"served bundle {served!r} is not the champion {champion!r}")
' || fail "served version != champion: $health"

# 3. The committed evidence is served.
splits="$(curl -fsS --max-time 20 "$BASE/eval/splits")" || fail "splits unreachable"
echo "$splits" | grep -q 'never_seen' || fail "splits payload missing never_seen"

versions="$(curl -fsS --max-time 20 "$BASE/eval/runs" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
[[ "$versions" -gt 0 ]] || fail "no eval runs served"

# 4. The demo pack is present, or chat is a dead end.
reports="$(curl -fsS --max-time 20 "$BASE/demo/reports" \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
[[ "$reports" -gt 0 ]] || fail "demo pack is empty"

# 5. The metrics endpoint answers, and answers with all three source groups.
#
#    This is the honesty contract in wire form. `serving`, `demo` and `eval`
#    describe three different populations — a live turn, a recording replayed at
#    a watchable pace, and an eval turn at concurrency 8 on a warm cache — and
#    the endpoint must never blend them. It must also return all three on a cold
#    container, before anyone has replayed anything: an absent group reads as
#    "no data" when it means "no turns yet", and a frontend that has to branch on
#    presence is a frontend that will eventually render the wrong empty state.
#
#    Note what is deliberately NOT asserted: nothing about latency or cost being
#    populated. They are `null` with `n_measured: 0` until a metered eval run is
#    paid for, and the board renders that as an em dash with a reason. Asserting
#    a number here would be asserting a lie.
metrics="$(curl -fsS --max-time 20 "$BASE/metrics/production")" \
  || fail "/metrics/production unreachable"
echo "$metrics" | python3 -c '
import json, sys

body = json.load(sys.stdin)
sources = body.get("sources")
if not isinstance(sources, dict):
    sys.exit("no sources object")
missing = [s for s in ("serving", "demo", "eval") if s not in sources]
if missing:
    sys.exit(f"source groups missing: {missing}")
for name in ("serving", "demo", "eval"):
    group = sources[name]
    for key in ("n_turns", "latency_ms", "cost_usd", "accuracy", "errors", "series"):
        if key not in group:
            sys.exit(f"source {name} is missing {key}")
    buckets = group["series"]
    if len(buckets) != 24:
        sys.exit(f"source {name} has {len(buckets)} series buckets, expected 24 hourly")
' || fail "/metrics/production payload is not the three-source shape: $metrics"

# 6. The gate holds: an admin write is refused.
code="$(curl -s -o /tmp/smoke_promote.json -w '%{http_code}' --max-time 20 \
  -X POST "$BASE/admin/registry/promote" \
  -H 'Content-Type: application/json' -d '{"version":"v2"}')"
[[ "$code" == "403" ]] || fail "POST /admin/registry/promote returned $code, expected 403"

# 6. The React shell serves from the same origin.
curl -fsS --max-time 20 "$BASE/" | grep -qi '<!doctype html' || fail "index did not render"

echo "SMOKE PASS: $BASE — mode=demo, $versions eval runs, $reports recorded reports, writes refused"
