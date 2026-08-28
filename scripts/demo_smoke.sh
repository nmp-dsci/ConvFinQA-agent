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

# 5. The gate holds: an admin write is refused.
code="$(curl -s -o /tmp/smoke_promote.json -w '%{http_code}' --max-time 20 \
  -X POST "$BASE/admin/registry/promote" \
  -H 'Content-Type: application/json' -d '{"version":"v2"}')"
[[ "$code" == "403" ]] || fail "POST /admin/registry/promote returned $code, expected 403"

# 6. The React shell serves from the same origin.
curl -fsS --max-time 20 "$BASE/" | grep -qi '<!doctype html' || fail "index did not render"

echo "SMOKE PASS: $BASE — mode=demo, $versions eval runs, $reports recorded reports, writes refused"
