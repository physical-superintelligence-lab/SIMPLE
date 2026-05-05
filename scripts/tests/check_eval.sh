#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${SIMPLE_TEST_OUTPUT_ROOT:-$ROOT_DIR/.test-output}"
DATA_RUN_DIR="${SIMPLE_EVAL_DATA_RUN_DIR:-$(find "$OUT_ROOT" -maxdepth 1 -type d -name 'datagen-smoke-*' | sort | tail -n 1)}"
if [[ -z "$DATA_RUN_DIR" || ! -d "$DATA_RUN_DIR" ]]; then
  echo "[check_eval] no datagen output found under $OUT_ROOT" >&2
  echo "[check_eval] run ./scripts/tests/check_datagen.sh first" >&2
  exit 1
fi
ENV_ID="simple/G1WholebodyBendPickMP-v0"
DATASET_DIR="$DATA_RUN_DIR/simple/G1WholebodyBendPickMP-v0/level-0"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[check_eval] dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
EVAL_ROOT="$OUT_ROOT/eval-smoke-$RUN_ID"
SERVER_LOG="$EVAL_ROOT/replay-server.log"
PORT="${SIMPLE_EVAL_SMOKE_PORT:-21090}"

mkdir -p "$EVAL_ROOT"

echo "[check_eval] dataset root: $DATA_RUN_DIR"
echo "[check_eval] lerobot dataset dir: $DATASET_DIR"
echo "[check_eval] eval root: $EVAL_ROOT"

if [[ -z "${ROBO_NIX_ACTIVE:-}" ]]; then
  echo "[check_eval] run this script inside 'robo shell'" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/eval" ]]; then
  echo "[check_eval] missing eval entrypoint" >&2
  echo "[check_eval] run: GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match" >&2
  exit 1
fi

"$UV_PROJECT_ENVIRONMENT/bin/python" ./scripts/tests/replay_policy_server.py \
  --data-root "$DATA_RUN_DIR" \
  --env-id "$ENV_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
echo $! > "$EVAL_ROOT/server.pid"

for _ in $(seq 1 60); do
  if "$UV_PROJECT_ENVIRONMENT/bin/python" - <<PY
import json
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:$PORT/healthz", timeout=1) as resp:
        payload = json.load(resp)
    raise SystemExit(0 if payload.get('ok') else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

"$UV_PROJECT_ENVIRONMENT/bin/eval" "$ENV_ID" replay_policy train \
  --host 127.0.0.1 \
  --port "$PORT" \
  --data-format lerobot \
  --data-dir "$DATASET_DIR" \
  --eval-dir "$EVAL_ROOT" \
  --num-episodes 1 \
  --sim-mode mujoco \
  --headless

SERVER_PID="$(cat "$EVAL_ROOT/server.pid")"

test -f "$EVAL_ROOT/eval_stats.txt"
test -d "$EVAL_ROOT/replay_policy/G1WholebodyBendPickMP-v0/train"
grep -q "success rate:" "$EVAL_ROOT/eval_stats.txt"
find "$EVAL_ROOT/replay_policy/G1WholebodyBendPickMP-v0/train" -type f \( -name '*.mp4' -o -name '*.png' \) | grep -q .

echo "[check_eval] success"
echo "[check_eval] eval root: $EVAL_ROOT"
