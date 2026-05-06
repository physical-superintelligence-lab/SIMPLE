#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="${SIMPLE_TEST_OUTPUT_ROOT:-$ROOT_DIR/.test-output}"
SAVE_DIR="$OUT_ROOT/datagen-smoke-$RUN_ID"
ENV_ID="simple/G1WholebodyBendPickMP-v0"
SIM_MODE="${SIMPLE_SMOKE_SIM_MODE:-mujoco_isaac}"
DATASET_DIR="$SAVE_DIR/simple/G1WholebodyBendPickMP-v0/level-0"

mkdir -p "$OUT_ROOT"

echo "[check_datagen] save dir: $SAVE_DIR"
echo "[check_datagen] sim mode: $SIM_MODE"

if [[ -z "${ROBO_NIX_ACTIVE:-}" ]]; then
  echo "[check_datagen] run this script inside 'robo shell'" >&2
  exit 1
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
fi

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/datagen" ]]; then
  echo "[check_datagen] missing datagen entrypoint" >&2
  echo "[check_datagen] run: bash scripts/setup_python_env.sh" >&2
  exit 1
fi

"$UV_PROJECT_ENVIRONMENT/bin/datagen" "$ENV_ID" \
  --num-episodes 1 \
  --save-dir "$SAVE_DIR" \
  --sim-mode "$SIM_MODE" \
  --headless \
  --no-webrtc \
  --render-hz 50

test -d "$DATASET_DIR"
test -f "$DATASET_DIR/meta/info.json"
test -f "$DATASET_DIR/meta/episodes.jsonl"

find "$DATASET_DIR/data" -type f -name '*.parquet' | grep -q .
find "$DATASET_DIR/videos" -type f -name '*.mp4' | grep -q .

echo "[check_datagen] success"
echo "[check_datagen] dataset dir: $DATASET_DIR"
