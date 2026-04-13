#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="${SIMPLE_TEST_OUTPUT_ROOT:-$ROOT_DIR/.test-output}"
SAVE_DIR="$OUT_ROOT/datagen-smoke-$RUN_ID"
DATASET_DIR="$SAVE_DIR/simple/G1WholebodyBendPick-v0/level-0"

mkdir -p "$OUT_ROOT"

echo "[check_datagen] save dir: $SAVE_DIR"

XDG_CACHE_HOME="$ROOT_DIR/.nix-cache" \
SIMPLE_AUTO_BOOTSTRAP=0 \
nix develop -c bash -lc "
  set -euo pipefail
  if [[ ! -x \"\$UV_PROJECT_ENVIRONMENT/bin/python\" ]]; then
    uv venv --python \"\${UV_PYTHON:-python3.10}\" \"\$UV_PROJECT_ENVIRONMENT\"
  fi

  if [[ ! -x \"\$UV_PROJECT_ENVIRONMENT/bin/datagen\" ]]; then
    ./scripts/nix/bootstrap.sh
  fi

  \"\$UV_PROJECT_ENVIRONMENT/bin/datagen\" simple/G1WholebodyBendPick-v0 \
    --num-episodes 1 \
    --save-dir '$SAVE_DIR' \
    --headless \
    --render-hz 50
"

test -d "$DATASET_DIR"
test -f "$DATASET_DIR/meta/info.json"
test -f "$DATASET_DIR/meta/episodes.jsonl"

find "$DATASET_DIR/data" -type f -name '*.parquet' | grep -q .
find "$DATASET_DIR/images" -type f -name '*.png' | grep -q .

echo "[check_datagen] success"
echo "[check_datagen] dataset dir: $DATASET_DIR"
