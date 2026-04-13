#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 TASK [HOST:PORT]" >&2
  echo "Example: $0 simple/G1WholebodyBendPick-v0 localhost:22085" >&2
  exit 1
fi

TASK="$1"
ADDR="${2:-${ADDR:-localhost:22085}}"
HOST="${ADDR%:*}"
PORT="${ADDR##*:}"

SIM_MODE="${SIM_MODE:-mujoco_isaac}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
EPISODE_START="${EPISODE_START:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
DATA_FORMAT="${DATA_FORMAT:-lerobot}"
POLICY="${POLICY:-psi0}"

if [[ -z "${DATA_DIR:-}" ]]; then
  task_name="${TASK#simple/}"
  if [[ -d "data/${task_name}-psi0/meta" ]]; then
    DATA_DIR="data/${task_name}-psi0"
  elif [[ -d "data/${task_name}/meta" ]]; then
    DATA_DIR="data/${task_name}"
  else
    DATA_DIR="data"
  fi
fi

video_flag="--save-video"
if [[ "$SAVE_VIDEO" == "0" ]]; then
  video_flag="--no-save-video"
fi

nix develop -c bash -lc '
  set -euo pipefail

  if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/eval" ]]; then
    echo "[eval_remote_policy] missing $UV_PROJECT_ENVIRONMENT/bin/eval"
    echo "[eval_remote_policy] run: ./scripts/nix/bootstrap.sh"
    exit 1
  fi

  "$UV_PROJECT_ENVIRONMENT/bin/eval" "'"$TASK"'" "'"$POLICY"'" \
    --host="'"$HOST"'" \
    --port="'"$PORT"'" \
    --sim-mode="'"$SIM_MODE"'" \
    '"$video_flag"' \
    --headless \
    --max-episode-steps="'"$MAX_EPISODE_STEPS"'" \
    --episode-start="'"$EPISODE_START"'" \
    --num-workers="'"$NUM_WORKERS"'" \
    --data-format="'"$DATA_FORMAT"'" \
    --data-dir="'"$DATA_DIR"'" \
    --num-episodes="'"$NUM_EPISODES"'"
'
