#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Launch a psi0 policy server + dockerized SIMPLE eval client as a pair in a new
# tmux window: server in the left pane, client in the right pane. The client
# waits for the server's port before starting.
#
# usage: launch_psi0_eval_pair.sh DR_LEVEL PORT SERVER_GPU CLIENT_GPU [extra eval options]
#   e.g. launch_psi0_eval_pair.sh dr-level-1 22086 1 3
#        launch_psi0_eval_pair.sh dr-level-2 22087 2 4 --num-episodes=5
#        TASK=G1WholebodyBendPickTeleop-v0 launch_psi0_eval_pair.sh dr-level-0 22086 0 4
#
# Overridable via env:
#   TASK         eval task id (without simple/ prefix)
#   RUN_DIR      psi0 checkpoint run dir, relative to PSI_DIR (default: the task's checkpoint below,
#                from huggingface.co/USC-PSI-Lab/psi-model psi0/simple-checkpoints)
#   CKPT_STEP    checkpoint step
#   PSI_DIR      psi repo (holds .venv-psi)
#   IMAGE_DATE   docker image tag, i.e. simple:$IMAGE_DATE
#   ISAAC_RENDER always: Isaac renders every step; on-query (default): only the steps whose images
#                psi0 reads, MuJoCo frames fill the rest of the video (see eval_decoupled_wbc.py)
#   CLIENT_ONLY  1: start only the eval client, against a psi0 server already listening on PORT (e.g. one
#                left running by an earlier launch with the same checkpoint); no server pane
#   TMUX_TARGET  tmux session to add the window to (default: current/first session)
#   STAMP        log folder name under logs/ (default: current time); give parallel runs the same
#                STAMP to collect them in one folder:
#                  export STAMP=$(date +%Y%m%d-%H%M%S)
#                  for i in 0 1 2; do launch_psi0_eval_pair.sh dr-level-$i 2208$i $i $((i+4)); done
set -euo pipefail

USAGE="usage: $(basename "$0") DR_LEVEL PORT SERVER_GPU CLIENT_GPU [extra eval options]"
DR=${1:?$USAGE}
PORT=${2:?$USAGE}
SERVER_GPU=${3:?$USAGE}
CLIENT_GPU=${4:?$USAGE}
shift 4
EXTRA_ARGS="$*"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=${TASK:-G1WholebodyLocomotionPickBetweenTablesTeleop-v0}
PSI_DIR=${PSI_DIR:-/hfm/songlin/psi}
case "$TASK" in
  G1WholebodyLocomotionPickBetweenTablesTeleop-v0)
    DEFAULT_RUN=g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126 ;;
  G1WholebodyBendPickTeleop-v0)
    DEFAULT_RUN=g1wholebodybendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605230955 ;;
  *) DEFAULT_RUN= ;;
esac
RUN_DIR=${RUN_DIR:-${DEFAULT_RUN:+.runs/hf/psi0/simple-checkpoints/$DEFAULT_RUN}}
[[ -n "$RUN_DIR" ]] || { echo "error: no default checkpoint for $TASK, set RUN_DIR" >&2; exit 1; }
CKPT_STEP=${CKPT_STEP:-40000}
IMAGE_DATE=${IMAGE_DATE:-20261027}
ISAAC_RENDER=${ISAAC_RENDER:-on-query}
TMUX_TARGET=${TMUX_TARGET:-}
CLIENT_ONLY=${CLIENT_ONLY:-0}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}

DATA_DIR="data/SIMPLE-eval/$TASK/$DR"
LOG_DIR="logs/$STAMP"
EVAL_DIR="$LOG_DIR/evals_decoupled_wbc_$DR"
WINDOW="eval-$DR-$PORT"

# Sanity checks: catch port / dr-level mistakes before anything starts.
[[ -f "$ROOT_DIR/$DATA_DIR/meta/info.json" ]] || {
  echo "error: $DATA_DIR/meta/info.json not found (available: $(ls "$ROOT_DIR/data/SIMPLE-eval/$TASK" 2>/dev/null | tr '\n' ' '))" >&2
  exit 1
}
if [[ "$CLIENT_ONLY" == 1 ]]; then
  ss -ltn | grep -q ":$PORT " || { echo "error: CLIENT_ONLY=1 but no server is listening on port $PORT" >&2; exit 1; }
elif ss -ltn | grep -q ":$PORT "; then
  echo "error: port $PORT is already in use" >&2
  exit 1
fi
[[ -d "$PSI_DIR/$RUN_DIR" ]] || { echo "error: run dir $PSI_DIR/$RUN_DIR not found" >&2; exit 1; }
command -v tmux >/dev/null || { echo "error: tmux not found" >&2; exit 1; }
mkdir -p "$ROOT_DIR/$LOG_DIR"

SERVER_CMD="CUDA_VISIBLE_DEVICES=$SERVER_GPU HF_HOME=\${HF_HOME:-/hfm/cache} .venv-psi/bin/serve_psi0_simple \
--host 0.0.0.0 --port $PORT --policy=psi0 --run-dir=$RUN_DIR --ckpt-step=$CKPT_STEP --action-exec-horizon=24 --rtc 2>&1 | tee $ROOT_DIR/$LOG_DIR/psi0_server_$DR.log"

# --headless is required: there is no X display, and Isaac segfaults on the first
# render step when it can't create a window.
# The client's live progress display (rich) only renders on a terminal, so its output is logged
# with `script` (keeps a pty) instead of piping it through tee.
if [[ "$TASK" == *MP-v0 ]]; then
  # motion-planning data (g1_wholebody, AMO lower body): cli/eval.py with the psi0 agent; the videos
  # go into the eval dir
  EVAL_DIR="$LOG_DIR/evals_$DR"
  CLIENT_ARGS="eval simple/$TASK psi0 $DR --host=localhost --port=$PORT --sim-mode=mujoco_isaac --headless \
--data-format=lerobot --data-dir=$DATA_DIR --eval-dir=$EVAL_DIR --isaac-render=$ISAAC_RENDER"
else
  CLIENT_ARGS="eval-decoupled-wbc simple/$TASK psi0_decoupled_wbc $DR --host=localhost --port=$PORT \
--sim-mode=mujoco_isaac --headless --data-format=lerobot --data-dir=$DATA_DIR --eval-dir=$EVAL_DIR \
--video-dir=$LOG_DIR/videos --isaac-render=$ISAAC_RENDER"
fi
CLIENT_CMD="until ss -ltn | grep -q ':$PORT '; do sleep 2; done; \
script -qfec 'DATE=$IMAGE_DATE GPUs=$CLIENT_GPU docker compose run --rm $CLIENT_ARGS $EXTRA_ARGS' $LOG_DIR/eval_client_$DR.log"

if [[ -n "$TMUX_TARGET" ]]; then
  WIN_ID=$(tmux new-window -P -F '#{window_id}' -t "$TMUX_TARGET:" -n "$WINDOW" -c "$PSI_DIR")
else
  WIN_ID=$(tmux new-window -P -F '#{window_id}' -n "$WINDOW" -c "$PSI_DIR")
fi
if [[ "$CLIENT_ONLY" == 1 ]]; then
  tmux send-keys -t "$WIN_ID" "cd $ROOT_DIR && $CLIENT_CMD" Enter
  echo "launched tmux window '$WINDOW': client gpu$CLIENT_GPU $DR -> $LOG_DIR (existing server :$PORT)"
  exit 0
fi
tmux send-keys -t "$WIN_ID" "$SERVER_CMD" Enter
CLIENT_PANE=$(tmux split-window -h -P -F '#{pane_id}' -t "$WIN_ID" -c "$ROOT_DIR")
tmux send-keys -t "$CLIENT_PANE" "$CLIENT_CMD" Enter

echo "launched tmux window '$WINDOW': server gpu$SERVER_GPU :$PORT | client gpu$CLIENT_GPU $DR -> $LOG_DIR"
