#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# One eval lane: pull (task, dr-level) jobs from the shared queue and run each eval client against
# this lane's psi0 server (already started in the server pane). Mirrors the client command of
# scripts/launch_psi0_eval_pair.sh, plus a read-only bind mount of the eval set into the container.
#
# usage: lane.sh LANE PORT CLIENT_GPU SERVER_PANE   (started by launch.sh, which sets the env below)
set -uo pipefail
LANE=$1 PORT=$2 CLIENT_GPU=$3 SERVER_PANE=$4
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:-}
CLIENT_THREADS=${CLIENT_THREADS:-8}
: "${ROOT_DIR:?}" "${LOG_ROOT:?}" "${EVAL_SRC:?}" "${IMAGE_DATE:?}" "${ISAAC_RENDER:?}"
cd "$ROOT_DIR"
QUEUE=$LOG_ROOT/queue.txt
STATUS=$LOG_ROOT/status

pop_job() {  # atomically take the first line of the queue
  (
    flock 9
    line=$(head -n 1 "$QUEUE")
    [[ -n "$line" ]] && sed -i '1d' "$QUEUE"
    echo "$line"
  ) 9>"$QUEUE.lock"
}

until ss -ltn | grep -q ":$PORT "; do sleep 5; done
echo "[lane $LANE] server :$PORT up"

while true; do
  job=$(pop_job)
  [[ -z "$job" ]] && break
  read -r TASK DR <<<"$job"
  LOG_DIR="${LOG_ROOT#$ROOT_DIR/}/$TASK"
  mkdir -p "$ROOT_DIR/$LOG_DIR"
  DATA_DIR="data/evals/simple-eval/$TASK/$DR"
  if [[ "$TASK" == *MP-v0 ]]; then
    EVAL_DIR="$LOG_DIR/evals_$DR"
    ARGS="eval simple/$TASK psi0 $DR --host=localhost --port=$PORT --sim-mode=mujoco_isaac --headless \
--data-format=lerobot --data-dir=$DATA_DIR --eval-dir=$EVAL_DIR --isaac-render=$ISAAC_RENDER"
  else
    EVAL_DIR="$LOG_DIR/evals_decoupled_wbc_$DR"
    ARGS="eval-decoupled-wbc simple/$TASK psi0_decoupled_wbc $DR --host=localhost --port=$PORT \
--sim-mode=mujoco_isaac --headless --data-format=lerobot --data-dir=$DATA_DIR --eval-dir=$EVAL_DIR \
--video-dir=$LOG_DIR/videos_$DR --isaac-render=$ISAAC_RENDER"
  fi
  echo "running lane=$LANE port=$PORT gpu=$CLIENT_GPU eval_dir=$EVAL_DIR start=$(date -u +%FT%TZ)" > "$STATUS/${TASK}__${DR}"
  echo "[lane $LANE] $(date -u +%T) START $TASK $DR"
  script -qfec "DATE=$IMAGE_DATE GPUs=$CLIENT_GPU docker compose run --rm \
-e OMP_NUM_THREADS=$CLIENT_THREADS -e MKL_NUM_THREADS=$CLIENT_THREADS \
-v $EVAL_SRC:/workspace/simple/data/evals/simple-eval:ro $ARGS $EXTRA_EVAL_ARGS" "$LOG_DIR/eval_client_$DR.log" > /dev/null
  rc=$?
  echo "done rc=$rc lane=$LANE eval_dir=$EVAL_DIR end=$(date -u +%FT%TZ)" > "$STATUS/${TASK}__${DR}"
  echo "[lane $LANE] $(date -u +%T) END rc=$rc $TASK $DR"
done

echo "[lane $LANE] queue empty, stopping server :$PORT"
tmux send-keys -t "$SERVER_PANE" C-c
