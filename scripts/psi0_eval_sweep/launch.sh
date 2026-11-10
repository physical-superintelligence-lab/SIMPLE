#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Launch a SIMPLE-eval sweep for one psi0 checkpoint: N lanes (tmux windows), each a psi0 server + a client
# loop pulling (task, dr-level) jobs from a shared queue, plus a window that keeps success_table.txt current.
# This folder's scripts are copied into the log folder, so a sweep records what it ran.
#
# usage: launch.sh RUN_DIR CKPT_STEP
#   RUN_DIR  psi0 run dir, relative to PSI_DIR
#   e.g. launch.sh .runs/finetune/allmp.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2611040055 40000
#        GPUS=4,5,6,7 LANES=8 TASK_FILTER='MP-v0$' launch.sh .runs/finetune/<run> 20000
#        TASK_FILTER='PickAndBendPlaceMP-v0$' LEVELS=dr-level-0 EXTRA_EVAL_ARGS=--max-episode-steps=1400 \
#          SERVER_ARGS=--action-exec-horizon=24 launch.sh .runs/finetune/<run> 20000   # no RTC
#
# Overridable via env:
#   NAME             log folder suffix and tmux window prefix (default: run dir name up to the first dot)
#   TASK_FILTER      grep -E regex over the task folders of EVAL_SRC (default: all, '-v0$')
#   LEVELS           space separated dr levels (default: dr-level-0 dr-level-1 dr-level-2 dr-level-hard)
#   GPUS             comma separated GPUs for servers and clients (default: all). Pass the free ones
#                    when a training run is using the others.
#   LANES            number of server + client pairs (default: 1 per GPU; 2 per GPU oversubscribed a
#                    128-core host with the decoupled-WBC clients)
#   CLIENT_THREADS   OMP_NUM_THREADS / MKL_NUM_THREADS inside each eval client container (default: 8)
#   BASE_PORT        lane w listens on BASE_PORT + w (default: 22100)
#   EPISODES         episodes per job; fewer finished ones mark the cell failed (default: 10)
#   SERVER_ARGS      serve_psi0_simple options (default: --action-exec-horizon=24 --rtc)
#   EXTRA_EVAL_ARGS  appended to every eval client command
#   EVAL_SRC         eval set, mounted read-only into the container (default: /hfm/data/simple/SIMPLE-eval)
#   PSI_DIR          psi repo, holds .venv-psi (default: /hfm/songlin/psi)
#   IMAGE_DATE       docker image tag, i.e. simple:$IMAGE_DATE (default: 260829)
#   ISAAC_RENDER     see launch_psi0_eval_pair.sh (default: on-query)
#   STAMP            log folder name under logs/ (default: <time>-<NAME>-ckpt<CKPT_STEP>)
#   DRY_RUN          1: print the jobs and lanes, start nothing
set -euo pipefail

USAGE="usage: $(basename "$0") RUN_DIR CKPT_STEP"
RUN_DIR=${1:?$USAGE}
CKPT_STEP=${2:?$USAGE}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$HERE/../.." && pwd)
RUN_DIR=${RUN_DIR%/}
PSI_DIR=${PSI_DIR:-/hfm/songlin/psi}
EVAL_SRC=${EVAL_SRC:-/hfm/data/simple/SIMPLE-eval}
IMAGE_DATE=${IMAGE_DATE:-260829}
ISAAC_RENDER=${ISAAC_RENDER:-on-query}
TASK_FILTER=${TASK_FILTER:--v0\$}
LEVELS=${LEVELS:-dr-level-0 dr-level-1 dr-level-2 dr-level-hard}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
IFS=, read -ra GPU_LIST <<<"$GPUS"
N_GPUS=${#GPU_LIST[@]}
LANES=${LANES:-$N_GPUS}
CLIENT_THREADS=${CLIENT_THREADS:-8}
BASE_PORT=${BASE_PORT:-22100}
EPISODES=${EPISODES:-10}
SERVER_ARGS=${SERVER_ARGS:---action-exec-horizon=24 --rtc}
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:-}
DRY_RUN=${DRY_RUN:-0}
RUN_NAME=$(basename "$RUN_DIR")
NAME=${NAME:-${RUN_NAME%%.*}}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)-$NAME-ckpt$CKPT_STEP}
LOG_ROOT=$ROOT_DIR/logs/$STAMP

# Sanity checks: catch mistakes before anything starts.
[[ -d "$PSI_DIR/$RUN_DIR/checkpoints/ckpt_$CKPT_STEP" ]] || {
  echo "error: $PSI_DIR/$RUN_DIR/checkpoints/ckpt_$CKPT_STEP not found" >&2; exit 1; }
command -v tmux >/dev/null || { echo "error: tmux not found" >&2; exit 1; }
[[ ! -e "$LOG_ROOT" ]] || { echo "error: $LOG_ROOT already exists" >&2; exit 1; }
mapfile -t TASKS < <(ls "$EVAL_SRC" | grep -E -- "$TASK_FILTER" || true)
[[ ${#TASKS[@]} -gt 0 ]] || { echo "error: no task in $EVAL_SRC matches '$TASK_FILTER'" >&2; exit 1; }
JOBS=()
for t in "${TASKS[@]}"; do
  for d in $LEVELS; do
    [[ -f "$EVAL_SRC/$t/$d/meta/info.json" ]] || { echo "error: $t/$d missing" >&2; exit 1; }
    JOBS+=("$t $d")
  done
done
for ((w = 0; w < LANES; w++)); do
  ss -ltn | grep -q ":$((BASE_PORT + w)) " && { echo "error: port $((BASE_PORT + w)) in use" >&2; exit 1; }
done

echo "${#JOBS[@]} jobs (${#TASKS[@]} tasks) -> $LOG_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  printf '  %s\n' "${JOBS[@]}"
  for ((w = 0; w < LANES; w++)); do
    echo "lane $w: server gpu${GPU_LIST[w % N_GPUS]} :$((BASE_PORT + w)) | client gpu${GPU_LIST[(w + N_GPUS / 2) % N_GPUS]}"
  done
  echo "DRY_RUN set - nothing started"
  exit 0
fi

mkdir -p "$LOG_ROOT/status" "$LOG_ROOT/servers" "$ROOT_DIR/data/evals/simple-eval"
cp "$HERE"/{launch.sh,lane.sh,table.py} "$LOG_ROOT/"
printf '%s\n' "${JOBS[@]}" > "$LOG_ROOT/jobs.txt"
cp "$LOG_ROOT/jobs.txt" "$LOG_ROOT/queue.txt"

LANE_ENV=$(printf '%q=%q ' ROOT_DIR "$ROOT_DIR" LOG_ROOT "$LOG_ROOT" EVAL_SRC "$EVAL_SRC" IMAGE_DATE "$IMAGE_DATE" \
  ISAAC_RENDER "$ISAAC_RENDER" EXTRA_EVAL_ARGS "$EXTRA_EVAL_ARGS" CLIENT_THREADS "$CLIENT_THREADS")

for ((w = 0; w < LANES; w++)); do
  PORT=$((BASE_PORT + w)); SGPU=${GPU_LIST[w % N_GPUS]}; CGPU=${GPU_LIST[(w + N_GPUS / 2) % N_GPUS]}
  WIN=$(tmux new-window -d -P -F '#{window_id}' -n "$NAME-L$w" -c "$PSI_DIR")
  SPANE=$(tmux display -p -t "$WIN" '#{pane_id}')
  tmux send-keys -t "$SPANE" "CUDA_VISIBLE_DEVICES=$SGPU HF_HOME=\${HF_HOME:-/hfm/cache} .venv-psi/bin/serve_psi0_simple \
--host 0.0.0.0 --port $PORT --policy=psi0 --run-dir=$RUN_DIR --ckpt-step=$CKPT_STEP $SERVER_ARGS \
2>&1 | tee $LOG_ROOT/servers/psi0_server_L$w.log" Enter
  CPANE=$(tmux split-window -d -h -P -F '#{pane_id}' -t "$WIN" -c "$ROOT_DIR")
  tmux send-keys -t "$CPANE" "$LANE_ENV bash $LOG_ROOT/lane.sh $w $PORT $CGPU $SPANE 2>&1 | tee $LOG_ROOT/servers/lane_L$w.log" Enter
  echo "lane $w: server gpu$SGPU :$PORT | client gpu$CGPU"
done

tmux new-window -d -n "$NAME-table" -c "$ROOT_DIR" \
  "$PSI_DIR/.venv-psi/bin/python $LOG_ROOT/table.py $LOG_ROOT 'psi0 $NAME ckpt_$CKPT_STEP on $EVAL_SRC' $EPISODES; \
echo table final; exec zsh"
echo "table: $LOG_ROOT/success_table.txt"
