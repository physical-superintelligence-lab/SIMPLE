#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Run a python module in the `sim` container, optionally sharded across GPUs.
#
#   bash docker.sh MODULE [--num-gpus N | --gpus LIST] [--shards-per-gpu K]
#                         [--total-episodes T] [module args...]
#
# Without --total-episodes: one run on GPU 0 (or $GPUs), args passed through.
# With it: episodes [--episode-start, +T) split into N*K contiguous shards of
# about equal frames (from <data-dir>/meta/episodes.jsonl, else equal episode
# counts) over GPUs 0..N-1, or the GPUs in LIST (e.g. 6,7); shard i runs on
# GPU i mod N (K shards per GPU), saving to <save-dir>/shard-i when N*K > 1.
#
#   bash docker.sh simple.cli.replay_decoupled_wbc \
#       simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
#       --data-dir=data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
#       --sim-mode=mujoco_isaac --headless --record \
#       --save-dir=data/simple_replay --success-criteria=0.2 \
#       --num-gpus 8 --total-episodes 100
#
#   # MuJoCo-only replay, 3 shards on each of GPUs 6 and 7:
#   bash docker.sh simple.cli.replay_decoupled_wbc \
#       simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
#       --data-dir=data/train-psi-data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
#       --sim-mode=mujoco --headless --no-record --save-dir=data/replay_train \
#       --gpus 6,7 --shards-per-gpu 3 --total-episodes 99
#
# Host paths under data/ become container paths; other host paths (e.g. plots/)
# are not mounted, so point outputs such as --replay-dir under data/.  Shard logs:
# logs/docker_shards/<save-dir under data/>/shard-i.log.
# Keep WebRTC off when sharding: containers share the host network.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_PWD="$PWD"
cd "$ROOT_DIR"

PYTHON=/workspace/simple/.venv/bin/python

usage() {
  echo "Usage: $0 MODULE [--num-gpus N] [--total-episodes T] [module args...]" >&2
  exit 1
}

[[ $# -ge 1 && "$1" != -* ]] || usage
MODULE="$1"
shift

# Pull out our options and the args sharding rewrites; the rest go to the module.
NUM_GPUS=1
GPU_LIST=""
SHARDS_PER_GPU=1
TOTAL_EPISODES=""
EPISODE_START=0
NUM_EPISODES_ARG=()
SAVE_DIR=""
MODULE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --num-gpus=*) NUM_GPUS="${1#*=}"; shift ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --gpus=*) GPU_LIST="${1#*=}"; shift ;;
    --shards-per-gpu) SHARDS_PER_GPU="$2"; shift 2 ;;
    --shards-per-gpu=*) SHARDS_PER_GPU="${1#*=}"; shift ;;
    --total-episodes) TOTAL_EPISODES="$2"; shift 2 ;;
    --total-episodes=*) TOTAL_EPISODES="${1#*=}"; shift ;;
    --episode-start) EPISODE_START="$2"; shift 2 ;;
    --episode-start=*) EPISODE_START="${1#*=}"; shift ;;
    --num-episodes) NUM_EPISODES_ARG=("$1" "$2"); shift 2 ;;
    --num-episodes=*) NUM_EPISODES_ARG=("$1"); shift ;;
    --save-dir) SAVE_DIR="$2"; shift 2 ;;
    --save-dir=*) SAVE_DIR="${1#*=}"; shift ;;
    *) MODULE_ARGS+=("$1"); shift ;;
  esac
done
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--num-gpus must be a positive integer" >&2; usage; }
[[ "$SHARDS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "--shards-per-gpu must be a positive integer" >&2; usage; }
if [[ -n "$GPU_LIST" ]]; then
  [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus must be a comma-separated list of GPU ids" >&2; usage; }
  IFS=, read -r -a GPUS <<< "$GPU_LIST"
else
  GPUS=()
  for (( i = 0; i < NUM_GPUS; i++ )); do GPUS+=("$i"); done
fi
NUM_SHARDS=$(( ${#GPUS[@]} * SHARDS_PER_GPU ))

# Data dir mount, as in docker-compose.yml.
HOST_DATA="$(cd "$ROOT_DIR" && realpath -m "${DATA_DIR:-./data}")"
CONTAINER_DATA=/workspace/simple/data

# Host path under the data dir -> container path, for VALUE or --opt=VALUE.
# Only clear paths (/, ./, ../, data/) are touched, so task ids pass through.
to_container() {
  local arg="$1" prefix="" val abs
  if [[ "$arg" == --*=* ]]; then prefix="${arg%%=*}="; val="${arg#*=}"; else val="$arg"; fi
  case "$val" in
    /*|./*|../*|data|data/*) ;;
    *) printf '%s' "$arg"; return ;;
  esac
  abs="$(cd "$CALLER_PWD" && realpath -m "$val")"
  if [[ "$abs" == "$HOST_DATA" || "$abs" == "$HOST_DATA"/* ]]; then
    val="$CONTAINER_DATA${abs#"$HOST_DATA"}"
  fi
  printf '%s' "$prefix$val"
}
for i in "${!MODULE_ARGS[@]}"; do MODULE_ARGS[$i]="$(to_container "${MODULE_ARGS[$i]}")"; done
[[ -n "$SAVE_DIR" ]] && SAVE_DIR="$(to_container "$SAVE_DIR")"

# Quote args for the container's `bash -c`.
build_cmd() {
  local cmd="$PYTHON -m $(printf '%q' "$MODULE")" arg
  for arg in "$@"; do cmd+=" $(printf '%q' "$arg")"; done
  printf '%s' "$cmd"
}

# `docker compose run -T` doesn't forward Ctrl+C, so send SIGINT to the named
# containers ourselves (SIGTERM is ignored by python as PID 1); kill after 30s.
NAMES=()
PIDS=()
cleanup() {
  trap - INT TERM
  echo
  echo "Stopping containers..."
  for name in "${NAMES[@]}"; do docker kill -s INT "$name" >/dev/null 2>&1 || true; done
  local alive
  for (( i = 0; i < 60; i++ )); do
    alive=0
    for name in "${NAMES[@]}"; do docker inspect "$name" >/dev/null 2>&1 && alive=1; done
    (( alive )) || break
    sleep 0.5
  done
  for name in "${NAMES[@]}"; do docker kill "$name" >/dev/null 2>&1 || true; done
  exit 130
}
trap cleanup INT TERM

# No sharding: single run, output to this terminal.
if [[ -z "$TOTAL_EPISODES" ]]; then
  (( NUM_SHARDS == 1 )) || { echo "more than one GPU or shard requires --total-episodes" >&2; usage; }
  args=("${MODULE_ARGS[@]}" "${NUM_EPISODES_ARG[@]}")
  [[ "$EPISODE_START" != 0 ]] && args+=("--episode-start=$EPISODE_START")
  [[ -n "$SAVE_DIR" ]] && args+=("--save-dir=$SAVE_DIR")
  NAMES+=("simple-run-$$")
  GPUs="${GPU_LIST:-${GPUs:-0}}" docker compose run --rm -T --name "${NAMES[0]}" sim -c "$(build_cmd "${args[@]}")" &
  wait $!
  exit $?
fi

[[ "$TOTAL_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "--total-episodes must be a positive integer" >&2; usage; }
(( ${#NUM_EPISODES_ARG[@]} == 0 )) || echo "Ignoring ${NUM_EPISODES_ARG[*]}: --total-episodes sets each shard's count" >&2

# One log dir per run, named after the save dir under data/ (else the module),
# so runs for different tasks don't overwrite each other's shard logs.
if [[ "$SAVE_DIR" == "$CONTAINER_DATA"/* ]]; then RUN_NAME="${SAVE_DIR#"$CONTAINER_DATA"/}"
elif [[ -n "$SAVE_DIR" ]]; then RUN_NAME="${SAVE_DIR#/}"
else RUN_NAME="$MODULE"; fi
LOG_DIR="$ROOT_DIR/logs/docker_shards/$RUN_NAME"
mkdir -p "$LOG_DIR"

# Host location of --data-dir's meta/episodes.jsonl, for frame counts.
EPISODES_JSONL=""
for (( i = 0; i < ${#MODULE_ARGS[@]}; i++ )); do
  case "${MODULE_ARGS[$i]}" in
    --data-dir=*) data_dir="${MODULE_ARGS[$i]#*=}" ;;
    --data-dir) data_dir="${MODULE_ARGS[$((i + 1))]:-}" ;;
    *) continue ;;
  esac
  [[ "$data_dir" == "$CONTAINER_DATA" || "$data_dir" == "$CONTAINER_DATA"/* ]] && data_dir="$HOST_DATA${data_dir#"$CONTAINER_DATA"}"
  EPISODES_JSONL="$data_dir/meta/episodes.jsonl"
done

# Print "start count frames" per shard: contiguous ranges cut nearest each 1/N
# of the total frames, or equal episode counts without episodes.jsonl.
plan_shards() {
  if [[ -f "$EPISODES_JSONL" ]]; then
    python3 -I - "$EPISODES_JSONL" "$EPISODE_START" "$TOTAL_EPISODES" "$NUM_SHARDS" <<'EOF'
import json, sys

path, start, total, n = sys.argv[1], *map(int, sys.argv[2:])
lengths = {}
with open(path) as f:
    for line in f:
        if line.strip():
            ep = json.loads(line)
            lengths[ep["episode_index"]] = ep["length"]
missing = [i for i in range(start, start + total) if i not in lengths]
if missing:
    sys.exit(f"episodes missing from {path}: {missing[:10]}")

prefix = [0]
for i in range(start, start + total):
    prefix.append(prefix[-1] + lengths[i])
cuts = [0]
for k in range(1, n):
    target = prefix[-1] * k / n
    cuts.append(min(range(cuts[-1], total + 1), key=lambda j: abs(prefix[j] - target)))
cuts.append(total)
for a, b in zip(cuts, cuts[1:]):
    if b > a:
        print(start + a, b - a, prefix[b] - prefix[a])
EOF
  else
    echo "No ${EPISODES_JSONL:-meta/episodes.jsonl (no --data-dir)} on the host — splitting by episode count" >&2
    local per=$(( (TOTAL_EPISODES + NUM_SHARDS - 1) / NUM_SHARDS )) offset
    for (( offset = 0; offset < TOTAL_EPISODES; offset += per )); do
      echo "$(( EPISODE_START + offset )) $(( TOTAL_EPISODES - offset < per ? TOTAL_EPISODES - offset : per )) ?"
    done
  fi
}
SHARDS="$(plan_shards)"

shard=0
while read -r start count frames; do
  gpu="${GPUS[$(( shard % ${#GPUS[@]} ))]}"
  name="simple-shard${shard}-$$"
  log="$LOG_DIR/shard-${shard}.log"

  args=("${MODULE_ARGS[@]}" "--num-episodes=$count" "--episode-start=$start")
  if [[ -n "$SAVE_DIR" ]]; then
    if (( NUM_SHARDS > 1 )); then args+=("--save-dir=$SAVE_DIR/shard-$shard"); else args+=("--save-dir=$SAVE_DIR"); fi
  fi

  echo "[shard $shard] GPU $gpu: episodes [$start, $((start + count))), $frames frames  log: $log"
  GPUs="$gpu" docker compose run --rm -T --name "$name" sim -c "$(build_cmd "${args[@]}")" > "$log" 2>&1 &
  NAMES+=("$name")
  PIDS+=($!)
  shard=$(( shard + 1 ))
done <<< "$SHARDS"

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[shard $i] done"
  else
    echo "[shard $i] FAILED (exit $?) — see $LOG_DIR/shard-$i.log"
    status=1
  fi
done
exit "$status"
