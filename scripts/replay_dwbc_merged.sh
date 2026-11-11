#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Plain MuJoCo replay of every task in the merged SIMPLE-decoupled-wbc dataset
# (HF SIMPLE-org/SIMPLE-decoupled-wbc, built by scripts/merge_simple_decoupled_wbc.py),
# one task after another, each sharded over the GPUs with docker.sh, then a per-task
# success-rate summary.
#
#   bash scripts/replay_dwbc_merged.sh [TASK_SUBSTRING ...]
#   e.g. bash scripts/replay_dwbc_merged.sh                 # all 17 tasks
#        bash scripts/replay_dwbc_merged.sh XMovePick BendPickMP
#        SUMMARY_ONLY=1 OUT=data/replays/dwbc-merged-<stamp> bash scripts/replay_dwbc_merged.sh
#
# The tasks are contiguous episode blocks of the merged dataset (merge_summary.json);
# each block is replayed with --episode-start/--total-episodes against the merged dir:
#   *Teleop-v0 -> simple.cli.replay_decoupled_wbc (decoupled WBC, 36-D psi0 actions)
#   *MP-v0     -> simple.cli.replay (g1_wholebody / AMO, psi0-format actions)
# Both stop an episode at the first success (eval-style).
#
# Env:
#   DATA            merged dataset under data/ (docker.sh only mounts data/; hard-link a
#                   copy in with `cp -al /hfm/data/simple/SIMPLE-decoupled-wbc data/`)
#                   default data/SIMPLE-decoupled-wbc
#   OUT             output root under data/ (default data/replays/dwbc-merged-<stamp>)
#   GPUS            GPU list for docker.sh (default 0,1,2,3,4,5,6,7)
#   SHARDS_PER_GPU  default 2
#   DATE            docker image tag simple:$DATE (default 20261027)
#   SUMMARY_ONLY=1  skip replays, only summarize OUT
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA=${DATA:-data/SIMPLE-decoupled-wbc}
OUT=${OUT:-data/replays/dwbc-merged-$(date +%y%m%d%H%M)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-2}
export DATE=${DATE:-20261027}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
FILTERS=("$@")

[[ -f "$DATA/merge_summary.json" ]] || { echo "error: $DATA/merge_summary.json not found" >&2; exit 1; }
[[ "$OUT" == data/* ]] || { echo "error: OUT must be under data/ (docker.sh mounts only data/)" >&2; exit 1; }
mkdir -p "$OUT"

# "source start count" per task, in merged episode order.
TASKS=$(python3 -I - "$DATA/merge_summary.json" <<'EOF'
import json, sys
start = 0
for t in json.load(open(sys.argv[1]))["tasks"]:
    print(t["source"], start, t["episodes"])
    start += t["episodes"]
EOF
)

summarize() {
  python3 -I - "$OUT" "$TASKS" <<'EOF'
import re, sys
from pathlib import Path
out, tasks = Path(sys.argv[1]), [l.split() for l in sys.argv[2].splitlines() if l.strip()]
rows, tot_s, tot_n = [], 0, 0
for src, start, count in tasks:
    tdir = out / src
    if not tdir.exists():
        continue
    res = {}
    # replay_decoupled_wbc: <save>/simple/<env>/level-0/meta/replay.txt  "ep: True"
    for f in tdir.rglob("replay.txt"):
        for line in f.read_text().splitlines():
            m = re.match(r"\s*(\d+):\s*(True|False)", line)
            if m: res[int(m[1])] = m[2] == "True"
    # replay: <save>/replay_stats.txt  "episode_<idx>: True"
    for f in tdir.rglob("replay_stats.txt"):
        for line in f.read_text().splitlines():
            m = re.match(r"\s*episode_(\d+):\s*(True|False)", line)
            if m: res[int(m[1])] = m[2] == "True"
    s, n, exp = sum(res.values()), len(res), int(count)
    tot_s += s; tot_n += n
    note = "" if n == exp else f"  (missing {exp - n} of {exp})"
    rows.append(f"| {src.replace('G1Wholebody', '').replace('-v0', '')} | {s}/{n} | {100 * s / max(n, 1):.1f}% |{note}")
print("| Task | Replay SR | % |\n|---|---|---|")
print("\n".join(rows))
print(f"| **Total** | **{tot_s}/{tot_n}** | **{100 * tot_s / max(tot_n, 1):.1f}%** |")
EOF
}

if [[ "$SUMMARY_ONLY" != 1 ]]; then
  echo "data=$DATA out=$OUT gpus=$GPUS shards/gpu=$SHARDS_PER_GPU image=simple:$DATE"
  while read -r src start count; do
    if (( ${#FILTERS[@]} )); then
      hit=0; for f in "${FILTERS[@]}"; do [[ "$src" == *"$f"* ]] && hit=1; done
      (( hit )) || continue
    fi
    echo "=== $(date +%H:%M:%S) $src: episodes [$start, $((start + count)))"
    common=(--gpus "$GPUS" --shards-per-gpu "$SHARDS_PER_GPU" --episode-start "$start"
            --total-episodes "$count" --save-dir="$OUT/$src")
    if [[ "$src" == *MP-v0 ]]; then
      bash docker.sh simple.cli.replay "simple/$src" --sim-mode=mujoco --headless \
        --max-episode-steps=1200 --data-dir="$DATA" "${common[@]}"
    else
      bash docker.sh simple.cli.replay_decoupled_wbc "simple/$src" --sim-mode=mujoco --headless \
        --no-webrtc --no-record --data-dir="$DATA" "${common[@]}"
    fi || echo "!!! $src: some shards failed (logs/docker_shards/${OUT#data/}/$src/)"
  done <<< "$TASKS"
fi

summarize | tee "$OUT/summary.md"
