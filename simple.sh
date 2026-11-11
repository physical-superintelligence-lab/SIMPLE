#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Run code in a long-lived SIMPLE docker container.
#
#   ./simple.sh start [--gpu N]       create the container in the background ($GPUs from .env, else
#                                     GPU 0; N may be a list such as 0,1), with $WORKSPACE at
#                                     /workspace/user
#   ./simple.sh bash [args]           shell in the container      (docker exec -it simple bash)
#   ./simple.sh python [-e K=V]... SCRIPT [args]
#                                     run SCRIPT with SIMPLE's Python from /workspace/simple; a host
#                                     path under $WORKSPACE becomes its /workspace/user/... path,
#                                     and a repo-relative path such as scripts/x.py works as is;
#                                     the host's $DISPLAY is passed in, and a local X server is
#                                     told to accept the container's root user, so GUIs can open
#   ./simple.sh exec CMD [args]       run any command in /workspace/simple
#   ./simple.sh download [--list|--verify] [--force] [args]
#                                     pre-download the eval assets into $DATA_DIR (default ./data);
#                                     no training data.  Needs no running container; args go to
#                                     data/_download_eval17_assets.sh
#   ./simple.sh replay simple/TASK [args]
#                                     replay a dataset; TASK picks the module (*Teleop-v*:
#                                     simple.cli.replay_decoupled_wbc, *MP-v*: simple.cli.replay)
#                                     and every argument goes to it unchanged
#   ./simple.sh status | stop         show / remove the container
#
# Settings come from .env next to this script: DATE (image tag simple:$DATE), HF_TOKEN, DATA_DIR,
# and optionally GPUs, CONTAINER (default simple) and WORKSPACE (host dir mounted at
# /workspace/user, relative to this folder; default ./workspace).
#
# The container runs as root, so after `bash`, `python` and `exec` finish (or are interrupted),
# files root created under /workspace/user are handed back to the host user:group.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# .env is docker compose's dotenv file and need not be valid shell (values may hold unquoted spaces),
# so pick out the settings used here instead of sourcing it.
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    case "$key" in
      DATE|HF_TOKEN|DATA_DIR|GPUs|CONTAINER|WORKSPACE)
        value="${value%%[[:space:]]#*}"; value="${value%"${value##*[![:space:]]}"}"; value="${value%[\"\']}"; value="${value#[\"\']}"
        export "$key=$value" ;;
    esac
  done < "$ENV_FILE"
else
  echo "warning: $ENV_FILE not found (copy .env.sample); using defaults" >&2
fi

DATE="${DATE:-20261027}"
CONTAINER="${CONTAINER:-simple}"
WORKSPACE="${WORKSPACE:-workspace}"
[[ "$WORKSPACE" == /* ]] || WORKSPACE="$SCRIPT_DIR/$WORKSPACE"
WORKSPACE_DIR="$(realpath -m "$WORKSPACE")"   # docker needs an absolute path
export DATE

usage() { awk 'NR > 4 && !/^#/ { exit } NR > 4' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }

# -t only when attached to a terminal, so piped or scripted runs work too.
TTY=(-i)
[[ -t 0 && -t 1 ]] && TTY=(-it)

state() {
  local s
  s="$(docker container inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" || s=missing
  echo "${s:-missing}"
}

require_running() {
  [[ "$(state)" == running ]] || die "container $CONTAINER is not running; start it with: $0 start"
}

# Host path under $WORKSPACE_DIR -> /workspace/user/...; anything else is left as is.
to_container() {
  local p="$1" abs
  for abs in "$(realpath -m "$p")" "$(realpath -m "$WORKSPACE_DIR/$p")"; do
    if [[ -e "$abs" && ( "$abs" == "$WORKSPACE_DIR" || "$abs" == "$WORKSPACE_DIR"/* ) ]]; then
      printf '%s' "/workspace/user${abs#"$WORKSPACE_DIR"}"; return
    fi
  done
  printf '%s' "$p"
}

# Give root-owned files in the workspace back to the host user (the container runs as root;
# running it as the host user would lock Isaac Sim out of /root and its caches).
HOST_UID="$(id -u)" HOST_GID="$(id -g)"
fix_owner() {
  [[ "$HOST_UID" == 0 || "$(state)" != running ]] && return 0
  docker exec "$CONTAINER" find /workspace/user -uid 0 -exec chown -h "$HOST_UID:$HOST_GID" {} + \
    || echo "warning: could not chown new files in $WORKSPACE_DIR to $HOST_UID:$HOST_GID" >&2
}

# docker exec in the container, then fix ownership; keeps the command's exit code.
run_in_container() {
  require_running
  trap fix_owner EXIT
  local rc=0
  docker exec "$@" || rc=$?
  exit "$rc"
}

cmd_start() {
  local gpus="${GPUs:-0}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|--gpus) gpus="${2:?--gpu needs a value}"; shift 2 ;;
      --gpu=*|--gpus=*) gpus="${1#*=}"; shift ;;
      *) die "unknown start option: $1" ;;
    esac
  done
  [[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu must be a GPU id or a list like 0,1"
  case "$(state)" in
    running) die "container $CONTAINER is already running; '$0 stop' first to recreate it" ;;
    missing) ;;
    *) echo "removing stopped container $CONTAINER"; docker rm "$CONTAINER" >/dev/null ;;
  esac
  docker image inspect "simple:$DATE" >/dev/null 2>&1 \
    || die "image simple:$DATE not found; build it with: docker compose build isaac-sim"
  mkdir -p "$WORKSPACE_DIR"
  # compose resolves ${PWD}/.uv-cache and ./data against this checkout.
  # Named `compose run` containers share project "simple"; never flag or remove them as orphans.
  (cd "$SCRIPT_DIR" && GPUs="$gpus" COMPOSE_IGNORE_ORPHANS=true docker compose run -d --name "$CONTAINER" \
    -v "$WORKSPACE_DIR:/workspace/user" sim -c "sleep infinity")
  echo "started $CONTAINER (image simple:$DATE, GPU $gpus, $WORKSPACE_DIR -> /workspace/user)"
}

# The container's DISPLAY is frozen at `start`, so pass the current one, empty if there is none.
# It runs as root, which the X server refuses until allowed; the grant lasts until X restarts.
display_env() {
  local display="${DISPLAY:-}"
  if [[ "$display" == :* ]] && command -v xhost >/dev/null; then
    xhost +SI:localuser:root >/dev/null 2>&1 || echo "warning: xhost failed on $display; GUIs may not open" >&2
  fi
  DISPLAY_ENV=(-e "DISPLAY=$display")
}

cmd_python() {
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -e) extra+=(-e "${2:?-e needs KEY=VALUE}"); shift 2 ;;
      -e*) extra+=(-e "${1#-e}"); shift ;;
      *) break ;;
    esac
  done
  [[ $# -ge 1 ]] || die "usage: $0 python [-e KEY=VALUE]... SCRIPT [args]"
  local args=()
  if [[ "$1" == -* ]]; then args=("$@")   # python -m / -c ...: pass through
  else args=("$(to_container "$1")" "${@:2}"); fi
  display_env
  run_in_container "${TTY[@]}" -w /workspace/simple "${DISPLAY_ENV[@]}" "${extra[@]}" \
    "$CONTAINER" /workspace/simple/.venv/bin/python "${args[@]}"
}

# Pick the replay module from the env id.  Matching the whole argument keeps option values such as
# --data-dir=data/simple/XTeleop-v0 from counting, wherever the env id sits among the args.
cmd_replay() {
  local a module=""
  for a in "$@"; do
    if [[ "$a" =~ ^(simple/)?[A-Za-z0-9_]+(Teleop|MP)-v[0-9]+$ ]]; then
      case "${BASH_REMATCH[2]}" in
        Teleop) module=simple.cli.replay_decoupled_wbc ;;
        MP) module=simple.cli.replay ;;
      esac
      break
    fi
  done
  [[ -n "$module" ]] || die "replay needs a simple/*Teleop-v* or simple/*MP-v* env id; for anything else use: $0 python -m simple.cli.replay ..."
  cmd_python -m "$module" "$@"
}

[[ $# -ge 1 ]] || { usage; exit 1; }
cmd="$1"; shift
case "$cmd" in
  start) cmd_start "$@" ;;
  stop) docker rm -f "$CONTAINER" ;;
  status) echo "$CONTAINER: $(state)" ;;
  bash) run_in_container "${TTY[@]}" "$CONTAINER" bash "$@" ;;
  python) cmd_python "$@" ;;
  exec)
    [[ $# -ge 1 ]] || die "usage: $0 exec CMD [args]"
    run_in_container "${TTY[@]}" -w /workspace/simple "$CONTAINER" "$@" ;;
  download) exec bash "$SCRIPT_DIR/data/_download_eval17_assets.sh" "$@" ;;
  replay) cmd_replay "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; die "unknown command: $cmd" ;;
esac
