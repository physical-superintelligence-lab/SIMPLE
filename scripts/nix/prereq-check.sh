#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/nix/common.sh"

MIN_NIX_FREE_GIB="${MIN_NIX_FREE_GIB:-80}"
MIN_ROOT_FREE_GIB="${MIN_ROOT_FREE_GIB:-20}"

failures=0
warnings=0

pass() {
  echo "[prereq-check] PASS: $1"
}

warn() {
  echo "[prereq-check] WARN: $1"
  warnings=$((warnings + 1))
}

fail() {
  echo "[prereq-check] FAIL: $1"
  failures=$((failures + 1))
}

bytes_to_gib() {
  awk -v bytes="$1" 'BEGIN { printf "%.1f", bytes / (1024 * 1024 * 1024) }'
}

free_bytes_for_path() {
  local path="$1"
  df -PB1 "$path" | awk 'NR==2 { print $4 }'
}

check_command() {
  local cmd="$1"
  local hint="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "$cmd is available"
  else
    fail "$cmd is missing${hint:+ ($hint)}"
  fi
}

check_optional_command() {
  local cmd="$1"
  local note="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "$cmd is available"
  else
    warn "$cmd is not installed in the outer shell${note:+ ($note)}"
  fi
}

check_direnv() {
  if command -v direnv >/dev/null 2>&1; then
    pass "direnv is available"
  else
    warn "direnv is not installed; you can still use 'env -u LD_LIBRARY_PATH nix develop', but direnv is the recommended entry path"
  fi
}

check_nix_loader_state() {
  if env -u LD_LIBRARY_PATH nix --version >/dev/null 2>&1; then
    pass "nix works with a clean loader environment"
  else
    fail "nix is not invocable even with LD_LIBRARY_PATH cleared"
    return 0
  fi

  if nix --version >/dev/null 2>&1; then
    pass "nix also works in the current shell"
  else
    warn "nix fails in the current shell; this usually means LD_LIBRARY_PATH is polluted outside the repo"
  fi
}

check_disk_space() {
  local nix_free root_free
  nix_free="$(free_bytes_for_path /nix)"
  root_free="$(free_bytes_for_path "$ROOT_DIR")"

  if (( nix_free >= MIN_NIX_FREE_GIB * 1024 * 1024 * 1024 )); then
    pass "/nix has $(bytes_to_gib "$nix_free") GiB free"
  else
    warn "/nix has only $(bytes_to_gib "$nix_free") GiB free; first bootstrap may fail while realizing CUDA packages"
  fi

  if (( root_free >= MIN_ROOT_FREE_GIB * 1024 * 1024 * 1024 )); then
    pass "workspace filesystem has $(bytes_to_gib "$root_free") GiB free"
  else
    warn "workspace filesystem has only $(bytes_to_gib "$root_free") GiB free; caches and virtualenv creation may fail"
  fi
}

check_host_nvidia() {
  simple_runtime_source_home_env
  if simple_runtime_assert_nvidia_runtime "prereq-check" "$ROOT_DIR" >/dev/null 2>&1; then
    pass "supported NVIDIA driver layout detected"
  else
    fail "supported NVIDIA driver files were not found (libcuda, Vulkan ICD, or EGL vendor JSON)"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi is available"
  else
    warn "nvidia-smi is missing from PATH; GPU arch detection may fail"
  fi
}

check_repo_state() {
  if [[ -f "$ROOT_DIR/uv.lock" && -f "$ROOT_DIR/pyproject.toml" && -f "$ROOT_DIR/flake.nix" ]]; then
    pass "repo contains flake.nix, pyproject.toml, and uv.lock"
  else
    fail "repo is missing one of flake.nix, pyproject.toml, or uv.lock"
  fi

  if simple_runtime_assert_vendor_tree "prereq-check" "nvidia-curobo" "$ROOT_DIR/third_party/curobo" >/dev/null 2>&1; then
    pass "third_party/curobo is initialized"
  else
    fail "third_party/curobo is missing or uninitialized"
  fi
}

check_user_shell_pollution() {
  local bad_vars=()
  [[ -n "${PYTHONPATH:-}" ]] && bad_vars+=("PYTHONPATH")
  [[ -n "${PYTHONHOME:-}" ]] && bad_vars+=("PYTHONHOME")
  [[ -n "${LD_PRELOAD:-}" ]] && bad_vars+=("LD_PRELOAD")
  [[ -n "${CONDA_PREFIX:-}" ]] && bad_vars+=("CONDA_PREFIX")

  if [[ "${#bad_vars[@]}" -eq 0 ]]; then
    pass "no obvious outer-shell Python/runtime overrides detected"
  else
    warn "outer shell exports ${bad_vars[*]}; the Nix runtime will sanitize them, but fresh shells are cleaner"
  fi

  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    warn "LD_LIBRARY_PATH is set in the current shell; use 'env -u LD_LIBRARY_PATH nix develop ...' when invoking nix directly"
  else
    pass "LD_LIBRARY_PATH is not set in the current shell"
  fi
}

echo "[prereq-check] checking SIMPLE host/runtime prerequisites"

check_command git "required for submodules"
check_command nix "required to realize the dev shell"
check_command bash ""
check_optional_command uv "the Nix dev shell provides uv even if the host does not"
check_optional_command git-lfs "the Nix dev shell provides git-lfs even if the host does not"
check_direnv
check_nix_loader_state
check_disk_space
check_host_nvidia
check_repo_state
check_user_shell_pollution

echo "[prereq-check] summary: $failures failure(s), $warnings warning(s)"

if (( failures > 0 )); then
  echo "[prereq-check] fix the failures above before relying on the Nix runtime"
  exit 1
fi

echo "[prereq-check] environment looks compatible with the SIMPLE Nix runtime"
