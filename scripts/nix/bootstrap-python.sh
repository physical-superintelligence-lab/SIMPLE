#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/nix/common.sh"

simple_runtime_source_home_env
simple_runtime_configure_uv_project_environment "$ROOT_DIR"
simple_runtime_sanitize_python_env
simple_runtime_stage_host_driver_libs "$ROOT_DIR"

if [[ -e "$ROOT_DIR/.venv" && ! -w "$ROOT_DIR/.venv" ]]; then
  echo "[bootstrap-python] detected non-writable $ROOT_DIR/.venv"
  echo "[bootstrap-python] this usually comes from running docker probe with bind mounts as root"
  echo "[bootstrap-python] fix once: sudo rm -rf $ROOT_DIR/.venv"
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.7}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

if ! simple_runtime_assert_nvidia_runtime "bootstrap-python" "$ROOT_DIR"; then
  echo "[bootstrap-python] host NVIDIA runtime is incomplete"
  exit 1
fi

if ! simple_runtime_detect_torch_cuda_arch_list "bootstrap-python"; then
  echo "[bootstrap-python] failed to determine TORCH_CUDA_ARCH_LIST automatically"
  echo "[bootstrap-python] set TORCH_CUDA_ARCH_LIST explicitly and rerun"
  exit 1
fi

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
simple_runtime_filter_path_var LD_LIBRARY_PATH $(simple_runtime_allowed_ld_prefixes "$ROOT_DIR")

if ! simple_runtime_assert_runtime_boundary "bootstrap-python" "$ROOT_DIR"; then
  exit 1
fi

if ! simple_runtime_assert_vendor_tree "bootstrap-python" "nvidia-curobo" "$ROOT_DIR/third_party/curobo"; then
  exit 1
fi

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[bootstrap-python] missing venv python at $VENV_PYTHON"
  echo "[bootstrap-python] creating venv with:"
  echo "  uv venv --python \"${UV_PYTHON:-python3.10}\" \"$UV_PROJECT_ENVIRONMENT\""
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
fi

echo "[bootstrap-python] seeding build backend tools into $UV_PROJECT_ENVIRONMENT"
uv pip install --python "$VENV_PYTHON" setuptools wheel

if [[ -d "$ROOT_DIR/third_party/evdev" ]]; then
  echo "[bootstrap-python] preinstalling vendored evdev with Nix toolchain"
  rm -rf \
    "$ROOT_DIR/third_party/evdev/build" \
    "$ROOT_DIR/third_party/evdev/src/evdev.egg-info"
  uv pip install --python "$VENV_PYTHON" --no-build-isolation "$ROOT_DIR/third_party/evdev"
fi

echo "[bootstrap-python] syncing uv env (docker-parity groups)"
uv_sync_args=(
  --python "$VENV_PYTHON"
  --group sonic
  --group rlds
  --group lerobot
  --group isaacsim-hotfix
  --no-install-package nvidia-curobo
  --index-strategy unsafe-best-match
)

if [[ -d "$ROOT_DIR/third_party/evdev" ]]; then
  uv_sync_args+=(--no-install-package evdev)
fi

run_uv_sync() {
  local log_file="$1"
  shift || true
  uv sync "${uv_sync_args[@]}" "$@" > >(tee "$log_file") 2> >(tee -a "$log_file" >&2)
}

extract_uv_failed_package() {
  local log_file="$1"
  sed -nE 's/.*\(([[:alnum:]_.-]+)==[^)]*\).*/\1/p' "$log_file" | tail -n1
}

sync_log="$(mktemp)"
if ! run_uv_sync "$sync_log"; then
  if grep -Eq 'The wheel is invalid|Missing \.dist-info directory|failed to read directory .*/archive-v0/' "$sync_log"; then
    failed_pkg="$(extract_uv_failed_package "$sync_log")"
    if [[ -n "$failed_pkg" ]]; then
      echo "[bootstrap-python] detected corrupt uv cache entry for $failed_pkg"
      echo "[bootstrap-python] clearing cached artifacts for $failed_pkg and retrying once"
      uv cache clean "$failed_pkg" >/dev/null 2>&1 || true
      retry_log="$(mktemp)"
      run_uv_sync "$retry_log" --refresh-package "$failed_pkg"
    else
      echo "[bootstrap-python] detected corrupt uv cache entry"
      echo "[bootstrap-python] pruning uv cache and retrying once"
      uv cache prune >/dev/null 2>&1 || true
      retry_log="$(mktemp)"
      run_uv_sync "$retry_log" --refresh
    fi
  else
    exit 1
  fi
fi

if [[ -d "$ROOT_DIR/third_party/evdev" ]]; then
  echo "[bootstrap-python] restoring vendored evdev after uv sync"
  uv pip install --python "$VENV_PYTHON" --no-build-isolation "$ROOT_DIR/third_party/evdev"
fi

echo "[bootstrap-python] installing local project in editable mode"
uv pip install --python "$VENV_PYTHON" --no-deps --editable .

echo "[bootstrap-python] done"
