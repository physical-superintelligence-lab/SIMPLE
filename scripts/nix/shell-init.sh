#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(simple_nix_repo_root)}"

simple_runtime_source_home_env
simple_runtime_configure_uv_project_environment "$ROOT_DIR"
simple_runtime_sanitize_python_env

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

export SIMPLE_RUNTIME_STATE_DIR="${SIMPLE_RUNTIME_STATE_DIR:-$ROOT_DIR/.runtime-state}"
mkdir -p "$SIMPLE_RUNTIME_STATE_DIR"

simple_runtime_use_writable_dir XDG_CACHE_HOME "$SIMPLE_RUNTIME_STATE_DIR/xdg-cache"
simple_runtime_use_writable_dir XDG_CONFIG_HOME "$SIMPLE_RUNTIME_STATE_DIR/xdg-config"
simple_runtime_use_writable_dir HF_HOME "$XDG_CACHE_HOME/huggingface"
simple_runtime_use_writable_dir HF_DATASETS_CACHE "$HF_HOME/datasets"
simple_runtime_use_writable_dir HUGGINGFACE_HUB_CACHE "$HF_HOME/hub"
simple_runtime_use_writable_dir MPLCONFIGDIR "$XDG_CONFIG_HOME/matplotlib"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.7}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

simple_runtime_stage_host_driver_libs "$ROOT_DIR"
simple_runtime_assert_nvidia_runtime "simple-dev" "$ROOT_DIR" || true

if simple_runtime_detect_torch_cuda_arch_list "simple-dev"; then
  echo "[simple-dev] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
else
  echo "[simple-dev] warning: could not detect TORCH_CUDA_ARCH_LIST automatically"
fi

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
simple_runtime_stage_python_runtime_libs
simple_runtime_assert_runtime_boundary "simple-dev" "$ROOT_DIR"

nix() {
  env -u LD_LIBRARY_PATH command nix "$@"
}

export SIMPLE_AUTO_BOOTSTRAP="${SIMPLE_AUTO_BOOTSTRAP:-1}"
export SIMPLE_BOOTSTRAP_STAMP="${SIMPLE_BOOTSTRAP_STAMP:-$ROOT_DIR/.nix-bootstrap.stamp}"

if [[ "$SIMPLE_AUTO_BOOTSTRAP" == "1" && -f "$ROOT_DIR/scripts/nix/bootstrap.sh" && -f "$ROOT_DIR/uv.lock" && -f "$ROOT_DIR/pyproject.toml" ]]; then
  if [[ -e "$ROOT_DIR/.venv" && ! -w "$ROOT_DIR/.venv" ]]; then
    echo "[simple-dev] skip auto-bootstrap: detected non-writable $ROOT_DIR/.venv"
    echo "[simple-dev] fix once: sudo rm -rf $ROOT_DIR/.venv"
  else
    BOOTSTRAP_KEY="$(
      cat "$ROOT_DIR/uv.lock" "$ROOT_DIR/pyproject.toml" \
        | sha256sum \
        | cut -d' ' -f1
    )"
    CURRENT_KEY=""
    if [[ -f "$SIMPLE_BOOTSTRAP_STAMP" ]]; then
      CURRENT_KEY="$(cat "$SIMPLE_BOOTSTRAP_STAMP" || true)"
    fi
    if [[ "${SIMPLE_FORCE_BOOTSTRAP:-0}" == "1" || "$BOOTSTRAP_KEY" != "$CURRENT_KEY" ]]; then
      echo "[simple-dev] bootstrap required, running scripts/nix/bootstrap.sh"
      if "$ROOT_DIR/scripts/nix/bootstrap.sh"; then
        echo "$BOOTSTRAP_KEY" > "$SIMPLE_BOOTSTRAP_STAMP"
        simple_runtime_stage_python_runtime_libs
      else
        echo "[simple-dev] bootstrap failed"
        echo "[simple-dev] rerun manually after fixing the cause: ./scripts/nix/bootstrap.sh"
        return 1
      fi
    fi
  fi
fi

echo "[simple-dev] nix shell ready"
echo "[simple-dev] python: $(python --version 2>&1)"
echo "[simple-dev] python path: $(command -v python)"
echo "[simple-dev] uv: $(uv --version 2>&1)"
echo "[simple-dev] uv project env: $UV_PROJECT_ENVIRONMENT"
echo "[simple-dev] vulkaninfo: $(command -v vulkaninfo || echo missing)"
echo "[simple-dev] nvidia-smi: $(command -v nvidia-smi || echo missing)"
