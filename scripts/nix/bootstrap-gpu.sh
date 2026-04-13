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

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.7}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

if ! simple_runtime_assert_nvidia_runtime "bootstrap-gpu" "$ROOT_DIR"; then
  echo "[bootstrap-gpu] host NVIDIA runtime is incomplete for GPU builds"
  exit 1
fi

if ! simple_runtime_detect_torch_cuda_arch_list "bootstrap-gpu"; then
  echo "[bootstrap-gpu] failed to determine TORCH_CUDA_ARCH_LIST automatically"
  echo "[bootstrap-gpu] set TORCH_CUDA_ARCH_LIST explicitly and rerun"
  exit 1
fi

export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
simple_runtime_filter_path_var LD_LIBRARY_PATH $(simple_runtime_allowed_ld_prefixes "$ROOT_DIR")
simple_runtime_stage_python_runtime_libs

if ! simple_runtime_assert_runtime_boundary "bootstrap-gpu" "$ROOT_DIR"; then
  exit 1
fi

if ! simple_runtime_assert_vendor_tree "bootstrap-gpu" "nvidia-curobo" "$ROOT_DIR/third_party/curobo"; then
  exit 1
fi

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[bootstrap-gpu] missing venv python at $VENV_PYTHON"
  echo "[bootstrap-gpu] run ./scripts/nix/bootstrap-python.sh first"
  exit 1
fi

compute_curobo_key() {
  local submodule_key
  local torch_key
  if git -C "$ROOT_DIR/third_party/curobo" rev-parse HEAD >/dev/null 2>&1; then
    submodule_key="$(
      git -C "$ROOT_DIR/third_party/curobo" rev-parse HEAD
      git -C "$ROOT_DIR/third_party/curobo" status --short
    )"
  else
    submodule_key="$(
      find "$ROOT_DIR/third_party/curobo" -type f ! -path '*/.git/*' -print0 \
        | sort -z \
        | xargs -0 sha256sum
    )"
  fi

  torch_key="$(
    "$VENV_PYTHON" - <<'PY'
import pathlib
import torch

print(torch.__version__)
print(torch.version.cuda)
print(pathlib.Path(torch.__file__).resolve())
PY
  )"

  {
    printf '%s\n' "$UV_PROJECT_ENVIRONMENT"
    printf '%s\n' "$TORCH_CUDA_ARCH_LIST"
    printf '%s\n' "$torch_key"
    sha256sum "$ROOT_DIR/scripts/install_curobo.sh"
    sha256sum "$ROOT_DIR/scripts/nix/bootstrap-gpu.sh"
    printf '%s\n' "$submodule_key"
  } | sha256sum | cut -d' ' -f1
}

curobo_installed() {
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib
import sys

mods = [
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]
for mod in mods:
    importlib.import_module(mod)
sys.exit(0)
PY
}

curobo_stamp="${SIMPLE_CUROBO_STAMP:-$ROOT_DIR/.nix-curobo.stamp}"
curobo_key="$(compute_curobo_key)"
current_curobo_key=""
if [[ -f "$curobo_stamp" ]]; then
  current_curobo_key="$(cat "$curobo_stamp" || true)"
fi

if [[ "$curobo_key" != "$current_curobo_key" ]] || ! curobo_installed; then
  echo "[bootstrap-gpu] installing curobo + warp pin"
  "$ROOT_DIR/scripts/install_curobo.sh"
  echo "$curobo_key" > "$curobo_stamp"
else
  echo "[bootstrap-gpu] curobo install up to date"
fi

echo "[bootstrap-gpu] done"
