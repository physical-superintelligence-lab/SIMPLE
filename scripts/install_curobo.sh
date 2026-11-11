#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Only source nix scripts if in nix shell
if [[ -n "${IN_NIX_SHELL:-}" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/nix/common.sh"

  simple_runtime_source_home_env
  simple_runtime_configure_uv_project_environment "$ROOT_DIR"
  simple_runtime_sanitize_python_env
  simple_runtime_stage_host_driver_libs "$ROOT_DIR"
else
  # Outside a nix shell (e.g., Docker): use the standard venv instead of .venv-nix.
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}"
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export GIT_LFS_SKIP_SMUDGE=1
export MAX_JOBS="${MAX_JOBS:-4}"

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[install_curobo] missing venv python at $VENV_PYTHON"
  # echo "[install_curobo] run ./scripts/nix/bootstrap-python.sh first"
  exit 1
fi

export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

if [[ -n "${IN_NIX_SHELL:-}" ]]; then
  # In nix: strip LD_LIBRARY_PATH down to only runtime-approved prefixes before
  # adding torch libs, then enforce the nix runtime boundary.
  simple_runtime_filter_path_var LD_LIBRARY_PATH $(simple_runtime_allowed_ld_prefixes "$ROOT_DIR")
fi

if [[ -n "${IN_NIX_SHELL:-}" ]]; then
  simple_runtime_stage_python_runtime_libs

  if ! simple_runtime_assert_runtime_boundary "install_curobo" "$ROOT_DIR"; then
    exit 1
  fi

  if ! simple_runtime_assert_vendor_tree "install_curobo" "nvidia-curobo" "$ROOT_DIR/third_party/curobo"; then
    exit 1
  fi

  apply_curobo_patch

  if ! simple_runtime_assert_nvidia_runtime "install_curobo" "$ROOT_DIR"; then
    echo "[install_curobo] host NVIDIA runtime is incomplete for GPU builds"
    exit 1
  fi

  if ! simple_runtime_detect_torch_cuda_arch_list "install_curobo"; then
    echo "[install_curobo] failed to determine TORCH_CUDA_ARCH_LIST automatically"
    echo "[install_curobo] set TORCH_CUDA_ARCH_LIST explicitly and rerun"
    exit 1
  fi
else
  # Outside nix (e.g. Docker): prefer the local GPU's compute capability; fall back to
  # 8.0+PTX when no GPU is visible (e.g. during `docker build`).
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    detected_arch="$("$VENV_PYTHON" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())' 2>/dev/null || true)"
    if [[ -n "$detected_arch" ]]; then
      echo "[install_curobo] TORCH_CUDA_ARCH_LIST not set, using detected GPU arch $detected_arch"
      export TORCH_CUDA_ARCH_LIST="$detected_arch"
    else
      echo "[install_curobo] TORCH_CUDA_ARCH_LIST not set, using default 8.0+PTX"
      export TORCH_CUDA_ARCH_LIST="8.0+PTX"
    fi
  fi
fi

if [[ -n "${IN_NIX_SHELL:-}" ]]; then
  if [[ -z "${SIMPLE_LIBC_DEV:-}" || ! -d "${SIMPLE_LIBC_DEV}/include" ]]; then
    echo "[install_curobo] missing libc development headers path"
    echo "[install_curobo] expected SIMPLE_LIBC_DEV/include to exist"
    exit 1
  fi
fi

echo "[install_curobo] using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[install_curobo] using Python environment at: $VENV_PYTHON"

# --no-build-isolation builds against the venv's torch, so curobo's build requirements must
# already be installed there. setuptools-scm>=10 imports vcs-versioning, which is not part of
# the project lock.
echo "[install_curobo] installing build requirements"
uv pip install --python "$VENV_PYTHON" setuptools "setuptools-scm>=6.2" vcs-versioning wheel ninja

uv pip install --python "$VENV_PYTHON" --no-build-isolation 'third_party/curobo[isaacsim]'

# No warp-lang re-pin: curobo only needs warp-lang>=0.9.0, and isaaclab 3.0.0 requires the
# warp-lang==1.17.0 already installed from the lock.

echo "[install_curobo] verifying curobo CUDA extensions"
"$VENV_PYTHON" - <<'PY'
import sys
import torch

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import curobo.curobolib.lbfgs_step_cu
    import curobo.curobolib.kinematics_fused_cu
    import curobo.curobolib.line_search_cu
    import curobo.curobolib.tensor_step_cu
    import curobo.curobolib.geom_cu
    print("All CUDA extensions loaded successfully")
except ImportError as exc:
    print(f"CUDA extension import failed: {exc}")
    sys.exit(1)
PY

echo "[install_curobo] done"
