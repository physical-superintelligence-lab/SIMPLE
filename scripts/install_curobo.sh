#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}"

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

if [[ ! -d "$ROOT_DIR/third_party/curobo" ]]; then
  echo "[install_curobo] missing third_party/curobo"
  echo "[install_curobo] run: git submodule update --init --recursive"
  exit 1
fi

if [[ -n "${ROBO_NIX_LIBC_DEV:-}" ]]; then
  export SIMPLE_LIBC_DEV="${SIMPLE_LIBC_DEV:-$ROBO_NIX_LIBC_DEV}"
fi

if [[ -n "${ROBO_NIX_ACTIVE:-}" && ( -z "${SIMPLE_LIBC_DEV:-}" || ! -d "${SIMPLE_LIBC_DEV}/include" ) ]]; then
  echo "[install_curobo] missing libc development headers path"
  echo "[install_curobo] run this script inside 'robo shell'"
  exit 1
fi

if [[ -n "${ROBO_NIX_ACTIVE:-}" && -z "${ROBO_NIX_LIBCUDA_PATH:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[install_curobo] warning: robo-nix did not report a host CUDA driver path" >&2
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$("$VENV_PYTHON" - <<'PY'
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")
else:
    print("8.0+PTX")
PY
)"
  export TORCH_CUDA_ARCH_LIST
fi

echo "[install_curobo] using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[install_curobo] using Python environment at: $VENV_PYTHON"

uv pip install --python "$VENV_PYTHON" setuptools_scm wheel

CUROBO_VERSION="$("$VENV_PYTHON" - <<'PY'
import setuptools_scm

print(
    setuptools_scm.get_version(
        root="third_party/curobo",
        version_scheme="no-guess-dev",
        local_scheme="dirty-tag",
    )
)
PY
)"
export SETUPTOOLS_SCM_PRETEND_VERSION="$CUROBO_VERSION"

uv pip install --python "$VENV_PYTHON" --no-build-isolation 'third_party/curobo[isaacsim]'

echo "[install_curobo] pinning warp-lang"
uv pip install --python "$VENV_PYTHON" "warp-lang==1.7.0" --index-strategy unsafe-best-match

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
