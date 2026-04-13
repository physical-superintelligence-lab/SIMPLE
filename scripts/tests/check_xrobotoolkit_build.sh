#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_STATE_DIR="$ROOT_DIR/.nix-cache/check-xrobotoolkit-build"
TEST_NIX_CACHE_DIR="$TEST_STATE_DIR/nix-cache"
TEST_UV_CACHE_DIR="$TEST_STATE_DIR/uv-cache"
TEST_VENV_DIR="$ROOT_DIR/.venv-nix-xrobotoolkit"

echo "[check_xrobotoolkit_build] removing repo-local bootstrap artifacts"
rm -rf \
  "$TEST_STATE_DIR" \
  "$TEST_VENV_DIR" \
  "$ROOT_DIR/third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/build" \
  "$ROOT_DIR/third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"/*.egg-info

mkdir -p "$TEST_NIX_CACHE_DIR" "$TEST_UV_CACHE_DIR"

echo "[check_xrobotoolkit_build] rebuilding focused XRoboToolkit build under nix develop"
XDG_CACHE_HOME="$TEST_NIX_CACHE_DIR" \
UV_CACHE_DIR="$TEST_UV_CACHE_DIR" \
UV_PROJECT_ENVIRONMENT="$TEST_VENV_DIR" \
SIMPLE_RESPECT_HOST_UV_PROJECT_ENV=1 \
SIMPLE_AUTO_BOOTSTRAP=0 \
LD_LIBRARY_PATH="" \
nix develop -c bash -lc '
  set -euo pipefail

  echo "[check_xrobotoolkit_build] creating venv at $UV_PROJECT_ENVIRONMENT"
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"

  echo "[check_xrobotoolkit_build] seeding shell runtime"
  source ./scripts/nix/common.sh
  simple_runtime_source_home_env
  simple_runtime_configure_uv_project_environment "$PWD"
  simple_runtime_sanitize_python_env
  simple_runtime_stage_host_driver_libs "$PWD"
  export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
  export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

  echo "[check_xrobotoolkit_build] installing minimal build backend tools"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" setuptools wheel

  echo "[check_xrobotoolkit_build] building XRoboToolkit editable package only"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation \
    -e ./third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64

  echo "[check_xrobotoolkit_build] verifying import"
  "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'"'"'PY'"'"'
import xrobotoolkit_sdk
print("[check_xrobotoolkit_build] import ok", xrobotoolkit_sdk.__file__)
PY
'

echo "[check_xrobotoolkit_build] success"
