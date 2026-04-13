#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_STATE_DIR="$ROOT_DIR/.nix-cache/check-curobo-build"
TEST_NIX_CACHE_DIR="$TEST_STATE_DIR/nix-cache"
TEST_UV_CACHE_DIR="$TEST_STATE_DIR/uv-cache"
TEST_VENV_DIR="$ROOT_DIR/.venv-nix-curobo"

echo "[check_curobo_build] removing focused build state"
rm -rf \
  "$TEST_STATE_DIR" \
  "$TEST_VENV_DIR" \
  "$ROOT_DIR/third_party/curobo/build" \
  "$ROOT_DIR/third_party/curobo/src/nvidia_curobo.egg-info"
rm -f "$ROOT_DIR"/third_party/curobo/src/curobo/curobolib/*.so

mkdir -p "$TEST_NIX_CACHE_DIR" "$TEST_UV_CACHE_DIR"

echo "[check_curobo_build] rebuilding curobo under nix develop"
XDG_CACHE_HOME="$TEST_NIX_CACHE_DIR" \
UV_CACHE_DIR="$TEST_UV_CACHE_DIR" \
UV_PROJECT_ENVIRONMENT="$TEST_VENV_DIR" \
SIMPLE_RESPECT_HOST_UV_PROJECT_ENV=1 \
SIMPLE_AUTO_BOOTSTRAP=0 \
LD_LIBRARY_PATH="" \
nix develop -c bash -lc '
  set -euo pipefail

  echo "[check_curobo_build] creating venv at $UV_PROJECT_ENVIRONMENT"
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"

  echo "[check_curobo_build] seeding Python deps needed for curobo build"
  source ./scripts/nix/common.sh
  simple_runtime_source_home_env
  simple_runtime_configure_uv_project_environment "$PWD"
  simple_runtime_sanitize_python_env
  simple_runtime_stage_host_driver_libs "$PWD"
  export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
  export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" setuptools wheel
  uv sync \
    --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    --group sonic \
    --group isaacsim-hotfix \
    --no-install-package nvidia-curobo \
    --index-strategy unsafe-best-match

  echo "[check_curobo_build] installing curobo only"
  ./scripts/install_curobo.sh
  simple_runtime_stage_python_runtime_libs

  echo "[check_curobo_build] verifying CUDA extension imports"
  "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'"'"'PY'"'"'
import curobo.curobolib.geom_cu
import curobo.curobolib.kinematics_fused_cu
import curobo.curobolib.lbfgs_step_cu
import curobo.curobolib.line_search_cu
import curobo.curobolib.tensor_step_cu
print("[check_curobo_build] import ok")
PY
'

echo "[check_curobo_build] success"
