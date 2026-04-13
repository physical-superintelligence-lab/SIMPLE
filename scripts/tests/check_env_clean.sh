#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_STATE_DIR="$ROOT_DIR/.nix-cache/check-env-clean"
TEST_NIX_CACHE_DIR="$TEST_STATE_DIR/nix-cache"
TEST_UV_CACHE_DIR="$TEST_STATE_DIR/uv-cache"
TEST_VENV_DIR="$ROOT_DIR/.venv-nix-check-env"

echo "[check_env_clean] removing repo-local bootstrap artifacts"
rm -rf \
  "$TEST_STATE_DIR" \
  "$TEST_VENV_DIR" \
  "$ROOT_DIR/.nix-bootstrap.stamp" \
  "$ROOT_DIR/.nix-curobo.stamp" \
  "$ROOT_DIR/third_party/curobo/build" \
  "$ROOT_DIR/third_party/curobo/src/nvidia_curobo.egg-info"
rm -f "$ROOT_DIR"/third_party/curobo/src/curobo/curobolib/*.so

mkdir -p "$TEST_NIX_CACHE_DIR" "$TEST_UV_CACHE_DIR"

echo "[check_env_clean] rebuilding env from scratch via nix develop"
XDG_CACHE_HOME="$TEST_NIX_CACHE_DIR" \
UV_CACHE_DIR="$TEST_UV_CACHE_DIR" \
UV_PROJECT_ENVIRONMENT="$TEST_VENV_DIR" \
SIMPLE_RESPECT_HOST_UV_PROJECT_ENV=1 \
SIMPLE_AUTO_BOOTSTRAP=0 \
LD_LIBRARY_PATH="" \
nix develop -c bash -lc '
  set -euo pipefail

  echo "[check_env_clean] creating venv at $UV_PROJECT_ENVIRONMENT"
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"

  echo "[check_env_clean] bootstrapping project"
  ./scripts/nix/bootstrap.sh

  test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
  test -x "$UV_PROJECT_ENVIRONMENT/bin/datagen"
  test -x "$UV_PROJECT_ENVIRONMENT/bin/eval"

  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "[check_env_clean] TORCH_CUDA_ARCH_LIST is not set"
    exit 1
  fi

  python - <<'"'"'PY'"'"'
import importlib
import os
import sys

import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert os.environ.get("TORCH_CUDA_ARCH_LIST"), "TORCH_CUDA_ARCH_LIST is unset"
assert os.environ.get("PYTHONNOUSERSITE") == "1", os.environ.get("PYTHONNOUSERSITE")
assert "PYTHONPATH" not in os.environ, os.environ.get("PYTHONPATH")
assert "PYTHONHOME" not in os.environ, os.environ.get("PYTHONHOME")
assert "LD_PRELOAD" not in os.environ, os.environ.get("LD_PRELOAD")
assert os.environ.get("VIRTUAL_ENV") == os.environ.get("UV_PROJECT_ENVIRONMENT"), (
    os.environ.get("VIRTUAL_ENV"),
    os.environ.get("UV_PROJECT_ENVIRONMENT"),
)

allowed_ld_prefixes = (
    "/nix/store/",
    f"{os.getcwd()}/.runtime-state/host-libcuda",
    "/run/opengl-driver/lib",
    "/run/opengl-driver-32/lib",
    "/usr/lib/x86_64-linux-gnu/nvidia/current",
    "/usr/lib/x86_64-linux-gnu/nvidia",
    "/usr/lib/nvidia",
    "/usr/lib/nvidia-570",
    "/usr/lib/nvidia-575",
    "/usr/lib/nvidia-580",
    "/usr/lib/nvidia-590",
    "/usr/lib/wsl/lib",
)
torch_lib = next(
    (
        str(path)
        for path in __import__("pathlib").Path(os.environ["UV_PROJECT_ENVIRONMENT"], "lib").glob(
            "python*/site-packages/torch/lib"
        )
        if path.is_dir()
    ),
    None,
)
if torch_lib is not None:
    allowed_ld_prefixes = allowed_ld_prefixes + (torch_lib,)
bad_ld = [
    entry
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if entry and not any(entry == prefix.rstrip("/") or entry.startswith(prefix) for prefix in allowed_ld_prefixes)
]
assert not bad_ld, f"unexpected LD_LIBRARY_PATH entries: {bad_ld}"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"

mods = [
    "simple",
    "openpi_client",
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]
for mod in mods:
    importlib.import_module(mod)

arch = os.environ["TORCH_CUDA_ARCH_LIST"]
print("[check_env_clean] python/runtime imports passed")
print(f"[check_env_clean] python={sys.executable}")
print(f"[check_env_clean] torch_cuda_devices={torch.cuda.device_count()}")
print(f"[check_env_clean] torch_cuda_arch_list={arch}")
PY
'

echo "[check_env_clean] success"
