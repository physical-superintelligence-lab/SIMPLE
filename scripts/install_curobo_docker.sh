#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-nix"
  fi
fi

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[install_curobo] missing venv python at $VENV_PYTHON"
  echo "[install_curobo] create the venv first"
  exit 1
fi

export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

assert_nvidia_runtime() {
  local cuda_ok=0
  for cuda_lib in \
    "${SIMPLE_NIX_COMPAT_DIR:-}/lib/libcuda.so.1" \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /run/opengl-driver/lib/libcuda.so.1 \
    /usr/lib/wsl/lib/libcuda.so.1
  do
    if [[ -f "$cuda_lib" ]]; then
      cuda_ok=1
      break
    fi
  done

  local vk_icd="${VK_ICD_FILENAMES:-}"
  if [[ -z "$vk_icd" ]]; then
    for vk_candidate in \
      /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json \
      /usr/share/vulkan/icd.d/nvidia_icd.json \
      /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
      /etc/vulkan/icd.d/nvidia_icd.json
    do
      if [[ -f "$vk_candidate" ]]; then
        vk_icd="$vk_candidate"
        break
      fi
    done
  fi

  local egl_vendor="${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  if [[ -z "$egl_vendor" ]]; then
    for egl_candidate in \
      /run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json \
      /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
      /usr/share/glvnd/egl_vendor.d/50_nvidia.json
    do
      if [[ -f "$egl_candidate" ]]; then
        egl_vendor="$egl_candidate"
        break
      fi
    done
  fi

  if [[ "$cuda_ok" -ne 1 ]]; then
    echo "[install_curobo] libcuda.so.1 not found in expected host locations"
    return 1
  fi
  if [[ -z "$vk_icd" ]]; then
    echo "[install_curobo] NVIDIA Vulkan ICD JSON not found"
    return 1
  fi
  if [[ -z "$egl_vendor" ]]; then
    echo "[install_curobo] NVIDIA EGL vendor JSON not found"
    return 1
  fi
}

detect_torch_cuda_arch_list() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[install_curobo] nvidia-smi not found"
    return 1
  fi

  local compute_caps
  compute_caps="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
      | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
  )"

  if [[ -z "$compute_caps" ]]; then
    echo "[install_curobo] no GPU compute capability reported"
    return 1
  fi

  local unique_count
  unique_count="$(printf '%s\n' "$compute_caps" | sed '/^$/d' | sort -u | wc -l)"
  if [[ "$unique_count" -ne 1 ]]; then
    echo "[install_curobo] mixed GPU compute capabilities detected: $(printf '%s\n' "$compute_caps" | sed '/^$/d' | paste -sd, -)"
    return 1
  fi

  local compute_cap
  compute_cap="$(printf '%s\n' "$compute_caps" | sed '/^$/d' | head -n 1)"
  if [[ -n "$compute_cap" ]]; then
    export TORCH_CUDA_ARCH_LIST="${compute_cap}+PTX"
    return 0
  fi

  return 1
}

echo "📦 Initializing cuRobo submodule..."
# git submodule update --init --recursive third_party/curobo

echo "⚠️  Skipping Git LFS pull (large files not downloaded)..."
echo "   If you encounter missing file errors, you may need to install Git LFS later."

echo "⚙️ Installing cuRobo into uv environment (This'll take a few minutes)..."
export GIT_LFS_SKIP_SMUDGE=1
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ "${SIMPLE_SKIP_NVIDIA_RUNTIME_CHECK:-0}" == "1" ]]; then
  echo "[install_curobo] SIMPLE_SKIP_NVIDIA_RUNTIME_CHECK=1 set; skipping host NVIDIA runtime assertion"
elif ! assert_nvidia_runtime; then
  echo "[install_curobo] host NVIDIA runtime is incomplete for GPU builds"
  exit 1
fi

if ! detect_torch_cuda_arch_list; then
  echo "[install_curobo] failed to determine TORCH_CUDA_ARCH_LIST automatically"
  echo "[install_curobo] set TORCH_CUDA_ARCH_LIST explicitly and rerun"
  exit 1
fi

echo "Using TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "Using Python environment at: $VENV_PYTHON"

uv pip install --python "$VENV_PYTHON" --no-build-isolation -e 'third_party/curobo[isaacsim]'

echo "🛠️ Fix warp-lang version"
uv pip install --python "$VENV_PYTHON" "warp-lang==1.7.0" --index-strategy unsafe-best-match

if [[ "${SIMPLE_SKIP_CUROBO_VERIFY:-0}" == "1" ]]; then
  echo "🔍 Skipping cuRobo CUDA extension verification (SIMPLE_SKIP_CUROBO_VERIFY=1)"
else
  echo "🔍 Verifying cuRobo CUDA extensions..."
  "$VENV_PYTHON" - <<'PY'
import sys
import torch

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')

try:
    import curobo.curobolib.lbfgs_step_cu
    import curobo.curobolib.kinematics_fused_cu
    import curobo.curobolib.line_search_cu
    import curobo.curobolib.tensor_step_cu
    import curobo.curobolib.geom_cu
    print('✅ All CUDA extensions loaded successfully')
except ImportError as e:
    print(f'❌ CUDA extension import failed: {e}')
    sys.exit(1)
PY
fi

echo "✅ cuRobo installed successfully."
