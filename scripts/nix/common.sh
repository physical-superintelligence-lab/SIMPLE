#!/usr/bin/env bash

simple_nix_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.." && pwd
}

simple_runtime_allowed_ld_prefixes() {
  local root_dir="$1"
  cat <<EOF
/nix/store
$root_dir/.runtime-state/host-libcuda
/run/opengl-driver/lib
/run/opengl-driver-32/lib
/usr/lib/x86_64-linux-gnu/nvidia/current
/usr/lib/x86_64-linux-gnu/nvidia
/usr/lib/nvidia
/usr/lib/nvidia-570
/usr/lib/nvidia-575
/usr/lib/nvidia-580
/usr/lib/nvidia-590
/usr/lib/wsl/lib
EOF

  simple_runtime_torch_lib_dirs
}

simple_runtime_host_cuda_candidates() {
  cat <<'EOF'
/run/opengl-driver/lib/libcuda.so.1
/usr/lib/x86_64-linux-gnu/libcuda.so.1
/usr/lib/x86_64-linux-gnu/nvidia/current/libcuda.so.1
/usr/lib/x86_64-linux-gnu/nvidia/libcuda.so.1
/usr/lib/wsl/lib/libcuda.so.1
EOF
}

simple_runtime_host_glx_candidates() {
  cat <<'EOF'
/run/opengl-driver/lib/libGLX_nvidia.so.0
/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
/usr/lib/x86_64-linux-gnu/nvidia/current/libGLX_nvidia.so.0
/usr/lib/x86_64-linux-gnu/nvidia/libGLX_nvidia.so.0
/usr/lib/wsl/lib/libGLX_nvidia.so.0
EOF
}

simple_runtime_vk_icd_candidates() {
  cat <<'EOF'
/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json
/usr/share/vulkan/icd.d/nvidia_icd.json
/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
/etc/vulkan/icd.d/nvidia_icd.json
EOF
}

simple_runtime_egl_vendor_candidates() {
  cat <<'EOF'
/run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json
/usr/share/glvnd/egl_vendor.d/10_nvidia.json
/usr/share/glvnd/egl_vendor.d/50_nvidia.json
EOF
}

simple_runtime_source_home_env() {
  if [[ -f "$HOME/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
  fi
}

simple_runtime_configure_uv_project_environment() {
  local root_dir="$1"

  if [[ "${SIMPLE_RESPECT_HOST_UV_PROJECT_ENV:-0}" == "1" ]]; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$root_dir/.venv-nix}"
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PATH="${PATH/#$VIRTUAL_ENV\/bin:/}"
    PATH="${PATH//:$VIRTUAL_ENV\/bin:/:}"
    PATH="${PATH/%:$VIRTUAL_ENV\/bin/}"
  fi
  unset VIRTUAL_ENV
  export UV_PROJECT_ENVIRONMENT="$root_dir/.venv-nix"
}

simple_runtime_sanitize_python_env() {
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  unset PYTHONHOME
  unset LD_PRELOAD
}

simple_runtime_filter_path_var() {
  local var_name="$1"
  shift
  local current_value
  current_value="$(printenv "$var_name" || true)"
  if [[ -z "$current_value" ]]; then
    return 0
  fi

  local filtered_value=""
  local old_ifs="$IFS"
  IFS=:
  for entry in $current_value; do
    [[ -z "$entry" ]] && continue
    local keep_entry=0
    local prefix
    for prefix in "$@"; do
      case "$entry" in
        "$prefix"|"$prefix"/*)
          keep_entry=1
          break
          ;;
      esac
    done
    if [[ "$keep_entry" -eq 1 ]]; then
      if [[ -n "$filtered_value" ]]; then
        filtered_value="${filtered_value}:$entry"
      else
        filtered_value="$entry"
      fi
    fi
  done
  IFS="$old_ifs"

  if [[ -n "$filtered_value" ]]; then
    export "$var_name=$filtered_value"
  else
    unset "$var_name"
  fi
}

simple_runtime_torch_lib_dirs() {
  local venv_dir="${UV_PROJECT_ENVIRONMENT:-}"
  [[ -n "$venv_dir" ]] || return 0
  [[ -d "$venv_dir/lib" ]] || return 0

  local torch_lib_dir
  for torch_lib_dir in "$venv_dir"/lib/python*/site-packages/torch/lib; do
    [[ -d "$torch_lib_dir" ]] || continue
    printf '%s\n' "$torch_lib_dir"
  done
}

simple_runtime_find_first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

simple_runtime_detect_host_driver_state() {
  local root_dir="$1"
  local host_lib_dir="${SIMPLE_RUNTIME_GPU_LIB_DIR:-$root_dir/.runtime-state/host-libcuda}"

  local libcuda=""
  libcuda="$(simple_runtime_find_first_file $(simple_runtime_host_cuda_candidates) || true)"

  local vk_icd="${VK_ICD_FILENAMES:-}"
  if [[ -z "$vk_icd" ]]; then
    vk_icd="$(simple_runtime_find_first_file $(simple_runtime_vk_icd_candidates) || true)"
  fi

  local egl_vendor="${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  if [[ -z "$egl_vendor" ]]; then
    egl_vendor="$(simple_runtime_find_first_file $(simple_runtime_egl_vendor_candidates) || true)"
  fi

  printf 'SIMPLE_HOST_LIB_DIR=%q\n' "$host_lib_dir"
  printf 'SIMPLE_HOST_LIBCUDA=%q\n' "$libcuda"
  printf 'SIMPLE_HOST_VK_ICD=%q\n' "$vk_icd"
  printf 'SIMPLE_HOST_EGL_VENDOR=%q\n' "$egl_vendor"
}

simple_runtime_assert_nvidia_runtime() {
  local prefix="$1"
  local root_dir="$2"

  # shellcheck disable=SC1090
  eval "$(simple_runtime_detect_host_driver_state "$root_dir")"

  if [[ -z "${SIMPLE_HOST_LIBCUDA:-}" ]]; then
    echo "[$prefix] libcuda.so.1 not found in expected host locations"
    echo "[$prefix] expected Ubuntu/NixOS-style NVIDIA driver layout under /usr/lib/x86_64-linux-gnu, /run/opengl-driver, or /usr/lib/wsl/lib"
    return 1
  fi
  if [[ -z "${SIMPLE_HOST_VK_ICD:-}" ]]; then
    echo "[$prefix] NVIDIA Vulkan ICD JSON not found"
    return 1
  fi
  if [[ -z "${SIMPLE_HOST_EGL_VENDOR:-}" ]]; then
    echo "[$prefix] NVIDIA EGL vendor JSON not found"
    return 1
  fi
}

simple_runtime_stage_host_driver_libs() {
  local root_dir="$1"
  # shellcheck disable=SC1090
  eval "$(simple_runtime_detect_host_driver_state "$root_dir")"

  mkdir -p "$SIMPLE_HOST_LIB_DIR"

  if [[ -n "${SIMPLE_HOST_LIBCUDA:-}" ]]; then
    ln -sf "$SIMPLE_HOST_LIBCUDA" "$SIMPLE_HOST_LIB_DIR/libcuda.so.1"
    ln -sf "$SIMPLE_HOST_LIB_DIR/libcuda.so.1" "$SIMPLE_HOST_LIB_DIR/libcuda.so"
  fi

  local glx_candidate
  for glx_candidate in $(simple_runtime_host_glx_candidates); do
    if [[ -f "$glx_candidate" ]]; then
      local glx_dir
      glx_dir="$(dirname "$glx_candidate")"
      local driver_lib
      for driver_lib in \
        "$glx_dir"/libGLX_nvidia.so* \
        "$glx_dir"/libEGL_nvidia.so* \
        "$glx_dir"/libGLESv2_nvidia.so* \
        "$glx_dir"/libnvcuvid.so* \
        "$glx_dir"/libnvidia-*.so*
      do
        [[ -f "$driver_lib" ]] || continue
        ln -sf "$driver_lib" "$SIMPLE_HOST_LIB_DIR/$(basename "$driver_lib")"
      done
      break
    fi
  done

  if [[ -z "${VK_ICD_FILENAMES:-}" && -n "${SIMPLE_HOST_VK_ICD:-}" ]]; then
    export VK_ICD_FILENAMES="$SIMPLE_HOST_VK_ICD"
  fi
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -n "${SIMPLE_HOST_EGL_VENDOR:-}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$SIMPLE_HOST_EGL_VENDOR"
  fi

  simple_runtime_filter_path_var LD_LIBRARY_PATH \
    /nix/store \
    "$SIMPLE_HOST_LIB_DIR" \
    /run/opengl-driver/lib \
    /run/opengl-driver-32/lib \
    /usr/lib/x86_64-linux-gnu/nvidia/current \
    /usr/lib/x86_64-linux-gnu/nvidia \
    /usr/lib/nvidia \
    /usr/lib/nvidia-570 \
    /usr/lib/nvidia-575 \
    /usr/lib/nvidia-580 \
    /usr/lib/nvidia-590 \
    /usr/lib/wsl/lib

  export LD_LIBRARY_PATH="$SIMPLE_HOST_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export XDG_DATA_DIRS="/run/opengl-driver/share:/usr/share${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"

  if [[ -z "${TRITON_LIBCUDA_PATH:-}" ]]; then
    local triton_dir
    for triton_dir in \
      "$SIMPLE_HOST_LIB_DIR" \
      /run/opengl-driver/lib \
      /usr/lib/x86_64-linux-gnu/nvidia/current \
      /usr/lib/x86_64-linux-gnu/nvidia \
      /usr/lib/nvidia-590 \
      /usr/lib/nvidia-580 \
      /usr/lib/nvidia-575 \
      /usr/lib/nvidia-570 \
      /usr/lib/wsl/lib
    do
      if [[ -d "$triton_dir" ]]; then
        export TRITON_LIBCUDA_PATH="$triton_dir"
        break
      fi
    done
  fi
}

simple_runtime_stage_python_runtime_libs() {
  local torch_lib_dir
  local added_dirs=()
  while IFS= read -r torch_lib_dir; do
    [[ -n "$torch_lib_dir" ]] || continue
    added_dirs+=("$torch_lib_dir")
  done < <(simple_runtime_torch_lib_dirs)

  [[ "${#added_dirs[@]}" -gt 0 ]] || return 0

  simple_runtime_filter_path_var LD_LIBRARY_PATH $(simple_runtime_allowed_ld_prefixes "$(simple_nix_repo_root)")

  local existing="${LD_LIBRARY_PATH:-}"
  local new_value=""
  local dir
  for dir in "${added_dirs[@]}"; do
    case ":$existing:" in
      *":$dir:"*) ;;
      *)
        if [[ -n "$new_value" ]]; then
          new_value="${new_value}:$dir"
        else
          new_value="$dir"
        fi
        ;;
    esac
  done

  if [[ -n "$new_value" ]]; then
    export LD_LIBRARY_PATH="$new_value${existing:+:$existing}"
  fi
}

simple_runtime_assert_runtime_boundary() {
  local prefix="$1"
  local root_dir="$2"
  local bad_ld=""
  local old_ifs="$IFS"
  IFS=:
  local entry
  for entry in ${LD_LIBRARY_PATH:-}; do
    [[ -z "$entry" ]] && continue
    local entry_ok=0
    local allowed_prefix
    while IFS= read -r allowed_prefix; do
      [[ -z "$allowed_prefix" ]] && continue
      case "$entry" in
        "$allowed_prefix"|"$allowed_prefix"/*)
          entry_ok=1
          break
          ;;
      esac
    done < <(simple_runtime_allowed_ld_prefixes "$root_dir")
    if [[ "$entry_ok" -ne 1 ]]; then
      if [[ -n "$bad_ld" ]]; then
        bad_ld="${bad_ld}:$entry"
      else
        bad_ld="$entry"
      fi
    fi
  done
  IFS="$old_ifs"

  if [[ -n "$bad_ld" ]]; then
    echo "[$prefix] LD_LIBRARY_PATH contains non-runtime paths: $bad_ld"
    return 1
  fi
  if [[ -n "${PYTHONPATH:-}" ]]; then
    echo "[$prefix] PYTHONPATH must be unset inside the Nix runtime"
    return 1
  fi
  if [[ -n "${PYTHONHOME:-}" ]]; then
    echo "[$prefix] PYTHONHOME must be unset inside the Nix runtime"
    return 1
  fi
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    echo "[$prefix] LD_PRELOAD must be unset inside the Nix runtime"
    return 1
  fi
  if [[ "${VIRTUAL_ENV:-}" != "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    echo "[$prefix] VIRTUAL_ENV mismatch: expected ${UV_PROJECT_ENVIRONMENT:-<unset>} got ${VIRTUAL_ENV:-<unset>}"
    return 1
  fi
}

simple_runtime_detect_torch_cuda_arch_list() {
  local prefix="$1"
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$prefix] nvidia-smi not found"
    return 1
  fi

  local compute_caps
  compute_caps="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
      | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
      | sed '/^$/d'
  )"

  if [[ -z "$compute_caps" ]]; then
    echo "[$prefix] no GPU compute capability reported"
    return 1
  fi

  local unique_count
  unique_count="$(printf '%s\n' "$compute_caps" | sort -u | wc -l)"
  if [[ "$unique_count" -ne 1 ]]; then
    echo "[$prefix] mixed GPU compute capabilities detected: $(printf '%s\n' "$compute_caps" | paste -sd, -)"
    return 1
  fi

  export TORCH_CUDA_ARCH_LIST="$(printf '%s\n' "$compute_caps" | head -n 1)+PTX"
}

simple_runtime_assert_vendor_tree() {
  local prefix="$1"
  local package_name="$2"
  local package_dir="$3"

  if [[ ! -d "$package_dir" ]]; then
    echo "[$prefix] required vendored dependency is missing: $package_name ($package_dir)"
    return 1
  fi

  if [[ ! -f "$package_dir/pyproject.toml" && ! -f "$package_dir/setup.py" ]]; then
    echo "[$prefix] vendored dependency is not initialized: $package_name ($package_dir)"
    echo "[$prefix] expected one of: $package_dir/pyproject.toml or $package_dir/setup.py"
    echo "[$prefix] if this comes from git submodules, run:"
    echo "  git submodule update --init --recursive third_party/${package_name#nvidia-}"
    return 1
  fi
}

simple_runtime_use_writable_dir() {
  local var_name="$1"
  local fallback_dir="$2"
  local current_dir
  current_dir="$(printenv "$var_name" || true)"

  if [[ -n "$current_dir" && -d "$current_dir" && -w "$current_dir" ]]; then
    return 0
  fi

  mkdir -p "$fallback_dir"
  export "$var_name=$fallback_dir"
}
