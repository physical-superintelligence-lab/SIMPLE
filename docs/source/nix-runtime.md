# Nix Runtime

## Goal

The Nix setup in this repo is designed to provide a stable Linux development runtime with a narrow host boundary:

- Nix owns Python, compiler toolchains, FFmpeg, OpenGL/Vulkan userspace, CUDA userspace, and project Python packages.
- The host only provides the NVIDIA driver boundary:
  - `libcuda.so.1`
  - NVIDIA GL/EGL/Vulkan driver libraries
  - Vulkan ICD JSON
  - EGL vendor JSON

This is the closest practical approximation to "one reproducible dev shell across Linux hosts with NVIDIA drivers installed".

The intended consumption model is library-first:

- prefer `import simple` from inside the dev shell
- treat `uv run eval`, `datagen`, and similar CLI entry points as convenience wrappers around the Python API
- if another project integrates SIMPLE, it should usually depend on the package/runtime, not shell out to the CLI

## Why The Setup Exists

The environment is more complex than a pure CPU Python project because this repo mixes:

1. Python packaging managed by `uv`
2. Native extension builds for vendored packages
3. CUDA userspace managed by Nix
4. NVIDIA driver userspace managed by the host
5. Isaac Sim / CuRobo GPU-specific post-install work

The complexity does not come from Nix alone. It comes from the boundary between:

- reproducible userspace we can manage in Nix
- GPU driver components that remain host-specific

## Supported Host Model

The intended host baseline is:

- Linux
- NVIDIA drivers already installed
- Ubuntu-style driver layout is explicitly supported
- NixOS-style `/run/opengl-driver/...` layout is also supported
- WSL driver layout is partially supported through `/usr/lib/wsl/lib`
- at least ~80 GiB free on `/nix` is strongly recommended for a first CUDA-heavy bootstrap
- at least ~20 GiB free on the workspace filesystem is recommended for caches, extracted wheels, and virtualenvs

Current host lookup paths:

- CUDA:
  - `/run/opengl-driver/lib/libcuda.so.1`
  - `/usr/lib/x86_64-linux-gnu/libcuda.so.1`
  - `/usr/lib/x86_64-linux-gnu/nvidia/current/libcuda.so.1`
  - `/usr/lib/x86_64-linux-gnu/nvidia/libcuda.so.1`
  - `/usr/lib/wsl/lib/libcuda.so.1`
- Vulkan ICD:
  - `/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json`
  - `/usr/share/vulkan/icd.d/nvidia_icd.json`
  - `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json`
  - `/etc/vulkan/icd.d/nvidia_icd.json`
- EGL vendor:
  - `/run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json`
  - `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
  - `/usr/share/glvnd/egl_vendor.d/50_nvidia.json`

If an Ubuntu host keeps NVIDIA files elsewhere, update [scripts/nix/common.sh](/home/zhenyu/src/SIMPLE/scripts/nix/common.sh).

## Runtime Layout

### 1. Base Nix runtime

[nix/runtime-base.nix](/home/zhenyu/src/SIMPLE/nix/runtime-base.nix)

Owns:

- Python runtime
- wrapped compiler toolchain
- libc handling via the wrapped compiler toolchain
- CUDA toolkit userspace
- FFmpeg and graphics userspace libs

Important rule:

- do not add `glibc.dev` as a regular shell package to "help" C/C++ builds
- the wrapped compiler already knows how to find libc headers in the right order
- adding `glibc.dev` directly can break libstdc++'s `#include_next` chain for `stdlib.h`

### 2. Host NVIDIA bridge

[nix/runtime-host-gpu.nix](/home/zhenyu/src/SIMPLE/nix/runtime-host-gpu.nix)

Owns:

- staging host NVIDIA driver libraries into `.runtime-state/host-libcuda`
- exposing Vulkan/EGL host metadata into the shell

### 3. Shared runtime policy

[scripts/nix/common.sh](/home/zhenyu/src/SIMPLE/scripts/nix/common.sh)

Owns:

- sourcing `~/.env` when present
- sanitizing Python-related env vars
- defining allowed runtime library paths
- validating the Nix/host-driver boundary
- checking vendored dependency trees
- detecting `TORCH_CUDA_ARCH_LIST`

### 4. Shell orchestration

[scripts/nix/shell-init.sh](/home/zhenyu/src/SIMPLE/scripts/nix/shell-init.sh)

Owns:

- per-shell writable cache/config dirs
- project virtualenv path policy
- shell-time assertions
- optional auto-bootstrap

### 5. Bootstrap phases

- Python/bootstrap phase:
  [scripts/nix/bootstrap-python.sh](/home/zhenyu/src/SIMPLE/scripts/nix/bootstrap-python.sh)
- GPU/bootstrap phase:
  [scripts/nix/bootstrap-gpu.sh](/home/zhenyu/src/SIMPLE/scripts/nix/bootstrap-gpu.sh)
- Convenience wrapper:
  [scripts/nix/bootstrap.sh](/home/zhenyu/src/SIMPLE/scripts/nix/bootstrap.sh)

This split is intentional:

- Python/bootstrap should remain readable and mostly host-independent.
- GPU/bootstrap is where CuRobo and GPU-specific work belongs.

## Runtime Invariants

The shell/bootstrap assert the following:

- `PYTHONPATH` is unset
- `PYTHONHOME` is unset
- `LD_PRELOAD` is unset
- `PYTHONNOUSERSITE=1`
- `VIRTUAL_ENV == UV_PROJECT_ENVIRONMENT`
- `LD_LIBRARY_PATH` only contains:
  - `/nix/store/...`
  - `.runtime-state/host-libcuda`
  - approved NVIDIA host-driver paths

If those invariants fail, the scripts should stop with an exact error instead of silently continuing with a polluted runtime.

## Entry Points

### Check prerequisites on a new host

Run this first on a new machine:

```bash
./scripts/nix/prereq-check.sh
```

It checks:

- required commands such as `nix`, `git`, `git-lfs`, and `uv`
- whether `nix` works with `LD_LIBRARY_PATH` cleared
- free space on `/nix` and the workspace filesystem
- supported NVIDIA driver file layout
- required vendored dependencies such as `third_party/curobo`
- obvious outer-shell pollution such as `LD_LIBRARY_PATH`, `PYTHONPATH`, or `CONDA_PREFIX`

### Start the shell

```bash
direnv allow
```

or

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

### Bootstrap explicitly

```bash
./scripts/nix/bootstrap.sh
```

### Use SIMPLE as a library

Inside the dev shell:

```python
import gymnasium as gym
import simple.envs as _

env = gym.make("simple/FrankaTabletopGrasp-v0", sim_mode="isaac", headless=True)
obs, info = env.reset()
env.close()
```

Prefer this integration model over invoking the CLI from another Python application.

### Run only the Python phase

```bash
./scripts/nix/bootstrap-python.sh
```

### Run only the GPU phase

```bash
./scripts/nix/bootstrap-gpu.sh
```

## Failure Model

The setup should fail early in these cases:

- host NVIDIA runtime files are missing
- runtime env variables are polluted by the outer shell
- vendored dependencies are missing or uninitialized
- GPU architecture cannot be detected automatically
- `/nix` or the workspace filesystem is too full to realize the CUDA/Python closure reliably

That is deliberate. Silent fallback to host libraries is harder to debug and less reproducible than a hard assertion.

## Why We Still Cannot Be Perfectly Host-Independent

Even with Nix managing almost all userspace, the following still come from the host:

- running NVIDIA kernel driver
- driver-provided user libraries matched to that kernel driver
- Vulkan/EGL metadata installed by the driver package

So the reproducibility boundary is:

- reproducible above the driver boundary
- host-dependent at the driver boundary

That is the practical limit for GPU-heavy Linux development without fully containerizing or fully standardizing the host OS image.

## Maintenance Rule

When changing the runtime:

1. Put shared policy in [scripts/nix/common.sh](/home/zhenyu/src/SIMPLE/scripts/nix/common.sh)
2. Keep [flake.nix](/home/zhenyu/src/SIMPLE/flake.nix) as orchestration only
3. Put Nix-owned userspace in [nix/runtime-base.nix](/home/zhenyu/src/SIMPLE/nix/runtime-base.nix)
4. Put host NVIDIA bridging in [nix/runtime-host-gpu.nix](/home/zhenyu/src/SIMPLE/nix/runtime-host-gpu.nix)
5. Keep Python bootstrap separate from GPU bootstrap
6. Keep the user-facing contract stable: `prereq-check -> nix develop -> import simple`

If a new fix requires copying shell logic into multiple files, that is usually a sign the fix belongs in `common.sh` instead.
