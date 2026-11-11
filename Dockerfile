ARG ISAAC_SIM_VERSION=6.2.0

ARG CUDA_VERSION=12.8.0
# Must match the isaac-sim image's Ubuntu release (6.2.0 is 24.04).
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cuda
# Trim here, not after the COPY below: deleting in a later layer leaves the bytes
# in the earlier one and shrinks nothing that gets pushed.
#   *.a        ~3.4 GB of static libs -- everything here links CUDA dynamically
#   compat/    forward-compat driver, older than any host we target (see
#              scripts/run_eval_wbc.sh); leaving it out removes the 803 hazard
#   the rest   profiler/sanitizer/sample payload nothing at runtime reads
RUN find /usr/local/cuda-* -name '*.a' -delete && \
    rm -rf /usr/local/cuda-*/compat \
           /usr/local/cuda-*/compute-sanitizer \
           /usr/local/cuda-*/doc \
           /usr/local/cuda-*/samples

ARG ISAAC_SIM_VERSION=6.2.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim
LABEL maintainer="Songlin Wei"

# The isaac-sim 6.x image ends as the unprivileged `isaac-sim` user; everything
# below (apt, the shared venv under /workspace) is installed as root.
USER root
# Inherited from the base image: it greps /isaac-sim's Kit logs, which this image
# never writes (Isaac Sim runs from the pip wheel in the venv), so it would
# always report the container unhealthy.
HEALTHCHECK NONE

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION=12.8

# ARG VSCODE_COMMIT_SHA
ARG TIMEZONE
ARG TORCH_CUDA_ARCH_LIST

# 0 = minimal install (default, sufficient for headless sim eval/replay/datagen).
# 1 = full install (pulls decoupled_wbc[full]: ROS bridge, PyQt6, pyrealsense2,
#     Ray, mujoco, rerun, ...) — needed for real-robot deployment or GUI teleop.
ARG SIMPLE_FULL_INSTALL=0

COPY --from=cuda /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda-${CUDA_VERSION}
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}

# Change apt source if you encouter connection issues
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
#     sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# Ubuntu 24.04 ships Python 3.12 (what pyproject.toml requires; uv picks it up,
# and python3-dev provides the headers for the curobo / xrobotoolkit builds) and
# a libstdc++ new enough for Isaac Sim 6.x, so the old toolchain PPA is gone.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim ca-certificates build-essential ninja-build cmake libgmp-dev libgmp10 ffmpeg \
    python3 python3-dev pybind11-dev software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0+PTX"}
ENV TZ=${TIMEZONE:-UTC}

# --- Install uv globally ---
RUN wget -q https://astral.sh/uv/install.sh && \
    bash install.sh && \
    rm -f install.sh && \
    rm -rf /tmp/* /var/tmp/*
ENV PATH=/root/.local/bin:$PATH

# --- Configure persistent uv cache ---
ENV UV_CACHE_DIR=/workspace/.uv-cache
RUN mkdir -p ${UV_CACHE_DIR} && chmod 777 ${UV_CACHE_DIR}
VOLUME ["/workspace/.uv-cache"]

# --- Copy dependency files (cache key for venv) ---
WORKDIR /workspace/simple
COPY pyproject.toml uv.lock README.md license.md ./

# --- Copy third-party packages (filtered by .dockerignore) ---
COPY third_party third_party

# --- Install dependencies (cached) ---
ENV UV_HTTP_TIMEOUT=1800
ENV UV_HTTP_RETRIES=10
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7
ENV OMNI_KIT_ACCEPT_EULA=Y

#     `lerobot` is an optional extra now (the dependency group was removed).
#     --no-install-project: src/ is not copied yet; `simple` is installed
#     editable further down.
RUN --mount=type=cache,target=/workspace/.uv-cache \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --extra lerobot && \
    rm -rf /tmp/* /var/tmp/*

# --- Build cuRobo against the venv's torch ---
#     install_curobo.sh uses TORCH_CUDA_ARCH_LIST (set above from the build arg). No warp-lang re-pin any
#     more: isaaclab 3.0.0 needs the warp-lang==1.17.0 from the lock.
#     Peak memory ~ MAX_JOBS x NVCC_THREADS concurrent cicc/ptxas processes; curobo's
#     setup.py hardcodes --threads=8, so a non-empty NVCC_THREADS overrides it (nvcc keeps
#     the last value). Lower both for many-arch release builds on small hosts.
ARG MAX_JOBS=4
ARG NVCC_THREADS=
COPY scripts/install_curobo.sh scripts/
RUN --mount=type=cache,target=/workspace/.uv-cache \
    MAX_JOBS="${MAX_JOBS}" NVCC_APPEND_FLAGS="${NVCC_THREADS:+--threads=${NVCC_THREADS}}" \
    ./scripts/install_curobo.sh && \
    rm -rf /tmp/* /var/tmp/*

# --- Copy source code before cuRobo build to avoid overwriting built artifacts later ---
#     At runtime this is bind-mounted over with the host checkout, so host edits
#     take effect immediately (the editable install below resolves `simple` to
#     /workspace/simple/src).  The copy keeps the image usable without mounts.
COPY src src

# --- Optional: install local project in editable mode ---
RUN --mount=type=cache,target=/workspace/.uv-cache \
    uv pip install --editable . --no-deps && \
    rm -rf /tmp/* /var/tmp/*
# Scope the mode fix to the tree this layer actually adds. `chmod -R` on
# /workspace/simple would restat .venv (18 GB) and third_party (3.7 GB), which
# live in earlier layers -- overlayfs copies every touched file up, and the
# metadata change alone cost ~14 GB of duplicated layer.
RUN chmod -R 755 /workspace/simple/src /workspace/simple/scripts

# --- Install vendor-neutral Vulkan loader (libvulkan1) ---
#     The NVIDIA container runtime with `graphics` capability mounts
#     libGLX_nvidia.so.0 from the host, but not the loader itself.
#     Without libvulkan.so.1, omni.kit's carb.graphics-vulkan can't
#     create a VkInstance, GPU Foundation fails to init, and any
#     extension that builds omni.ui widgets at startup (e.g. the
#     URDF importer's string_filed_builder) segfaults.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libvulkan1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install the Sonic / decoupled-WBC stack ---
#     The vendored packages carry stale pins (numpy==1.26.4, scipy==1.15.3,
#     cyclonedds==0.10.2) that pyproject.toml's override-dependencies replace in
#     the lock, but `uv pip` does not apply those overrides. The minimal path
#     therefore installs them --no-deps and pulls their runtime deps constrained
#     to the locked versions, so numpy/torch/mujoco stay what isaaclab expects.
ARG SIMPLE_FULL_INSTALL
RUN --mount=type=cache,target=/workspace/.uv-cache \
    if [ "${SIMPLE_FULL_INSTALL}" = "1" ]; then \
        echo "[sonic-install] SIMPLE_FULL_INSTALL=1 — installing full sonic group" && \
        GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --inexact --extra lerobot --group sonic ; \
    else \
        echo "[sonic-install] SIMPLE_FULL_INSTALL=0 — installing minimal sonic subset" && \
        uv export --frozen --no-hashes --no-header --no-annotate --no-emit-project \
            --no-emit-local --extra lerobot --group sonic \
            | grep -v ' @ ' > /tmp/sonic-constraints.txt && \
        uv pip install --python /workspace/simple/.venv/bin/python \
            --no-build-isolation --no-deps \
            -e third_party/gear_sonic \
            -e third_party/decoupled_wbc \
            -e third_party/unitree_sdk2_python && \
        uv pip install --python /workspace/simple/.venv/bin/python \
            -c /tmp/sonic-constraints.txt \
            tyro pin pin-pink pyyaml onnxruntime loguru termcolor qpsolvers \
            pyzmq msgpack-numpy cyclonedds opencv-python ; \
    fi && \
    rm -rf /tmp/* /var/tmp/*

# --- Build the SONIC whole-body controller (docs/source/sonic-wbc/setup.md) ---
ARG SIMPLE_BUILD_SONIC_WBC=1

# System deps for build_sonic_controller.sh (cmake, ninja, g++, curl, git are
# already installed above):
RUN if [ "${SIMPLE_BUILD_SONIC_WBC}" = "1" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends pkg-config libboost-dev iproute2 && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

COPY scripts/build_sonic_controller.sh scripts/

# --- XRoboToolkit VR SDK (PICO headset bindings) ---
RUN --mount=type=cache,target=/workspace/.uv-cache \
    if [ "${SIMPLE_BUILD_SONIC_WBC}" = "1" ] && [ "${SIMPLE_FULL_INSTALL}" != "1" ]; then \
        if [ -f third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/setup.py ]; then \
            echo "[sonic-wbc] building xrobotoolkit-sdk" && \
            uv pip install --python /workspace/simple/.venv/bin/python \
                pybind11 "setuptools-scm<9" && \
            CMAKE_PREFIX_PATH="$(/workspace/simple/.venv/bin/python -m pybind11 --cmakedir)" \
            uv pip install --python /workspace/simple/.venv/bin/python --no-build-isolation \
                -e third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 ; \
        else \
            echo "[sonic-wbc] WARNING: XRoboToolkit submodule missing — skipping VR SDK" ; \
        fi ; \
    fi && \
    rm -rf /tmp/* /var/tmp/*

# --- SONIC ONNX checkpoints (nvidia/GEAR-SONIC on HuggingFace, public) ---
ARG SIMPLE_DOWNLOAD_SONIC_CKPT=1
RUN if [ "${SIMPLE_BUILD_SONIC_WBC}" = "1" ] && [ "${SIMPLE_DOWNLOAD_SONIC_CKPT}" = "1" ]; then \
        if [ -f third_party/GR00T-WholeBodyControl/download_from_hf.py ]; then \
            echo "[sonic-wbc] downloading SONIC checkpoints" && \
            cd third_party/GR00T-WholeBodyControl && \
            /workspace/simple/.venv/bin/python download_from_hf.py ; \
        else \
            echo "[sonic-wbc] WARNING: GR00T-WholeBodyControl missing — skipping checkpoints" ; \
        fi ; \
    fi && \
    rm -rf /root/.cache/huggingface /tmp/* /var/tmp/*

# --- SONIC controller binary (CMake + TensorRT/onnxruntime, no GPU needed) ---
#     SONIC_BUILD_JOBS caps make/cmake parallelism (empty = nproc).
ARG SONIC_BUILD_JOBS=
RUN --mount=type=cache,target=/workspace/.uv-cache \
    if [ "${SIMPLE_BUILD_SONIC_WBC}" != "1" ]; then \
        echo "[sonic-wbc] SIMPLE_BUILD_SONIC_WBC=${SIMPLE_BUILD_SONIC_WBC} — skipping controller build" ; \
    elif [ ! -f third_party/GR00T-WholeBodyControl/gear_sonic_deploy/CMakeLists.txt ]; then \
        echo "[sonic-wbc] WARNING: gear_sonic_deploy missing from the build context —" && \
        echo "[sonic-wbc]          controller not built; see the submodule hint above." ; \
    else \
        HOME=/root SONIC_BUILD_JOBS="${SONIC_BUILD_JOBS}" bash scripts/build_sonic_controller.sh && \
        rm -rf /root/tools/zeromq-4.3.5 \
               /root/tools/eigen-3.4.0 \
               /root/tools/googletest-1.14.0 \
               /root/tools/msgpack-cxx-6.1.1 \
               /root/tools/TensorRT-10.9.0 && \
        chmod -R a+rX /root/tools ; \
    fi && \
    rm -rf /tmp/* /var/tmp/*

# --- Make the venv the default Python (`python`, `python3`, its console scripts) ---
#     PATH, not a symlink in /usr/local/bin: Python finds pyvenv.cfg next to the path it
#     was launched from, so a symlink elsewhere would start without the venv's packages.
#     Last in the file so the build steps above keep the system cmake / ninja, which
#     this shadows with the venv's at runtime.
ENV VIRTUAL_ENV=/workspace/simple/.venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH

ENTRYPOINT ["/usr/bin/env"]
CMD ["/bin/bash"]
