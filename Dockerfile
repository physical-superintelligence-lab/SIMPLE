ARG ISAAC_SIM_VERSION=4.5.0 

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda

ARG ISAAC_SIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim
LABEL maintainer="Songlin Wei"

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

# fix libstdc++.so.6 version issue 
# strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep 3.4.32
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg dirmngr \
    git curl wget vim ca-certificates build-essential ninja-build cmake libgmp-dev libgmp10 ffmpeg \
    python3 python3-pip python3-dev pybind11-dev software-properties-common \
    && curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x60C317803A41BA51845E371A1E9377A2BA9EF27F' \
       | gpg --dearmor -o /etc/apt/trusted.gpg.d/ubuntu-toolchain-r.gpg \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install everything as root for shared usage
USER root
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
WORKDIR /workspace/SIMPLE
COPY pyproject.toml uv.lock ./

# --- Copy third-party packages (filtered by .dockerignore) ---
COPY third_party third_party

# --- Install dependencies (cached) ---
ENV UV_HTTP_TIMEOUT=1800
ENV UV_HTTP_RETRIES=10
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7
ENV OMNI_KIT_ACCEPT_EULA=Y

RUN --mount=type=cache,target=/workspace/.uv-cache \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot --index-strategy unsafe-best-match && \
    rm -rf /tmp/* /var/tmp/*

# --- Copy install script (separate from cuRobo files) ---
#     Use install_curobo_docker.sh — the docker-build variant of install_curobo.sh
#     that is stripped of nix-shell machinery and gates GPU-dependent checks behind
#     env vars (docker build runs without a GPU attached). install_curobo.sh
#     remains unchanged for bare-host (README Option 1) and nix (Option 2) flows.
COPY scripts/install_curobo_docker.sh scripts/

# --- Fix curobo + warp-lang installations (with cache for smaller image) ---
#     Skip host NVIDIA runtime assertion and GPU-dependent verification because
#     docker build runs without a GPU attached — runtime tests happen later.
ENV SIMPLE_SKIP_NVIDIA_RUNTIME_CHECK=1
ENV SIMPLE_SKIP_CUROBO_VERIFY=1
RUN --mount=type=cache,target=/workspace/.uv-cache \
    ./scripts/install_curobo_docker.sh && \
    rm -rf /tmp/* /var/tmp/*

# --- Copy source code before cuRobo build to avoid overwriting built artifacts later ---
COPY src src

# --- Optional: install local project in editable mode ---
RUN --mount=type=cache,target=/workspace/.uv-cache \
    uv pip install --editable . && \
    rm -rf /tmp/* /var/tmp/* && \
    chmod -R 755 /workspace/SIMPLE

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
#
#   SIMPLE_FULL_INSTALL=0 (default, minimal):
#     Hand-picked subset sufficient for all headless sim use cases that the
#     public DockerHub image is designed for (eval, replay, datagen, teleop
#     sim-side, eval-decoupled-wbc, replay-decoupled-wbc, G1Sonic robot,
#     SonicLocoManipEnv, ReplayDecoupledAgent). Avoids pulling in the
#     decoupled_wbc[full] extra, which brings ROS bridge (cv-bridge), PyQt6
#     GUI, pyrealsense2 hardware driver, Ray, mujoco, and rerun (≈ 2-4 GB).
#
#   SIMPLE_FULL_INSTALL=1 (full):
#     uv sync --group sonic — byte-equivalent to what README Option 1's
#     `uv sync --all-groups` installs on a bare host. Needed for real-robot
#     deployment, GUI teleop, distributed Ray eval, etc.
ARG SIMPLE_FULL_INSTALL
RUN --mount=type=cache,target=/workspace/.uv-cache \
    if [ "${SIMPLE_FULL_INSTALL}" = "1" ]; then \
        echo "[sonic-install] SIMPLE_FULL_INSTALL=1 — installing full sonic group" && \
        GIT_LFS_SKIP_SMUDGE=1 uv sync --group sonic --index-strategy unsafe-best-match ; \
    else \
        echo "[sonic-install] SIMPLE_FULL_INSTALL=0 — installing minimal sonic subset" && \
        uv pip install --python /workspace/SIMPLE/.venv/bin/python \
            --no-build-isolation \
            -e third_party/gear_sonic \
            -e third_party/decoupled_wbc \
            -e third_party/unitree_sdk2_python \
            tyro pin pin-pink pyyaml onnxruntime loguru termcolor qpsolvers ; \
    fi && \
    rm -rf /tmp/* /var/tmp/*

ENTRYPOINT ["/usr/bin/env"]
CMD ["/bin/bash"]
