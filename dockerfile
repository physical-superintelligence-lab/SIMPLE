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

COPY --from=cuda /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda-${CUDA_VERSION}
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}

# Change apt source if you encouter connection issues
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
#     sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# fix libstdc++.so.6 version issue 
# strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep 3.4.32
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg dirmngr \
    git curl wget vim ca-certificates build-essential ninja-build libgmp-dev libgmp10 ffmpeg \
    python3 python3-pip python3-dev software-properties-common \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
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

# --- Copy dependency files and cuRobo (cache key) ---
WORKDIR /workspace/SIMPLE
COPY pyproject.toml uv.lock ./
COPY third_party/curobo third_party/curobo
COPY third_party/openpi-client third_party/openpi-client

# --- Install dependencies (cached) ---
ENV UV_HTTP_TIMEOUT=1800
ENV UV_HTTP_RETRIES=10
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7
ENV OMNI_KIT_ACCEPT_EULA=Y

RUN --mount=type=cache,target=/workspace/.uv-cache \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds --group lerobot --group isaacsim --index-strategy unsafe-best-match && \
    rm -rf /tmp/* /var/tmp/*

# --- Copy install script (separate from cuRobo files) ---
COPY scripts/install_curobo.sh scripts/

# --- Fix curobo + warp-lang installations (with cache for smaller image) ---
RUN --mount=type=cache,target=/workspace/.uv-cache \
    ./scripts/install_curobo.sh && \
    rm -rf /tmp/* /var/tmp/*

# --- Copy source code before cuRobo build to avoid overwriting built artifacts later ---
COPY src src

# --- Optional: install local project in editable mode ---
RUN --mount=type=cache,target=/workspace/.uv-cache \
    uv pip install --editable . && \
    rm -rf /tmp/* /var/tmp/* && \
    chmod -R 755 /workspace/SIMPLE

ENTRYPOINT ["/usr/bin/env"]
CMD ["/bin/bash"]
