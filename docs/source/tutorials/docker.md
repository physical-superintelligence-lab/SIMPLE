# Docker Setup

## Prebuilt image

If you just want to run SIMPLE without building, pull the prebuilt minimal
image from DockerHub:

```
docker pull ghcr.io/physical-superintelligence-lab/simple:latest
```

Available tags and image history: https://github.com/physical-superintelligence-lab/SIMPLE/pkgs/container/simple

The published image is the **minimal** variant — sufficient for all headless
sim use cases (`eval`, `replay`, `datagen`, decoupled-WBC sim side). For the
full stack (real-robot / GUI teleop), build locally with
`SIMPLE_FULL_INSTALL=1` (see [Build variants](#build-variants) below) or
[upgrade a running minimal container](#upgrading-a-running-minimal-container-to-the-full-stack).

## Build from source

build and start docker container

First pull all submodules

```
git submodule update --init --recursive
```

create `.uv-cache` folder for building and running the image
```
mkdir -p .uv-cache
```

(a)  create and edit `.env`
```
cp .env.sample .env  
# --> change DATA_DIR to your path on host
# --> DATE is used as the docker image tag
# --> other variables are optional
```

> 💡 Compiling cuda kernels for every compute capability can significantly increase the install time of `CuRobo`, it’s reccommended to set the environment variable `TORCH_CUDA_ARCH_LIST` to the correct computablity according to [offical doc](https://developer.nvidia.com/cuda-gpus)
> ```
> TORCH_CUDA_ARCH_LIST=12.0+PTX # for 5090 etc., 
> TORCH_CUDA_ARCH_LIST=8.9+PTX # for 4090 etc., 
> TORCH_CUDA_ARCH_LIST=8.0+PTX # for A100 etc., 
> ```

(b)  build isaac-sim image
```
docker buildx bake --allow=network.host isaac-sim
# or build without cache
docker buildx bake --allow=network.host --no-cache isaac-sim
```

>  ***Build variants***
> 
> By default, `docker buildx bake isaac-sim` produces a **minimal** image sufficient
> for all headless sim use cases: `eval`, `replay`, `datagen`, `teleop-decoupled-wbc`
> (sim-side), `eval-decoupled-wbc`, `replay-decoupled-wbc`, and any environment
> that instantiates `G1Sonic` / `SonicLocoManipEnv` / `ReplayDecoupledAgent`.
>
> To include the **full** `decoupled_wbc[full]` stack (ROS bridge, PyQt6 GUI,
> pyrealsense2, Ray, mujoco, rerun) — needed for real-robot deployment or GUI
> teleop — set `SIMPLE_FULL_INSTALL=1` at build time:
> 
> ```
> SIMPLE_FULL_INSTALL=1 docker buildx bake --allow=network.host isaac-sim
> ```
> 
> The full variant is byte-equivalent to what README Option 1's
> `uv sync --all-groups` installs on a bare host. Image size increases by
> roughly 2-4 GB.

Test installtion by running a demo data generation

```
docker compose -p wsl run datagen \
    simple/G1WholebodyBendPickMP-v0 \
    --render-hz=50 \
    --sim-mode=mujoco_isaac \
    --headless \
    --num-episodes=1
```
You should see 1 episode generated succesfully!
The video is saved to `data/datagen/simple`.

Or you can follow below (c-e) steps to launch the container manually.

(c) start sim container in detached mode
```
docker compose up sim -d
# or use -p to specify a project name (usefull when multiple developer share same host)
docker compose -p wsl up -d sim
```

(d) attach to the container
```
docker attach simple-sim-1
```

(e) run test code to verify installation
```
cd /workspace/SIMPLE
source .venv/bin/activate
python src/simple/cli/datagen.py simple/G1WholebodyBendPickMP-v0 --render-hz=50 --sim-mode=mujoco_isaac --headless --num-episodes=1
```



### Useful Commands

#### Upgrading a running minimal container to the full stack

If you built with the default (minimal) variant but need the full
`decoupled_wbc[full]` stack (ROS, PyQt6, realsense, Ray, mujoco, rerun)
without rebuilding the image:

```
docker exec -it simple-sim-1 bash -c "\
  uv pip install --python /workspace/SIMPLE/.venv/bin/python \
    --no-build-isolation \
    -e 'third_party/decoupled_wbc[full]' \
    -e third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
    -e third_party/evdev"
```

The `third_party/` tree inside the container is baked at image build time,
so the install uses that frozen snapshot. If you need bleeding-edge
`decoupled_wbc`, either rebuild with `SIMPLE_FULL_INSTALL=1` or bind-mount
`third_party/` when launching the container.

#### take down all running containers
```
docker compose down --remove-orphans
```

#### GPU selection

If you want to specify which GPU to run the command, just prepend `GPUs=0` to any following command:
```
# eg.,
GPUs=0 ...
```

> If `GPUs` has no effect, then try to change `privileged=true/false`. I've experienced different environments with both settings.

Run `datagen`
```
GPUs=0 docker compose -p wsl run datagen \
    simple/G1WholebodyBendPickMP-v0 \
    --render-hz=50 \
    --sim-mode=mujoco_isaac \
    --headless \
    --num-episodes=1
```


***If you are behind the GFW, please setup proxy following guide at the end of this doc.***


### *Setup docker proxies

1. Setup local Clash proxy and use your local ipv4 address. 
DO NOT SET IP ADDRESS TO `127.0.0.1`!
It is used when docker build. 
    ```
    # vim ~/.docker/config.json
    "proxies": {
        "default": {
            "httpProxy": "http://192.168.1.55:7890",
            "httpsProxy": "http://192.168.1.55:7890",
            "allProxy": "socks5://192.168.1.55:7890",
            "noProxy": "192.168.1.55/8"
        }
    }
    ```

2. Setup proxy mirros used when docker pull, etc

    ```
    # sudo vim /etc/docker/daemon.json
    {
        ...
        "registry-mirrors": [
            "https://mirror.ccs.tencentyun.com",
            "https://05f073ad3c0010ea0f4bc00b7105ec20.mirror.swr.myhuaweicloud.com",
            "https://registry.docker-cn.com",
            "http://hub-mirror.c.163.com",
            "http://f1361db2.m.daocloud.io"
        ]
    }
    ```

3. Turn on clash to allow LAN
    ```
    # vim ~/Clash/config.yaml
    allow-lan: true

    # test in your terminal
    export HOST_IP=192.168.61.221
    export all_proxy=socks5://${HOST_IP}:7890
    export all_proxy=socks5://${HOST_IP}:7890
    export https_proxy=http://${HOST_IP}:7890
    export http_proxy=http://${HOST_IP}:7890
    export no_proxy=localhost,${HOST_IP}/8,::1
    export ftp_proxy=http://${HOST_IP}:7890/

    # check env variables are set
    env | grep proxy

    # test connection
    curl -I https://www.google.com
    ```

4. Try to uncomment the lines in dockerfile which changes ubuntu apt sources to aliyun.
    ```
    # Change apt source if you encouter connection issues
    RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
        sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
    ```

5. Restart docker [and then build again]
    ```
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

6. If still not work, add following building options to `docker build`
   ```
    ...
    --build-arg http_proxy=http://192.168.1.55:7890 \
    --build-arg https_proxy=http://192.168.1.55:7890 \
    ...
   ```