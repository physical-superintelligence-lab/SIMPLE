# Docker Setup

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
# --> then change all the env variables like: MY_USER, MY_UID, MY_GID (should be same as your host)
# --> change DATA_DIR to your path on host
# --> other variables are optional
```

> 💡 Compiling cuda kernels for every compute capability can significantly increase the install time of `CuRobo`, it’s reccommended to set the environment variable `TORCH_CUDA_ARCH_LIST` to the correct computablity according to [offical doc](https://developer.nvidia.com/cuda-gpus)
> ```
> TORCH_CUDA_ARCH_LIST=8.9+PTX # for 4090 etc., 
> TORCH_CUDA_ARCH_LIST=8.0+PTX # for A100 etc., 
> ```

(b)  build isaac-sim image
```
docker buildx bake --allow=network.host isaac-sim
# or build without cache
docker buildx bake --allow=network.host --no-cache isaac-sim
```

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
cd ~/SIMPLE
uv run replay
```

You should see episode replayed succesfully!
The video is saved to `output/replays`

(f) usefull commands

take down all running containers
```
docker compose down --remove-orphans
```

### Useful Commands

If you want to specify which GPU to run the command, just prepend `GPUs=0` to any following command:
```
# eg.,
GPUs=0 ...
```

> If `GPUs` has no effect, then try to change `privileged=true/false`. I've experienced different environments with both settings.

Run `datagen`
```
docker compose -p wsl run datagen simple/FrankaTabletopGrasp-v0 --headless --ignore-target-collision
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