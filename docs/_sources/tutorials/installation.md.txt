# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment
```
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match
```

Check if installation is done. You should see version number printed.
```
python -c "import simple; print(simple.__version__)"
```

Activate virtual env to start hacking!
```
source .venv/bin/activate
```

## Install CuRobo

> Currently, CuRobo is a must-have dependency, but we might remove it in the future.

Some modules use [CuRobo](https://curobo.org/index.html) for GPU-accelerated motion planning, forward-kinematics and inverse kinematics.

CuRobo requires `git-lfs`.

```
sudo apt install git-lfs
```

Then install [`CUDA`](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux). We tested versions `11.8` and `12.4`. Similar versions should also work.

Now install `CuRobo`, run:

> 💡 Compiling cuda kernels for every compute capability can significantly increase the install time of `CuRobo`, it’s reccommended to set the environment variable `TORCH_CUDA_ARCH_LIST` to the correct computablity according to [offical doc](https://developer.nvidia.com/cuda-gpus)
> ```
> export TORCH_CUDA_ARCH_LIST=12.0+PTX # for 5090 etc., 
> # export TORCH_CUDA_ARCH_LIST=8.9+PTX # for 4090 etc., 
> ```


```
./scripts/install_curobo.sh
```

Test CuRobo installation:

> If you happen to have a display then you can run the official arm reacher example with GUI.
> Please refer to the [official cuRobo doc](https://curobo.org/get_started/2b_isaacsim_examples.html#multi-arm-reacher) to learn more about operating the examples.

> Hint: It may take a while the first time you launch Isaac Sim. 

```
python examples/multi_arm_reacher.py
```

Test SIMPLE installation
```
python scripts/list_env.py
```


## [Optional] Download Resource Files

Many scripts in this project requires downloading extra data and files.
Most of them will be downloaded ***automatically*** when running the script.
However, if you want to ***pre-download ALL neccessary***  the resources and data, run the following command:
```bash
scripts/pre-minimal-download.sh
```
Optionally, append `--cleanup` to delete the zip files once extracted.

> Whenever a download is interrupted, simply re-run the script to resume from where it left off.

The structure of the ```data``` folder should be like this:
```text
data
├── assets
│   ├── graspnet
│   └── ...
├── robots
│   ├── [robot_1]
│   ├── [robot_2]
│   └── ...
├── scenes
└── vMaterials
└── ...
```
