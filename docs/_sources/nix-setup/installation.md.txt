# Installation

We also support install SIMPLE using `nix`. The supported contract is:

1. check host prerequisites
2. enter the Nix dev shell
3. import `simple` as a Python library inside that shell

## Supported Host Shape

The intended baseline is:

- Linux
- NVIDIA drivers already installed on the host
- Ubuntu-style NVIDIA driver layouts or NixOS `/run/opengl-driver/...`
- enough free disk for the CUDA closure and local caches

## Install Steps

Clone the repo and initialize submodules:

```bash
git clone git@github.com:songlin/SIMPLE.git
cd SIMPLE
git submodule update --init --recursive
```

Install Nix if needed:

```bash
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

Run the host prerequisite check:

```bash
./scripts/nix/prereq-check.sh
```

Enter the dev shell:

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

Check the install:

```bash
python -c "import simple; print(simple.__version__)"
```

## Preferred Usage: Import The Library

Prefer using SIMPLE as a Python library from inside the dev shell:

```python
import gymnasium as gym
import simple.envs as _

env = gym.make("simple/FrankaTabletopGrasp-v0", sim_mode="isaac", headless=True)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

CLI commands such as `eval` and `datagen` are supported, but they are thin convenience wrappers. If another codebase integrates SIMPLE, it should usually import `simple` directly instead of launching subprocesses.

If you need to invoke `nix` directly from an uncertain shell, prefer:

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

## Build The Docs

```bash
make live
```

Then open [http://127.0.0.1:8005](http://127.0.0.1:8005) to read the documentation.

## Optional Resource Downloads

Many scripts download extra data automatically when needed.
If you want to pre-download the shared resources and data, run:

```bash
./scripts/download_data.sh
```

The structure of the `data` folder should look like:

```text
data
├── assets
│   ├── graspnet
│   └── ...
├── robots
│   ├── robot_1
│   └── ...
├── scenes
└── vMaterials
    └── ...
```

## Optional Bootstrap Entry Points

Use these only when you need to run parts of the install explicitly:

```bash
./scripts/nix/bootstrap-python.sh
./scripts/nix/bootstrap-gpu.sh
./scripts/nix/bootstrap.sh
```
