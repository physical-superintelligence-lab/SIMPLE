Contributors: [Songlin Wei](https://songlin.github.io/), [Zhenhao Ni](https://nizhenhao-3.github.io/), [Jie Liu](https://jie0530.github.io/), [Zhenyu Zhao](https://zhenyuzhao.com/) and more (to appear) ...

> 

## What is SIMPLE?

SIMPLE stands for SIMulation-based Policy Learning and Evaluation.

It is a `simple` simulation environment supports:
  + multiple agents: (franka arm/aloha bimanual arms/dexmate wheeled robot and unitree g1 humanoid!)
  + 1000+ Objaverse assets
  + 50+ Habitat HSSD scenes
  + 50+ humanoid wholebody loco-manipulation tasks

## System Requirements

SIMPLE is built on top of `IsaacSim 4.5` and `MuJoCo 3.3`.

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 |
| **CPU** | Intel Core i7 / AMD Ryzen 7 | Intel Core i9 / AMD Ryzen 9 |
| **RAM** | 32 GB | 64 GB |
| **GPU** | NVIDIA RTX 2070 (8 GB VRAM) | NVIDIA RTX 3080 Ti / 4090 (16+ GB VRAM) |
| **NVIDIA Driver** | 535.x | Latest |
| **CUDA** | 12.x | 12.x |
| **Python** | 3.10 | 3.10 |
| **Storage** | 50 GB SSD | 100+ GB NVMe SSD |

> An RTX-class NVIDIA GPU is required. GTX and older architectures are not supported.


## Installation

Clone the project:

```
git clone git@github.com:physical-superintelligence-lab/SIMPLE.git
```

Change directory to the project root:

```
cd SIMPLE
```

Pull all submodules
```
git submodule update --init --recursive
```

We offer three options for setting up SIMPLE:

## [Option 1] UV setup (Quickest)

Install `uv` if not already done
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install all dependencies at once
```
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match
```

Install CuRobo
```
bash scripts/install_curobo.sh
```

Activate the environment:
```
source .venv/bin/activate
```

Verify the installation by printing the version number
```
python -c "import simple; print(simple.__version__)"
```

[Optional] Build the docs.

```
make live
```

Open http://127.0.0.1:8000 in a browser to view the documentation. 

> The document are working in progress. Feel free to raise questions using github issue, we will try to complete the document construction as soon as possible.

## [Option 2] Nix setup

We recommend using [nix](https://nixos.org/) on fresh new linux host, otherwise, if you alread have install NVIDIA driver and CUDA, it will be faster to setup SIMPLE through `uv`.

> [Nix](https://nixos.org/) is a modern package manager and build system that focuses on reproducibility, isolation, and declarative system configuration.

> Instead of installing software directly into your system (like apt or pip), Nix builds everything in isolated environments and stores them in the /nix/store, where each package version is uniquely identified by a hash.


<details>
<summary>Nix Setup (Recommend on fresh new hosts) </summary>

1. Install Nix first, for all interactive questions, enter `y`:

```bash
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

2. After Nix installation, open up a new shell to proceed.

If you encounter issues with `nix` command not found, try

```bash
export PATH=/nix/var/nix/profiles/default/bin:$PATH

```

3. Pull git modules recursively

```bash
git submodule update --init --recursive
```

Run the prerequisite check once on a new host:

```bash
./scripts/nix/prereq-check.sh
```

`nix develop` auto-bootstraps dependencies on first entry (or when `uv.lock` / `pyproject.toml` changes).

Start the dev shell:

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

Or run a single command inside the dev shell:

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop -c <command>
```

Do not activate the virtual environment directly with `source .venv/bin/activate` or `source .venv-nix/bin/activate`.
This repo expects the Nix shell and the Python environment to be used together. The virtual environment alone is not a supported runtime.
If your IDE terminal auto-sources `.venv-nix/bin/activate`, disable that behavior for this workspace or `deactivate` before entering through `nix develop`.

Check if install successfully.

```
python -c "import simple; print(simple.__version__)"
```

You should see version number printed.

- If encouter installtion or running issues, please checkout `Troubleshootings` in the Docs

### Download a task

```bash
export task=G1WholebodyXMovePickTeleop-v0
hf download USC-PSI-Lab/psi-data simple-eval/$task.zip --local-dir=./data/evals/ --repo-type=dataset
unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

### Running evaluation

CLI usage is supported, but it is not the preferred integration surface.

```bash
export task=G1WholebodyXMovePickTeleop-v0
export entry=eval_decoupled_wbc
export agent=gr00t_n16_decoupled_wbc
export dr=level-0

env -u LD_LIBRARY_PATH nix --extra-experimental-features 'nix-command flakes' develop -c \
  python -m simple.cli.$entry simple/$task $agent $dr \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$task/$dr \
  --host 127.0.0.1 \
  --port 21000 \
  --headless
```

### Evalating on tasks collected using the SONIC stack

```bash
 TASK_NAME=G1WholebodyXMovePickTeleop-v0 uv run eval-decoupled-wbc simple/G1WholebodyXMovePickTeleop-v0 gr00t_n16_decoupled_wbc train --data-format lerobot --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/level-0 --host 127.0.0.1 --port 21000 --headless
```

### Nix Notes

The Nix runtime is documented in detail in [`docs/source/nix-runtime.md`](./docs/source/nix-runtime.md).

Short version:

- Mutually exclusive with Docker.
- Intended host baseline: Linux with NVIDIA drivers already installed, especially Ubuntu hosts.
- Run `./scripts/nix/prereq-check.sh` first on a new host.
- Nix owns userspace; the host only owns the NVIDIA driver boundary.
- The shell fails early on runtime pollution from `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`, or `LD_PRELOAD`.
- The default Python environment is `.venv-nix`.
- Bootstrap entry points are `./scripts/nix/bootstrap-python.sh`, `./scripts/nix/bootstrap-gpu.sh`, and `./scripts/nix/bootstrap.sh`.
- Prefer importing `simple` as a library from inside the dev shell; treat the CLI as a thin convenience layer.

Operational notes:

- Remove a root-owned `.venv` left by older Docker runs with `sudo rm -rf .venv`.
- Use `SIMPLE_AUTO_BOOTSTRAP=0` to skip auto-setup, or `SIMPLE_FORCE_BOOTSTRAP=1` to force re-bootstrap.
- If you need to run `nix` from inside the dev shell, prefer `env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" ...`.


</details>

### [Option 3] Docker setup

We also support building and running SIMPLE in docker. Please refer to the documents for docker setup.

## 📊 Simulation Benchmarking Results

> This is a preliminary benchmark with 6 tasks accompanying the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) project. Please also checkout Psi-0 for more details of intergrating Psi-0 with SIMPLE.

![Quick Preview](assets/video/tasks.gif)

To rigorously evaluate the robustness and generalization of the learned policies, we design three evaluation levels with progressive out-of-distribution variations applied to the training environment:

> The evaluation environments are provided in the huggingface repository [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).

- **Level 0 (Visual & Distractors):** Randomizes table materials and the types/initial positions of distractor objects.
- **Level 1 (Lighting):** Includes Level 0 variations + extreme changes in lighting conditions.
- **Level 2 (Spatial pose):** Includes Level 1 variations + perturbations to the initial positions of the target objects.

_Success rates are reported out of 10 evaluation trials per level (**Level 0 | Level 1 | Level 2**)._
| Baseline / Task | G1Wholebody<br>XMove<br>PickTeleop-v0 | G1Wholebody<br>BendPickMP-v0 | G1Wholebody<br>Handover<br>Teleop-v0 | G1Wholebody<br>Locomotion<br>PickBetweenTables<br>Teleop-v0 | G1Wholebody<br>Tabletop<br>GraspMP-v0 | G1Wholebody<br>XMove<br>BendPick<br>Teleop-v0 |
| :--------------- | :-----------------------------------: | :--------------------------: | :----------------------------------: | :---------------------------------------------------------: | :-----------------------------------: | :-------------------------------------------: |
| **Psi0** | 10 &#124; 10 &#124; 6 | 10 &#124; 10 &#124; 10 | 7 &#124; 7 &#124; 10 | 7 &#124; 5 &#124; 6 | 10 &#124; 10 &#124; 8 | 10 &#124; 9 &#124; 9 |
| **GR00T N1.6** | 10 &#124; 10 &#124; 7 | 7 &#124; 7 &#124; 6 | 1 &#124; 3 &#124; 3 | 0 &#124; 0 &#124; 0 | 9 &#124; 9 &#124; 7 | 4 &#124; 4 &#124; 1 |
| **OpenPi π0.5** | 7 &#124; 5 &#124; 1 | 10 &#124; 10 &#124; 8 | 5 &#124; 4 &#124; 5 | 3 &#124; 3 &#124; 3 | 10 &#124; 10 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **InternVLA-M1** | 0 &#124; 0 &#124; 0 | 5 &#124; 5 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 3 &#124; 5 &#124; 7 |
| **H-RDT** | 0 &#124; 0 &#124; 2 | 0 &#124; 0 &#124; 1 | 0 &#124; 1 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 |
| **DreamZero** | - &#124; - &#124; - | - &#124; - &#124; - | - &#124; - &#124; - | - &#124; - &#124; - | 9 &#124; 10 &#124; 10 | - &#124; - &#124; - |
| **EgoVLA** | 0 &#124; 1 &#124; 2 | 7 &#124; 5 &#124; 8 | 0 &#124; 4 &#124; 3 | 0 &#124; 0 &#124; 0 | 10 &#124; 10 &#124; 7 | 3 &#124; 5 &#124; 4 |
| **Diff. Policy** | 3 &#124; 3 &#124; 2 | 10 &#124; 8 &#124; 6 | 3 &#124; 2 &#124; 4 | 4 &#124; 0 &#124; 0 | 8 &#124; 9 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **ACT** | 10 &#124; 9 &#124; 6 | 10 &#124; 9 &#124; 9 | 4 &#124; 4 &#124; 6 | 6 &#124; 5 &#124; 7 | 10 &#124; 10 &#124; 8 | 6 &#124; 8 &#124; 8 |

## Citation
```
@misc{wei2026psi0,
  title={$\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation}, 
  author={Songlin Wei and Hongyi Jing and Boqian Li and Zhenyu Zhao and Jiageng Mao and Zhenhao Ni and Sicheng He and Jie Liu and Xiawei Liu and Kaidi Kang and Sheng Zang and Weiduo Yuan and Marco Pavone and Di Huang and Yue Wang},
  year={2026},
  eprint={2603.12263},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2603.12263}, 
}
```

## License
This project is licensed under the MIT.

See the [LICENSE](license.md) file for details.
