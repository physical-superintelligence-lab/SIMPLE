<h1 align="center">SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2606.08278-df2a2a.svg)](https://arxiv.org/abs/2606.08278)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://psi-lab.ai/SIMPLE)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/USC-PSI-Lab/psi-model)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/datasets/USC-PSI-Lab/psi-data)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>


<p align="center">
  <img src="assets/teaser.webp" alt="SIMPLE teaser image" />
</p>


Contributors: [Songlin Wei](https://songlin.github.io/)\*, [Zhenhao Ni](https://nizhenhao-3.github.io/)\*, [Jie Liu](https://jie0530.github.io/)\*, [Zhenyu Zhao](https://zhenyuzhao.com/)\*, [Junjie Ye](https://junjieye.com/), [Hongyi Jing](https://hongyijing.me/), Junkai Xia, [Xiawei Liu](https://www.xiaweiliu.com/), [Michael Leong](https://leongmichael.github.io/), [Liang Heng](https://liangheng121.github.io/), Di Huang, [Yue Wang](https://yuewang.xyz/)†

> 

## Table of Contents
- [What is SIMPLE?](#what～is～SIMPLE)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [[Option 1] UV setup (Quickest)](#option-1-uv-setup-quickest)
  - [[Option 2] Nix setup](#option-2-nix-setup)
  - [[Option 3] Docker setup](#option-3-docker-setup)
- [Data Generation & Pipeline](#-data-generation--pipeline)
  - [1. Data Collection ](#1-data-collection-methods)
  - [2. Data Post-processing](#2-post-processing)
  - [3. Fine-Tuning](#3-fine-tuning)
- [Evaluation in SIMPLE](#-evaluation-in-simple)
- [📊 Simulation Benchmarking Results](#-simulation-benchmarking-results)
- [Citation](#citation)
- [License](#license)

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
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

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

Open http://127.0.0.1:8005 in a browser to view the documentation.

> The document are working in progress. Feel free to raise questions using github issue, we will try to complete the document construction as soon as possible.

## [Option 2] Nix setup

We recommend using [nix](https://nixos.org/) on fresh new linux host, otherwise, if you alread have install NVIDIA driver and CUDA, it will be faster to setup SIMPLE through `uv`.

> [Nix](https://nixos.org/) is a modern package manager and build system that focuses on reproducibility, isolation, and declarative system configuration.

> Instead of installing software directly into your system (like apt or pip), Nix builds everything in isolated environments and stores them in the /nix/store, where each package version is uniquely identified by a hash.

1. Install Nix first, for all interactive questions, enter `y`:

```bash
sh <(curl --proto '=https' --tlsv1.2 -L [https://nixos.org/nix/install](https://nixos.org/nix/install)) --daemon

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

`nix develop` auto-booststraps dependencies on first entry (or when `uv.lock` / `pyproject.toml` changes).

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

* If encouter installtion or running issues, please checkout `Troubleshootings` in the Docs


### Nix Notes

The Nix runtime is documented in detail in [`docs/source/nix-runtime.md`](https://www.google.com/search?q=./docs/source/nix-runtime.md).

Short version:

* Mutually exclusive with Docker.
* Intended host baseline: Linux with NVIDIA drivers already installed, especially Ubuntu hosts.
* Run `./scripts/nix/prereq-check.sh` first on a new host.
* Nix owns userspace; the host only owns the NVIDIA driver boundary.
* The shell fails early on runtime pollution from `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`, or `LD_PRELOAD`.
* The default Python environment is `.venv-nix`.
* Bootstrap entry points are `./scripts/nix/bootstrap-python.sh`, `./scripts/nix/bootstrap-gpu.sh`, and `./scripts/nix/bootstrap.sh`.
* Prefer importing `simple` as a library from inside the dev shell; treat the CLI as a thin convenience layer.

Operational notes:

* Remove a root-owned `.venv` left by older Docker runs with `sudo rm -rf .venv`.
* Use `SIMPLE_AUTO_BOOTSTRAP=0` to skip auto-setup, or `SIMPLE_FORCE_BOOTSTRAP=1` to force re-bootstrap.
* If you need to run `nix` from inside the dev shell, prefer `env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" ...`.

### [Option 3] Docker setup

We also support building and running SIMPLE in docker. Please refer to the documents for [docker setup](https://www.google.com/search?q=docs/source/tutorials/docker.md).

---

## ⚙️ Data Generation & Pipeline

SIMPLE provides a scalable pipeline to generate, process, and train policies using synthesized simulation data.

### 1. Data Collection 

We support two primary interfaces for gathering  data: **Teleoperation (human-in-the-loop)** and **Automated Motion Planning**. 

Before running, adjust your environment variables to match your system topology.
```bash
# Example configurations (Adjust CUDA_VISIBLE_DEVICES and DISPLAY based on your host)
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="0" 
export DISPLAY=":1"
```




##### Stage 1: Teleoperation in MuJoCo

We perform the initial human-in-the-loop teleoperation inside the lightweight MuJoCo engine. This ensures minimal control loop latency and high-frequency physical interactions during the demonstration tracking.

**Example Usage:**

```bash
export TASK_NAME=G1WholebodyOpenTrashCanTeleop-v0

python -m simple.cli.teleop_decoupled_wbc \
  simple/$TASK_NAME \
  --target=graspnet1b:0 \
  --sim-mode=mujoco \
  --record \
  --no-headless \
  --success-criteria=2


```
> 🥽 **Hardware Setup:** We utilize **Pico VR headsets** for immersive human-in-the-loop teleoperation. For specific hardware configuration, controller mapping, and connection details, please refer to the [Teleoperation Setup Guide](docs/source/tutorials/teleop.md).


> 💡 *To explore additional customizable options for teleoperation, run:*
> `python -m simple.cli.teleop_decoupled_wbc --help`

Supported Wholebody Teleop Tasks Include:

* `simple/G1WholebodyOpenTrashCanTeleop-v0`
* `simple/G1WholebodyBendPickTeleop-v0`
* `simple/G1WholebodyBendPickAndPlaceTeleop-v0`
* `simple/G1WholebodyBendHandoverTeleop-v0`
* `simple/G1WholebodyPushOfficeChairTeleop-v0`
* `simple/G1WholebodyOpenFaucetTeleop-v0`
* `simple/G1WholebodyOpenOvenTeleop-v0`
* `simple/G1WholebodyCloseDoorTeleop-v0`
* `simple/G1WholebodyXMovePickTeleop-v0`
* `simple/G1WholebodyXMoveBendPickTeleop-v0`
* `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
* `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
* `simple/G1WholebodyHandoverTeleop-v0`


##### Stage 2: Photorealistic Replay & Isaac Sim Rendering

Once raw trajectories are successfully captured, pass them into the `replay_decoupled_wbc` suite. By specifying `--sim-mode=mujoco_isaac`, this stage replays the actions in MuJoCo while driving **Isaac Sim** simultaneously as a synchronized rendering engine. This step processes the raw stream into standard dataset structures (LeRobot format).

**Example Usage:**

```bash
# Ensure $TASK_NAME matches the task used in Stage 1
python -m simple.cli.replay_decoupled_wbc \
  simple/$TASK_NAME \
  --data-dir=data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0/ \
  --sim-mode=mujoco_isaac \
  --no-headless \
  --render-hz=50 \
  --save-dir=data/replay_decoupled_wbc_output \
  --record \
  --resume \
  --success-criteria=0.2

```

> 💡 **Tip:** If the replay success rate is low, try lowering the `--success-criteria` first.



#### B. Automated Motion Planning 

To bypass manual human interaction and scale up synthetic data generation, the `simple.cli.datagen` pipeline directly integrates **CuRobo for automated motion planning**. This allows us to procedurally batch-produce optimal demonstration trajectories without human teleop.

Unlike the two-stage teleoperation process, **Motion Planning can be executed in a single step**. By setting `--sim-mode=mujoco_isaac`, the pipeline resolves the fast contact physics and motion planning within MuJoCo, while simultaneously driving Isaac Sim for photorealistic rendering. This directly outputs the final dataset in the standard LeRobot format.

**Example Usage:**

```bash
export TASK_NAME=G1WholebodyTabletopHandoverMP-v0

python -m simple.cli.datagen \
  simple/$TASK_NAME \
  --sim-mode=mujoco_isaac \
  --render-hz=50 \
  --no-headless \
  --num-episodes=10

```





### 2. Post-processing

To prepare the generated datasets for policy learning, we need to post-process the raw output data to be strictly compatible with the training pipeline of our foundation model, [Psi-0](https://github.com/physical-superintelligence-lab/Psi0).

We provide two distinct post-processing scripts depending on how the data was collected:

#### A. Post-processing Motion Planning Data
For data generated via the automated motion planning pipeline (`datagen.py`), use `postprocess_psi0.py`. **This script supports wildcard matching (`*`)** to seamlessly merge data from multiple parallel generation batches into a single unified dataset.

**Example Usage:**
```bash
python scripts/postprocess_psi0.py \
  --sim-root="data/datagen*/simple/G1WholebodyXMoveBendPickMP-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyXMoveBendPickMP-v0 \
  --skip=60

```

#### B. Post-processing Teleoperation Data

For data captured through human teleoperation and rendered via Isaac Sim , use `postprocess_psi0_sonic.py`. Similarly, this script utilizes wildcard matching (`*`) to merge data from multiple teleop replay sessions.

**Example Usage:**

```bash
python scripts/postprocess_psi0_sonic.py \
  --sim-root="data/replay_decoupled_wbc_output*/simple/G1WholebodyPushOfficeChairTeleop-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyPushOfficeChairTeleop-v0 \
  --skip=0 \
  --total_episodes=100

```

**Key Arguments:**

* `--sim-root`: The input directory containing the generated dataset. Note that quotes `""` are highly recommended when using wildcards (`*`) to prevent premature shell expansion.
* `--out-dir`: The output directory where the Psi-0 compatible dataset will be saved.
* `--skip`: Number of initial frames to skip (useful for bypassing static setup or initialization frames).
* `--total_episodes`: Limits the total number of valid episodes to process and merge.



### 3. Fine-Tuning

To train or fine-tune foundation models directly using the structured datasets generated from the pipeline, we provide seamless integration with the **Psi-0** training stack.

> 👉 **Quick Start:** You can skip fine-tuning entirely and evaluate right away by downloading our pre-trained [checkpoints for SIMPLE](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints).

**Data Preparation:**
If you wish to train from scratch or fine-tune, download the required [SIMPLE task data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple) and extract it to your local workspace:

```bash
export TASK_NAME=G1WholebodyXMovePickTeleop-v0

hf download USC-PSI-Lab/psi-data \
  simple/$TASK_NAME.zip \
  --local-dir=data \
  --repo-type=dataset

unzip data/simple/$TASK_NAME.zip -d data/simple

```

**Training Integration:**

> 💡 **For full training instructions, please refer to the [Psi-0 Project README](https://github.com/physical-superintelligence-lab/Psi0).** >
> The Psi-0 repository contains comprehensive, up-to-date documentation on setting up training environment variables, visualizing episodes, and launching the training scripts (e.g., `bash scripts/train/psi0/finetune-simple-psi0.sh`).



## 🎯 Evaluation in SIMPLE

To rigorously evaluate the robustness and generalization of learned policies, we benchmark our foundation model [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) using a decoupled **Client-Server architecture**. The server hosts the model inference, while the SIMPLE client runs the simulation environment.

---

### 🖥️ Server Side: Model Inference (Executed in the Psi-0 Repository)

#### Step 1: Environment & Checkpoint Setup
Configure the evaluation environment variables and paths within your **Psi-0** project workspace.

1. **Configure Environment Variables:** Inside the **Psi-0** project root, create and source your `.env` file based on the sample:
```bash
  cp .env.sample .env
  # Edit .env to include your HF_TOKEN, WANDB variables, and PSI_HOME path
  source .env
  echo $PSI_HOME # Verify the path is correctly set
```

2. **Download Pre-trained Weights:** Pull the Psi-0 checkpoints for the SIMPLE benchmark from our Hugging Face repository. Psi0's pre-trained weights for the SIMPLE benchmark are hosted on the Hugging Face Model Hub at [USC-PSI-Lab/psi-model](https://huggingface.co/USC-PSI-Lab/psi-model).

```bash
hf download USC-PSI-Lab/psi-model \
  --include="psi0/simple-checkpoints/*" \
  --local-dir=$PSI_HOME/.runs \
  --repo-type=model

```

### Step 2: Start the Psi-0 Inference Server

Before launching the simulation, initialize the model inference server.

```bash
# Set your target run directory and checkpoint step
export RUN_DIR=xxxx
export CKPT_STEP=40000

# Start the server (Listens on port 22085 by default)
bash scripts/deploy/serve_psi0_simple.sh $RUN_DIR $CKPT_STEP

```

> ⚠️ **Important:** Keep this terminal window open. The server must remain active for the duration of the evaluation.

### Step 3: Run the SIMPLE Simulation Client

Open a **new terminal window** to launch the environment. The execution parameters differ slightly based on the data source of the task:

* **For Teleop Tasks (suffix `*Teleop-v0`):** Use decoupled Whole-Body Control.
* `export entry=eval_decoupled_wbc`
* `export agent=psi0_decoupled_wbc`


* **For Motion Planning Tasks (suffix `*MP-v0`):** Use standard evaluation.
* `export entry=eval`
* `export agent=psi0`



**Execution Example (Teleop Task):**


#### Option A: UV Environment

```bash
export task=G1WholebodyXMovePickTeleop-v0
export agent=psi0_decoupled_wbc
export dr=level-0

TASK_NAME=$task uv run eval-decoupled-wbc \
    simple/$task \
    $agent \
    train \
    --data-format lerobot \
    --data-dir data/evals/simple-eval/$task/$dr \
    --host 127.0.0.1 \
    --port 21000 \
    --headless
```

#### Option B: Nix Environment

```bash
export task=G1WholebodyXMovePickTeleop-v0
export entry=eval_decoupled_wbc
export agent=psi0_decoupled_wbc
export dr=level-0

env -u LD_LIBRARY_PATH nix --extra-experimental-features 'nix-command flakes' develop -c \
  python -m simple.cli.$entry \
  simple/$task \
  $agent \
  train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$task/$dr \
  --host 127.0.0.1 \
  --port 21000 \
  --headless
```

### Step 4: View Evaluation Results & Videos

**Task Success Rate Statistics:**
Upon completion, the terminal will display a summary of the results. A detailed log is also preserved automatically:

```bash
cat data/evals_decoupled_wbc/eval_stats.txt

```

**Execution Videos:**
Visual records of each episode are automatically rendered and saved. The files are named using the pattern `episode_id/cam_name_{success_flag}.mp4` (e.g., `success` or `failed`).

```bash
# Example: Play a successful teleop evaluation video
mpv data/evals_decoupled_wbc/psi0_decoupled_wbc/G1WholebodyXMovePickTeleop-v0/level-0/episode_0/head_stereo_left_success.mp4

# Example: Play a successful motion planning evaluation video
mpv data/evals/psi0/G1WholebodyBendPickMP-v0/level-0/episode_0/front_stereo_left_success.mp4

```




## 📊 Simulation Benchmarking Results

> This is a preliminary benchmark with 6 tasks accompanying the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) project. Please also checkout Psi-0 for more details of intergrating Psi-0 with SIMPLE.

To rigorously evaluate the robustness and generalization of the learned policies, we design three evaluation levels with progressive out-of-distribution variations applied to the training environment:

> The evaluation environments are provided in the huggingface repository [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).

* **Level 0 (Visual & Distractors):** Randomizes table materials and the types/initial positions of distractor objects.
* **Level 1 (Lighting):** Includes Level 0 variations + extreme changes in lighting conditions.
* **Level 2 (Spatial pose):** Includes Level 1 variations + perturbations to the initial positions of the target objects.

_Success rates are reported out of 10 evaluation trials per level (**Level 0 | Level 1 | Level 2**)._
| Baseline / Task | G1Wholebody<br>XMove<br>PickTeleop-v0 | G1Wholebody<br>BendPickMP-v0 | G1Wholebody<br>Handover<br>Teleop-v0 | G1Wholebody<br>Locomotion<br>PickBetweenTables<br>Teleop-v0 | G1Wholebody<br>Tabletop<br>GraspMP-v0 | G1Wholebody<br>XMove<br>BendPick<br>Teleop-v0 |
| :--------------- | :-----------------------------------: | :--------------------------: | :----------------------------------: | :---------------------------------------------------------: | :-----------------------------------: | :-------------------------------------------: |
| **Psi0** | 10 &#124; 10 &#124; 6 | 10 &#124; 10 &#124; 10 | 7 &#124; 7 &#124; 10 | 7 &#124; 5 &#124; 6 | 10 &#124; 10 &#124; 8 | 10 &#124; 9 &#124; 9 |
| **GR00T N1.6** | 10 &#124; 10 &#124; 7 | 7 &#124; 7 &#124; 6 | 1 &#124; 3 &#124; 3 | 0 &#124; 0 &#124; 0 | 9 &#124; 9 &#124; 7 | 4 &#124; 4 &#124; 1 |
| **OpenPi π0.5** | 7 &#124; 5 &#124; 1 | 10 &#124; 10 &#124; 8 | 5 &#124; 4 &#124; 5 | 3 &#124; 3 &#124; 3 | 10 &#124; 10 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **InternVLA-M1** | 0 &#124; 0 &#124; 0 | 5 &#124; 5 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 3 &#124; 5 &#124; 7 |
| **H-RDT** | 0 &#124; 0 &#124; 2 | 0 &#124; 0 &#124; 1 | 0 &#124; 1 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 |
| **DreamZero** | 10 &#124; 10 &#124; 10 | 9 &#124; 9 &#124; 8 | 7 &#124; 8 &#124; 9 | 5 &#124; 3 &#124; 3 | 9 &#124; 10 &#124; 7 | 0 &#124; 0 &#124; 1 |
| **EgoVLA** | 0 &#124; 1 &#124; 2 | 7 &#124; 5 &#124; 8 | 0 &#124; 4 &#124; 3 | 0 &#124; 0 &#124; 0 | 10 &#124; 10 &#124; 7 | 3 &#124; 5 &#124; 4 |
| **Diff. Policy** | 3 &#124; 3 &#124; 2 | 10 &#124; 8 &#124; 6 | 3 &#124; 2 &#124; 4 | 4 &#124; 0 &#124; 0 | 8 &#124; 9 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **ACT** | 10 &#124; 9 &#124; 6 | 10 &#124; 9 &#124; 9 | 4 &#124; 4 &#124; 6 | 6 &#124; 5 &#124; 7 | 10 &#124; 10 &#124; 8 | 6 &#124; 8 &#124; 8 |

_More interesting tasks, including articulated objects._

| Baseline / Task | G1Wholebody<br>CloseDoor<br>Teleop-v0 | G1Wholebody<br>OpenOven<br>Teleop-v0 | G1Wholebody<br>OpenFaucet<br>Teleop-v0 | G1Wholebody<br>PickAndPlace<br>AndHugContainer<br>Teleop-v0 | 
| :--------------- | :-----------------------------------: | :--------------------------: | :----------------------------------: | :---------------------------------------------------------: | 
| **Psi0** | 10 &#124; 10 &#124; 10 | 7 &#124; 5 &#124; 4 | 3 &#124; 3 &#124; 4 | 7 &#124; 6 &#124; 3 | 

## Citation

> Please also consider citing `Psi-0` if you use its training code.

```
@article{wei2026simple,
  title={SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation},
  author={Wei, Songlin and Ni, Zhenhao and Liu, Jie and Zhao, Zhenyu and Ye, Junjie and Jing, Hongyi and Xia, Junkai and Liu, Xiawei and Leong, Michael and Heng, Liang and Huang, Di and Wang, Yue},
  journal={arXiv preprint arXiv:2606.08278},
  year={2026}
}
```

```
@article{wei2026psi0,
  title={{$\Psi_0$}: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation},
  author={Wei, Songlin and Jing, Hongyi and Li, Boqian and Zhao, Zhenyu and Mao, Jiageng and Ni, Zhenhao and He, Sicheng and Liu, Jie and Liu, Xiawei and Kang, Kaidi and others},
  journal={arXiv preprint arXiv:2603.12263},
  year={2026}
}
```

## License

This project is licensed under the MIT.

See the [LICENSE](https://www.google.com/search?q=license.md) file for details.

