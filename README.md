<h1 align="center">SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2606.08278-df2a2a.svg)](https://arxiv.org/abs/2606.08278)
[![Documentation](https://img.shields.io/badge/Documentation-a)](https://psi-lab.ai/SIMPLE/docs/)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/collections/USC-PSI-Lab/simple)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/collections/USC-PSI-Lab/simple)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>


<p align="center">
  <img src="assets/teaser.webp" alt="SIMPLE teaser image" />
</p>


Contributors: [Songlin Wei](https://songlin.github.io/)\*, [Zhenhao Ni](https://nizhenhao-3.github.io/)\*, [Jie Liu](https://jie0530.github.io/)\*, [Zhenyu Zhao](https://zhenyuzhao.com/)\*, [Junjie Ye](https://junjieye.com/), [Hongyi Jing](https://hongyijing.me/), Junkai Xia, [Xiawei Liu](https://www.xiaweiliu.com/), [Michael Leong](https://leongmichael.github.io/), [Liang Heng](https://liangheng121.github.io/), Di Huang, [Yue Wang](https://yuewang.xyz/)†

> 


## 📢 News & Updates
+ [x] [2026-11-11] SIMPLE now runs on `IsaacSim 6.2` (with `IsaacLab 3.0`) + `MuJoCo 3.12` on Python 3.12, on the [`isaacsim-6.2-mujoco-3.12`](https://github.com/physical-superintelligence-lab/SIMPLE/tree/isaacsim-6.2-mujoco-3.12) branch (docker image `simple:20261027`). The `main` branch stays on `IsaacSim 4.5` + `MuJoCo 3.3`.
+ [x] [2026-11-11] We release Psi0 checkpoints finetuned on the decoupled-WBC Teleop and MP tasks: [Psi0-SIMPLE-decoupled-wbc-Teleop](https://huggingface.co/USC-PSI-Lab/Psi0-SIMPLE-decoupled-wbc-Teleop) (12 tasks) and [Psi0-SIMPLE-decoupled-wbc-MP](https://huggingface.co/USC-PSI-Lab/Psi0-SIMPLE-decoupled-wbc-MP) (5 tasks).
+ [x] [2026-11-09] Stability and evaluation fixes for decoupled-WBC tasks and Psi0 eval tooling (PR #33). See [Release Notes](#-release-notes).
+ [x] [2026-11-02] We release the [Psi0 multi-task checkpoint](https://huggingface.co/collections/USC-PSI-Lab/simple) for the dataset `SIMPLE-decoupled-wbc`.
+ [x] [2026-11-01] We released the 17 decoupled wbc tasks as two separate datasets, [SIMPLE-decoupled-wbc-Teleop](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-decoupled-wbc-Teleop) and [SIMPLE-decoupled-wbc-MP](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-decoupled-wbc-MP), and updated the eval episodes [SIMPLE-decouple-wbc-eval](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE-decoupled-wbc-eval).
+ [x] [2026-08-30] [Integrate SONIC whole-body controller](#quick-start-for-sonic-wholebody-vla).
+ [x] [2026-07-14] We released support for World Action Models: [Cosmos3](https://github.com/songlin/cosmos-framework/blob/main/docs/action_policy_simple_posttrain.md) and [DreamZero](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/dreamzero/README.md). 


## 📝 Release Notes
**2026-11-09** (PR #33)
- Fixed AMO/GR00T decoupled controller to initialize more stably.
- Fixed MP tasks' contact parameters for stable grasping and no slipping.
- Fixed Docker Compose to enable Isaac Sim caching between launches.
- Added support for fast VLA evaluation by skipping unneeded renderings.
- Fixed some episodes' eval spawning positions to avoid self-collision.
- Fixed some tasks' success checker to avoid false positives.

## Table of Contents
- [What is SIMPLE?](#what～is～SIMPLE)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [[Option 1] UV setup (Quickest)](#option-1-uv-setup-quickest)
  - [[Option 2] Nix setup](#option-2-nix-setup)
  - [[Option 3] Docker setup](#option-3-docker-setup)
- [Quick start for SONIC wholebody VLA](#quick-start-for-sonic-wholebody-vla)
- [Data Generation & Pipeline](#-data-generation--pipeline)
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

SIMPLE is built on top of `IsaacSim 4.5` and `MuJoCo 3.3`, and requires an RTX-class NVIDIA GPU on Ubuntu 22.04.

> 📖 **See the full hardware and software requirements at [psi-lab.ai/SIMPLE/docs](https://psi-lab.ai/SIMPLE/docs/).**

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

### [Option 1] UV setup (Quickest)


Prerequisits:
```
sudo apt-get update
sudo apt-get install curl cmake python3-dev ffmpeg
sudo apt-get install gstreamer1.0-libav
sudo apt-get install git-lfs && git lfs install && git lfs pull
```

Install `uv` if not already done
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

Install all dependencies at once

```
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match

```

> **If the sync fails with** `Unable to uninstall lerobot==0.3.3. distutils-installed
> distributions do not include the metadata required to uninstall safely.` — the `lerobot`
> wheel ships a stray top-level `lerobot-<ver>.egg-info` file next to its `.dist-info`, and
> uv reads it as a second, legacy copy of the package. Delete it and re-run the sync:
>
> ```
> rm -f .venv/lib/python3.10/site-packages/lerobot-*.egg-info
> ```

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

> See [Installation Troubleshootings](docs/source/troubleshooting.md)


> The document are working in progress. Feel free to raise questions using github issue, we will try to complete the document construction as soon as possible.

### [Option 2] Nix setup

We recommend [nix](https://nixos.org/) on a fresh Linux host; if you already have the NVIDIA driver and CUDA installed, `uv` is the faster path.

> 📖 **See the full Nix setup and runtime guide at [psi-lab.ai/SIMPLE/docs/nix-setup](https://psi-lab.ai/SIMPLE/docs/nix-setup/).**

### [Option 3] Docker setup

We also support building and running SIMPLE in docker. Please refer to the documents for [docker setup](https://psi-lab.ai/SIMPLE/docs/docker.html).

---

## Quick start for SONIC wholebody VLA

Evaluate a trained Psi-0 policy on the whole-body carry-box task with both sides in
Docker. The two terminals below run **two different images from two different
repositories** — the server on Psi-0's image, the client on SIMPLE's — and both are
published on the GitHub Container Registry:

```bash
# Psi-0 server image
docker pull ghcr.io/physical-superintelligence-lab/psi0:latest
docker tag  ghcr.io/physical-superintelligence-lab/psi0:latest psi:train

# SIMPLE client image
docker pull ghcr.io/physical-superintelligence-lab/simple:latest
docker tag  ghcr.io/physical-superintelligence-lab/simple:latest simple:latest
```

The retags matter: each `docker-compose.yml` refers to its image by a bare local
tag (`psi:${PSI_TAG:-train}` and `simple:${DATE:-latest}`), so **neither `docker
compose run` below pulls from ghcr on its own** — without the retag the Psi-0
service cannot resolve its image and the SIMPLE service rebuilds from source
instead.

The remaining prerequisites — checkpoint, eval episodes — and a native `uv`
alternative are in
[Wholebody Loco-manipulation](docs/source/tutorials/wholebody_loco_manipulation.md).

**Terminal A — policy server** (`psi:train`), from the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0#docker-support) workspace:

```bash
export RUN=sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223

docker compose run --rm serve-psi0-sonic-http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step 40000 \
    --run-dir .runs/finetune/$RUN \
    --rtc \
    --action-exec-horizon 24
```

Wait for `Server listens on 0.0.0.0:8014`, then leave it running.

**Terminal B — evaluation client** (`simple:latest`), from the SIMPLE repository root:

```bash
GPUs=1 docker compose run --rm eval-sonic-wbc \
    simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
    --data-dir data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 \
    --host 127.0.0.1 \
    --port 8014 \
    --episode-start 0 \
    --num-episodes 5 \
    --dr-level 0 \
    --eval-dir data/eval/sonic-psi0
```

Both compose stacks are host-networked, so `127.0.0.1` reaches across them; `--port`
must match the server's. Keep `--data-dir` and `--eval-dir` under `data/` — only that
tree is bind-mounted. Per-episode videos land in
`data/eval/sonic-psi0/psi0/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0/episode_*/`.

---

## ⚙️ Data Generation & Pipeline [Decouple Wholebody]

SIMPLE provides a scalable pipeline to generate, process, and train policies using synthesized simulation data — covering data collection (teleoperation and automated motion planning), post-processing, and fine-tuning.

> 📖 **See the full documentation at [psi-lab.ai/SIMPLE/docs](https://psi-lab.ai/SIMPLE/docs/).**

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
export dr=dr-level-0

TASK_NAME=$task uv run eval-decoupled-wbc \
    simple/$task \
    $agent \
    train \
    --data-format lerobot \
    --data-dir data/SIMPLE-eval/$task/$dr \
    --host 127.0.0.1 \
    --port 21000 \
    --headless
```

#### Option B: Nix Environment

```bash
export task=G1WholebodyXMovePickTeleop-v0
export entry=eval_decoupled_wbc
export agent=psi0_decoupled_wbc
export dr=dr-level-0

env -u LD_LIBRARY_PATH nix --extra-experimental-features 'nix-command flakes' develop -c \
  python -m simple.cli.$entry \
  simple/$task \
  $agent \
  train \
  --data-format lerobot \
  --data-dir data/SIMPLE-eval/$task/$dr \
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
mpv data/evals_decoupled_wbc/psi0_decoupled_wbc/G1WholebodyXMovePickTeleop-v0/dr-level-0/episode_0/head_stereo_left_success.mp4

# Example: Play a successful motion planning evaluation video
mpv data/evals/psi0/G1WholebodyBendPickMP-v0/dr-level-0/episode_0/front_stereo_left_success.mp4

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

