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
UV_HTTP_TIMEOUT=3000 bash scripts/setup_python_env.sh
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

## [Option 2] robo-nix setup

SIMPLE keeps Python dependencies in `uv`, but uses [robo-nix](https://github.com/ausbxuse/robo-nix) for the native robotics runtime: CUDA, graphics, MuJoCo, Isaac Sim runtime libraries, media libraries, and native build tools.

Install `robo`:

```bash
curl -fsSL https://raw.githubusercontent.com/ausbxuse/robo-nix/master/scripts/install.sh | sh
```

Prepare the runtime, enter it, and install Python dependencies:

```bash
robo shell
bash scripts/setup_python_env.sh
bash scripts/install_curobo.sh
```

Verify the runtime:

```bash
python -c "import simple; print(simple.__version__)"
bash scripts/tests/check_datagen.sh
bash scripts/tests/check_eval.sh
```

Isaac Sim 4.5 is not stable with every NVIDIA driver branch. NVIDIA driver 595
has been observed to segfault during Isaac `SimulationApp` startup on an RTX
5090 Laptop GPU, while drivers 575 and 580 have both been tested and worked. If
datagen crashes right after Isaac/RTX startup, try a 575- or 580-series driver
and rerun `robo up` after removing `.robo-nix/host-graphics`,
`.robo-nix/shell-env`, and `.robo-nix/shell-env.key`.

See [robo-nix setup](docs/source/robo-nix.md) for notes on the project runtime contract.

## [Option 3] Docker setup

We also support building and running SIMPLE in docker. Please refer to the documents for [docker setup](docs/source/tutorials/docker.md).  

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
