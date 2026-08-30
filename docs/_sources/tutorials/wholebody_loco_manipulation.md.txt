# Wholebody Loco-manipulation

```{admonition} End-to-end quick start
:class: tip

**Download a trained Psi-0 checkpoint, serve it, and evaluate it on the
whole-body carry-box task — producing evaluation videos.**

Two terminals: **A** serves the policy, **B** runs the simulation client.
Everything below runs headless — no display required — and produces an MP4
per camera plus a success rate.
```

| | Terminal A — policy server | Terminal B — evaluation client |
|---|---|---|
| Repository | [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) | [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) |
| Needs a GPU | yes (loads the checkpoint) | yes (Isaac Sim + MuJoCo) |
| Setup | Docker **or** `uv`, [step 1](#step-1-set-up-the-psi-0-environment) | Docker **or** native `uv`, [step 2](#step-2-set-up-the-evaluation-client) |

Each side can run **either** from Docker (nothing to build, recommended for a
first run) **or** natively in a `uv` environment. The two choices are
independent — a containerised server happily serves a native client and vice
versa, because both compose stacks use `network_mode: host`.

## Step 1: Set up the Psi-0 environment

Skip if you already have a working $\Psi_0$.

```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git
cd Psi0
```

Copy `.env.sample` to `.env` and fill in at least `HF_TOKEN` (a read token, for
the checkpoint download) and `PSI_HOME` (where checkpoints and data live by
convention). Both options below read that file — the native server loads it on
startup, and the compose services declare it as `env_file`.

### Option A — Docker

Pull the official image from the GitHub Container Registry:

```bash
docker pull ghcr.io/physical-superintelligence-lab/psi0:latest
docker tag ghcr.io/physical-superintelligence-lab/psi0:latest psi:train
```

The retag is what lets `docker compose` find it — `docker-compose.yml` refers to
the image as `psi:${PSI_TAG:-train}`, and `PSI_TAG` overrides the tag only, not
the registry path. Pin an exact build with a dated tag such as `:260830`.

Requirements: Docker with the NVIDIA Container Toolkit; the services run with
`runtime: nvidia`. The image carries **dependencies only** — `src/`, `.runs/`
and `$PSI_HOME` are bind-mounted, so checkpoints you download on the host are
visible inside the container and code edits need only a restart, never a
rebuild.

To build it yourself instead:

```bash
docker compose build psi
```

`psi` is the only service that declares `build:`, so a bare `docker compose
build` does the same thing. Every other service — including the server in
[step 5](#step-5-serve-the-policy) — consumes the image and never rebuilds it.

Verify:

```bash
docker compose run --rm serve-psi0-sonic-http --help
```

### Option B — Native `uv`

```bash
# uv manages the Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .venv-psi --python 3.11
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve \
  --group viz \
  --group psi \
  --index-strategy unsafe-best-match \
  --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

Verify:

```bash
python -c "import psi; print(psi.__version__)"
```

See the [Psi-0 README](https://github.com/physical-superintelligence-lab/Psi0#installation) for the full installation notes.

## Step 2: Set up the evaluation client

### Option A — Docker (recommended)

The published image already contains Isaac Sim, the SIMPLE package, and a
**prebuilt SONIC whole-body controller**, so there is nothing to compile:

```bash
docker pull ghcr.io/physical-superintelligence-lab/simple:latest
docker tag ghcr.io/physical-superintelligence-lab/simple:latest simple:latest
```

The `docker tag` is what lets `docker compose` find it — `docker-compose.yml`
refers to the image as `simple:${DATE:-latest}`.

Requirements: Docker with the NVIDIA Container Toolkit (`runtime: nvidia`), and
an NVIDIA driver **at least as new as** the image's CUDA toolkit release. The
image is large; allow room for it.

```{note}
The image is published from the SIMPLE GitHub project's package registry. If
`docker pull` reports the package is not found, it has not been published for
your platform yet — use Option B, or build locally with
`docker compose build isaac-sim`.
```

### Option B — Native `uv`, no Docker

Runs the same client directly on the host. This one **does** need the SONIC
whole-body stack compiled locally:

```bash
git clone git@github.com:physical-superintelligence-lab/SIMPLE.git
cd SIMPLE

uv sync                                # creates ./.venv
bash scripts/setup_teleop_wbc.sh       # builds the SONIC controller into ~/tools
```

See [Setup](../sonic-wbc/setup.md) for the whole-body stack in detail. Verify:

```bash
./.venv/bin/python -c "import simple; print(simple.__version__)"
ls third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

The evaluation refuses to start if that binary is missing — unlike teleop and
replay, it does not build it on demand.

## Step 3: Download the checkpoint

From your Psi-0 workspace, with `PSI_HOME` set:

```bash
export RUN=sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223

hf download USC-PSI-Lab/psi-model \
  --include="psi0/simple-checkpoints/$RUN/*" \
  --local-dir=.runs/finetune/ \
  --repo-type=model
```

The run directory ships a single checkpoint, `checkpoints/ckpt_40000/`, and
lands at `$PSI_HOME/.runs/psi0/simple-checkpoints/$RUN` (~6 GB).

## Step 4: Download the evaluation dataset

The client loads domain randomized **SIMPLE eval dataset**: it reads each episode's
`environment_config` from `meta/episodes.jsonl` to rebuild the scene, then lets
the policy drive.

```{warning}
A *training-converted* LeRobot copy of the same task will **not** work — it has
the parquet files but no `environment_config`, and the evaluation stops with:

    ValueError: no environment_config in <dir>/meta/episodes.jsonl ...;
    point --data-dir at the SIMPLE teleop dataset (…/<env-id>/level-N),
    not a training-converted copy
```

From the SIMPLE repository root:

```bash
hf download USC-PSI-Lab/psi-data \
  --repo-type=dataset \
  --include="simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip" \
  --local-dir=/tmp/psi-eval-data

unzip /tmp/psi-eval-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip \
  -d data/simple/
```

That is 15 MB and holds 5 episodes, unpacking to
`data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0/` — the path
[step 6](#step-6-run-the-evaluation) passes to `--data-dir`. Raise
`--num-episodes` up to 5 to run more of them.

`--data-dir` must point at a directory laid out as:

```
<env-id>/<level-dir>/
├── data/chunk-000/episode_*.parquet
├── meta/episodes.jsonl          # must contain environment_config
└── videos/chunk-000/
```

When using Docker, keep it under `data/` — that tree is bind-mounted into the
container, so anything outside it is invisible to the client.

## Step 5: Serve the policy

Terminal **A**, from the Psi-0 workspace. Both options take the same flags and
bind `0.0.0.0:8014`.

### Option A — Docker

```bash
docker compose run --rm serve-psi0-sonic-http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step 40000 \
    --run-dir .runs/finetune/$RUN \
    --rtc \
    --action-exec-horizon 24
```

`serve_psi0_sonic_http` is the service's entrypoint, so these arguments replace
the compose defaults wholesale. `--run-dir` is relative to the container's
working directory, where the repo's `.runs/` is bind-mounted — the same path
works in both options.

Those defaults already match the command above, so the short form is

```bash
docker compose up serve-psi0-sonic-http
```

and single values are overridable from the shell or `.env` without repeating
the argument list: `POLICY`, `PORT`, `CKPT_STEP`, `RUN` (or `RUN_DIR`),
`RTC_FLAG` and `ACTION_EXEC_HORIZON`. `GPUS` picks the device, e.g.
`GPUS=1 docker compose up serve-psi0-sonic-http`.

The service is host-networked, so the client reaches it at `127.0.0.1:8014`
whether the client itself runs in Docker or natively.

### Option B — Native `uv`

```bash
serve_psi0_sonic_http \
    --policy psi0 \
    --port 8014 \
    --ckpt-step 40000 \
    --run-dir .runs/finetune/$RUN \
    --rtc \
    --action-exec-horizon 24
```

### Either way

Keep this terminal open — the server must stay up for the whole evaluation.
Startup takes a minute or two while the VLM backbone and action head load; wait
for `Server listens on 0.0.0.0:8014` before starting the client.

```{important}
`--rtc` is the flag that turns real-time chunking **on**. 
```

Useful flags: `--rtc-mode` (`auto`, the default, picks a train-time frozen
prefix if the checkpoint has one, otherwise test-time guidance),
`--action-exec-horizon` (must be smaller than the chunk size),
`--num-inference-steps` (default 8), `--device` (default `cuda:0`) and
`--port` (default 21074).

Sanity-check it from another shell:

```bash
curl -s localhost:8014/info | head -c 400
```

## Step 6: Run the evaluation

Terminal **B**. Both options take the same arguments: the environment id, the
policy name, then flags forwarded to the evaluation CLI. `--port` must match
the server.

### Option A — Docker

```bash
docker compose run --rm eval-sonic-wbc \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 \
  --host 127.0.0.1 \
  --port 8014 \
  --episode-start 1 \
  --num-episodes 1 \
  --dr-level 0 \
  --eval-dir data/eval/sonic-psi0
```

The service uses `network_mode: host`, so `127.0.0.1` reaches the server running
on the host, and it runs headless by default.

### Option B — Native `uv`

```bash
HEADLESS=1 bash scripts/run_eval_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 psi0 \
  --data-dir data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 \
  --host 127.0.0.1 \
  --port 8014 \
  --episode-start 1 \
  --num-episodes 1 \
  --dr-level 0 \
  --eval-dir data/eval/sonic-psi0
```

This is the same script the Docker service runs as its entrypoint — the image
just supplies the environment. Either way it starts the lockstep SONIC
controller itself and tears it down on exit.

On a workstation with a live X display you can watch the run instead of only
the recordings: `HEADLESS=0` opens the desktop viewer (the compose file already
wires up `DISPLAY` and the X11 socket). Videos are written either way, and the
viewer is not needed on a headless host.

```{note}
The controller binds host ports **5556**, **5557** and **13579**, so only one
evaluation can run at a time. The script refuses to start rather than killing a
process it does not own.
```

## Step 7: Watch the results

Everything lands under `--eval-dir`:

```
data/eval/sonic-psi0/
├── episode_0/
│   ├── head_stereo_left.mp4
│   └── head_stereo_right.mp4
├── eval_stats.txt
└── summary.json
```

```bash
mpv data/eval/sonic-psi0/episode_0/head_stereo_left.mp4
cat data/eval/sonic-psi0/eval_stats.txt
```

```text
episode_0: True (task_success)
success rate: 1.000000
```

`summary.json` carries the full record — per-episode `task_success`,
`termination_reason`, `executed_steps` against `budget_steps`, `inference_count`
— plus the aggregate `success_rate`.

Output directories are numbered from 0 in selection order, not by dataset index:
with `--episode-start 1` the first result is still `episode_0`.

```{note}
Under Docker the files are written by root, since the container runs as root.
Use `sudo` to remove them, or delete them from inside the container.
```