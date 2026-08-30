# Evaluation

Evaluating a policy on decoupled whole-body control tasks uses a **client-server
architecture**: the server hosts model inference (in the
[Psi-0](https://github.com/physical-superintelligence-lab/Psi0) repository),
while the SIMPLE client runs the simulation environment.

Teleop tasks (suffix `*Teleop-v0`) evaluate through decoupled WBC:

```bash
export entry=eval_decoupled_wbc
export agent=psi0_decoupled_wbc
```

Motion planning tasks (suffix `*MP-v0`) use the standard `eval` / `psi0` path
instead — see [Motion Planning](../data_gen_mp.md).

## Server side: model inference

Run these steps inside your **Psi-0** workspace.

**1. Configure environment variables**

```bash
cp .env.sample .env
# Edit .env to include your HF_TOKEN, WANDB variables, and PSI_HOME path
source .env
echo $PSI_HOME # Verify the path is correctly set
```

**2. Download pre-trained weights**

```bash
hf download USC-PSI-Lab/psi-model \
  --include="psi0/simple-checkpoints/*" \
  --local-dir=$PSI_HOME/.runs \
  --repo-type=model
```

**3. Start the inference server**

```bash
export RUN_DIR=xxxx
export CKPT_STEP=40000

# Listens on port 22085 by default
bash scripts/deploy/serve_psi0_simple.sh $RUN_DIR $CKPT_STEP
```

> ⚠️ Keep this terminal open — the server must stay active for the whole
> evaluation.

## Client side: simulation

Open a new terminal to launch the environment.

### Option A: UV environment

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

### Option B: Nix environment

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

## Results and videos

Task success rate statistics are printed on completion and preserved in a log:

```bash
cat data/evals_decoupled_wbc/eval_stats.txt
```

Per-episode videos are rendered automatically, named
`episode_id/cam_name_{success_flag}.mp4`:

```bash
mpv data/evals_decoupled_wbc/psi0_decoupled_wbc/G1WholebodyXMovePickTeleop-v0/level-0/episode_0/head_stereo_left_success.mp4
```
