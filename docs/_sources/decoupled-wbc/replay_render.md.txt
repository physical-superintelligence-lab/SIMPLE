# Replay & Render

Stage 2 of the decoupled whole-body control data pipeline: photorealistic replay
and Isaac Sim rendering.

Once raw trajectories have been captured with
[Teleoperation](teleoperation.md), pass them into the `replay_decoupled_wbc`
suite. With `--sim-mode=mujoco_isaac` the recorded actions are replayed in
MuJoCo while **Isaac Sim** is driven simultaneously as a synchronized rendering
engine. This turns the raw stream into a standard dataset in **LeRobot** format.

## Example usage

```bash
# Ensure $TASK_NAME matches the task used during teleoperation
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

The `replay-decoupled-wbc` entry point runs the same command.

> 💡 **Tip:** If the replay success rate is low, try lowering
> `--success-criteria` first.

## Post-processing

To make the rendered data compatible with the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0)
training pipeline, run `postprocess_psi0_sonic.py`. It accepts wildcards (`*`)
so several replay sessions can be merged into one dataset.

```bash
python scripts/postprocess_psi0_sonic.py \
  --sim-root="data/replay_decoupled_wbc_output*/simple/G1WholebodyPushOfficeChairTeleop-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyPushOfficeChairTeleop-v0 \
  --skip=0 \
  --total_episodes=100
```

**Key arguments**

* `--sim-root` — input directory holding the generated dataset. Quote the value
  when using wildcards so the shell does not expand them early.
* `--out-dir` — output directory for the Psi-0 compatible dataset.
* `--skip` — number of initial frames to skip (useful to bypass static setup
  frames).
* `--total_episodes` — cap on the number of valid episodes to merge.
