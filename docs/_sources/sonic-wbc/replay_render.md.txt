# Replay & Render

SONIC replay drives an **external controller process**, so rendering Isaac Sim
inside the live control loop can starve ZMQ/DDS communication and trip
controller heartbeat timeouts. Replay and rendering are therefore split:

```text
Phase 1: run SONIC + MuJoCo replay, save the qpos/qvel state trace
Phase 2: after replay, load the trace into Isaac Sim, sync each state, render
```

## Run

```bash
HEADLESS=1 bash scripts/run_replay_wbc.sh \
  simple/G1WholebodyXMoveBendCarryBoxSonic-v0 \
  --sim-mode mujoco_isaac \
  --offline-isaac-render \
  --no-webrtc \
  --data-dir data/teleop_wbc/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/level-0 \
  --num-episodes 1 \
  --replay-dir logs/sonic_replay_isaac_render
```

`--data-dir` defaults to `data/teleop_wbc/<env_id>/level-0`, so it can be
omitted when replaying the dataset the launcher just recorded.

Outputs:

```text
logs/sonic_replay_isaac_render/episode_0_mujoco_state_trace.npz   # qpos / qvel
logs/sonic_replay_isaac_render/episode_0/*.mp4                    # Isaac Sim frames
```

A `_failed` suffix on an MP4 is the task success label — not a rendering
failure.

## How it works

During live replay the controller runs against MuJoCo only, and each control
frame stores `mjData.qpos` / `mjData.qvel`. Afterwards a separate headless
`mujoco_isaac` environment is created and reset from the same episode
configuration, then every saved state is pushed in:

```python
for state in states:
    sonic_render_env.mjData.qpos[:] = state["qpos"]
    sonic_render_env.mjData.qvel[:] = state["qvel"]
    mujoco.mj_forward(sonic_render_env.mjModel, sonic_render_env.mjData)
    sonic_render_env.isaac.step(sonic_render_env.mujoco)
    frame_dict = sonic_render_env._render_frame()
```

Each camera stream in `frame_dict` is written to its own MP4 with `VideoWriter`.

Because the trace is a plain `.npz`, rendering can be rerun from it without
launching the controller again — useful for debugging rendering or validating
Isaac Sim synchronization.

## Verify output

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,r_frame_rate,duration,width,height \
  -of default=noprint_wrappers=1 \
  logs/sonic_replay_isaac_render/episode_0/head_stereo_left_failed.mp4
```

## Troubleshooting

* **Isaac Sim extension warnings** — deprecated extensions, missing material USD
  files and optional robot-motion extensions log warnings even on success. Treat
  them as fatal only if Python raises or no video is written.
* **No video produced** — check that `--replay-dir` is set, that
  `--offline-isaac-render` is paired with an Isaac sim mode
  (`--sim-mode mujoco_isaac`), that the replay produced a non-empty state trace,
  that Isaac Sim starts headless, and that the task layout has cameras.
