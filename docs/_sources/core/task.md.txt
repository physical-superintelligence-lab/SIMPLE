# Task

`Task` (`core/task.py`) is the abstract base bundling everything task-specific:
the `DRManager`, the robot descriptor, sensor configs, and the Gym action and
observation spaces.

## Class attributes

| Attribute | Purpose |
| :--- | :--- |
| `uid`, `label`, `description` | Identity and human-readable name |
| `metadata` | `physics_dt` (0.002), `render_hz`, `split` |
| `robot_cfg` | Robot uid and controller selection |
| `sensor_cfgs` | Camera definitions (`eye_on_base`, `eye_in_hand`, head) |
| `dr_cfgs` | Per-randomizer configuration, see [Domain Randomization](../dr/index.md) |

## Key methods

* **`reset(seed, options)`** — sequences through every registered randomizer
  (scene, target, container, distractor, spatial, lighting, camera, material)
  and builds a `Layout` that both simulators ingest to rebuild the scene.
* **`check_success()` / `compute_reward()`** — task completion and reward.
* **`instruction`** — the natural-language instruction for this episode,
  sampled by `LanguageDR`.
* **`decompose()`** — splits the objective into `SubtaskSpec` atoms, used by the
  motion-planning data generator.
* **`state_dict()` / `load_state_dict()`** — serialize an episode's layout so it
  can be reproduced exactly during replay or evaluation.

Concrete tasks live in `simple/tasks/` and are named
`<robot>_<skill>_<pipeline>.py`, where the pipeline suffix is `_mp` (motion
planning) or `_teleop` (VR teleoperation).
