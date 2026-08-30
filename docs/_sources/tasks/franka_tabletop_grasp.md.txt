# Franka Tabletop Grasp

`simple/FrankaTabletopGraspMP-v0` — a Franka FR3 must grasp and lift a target
object from a tabletop. This is the smallest end-to-end example in SIMPLE and the
one used in [Motion Planning](../data_gen_mp.md).

| | |
| :--- | :--- |
| Task class | `FrankaTabletopGraspTaskMP` (`tasks/franka_tabletop_grasp_mp.py`) |
| Environment | `TabletopGraspEnv` |
| Robot | `franka_fr3`, controller `pd_joint_pos` |
| Target | `graspnet1b:5`, randomized by `TargetDR` |
| Cameras | Realsense D415 (`eye_on_base`), Logitech C930e (`eye_in_hand`) |
| Success | Target lifted 2 cm above its initial height (`_LIFT_HEIGHT = 0.02`) |

`compute_reward()` returns the lift progress clipped to `[0, 1]`;
`check_success()` fires at `1.0`.

```bash
uv run datagen simple/FrankaTabletopGraspMP-v0 --headless --num-episodes 10
```

A related variant, `simple/FrankaTabletopPickNPlaceMP-v0`, adds a container the
object must be placed into.
