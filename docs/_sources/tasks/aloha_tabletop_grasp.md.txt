# Aloha Tabletop Grasp

`simple/AlohaTabletopGraspMP-v0` — the bimanual Aloha arms must grasp and lift a
target object from a tabletop.

| | |
| :--- | :--- |
| Task class | `AlohaTabletopGraspTaskMP` (`tasks/aloha_tabletop_grasp_mp.py`) |
| Environment | `TabletopGraspEnv` |
| Robot | `aloha` (dual-arm, `CuRoboMixin`, wrist cameras) |
| Target | `graspnet1b:63`, randomized by `TargetDR` |
| Cameras | Realsense D415 (`eye_on_base`), Logitech C930e per wrist (`eye_in_hand`) |
| Success | Target lifted 2 cm above its initial height (`_LIFT_HEIGHT = 0.02`) |

```bash
uv run datagen simple/AlohaTabletopGraspMP-v0 --headless --num-episodes 10
```

Related Aloha environments: `simple/AlohaTabletopHandoverMP-v0` (pass the object
between arms) and `simple/AlohaTabletopFindNGraspMP-v0` (locate the target first).
