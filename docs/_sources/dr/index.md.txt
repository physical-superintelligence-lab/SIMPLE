# Domain Randomization

Domain randomization (DR) closes the visual sim-to-real gap. Because trajectories
are replayed offline in Isaac Sim, randomization is applied at render time: object
instances, initial poses, table and scene textures, lighting, camera viewpoints
and language instructions all vary between episodes. Material shaders are sampled
from NVIDIA vMaterials.

## Randomizers

`DRManager` (`dr/manager.py`) holds a named registry of `Randomizer` instances
and dispatches them on every `Task.reset()`. Each task declares its own
`dr_cfgs`:

| Randomizer | Config | Varies |
| :--- | :--- | :--- |
| `TargetDR` | `TargetDRCfg` | Target / container object instance |
| `DistractorDR` | `DistractorDRCfg` | Number and identity of distractor objects |
| `SpatialDR` | `SpatialDRCfg` | Object and robot initial poses, stable-pose index |
| `MaterialDR` | `MaterialDRCfg` | vMaterials surface shaders |
| `LightingDR` | `LightingDRCfg` | Light positions and intensities |
| `TabletopSceneDR` | `TabletopSceneDRCfg` | Room / scene choice, table geometry |
| `CameraDR` | `CameraDRCfg` | Camera viewpoint |
| `LanguageDR` | `LanguageDRCfg` | Instruction phrasing |
| `ArticulatedObjectDr` | `ArticulatedObjectDrCfg` | Articulated-object joint state |

Example, from a whole-body pick task:

```python
dr_cfgs: dict[str, RandomizerCfg] = dict(
    language    = LanguageDRCfg(instructions=["move forward and pick up the apple."]),
    target      = TargetDRCfg(asset_id="graspnet1b:12"),
    distractors = DistractorDRCfg(res_id="graspnet1b", number_of_distractors=3,
                                  allow_duplicates=False, exclude=["12", "46"]),
    spatial     = SpatialDRCfg(spatial_mode="random",
                               robot_region=Box(low=[-1.4, 0.0, 0.0], high=[-1.5, 0.0, 0.0]),
                               target_region=Box(low=[-0.78, -0.06], high=[-0.85, 0.06])),
    scene       = TabletopSceneDRCfg(scene_mode="random"),
)
```

## DR levels

`--dr-level` selects a difficulty level, applied by
`TabletopGraspDRManager.set_level()`. **Level 0 is the most randomized**; each
higher level pins one more factor down:

| Level | Effect |
| :--- | :--- |
| **0** | Everything the task's `dr_cfgs` declare stays random (scene, lighting, materials, distractors, spatial) |
| **1** | Lighting, materials and scene fixed (`scene0`) |
| **2** | Also removes distractor objects (`number_of_distractors = 0`) |
| **3** | Also fixes spatial poses to the first stable pose |

```{note}
The paper describes levels the other way round — progressively *adding*
distractors, then visual randomization, then spatial randomization. The code is
the authority for `--dr-level`: level 0 is fully randomized and higher levels
remove variation.
```

## Generating evaluation datasets

An evaluation dataset is a minimal LeRobot dataset: every episode contains one
rendered initial frame and the corresponding `environment_config`. It describes
the layouts used to start evaluation; it is not a demonstration trajectory.

Run the generators from the repository root with the project virtual
environment. The environment variables below match the `cli/dr` and
`cli/dr_decoupled_wbc` entries in `.vscode/launch.json`; adjust the GPU and
display for the local machine. For unattended generation, use `--headless`.

```bash
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:1
```

### Motion-planning tasks

Use `src/simple/cli/dr.py` for an MP environment. With no
`--env-config-dir`, every reset samples a new base layout. To derive a new DR
level from existing layouts, point `--env-config-dir` either at a LeRobot
dataset directory containing `meta/episodes.jsonl`, or directly at that JSONL
file. The environment ID and source configurations must belong to the same
task.

```bash
.venv/bin/python src/simple/cli/dr.py \
  simple/G1WholebodyBendPickMP-v0 \
  --render-hz=50 \
  --sim-mode=mujoco_isaac \
  --no-headless \
  --dr-level=2 \
  --env-config-dir=data/G1WholebodyBendPickMP-v0 \
  --save-dir=data/evals \
  --num-episodes=10
```

Remove `--env-config-dir=...` when there is no source dataset. If fewer source
configurations than `--num-episodes` are available, the generator cycles through
them. Output is written to
`<save-dir>/<env-id>/level-<dr-level>/`, for example
`data/evals/simple/G1WholebodyBendPickMP-v0/level-2/`.

### Teleoperation and decoupled-WBC tasks

Use `src/simple/cli/dr_decoupled_wbc.py` for Teleop/SONIC environments. Its
`--data-dir` is required in practice: it must be an existing LeRobot dataset
with episode Parquet files and an `environment_config` for every selected
episode.

```bash
.venv/bin/python src/simple/cli/dr_decoupled_wbc.py \
  simple/G1WholebodyPushOfficeChairTeleop-v0 \
  --data-dir=data/G1WholebodyPushOfficeChairTeleop-v0 \
  --dr-level=2 \
  --num-episodes=10 \
  --headless \
  --save-dir=data/evals
```

Output is written to `<save-dir>/<env-id>/dr-level-<dr-level>/`, for example
`data/evals/simple/G1WholebodyPushOfficeChairTeleop-v0/dr-level-2/`.
Repeat the command with each required
`--dr-level` to build multiple evaluation splits.



On replay, `DRManager.load_state_dict(state_dict, dr_level)` controls how much of
a recorded layout is overridden: level 0 re-randomizes distractors and table
material, level 1 also lighting and materials, level 2 also spatial poses; passing
`None` restores the entire recorded layout.
