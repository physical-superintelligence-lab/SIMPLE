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

Datasets are written per level, e.g. `data/datagen/simple/<env_id>/level-0/`.

On replay, `DRManager.load_state_dict(state_dict, dr_level)` controls how much of
a recorded layout is overridden: level 0 re-randomizes distractors and table
material, level 1 also lighting and materials, level 2 also spatial poses; passing
`None` restores the entire recorded layout.
