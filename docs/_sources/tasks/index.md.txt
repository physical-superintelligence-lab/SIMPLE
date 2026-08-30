# Built-In Tasks

SIMPLE ships **31 registered environments** built from over 1,500 Objaverse
objects, 75 GraspNet-1B objects and 50 HSSD indoor scenes. Tasks span rigid
pick-and-place, non-prehensile interaction and articulated-object manipulation.

List them at runtime:

```python
import gymnasium as gym
import simple.envs as _  # register all envs

[k for k in gym.registry if k.startswith("simple/")]
```

## Naming

```text
simple/<Robot><Skill><Variant?><Pipeline>-v0
```

The pipeline suffix says how demonstrations are produced — **`MP`** (automated
CuRobo motion planning, 18 envs) or **`Teleop`** / **`Sonic`** (VR
teleoperation, 13 envs) — and selects the data-generation and evaluation path.

## Environment families

| Entry point | Envs | Scope |
| :--- | ---: | :--- |
| `SonicLocoManipEnv` | 13 | Whole-body teleoperation via the external SONIC controller |
| `LocoManipulationEnv` | 11 | G1 whole-body and fixed-base G1 motion planning |
| `TabletopGraspEnv` | 7 | Fixed-base tabletop arms (Franka, Aloha, Vega) |

Whole-body task names compose a locomotion primitive with a manipulation skill
— `XMove`, `Locomotion` for the base; `Pick`, `PickAndPlace`, `Bend`,
`Handover` for the body — for example
`simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`.

```{toctree}
:maxdepth: 1

g1_mp_amo
g1_decoupled
g1_sonic
franka_tabletop_grasp
aloha_tabletop_grasp
```
