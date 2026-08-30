# G1 MP (AMO)

Whole-body G1 tasks whose demonstrations are produced by **automated motion
planning**. They run on `LocoManipulationEnv` with the `g1_wholebody` robot,
whose lower body is driven by the **AMO** policy
(`robots/policy/AMO_Policy.py`, `amo_jit.pt`): CuRobo plans the dual-arm
kinematic trajectory while a scripted policy issues the base locomotion
commands.

Collect data with the [Motion Planning](../data_gen_mp.md) pipeline:

```bash
uv run datagen simple/G1WholebodyBendPickMP-v0 \
  --sim-mode=mujoco_isaac --render-hz=50 --no-headless --num-episodes=10
```

## Environments (5)

* `simple/G1WholebodyBendPickAndPlaceOnSofaMP-v0`
* `simple/G1WholebodyBendPickMP-v0`
* `simple/G1WholebodyPickAndBendPlaceMP-v0`
* `simple/G1WholebodyTabletopGraspMP-v0`
* `simple/G1WholebodyTabletopHandoverMP-v0`

The fixed-base Franka, Aloha, Vega, G1 and G1-Inspire tabletop environments also
use the motion-planning pipeline; see
[Franka Tabletop Grasp](franka_tabletop_grasp.md) and
[Aloha Tabletop Grasp](aloha_tabletop_grasp.md).
