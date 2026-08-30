# G1 Decoupled

Whole-body G1 tasks collected by **VR teleoperation** through the GR00T
decoupled whole-body control stack. They run on `SonicLocoManipEnv` with the
`g1_sonic` robot: the operator's upper-body pose is retargeted by IK while the
whole-body controller keeps balance and handles locomotion.

See [GR00T Decoupled Wholebody](../decoupled-wbc/index.md) for the
teleoperation, replay and evaluation workflow.

```bash
python -m simple.cli.teleop_decoupled_wbc simple/G1WholebodyOpenTrashCanTeleop-v0 \
  --sim-mode=mujoco --record --no-headless
```

## Environments (12)

Alongside pick, place and handover, this set covers the **articulated-object**
and **non-prehensile** skills that motion planning cannot synthesize — opening a
trash can, oven or faucet, closing a door, and pushing an office chair.

* `simple/G1WholebodyBendHandoverTeleop-v0`
* `simple/G1WholebodyBendPickAndPlaceTeleop-v0`
* `simple/G1WholebodyBendPickTeleop-v0`
* `simple/G1WholebodyCloseDoorTeleop-v0`
* `simple/G1WholebodyHandoverTeleop-v0`
* `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
* `simple/G1WholebodyOpenFaucetTeleop-v0`
* `simple/G1WholebodyOpenOvenTeleop-v0`
* `simple/G1WholebodyOpenTrashCanTeleop-v0`
* `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
* `simple/G1WholebodyPushOfficeChairTeleop-v0`
* `simple/G1WholebodyXMovePickTeleop-v0`
