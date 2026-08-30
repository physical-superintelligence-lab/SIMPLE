# G1 SONIC

Whole-body G1 tasks driven by the **SONIC** controller running as an external
process. They share `SonicLocoManipEnv` and the `g1_sonic` robot with the
[decoupled tasks](g1_decoupled.md), but the teleoperation, replay and evaluation
workflow differs — see [SONIC Wholebody](../sonic-wbc/index.md).

```bash
bash scripts/run_teleop_wbc.sh simple/G1WholebodyXMoveBendCarryBoxSonic-v0
```

## Environments (1)

* `simple/G1WholebodyXMoveBendCarryBoxSonic-v0`
