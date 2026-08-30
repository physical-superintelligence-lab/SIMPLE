# Teleoperation

Stage 1 of the decoupled whole-body control data pipeline: human-in-the-loop
demonstration collection.

Teleoperation is run inside the lightweight **MuJoCo** engine only. This keeps
the control loop latency minimal and allows high-frequency physical interaction
while the demonstration is being tracked. Photorealistic rendering happens later,
in [Replay & Render](replay_render.md).

For headset, controller and motion-tracker configuration, see
[Teleop Setup](teleop_decoupled_setup.md) first.

## Environment variables

Adjust these to match your host topology before running:

```bash
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="0"
export DISPLAY=":1"
```

## Example usage

```bash
export TASK_NAME=G1WholebodyOpenTrashCanTeleop-v0

python -m simple.cli.teleop_decoupled_wbc \
  simple/$TASK_NAME \
  --target=graspnet1b:0 \
  --sim-mode=mujoco \
  --record \
  --no-headless \
  --success-criteria=2
```

The same command is installed as the `teleop-decoupled-wbc` entry point, so
`uv run teleop-decoupled-wbc simple/$TASK_NAME ...` is equivalent.

Recorded trajectories are written under
`data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0/`, which is the input
directory for the replay stage.

> 💡 To list all customizable options, run
> `python -m simple.cli.teleop_decoupled_wbc --help`.

## Supported decoupled whole-body teleop tasks

* `simple/G1WholebodyOpenTrashCanTeleop-v0`
* `simple/G1WholebodyBendPickTeleop-v0`
* `simple/G1WholebodyBendPickAndPlaceTeleop-v0`
* `simple/G1WholebodyBendHandoverTeleop-v0`
* `simple/G1WholebodyPushOfficeChairTeleop-v0`
* `simple/G1WholebodyOpenFaucetTeleop-v0`
* `simple/G1WholebodyOpenOvenTeleop-v0`
* `simple/G1WholebodyCloseDoorTeleop-v0`
* `simple/G1WholebodyXMovePickTeleop-v0`
* `simple/G1WholebodyXMoveBendCarryBoxSonic-v0`
* `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
* `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
* `simple/G1WholebodyHandoverTeleop-v0`
