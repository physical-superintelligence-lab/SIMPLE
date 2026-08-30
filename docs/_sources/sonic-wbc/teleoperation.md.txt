# Teleoperation

Full-body SMPL pose tracking — squats, crouching, arm manipulation and gait —
streamed from the PICO headset and ankle trackers to the robot. Complete
[Setup](setup.md) first.

## Launch

From the repository root:

```bash
bash scripts/run_teleop_wbc.sh simple/G1WholebodyXMoveBendCarryBoxSonic-v0
```

The launcher starts the external SONIC controller and the simulation together;
`Ctrl+C` stops both.

**Options**

* **Recording is on by default** — the launcher passes `--record`, writing a
  LeRobot dataset under `data/teleop_wbc/<env_id>/level-0`. Pass `--no-record`
  to disable it, or `--save-dir` / `--num-episodes` to change the destination
  and episode count.
* **`HEADLESS=1`** — skip the desktop viewer and stream only to the headset.
* Engaging enters **PLANNER**, then auto-advances to **POSE**; `A+X` toggles
  between them at runtime.

## Stabilize and stand

***There is no crane used***. Each episode reset disables the elastic band
(`band.enable = False`) and the robot is held instead by an **in-process
decoupled WBC**: the lower-body RL policy balances while the upper body ramps to
the WBC default pose. This runs from spawn until `robot.stabilized` latches —
max `|qvel[0:6]| < 1e-3` after at least 100 steps (1 s at 200 Hz) — at which
point control hands over to the external SONIC controller and the terminal
prints `Robot stabilized — handing control to the SONIC controller`.

To engage:

1. **Stand in the calibration pose:** stiff upright stance, feet together, upper
   arms at your sides, forearms bent 90° forward (L-shape at the elbows), palms
   facing inward.
2. **Press `A+X`** (`A+B+X+Y` also works). This calibrates VR 3-point tracking
   against the current SMPL frame and engages the controller.

Engaging always enters **PLANNER** mode first — the controller only latches
`operator_state.start` from a PLANNER command, so entering POSE straight from
OFF would leave the policy in `WAIT_FOR_CONTROL` with the arms frozen. The robot
stands in PLANNER. With `--auto_pose` (set by the launcher) the manager then
advances to POSE on its own as soon as control is confirmed live via `g1_debug`,
or after a 6 s fallback if that telemetry is unavailable:

```text
[Manager] auto_pose: engaging PLANNER first to latch control, then advancing to POSE
[Manager] auto_pose: control confirmed via g1_debug -> advancing to POSE
```

## Controls

| Action | Input | Description |
| :--- | :--- | :--- |
| **Engage & calibrate** | `A+X` | Calibrate + engage; enters PLANNER, then auto-advances to **POSE**. |
| **Toggle POSE / PLANNER** | `A+X` | Full-body SMPL teleop ⟷ stick locomotion. |
| **VR 3-point mode** | Left joystick click | Head + wrists tracking, planner legs. |
| **Left / right hand** | Left / right index trigger | Open / close the dexterous hand. |
| **Reset episode** | Left grip + right grip | Force a simulation episode reset. |
| **Emergency stop** | `A+B+X+Y` (when active) | Disengage the policy, manager returns to `OFF`. |

## Modes

* **POSE** — streams all 24 SMPL joints from the headset and ankle trackers;
  squatting, bending and arm reach are mirrored in real time.
* **PLANNER** — joystick base movement, VR body tracking disabled.
* **VR_3PT** — head and wrists tracked, legs driven by the reference planner.

## Workflow

1. Start the stack with `run_teleop_wbc.sh`.
2. Wait for the in-process WBC to settle the robot (`[status]` line shows
   `free`, then the handover message).
3. Stand in the calibration pose and press `A+X`.
4. The manager latches control in PLANNER and advances to POSE.
5. Teleoperate.
6. Squeeze both grips to reset the episode when done.
