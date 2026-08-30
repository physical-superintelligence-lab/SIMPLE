# Robot

`Robot` (`core/robot.py`) is a configuration container, not a controller. It
holds the asset paths and kinematics description that the simulators and the
whole-body controllers consume.

| Field | Purpose |
| :--- | :--- |
| `mjcf_path` | MuJoCo physics model |
| `usd_path` | Isaac Sim rendering model |
| `curobo_usd`, `robot_cfg` | CuRobo kinematics / planning config |
| `joint_names`, `joint_limits`, `init_joint_states` | Joint description and default pose |
| `eef_link_name`, `eef_prim_path`, `robot_eef_offset` | End-effector frames |
| `controller_cfg` | Controller selection (e.g. `pd_joint_pos`) |

`fk()`, `ik()` and `reset()` are provided; low-level actuation is left to the
simulator backend and the whole-body controller.

## Implementations

Concrete robots in `simple/robots/` compose `Robot` with mixins and protocols —
`CuRoboMixin` (batch planning, grasping, kinematics), `Humanoid`, `DualArm`,
`HasDexterousHand`, `HeadCamMountable`, `WristCamMountable`:

| Class | Notes |
| :--- | :--- |
| `G1Wholebody` | Unitree G1, whole-body loco-manipulation |
| `G1Sonic` | G1 driven by the external SONIC controller |
| `G1`, `G1Inspire`, `G1InspireWholebody` | G1 variants with Inspire dexterous hands |
| `Vega1` | Humanoid with dexterous hands, head and wrist cameras |
| `Aloha` | Bimanual tabletop arms |
| `FrankaResearch3` | Single-arm tabletop (Franka FR3) |

## Whole-body control

High-level policies emit upper-body kinematic targets plus base navigation
commands; a lower-body tracking controller runs at higher frequency to keep
balance. SIMPLE ships three, behind a common
`setup_controller()` / `apply_action()` interface: **AMO**, **SONIC**, and
**decoupled SONIC** — see [GR00T Decoupled Wholebody](../decoupled-wbc/index.md)
and [SONIC Wholebody](../sonic-wbc/index.md).
