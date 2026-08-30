# Architecture

## Dual-simulator design

SIMPLE strictly decouples physics from rendering. **MuJoCo** handles all
rigid-body dynamics, contact resolution and robot control; the resulting states
are synchronized to **Isaac Sim**, which performs photorealistic ray-traced
rendering. The policy consumes Isaac Sim images and returns actions to MuJoCo.

```{image} ../../../assets/img/system_diagram.png
:alt: MuJoCo simulates physics, Isaac Sim renders images for the policy
:width: 380px
```

This split buys MuJoCo's contact fidelity and locomotion stability together with
Isaac Sim's visual diversity. The whole loop is wrapped in a standard Gym
interface, and high-level policies emit upper-body kinematic targets plus base
navigation commands that a higher-rate lower-body tracking controller executes.

## Class diagram

```{image} ../../../assets/img/class_diagram.png
:alt: Class diagram of the SIMPLE framework
:width: 100%
```

* **`BaseDualSim`** (`envs/base_dual_env.py`) — the `gymnasium.Env`; owns two
  `Simulator` instances and the active `Task`. Isaac Sim is created lazily, only
  when `"isaac"` is in `sim_mode`. `LocoManipulationEnv`, `SonicLocoManipEnv` and
  `TabletopGraspEnv` subclass it.
* **`Simulator`** (`core/simulator.py`) — `update_layout()`, `set_states()`,
  `get_states()`, `step()`, `render()`. Implemented by `MujocoSimulator`
  (physics) and `IsaacSimSimulator` (rendering).
* **`Task`** — composes a `DRManager` and a `Robot`, and drives scene
  randomization at each reset. See [Task](task.md).
* **`Robot`** — asset paths and kinematics config only. See [Robot](robot.md).
* **`DRManager`** — registry of `MaterialDR`, `SpatialDR`, `LightingDR`,
  `DistractorDR` and friends. See [Domain Randomization](../dr/index.md).
* **Agents** — teleoperation, motion planning and remote inference share one
  `get_action()` interface over a buffered action queue.
* **Whole-body controllers** — `AmoWBC`, `SonicWBC` and `SonicDecoupledWBC`
  share `setup_controller()` / `apply_action()` and turn high-level commands into
  low-level joint targets.
* **Policy servers** — `Psi0Server`, `Pi05Server`, `GR00TServer` run as separate
  processes behind `health()` / `act()`, reached over HTTP or WebSocket, so
  inference can sit on another GPU or machine.

## One `env.step()`

```{image} ../../../assets/img/sequence_diagram.png
:alt: Sequence diagram of a single env.step() call
:width: 100%
```

`Task.reset()` applies domain randomization once per episode. Then each
`env.step(action)`:

1. MuJoCo applies the action; the robot routes it through the whole-body
   controller and RL tracker, which return torque / qpos targets.
2. MuJoCo integrates physics at 500 Hz and reports `info` and termination.
3. Isaac Sim reads the states (`get_states()` → `set_states()`) and renders at
   the task's `render_hz` (50 Hz for whole-body pipelines).
4. The observation is returned to the agent.

MuJoCo sub-steps ahead of the slower rendering pass, so every inference step
receives a physically consistent, photorealistic observation.
