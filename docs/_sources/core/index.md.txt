# Core Concepts

SIMPLE couples **MuJoCo** (contact-rich rigid-body physics and robot control)
with **Isaac Sim** (photorealistic ray-traced rendering). Physics and rendering
are strictly decoupled: MuJoCo integrates at 500 Hz (`physics_dt = 0.002`),
Isaac Sim consumes the synchronized states and renders RGB frames at the task's
`render_hz` (50 Hz for the whole-body pipelines).

**`BaseDualSim`** (`envs/base_dual_env.py`) is the `gymnasium.Env` that owns
both simulators and the active `Task`. Isaac Sim is initialized lazily — the
`SimulationApp` is created only when `"isaac"` appears in `sim_mode` — so
MuJoCo-only workflows pay no rendering cost. Concrete environments
(`TabletopGraspEnv`, `LocoManipulationEnv`, `SonicLocoManipEnv`) subclass it.

```python
import gymnasium as gym
import simple.envs as _  # register all envs

env = gym.make("simple/G1WholebodyBendPickMP-v0", sim_mode="mujoco_isaac")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

```{toctree}
:maxdepth: 1

architecture
task
robot
```
