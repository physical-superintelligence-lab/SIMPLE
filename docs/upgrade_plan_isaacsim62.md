# Upgrade plan: environment upgrade (Isaac Sim 6.2 / MuJoCo 3.12 / Python 3.12)

Status: planned, not started. Run **after** `docs/migration_plan_spawn_retract.md` (spawn fix and
trailing patch for all g1_sonic teleop tasks) is done, as a pure environment upgrade: package and API
changes only, no task or behavior changes.

## Findings

- `origin/migrate` (worked on in `/hfm/songlin/simple-migrate`) branches from `e8a4e7e`, the base of
  this repo's current work, and has 55 commits on top. Package versions:

  | | current | migrate |
  |---|---|---|
  | Python | 3.10 | 3.12 |
  | Isaac Sim | 4.5.0 | 6.2.0 (+ Isaac Lab 3.0) |
  | MuJoCo | 3.3.6 | 3.12.0 (+ mujoco-warp 3.12) |
  | torch / torchvision | 2.7.0 / 0.22.0 | 2.11.0 / 0.26.0 |
  | numpy | 1.26.4 | >= 2 |

- The API changes sit in a few commits, mixed with behavior changes:
  - `556a460` upgrade python to 3.12: `pyproject.toml`, `uv.lock`, `.python-version`
  - `ed8a64f` fix mujoco api migration: `bodyid` -> `bodyid.item()` in 15 task files. **Also**
    rewrites the LocomotionPickBetweenTables success check (position only, contact check removed):
    a behavior change, must be left out (eval success checkers stay as they are)
  - `1326dd3` upgrade curobo (submodule bump)
  - `8b38f39` migration to isaacsim 6.2.0 and isaaclab 3.0 (17 files): engine API changes
    (`engines/isaacsim.py`, `engines/mujoco.py`, `envs/video_writer.py`, `envs/base_dual_env.py`,
    `envs/wrappers/video_recorder.py`, `robots/g1_sonic.py`, ...) tangled with feature work
    (`cli/replay_decoupled_wbc.py` refactor, `cli/_decoupled_wbc_recording.py`, agents)
  - `45456d0` isaacsim: drop the local pxr import that left UsdGeom unbound on scene reload
  - to decide during testing (bug fixes that can change physics or rendering): `ae09a7d` isaacsim:
    disable robot colliders so render matches MuJoCo; `685b5ec` sort loaded collision meshes
- Image: `ghcr.io/songlin/simple:migrate` (already pulled) has the full upgraded venv (Python 3.12.3,
  MuJoCo 3.12.0, numpy 2.5.4, torch 2.11.0+cu128). Compose mounts `src/` and `scripts/` into the
  container, so no image build is needed.

## Steps

1. **Branch and setup**
   - commit the current work on `main`; create branch `upgrade-isaacsim62`
   - `docker tag ghcr.io/songlin/simple:migrate simple:migrate` (compose uses `image: simple:${DATE}`)
   - `docker-compose.yml`: the Kit cache mount is hardcoded to `.venv/lib/python3.10/...`; make the
     Python version a variable (3.12 for the migrate image)
   - check whether the image's baked-in `third_party/decoupled_wbc` is what migrate expects (migrate's
     compose mounts it from the host)
2. **Port the API changes only**
   - go through `e8a4e7e..origin/migrate` file by file; take only hunks needed for the new packages
     (MuJoCo 3.12, numpy 2, Isaac Sim 6.2 / pxr / Isaac Lab 3.0, torch 2.11, Python 3.12) plus
     `pyproject.toml` / `uv.lock` / `.python-version` / curobo submodule
   - skip success checkers, task logic, replay/recording refactors, docs
   - keep a list of every hunk taken or skipped, with the reason
   - smoke test in the migrate container: imports; `gym.make` in `mujoco` and `mujoco_isaac` modes;
     one reset + step for each migrated task
3. **Spawn check in the new env**: `scripts/check_eval_spawn.py` for every migrated task (MuJoCo 3.12
   contact/solver changes could move where the robot settles and invalidate the spawn shift)
4. **Test 1: MuJoCo replay success rate** of the patched training datasets
   (`data/train-psi-data-retract/simple/<Task>`), MuJoCo only, sharded over 8 GPUs with `docker.sh`
   (`DATE=migrate`), same defaults (`--num-pad-frames` 50) in the old (`simple:260829`) and new
   (`simple:migrate`) image; compare per episode. Replays vary by about +-6 episodes run to run
   (wall-clock pacing of the WBC), so run each image twice before calling a drop. Then
   `scripts/check_retract_patch.py --replay-dir ...` on the new runs
5. **Test 2: closed-loop eval** with `scripts/launch_psi0_eval_pair.sh` and `IMAGE_DATE=migrate`, all dr
   levels in parallel with one `STAMP`, outputs under `logs/<stamp>/`. Checks that Isaac 6.2 headless,
   the Kit cache, `--video-dir` and `--trailing-episode-steps` work. Baseline: the same eval in the old
   image (for LocomotionPickBetweenTables: stamp `20261029-203740`, fixed spawns). Report success per
   dr level side by side; with 10 episodes per level, 1-2 episodes of difference is within noise

## Risks

- `8b38f39`: the Isaac changes are tangled with the replay/recording refactor; some API fixes may
  depend on it
- physics: MuJoCo 3.12 may change contacts, stabilization and replay success (steps 3 and 4 catch it)
- the image ships its own `gear_sonic` / `decoupled_wbc` builds; our `src/` must work with them (the
  smoke test catches interface drift)
