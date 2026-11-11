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

## Port log (branch `upgrade`)

Ported from `/hfm/songlin/simple-migrate` (`origin/migrate` @ `15119f6`) onto `main` @ `f705a62`.
Only package / API hunks were taken; every other migrate change stays out.

### Taken

| Source | Files | Why |
|---|---|---|
| `556a460` | `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.12, Isaac Sim 6.2 + Isaac Lab 3.0, MuJoCo 3.12, torch 2.11, numpy>=2 (main never changed these since `e8a4e7e`, so taken whole) |
| `8b38f39` / `1326dd3` | `third_party/curobo` -> `b12518e` | warp 1.17 API (`wp.device_from_torch`) |
| `ed8a64f` | 15 task files + `cli/control.py` (64 call sites, incl. 2 main-only ones) | MuJoCo 3.12: `MjModel.body()` rejects a shape-(1,) array -> `bodyid.item()`. Done as a mechanical sed; the LocomotionPickBetweenTables success-check rewrite in the same commit is **not** taken |
| `8b38f39` | `engines/mujoco.py` | MuJoCo 3.12: `MjSpec.attach(..., prefix="")` (default prefix became `/`); restore 3.3.x CCD defaults (`ccd_iterations=50`, multi-CCD off) |
| `8b38f39` | `engines/isaacsim.py` | `omni.isaac.*` -> `isaacsim.*` imports (`Single*Prim`, `isaacsim.sensors.camera`); RTX real-time with DLSS; scene `xformOp:orient` with unitsResolve factored out + quarter-turn GPU-crash nudge; object USDs as plain references; object collision schemas removed |
| `685b5ec` (isaacsim hunk only) | `engines/isaacsim.py` | scene USD as a plain reference (6.x metrics assembler would rescale cm scenes 100x) |
| `8b38f39` | `envs/base_dual_env.py` | `isaacsim.core.utils.extensions` import |
| `8b38f39` | `cli/_decoupled_wbc_recording.py` + 5 recorders | `create_gr00t_exporter`: numpy>=2.5 rejects `float()` on shape-(1,) features |
| `3e12155` | `examples/multi_arm_reacher.py` | Isaac Sim 6.2 imports + new URDF importer (main never changed this file) |
| `3e12155` | `scripts/install_curobo.sh` | GPU-arch autodetect, explicit build requirements, no `warp-lang==1.7.0` re-pin (nix branch kept) |
| `a2c4ee8`, `ea91af4`, `b8d3b6c` | `Dockerfile`; `scripts/install_curobo_docker.sh` deleted | 6.2.0 / Ubuntu 24.04 base image; Dockerfile now uses `install_curobo.sh` |
| `a2c4ee8` | `docker-compose.yml` | `ISAAC_SIM_VERSION` build arg; `--no-sync` replay service; Kit cache mounted at both the `python3.10` (old 4.5 images, host `.isaac-cache/kit` as before) and `python3.12` (6.2 images, host `.isaac-cache/kit-py312`) venv paths, so no setting is needed for either image |

### Left out (not API changes)

- `ed8a64f` LocomotionPickBetweenTables success check; `10ffaba` / `b9f1cff` success_criteria / reward rounding; all per-task replay fixes (`aba0130` .. `5db2776`, `ce8ec98`, `9854050`, `e331836`, `b7af5ed`, `3c4b0b6`, `0315b1a`)
- `8b38f39` agent refactors (`use_policy_action` constructor arg, `ReplayDecoupledAgent` pad 10->60, `warm_start`, `get_initial_body_q`), `G1Sonic.set_standing_pose`, logging-only changes in `video_writer.py` / `video_recorder.py`, the `replay_decoupled_wbc.py` refactor
- `2032964` AMO stance spawn / history priming, `LeRobotParquetDataset` (see test notes), `588cf65`, `ab0878b` viewer fixes
- `685b5ec` `natural_sorted` collision meshes (changes the compiled MuJoCo model; would confound the replay comparison), `ae09a7d` robot colliders off in Isaac, `3c512f2` table material reuse — render-only bug fixes, candidates once the upgrade is validated
- `third_party/decoupled_wbc` bump and its host bind mount: the host submodule is not checked out; the image's baked copy is used
- `a5ff2a0` cleanup (nix removal etc.), docs

### Test 1 result: MuJoCo replay of the merged SIMPLE-decoupled-wbc dataset (2026-11-02)

`scripts/replay_dwbc_merged.sh` (8 GPUs x 3 shards), old image `38a674039f41` (now tagged `simple:20260830`; was `simple:latest` / `ghcr-20260830`; main) ->
`data/replays/dwbc-merged-2611020514`; new image `bc55dc07dc01` (now `simple:latest`; was `simple:migrate`; this branch) -> `data/replays/dwbc-merged-upgrade`.

| Task | Old | New | Δ | lost | gained |
|---|---|---|---|---|---|
| BendHandoverTeleop | 77/96 | 79/96 | +2 | 3 | 5 |
| BendPickAndPlaceOnSofaMP | 75/75 | 73/75 | -2 | 2 | 0 |
| BendPickAndPlaceTeleop | 46/48 | 45/48 | -1 | 2 | 1 |
| BendPickMP | 83/91 | 83/91 | +0 | 3 | 3 |
| BendPickTeleop | 91/97 | 92/97 | +1 | 0 | 1 |
| CloseDoorTeleop | 99/99 | 99/99 | +0 | 0 | 0 |
| HandoverTeleop | 86/100 | 89/100 | +3 | 2 | 5 |
| LocomotionPickBetweenTablesTeleop | 85/99 | 78/99 | -7 | 9 | 2 |
| OpenFaucetTeleop | 74/99 | 71/99 | -3 | 3 | 0 |
| OpenOvenTeleop | 73/99 | 73/99 | +0 | 3 | 3 |
| OpenTrashCanTeleop | 74/99 | 73/99 | -1 | 4 | 3 |
| PickAndBendPlaceMP | 49/49 | 47/49 | -2 | 2 | 0 |
| PickAndPlaceAndHugContainerTeleop | 60/94 | 63/94 | +3 | 2 | 5 |
| PushOfficeChairTeleop | 77/99 | 77/99 | +0 | 6 | 6 |
| TabletopGraspMP | 94/98 | 91/98 | -3 | 3 | 0 |
| TabletopHandoverMP | 76/76 | 75/76 | -1 | 1 | 0 |
| XMovePickTeleop | 98/99 | 98/99 | +0 | 0 | 0 |
| **Total** | **1317/1517 (86.8%)** | **1306/1517 (86.1%)** | **-11** | | |

Run-to-run noise, from repeating the 2-task smoke on each image: MP replays (`simple.cli.replay`, AMO)
are deterministic on a given image (BendPickMP identical per shard on both images); teleop replays
(`replay_decoupled_wbc`) are not (BendPickAndPlaceTeleop: 7/24 shards differ between two old-image runs,
43 vs 46 of 48).

- **MP tasks: -8 of 389, all caused by the upgrade** (deterministic). Lost episodes: OnSofaMP 107, 161;
  BendPickMP 258, 298, 302 (gained 277, 284, 291); PickAndBendPlaceMP 1014, 1049; TabletopGraspMP 1260,
  1269, 1311; TabletopHandoverMP 1405. Likely the remaining MuJoCo 3.12 difference the migrate notes list as
  uncompensated (mesh polygon data), not yet confirmed.
- **Teleop tasks: -3 of 1128 overall, within noise.** LocomotionPickBetweenTablesTeleop (-7 in the full
  run) was re-run twice per image (old image with `main`'s code from a worktree, new image with this branch):

  | | full run | rerun 1 | rerun 2 | mean |
  |---|---|---|---|---|
  | old (`simple:20260830`, main) | 85/99 | 80/99 | 82/99 | 82.3 |
  | new (`simple:latest`, upgrade) | 78/99 | 81/99 | 80/99 | 79.7 |

  The ranges overlap and no episode passes in all three old runs while failing in all three new ones
  (25 episodes flip between runs at random), so the -7 was mostly noise. A small downward shift (~2-3
  episodes) can't be ruled out with three runs each.
