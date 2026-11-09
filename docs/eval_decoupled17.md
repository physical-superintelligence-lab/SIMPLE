# psi0 decoupled17: Isaac Sim evals, all 17 SIMPLE tasks

Checkpoint: `USC-PSI-Lab/psi-model` `psi0/simple-checkpoints/decoupled17.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2610310854`,
step 20000 (one checkpoint for all 17 tasks, trained on `simple-decoupled-wbc-train`).

Run: `scripts/launch_psi0_eval_pair.sh` (psi0 server `--action-exec-horizon=24 --rtc`), image `simple:260829`,
`--sim-mode=mujoco_isaac`; teleop tasks with `--isaac-render=on-query`, MP tasks via `cli/eval.py` (Isaac renders
every step there). One psi0 server per slot, reused across tasks (`CLIENT_ONLY=1`), 8 slots on 8 GPUs.
Eval data: `data/SIMPLE-eval/<Task>/<level>` (= HF `USC-PSI-Lab/SIMPLE-eval`), 10 episodes per level.
Logs: `logs/20261031-223949-decoupled17/<Task>/`.

Successes per level out of 10. "Per-task ckpt" = the task's own checkpoint, from `docs/migration.md`.

| Task | dr-level-0 | dr-level-1 | dr-level-2 | dr-level-hard | Total | Per-task ckpt | Δ (decoupled17 - per-task) |
|---|---|---|---|---|---|---|---|
| BendHandoverTeleop | 8 | 8 | 5 | 5 | **26/40** | 28/40 | -2 |
| BendPickAndPlaceOnSofaMP | 1 | 1 | 0 | 0 | **2/40** | 2/40 (old) | 0 |
| BendPickAndPlaceTeleop | 0 | 4 | 3 | 4 | **11/40** | 29/40 | -18 |
| BendPickMP | 1 | 1 | 0 | 0 | **2/40** | 32/40 | -30 |
| BendPickTeleop | 8 | 8 | 8 | 7 | **31/40** | 30/40 | +1 |
| CloseDoorTeleop | 9 | 10 | 9 | 10 | **38/40** | 38/40 | 0 |
| HandoverTeleop | 8 | 8 | 8 | 8 | **32/40** | 36/40 | -4 |
| LocomotionPickBetweenTablesTeleop | 6 | 6 | 3 | 2 | **17/40** | 27/40 | -10 |
| OpenFaucetTeleop | 4 | 3 | 5 | 4 | **16/40** | 21/40 | -5 |
| OpenOvenTeleop | 9 | 9 | 9 | 6 | **33/40** | 11/40 | +22 |
| OpenTrashCanTeleop | 6 | 4 | 4 | 5 | **19/40** | 25/40 | -6 |
| PickAndBendPlaceMP | 0 | 0 | 0 | 0 | **0/40** | - | - |
| PickAndPlaceAndHugContainerTeleop | 6 | 4 | 3 | 0 | **13/40** | 13/40 | 0 |
| PushOfficeChairTeleop | 10 | 10 | 10 | 7 | **37/40** | 36/40 | +1 |
| TabletopGraspMP | 3 | 6 | 3 | 0 | **12/40** | 29/40 | -17 |
| TabletopHandoverMP | 0 | 3 | 2 | 2 | **7/40** | 15/40 | -8 |
| XMovePickTeleop | 1 | 2 | 0 | 1 | **4/40** | 38/40 | -34 |
| **Teleop (12 tasks)** | 75/120 (62%) | 76/120 (63%) | 67/120 (56%) | 59/120 (49%) | **277/480** (57.7%) | 332/480 | **-55** (-11.5 pts) |
| **MP (5 tasks)** | 5/50 (10%) | 11/50 (22%) | 5/50 (10%) | 2/50 (4%) | **23/200** (11.5%) | 78/160 (4 tasks) | **-55** (-34.4 pts, 4 tasks) |
| **All (17 tasks)** | 80/170 (47%) | 87/170 (51%) | 72/170 (42%) | 61/170 (36%) | **300/680** (44.1%) | 410/640 (16 tasks) | **-110** (-17.2 pts, 16 tasks) |

## Per-task checkpoints

The per-task column above, per level (from `docs/migration.md`; each task's own checkpoint, all in
`USC-PSI-Lab/psi-model` `psi0/simple-checkpoints/` except mix24, which is local only).

| Task | dr-level-0 | dr-level-1 | dr-level-2 | dr-level-hard | Total | Checkpoint (step) |
|---|---|---|---|---|---|---|
| BendHandoverTeleop | 8 | 7 | 8 | 5 | **28/40** | `g1wholebodybendhandoverteleop-v0-lv0...2605220414` (40k) |
| BendPickAndPlaceOnSofaMP | 0 | 0 | 1 | 1 | **2/40** (old) | `mix24-q99-150k...2608092342` (150k) |
| BendPickAndPlaceTeleop | 8 | 8 | 8 | 5 | **29/40** | `simple-bendpickplace...2610300540` (20k) |
| BendPickMP | 10 | 9 | 10 | 3 | **32/40** | `g1wholebodybendpick-v0...2603151312` (40k) |
| BendPickTeleop | 8 | 8 | 6 | 8 | **30/40** | `g1wholebodybendpickteleop-v0...2605230955` (40k) |
| CloseDoorTeleop | 9 | 9 | 10 | 10 | **38/40** | `g1wholebodyclosedoorteleop-v0...2605070100` (40k) |
| HandoverTeleop | 9 | 9 | 9 | 9 | **36/40** | `handover...2610310807` (20k) |
| LocomotionPickBetweenTablesTeleop | 9 | 9 | 5 | 4 | **27/40** | `g1wholebodylocomotionpickbetweentablesteleop-v0...2604081126` |
| OpenFaucetTeleop | 6 | 3 | 8 | 4 | **21/40** | `g1wholebodyopenfaucetteleop-v0...2605081439` (40k) |
| OpenOvenTeleop | 2 | 4 | 5 | 0 | **11/40** | `g1wholebodyopenoventeleop-v0...2605120604` (40k) |
| OpenTrashCanTeleop | 6 | 6 | 6 | 7 | **25/40** | `g1wholebodyopentrashcanteleop-v0...2605291607` (40k) |
| PickAndBendPlaceMP | - | - | - | - | - | none on the current data |
| PickAndPlaceAndHugContainerTeleop | 4 | 5 | 3 | 1 | **13/40** | `g1wholebodypickandplaceandhugcontainerteleop-v0...2604280201` (40k) |
| PushOfficeChairTeleop | 10 | 10 | 10 | 6 | **36/40** | `g1wholebodypushofficechairteleop-v0...2605291520` (40k) |
| TabletopGraspMP | 10 | 10 | 8 | 1 | **29/40** | `g1wholebodytabletopgrasp-v0...2603181503` (40k) |
| TabletopHandoverMP | 4 | 3 | 4 | 4 | **15/40** | `mix24-q99-150k...2608092342` (150k) |
| XMovePickTeleop | 10 | 10 | 10 | 8 | **38/40** | `xmovepick...2610310809` (20k) |
| **Teleop (12 tasks)** | 89/120 (74%) | 88/120 (73%) | 88/120 (73%) | 67/120 (56%) | **332/480** (69.2%) | |
| **MP (4 tasks, no PickAndBendPlaceMP)** | 24/40 (60%) | 22/40 (55%) | 23/40 (58%) | 9/40 (22%) | **78/160** (48.8%) | |
| **All (16 tasks)** | 113/160 (71%) | 110/160 (69%) | 111/160 (69%) | 76/160 (48%) | **410/640** (64.1%) | |

decoupled17 on the same 16 tasks: Teleop 277/480 (57.7%, -11.5 pts), MP 23/160 (14.4%, -34.4 pts),
all 300/640 (46.9%, -17.2 pts). decoupled17 is better on OpenOvenTeleop (33 vs 11), about equal on
BendPickTeleop, CloseDoorTeleop, PickAndPlaceAndHugContainerTeleop, PushOfficeChairTeleop and
BendPickAndPlaceOnSofaMP, and much worse on XMovePickTeleop (4 vs 38), BendPickMP (2 vs 32),
BendPickAndPlaceTeleop (11 vs 29) and TabletopGraspMP (12 vs 29).

Notes:
- Per-task ckpt column: those evals predate some fixes (see `docs/migration.md`); BendPickAndPlaceOnSofaMP's
  2/40 is mix24 on the old hard level; PickAndBendPlaceMP had no eval on its current (57-episode) data.
- The training data (`SIMPLE-org/SIMPLE-decoupled-wbc`, 1584 episodes, all 17 tasks) holds the same
  episodes as `data-fixed` (BendPickMP checked: actions and states identical) and the eval instructions
  match the trained ones.
- MP tasks: these evals ran with the Dex3 hand torque limits (60d1abc); the per-task MP evals ran
  before it. A/B on BendPickMP dr-level-0 with the old limits (5 / 3 N·m, via a temporary
  `--hand-torque=legacy` eval option, since removed): 0/10 vs 1/10, so the torque limits do not explain the
  drop (`logs/20261031-223949-decoupled17-handtorque-ab/legacy`). The Dex3 limits were reverted afterwards
  (2026-11-01); the code now uses the old limits.
- `cli/eval.py` (MP tasks) renders only the head cameras since this run (front_stereo dropped).
