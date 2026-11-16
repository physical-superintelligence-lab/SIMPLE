# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Recover per-frame raw states (43-D joint state, 7-D base pose, base velocity,
object poses) for a psi0-format SIMPLE dataset (psi-data ``simple/<Task>``) by
*state replay*.  See GitHub issue #38.

Why state replay
----------------
The psi0-format datasets keep every measured joint (``observation.leg_joints``,
``observation.arm_joints``, ``observation.hand_joints`` = 15 + 14 + 14 = 43) but
not the floating base pose.  Re-running the 36-D actions through the WBC
(``simple.cli.replay_decoupled_wbc``) is an *action* replay: it produces a new
rollout whose joints, base pose and objects drift away from the recorded ones,
so the "recovered" base pose would belong to a different trajectory.

Here the robot's 43 joints are instead *prescribed* to follow the recorded
trajectory exactly, and MuJoCo integrates only what the dataset does not
record: the floating base (through foot contacts and the momentum of the
moving limbs) and the scene objects.  Every output row therefore matches the
source row joint-for-joint, and the base pose is the one implied by those
joints.

How the joints are prescribed
-----------------------------
Each 50 Hz frame interval is split into the physics substeps (200 Hz) and the
joints follow the linear interpolation of the recorded positions.  The joint
DoFs get a very large armature A and their limits are disabled; each substep a
deadbeat generalized force  tau = A * (qd_des - qd) / dt,  qd_des = (q_des - q) / dt
lands every joint on its target.  Because the joints are driven by forces
inside one MuJoCo solve (rather than by overwriting qpos/qvel), their reaction
on the floating base (momentum transfer when the limbs accelerate) and the foot
and object contact forces are all resolved together, and the base moves as the
recorded joint motion makes it move.  Actuators and the elastic band are off.

Output
------
A self-contained LeRobot dataset: meta, a copy of each replayed episode's
video(s), and one parquet per source episode, with the same ``episode_index``,
``frame_index``, ``index`` and ``timestamp``, and the original columns kept
(including the 36-D ``action``), plus:

  observation.state        (43) joints in G1 ``WHOLE_BODY_JOINTS`` order
                                (legs, waist yaw/roll/pitch, arms, hands)
  observation.base_pose    (7)  pelvis pos xyz + quat wxyz (MuJoCo qpos[:7])
  observation.base_vel     (6)  MuJoCo qvel[:6] (linear world, angular local)
  observation.object_poses (7 * n_objects) pos xyz + quat wxyz per object

The 36-D psi0 ``action`` is passed through unchanged.

Data
----
Source data: the per-task zips in https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple,
extracted to ``data/simple/<Task>``.  (If a dataset carries a recorded
``observation.base_pose``, the replayed one is written next to it as
``*_replay`` and the error against it is reported.)

Usage (inside the sim container, MuJoCo only, no GPU rendering needed)
----------------------------------------------------------------------
  ./simple.sh start        # once: long-lived container
  ./simple.sh exec bash -c '.venv/bin/hf download USC-PSI-Lab/psi-data --repo-type dataset \\
      --include "simple/G1WholebodyBendPickTeleop-v0.zip" --local-dir data \\
      && cd data/simple && unzip -q -o G1WholebodyBendPickTeleop-v0.zip'
  ./simple.sh python -m simple.cli.state_replay \\
      simple/G1WholebodyBendPickTeleop-v0 \\
      --data-dir=data/simple/G1WholebodyBendPickTeleop-v0 \\
      --save-dir=data/simple-raw-states

  # a slice of the episodes (e.g. to split a task over several runs; each run
  # writes only its own episodes, so the outputs can be copied together)
  ./simple.sh python -m simple.cli.state_replay simple/G1WholebodyBendPickTeleop-v0 \\
      --data-dir=data/simple/G1WholebodyBendPickTeleop-v0 \\
      --save-dir=data/simple-raw-states --episode-start 50 --num-episodes 47
"""

from __future__ import annotations

import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import json
import re
import shutil
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm
from typing_extensions import Annotated, TYPE_CHECKING

import simple.envs as _  # noqa: F401  register all envs
from simple.cli._replay_checks import require_episode_task
from simple.cli.render_decoupled_wbc import _load_episode_configs, _load_episodes
from simple.robots.g1_sonic import WHOLE_BODY_JOINTS

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

# Armature added to the robot joint DoFs: large enough that gravity and
# contact torques change a joint's velocity by < 1e-4 rad/s over one 5 ms
# substep, small enough to keep the mass matrix well conditioned.
PRESCRIBED_ARMATURE = 1e4
SETTLE_MIN_STEPS = 100        # 0.5 s at 200 Hz
SETTLE_MAX_STEPS = 2000       # 10 s
SETTLE_VEL_THRESHOLD = 1e-3   # max |base qvel| (m/s, rad/s), same as G1Sonic.stabilized


def _joint_trajectory(ep_df, state_names: list[str] | None) -> np.ndarray:
    """(T, 43) joint positions in WHOLE_BODY_JOINTS order."""
    if "observation.state" in ep_df.columns:  # raw decoupled-WBC recording
        q = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float64)
        assert state_names is not None
        order = [state_names.index(j) for j in WHOLE_BODY_JOINTS]
        return q[:, order]
    # psi0 format: simple.datasets.psi0 writes
    #   leg_joints  = left leg (6) + right leg (6) + waist yaw/roll/pitch (3)
    #   arm_joints  = left arm (7) + right arm (7)
    #   hand_joints = left hand (7) + right hand (7), thumb/index/middle
    # whose concatenation is exactly WHOLE_BODY_JOINTS.
    return np.concatenate(
        [np.stack(ep_df[c].to_numpy()).astype(np.float64)
         for c in ("observation.leg_joints", "observation.arm_joints", "observation.hand_joints")],
        axis=1,
    )


class PrescribedJointReplay:
    """Drives the robot joints along a recorded trajectory; MuJoCo simulates the rest."""

    def __init__(self, sonic_env: "SonicLocoManipEnv"):
        self.env = sonic_env
        self.m: mujoco.MjModel = sonic_env.mujoco.mjModel
        self.d: mujoco.MjData = sonic_env.mujoco.mjData
        m = self.m
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in WHOLE_BODY_JOINTS]
        missing = [j for j, i in zip(WHOLE_BODY_JOINTS, jids) if i < 0]
        if missing:
            raise KeyError(f"robot joints not in the MuJoCo model: {missing}")
        self.qadr = m.jnt_qposadr[jids]
        self.vadr = m.jnt_dofadr[jids]
        self.n_sub = int(round(1.0 / (sonic_env.mujoco.physics_dt * sonic_env.task.render_hz)))

        # Prescribed joints: rigid over a substep, no limit constraints.
        m.dof_armature[self.vadr] = PRESCRIBED_ARMATURE
        m.jnt_limited[jids] = 0
        robot = sonic_env.task.robot
        if getattr(robot, "elastic_band", None) is not None:
            robot.elastic_band.enable = False

    def _step_to(self, q_des: np.ndarray) -> None:
        """One physics step whose joint forces land the joints on q_des."""
        d, dt = self.d, self.m.opt.timestep
        qd_des = (q_des - d.qpos[self.qadr]) / dt
        d.ctrl[:] = 0.0
        d.xfrc_applied[:] = 0.0
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.vadr] = PRESCRIBED_ARMATURE * (qd_des - d.qvel[self.vadr]) / dt
        mujoco.mj_step(self.m, d)

    def start(self, q0: np.ndarray, base_pose: np.ndarray | None = None) -> int:
        """Put the robot in the first recorded pose.  With base_pose (validation)
        the base starts there; otherwise hold q0 until the base comes to rest,
        as the recordings also start after the robot stabilized."""
        d = self.d
        d.qpos[self.qadr] = q0
        d.qvel[self.vadr] = 0.0
        if base_pose is not None:
            d.qpos[:7] = base_pose
            d.qvel[:6] = 0.0
            mujoco.mj_forward(self.m, d)
            return 0
        for i in range(SETTLE_MAX_STEPS):
            self._step_to(q0)
            if i >= SETTLE_MIN_STEPS and np.max(np.abs(d.qvel[:6])) < SETTLE_VEL_THRESHOLD:
                break
        mujoco.mj_forward(self.m, d)
        return i + 1

    def advance(self, q_from: np.ndarray, q_to: np.ndarray) -> None:
        """Integrate one dataset frame interval with the joints on the recorded path."""
        for k in range(1, self.n_sub + 1):
            self._step_to(q_from + (k / self.n_sub) * (q_to - q_from))
        mujoco.mj_forward(self.m, self.d)

    def joint_tracking_error(self, q: np.ndarray) -> float:
        return float(np.max(np.abs(self.d.qpos[self.qadr] - q)))

    def base_pose(self) -> np.ndarray:
        return self.d.qpos[:7].copy()

    def base_vel(self) -> np.ndarray:
        return self.d.qvel[:6].copy()

    def object_poses(self) -> np.ndarray:
        objs = self.env.mujoco.mj_objects.values()
        if not objs:
            return np.zeros(0)
        return np.concatenate([np.concatenate([o.xpos, o.xquat]) for o in objs])


def _yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _base_pose_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    pos = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    xy = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    dz = np.abs(pred[:, 2] - gt[:, 2])
    dyaw = np.abs(np.angle(np.exp(1j * (_yaw(pred[:, 3:]) - _yaw(gt[:, 3:])))))
    dot = np.clip(np.abs(np.sum(pred[:, 3:] * gt[:, 3:], axis=1)), 0, 1)
    rot = 2 * np.arccos(dot)
    path = np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))
    return {
        "pos_mean_m": float(pos.mean()), "pos_max_m": float(pos.max()),
        "xy_final_m": float(xy[-1]), "z_max_m": float(dz.max()),
        "yaw_max_deg": float(np.degrees(dyaw.max())), "rot_max_deg": float(np.degrees(rot.max())),
        "gt_xy_path_m": float(path),
    }


def _write_episode(out_dir: Path, info: dict, ep_df, new_cols: dict) -> None:
    ep_idx = int(ep_df["episode_index"].iloc[0])
    chunk = ep_idx // int(info.get("chunks_size", 1000))
    path = out_dir / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(ep_df.reset_index(drop=True), preserve_index=False)
    for name, arr in new_cols.items():
        col = pa.array(list(arr), type=pa.list_(pa.float64()))
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, col)
        else:
            table = table.append_column(name, col)
    pq.write_table(table, path)


def _write_meta(src: Path, out_dir: Path, n_objects: int, obj_names: list[str], validate: bool) -> dict:
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").iterdir():
        if f.is_file() and f.name != "info.json":
            shutil.copy2(f, out_dir / "meta" / f.name)
    info = json.loads((src / "meta" / "info.json").read_text())
    feats = info["features"]
    sfx = "_replay" if validate else ""
    if not validate:
        feats["observation.state"] = {"dtype": "float64", "shape": [43], "names": list(WHOLE_BODY_JOINTS)}
    feats[f"observation.base_pose{sfx}"] = {"dtype": "float64", "shape": [7],
                                            "names": ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]}
    feats[f"observation.base_vel{sfx}"] = {"dtype": "float64", "shape": [6],
                                           "names": ["lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]}
    if n_objects:
        feats[f"observation.object_poses{sfx}"] = {
            "dtype": "float64", "shape": [7 * n_objects],
            "names": [f"{o}.{s}" for o in obj_names
                      for s in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]}
    info["state_replay"] = {
        "source": str(src),
        "method": "simple.cli.state_replay: joints prescribed from the recording, "
                  "floating base and objects simulated in MuJoCo",
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    # Earlier versions linked videos/ to the source; copying through that link
    # would write into the source dataset, so drop it (the link only).
    if (out_dir / "videos").is_symlink():
        (out_dir / "videos").unlink()
    return info


def _copy_episode_videos(src: Path, out_dir: Path, info: dict, ep_idx: int) -> None:
    """Copy (not link) the episode's videos so the output dataset is self-contained."""
    template = info.get("video_path")
    if not template:
        return
    chunk = ep_idx // int(info.get("chunks_size", 1000))
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for key in video_keys or [None]:
        rel = template.format(episode_chunk=chunk, episode_index=ep_idx, video_key=key)
        if not (src / rel).exists():
            print(f"[ep {ep_idx}] warning: video not found: {src / rel}")
            continue
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, out_dir / rel)


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPickTeleop-v0",
    data_dir: Annotated[str, typer.Option()] = "data/simple/G1WholebodyBendPickTeleop-v0",
    save_dir: Annotated[str, typer.Option()] = "data/simple-raw-states",
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = True,
    episode_start: Annotated[int, typer.Option()] = 0,
    num_episodes: Annotated[int, typer.Option()] = -1,
):
    """State-replay a SIMPLE dataset and export 43-D state + base pose per frame."""
    # Only the decoupled-WBC teleop tasks are supported: the MP tasks (AMO controller)
    # use another env/robot setup, and the Sonic datasets already record these states.
    if not re.fullmatch(r"(simple/)?[A-Za-z0-9_]+Teleop-v\d+", env_id):
        raise SystemExit(
            f"error: {env_id!r} is not a teleop task; state_replay only supports the "
            "decoupled-WBC teleop tasks (simple/*Teleop-v*)."
        )
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    src = Path(data_dir)
    src_info = json.loads((src / "meta" / "info.json").read_text())
    state_names = (src_info["features"].get("observation.state") or {}).get("names")
    episodes = _load_episodes(data_dir)
    configs = _load_episode_configs(data_dir)
    ep_ids = sorted(episodes)[episode_start:]
    if num_episodes >= 0:
        ep_ids = ep_ids[:num_episodes]
    # A raw decoupled-WBC recording carries the ground-truth base pose: validate against it.
    validate = "observation.base_pose" in src_info["features"]
    print(f"{data_dir}: replaying {len(ep_ids)} of {len(episodes)} episodes"
          + (" (validation against recorded base pose)" if validate else ""))

    cfg = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = cfg.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    env = gym.make(
        env_id, sim_mode=sim_mode, render_hz=int(src_info["fps"]),
        physics_dt=sonic_config["SIMULATE_DT"], headless=headless, webrtc=False,
        max_episode_steps=100000, sonic_config=sonic_config,
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore

    out_dir = Path(save_dir) / src.name if src.name != "level-0" else Path(save_dir) / src.parent.name
    info = None
    errors = {}
    try:
        for ep_idx in tqdm(ep_ids, desc="Episodes", unit="ep"):
            ep_df = episodes[ep_idx]
            env_conf = configs[ep_idx]
            require_episode_task(env_conf, sonic_env.task, ep_idx, data_dir, env_id)
            env.reset(options={"state_dict": env_conf})
            # The model is rebuilt on reset: re-apply the prescribed-joint setup.
            replay = PrescribedJointReplay(sonic_env)

            q = _joint_trajectory(ep_df, state_names)
            gt = (np.stack(ep_df["observation.base_pose"].to_numpy()).astype(np.float64)
                  if validate else None)
            if validate:
                # How far the settled spawn pose is from the recording's first frame
                # (the only start available for psi0-format data).
                replay.start(q[0])
                spawn_offset = float(np.linalg.norm(replay.base_pose()[:3] - gt[0, :3]))
                # Then measure the replay itself from the recorded start.
                replay.start(q[0], base_pose=gt[0])
            else:
                replay.start(q[0])
            T = len(q)
            base_pose, base_vel, obj_poses = np.zeros((T, 7)), np.zeros((T, 6)), []
            track_err = 0.0
            for t in range(T):
                if t > 0:
                    replay.advance(q[t - 1], q[t])
                track_err = max(track_err, replay.joint_tracking_error(q[t]))
                base_pose[t], base_vel[t] = replay.base_pose(), replay.base_vel()
                obj_poses.append(replay.object_poses())
            if track_err > 1e-3:
                print(f"[ep {ep_idx}] warning: max joint tracking error {track_err:.2e} rad")

            if info is None:
                obj_labels = [str(k) for k in sonic_env.mujoco.mj_objects.keys()]
                info = _write_meta(src, out_dir, len(obj_labels), obj_labels, validate)
            new_cols = {
                "observation.state": q,
                "observation.base_pose": base_pose,
                "observation.base_vel": base_vel,
            }
            if len(obj_poses[0]):
                new_cols["observation.object_poses"] = np.stack(obj_poses)
            if validate:
                # Raw recording: keep its columns, add the replayed ones next to them.
                errors[ep_idx] = {**_base_pose_error(base_pose, gt),
                                  "spawn_offset_m": spawn_offset, "joint_track_max_rad": track_err}
                print(f"[ep {ep_idx}] " + "  ".join(f"{k}={v:.4f}" for k, v in errors[ep_idx].items()))
                new_cols.pop("observation.state")
                new_cols = {f"{k}_replay": v for k, v in new_cols.items()}
            _write_episode(out_dir, info, ep_df, new_cols)
            _copy_episode_videos(src, out_dir, info, ep_idx)
    finally:
        env.close()

    if errors:
        keys = next(iter(errors.values())).keys()
        summary = {k: float(np.mean([e[k] for e in errors.values()])) for k in keys}
        summary["worst_pos_max_m"] = float(max(e["pos_max_m"] for e in errors.values()))
        print("\n[validation] mean over episodes: " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        (out_dir / "meta" / f"state_replay_errors_{ep_ids[0]}.json").write_text(
            json.dumps({"per_episode": errors, "mean": summary}, indent=2))
    print(f"wrote {len(ep_ids)} episodes to {out_dir}")


if __name__ == "__main__":
    typer.run(main)
