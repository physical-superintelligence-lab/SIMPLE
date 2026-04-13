"""
Render/observation synchronization check.

This utility records an episode and writes an EEF-overlay debug video.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import typer
import transforms3d as t3d
from tqdm import tqdm
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401  # ensure envs are registered
from simple.baselines.vlt import project_2d_traj
from simple.utils import env_flag, snake_to_pascal


def _resolve_mujoco_eef_body_name(env, task) -> str | None:
    try:
        model = env.unwrapped.mujoco.mj_physics_model  # type: ignore[attr-defined]
        nbody = int(model.nbody)
        body_names = {model.body(i).name for i in range(nbody)}
    except Exception:
        return None

    candidates = [
        getattr(task.robot, "eef_prim_path", None),
        getattr(task.robot, "RIGHT_ARM_EE_LINK", None),
        getattr(task.robot, "hand_prim_path", None),
        "right_hand_index_finger_tip",
        "right_hand_thumb_finger_tip",
        "right_hand_middle_finger_tip",
        "right_wrist_yaw_link",
    ]
    for c in candidates:
        if isinstance(c, str) and c in body_names:
            return c
    return None


def _resolve_mujoco_tip_body_name(env) -> str | None:
    try:
        model = env.unwrapped.mujoco.mj_physics_model  # type: ignore[attr-defined]
        nbody = int(model.nbody)
        body_names = {model.body(i).name for i in range(nbody)}
    except Exception:
        return None

    tip_candidates = [
        "right_hand_index_finger_tip",
        "right_hand_middle_finger_tip",
        "right_hand_thumb_finger_tip",
    ]
    for name in tip_candidates:
        if name in body_names:
            return name
    return None


def _pose_to_T(p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = t3d.quaternions.quat2mat(q_wxyz.astype(np.float64)).astype(np.float32)
    T[:3, 3] = p.astype(np.float32)
    return T


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    return t3d.quaternions.mat2quat(R.astype(np.float64)).astype(np.float32)


def _get_mujoco_eef_pose(env, body_name: str | None, base_body_name: str = "pelvis") -> np.ndarray | None:
    if not body_name:
        return None
    try:
        body = env.unwrapped.mujoco.mj_physics_data.body(body_name)  # type: ignore[attr-defined]
        p_w = np.asarray(body.xpos, dtype=np.float32)
        q_w = np.asarray(body.xquat, dtype=np.float32)  # mujoco: wxyz

        # Convert world-frame body pose to robot(base)-frame pose expected by projector.
        base = env.unwrapped.mujoco.mj_physics_data.body(base_body_name)  # type: ignore[attr-defined]
        p_base_w = np.asarray(base.xpos, dtype=np.float32)
        q_base_w = np.asarray(base.xquat, dtype=np.float32)  # wxyz

        T_w_eef = _pose_to_T(p_w, q_w)
        T_w_base = _pose_to_T(p_base_w, q_base_w)
        T_base_eef = np.linalg.inv(T_w_base) @ T_w_eef

        p = T_base_eef[:3, 3].astype(np.float32)
        q = _mat_to_quat_wxyz(T_base_eef[:3, :3])
        pose = np.concatenate([p, q]).astype(np.float32)
        if np.all(np.isfinite(pose)):
            return pose
    except Exception:
        return None
    return None


def _get_eef_pose(obs: dict, env, task) -> np.ndarray | None:
    # Preferred: direct env observation.
    if "eef_pose" in obs:
        return np.asarray(obs["eef_pose"], dtype=np.float32)

    # Fallback: reconstruct from joint_qpos via robot kinematics.
    try:
        qpos = obs.get("joint_qpos", None)
        if qpos is None:
            qpos = np.asarray(list(env.unwrapped.mujoco.get_robot_qpos().values()), dtype=np.float32)  # type: ignore[attr-defined]
        if hasattr(task.robot, "fk"):
            p, q = task.robot.fk(qpos)  # type: ignore[misc]
            pose = np.asarray(np.concatenate([np.asarray(p), np.asarray(q)]), dtype=np.float32)
            if pose.size == 7 and np.all(np.isfinite(pose)):
                return pose
        hand_pose = np.asarray(task.robot.get_link_pose(task.robot.eef_prim_path, qpos), dtype=np.float32)
        if hand_pose.size == 7 and hasattr(task.robot, "get_eef_pose_from_hand_pose"):
            p, q = task.robot.get_eef_pose_from_hand_pose(hand_pose[:3], hand_pose[3:])  # type: ignore[misc]
            return np.asarray(np.concatenate([p, q]), dtype=np.float32)
        if hand_pose.size == 7:
            return hand_pose
    except Exception as e:
        if env_flag("SIMPLE_SYNC_DEBUG_KEYS", default=False):
            print(f"[sync-debug] eef fallback failed: {type(e).__name__}: {e}")
        return None
    return None


def _camera_intrinsics_from_infos(camera_infos: list[dict], cam_name: str) -> tuple[float, float, float, float]:
    for cam_info in camera_infos:
        name = cam_info.get("name")
        if name == "front_stereo" and cam_name in {"front_stereo_left", "front_stereo_right"}:
            return float(cam_info["fx"]), float(cam_info["fy"]), float(cam_info["cx"]), float(cam_info["cy"])
        if name == cam_name:
            return float(cam_info["fx"]), float(cam_info["fy"]), float(cam_info["cx"]), float(cam_info["cy"])
    raise ValueError(f"camera intrinsics missing for {cam_name}")


def _get_live_camera_pose_robot(env, cam_name: str) -> np.ndarray | None:
    try:
        isaac = getattr(env.unwrapped, "isaac", None)  # type: ignore[attr-defined]
        if isaac is None:
            return None
        camera = isaac.cameras.get(cam_name, None)
        if camera is None:
            return None
        cam_p_w, cam_q_w = camera.get_world_pose()
        robot_p_w, robot_q_w = isaac.robot.get_world_pose()
        T_w_cam = _pose_to_T(np.asarray(cam_p_w, dtype=np.float32), np.asarray(cam_q_w, dtype=np.float32))
        T_w_robot = _pose_to_T(np.asarray(robot_p_w, dtype=np.float32), np.asarray(robot_q_w, dtype=np.float32))
        T_robot_cam = np.linalg.inv(T_w_robot) @ T_w_cam
        p = T_robot_cam[:3, 3].astype(np.float32)
        q = _mat_to_quat_wxyz(T_robot_cam[:3, :3])
        pose = np.concatenate([p, q]).astype(np.float32)
        if np.all(np.isfinite(pose)):
            return pose
    except Exception:
        return None
    return None


def _project_2d_traj_live(camera_infos: list[dict], eef_pose: np.ndarray, cam_poses_robot: np.ndarray, cam_name: str) -> np.ndarray:
    fx, fy, cx, cy = _camera_intrinsics_from_infos(camera_infos, cam_name)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    n = min(eef_pose.shape[0], cam_poses_robot.shape[0])
    traj_2d: list[np.ndarray] = []
    switch_frame = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    for i in range(n):
        p_cam = cam_poses_robot[i, :3]
        q_cam = cam_poses_robot[i, 3:]
        T_robot_cam = np.eye(4, dtype=np.float32)
        T_robot_cam[:3, :3] = t3d.quaternions.quat2mat(q_cam.astype(np.float64)).astype(np.float32)
        T_robot_cam[:3, 3] = p_cam

        p_eef = eef_pose[i, :3]
        q_eef = eef_pose[i, 3:]
        T_robot_eef = np.eye(4, dtype=np.float32)
        T_robot_eef[:3, :3] = t3d.quaternions.quat2mat(q_eef.astype(np.float64)).astype(np.float32)
        T_robot_eef[:3, 3] = p_eef

        T_cam_eef = switch_frame @ (np.linalg.inv(T_robot_cam) @ T_robot_eef)
        p_in_cam = T_cam_eef[:3, 3]
        if not np.isfinite(p_in_cam).all() or p_in_cam[2] <= 1e-6:
            traj_2d.append(np.array([np.nan, np.nan], dtype=np.float32))
            continue
        p_in_2d = K @ p_in_cam / p_in_cam[2]
        traj_2d.append(p_in_2d[:2].astype(np.float32))
    return np.asarray(traj_2d, dtype=np.float32)


def _write_overlay_video(path: str, frames: list[np.ndarray], traj_2d: np.ndarray):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (w, h),
    )
    if not writer.isOpened():
        return
    traj = np.asarray(traj_2d, dtype=np.float32)
    for i, frame in enumerate(frames):
        overlay_canvas = frame.copy()
        for j in range(max(0, i - 20), i):
            if j + 1 >= traj.shape[0]:
                continue
            p0 = tuple(np.round(traj[j]).astype(int))
            p1 = tuple(np.round(traj[j + 1]).astype(int))
            cv2.line(overlay_canvas, p0, p1, (255, 255, 0), 1, cv2.LINE_AA)
        if i < traj.shape[0]:
            p = tuple(np.round(traj[i]).astype(int))
            cv2.circle(overlay_canvas, p, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(overlay_canvas, f"overlay t={i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(overlay_canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def _resolve_obs_camera_key(obs: dict, camera_infos: list[dict], requested_key: str) -> tuple[str, str]:
    # proj camera name should match env_config camera names used by project_2d_traj.
    available_cam_names = [str(c.get("name")) for c in camera_infos if c.get("name") is not None]

    if requested_key in obs:
        proj_name = requested_key if requested_key in available_cam_names else requested_key
        return requested_key, proj_name

    if requested_key in available_cam_names and requested_key in obs:
        return requested_key, requested_key

    alias_candidates = {
        "head": ["head_stereo_left", "head_stereo_right"],
        "front": ["front_stereo_left", "front_stereo_right"],
        "wrist": ["wrist", "wrist_left"],
    }
    for cand in alias_candidates.get(requested_key, []):
        if cand in obs:
            proj_name = cand if cand in available_cam_names else requested_key
            return cand, proj_name

    # If requested key is only present in camera infos, map to the first obs key with that prefix.
    if requested_key in available_cam_names:
        for k in obs.keys():
            if str(k) == requested_key:
                return str(k), requested_key

    # Fallback to any image-like observation key.
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (1, 3, 4):
            fallback = str(k)
            proj_name = fallback if fallback in available_cam_names else requested_key
            return fallback, proj_name

    raise KeyError(f"Unable to resolve camera key '{requested_key}' from obs keys: {list(obs.keys())}")


def _camera_infos_for_episode(data_format: str, eval_dir: str, env_id: str, split: str, episode_idx: int, data_dir: str):
    def _extract_camera_infos(env_cfg: dict) -> list[dict]:
        if isinstance(env_cfg.get("camera_info"), list):
            return env_cfg["camera_info"]  # type: ignore[return-value]
        scene_info = env_cfg.get("scene_info", {})
        if isinstance(scene_info, dict) and isinstance(scene_info.get("camera_info"), list):
            return scene_info["camera_info"]  # type: ignore[return-value]

        # LeRobot env config may store camera configs under task.layout.camera dict.
        task = env_cfg.get("task", {})
        layout = task.get("layout", {}) if isinstance(task, dict) else {}
        if not layout and isinstance(env_cfg.get("layout"), dict):
            layout = env_cfg["layout"]

        cam_dict = {}
        if isinstance(layout, dict):
            if isinstance(layout.get("camera"), dict):
                cam_dict = layout["camera"]
            elif isinstance(layout.get("cameras"), dict):
                cam_dict = layout["cameras"]

        if isinstance(cam_dict, dict) and cam_dict:
            out = []
            for cam_name, cam_cfg in cam_dict.items():
                if not isinstance(cam_cfg, dict):
                    continue
                pose = cam_cfg.get("pose", {})
                out.append(
                    {
                        "name": cam_name,
                        "fx": cam_cfg.get("fx"),
                        "fy": cam_cfg.get("fy"),
                        "cx": cam_cfg.get("cx"),
                        "cy": cam_cfg.get("cy"),
                        "position": pose.get("position"),
                        "orientation": pose.get("quaternion"),
                    }
                )
            if out:
                return out
        raise KeyError("camera_info")

    if data_format == "rlds_numpy":
        data_path = Path(eval_dir) / env_id / split
        episodes = sorted(list(data_path.glob("*episode_*.pkl")))
        with open(episodes[episode_idx], "rb") as f:
            episode = pickle.load(f)
        env_cfg = json.loads(episode["task"]["environment_config"][0].decode("utf-8"))
        return _extract_camera_infos(env_cfg), env_cfg

    if data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from simple.datasets.lerobot import get_episode_lerobot

        dataset = LeRobotDataset(repo_id=env_id, root=data_dir)
        env_cfg, _episode = get_episode_lerobot(dataset, episode_idx)
        return _extract_camera_infos(env_cfg), env_cfg

    raise NotImplementedError(f"Unsupported data_format: {data_format}")


def main(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    host: Annotated[str, typer.Option()] = "localhost",
    port: Annotated[int, typer.Option()] = 5556,
    split: Annotated[str, typer.Option()] = "train",
    data_format: Annotated[str, typer.Option()] = "lerobot",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = True,
    data_dir: Annotated[str, typer.Option()] = "data",
    eval_dir: Annotated[str, typer.Option()] = "data/evals",
    num_episodes: Annotated[int, typer.Option()] = 3,
    episode_start: Annotated[int, typer.Option()] = 0,
    episode_stride: Annotated[int, typer.Option()] = 1,
    max_episode_steps: Annotated[int, typer.Option()] = 200,
    camera_key: Annotated[str, typer.Option()] = "front_stereo_left",
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    out_dir: Annotated[str, typer.Option()] = "artifacts/render_sync_check",
):
    os.makedirs(out_dir, exist_ok=True)

    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")

    if data_format == "rlds_numpy":
        render_hz = 3
    elif data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=env_id, root=data_dir)
        render_hz = int(dataset.meta.fps)
    else:
        raise NotImplementedError(f"Unsupported data_format: {data_format}")

    make_kwargs = dict(
        sim_mode=sim_mode,
        headless=headless,
        max_episode_steps=max_episode_steps,
        render_hz=render_hz,
    )
    env = gym.make(env_id, **make_kwargs)
    task = env.unwrapped.task  # type: ignore
    agent = agent_clazz(task.robot, host, port)

    episode_ids = list(range(episode_start, episode_start + num_episodes * episode_stride, episode_stride))
    for eps_idx in tqdm(episode_ids, desc="sync-check"):
        camera_infos, env_conf = _camera_infos_for_episode(
            data_format=data_format,
            eval_dir=eval_dir,
            env_id=env_id,
            split=split,
            episode_idx=eps_idx,
            data_dir=data_dir,
        )
        obs, info = env.reset(options={"state_dict": env_conf})
        mujoco_eef_body = _resolve_mujoco_eef_body_name(env, task)
        # For G1-like robots where EE is palm link, draw at fingertip position for visual match.
        mujoco_tip_body = None
        if isinstance(mujoco_eef_body, str) and "palm" in mujoco_eef_body:
            mujoco_tip_body = _resolve_mujoco_tip_body_name(env)
        obs_camera_key, proj_camera_name = _resolve_obs_camera_key(obs, camera_infos, camera_key)
        instruction = task.instruction
        agent.reset()

        imgs: list[np.ndarray] = []
        eef_poses: list[np.ndarray] = []
        cam_poses_robot: list[np.ndarray] = []
        done = False
        steps = 0
        early_stop_reason = "unknown"
        last_terminated = False
        last_truncated = False
        while not done:
            imgs.append(obs[obs_camera_key].copy())
            live_cam_pose = _get_live_camera_pose_robot(env, proj_camera_name)
            if live_cam_pose is not None:
                cam_poses_robot.append(live_cam_pose)
            eef_pose = _get_eef_pose(obs, env, task)
            if eef_pose is None:
                eef_pose = _get_mujoco_eef_pose(env, mujoco_eef_body)
            if eef_pose is not None and mujoco_tip_body is not None:
                tip_pose = _get_mujoco_eef_pose(env, mujoco_tip_body)
                if tip_pose is not None and tip_pose.size == 7:
                    eef_pose = eef_pose.copy()
                    eef_pose[:3] = tip_pose[:3]
            if eef_pose is not None:
                eef_poses.append(eef_pose)

            action = agent.get_action(obs, info=info, instruction=instruction)
            obs, _reward, terminated, truncated, info = env.step(action)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            if env_flag("SIMPLE_SYNC_DEBUG_KEYS", default=False):
                print(f"[sync-debug] step={steps} terminated={terminated} truncated={truncated}")
            done = bool(terminated or truncated)
            if done:
                early_stop_reason = "terminated_or_truncated"
            steps += 1
            if steps >= max_episode_steps:
                early_stop_reason = "max_episode_steps"
                break

        if len(imgs) < 1:
            print(
                f"[sync-check] episode={eps_idx} skipped: no frames "
                f"(reason={early_stop_reason}, terminated={last_terminated}, truncated={last_truncated})"
            )
            continue

        if len(eef_poses) < len(imgs):
            print(
                f"[sync-check] episode={eps_idx} skipped: missing eef poses "
                f"({len(eef_poses)}/{len(imgs)})"
            )
            continue

        eef_arr = np.asarray(eef_poses[: len(imgs)], dtype=np.float32)
        use_live_cam = proj_camera_name.startswith("head_") and len(cam_poses_robot) >= len(imgs)
        if use_live_cam:
            cam_arr = np.asarray(cam_poses_robot[: len(imgs)], dtype=np.float32)
            traj_2d = _project_2d_traj_live(camera_infos, eef_arr, cam_arr, cam_name=proj_camera_name)
        else:
            traj_2d = project_2d_traj(camera_infos, eef_arr, cam_name=proj_camera_name)

        if save_video:
            out_path = os.path.join(out_dir, f"episode_{eps_idx}_eef_overlay.mp4")
            _write_overlay_video(out_path, imgs, traj_2d)
            print(f"[sync-check] wrote {out_path}")

    print(f"Saved sync-check artifacts to: {out_dir}")
    try:
        env.close()
    except Exception:
        # Isaac shutdown can be noisy; artifacts are already persisted.
        pass


if __name__ == "__main__":
    typer.run(main)
