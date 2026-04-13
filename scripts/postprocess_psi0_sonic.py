#!/usr/bin/env python3
import argparse
import subprocess
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import glob

from simple.robots.g1_sonic import WHOLE_BODY_JOINTS

# see WHOLE_BODY_JOINTS in g1_sonic.py 
STATE_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]

ACTION_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
    ("torso_rp", 13, 15), # waist roll/pitch
    ("torso_y", 12, 13),  # waist yaw
]


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"), default=_json_default)
            f.write("\n")


def modality_dict():
    def _entry(start, end, original_key, absolute=True):
        return {
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": absolute,
            "dtype": "float32",
            "original_key": original_key,
        }

    return {
        "state": {
            "left_hand": _entry(0, 7, "states"),
            "right_hand": _entry(7, 14, "states"),
            "left_arm": _entry(14, 21, "states"),
            "right_arm": _entry(21, 28, "states"),
            "rpy": _entry(28, 31, "states"),
            "height": _entry(31, 32, "states"),
        },
        "action": {
            "left_hand": _entry(0, 7, "action"),
            "right_hand": _entry(7, 14, "action"),
            "left_arm": _entry(14, 21, "action"),
            "right_arm": _entry(21, 28, "action"),
            "rpy": _entry(28, 31, "action"),
            "height": _entry(31, 32, "action"),
            "torso_vx": _entry(32, 33, "action", absolute=False),
            "torso_vy": _entry(33, 34, "action", absolute=False),
            "torso_vyaw": _entry(34, 35, "action", absolute=False),
            "target_yaw": _entry(35, 36, "action"),
        },
        "video": {
            "rs_view": {
                "original_key": "observation.images.egocentric"
            }
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        },
    }

def build_vectors(proprio, history_cmd, cmd, action):
    simple_order_states = np.concatenate([proprio[jName][None,...] for jName in WHOLE_BODY_JOINTS], axis=0).T
    simple_order_actions = np.concatenate([action[jName][None,...] for jName in WHOLE_BODY_JOINTS], axis=0).T
    waist_rpy = np.concatenate([proprio[jName][None,...] 
                                for jName in ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]], axis=0).T

    # states: match to_psi0_state_format ordering
    states = np.concatenate(
        [simple_order_states[:, s:e] for _, s, e in STATE_SLICES] + [
            waist_rpy,  # torso_rpy -> waist roll/pitch/yaw 
            history_cmd[:, 4:5] # base height
        ],
        axis=1,
    ).astype(np.float32)

    # actions: match to_psi0_action_format ordering
    actions = np.concatenate(
        [simple_order_actions[:, s:e] for _, s, e in ACTION_SLICES]
        + [
            cmd[:, 4:5],  # base height
            cmd[:, 0:4],  # vx, vy, vyaw, target_yaw
        ],
        axis=1,
    ).astype(np.float32)
    return states, actions


def build_proprio_obs(proprio, history_cmd):
    # proprio: see WHOLE_BODY_JOINTS in g1_sonic.py 
    # history_cmd: [vx, vy, vyaw, target_yaw, base_height] from previous timestep

    simple_order_proprio = np.concatenate([proprio[jName][None,...] for jName in WHOLE_BODY_JOINTS], axis=0).T

    # hand joints: left thumb(29:32), left index(32:34), left middle(34:36),
    #              right thumb(36:39), right index(39:41), right middle(41:43)
    hand = np.concatenate(
        [
            simple_order_proprio[:, 29:32],
            simple_order_proprio[:, 32:34],
            simple_order_proprio[:, 34:36],
            simple_order_proprio[:, 36:39],
            simple_order_proprio[:, 39:41],
            simple_order_proprio[:, 41:43],
        ],
        axis=1,
    ).astype(np.float32)

    # arm joints: left arm(15:22), right arm(22:29)
    arm = np.concatenate([simple_order_proprio[:, 15:22], simple_order_proprio[:, 22:29]], axis=1).astype(
        np.float32
    )

    # leg joints: 12 leg joints + 3 waist joints (12:15)
    leg = np.concatenate([simple_order_proprio[:, 0:12], simple_order_proprio[:, 12:15]], axis=1).astype(
        np.float32
    )
    
    # torso rpy -> waist rpy
    waist_rpy = np.concatenate([proprio[jName][None,...] 
        for jName in ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
    ], axis=0).T
    prev_waist_rpy = np.concatenate([
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32), # default for the first step
        waist_rpy[:-1], # one-step history of waist rpy
    ], axis=0).astype(np.float32)
    
    # base height from command
    prev_height = history_cmd[:, 4:5].astype(np.float32)

    return hand, arm, leg, prev_waist_rpy, prev_height


def stats_block(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


# default history command for the initial state
initial_command = np.array([0, 0, 0, 0, 0, 0, 0.75, 0.75, 0.75], dtype=np.float32) 
default_fps=50


def write_downsampled_video(src_video: Path, dst_video: Path, skip: int, downsample: int, fps: int):
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if downsample <= 0:
        raise ValueError(f"downsample must be > 0, got {downsample}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    if skip == 0 and downsample == 1:
        shutil.copyfile(src_video, dst_video)
        return

    vf = (
        f"select='gte(n\\,{skip})*not(mod(n-{skip}\\,{downsample}))',"
        f"setpts=N/({fps}*TB)"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-an",
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst_video),
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to write downsampled videos but was not found in PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for source video: {src_video} -> {dst_video} (exit code {result.returncode})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", required=True, help="Path or glob pattern, e.g. data/datagen*/simple/G1WholebodyBendPick-v0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--total_episodes", type=int, default=99)
    parser.add_argument("--fps", type=int, default=default_fps)
    parser.add_argument("--video-key", default="observation.images.ego_view")
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()

    # last_episode_idx = 0
    episode_idx = 0
    last_index = 0
    all_tasks = []

    total_frames = 0
    episodes = []
    episode_stats_rows = []

    all_states = []
    all_actions = []
    all_timestamp = []
    all_frame_index = []
    all_episode_index = []
    all_index = []
    all_task_index = []
    all_done = []

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    print("created output directories at", out_dir)

    sim_info = None

    all_sim_roots = sorted(Path(p).resolve() for p in glob.glob(args.sim_root))
    for sim_root in all_sim_roots:
        if episode_idx >= args.total_episodes:
            print(f"Reached total_episodes={args.total_episodes}, stopping further processing.")
            break

        print(f"Merging data: {sim_root}")

        sim_info = json.loads((sim_root / "meta" / "info.json").read_text())
        sim_tasks = load_jsonl(sim_root / "meta" / "tasks.jsonl")
        episodes_info = load_jsonl(sim_root / "meta" / "episodes.jsonl")

        original_fps = float(sim_info["fps"])
        if args.fps / original_fps != args.downsample:
            print(f"Warning: The specified fps {args.fps} is not consistent with the original fps {original_fps} and downsample factor {args.downsample}." 
                  f"The timestamps will be computed based on the specified fps.")

        curr_task_index_to_new_task_index = {}

        def merge_task(task, all_tasks):
            for t in all_tasks:
                if t["task"] == task:
                    return t["task_index"]
            all_tasks.append({
                "task_index": len(all_tasks),
                "task": task,
            })
            return len(all_tasks) -1

        for t in sim_tasks:
            task_index = t["task_index"]
            task = t["task"]
            curr_task_index_to_new_task_index[task_index] = merge_task(task, all_tasks)

        data_files = sorted((sim_root / "data").glob("chunk-*/episode_*.parquet"))
        for data_path in tqdm(data_files): # for each episode
            if episode_idx >= args.total_episodes:
                print(f"Reached total_episodes={args.total_episodes}, stopping further processing.")
                break

            ep_index = int(data_path.stem.split("_")[-1]) 
            chunk_id = episode_idx // args.chunks_size

            table = pq.read_table(data_path)
            proprio = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            state_joint_names = sim_info["features"]["observation.state"]["names"]
            proprio = dict(zip(state_joint_names, proprio.T)) # dict of (joint_name -> [values over time])

            cmd = np.concatenate([
                np.asarray(table["teleop.navigate_command"].to_pylist(), dtype=np.float32),
                np.asarray(table["teleop.base_height_command"].to_pylist(), dtype=np.float32)[:, None]
            ], axis=1) # [vx, vy, vyaw, target_yaw, base_height]

            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            action_joint_names = sim_info["features"]["action"]["names"]
            action = dict(zip(action_joint_names, action.T)) # dict of (joint_name -> [values over time])

            default_cmd = np.array([0, 0, 0, 0, 0.74], dtype=np.float32) # vx, vy, vyaw, target_yaw, base_height
            history_cmd = np.concatenate([default_cmd[None, :], cmd[:-1]], axis=0) # one-step history of command

            # history_cmd = np.concatenate([initial_command[None, :], cmd[:-1]], axis=0)
            states, actions = build_vectors(proprio, history_cmd, cmd, action)
            hand_joints, arm_joints, leg_joints, prev_torso_rpy, prev_height = build_proprio_obs(
                proprio, history_cmd
            )

            m = states.shape[0]
            try:
                assert m >= args.skip, f"Episode {episode_idx} has only {m} frames, which is less than skip={args.skip}"
            except AssertionError:
                continue
            
            n = (m - args.skip) // args.downsample
                
            done = np.zeros((n,), dtype=bool)
            if n > 0:
                done[-1] = True

            # frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
            frame_index = np.asarray(range(n), dtype=np.int64)

            # episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            episode_index = np.asarray([episode_idx]*n, dtype=np.int64)

            # index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            index = np.asarray(table["index"].to_pylist(), dtype=np.int64) + last_index
            
            # timesteps = np.asarray([round(ts * original_fps) for ts in table["timestamp"].to_pylist()], dtype=np.float32)
            timestamp = frame_index * 1.0 / args.fps 

            curr_task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            assert np.all(curr_task_indices == curr_task_indices[0]), f"Episode {episode_idx} has multiple task indices: {set(curr_task_indices)}"
            new_task_index = curr_task_index_to_new_task_index[curr_task_indices[0]]
            task_index = np.asarray([new_task_index]*n, dtype=np.int64)

            out_table = pa.table({
                "states": states[args.skip:][::args.downsample].tolist(),
                "action": actions[args.skip:][::args.downsample].tolist(),
                "observation.hand_joints": hand_joints[args.skip:][::args.downsample].tolist(),
                "observation.arm_joints": arm_joints[args.skip:][::args.downsample].tolist(),
                "observation.leg_joints": leg_joints[args.skip:][::args.downsample].tolist(),
                "observation.prev_torso_rpy": prev_torso_rpy[args.skip:][::args.downsample].tolist(),
                "observation.prev_height": prev_height[args.skip:][::args.downsample].tolist(),
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": index[args.skip:][::args.downsample],
                "task_index": task_index,
                "next.done": done,
            })

            out_data_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
            out_data_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(out_table, out_data_dir / f"episode_{episode_idx:06d}.parquet")

            src_chunk = data_path.parent.name
            src_episode_idx = int(data_path.stem.split("_")[-1])
            src_video = sim_root / "videos" / src_chunk / args.video_key / f"episode_{src_episode_idx:06d}.mp4"
            dst_video_dir = out_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_video_dir / f"episode_{episode_idx:06d}.mp4"
            if src_video.exists():
                write_downsampled_video(
                    src_video=src_video,
                    dst_video=dst_video,
                    skip=args.skip,
                    downsample=args.downsample,
                    fps=args.fps,
                )

            total_frames += n
            ep_task = task_index[0] # ) if len(task_index) else 0
            episodes.append({
                "episode_index": episode_idx,
                "tasks": [ep_task],
                "length": n,
                "dataset_from_index": total_frames - n,
                "dataset_to_index": total_frames - 1,
                "robot_type": "g1",
                "instruction": all_tasks[task_index[0]],
                "environment_config": episodes_info[ep_index]["environment_config"]
            })

            ep_stats = {
                "episode_index": episode_idx,
                "stats": {
                    "action": {**stats_block(actions), "count": [int(n)]},
                    "timestamp": {**stats_block(timestamp), "count": [int(n)]},
                },
            }
            episode_stats_rows.append(ep_stats)

            all_states.append(states)
            all_actions.append(actions)
            all_timestamp.append(timestamp)
            all_frame_index.append(frame_index)
            all_episode_index.append(episode_index)
            all_index.append(index)
            all_task_index.append(task_index)
            all_done.append(done.astype(np.float32))

            last_index += n
            episode_idx += 1

    all_states = np.concatenate(all_states, axis=0) if all_states else np.zeros((0, 32), dtype=np.float32)
    all_actions = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 36), dtype=np.float32)
    all_timestamp = np.concatenate(all_timestamp, axis=0) if all_timestamp else np.zeros((0,), dtype=np.float32)
    all_frame_index = np.concatenate(all_frame_index, axis=0) if all_frame_index else np.zeros((0,), dtype=np.float32)
    all_episode_index = np.concatenate(all_episode_index, axis=0) if all_episode_index else np.zeros((0,), dtype=np.float32)
    all_index = np.concatenate(all_index, axis=0) if all_index else np.zeros((0,), dtype=np.float32)
    all_task_index = np.concatenate(all_task_index, axis=0) if all_task_index else np.zeros((0,), dtype=np.float32)
    all_done = np.concatenate(all_done, axis=0) if all_done else np.zeros((0,), dtype=np.float32)

    task_by_index = {}
    tasks_rows = []
    for t in all_tasks:
        ti = t.get("task_index", 0)
        task_by_index[int(ti)] = t.get("task", "")
        tasks_rows.append({
            "task_index": int(ti),
            "task": t.get("task", ""),
            "category": "",
            "description": t.get("task", ""),
        })

    meta_dir = out_dir / "meta"
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    print("Wrote tasks.jsonl with", len(tasks_rows), "tasks")
    write_jsonl(meta_dir / "episodes.jsonl", sorted(episodes, key=lambda r: r["episode_index"]))
    print("Wrote episodes.jsonl with", len(episodes), "episodes")
    write_jsonl(meta_dir / "episodes_stats.jsonl", sorted(episode_stats_rows, key=lambda r: r["episode_index"]))
    print("Wrote episodes_stats.jsonl")

    assert sim_info is not None
    video_feat = sim_info["features"][args.video_key]

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks_rows),
        "total_videos": len(episodes),
        "total_chunks": math.ceil(len(episodes) / args.chunks_size) if args.chunks_size else 1,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": video_feat.get("dtype", "video"),
                "shape": video_feat.get("shape", [360, 640, 3]),
                "names": ["height", "width", "channel"],
                "video_info": video_feat.get("info", video_feat.get("video_info", {})),
            },
            "observation.hand_joints": {"dtype": "float32", "shape": [14], "names": ["hand_joints"]},
            "observation.arm_joints": {"dtype": "float32", "shape": [14], "names": ["arm_joints"]},
            "observation.leg_joints": {"dtype": "float32", "shape": [15], "names": ["leg_joints"]},
            "observation.prev_torso_rpy": {"dtype": "float32", "shape": [3], "names": ["prev_roll", "prev_pitch", "prev_yaw"]},
            "observation.prev_height": {"dtype": "float32", "shape": [1], "names": ["prev_height"]},
            "states": {"dtype": "float32", "shape": [-1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    print("Wrote info.json")

    stats = {
        "states": stats_block(all_states),
        "action": stats_block(all_actions),
        "timestamp": stats_block(all_timestamp),
        "frame_index": stats_block(all_frame_index),
        "episode_index": stats_block(all_episode_index),
        "index": stats_block(all_index),
        "task_index": stats_block(all_task_index),
        "next.done": stats_block(all_done),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    print("Wrote stats.json")
    (meta_dir / "stats_psi0.json").write_text(json.dumps(stats, indent=4))
    print("Wrote stats_psi0.json")
    (meta_dir / "relative_stats.json").write_text("{}")
    print("Wrote relative_stats.json")
    (meta_dir / "lang_map.json").write_text("{}")
    print("Wrote lang_map.json")
    (meta_dir / "modality.json").write_text(json.dumps(modality_dict(), indent=2))
    print("Wrote modality.json")


if __name__ == "__main__":
    main()
