# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Write G1 recordings directly in the psi0 LeRobot v2.1 format (the layout of the merged
SIMPLE-decoupled-wbc dataset, see its README / meta/modality.json):

* ``states`` (32): measured hand (left thumb, middle, index; right thumb, index, middle) and arm joints,
  waist roll/pitch/yaw, previous base height command;
* ``action`` (36): hand/arm/waist joint targets in the same order, base height, vx, vy,
  vyaw (decoupled WBC) or the AMO turning flag, target yaw;
* ``observation.{hand,arm,leg}_joints``, ``observation.prev_torso_rpy``, ``observation.prev_height``;
* one H.264 egocentric video per episode.

``states[t]`` is the observation the command ``action[t]`` was computed from. The waist / height state
follows the controller, as in the merged dataset:

=====================  ==========================================  ==========================================
                       ``amo`` (MP tasks, g1_wholebody)             ``decoupled_wbc`` (teleop tasks, g1_sonic)
=====================  ==========================================  ==========================================
``states[28:31]``      ``action[t-1, 28:31]`` (0 at frame 0)        measured waist roll, pitch, yaw
``prev_torso_rpy``     ``states[28:31]``                            ``states[t-1, 28:31]`` (0 at frame 0)
``states[31]``         ``action[t-1, 31]`` (0.75 at frame 0)        ``action[t-1, 31]`` (0.74 at frame 0)
=====================  ==========================================  ==========================================
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

LEFT_LEG_JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
RIGHT_LEG_JOINTS = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_RPY_JOINTS = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
LEFT_ARM_JOINTS = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
RIGHT_ARM_JOINTS = [j.replace("left_", "right_") for j in LEFT_ARM_JOINTS]
# natural hand order (thumb, index, middle), as in G1 joint_names and observation.hand_joints
LEFT_HAND_JOINTS = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint"]
RIGHT_HAND_JOINTS = [j.replace("left_", "right_") for j in LEFT_HAND_JOINTS]

# states[0:28] / action[0:28]: the left hand is thumb, middle, index; the right hand thumb, index, middle
PSI0_UPPER_JOINTS = (
    LEFT_HAND_JOINTS[0:3] + LEFT_HAND_JOINTS[5:7] + LEFT_HAND_JOINTS[3:5]
    + RIGHT_HAND_JOINTS
    + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
)
PSI0_STATE_NAMES = PSI0_UPPER_JOINTS + ["waist_roll", "waist_pitch", "waist_yaw", "prev_base_height_command"]
PSI0_ACTION_NAMES = PSI0_UPPER_JOINTS + WAIST_RPY_JOINTS + [
    "base_height_command", "base_vx_command", "base_vy_command", "base_vyaw_or_turning_flag", "target_yaw_command"]
assert len(PSI0_STATE_NAMES) == 32 and len(PSI0_ACTION_NAMES) == 36

# initial "previous base height command" per controller (see the module docstring)
INITIAL_HEIGHT = {"amo": 0.75, "decoupled_wbc": 0.74}

VIDEO_KEY = "observation.images.egocentric"
CHUNK_SIZE = 1000
NUMERIC_COLUMNS = ["observation.hand_joints", "observation.arm_joints", "observation.leg_joints",
                   "observation.prev_torso_rpy", "observation.prev_height", "states", "action",
                   "timestamp", "frame_index", "episode_index", "index", "next.done", "task_index"]


def modality_dict() -> dict:
    def _entry(start, end, original_key, absolute=True):
        return {"start": start, "end": end, "rotation_type": None, "absolute": absolute,
                "dtype": "float32", "original_key": original_key}

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
        "video": {"rs_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _stats(x: np.ndarray, with_count: bool = False) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    d = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0),
         "q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0)}
    d = {k: v.astype(np.float32).tolist() for k, v in d.items()}
    if with_count:
        d["count"] = [int(x.shape[0])]
    return d


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def _write_json(path: Path, obj, indent=4) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)


class _VideoWriter:
    """Pipe RGB frames to ffmpeg (libx264, yuv420p: the codec of the merged dataset's videos)."""

    def __init__(self, path: Path, height: int, width: int, fps: int):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264",
               # keyframe every 10 frames, no B-frames: training samples random frames, and with
               # x264's default keyint=250 each lookup decodes ~125 frames (~10x slower).
               "-g", "10", "-keyint_min", "10", "-sc_threshold", "0", "-bf", "0",
               "-pix_fmt", "yuv420p", "-f", "mp4", str(path)]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to write psi0 videos but was not found in PATH") from exc
        self.shape = (height, width, 3)

    def write(self, image: np.ndarray) -> None:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        assert image.shape == self.shape, f"video frame shape {image.shape} != {self.shape}"
        assert self.proc.stdin is not None
        self.proc.stdin.write(image.tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path} (exit code {self.proc.returncode})")

    def abort(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.kill()
        self.proc.wait()
        self.path.unlink(missing_ok=True)


class Psi0DatasetWriter:
    """Append episodes to a psi0-format LeRobot v2.1 dataset at ``root`` (created if missing).

    Per episode: ``start_episode()``, ``add_frame(...)`` per control step, then ``save_episode(...)`` or
    ``discard_episode()``. The metadata (info, episodes, stats, ...) is rewritten after every saved
    episode, so the dataset on disk stays loadable if the recording is interrupted.
    """

    def __init__(self, root: str | Path, fps: int, controller: str, robot_type: str = "g1"):
        assert controller in INITIAL_HEIGHT, f"controller must be one of {list(INITIAL_HEIGHT)}, got {controller!r}"
        self.root = Path(root).resolve()
        self.fps = int(fps)
        self.controller = controller
        self.robot_type = robot_type
        meta = self.root / "meta"

        info_path = meta / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            if "states" not in info.get("features", {}):
                raise ValueError(f"{self.root} holds a dataset that is not in the psi0 format; use another --save-dir.")
            if int(info["fps"]) != self.fps:
                raise ValueError(f"{self.root} was recorded at {info['fps']} fps, not {self.fps}.")
            self.video_shape = list(info["features"][VIDEO_KEY]["shape"])
        else:
            self.video_shape = None
        meta.mkdir(parents=True, exist_ok=True)
        self.tasks = _read_jsonl(meta / "tasks.jsonl")
        self.episodes = _read_jsonl(meta / "episodes.jsonl")
        self.episodes_stats = _read_jsonl(meta / "episodes_stats.jsonl")
        # per-episode numeric columns, for the dataset-wide stats.json
        self._columns: list[dict[str, np.ndarray]] = [self._load_columns(i) for i in range(len(self.episodes))]
        self._video: _VideoWriter | None = None
        self._buffer: dict[str, list] | None = None

    # ------------------------------------------------------------------ paths
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return sum(int(e["length"]) for e in self.episodes)

    def _data_path(self, ep: int) -> Path:
        return self.root / f"data/chunk-{ep // CHUNK_SIZE:03d}/episode_{ep:06d}.parquet"

    def _video_path(self, ep: int) -> Path:
        return self.root / f"videos/chunk-{ep // CHUNK_SIZE:03d}/egocentric/episode_{ep:06d}.mp4"

    def _load_columns(self, ep: int) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        table = pq.read_table(self._data_path(ep))
        return {k: np.asarray(table.column(k).to_pylist(), dtype=np.float64).reshape(table.num_rows, -1)
                for k in NUMERIC_COLUMNS}

    # --------------------------------------------------------------- episode
    def start_episode(self) -> None:
        self.discard_episode()
        self._buffer = {k: [] for k in ["states", "action", "observation.hand_joints", "observation.arm_joints",
                                        "observation.leg_joints", "observation.prev_torso_rpy", "observation.prev_height"]}

    @property
    def episode_length(self) -> int:
        return 0 if self._buffer is None else len(self._buffer["action"])

    def add_frame(
        self,
        image: np.ndarray,
        qpos: dict[str, float],
        targets: dict[str, float],
        base_height: float,
        vx: float,
        vy: float,
        vyaw_or_turning_flag: float,
        target_yaw: float,
    ) -> None:
        """One control step: ``image`` / ``qpos`` (joint name -> position) are the observation the command was
        computed from; ``targets`` (joint name -> target) and the base command are what was sent at this step."""
        if self._buffer is None:
            self.start_episode()
        assert self._buffer is not None
        prev_action = self._buffer["action"][-1] if self._buffer["action"] else None
        prev_height = prev_action[31] if prev_action is not None else INITIAL_HEIGHT[self.controller]
        measured_rpy = np.array([qpos[j] for j in WAIST_RPY_JOINTS], dtype=np.float64)
        if self.controller == "amo":
            # AMO does not use waist feedback: its input is the last commanded waist
            waist_state = prev_action[28:31] if prev_action is not None else np.zeros(3)
            prev_torso_rpy = waist_state
        else:
            waist_state = measured_rpy
            prev_torso_rpy = self._buffer["states"][-1][28:31] if self._buffer["states"] else np.zeros(3)

        states = np.concatenate([[qpos[j] for j in PSI0_UPPER_JOINTS], waist_state, [prev_height]])
        action = np.concatenate([
            [targets[j] for j in PSI0_UPPER_JOINTS],
            [targets[j] for j in WAIST_RPY_JOINTS],
            [base_height, vx, vy, vyaw_or_turning_flag, target_yaw],
        ]).astype(np.float64)
        b = self._buffer
        b["states"].append(states.astype(np.float64))
        b["action"].append(action)
        b["observation.hand_joints"].append(np.array([qpos[j] for j in LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS]))
        b["observation.arm_joints"].append(np.array([qpos[j] for j in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS]))
        b["observation.leg_joints"].append(np.array([qpos[j] for j in LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + WAIST_JOINTS]))
        b["observation.prev_torso_rpy"].append(np.asarray(prev_torso_rpy, dtype=np.float64).copy())
        b["observation.prev_height"].append(np.array([prev_height], dtype=np.float64))

        image = np.asarray(image)
        if self._video is None:
            if self.video_shape is None:
                self.video_shape = list(image.shape)
            h, w = self.video_shape[:2]
            self._video = _VideoWriter(self._video_path(self.num_episodes).with_suffix(".tmp.mp4"), h, w, self.fps)
        self._video.write(image)

    def discard_episode(self) -> None:
        if self._video is not None:
            self._video.abort()
            self._video = None
        self._buffer = None

    def save_episode(self, task: str, environment_config: dict | str | None = None) -> int:
        """Write the buffered episode; returns its episode index."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from simple.utils import NumpyArrayEncoder

        assert self._buffer is not None and self.episode_length > 0, "no frames to save"
        assert self._video is not None
        ep, n, start = self.num_episodes, self.episode_length, self.num_frames

        task_index = next((t["task_index"] for t in self.tasks if t["task"] == task), None)
        if task_index is None:
            task_index = len(self.tasks)
            self.tasks.append({"task_index": task_index, "task": task, "category": "", "description": task})

        frame_index = np.arange(n, dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        columns = {k: np.stack(v).astype(np.float64) for k, v in self._buffer.items()}
        table = pa.table({
            **{k: columns[k].tolist() for k in ["states", "action", "observation.hand_joints", "observation.arm_joints",
                                                 "observation.leg_joints", "observation.prev_torso_rpy", "observation.prev_height"]},
            "timestamp": frame_index / float(self.fps),
            "frame_index": frame_index,
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": start + frame_index,
            "task_index": np.full(n, task_index, dtype=np.int64),
            "next.done": done,
        })
        data_path = self._data_path(ep)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        self._video.close()
        self._video.path.replace(self._video_path(ep))
        self._video = None
        self._buffer = None

        if environment_config is not None and not isinstance(environment_config, str):
            environment_config = json.dumps(environment_config, cls=NumpyArrayEncoder)
        self.episodes.append({
            "episode_index": ep,
            "tasks": [task_index],
            "length": n,
            "dataset_from_index": start,
            "dataset_to_index": start + n - 1,
            "robot_type": self.robot_type,
            "instruction": {"task_index": task_index, "task": task},
            "environment_config": environment_config,
        })
        self.episodes_stats.append({"episode_index": ep, "stats": {
            "action": _stats(columns["action"], with_count=True),
            "timestamp": _stats(frame_index / float(self.fps), with_count=True),
        }})
        self._columns.append({**{k: columns[k] for k in columns},
                              "timestamp": (frame_index / float(self.fps)).reshape(-1, 1),
                              "frame_index": frame_index.reshape(-1, 1).astype(np.float64),
                              "episode_index": np.full((n, 1), ep, dtype=np.float64),
                              "index": (start + frame_index).reshape(-1, 1).astype(np.float64),
                              "task_index": np.full((n, 1), task_index, dtype=np.float64),
                              "next.done": done.reshape(-1, 1).astype(np.float64)})
        self._write_meta()
        return ep

    def close(self) -> None:
        self.discard_episode()

    # ------------------------------------------------------------------ meta
    def _write_meta(self) -> None:
        meta = self.root / "meta"
        _write_jsonl(meta / "tasks.jsonl", self.tasks)
        _write_jsonl(meta / "episodes.jsonl", self.episodes)
        _write_jsonl(meta / "episodes_stats.jsonl", self.episodes_stats)
        stats = {k: _stats(np.concatenate([c[k] for c in self._columns])) for k in NUMERIC_COLUMNS}
        _write_json(meta / "stats.json", stats)
        _write_json(meta / "stats_psi0.json", stats)
        for name in ("relative_stats.json", "lang_map.json"):
            if not (meta / name).exists():
                (meta / name).write_text("{}")
        _write_json(meta / "modality.json", modality_dict(), indent=2)
        n_eps = self.num_episodes
        _write_json(meta / "info.json", {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": n_eps,
            "total_frames": self.num_frames,
            "total_tasks": len(self.tasks),
            "total_videos": n_eps,
            "total_chunks": math.ceil(n_eps / CHUNK_SIZE),
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            "features": {
                VIDEO_KEY: {"dtype": "video", "shape": self.video_shape, "names": ["height", "width", "channel"],
                            "video_info": {"video.fps": float(self.fps), "video.codec": "h264",
                                           "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                                           "has_audio": False}},
                "observation.hand_joints": {"dtype": "float32", "shape": [14], "names": ["hand_joints"]},
                "observation.arm_joints": {"dtype": "float32", "shape": [14], "names": ["arm_joints"]},
                "observation.leg_joints": {"dtype": "float32", "shape": [15], "names": ["leg_joints"]},
                "observation.prev_torso_rpy": {"dtype": "float32", "shape": [3], "names": ["prev_roll", "prev_pitch", "prev_yaw"]},
                "observation.prev_height": {"dtype": "float32", "shape": [1], "names": ["prev_height"]},
                "states": {"dtype": "float32", "shape": [-1], "names": PSI0_STATE_NAMES},
                "action": {"dtype": "float32", "shape": [-1], "names": PSI0_ACTION_NAMES},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        })

