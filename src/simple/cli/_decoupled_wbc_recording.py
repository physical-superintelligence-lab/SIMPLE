# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path


EGO_VIEW_FEATURE = "observation.images.ego_view"


def _normalize_shape(ego_view_shape) -> list[int] | None:
    if ego_view_shape is None:
        return None

    shape = [int(dim) for dim in ego_view_shape]
    if len(shape) != 3:
        raise ValueError(f"Expected ego-view image shape to be HWC, got {tuple(shape)}")
    return shape


def set_ego_view_feature_shape(features: dict, ego_view_shape) -> None:
    """Match the LeRobot video metadata to the actual egocentric image shape."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    features[EGO_VIEW_FEATURE]["shape"] = shape


def validate_existing_ego_view_feature_shape(save_dir: str | Path, ego_view_shape) -> None:
    """Refuse to append 480p frames to a dataset initialized with old 360p metadata."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    info_path = Path(save_dir) / "meta" / "info.json"
    if not info_path.exists():
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    existing_shape = info.get("features", {}).get(EGO_VIEW_FEATURE, {}).get("shape")
    if existing_shape is None:
        return

    existing_shape = [int(dim) for dim in existing_shape]
    if existing_shape != shape:
        raise ValueError(
            f"Existing LeRobot dataset at {save_dir} declares {EGO_VIEW_FEATURE} "
            f"shape {existing_shape}, but the current ego-view frame is {shape}. "
            "Use a clean --save-dir or remove the old dataset before recording."
        )


def create_gr00t_exporter(**create_kwargs):
    """``Gr00tDataExporter.create`` with shape-(1,) features saved as scalars.

    LeRobot stores shape-(1,) features as scalar hf ``Value`` columns but buffers
    them as (1,) arrays; numpy>=2.5 refuses ``float()`` on those, so
    ``save_episode`` fails.  Flatten (N, 1) -> (N,) just for the parquet table
    (episode stats still see the (N, 1) arrays, as before).
    """
    from decoupled_wbc.data.exporter import Gr00tDataExporter

    class _Exporter(Gr00tDataExporter):
        def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
            episode_buffer = dict(episode_buffer)
            for key, ft in self.features.items():
                if ft["dtype"] not in ("image", "video") and tuple(ft["shape"]) == (1,):
                    episode_buffer[key] = episode_buffer[key].reshape(-1)
            super()._save_episode_table(episode_buffer, episode_index)

    return _Exporter.create(**create_kwargs)


# SONIC whole-body "token": the 64-dim encoder output / decoder input that
# drives the 29 body joints.  WBC teleop records it (+ the 14 hand joints) as the
# LeRobot `action` instead of the 43 decoded joint targets.  Fixed at 64 by every
# SONIC release config (encoder: dimension: 64).
WBC_TOKEN_DIM = 64


def apply_wbc_token_action(
    features: dict,
    modality_config: dict,
    joint_names: list[str],
    token_dim: int = WBC_TOKEN_DIM,
) -> None:
    """Redefine the LeRobot ``action`` as [SONIC token(token_dim) | hands(14)].

    Applied ONLY in the WBC recorder/replay (teleop_wbc / replay_wbc), never in
    decoupled_wbc.get_dataset_features, so the decoupled teleop/replay pipeline
    keeps its 43-DOF joint action.  Both the WBC recorder and its replay call
    this so the two exporters declare the identical action schema.

    ``joint_names`` is WHOLE_BODY_JOINTS; its last 14 entries are the L-hand(7)
    + R-hand(7) names, reused verbatim so the hand block matches how the
    decoupled action already encoded the hands.
    """
    hand_names = list(joint_names[-14:])
    features["action"] = {
        "dtype": "float64",
        "shape": (token_dim + 14,),
        "names": [f"token_{i}" for i in range(token_dim)] + hand_names,
    }
    # The token dims are one opaque block; hands keep per-joint slices at the new
    # offsets.  The eef / base_height / navigate entries reference OTHER fields
    # via original_key and are unchanged.  (Confirm the layout GR00T expects.)
    modality_config["action"] = {
        "token": {"start": 0, "end": token_dim},
        "left_hand": {"start": token_dim, "end": token_dim + 7},
        "right_hand": {"start": token_dim + 7, "end": token_dim + 14},
        "left_wrist_pos": {"start": 0, "end": 3, "original_key": "action.eef"},
        "left_wrist_abs_quat": {
            "start": 3, "end": 7, "original_key": "action.eef",
            "rotation_type": "quaternion",
        },
        "right_wrist_pos": {"start": 7, "end": 10, "original_key": "action.eef"},
        "right_wrist_abs_quat": {
            "start": 10, "end": 14, "original_key": "action.eef",
            "rotation_type": "quaternion",
        },
        "base_height_command": {
            "start": 0, "end": 1, "original_key": "teleop.base_height_command",
        },
        "navigate_command": {"start": 0, "end": 4, "original_key": "teleop.navigate_command"},
    }
