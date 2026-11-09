# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from simple.core.types import Pose
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import math

SUPPORTED_MOUNTS = ("eye_in_hand", "eye_on_base", "eye_in_head")


def _require(cond: bool, cfg: Any, msg: str) -> None:
    if not cond:
        uid = getattr(cfg, "uid", "?")
        raise ValueError(f"Invalid {type(cfg).__name__} (uid={uid!r}): {msg}")


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class SensorCfg:
    uid: str
    mount: str
    # parent_link: str

    def __post_init__(self):
        _require(isinstance(self.uid, str) and self.uid != "", self, "uid must be a non-empty string")
        _require(
            self.mount in SUPPORTED_MOUNTS, self,
            f"unsupported mount {self.mount!r}; expected one of {SUPPORTED_MOUNTS}",
        )

    @property
    def observation_space(self) -> spaces.Space:
        ...

@dataclass
class CameraCfg(SensorCfg):

    # uid: str 
    
    width: int
    height: int

    fov: float
    near: float
    far: float

    focal_length: float

    pose: dict

    intrinsics: dict[str, Any] | None = field(default=None, kw_only=True)
    depth_intrinsics: dict[str, Any] | None = field(default=None, kw_only=True)
    depth_to_color_extrinsics: dict[str, Any] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        for name in ("width", "height"):
            v = getattr(self, name)
            _require(
                isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0,
                self, f"{name} must be a positive integer, got {v!r}",
            )
        _require(
            _is_finite_number(self.fov) and 0 < float(self.fov) < math.pi, self,
            f"fov must be in radians within (0, pi), got {self.fov!r}",
        )
        _require(
            _is_finite_number(self.focal_length) and float(self.focal_length) > 0, self,
            f"focal_length must be > 0, got {self.focal_length!r}",
        )
        _require(
            _is_finite_number(self.near) and _is_finite_number(self.far)
            and 0 < float(self.near) < float(self.far), self,
            f"clipping range must satisfy 0 < near < far, got near={self.near!r}, far={self.far!r}",
        )
        _require(isinstance(self.pose, dict), self, f"pose must be a dict, got {type(self.pose).__name__}")
        for key, length in (("position", 3), ("quaternion", 4)):
            v = self.pose.get(key)
            if v is not None:  # quaternion may be filled in later from robot defaults
                _require(
                    len(v) == length and all(_is_finite_number(x) for x in v), self,
                    f"pose[{key!r}] must be {length} finite numbers, got {v!r}",
                )
        for name in ("intrinsics", "depth_intrinsics"):
            intr = getattr(self, name)
            if intr is None:
                continue
            _require(isinstance(intr, dict), self, f"{name} must be a dict or None")
            for k in ("fx", "fy"):
                if intr.get(k) is not None:
                    _require(
                        _is_finite_number(intr[k]) and float(intr[k]) > 0, self,
                        f"{name}[{k!r}] must be > 0, got {intr[k]!r}",
                    )
            for k in ("cx", "cy"):
                if intr.get(k) is not None:
                    _require(_is_finite_number(intr[k]), self, f"{name}[{k!r}] must be a finite number")


    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

    @property
    def resolution(self) -> tuple[int, int]: # [W, H]
        return (self.width, self.height)
    
    @property
    def fy(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("fy") is not None:
            return float(self.intrinsics["fy"])
        fx = self.width / (2 * np.tan(self.fov / 2))
        fy = fx
        return fy

    @property
    def fx(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("fx") is not None:
            return float(self.intrinsics["fx"])
        return self.width / (2 * np.tan(self.fov / 2))

    @property
    def cx(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("cx") is not None:
            return float(self.intrinsics["cx"])
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("cy") is not None:
            return float(self.intrinsics["cy"])
        return 0.5 * self.height
    
    @property
    def position(self) -> list[float]:
        if 'position' in self.pose:
            return self.pose['position']

    @property
    def quaternion(self) -> list[float]:
        if 'quaternion' in self.pose:
            return self.pose['quaternion']

@dataclass
class StereoCameraCfg(CameraCfg):

    baseline: float

    def __post_init__(self):
        super().__post_init__()
        _require(
            _is_finite_number(self.baseline) and float(self.baseline) > 0, self,
            f"baseline must be > 0, got {self.baseline!r}",
        )

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({
            f"left": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
            f"right": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)
        })
