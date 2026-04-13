"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from .actor import Actor
from .asset import Asset, ArticulatedAsset
from .types import Pose, Vec3, Vec7, GraspPose
from typing import List, Protocol, runtime_checkable

class Object (Actor):
    
    asset: Asset

    def __init__(self, asset: Asset) -> None:
        self.asset = asset

class ArticulatedObject(Actor):
    asset: ArticulatedAsset
    def __init__(self, asset: ArticulatedAsset) -> None:
        self.asset = asset
@runtime_checkable
class SemanticAnnotated(Protocol):
    label: str
    name: str
    description: str | None

@runtime_checkable
class SpatialAnnotated(Protocol):
    """Spatial annotation for the object."""

    keypoints: dict[str, Vec3]
    axes: dict[str, Vec3]
    stable_poses: list #[Vec7] #: np.ndarray #dict[str, Pose]

# class Graspable:
    canonical_grasps: List# [GraspPose]
    functional_grasps: dict#[str, GraspPose]

# class 