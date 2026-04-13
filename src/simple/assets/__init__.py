"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""


from .asset_manager import AssetManager

from .graspnet import Graspnet1BAssetManager
from .objects import ObjectsAssetManager
from .objaverse import ObjaverseAssetManager
from .primitive import PrimitiveAsseManager, Primitive, Box
from .articulate import ArticulatedAssetManager
