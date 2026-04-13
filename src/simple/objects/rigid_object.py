"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from pathlib import Path

class RigidObject:

    usd_path: Path
    visual_mesh_path: Path
    collision_mesh_path: Path | None

