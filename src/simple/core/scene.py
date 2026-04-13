"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.core.asset import Asset

class Scene:
    uid: str
    data_dir: str
    name: str

    def to_dict(self):
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                return {k: _convert(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_convert(v) for v in obj)
            else:
                return obj

        return _convert(self)

class TabletopScene(Scene):
    table: Asset
    table2: Asset | None = None