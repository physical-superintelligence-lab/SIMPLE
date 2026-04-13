"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from simple.core.scene import TabletopScene
from simple.core.asset import Asset

class ShowHouse(TabletopScene):

    def __init__(self, table: Asset) -> None:
        self.uid = "show_house"
        self.table = table