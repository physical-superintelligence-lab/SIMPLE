"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Protocol, Any, Dict, List, Union

from abc import ABC
import numpy as np

import importlib.resources as res
from importlib.resources import as_file

class Simulator(ABC):

    def update_layout(self) -> None: ...

    def set_states(self, states: Dict[str, Any]) -> None: ...

    def get_states(self) -> Dict[str, Any]: ...

    def step(self) -> Dict[str, np.ndarray] | None: ...

    def render(self, *args, **kwargs) -> Dict[str, np.ndarray]: ...

    def resolve_res_path(self,rel_path=None) -> str:
        res_dir = self.get_res_dir()
        if not rel_path:
            return res_dir
        with as_file(res_dir / rel_path) as res_path:
            if not res_path.exists():
                raise FileNotFoundError(res_path)
        return str(res_path)
    def get_res_dir(self) -> str:
        return res.files("simple") / "resources" # type: ignore

    def get_data_dir(self) -> str:
        return res.files("simple").parent.parent / "data" # type: ignore

    def resolve_data_path(self,rel_path = None, create_if_not_exist=False) -> str:
        data_dir = self.get_data_dir()
        if not rel_path:
            return data_dir
        with as_file(data_dir / rel_path) as res_path:
            # assert res_path.exists(), f"Data not found: {res_path}"
            if not res_path.exists():
                if create_if_not_exist:
                    os.makedirs(res_path, exist_ok=True)
                    
                else:
                    raise FileNotFoundError(res_path)
        return str(res_path)

        