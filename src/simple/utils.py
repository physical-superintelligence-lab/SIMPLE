"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import importlib.resources as res
import json
import os
import re
import subprocess
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from importlib.resources import as_file
from json import JSONEncoder

import numpy as np
from huggingface_hub import snapshot_download


def resolve_res_path(rel_path=None) -> str:
    res_dir = get_res_dir()
    if not rel_path:
        return res_dir
    with as_file(res_dir / rel_path) as res_path:
        if not res_path.exists():
            raise FileNotFoundError(res_path)
    return str(res_path)


def get_res_dir() -> str:
    return res.files("simple") / "resources"


def get_data_dir() -> str:
    return res.files("simple").parent.parent / "data"  # type: ignore


def _parse_zip_file_from_rel_path(rel_path: str) -> str:
    rel_path = rel_path.rstrip("/")

    pattern = r"^vMaterials_2/(?:[^/]+/)*[^/]+\.mdl$"
    match = re.match(pattern, rel_path)
    if match:
        return "vMaterials_2.zip"

    pattern = r"^robots/([^/]+)/.+\.(?:yml|xml)$"
    match = re.match(pattern, rel_path)
    if match:
        robot_name = match.group(1)
        return f"robots_{robot_name}.zip"

    pattern = r"^scenes/([^/]+)/([^/]+)$"
    match = re.match(pattern, rel_path)
    if match:
        scene_category = match.group(1)
        scene_name = match.group(2)
        return f"scenes_{scene_category}_{scene_name}.zip"

    pattern = r"^assets/graspnet/(.*)$"
    match = re.match(pattern, rel_path)
    if match:
        return f"assets_graspnet.zip"

    pattern = r"^assets/([^/]+)/([^/]+)/([^/]+)/(.+)$"

    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        dex_category = match.group(2)
        hand_uid = match.group(3)
        asset_label = match.group(4)

        return f"assets_{asset_category}_{dex_category}_{hand_uid}_{asset_label}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)$"
    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        asset_name = match.group(2)
        return f"assets_{asset_category}_{asset_name}.zip"

    pattern = r"^assets/([^/]+)/(.+)$"
    match = re.match(pattern, rel_path)

    if match:
        asset_category = match.group(1)
        asset_name = match.group(2).replace("/", "_")
        return f"assets_{asset_category}.zip"

    raise FileNotFoundError(f"{rel_path}")


def resolve_data_path(
    rel_path=None, create_if_not_exist=False, auto_download=False
) -> str:
    data_dir = get_data_dir()
    if not rel_path:
        return data_dir
    with as_file(data_dir / rel_path) as res_path:
        # assert res_path.exists(), f"Data not found: {res_path}"
        if not res_path.exists():
            if not auto_download:
                raise FileNotFoundError(res_path)

            if create_if_not_exist:
                os.makedirs(res_path, exist_ok=True)
            else:
                zip_file = _parse_zip_file_from_rel_path(rel_path)
                zip_path = os.path.join(data_dir, zip_file)
                snapshot_download(
                    repo_id="USC-PSI-Lab/SIMPLE",
                    allow_patterns=[zip_file],
                    local_dir=data_dir,
                    repo_type="dataset",
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
                )

                if not os.path.exists(zip_path):
                    raise FileNotFoundError(
                        f"Download did not materialize {zip_path} for {rel_path}"
                    )
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    print(f"Deleted {zip_path}")
                return str(res_path)

    return str(res_path)


def is_ffmpeg_installed():
    try:
        # Run 'ffmpeg -version' and suppress output
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# def resolve_output_path(rel_path = None) -> str:


def timestamp_str() -> str:
    now = datetime.now()
    time_prefix = now.isoformat().replace(":", "-").replace(".", "-")
    return time_prefix


def class_to_str(cls: type) -> str:
    """Get fully qualified class path: module + class name."""
    return f"{cls.__module__}.{cls.__qualname__}"


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


def dump_json(obj, fp, **kwargs):
    return json.dump(obj, fp, cls=NumpyArrayEncoder, **kwargs)


def snake_to_pascal(snake_str: str) -> str:
    parts = re.split(r"[-_]", snake_str)
    return "".join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_yaml(file_path) -> dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """

    import re

    import yaml
    from yaml import SafeLoader as Loader

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params


def is_valid_rotation_matrix(R, tol=1e-6):
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=tol)
        and np.isclose(np.linalg.det(R), 1.0, atol=tol)
    )


class Timer:
    """
    Timer utility. Usage:

        timer = Timer()
        with timer("foo"):
            do_something()

        timer.tick("bar")
        do_something_else()
        timer.tock("bar")

        timer.get_average_times() -> {"foo": 0.1, "bar": 0.2}
    """

    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, key):
        self.tick(key)
        try:
            yield None
        finally:
            self.tock(key)

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        """Do nothing when key is already ticking."""
        if key not in self.start_times:
            self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def get_time(self, key, reset=True):
        ret = self.times[key]
        if reset:
            self.times[key] = 0
            self.counts[key] = 0
        return ret

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
