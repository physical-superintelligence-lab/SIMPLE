# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import json

def get_episode_lerobot(dataset, eps_idx, data_format=None, load_videos=True):
    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
    to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
    if load_videos:
        episode = [dataset[i] for i in range(from_idx, to_idx)]
    else:
        # parquet columns only: dataset[i] decodes every video frame, which dominates loading
        episode = [dataset.hf_dataset[i] for i in range(from_idx, to_idx)]

    env_conf = json.loads(dataset.meta.episodes[eps_idx]['environment_config'])
    env_conf["dr_state_dict"]["scene"]["uid"] = env_conf["dr_state_dict"]["scene"]["uid"].replace("102344280", "scene3") # FIXME
    # import pickle; pickle.dump(env_conf, open(f"env_conf_{eps_idx}.pkl", "wb"))
    # import pickle; env_conf = pickle.loads(open(f"env_conf_{eps_idx}.pkl", "rb").read())
    return env_conf, episode


class LeRobotParquetDataset:
    """Minimal LeRobot v2.1 reader for replay: reads data/**/*.parquet and meta/ directly, so it works
    with `datasets` versions that lerobot's LeRobotDataset can't read. Frames are dicts of torch tensors
    (no images); exposes the attributes get_episode_lerobot uses."""

    def __init__(self, root):
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        import pandas as pd

        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
        for ep in episodes:
            if not isinstance(ep.get("environment_config"), str):
                ep["environment_config"] = json.dumps(ep.get("environment_config"))
        episodes.sort(key=lambda ep: ep["episode_index"])
        frames = pd.concat([self._read_episode(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
        frames = frames.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
        self._frames = frames
        self._dtypes = {}
        for key, ft in info.get("features", {}).items():
            try:
                self._dtypes[key] = np.dtype(ft["dtype"])
            except TypeError:  # image / video / string features
                pass
        self.meta = SimpleNamespace(fps=info["fps"], episodes=episodes, info=info)
        self.num_episodes = len(episodes)
        bounds = frames.groupby("episode_index").indices
        starts = [int(bounds[ep["episode_index"]][0]) for ep in episodes]
        ends = [int(bounds[ep["episode_index"]][-1]) + 1 for ep in episodes]
        self.episode_data_index = {"from": np.array(starts), "to": np.array(ends)}

    @staticmethod
    def _read_episode(path):
        """Read one episode_{idx:06d}.parquet. The file name is authoritative (as in LeRobot v2.1); some
        SIMPLE-eval files carry a stale episode_index column (e.g. LocomotionPickBetweenTables dr-level-2
        episode_000003 says 4), which would drop that episode and duplicate another."""
        import re
        import pandas as pd

        df = pd.read_parquet(path)
        m = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
        if m is not None and "episode_index" in df:
            idx = int(m.group(1))
            if (df["episode_index"] != idx).any():
                print(f"Warning: {path} has episode_index {sorted(df['episode_index'].unique().tolist())}, using {idx} from the file name")
                df["episode_index"] = idx
        return df

    @property
    def hf_dataset(self):
        """get_episode_lerobot(load_videos=False) reads rows from here; this reader never loads images."""
        return self

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        # Cast to the info.json feature dtypes, as LeRobotDataset does (parquet may store e.g. float64).
        out = {}
        for k, v in self._frames.iloc[idx].items():
            dtype = self._dtypes.get(k)
            out[k] = torch.tensor(np.asarray(v, dtype=dtype) if dtype is not None else np.array(v))
        return out
