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
