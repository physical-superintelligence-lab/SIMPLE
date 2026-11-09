# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT


def require_episode_task(env_conf: dict, task, episode: int, data_dir: str, env_id: str) -> None:
    """Exit unless the episode was recorded for this task; replaying another task's scene is
    meaningless and usually dies later in env.reset with an unrelated DR error.

    SystemExit, not an Exception: the replay CLIs catch Exception to log and carry on.
    """
    uid = env_conf.get("uid")
    if uid == task.uid:
        return
    if f"{uid}_mp" == task.uid:  # recorded before the task gained its "_mp" suffix
        env_conf["uid"] = task.uid
        return
    raise SystemExit(
        f"error: task mismatch: episode {episode} of {data_dir} was recorded for {uid!r}, but "
        f"{env_id} is {task.uid!r}. In a merged dataset --episode-start is a global index: use "
        "an episode of this task (see merge_summary.json) or the env id it was recorded for."
    )
