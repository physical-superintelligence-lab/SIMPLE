from __future__ import annotations

from dataclasses import dataclass
import atexit
import importlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Protocol


class SupportsReset(Protocol):
    def reset(self, **kwargs: Any) -> None: ...


class SupportsGetAction(Protocol):
    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Any: ...


class PolicyLike(SupportsGetAction, Protocol):
    def reset(self, **kwargs: Any) -> None: ...


@dataclass(slots=True)
class EvalConfig:
    env_id: str
    split: str = "train"
    data_format: str = "rlds_numpy"
    sim_mode: str = "mujoco_isaac"
    headless: bool = False
    eval_dir: str = "data/evals"
    max_episode_steps: int = 15000
    data_dir: str = "data/datagen"
    success_criteria: float = 0.9
    save_video: bool = True
    host: str = "172.17.0.1"
    port: int = 21000


@dataclass(slots=True)
class EvalEpisodeResult:
    episode_idx: int
    episode_id: str
    success: bool
    steps: int
    duration_seconds: float


@dataclass(slots=True)
class EvalResult:
    env_id: str
    policy_name: str
    split: str
    episode_results: list[EvalEpisodeResult]

    @property
    def successes(self) -> int:
        return sum(result.success for result in self.episode_results)

    @property
    def success_rate(self) -> float:
        if not self.episode_results:
            return 0.0
        return self.successes / len(self.episode_results)

    def by_episode_id(self) -> dict[str, bool]:
        return {result.episode_id: result.success for result in self.episode_results}


class _CallablePolicyAdapter:
    def __init__(
        self,
        action_fn: Callable[..., Any],
        reset_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._action_fn = action_fn
        self._reset_fn = reset_fn

    def reset(self, **kwargs: Any) -> None:
        if self._reset_fn is not None:
            self._reset_fn(**kwargs)

    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._action_fn(observation, instruction=instruction, **kwargs)


class _PolicyObjectAdapter:
    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def reset(self, **kwargs: Any) -> None:
        reset_fn = getattr(self._policy, "reset", None)
        if callable(reset_fn):
            reset_fn(**kwargs)

    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._policy.get_action(observation, instruction=instruction, **kwargs)


class EnvRunner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self._dataset, self._dataset_size, self._render_hz, self._get_episode = self._load_dataset()
        self._raw_env = self._make_env()
        self._closed = False
        self._close_registered = False
        self.task = self._raw_env.unwrapped.task  # type: ignore[attr-defined]
        self.eval_output_dir = Path(config.eval_dir) / self.policy_output_name("library") / config.split
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._raw_env.close()

    def __enter__(self) -> EnvRunner:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Isaac Sim shutdown tears down the interpreter aggressively enough that
        # code after `with EnvRunner(...)` may never run. Defer that close to
        # interpreter shutdown so library callers can still read and print the result.
        if "isaac" in self.config.sim_mode and not self._close_registered:
            atexit.register(self.close)
            self._close_registered = True
            return
        self.close()

    def run(
        self,
        policy: str | PolicyLike | Callable[..., Any],
        *,
        num_episodes: int = 100,
        episode_start: int = 0,
        policy_name: str | None = None,
        policy_reset_fn: Callable[..., Any] | None = None,
        episode_reset_kwargs_fn: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
        progress_callback: Callable[[EvalEpisodeResult], None] | None = None,
        progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvalResult:
        if episode_start < 0:
            raise ValueError(f"episode_start must be >= 0, got {episode_start}")

        agent, resolved_policy_name = self._resolve_policy(
            policy,
            policy_name=policy_name,
            policy_reset_fn=policy_reset_fn,
        )
        policy_output_dir = Path(self.config.eval_dir) / self.policy_output_name(resolved_policy_name) / self.config.split
        policy_output_dir.mkdir(parents=True, exist_ok=True)

        episode_indices = list(range(episode_start, self._dataset_size))
        episode_indices = episode_indices[:num_episodes]
        results: list[EvalEpisodeResult] = []
        successes = 0

        if progress_reporter is not None:
            progress_reporter(
                {
                    "event": "worker_init",
                    "total_episodes": len(episode_indices),
                    "status": "ready",
                    "setup_seconds": 0.0,
                }
            )
        for eps_idx in episode_indices:
            env_conf, episode = self._get_episode(self._dataset, eps_idx)
            result = self.run_episode(
                agent,
                env_conf,
                episode,
                episode_idx=eps_idx,
                policy_name=resolved_policy_name,
                eval_output_dir=policy_output_dir,
                episode_reset_kwargs_fn=episode_reset_kwargs_fn,
                progress_reporter=progress_reporter,
            )
            results.append(result)
            successes += int(result.success)
            if progress_callback is not None:
                progress_callback(result)
            if progress_reporter is not None:
                progress_reporter(
                    {
                        "event": "episode_end",
                        "episode": result.episode_id,
                        "step": result.steps,
                        "completed_episodes": len(results),
                        "successes": successes,
                        "episode_seconds": result.duration_seconds,
                        "steps_per_second": (
                            result.steps / result.duration_seconds
                            if result.duration_seconds > 0
                            else 0.0
                        ),
                    }
                )

        if progress_reporter is not None:
            progress_reporter({"event": "worker_status", "status": "closing"})
            progress_reporter(
                {
                    "event": "worker_done",
                    "completed_episodes": len(results),
                    "successes": successes,
                }
            )

        return EvalResult(
            env_id=self.config.env_id,
            policy_name=resolved_policy_name,
            split=self.config.split,
            episode_results=results,
        )

    def run_with_tui(
        self,
        policy: str | PolicyLike | Callable[..., Any],
        **kwargs: Any,
    ) -> EvalResult:
        from rich.live import Live
        from simple.evals.tui import (
            WorkerProgress,
            make_console,
            render_progress,
            restore_cursor,
            update_progress,
        )

        total_episodes = int(kwargs.get("num_episodes", 100))
        policy_name = kwargs.get("policy_name")
        resolved_policy_label = policy if isinstance(policy, str) else (
            policy_name or getattr(policy, "__name__", policy.__class__.__name__)
        )
        console = make_console()
        worker_states = {0: WorkerProgress(total_episodes=total_episodes)}
        log_path = str(Path(self.config.eval_dir) / "eval_latest.log")

        try:
            with Live(
                render_progress(self.config.env_id, str(resolved_policy_label), worker_states, log_path),
                console=console,
                refresh_per_second=4,
            ) as live:
                def report(payload: dict[str, Any]) -> None:
                    update_progress(worker_states, 0, payload)
                    live.update(
                        render_progress(
                            self.config.env_id,
                            str(resolved_policy_label),
                            worker_states,
                            log_path,
                        ),
                        refresh=False,
                    )

                return self.run(
                    policy,
                    progress_reporter=report,
                    **kwargs,
                )
        finally:
            restore_cursor(console)

    def run_episode(
        self,
        policy: PolicyLike,
        env_conf: dict[str, Any],
        episode: Any,
        *,
        episode_idx: int,
        policy_name: str,
        eval_output_dir: Path | None = None,
        episode_reset_kwargs_fn: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
        progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvalEpisodeResult:
        task_id = f"episode_{episode_idx}"
        env = self._raw_env
        if self.config.save_video:
            from simple.envs.wrappers.video_recorder import VideoRecorder

            output_dir = eval_output_dir or (Path(self.config.eval_dir) / self.policy_output_name(policy_name) / self.config.split)
            env = VideoRecorder(
                env=self._raw_env,
                video_folder=str(output_dir),
                name_prefix=task_id,
                framerate=self._render_hz,
                write_png=False,
            )

        observation, info = env.reset(options={"state_dict": env_conf})
        instruction = self.task.instruction
        reset_kwargs = self._default_reset_kwargs(policy_name, env_conf, episode)
        if episode_reset_kwargs_fn is not None:
            reset_kwargs.update(episode_reset_kwargs_fn(env_conf, episode))
        policy.reset(**reset_kwargs)

        frame_idx = 0
        episode_start_time = time.perf_counter()
        episode_over = False
        step_update_every = 5
        if progress_reporter is not None:
            progress_reporter({"event": "episode_start", "episode": task_id})
        while not episode_over:
            try:
                action = policy.get_action(observation, info=info, instruction=instruction)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                frame_idx += 1
                if progress_reporter is not None and (
                    frame_idx == 1 or frame_idx % step_update_every == 0 or episode_over
                ):
                    progress_reporter(
                        {
                            "event": "episode_step",
                            "episode": task_id,
                            "step": frame_idx,
                        }
                    )
            except StopIteration:
                episode_over = True

        success = bool(self._raw_env.unwrapped._success)  # type: ignore[attr-defined]
        duration_seconds = time.perf_counter() - episode_start_time

        if self.config.save_video and isinstance(env, VideoRecorder):
            env.release()

        return EvalEpisodeResult(
            episode_idx=episode_idx,
            episode_id=task_id,
            success=success,
            steps=frame_idx,
            duration_seconds=duration_seconds,
        )

    def policy_output_name(self, policy_name: str) -> str:
        return policy_name.replace("/", "_")

    def _make_env(self):
        import gymnasium as gym

        import simple.envs as _  # noqa: F401

        make_kwargs = dict(
            sim_mode=self.config.sim_mode,
            render_hz=self._render_hz,
            headless=self.config.headless,
            max_episode_steps=self.config.max_episode_steps,
            success_criteria=self.config.success_criteria,
        )
        return gym.make(self.config.env_id, **make_kwargs)

    def _load_dataset(self):
        if self.config.data_format == "rlds_numpy":
            data_path = Path(self.config.eval_dir) / self.config.env_id / self.config.split
            if not data_path.exists():
                raise FileNotFoundError(f"Data path {data_path} does not exist.")
            dataset = sorted(list(data_path.glob("*episode_*.pkl")))
            render_hz = 3

            from simple.datasets.rlds import convert_env_config_to_state_dict

            def get_episode(dataset_obj, idx):
                with open(dataset_obj[idx], "rb") as f:
                    episode = pickle.load(f)
                env_cfg = json.loads(episode["task"]["environment_config"][0].decode("utf-8"))
                state_dict = convert_env_config_to_state_dict(env_cfg)
                state_dict["camera_info"] = env_cfg["scene_info"]["camera_info"]
                return state_dict, episode

            return dataset, len(dataset), render_hz, get_episode

        if self.config.data_format == "lerobot":
            from simple.datasets.lerobot import get_episode_lerobot

            def get_episode(dataset_obj, idx):
                return get_episode_lerobot(dataset_obj, idx)

            try:
                dataset = _LocalLeRobotDataset(self.config.env_id, self.config.data_dir)
                return dataset, dataset.num_episodes, dataset.meta.fps, get_episode
            except FileNotFoundError:
                pass

            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                dataset = LeRobotDataset(repo_id=self.config.env_id, root=self.config.data_dir)
                return dataset, dataset.num_episodes, dataset.meta.fps, get_episode
            except ModuleNotFoundError as exc:
                if exc.name != "lerobot":
                    raise
                raise RuntimeError(
                    "lerobot dataset support requires either a local dataset under "
                    f"{self.config.data_dir!r} or the lerobot package to be installed."
                ) from exc

        raise NotImplementedError(f"Data format {self.config.data_format} not supported.")

    def _resolve_policy(
        self,
        policy: str | PolicyLike | Callable[..., Any],
        *,
        policy_name: str | None,
        policy_reset_fn: Callable[..., Any] | None,
    ) -> tuple[PolicyLike, str]:
        if isinstance(policy, str):
            from simple.utils import snake_to_pascal

            policy_module = importlib.import_module(f"simple.baselines.{policy}")
            agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")
            return agent_clazz(self.task.robot, self.config.host, self.config.port), policy

        if hasattr(policy, "get_action"):
            resolved_name = policy_name or policy.__class__.__name__
            if hasattr(policy, "reset"):
                return policy, resolved_name
            return _PolicyObjectAdapter(policy), resolved_name

        if callable(policy):
            resolved_name = policy_name or getattr(policy, "__name__", "callable_policy")
            return _CallablePolicyAdapter(policy, reset_fn=policy_reset_fn), resolved_name

        raise TypeError(f"Unsupported policy type: {type(policy)!r}")

    def _default_reset_kwargs(
        self,
        policy_name: str,
        env_conf: dict[str, Any],
        episode: Any,
    ) -> dict[str, Any]:
        if policy_name == "vlt":
            return {
                "camera_infos": env_conf["camera_info"],
                "episode": episode,
                "condition": "forward_all",
                "save_cond_images": True,
            }
        return {}


def evaluate_policy(
    config: EvalConfig,
    policy: str | PolicyLike | Callable[..., Any],
    **kwargs: Any,
) -> EvalResult:
    with EnvRunner(config) as runner:
        return runner.run(policy, **kwargs)


def run_with_tui(
    config: EvalConfig,
    policy: str | PolicyLike | Callable[..., Any],
    **kwargs: Any,
) -> EvalResult:
    with EnvRunner(config) as runner:
        return runner.run_with_tui(policy, **kwargs)


@dataclass(slots=True)
class _LocalLeRobotMeta:
    fps: int
    episodes: list[dict[str, Any]]


class _LocalLeRobotDataset:
    def __init__(self, repo_id: str, root: str):
        import pandas as pd

        self.repo_id = repo_id
        self.root = self._resolve_dataset_root(Path(root), repo_id)
        meta_dir = self.root / "meta"
        info = json.loads((meta_dir / "info.json").read_text())
        with (meta_dir / "episodes.jsonl").open() as f:
            episodes = [json.loads(line) for line in f if line.strip()]

        self.meta = _LocalLeRobotMeta(fps=info["fps"], episodes=episodes)
        self.num_episodes = info["total_episodes"]
        self._data_path_pattern = info["data_path"]
        self._chunks_size = info["chunks_size"]
        self._pd = pd

        from_indices = [episode["dataset_from_index"] for episode in episodes]
        to_indices = [episode["dataset_to_index"] + 1 for episode in episodes]
        self.episode_data_index = {"from": from_indices, "to": to_indices}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx = self._episode_index_for_global_index(idx)
        episode_start = self.episode_data_index["from"][episode_idx]
        local_idx = idx - episode_start
        parquet_path = self.root / self._data_path_pattern.format(
            episode_chunk=episode_idx // self._chunks_size,
            episode_index=episode_idx,
        )
        frame = self._pd.read_parquet(parquet_path).iloc[local_idx]
        return frame.to_dict()

    def _episode_index_for_global_index(self, idx: int) -> int:
        for episode_idx, (from_idx, to_idx) in enumerate(
            zip(self.episode_data_index["from"], self.episode_data_index["to"])
        ):
            if from_idx <= idx < to_idx:
                return episode_idx
        raise IndexError(idx)

    @staticmethod
    def _resolve_dataset_root(root: Path, repo_id: str) -> Path:
        direct = root / repo_id
        if direct.exists():
            return direct

        stripped = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
        stripped_path = root / stripped
        if stripped_path.exists():
            return stripped_path

        matches = sorted(path for path in root.glob(f"{stripped}*") if path.is_dir())
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"Could not locate local LeRobot dataset for repo_id={repo_id!r} under {root}. "
            f"Tried {direct} and {stripped_path}."
        )
