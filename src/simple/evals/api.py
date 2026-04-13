from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    policy: str
    split: str = "train"
    host: str = "172.17.0.1"
    port: int = 21000
    data_format: str = "rlds_numpy"
    sim_mode: str = "mujoco_isaac"
    headless: bool = False
    eval_dir: str = "data/evals"
    max_episode_steps: int = 15000
    num_episodes: int = 100
    episode_start: int = 0
    data_dir: str = "data/datagen"
    success_criteria: float = 0.9
    save_video: bool = True
    num_workers: int = 1


@dataclass
class EvalResult:
    env_id: str
    policy: str
    split: str
    stats: dict[str, bool]
    success_rate: float
    log_path: str
    eval_dir: str


class EvalRunner:
    def __init__(
        self,
        config: EvalConfig,
        *,
        agent_factory: Callable[..., Any] | None = None,
        show_progress: bool = True,
    ):
        self.config = config
        self.agent_factory = agent_factory
        self.show_progress = show_progress

    def run(self, policy: Any | None = None) -> EvalResult:
        config = self.config
        agent_factory = self.agent_factory

        if policy is not None:
            if isinstance(policy, str):
                config = replace(config, policy=policy)
                agent_factory = None
            elif callable(policy):
                agent_factory = policy
            elif hasattr(policy, "make_agent"):
                agent_factory = policy.make_agent
            else:
                raise TypeError(
                    "policy must be a baseline name, an agent factory, or an object "
                    "with make_agent(robot, host, port)."
                )

        from simple.cli.eval import run_eval as _run_eval

        return _run_eval(
            config,
            agent_factory=agent_factory,
            show_progress=self.show_progress,
        )
