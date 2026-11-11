# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

__version__ = "0.1.0"

# Exit with upgrade instructions in a pre-upgrade environment (Python 3.10 / Isaac Sim 4.5 / MuJoCo 3.3).
from simple._env_check import check_environment as _check_environment

_check_environment()

__all__ = ["__version__", "EvalConfig", "EvalEpisodeResult", "EvalResult", "EnvRunner", "evaluate_policy", "run_with_tui"]


def __getattr__(name: str):
    if name in {"EvalConfig", "EvalEpisodeResult", "EvalResult", "EnvRunner", "evaluate_policy", "run_with_tui"}:
        from simple.evals import (
            EvalConfig,
            EvalEpisodeResult,
            EvalResult,
            EnvRunner,
            evaluate_policy,
            run_with_tui,
        )

        exports = {
            "EvalConfig": EvalConfig,
            "EvalEpisodeResult": EvalEpisodeResult,
            "EvalResult": EvalResult,
            "EnvRunner": EnvRunner,
            "evaluate_policy": evaluate_policy,
            "run_with_tui": run_with_tui,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
