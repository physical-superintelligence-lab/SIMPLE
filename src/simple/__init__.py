"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

__version__ = "0.1.0"
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
