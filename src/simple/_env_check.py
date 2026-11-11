# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Refuse to run in an environment that predates the Isaac Sim 6.2 / MuJoCo 3.12 upgrade.

Stdlib only and cheap: reads installed package metadata, imports nothing heavy. On a mismatch it
prints the upgrade steps and exits (status 1); the only way past it is to upgrade the environment.
A failure of the check itself never blocks the import.
"""

import sys

# What pyproject.toml / uv.lock require: (label, distribution, expected version prefix)
_EXPECTED_PYTHON = (3, 12)
_EXPECTED_PACKAGES = (
    ("Isaac Sim", "isaacsim", "6.2"),
    ("MuJoCo", "mujoco", "3.12"),
    ("NumPy", "numpy", "2."),
)


def _find_problems() -> list[str]:
    from importlib import metadata

    problems: list[str] = []
    if sys.version_info[:2] != _EXPECTED_PYTHON:
        expected = ".".join(map(str, _EXPECTED_PYTHON))
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} (expected {expected})")
    for label, dist, prefix in _EXPECTED_PACKAGES:
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue  # not installed here (e.g. a MuJoCo-only tool env): nothing to compare
        if not (version == prefix or version.startswith(prefix if prefix.endswith(".") else prefix + ".")):
            expected = prefix + "x" if prefix.endswith(".") else prefix
            problems.append(f"{label} {version} (expected {expected})")
    return problems


def check_environment() -> None:
    try:
        problems = _find_problems()
    except Exception:
        return  # a broken check must not block a working environment
    if not problems:
        return

    print(
        "\n[simple] Please Upgrade: SIMPLE now requires Python 3.12 / Isaac Sim 6.2 / MuJoCo 3.12, but the\n"
        + f"[simple]        running environment has: {'; '.join(problems)}.\n"
        + "[simple]        Upgrade the environment to run this version:\n"
        + "[simple]          host:   rm -rf .venv && uv sync --all-groups --index-strategy unsafe-best-match\n"
        + "[simple]                  && git submodule update third_party/curobo && bash scripts/install_curobo.sh\n"
        + "[simple]          docker: docker pull ghcr.io/physical-superintelligence-lab/simple:20261027\n"
        + "[simple]                  && docker tag ghcr.io/physical-superintelligence-lab/simple:20261027 simple:20261027\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)
