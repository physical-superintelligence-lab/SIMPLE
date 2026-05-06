#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv venv
uv pip install --python .venv/bin/python setuptools wheel pybind11
GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}" \
  uv sync --all-groups --index-strategy unsafe-best-match
