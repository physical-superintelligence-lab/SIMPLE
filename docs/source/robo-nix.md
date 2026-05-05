# robo-nix Setup

SIMPLE uses `robo-nix` for the native robotics runtime and keeps Python package resolution in `uv`.

The committed runtime contract is:

- `flake.nix`: generated plumbing that points at `robo-nix`.
- `robo.nix`: SIMPLE's runtime manifest.
- `.robo-nix/`: local generated cache, ignored by git.

## Setup

Install `robo`:

```bash
curl -fsSL https://raw.githubusercontent.com/ausbxuse/robo-nix/develop/scripts/install.sh | sh
```

From the SIMPLE checkout:

```bash
git submodule update --init --recursive
robo up
robo shell
uv sync --all-groups --index-strategy unsafe-best-match
bash scripts/install_curobo.sh
```

Run smoke checks inside the robo shell:

```bash
bash scripts/tests/check_datagen.sh
bash scripts/tests/check_eval.sh
```

## Cache Trust

`robo-nix` uses the `nixpkgs-python` cache for Python interpreters. If the host Nix daemon does not trust that cache, `robo up` stops before doing a slow local CPython build.

Prefer configuring the daemon to trust the cache. If a slow local CPython build is acceptable for this machine, opt in explicitly:

```bash
ROBO_NIX_ALLOW_SOURCE_PYTHON=1 robo up
```

## Updating the Runtime Contract

Regenerate the robo-nix files only when the runtime requirements change:

```bash
robo init . \
  --force \
  --name simple \
  --python-version 3.10 \
  --components base,python-uv,native-build,isaac-sim,x11-gl,mujoco,media,linux-headers \
  --robo-nix-url github:ausbxuse/robo-nix/develop
```

Review `flake.nix` and `robo.nix` before committing regenerated files.
