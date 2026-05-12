# robo-nix Setup

SIMPLE uses `robo-nix` for the native robotics runtime and keeps Python package resolution in `uv`.

The committed runtime contract is:

- `flake.nix`: generated plumbing that points at `robo-nix`.
- `robo.nix`: SIMPLE's runtime manifest.
- `.robo-nix/`: local generated cache, ignored by git.

## Setup

Install `robo`:

```bash
curl -fsSL https://raw.githubusercontent.com/ausbxuse/robo-nix/master/scripts/install.sh | sh
```

From the SIMPLE checkout:

```bash
git submodule update --init --recursive
robo shell
bash scripts/setup_python_env.sh
bash scripts/install_curobo.sh
```

Run smoke checks inside the robo shell:

```bash
bash scripts/tests/check_datagen.sh
bash scripts/tests/check_eval.sh
```


Review `flake.nix` and `robo.nix` before committing regenerated files.
