#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import selectors
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from rich.console import Console
from rich.table import Table


DEFAULT_ENV_ID = "simple/G1WholebodyBendPick-v0"
DEFAULT_TEST_ROOT_NAME = ".test_regression"
DATAGEN_MAX_ATTEMPTS = 3
AMO_POLICY_ASSETS = [
    Path("src/simple/robots/policy/amo_jit.pt"),
    Path("src/simple/robots/policy/adapter_jit.pt"),
    Path("src/simple/robots/policy/adapter_norm_stats.pt"),
]


@dataclass
class TestResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class RuntimeRoots:
    source_root: Path
    simple_root: Path
    tmp_parent: Path
    test_root: Path
    log_root: Path
    output_root: Path
    datagen_root: Path
    eval_root: Path


class Runner:
    def __init__(self, root: Path, log_root: Path):
        self.root = root
        self.log_root = log_root
        self.console = Console()
        self.results: list[TestResult] = []
        self._log_index = 0

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(TestResult(name=name, status=status, detail=detail))
        style = {
            "PASS": "bold green",
            "FAIL": "bold red",
            "SKIP": "bold yellow",
        }.get(status, "bold")
        line = f"[{style}]{status:>4}[/{style}] {name}"
        if detail:
            line += f" [dim]- {detail}[/dim]"
        self.console.print(line)

    def log_path(self, name: str, suffix: str = ".log") -> Path:
        self._log_index += 1
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "test"
        return self.log_root / f"{self._log_index:02d}_{slug}{suffix}"

    @staticmethod
    def _prepare_log_path(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w"):
            pass

    @staticmethod
    def _status(console: Console, prefix: str):
        class _CompatStatus:
            def __enter__(self_nonlocal):
                console.print(f"[bold blue]{prefix}[/bold blue]")
                return self_nonlocal

            def __exit__(self_nonlocal, exc_type, exc, tb):
                return False

            def update(self_nonlocal, text: str) -> None:
                console.print(text)

        try:
            return console.status(f"[bold blue]{prefix}[/bold blue]", spinner="dots")
        except ModuleNotFoundError:
            return _CompatStatus()

    @staticmethod
    def _append_chunk(
        *,
        chunk: str,
        output_parts: list[str],
        log_file,
        status,
        prefix: str,
    ) -> None:
        if not chunk:
            return
        output_parts.append(chunk)
        log_file.write(chunk)
        log_file.flush()
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            status.update(f"[bold blue]{prefix}[/bold blue] [dim]- {trim_line(lines[-1])}[/dim]")

    def run_capture(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        prefix: str,
        log_path: Path,
    ) -> tuple[int, str]:
        output_parts: list[str] = []
        self._prepare_log_path(log_path)
        with open(log_path, "w", buffering=1) as log_file:
            proc = subprocess.Popen(
                nix_cmd(cmd),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            with self._status(self.console, prefix) as status:
                while True:
                    if proc.poll() is not None:
                        self._append_chunk(
                            chunk=proc.stdout.read(),
                            output_parts=output_parts,
                            log_file=log_file,
                            status=status,
                            prefix=prefix,
                        )
                        break

                    events = selector.select(timeout=0.2)
                    for key, _ in events:
                        line = key.fileobj.readline()
                        self._append_chunk(
                            chunk=line,
                            output_parts=output_parts,
                            log_file=log_file,
                            status=status,
                            prefix=prefix,
                        )
                proc.wait()
            selector.unregister(proc.stdout)
            proc.stdout.close()
        return proc.returncode, "".join(output_parts)

    def run(
        self,
        name: str,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        skip_if: bool = False,
        skip_reason: str = "",
    ) -> bool:
        if skip_if:
            self.record(name, "SKIP", skip_reason)
            return False
        log_path = self.log_path(name)
        returncode, output = self.run_capture(
            cmd,
            cwd=cwd,
            env=env,
            prefix=f"RUN  {name}",
            log_path=log_path,
        )
        if returncode != 0:
            self.record(name, "FAIL", f"{tail_detail(output)} [log: {log_path}]")
            return False
        self.record(name, "PASS")
        return True

    def summary(self) -> int:
        counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1

        table = Table(title="Regression Test Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("PASS", str(counts["PASS"]), style="green")
        table.add_row("FAIL", str(counts["FAIL"]), style="red")
        table.add_row("SKIP", str(counts["SKIP"]), style="yellow")
        self.console.print()
        self.console.print(table)
        return 1 if counts["FAIL"] else 0

    def banner(self, *, stage: str, roots: RuntimeRoots, env: dict[str, str]) -> None:
        mode = "isolated-worktree" if roots.simple_root != roots.source_root else "current-worktree"
        self.console.print(
            "\n".join(
                [
                    f"[bold]Stage:[/bold] {stage}",
                    f"[bold]Mode:[/bold] {mode}",
                    f"[bold]Workspace:[/bold] {roots.simple_root}",
                    f"[bold]UV env:[/bold] {env['UV_PROJECT_ENVIRONMENT']}",
                ]
            )
        )


def trim_line(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tail_detail(output: str) -> str:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    return " | ".join(lines[-8:]) if lines else "exit!=0"


def nix_cmd(cmd: list[str]) -> list[str]:
    import shlex

    return [
        "nix",
        "develop",
        "--impure",
        "-c",
        "bash",
        "-lc",
        "exec " + shlex.join(cmd),
    ]


def make_runtime_roots(source_root: Path, simple_root: Path, tmp_parent: Path) -> RuntimeRoots:
    test_root = simple_root / DEFAULT_TEST_ROOT_NAME
    output_root = test_root / "output"
    roots = RuntimeRoots(
        source_root=source_root,
        simple_root=simple_root,
        tmp_parent=tmp_parent,
        test_root=test_root,
        log_root=test_root / "logs",
        output_root=output_root,
        datagen_root=output_root / "datagen",
        eval_root=output_root / "eval",
    )
    for path in [roots.test_root, roots.log_root, roots.output_root, roots.datagen_root, roots.eval_root]:
        path.mkdir(parents=True, exist_ok=True)
    return roots


def pick_free_port(start: int = 21090, end: int = 21999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def base_env(source_root: Path, simple_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("HF_TOKEN") and not env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]
    env["UV_PROJECT_ENVIRONMENT"] = str(simple_root / ".venv-nix")
    env["XDG_CACHE_HOME"] = str(simple_root / ".nix-cache")
    tmp_root = simple_root / DEFAULT_TEST_ROOT_NAME / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_root)
    env["TMP"] = str(tmp_root)
    env["TEMP"] = str(tmp_root)
    torch_extensions_dir = (
        env.get("TEST_REGRESSION_TORCH_EXTENSIONS_DIR")
        or env.get("TORCH_EXTENSIONS_DIR")
        or str(simple_root / ".torch_extensions_regression")
    )
    clean_stale_torch_extensions_cache(
        Path(torch_extensions_dir),
        simple_root=simple_root,
        uv_project_environment=Path(env["UV_PROJECT_ENVIRONMENT"]),
    )
    Path(torch_extensions_dir).mkdir(parents=True, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir
    env["SIMPLE_AUTO_BOOTSTRAP"] = "0"
    return env


def clean_stale_torch_extensions_cache(
    torch_extensions_dir: Path,
    *,
    simple_root: Path,
    uv_project_environment: Path,
) -> None:
    if not torch_extensions_dir.exists():
        return

    expected_root = str(simple_root.resolve())
    expected_uv_env = str(uv_project_environment.resolve(strict=False))

    for build_ninja in torch_extensions_dir.rglob("build.ninja"):
        try:
            text = build_ninja.read_text()
        except OSError:
            continue
        if expected_root in text and expected_uv_env in text:
            continue
        shutil.rmtree(torch_extensions_dir, ignore_errors=True)
        return


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def link_shared_path(src: Path, dst: Path, *, replace: bool = False) -> None:
    if not src.exists():
        return
    if dst.is_symlink() or dst.is_file():
        if replace:
            dst.unlink(missing_ok=True)
        else:
            return
    elif dst.is_dir():
        if replace:
            shutil.rmtree(dst, ignore_errors=True)
        elif not any(dst.iterdir()):
            dst.rmdir()
        else:
            return
    elif dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def copy_shared_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.is_symlink() or dst.is_file():
        dst.unlink(missing_ok=True)
    elif dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name == "__pycache__":
                ignored.add(name)
            elif name == "build":
                ignored.add(name)
            elif name.endswith(".egg-info"):
                ignored.add(name)
            elif name.endswith(".so"):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as handle:
            first_line = handle.readline().strip()
    except UnicodeDecodeError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def ensure_lfs_assets(source_root: Path, rel_paths: list[Path]) -> None:
    pending = [rel for rel in rel_paths if not (source_root / rel).is_file() or is_git_lfs_pointer(source_root / rel)]
    if not pending:
        return
    include = ",".join(str(rel) for rel in pending)
    subprocess.run(
        ["git", "-C", str(source_root), "lfs", "pull", "--include", include],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unresolved = [rel for rel in pending if not (source_root / rel).is_file() or is_git_lfs_pointer(source_root / rel)]
    if unresolved:
        joined = ", ".join(str(rel) for rel in unresolved)
        raise RuntimeError(f"required Git LFS assets are unavailable after git lfs pull: {joined}")


def materialize_required_assets(source_root: Path, simple_root: Path) -> None:
    ensure_lfs_assets(source_root, AMO_POLICY_ASSETS)
    for rel in AMO_POLICY_ASSETS:
        link_shared_path(source_root / rel, simple_root / rel, replace=True)


def link_shared_inputs(source_root: Path, simple_root: Path) -> None:
    if source_root == simple_root:
        return
    copy_shared_tree(source_root / "third_party/curobo", simple_root / "third_party/curobo")
    materialize_required_assets(source_root, simple_root)
    link_shared_path(source_root / "data", simple_root / "data", replace=True)
    for rel in [
        ".uv-cache",
        "third_party/evdev",
        "third_party/openpi-client",
        "third_party/gear_sonic",
        "third_party/unitree_sdk2_python",
        "third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64",
        "third_party/decoupled_wbc",
    ]:
        link_shared_path(source_root / rel, simple_root / rel)


def create_clean_worktree(source_root: Path, tmpdir: Path) -> tuple[Path, Path]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix="simple-regression.", dir=tmpdir))
    worktree_root = tmp_parent / "repo"
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(worktree_root), "HEAD"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    link_shared_inputs(source_root, worktree_root)
    return tmp_parent, worktree_root


def cleanup_worktree(roots: RuntimeRoots, keep_worktree: bool, preserve_worktree: bool = False) -> None:
    if keep_worktree:
        print(f"[keep] worktree kept at {roots.simple_root}")
        return
    if roots.simple_root == roots.source_root:
        shutil.rmtree(roots.test_root, ignore_errors=True)
        return
    if preserve_worktree:
        return
    subprocess.run(
        ["git", "-C", str(roots.source_root), "worktree", "remove", "--force", str(roots.simple_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    shutil.rmtree(roots.tmp_parent, ignore_errors=True)


def reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def prepare_stage(roots: RuntimeRoots, stage: str) -> None:
    reset_path(roots.log_root)
    for path in [roots.test_root / "tmp", roots.output_root]:
        path.mkdir(parents=True, exist_ok=True)

    if stage == "env":
        return
    if stage == "datagen":
        reset_path(roots.datagen_root)
        return
    if stage == "eval":
        reset_path(roots.eval_root)
        return
    if stage == "all":
        reset_path(roots.datagen_root)
        reset_path(roots.eval_root)
        return
    raise ValueError(f"unsupported stage: {stage}")


def existing_datagen_run_dir(roots: RuntimeRoots, env_id: str) -> Path | None:
    save_dir = roots.datagen_root / env_id.split("/", 1)[1]
    dataset_dir = resolve_dataset_dir(save_dir, env_id)
    required = [
        dataset_dir / "meta/info.json",
        dataset_dir / "meta/episodes.jsonl",
        dataset_dir / "meta/tasks.jsonl",
    ]
    if not all(path.is_file() for path in required):
        return None
    if not list(dataset_dir.rglob("*.parquet")):
        return None
    media_files = list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.mp4"))
    if not media_files:
        return None
    return save_dir


def resolve_dataset_dir(save_dir: Path, env_id: str) -> Path:
    direct = save_dir / "level-0"
    if direct.is_dir():
        return direct

    env_name = env_id.split("/", 1)[1] if "/" in env_id else env_id
    task_nested = save_dir / env_name / "level-0"
    if task_nested.is_dir():
        return task_nested

    nested = save_dir / "simple" / env_name / "level-0"
    if nested.is_dir():
        return nested

    return direct


def bootstrap_shell(*, allow_rebuild: bool) -> str:
    if allow_rebuild:
        return """
set -euo pipefail
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  if [[ -d "$UV_PROJECT_ENVIRONMENT" ]]; then
    rm -rf "$UV_PROJECT_ENVIRONMENT"
  fi
  uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
fi

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' >/dev/null 2>&1; then
import importlib

for mod in [
    "simple.cli.datagen",
    "simple.cli.eval",
]:
    importlib.import_module(mod)
PY
  ./scripts/nix/bootstrap.sh
elif ! "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' >/dev/null 2>&1; then
import importlib

for mod in [
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]:
    importlib.import_module(mod)
PY
  ./scripts/nix/bootstrap.sh
fi

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' >/dev/null 2>&1; then
from pathlib import Path
import simple

repo_root = Path.cwd().resolve()
simple_path = Path(simple.__file__).resolve()
raise SystemExit(0 if repo_root in simple_path.parents else 1)
PY
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --editable .
fi
"""
    return """
set -euo pipefail
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  echo "[stage] missing ready virtualenv at $UV_PROJECT_ENVIRONMENT"
  echo "[stage] run --stage env first"
  exit 1
fi

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' >/dev/null 2>&1; then
import importlib

for mod in [
    "simple.cli.datagen",
    "simple.cli.eval",
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]:
    importlib.import_module(mod)
PY
  echo "[stage] virtualenv is not ready at $UV_PROJECT_ENVIRONMENT"
  echo "[stage] run --stage env first"
  exit 1
fi

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' >/dev/null 2>&1; then
from pathlib import Path
import simple

repo_root = Path.cwd().resolve()
simple_path = Path(simple.__file__).resolve()
raise SystemExit(0 if repo_root in simple_path.parents else 1)
PY
  echo "[stage] editable install does not point at current workspace"
  echo "[stage] run --stage env first"
  exit 1
fi
"""


def remove_bootstrap_artifacts(simple_root: Path) -> None:
    for rel in [
        ".venv-nix",
        ".uv-cache",
        ".nix-cache",
        ".nix-bootstrap.stamp",
        ".nix-curobo.stamp",
        "third_party/curobo/build",
        "third_party/curobo/src/nvidia_curobo.egg-info",
    ]:
        path = simple_root / rel
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    for so_path in (simple_root / "third_party/curobo/src/curobo/curobolib").glob("*.so"):
        so_path.unlink(missing_ok=True)

    (simple_root / ".nix-cache").mkdir(parents=True, exist_ok=True)


def run_env_clean_rebuild(runner: Runner, roots: RuntimeRoots, env: dict[str, str]) -> bool:
    remove_bootstrap_artifacts(roots.simple_root)
    shell = """
set -euo pipefail
echo "[env_clean] creating venv at $UV_PROJECT_ENVIRONMENT"
uv venv --python "${UV_PYTHON:-python3.10}" "$UV_PROJECT_ENVIRONMENT"
echo "[env_clean] bootstrapping project"
./scripts/nix/bootstrap.sh
test -x "$UV_PROJECT_ENVIRONMENT/bin/python"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  echo "TORCH_CUDA_ARCH_LIST is not set"
  exit 1
fi
python - <<'PY'
import importlib
import os
import sys
import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert os.environ.get("TORCH_CUDA_ARCH_LIST"), "TORCH_CUDA_ARCH_LIST is unset"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
for mod in [
    "simple",
    "simple.cli.datagen",
    "simple.cli.eval",
    "openpi_client",
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]:
    importlib.import_module(mod)
print("[env_clean] imports passed")
PY
"""
    return runner.run("env clean rebuild", ["bash", "-lc", shell], cwd=roots.simple_root, env=env)


def run_datagen(runner: Runner, roots: RuntimeRoots, env: dict[str, str], env_id: str) -> Path | None:
    save_dir = roots.datagen_root / env_id.split("/", 1)[1]
    shell = f"""
{bootstrap_shell(allow_rebuild=False)}
"$UV_PROJECT_ENVIRONMENT/bin/python" -m simple.cli.datagen {env_id} \
  --num-episodes 1 \
  --save-dir {save_dir} \
  --headless \
  --render-hz 50
"""
    for attempt in range(1, DATAGEN_MAX_ATTEMPTS + 1):
        if save_dir.exists():
            shutil.rmtree(save_dir)
        log_path = runner.log_path("datagen")
        returncode, output = runner.run_capture(
            ["bash", "-lc", shell],
            cwd=roots.simple_root,
            env=env,
            prefix="RUN  datagen" if attempt == 1 else f"RUN  datagen (attempt {attempt}/{DATAGEN_MAX_ATTEMPTS})",
            log_path=log_path,
        )
        if returncode != 0:
            runner.record("datagen", "FAIL", f"{tail_detail(output)} [log: {log_path}]")
            return None

        dataset_dir = resolve_dataset_dir(save_dir, env_id)
        log_text = log_path.read_text() if log_path.exists() else ""
        if "Traceback (most recent call last):" in log_text or "RuntimeError: HF rate-limited" in log_text:
            runner.record("datagen", "FAIL", f"incomplete datagen output [log: {log_path}]")
            return None

        required = [
            dataset_dir / "meta/info.json",
        ]
        optional_markers = [
            dataset_dir / "meta/episodes.jsonl",
            dataset_dir / "meta/tasks.jsonl",
        ]
        parquet_files = list(dataset_dir.rglob("*.parquet"))
        image_files = list(dataset_dir.rglob("*.png"))
        video_files = list(dataset_dir.rglob("*.mp4"))
        has_required = all(path.is_file() for path in required)
        has_episode_metadata = any(path.is_file() for path in optional_markers)
        has_media = bool(image_files or video_files)
        if has_required and has_episode_metadata and parquet_files and has_media:
            runner.record("datagen", "PASS")
            return save_dir

        if attempt == DATAGEN_MAX_ATTEMPTS:
            if has_required and has_media and (not has_episode_metadata or not parquet_files):
                runner.record(
                    "datagen",
                    "FAIL",
                    f"partial datagen output under {dataset_dir}: media written but no saved episode metadata/parquet [log: {log_path}]",
                )
            else:
                runner.record("datagen", "FAIL", f"missing expected files under {dataset_dir} [log: {log_path}]")
            return None
    return None


def wait_for_health(url: str, tries: int = 60, sleep_s: float = 1.0) -> bool:
    import time

    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                payload = resp.read().decode("utf-8")
            if '"ok": true' in payload or '"status": "ok"' in payload or '"status":"ok"' in payload:
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(sleep_s)
    return False


def run_eval(runner: Runner, roots: RuntimeRoots, env: dict[str, str], env_id: str, data_run_dir: Path) -> None:
    if roots.eval_root.exists():
        shutil.rmtree(roots.eval_root)
    roots.eval_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = resolve_dataset_dir(data_run_dir, env_id)
    port = int(os.environ.get("SIMPLE_REGRESSION_PORT", str(pick_free_port())))
    server_log = runner.log_path("replay_policy_server")
    eval_log = runner.log_path("eval")
    Runner._prepare_log_path(server_log)
    Runner._prepare_log_path(eval_log)

    server_shell = f"""
{bootstrap_shell(allow_rebuild=False)}
"$UV_PROJECT_ENVIRONMENT/bin/python" ./scripts/tests/replay_policy_server.py \
  --data-root {data_run_dir} \
  --env-id {env_id} \
  --host 127.0.0.1 \
  --port {port}
"""
    with open(server_log, "w", buffering=1) as server_file:
        server = subprocess.Popen(
            nix_cmd(["bash", "-lc", server_shell]),
            cwd=roots.simple_root,
            env=env,
            stdout=server_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        if not wait_for_health(f"http://127.0.0.1:{port}/healthz"):
            detail = server_log.read_text() if server_log.exists() else "server did not become healthy"
            runner.record("eval replay server", "FAIL", f"{tail_detail(detail)} [log: {server_log}]")
            return
        runner.record("eval replay server", "PASS")

        eval_shell = f"""
{bootstrap_shell(allow_rebuild=False)}
python - <<'PY'
import importlib
for mod in [
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]:
    importlib.import_module(mod)
PY
"$UV_PROJECT_ENVIRONMENT/bin/python" -m simple.cli.eval {env_id} replay_policy train \
  --host 127.0.0.1 \
  --port {port} \
  --data-format lerobot \
  --data-dir {dataset_dir} \
  --eval-dir {roots.eval_root} \
  --num-episodes 1 \
  --headless
"""
        ok = runner.run("eval", ["bash", "-lc", eval_shell], cwd=roots.simple_root, env=env)
        if not ok:
            return

        eval_stats = roots.eval_root / "eval_stats.txt"
        media_root = roots.eval_root / "replay_policy/train"
        media_files = list(media_root.rglob("*.mp4")) + list(media_root.rglob("*.png"))
        eval_stats_text = eval_stats.read_text() if eval_stats.is_file() else ""
        has_episode_result = any(
            line.strip().startswith("episode_") and ":" in line
            for line in eval_stats_text.splitlines()
        )
        if not eval_stats.is_file() or not has_episode_result or not media_files:
            runner.record("eval outputs", "FAIL", f"missing expected eval outputs under {roots.eval_root}")
            return
        runner.record("eval outputs", "PASS")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIMPLE regression stages: env setup, datagen, eval, or all.")
    parser.add_argument(
        "--env-id",
        default=os.environ.get("SIMPLE_REGRESSION_ENV_ID", DEFAULT_ENV_ID),
        help="Environment id to use for datagen/eval.",
    )
    parser.add_argument(
        "--stage",
        choices=["env", "datagen", "eval", "all"],
        default="all",
        help="Which regression stage to run. Later stages reuse outputs from earlier stages.",
    )
    parser.add_argument(
        "--isolated-worktree",
        action="store_true",
        help="Run in a fresh detached worktree instead of the current checkout.",
    )
    parser.add_argument("--current-worktree", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fresh-worktree", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-worktree", action="store_true", help="Keep the detached worktree and test artifacts after the run.")
    parser.add_argument("--tmpdir", default=os.environ.get("TEST_REGRESSION_TMPDIR", "/hfm/zhenyu/tmp"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(run_git(Path(__file__).resolve().parents[1], "rev-parse", "--show-toplevel"))
    preserve_worktree = False
    use_isolated_worktree = args.isolated_worktree or args.fresh_worktree
    if use_isolated_worktree:
        tmp_parent, simple_root = create_clean_worktree(source_root, Path(args.tmpdir))
    else:
        tmp_parent = Path(args.tmpdir)
        simple_root = source_root
    roots = make_runtime_roots(source_root, simple_root, tmp_parent)
    prepare_stage(roots, args.stage)
    env = base_env(source_root, simple_root)
    runner = Runner(simple_root, roots.log_root)
    runner.banner(stage=args.stage, roots=roots, env=env)

    try:
        data_run_dir: Path | None = None
        if args.stage in {"env", "all"}:
            run_env_clean_rebuild(runner, roots, env)
            if args.stage == "env":
                return runner.summary()

        if args.stage in {"datagen", "all"}:
            data_run_dir = run_datagen(runner, roots, env, args.env_id)
            if args.stage == "datagen":
                return runner.summary()
        else:
            data_run_dir = existing_datagen_run_dir(roots, args.env_id)

        if args.stage in {"eval", "all"}:
            if data_run_dir is None:
                runner.record("eval", "SKIP", "datagen output not available; run --stage datagen first")
            else:
                run_eval(runner, roots, env, args.env_id, data_run_dir)

        return runner.summary()
    finally:
        cleanup_worktree(roots, args.keep_worktree, preserve_worktree=preserve_worktree)


if __name__ == "__main__":
    raise SystemExit(main())
