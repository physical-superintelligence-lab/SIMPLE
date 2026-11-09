# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# Statuses that should show a live (animated) spinner in the STATE column.
_ACTIVE_STATUSES = {"creating_env", "ready", "running", "closing"}


# Persistent spinner per worker row so the animation frame (derived from each
# spinner's start_time) advances on Live auto-refresh instead of resetting to
# frame 0 on every render.
_state_spinners: dict[int, Spinner] = {}


@dataclass
class WorkerProgress:
    total_episodes: int = 0
    completed_episodes: int = 0
    successes: int = 0
    current_episode: str = "-"
    current_step: int = 0
    max_episode_steps: int | None = None
    status: str = "pending"
    error: str | None = None
    setup_seconds: float | None = None
    last_episode_seconds: float | None = None
    last_steps_per_second: float | None = None


def make_console(stream=None) -> Console:
    stream = sys.stderr if stream is None else stream
    return Console(file=stream, force_terminal=stream.isatty(), no_color=not stream.isatty())


def restore_cursor(console: Console) -> None:
    streams = []

    for stream in (getattr(console, "file", None), sys.stderr, sys.__stderr__, sys.stdout, sys.__stdout__):
        if stream is None:
            continue
        if stream in streams:
            continue
        streams.append(stream)

    try:
        console.show_cursor(True)
    except Exception:
        pass

    for stream in streams:
        try:
            stream.write("\x1b[?25h")
            stream.flush()
        except Exception:
            pass

    try:
        with open("/dev/tty", "w", buffering=1) as tty:
            tty.write("\x1b[?25h")
            tty.flush()
    except Exception:
        pass


_saved_tty_attrs: tuple[int, list] | None = None


def _snapshot_tty_attrs() -> None:
    """Remember the terminal's line settings before any child can change them."""
    global _saved_tty_attrs
    if _saved_tty_attrs is not None:
        return
    try:
        import termios

        fd = sys.stdin.fileno()
        if os.isatty(fd):
            _saved_tty_attrs = (fd, termios.tcgetattr(fd))
    except Exception:
        _saved_tty_attrs = None


def restore_terminal_modes() -> None:
    """Undo raw/no-echo modes a child left behind.

    Isaac Sim (via GLFW) puts the controlling terminal into a non-canonical,
    echo-off mode and does not always restore it, which leaves the shell unable
    to display anything the user types -- `stty sane` territory. Showing the
    cursor again is not enough; the line discipline has to be put back too.
    """
    if _saved_tty_attrs is None:
        return
    fd, attrs = _saved_tty_attrs
    try:
        import termios

        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except Exception:
        pass


def restore_terminal(console: Console) -> None:
    """Put the terminal back the way we found it: cursor visible, echo on."""
    restore_cursor(console)
    restore_terminal_modes()


def ensure_cursor_restored_at_exit(console: Console) -> None:
    """Restore cursor *and* line settings at interpreter exit.

    The parent stops its Live display -- and restores the cursor -- while worker
    processes are still being joined, and Isaac Sim writes to the same terminal
    as it tears down. Anything that hides the cursor or disables echo during that
    shutdown would otherwise win, leaving the terminal cursorless or silent when
    typed into. Registering here gives us the last word on every exit path:
    normal return, typer.Exit, an unhandled exception, or Ctrl-C.
    """
    _snapshot_tty_attrs()
    # Keep exactly one registration even if run_eval is invoked repeatedly.
    atexit.unregister(restore_terminal)
    atexit.register(restore_terminal, console)


def format_episode_label(value: str) -> str:
    if value.startswith("episode_"):
        return f"ep{value.split('_', 1)[1]}"
    return value


def update_progress(
    worker_states: dict[int, WorkerProgress],
    worker_id: int,
    payload: dict[str, object],
) -> None:
    state = worker_states.setdefault(worker_id, WorkerProgress())
    event = payload["event"]

    if event == "worker_init":
        state.total_episodes = int(payload["total_episodes"])
        state.status = str(payload.get("status", "running"))
        if "setup_seconds" in payload:
            state.setup_seconds = float(payload["setup_seconds"])  # type: ignore[arg-type]
        if payload.get("max_episode_steps") is not None:
            state.max_episode_steps = int(payload["max_episode_steps"])  # type: ignore[arg-type]
    elif event == "worker_status":
        state.status = str(payload["status"])
    elif event == "episode_start":
        state.current_episode = str(payload["episode"])
        state.current_step = 0
        state.status = "running"
    elif event == "episode_step":
        state.current_episode = str(payload["episode"])
        state.current_step = int(payload["step"])
        state.status = "running"
    elif event == "episode_end":
        state.current_episode = str(payload["episode"])
        state.current_step = int(payload["step"])
        state.completed_episodes = int(payload["completed_episodes"])
        state.successes = int(payload["successes"])
        if "episode_seconds" in payload:
            state.last_episode_seconds = float(payload["episode_seconds"])  # type: ignore[arg-type]
        if "steps_per_second" in payload:
            state.last_steps_per_second = float(payload["steps_per_second"])  # type: ignore[arg-type]
        state.status = "running"
    elif event == "worker_done":
        state.completed_episodes = int(payload["completed_episodes"])
        state.successes = int(payload["successes"])
        state.status = "done"
    elif event == "worker_error":
        state.status = "error"
        state.error = str(payload["message"])


def render_progress(
    env_id: str,
    policy: str,
    worker_states: dict[int, WorkerProgress],
    log_path: str,
    total_target: int | None = None,
) -> Panel:
    total_completed = sum(state.completed_episodes for state in worker_states.values())
    # Under dynamic dispatch a worker's `total_episodes` only counts what it has
    # claimed so far, so summing them understates the run until the queue drains.
    # Callers that know the real episode count pass it as `total_target`.
    total_assigned = (
        total_target
        if total_target is not None
        else sum(state.total_episodes for state in worker_states.values())
    )
    total_successes = sum(state.successes for state in worker_states.values())
    success_rate = f"{(total_successes / total_completed):.2%}" if total_completed else "n/a"
    setup_values = [state.setup_seconds for state in worker_states.values() if state.setup_seconds is not None]
    episode_values = [state.last_episode_seconds for state in worker_states.values() if state.last_episode_seconds is not None]
    sps_values = [state.last_steps_per_second for state in worker_states.values() if state.last_steps_per_second is not None]

    active_workers = sum(
        1
        for state in worker_states.values()
        if state.status in {"creating_env", "ready", "running", "closing"}
    )

    avg_setup = f"{sum(setup_values) / len(setup_values):.1f}s" if setup_values else "-"
    avg_episode = f"{sum(episode_values) / len(episode_values):.1f}s" if episode_values else "-"
    avg_sps = f"{sum(sps_values) / len(sps_values):.1f}" if sps_values else "-"

    summary = Table.grid(expand=True)
    summary.add_column(ratio=1)
    summary.add_column(justify="right")
    active = any(
        state.status in {"creating_env", "ready", "running", "closing"}
        for state in worker_states.values()
    )
    title_text = "[bold cyan]SIMPLE Eval[/bold cyan]" if active else "[bold]SIMPLE Eval[/bold]"
    summary.add_row(
        title_text,
        "",
    )
    summary.add_row(
        f"[bold]{env_id}[/bold]  [dim]policy[/dim] {policy}",
        f"[bold]{total_completed}/{total_assigned}[/bold] episodes",
    )
    summary.add_row(
        f"[dim]success[/dim] {total_successes}/{total_completed or 0} ({success_rate})",
        f"[dim]workers[/dim] {active_workers}/{len(worker_states)} active",
    )
    summary.add_row(
        f"[dim]setup[/dim] {avg_setup}  [dim]ep[/dim] {avg_episode}  [dim]step/s[/dim] {avg_sps}",
        "",
    )
    # Own full-width line: in a grid cell a long path widens its whole column and squeezes the
    # others. log_path is <--eval-dir>/eval_latest.log, so show only the eval dir.
    eval_dir_line = Text.assemble(("eval-dir ", "dim"), os.path.dirname(log_path) or ".", overflow="fold")

    workers = Table(
        expand=True,
        pad_edge=False,
        collapse_padding=True,
        box=None,
        show_edge=False,
        header_style="bold cyan",
    )
    workers.add_column("WORKER", justify="left", width=4)
    workers.add_column("STATE", justify="left", width=6, no_wrap=True)
    workers.add_column("EP", justify="left", width=4, no_wrap=True)
    workers.add_column("DONE", justify="left", width=4, no_wrap=True)
    workers.add_column("STEP", justify="left", width=9, no_wrap=True)

    status_style = {
        "pending": ("dots", "dim", "wait "),
        "creating_env": ("dots", "yellow", "setup"),
        "ready": ("dots", "cyan", "ready"),
        "running": ("dots", "cyan", "run  "),
        "closing": ("dots", "yellow", "close"),
        "done": ("dots", "green", "done "),
        "error": ("dots", "red", "error"),
    }

    for worker_id in sorted(worker_states):
        state = worker_states[worker_id]
        spinner_name, color, label = status_style.get(state.status, ("dots", "white", state.status[:5]))
        label = label.strip()
        if state.status in _ACTIVE_STATUSES:
            spinner = _state_spinners.get(worker_id)
            if spinner is None or spinner.name != spinner_name:
                spinner = Spinner(spinner_name, style=color)
                _state_spinners[worker_id] = spinner
            spinner.text = Text(label, style=color)
            spinner.style = color
            state_cell: object = spinner
        else:
            _state_spinners.pop(worker_id, None)
            state_cell = Text(label, style=color)
        worker_progress = f"{state.completed_episodes}/{state.total_episodes}" if state.total_episodes else "-"
        episode_label = format_episode_label(state.current_episode)[:5]
        if state.max_episode_steps:
            step_text = f"{state.current_step}/{state.max_episode_steps}"
        else:
            step_text = str(state.current_step) if state.current_step else "-"
        workers.add_row(
            str(worker_id),
            state_cell,
            f"[dim]{episode_label:<5}[/dim]",
            f"[dim]{worker_progress}[/dim]",
            f"[dim]{step_text}[/dim]",
        )

    return Panel(Group(summary, eval_dir_line, workers), title=Text("SIMPLE Eval"), border_style="cyan")
