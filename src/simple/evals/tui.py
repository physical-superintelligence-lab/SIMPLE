from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class WorkerProgress:
    total_episodes: int = 0
    completed_episodes: int = 0
    successes: int = 0
    current_episode: str = "-"
    current_step: int = 0
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
) -> Panel:
    total_completed = sum(state.completed_episodes for state in worker_states.values())
    total_assigned = sum(state.total_episodes for state in worker_states.values())
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
        f"[dim]log[/dim] {log_path}",
    )

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
    workers.add_column("STEP", justify="left", width=4, no_wrap=True)

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
        _spinner_name, color, label = status_style.get(state.status, ("dots", "white", state.status[:5]))
        state_cell = f"[{color}]{label}[/{color}]"
        worker_progress = f"{state.completed_episodes}/{state.total_episodes}" if state.total_episodes else "-"
        episode_label = format_episode_label(state.current_episode)[:5]
        step_text = str(state.current_step) if state.current_step else "-"
        workers.add_row(
            str(worker_id),
            state_cell,
            f"[dim]{episode_label:<5}[/dim]",
            f"[dim]{worker_progress}[/dim]",
            f"[dim]{step_text}[/dim]",
        )

    return Panel(Group(summary, workers), title=Text("SIMPLE Eval"), border_style="cyan")
