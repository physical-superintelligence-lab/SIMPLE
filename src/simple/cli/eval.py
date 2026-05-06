from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing.connection import wait

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import requests
import torch  # noqa: F401
import typer
from gymnasium.wrappers import TimeLimit
from rich.console import Group
from rich.live import Live
from typing_extensions import Annotated

import simple.envs as _  # noqa: F401
from simple.envs.wrappers import VideoRecorder
from simple.evals.api import EvalConfig, EvalResult
from simple.evals.tui import (
    WorkerProgress,
    make_console,
    render_progress,
    restore_cursor,
    update_progress,
)
from simple.utils import snake_to_pascal


def _check_server_health(host: str, port: int, policy: str) -> None:
    if policy not in {"hrdt", "egovla"}:
        return

    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url, timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Policy server health check failed for {url}") from exc


def _append_eval_stats_line(eval_dir: str, line: str) -> None:
    path = Path(eval_dir) / "eval_stats.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _redirect_stdio_to_log(log_path: str | None):
    if not log_path:
        yield
        return

    log_file = open(log_path, "a", buffering=1)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    redirected_stdout = None
    redirected_stderr = None

    try:
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        redirected_stdout = os.fdopen(os.dup(1), "w", buffering=1)
        redirected_stderr = os.fdopen(os.dup(2), "w", buffering=1)
        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if redirected_stdout is not None:
            redirected_stdout.close()
        if redirected_stderr is not None:
            redirected_stderr.close()

        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        log_file.close()


def _run_eval_worker_entry(
    worker_result_path: str,
    worker_id: int,
    num_workers: int,
    worker_kwargs: dict[str, Any],
    log_path: str | None,
    progress_conn: Any | None,
):
    def report(payload: dict[str, Any]) -> None:
        if progress_conn is not None:
            progress_conn.send((worker_id, payload))

    try:
        with _redirect_stdio_to_log(log_path):
            try:
                _run_eval_worker(
                    **worker_kwargs,
                    worker_id=worker_id,
                    num_workers=num_workers,
                    worker_result_path=worker_result_path,
                    progress_reporter=report,
                )
            except Exception:
                report(
                    {
                        "event": "worker_error",
                        "message": "see eval log",
                    }
                )
                with open(worker_result_path, "wb") as f:
                    pickle.dump(("err", worker_id, traceback.format_exc()), f)
    finally:
        if progress_conn is not None:
            try:
                progress_conn.close()
            except Exception:
                pass


def _run_eval_worker(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "train",
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "rlds_numpy",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 100,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    success_criteria: Annotated[float, typer.Option()] = 0.9,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    worker_id: int = 0,
    num_workers: int = 1,
    worker_result_path: str | None = None,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
):
    def persist_payload(kind: str, payload: Any):
        if not worker_result_path:
            return
        with open(worker_result_path, "wb") as f:
            pickle.dump((kind, worker_id, payload), f)

    def report(event: str, **payload: Any) -> None:
        if progress_reporter is not None:
            progress_reporter({"event": event, **payload})

    eval_output_dir = os.path.join(eval_dir, policy, env_id.split("/")[1], split)
    os.makedirs(eval_output_dir, exist_ok=True)

    if episode_start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}")

    if data_format == "rlds_numpy":
        data_path = Path(eval_dir) / env_id / split
        assert data_path.exists(), f"Data path {data_path} does not exist."
        dataset = sorted(list(data_path.glob("*episode_*.pkl")))
        dataset_size = len(dataset)
        render_hz = 3

        from simple.datasets.rlds import convert_env_config_to_state_dict

        def get_episode(dataset_obj, idx):
            with open(dataset_obj[idx], "rb") as f:
                episode = pickle.load(f)
            env_cfg = json.loads(
                episode["task"]["environment_config"][0].decode("utf-8")
            )
            state_dict = convert_env_config_to_state_dict(env_cfg)
            state_dict["camera_info"] = env_cfg["scene_info"]["camera_info"]
            return state_dict, episode

    elif data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from simple.datasets.lerobot import get_episode_lerobot

        dataset = LeRobotDataset(repo_id=env_id, root=data_dir)
        dataset_size = dataset.num_episodes
        render_hz = dataset.meta.fps
        print(f"loaded dataset with {dataset_size} episodes.")

        def get_episode(dataset_obj, idx):
            return get_episode_lerobot(dataset_obj, idx)

    else:
        raise NotImplementedError(f"Data format {data_format} not supported YET.")

    global_episode_indices = list(range(episode_start, dataset_size))
    global_episode_indices = global_episode_indices[:num_episodes]
    episode_indices = global_episode_indices[worker_id::num_workers]
    print(
        f"Evaluating {len(episode_indices)} episodes "
        f"(worker={worker_id}/{num_workers}, requested_total={num_episodes}, "
        f"assigned={len(episode_indices)})."
    )
    report("worker_init", total_episodes=len(episode_indices), status="creating_env")

    setup_start_time = time.perf_counter()
    make_kwargs = dict(
        sim_mode=sim_mode,
        render_hz=render_hz,
        headless=headless,
        success_criteria=success_criteria,
    )
    raw_env = gym.make(env_id, **make_kwargs)
    task = raw_env.unwrapped.task  # type: ignore[attr-defined]

    # Use provided max_episode_steps or fall back to task's metadata
    if max_episode_steps is None:
        max_episode_steps = task.metadata.get("max_episode_steps")
        print("max_episode_steps", max_episode_steps)

    # Apply TimeLimit wrapper if max_episode_steps is specified
    if max_episode_steps is not None:
        raw_env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)

    print("raw_env: ", raw_env)
    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")
    agent = agent_clazz(task.robot, host, port)
    setup_seconds = time.perf_counter() - setup_start_time
    report(
        "worker_init",
        total_episodes=len(episode_indices),
        status="ready",
        setup_seconds=setup_seconds,
    )

    step_update_every = 5
    stats = defaultdict(bool)

    for eps_idx in episode_indices:
        env_conf, episode = get_episode(dataset, eps_idx)  # type: ignore[arg-type]
        task_id = f"episode_{eps_idx}"
        report("episode_start", episode=task_id)

        if save_video:
            env = VideoRecorder(
                env=raw_env,
                video_folder=eval_output_dir,
                name_prefix=task_id,
                framerate=render_hz,
                write_png=False,
            )
        else:
            env = raw_env

        observation, info = env.reset(options={"state_dict": env_conf})
        frame_idx = 0
        episode_start_time = time.perf_counter()
        instruction = task.instruction

        if policy == "vlt":
            reset_kwargs = {
                "camera_infos": env_conf["camera_info"],
                "episode": episode,
                "condition": "forward_all",
                "save_cond_images": True,
            }
        else:
            reset_kwargs = {}

        agent.reset(**reset_kwargs)
        episode_over = False
        while not episode_over:
            try:
                action = agent.get_action(
                    observation, info=info, instruction=instruction
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                frame_idx += 1
                if frame_idx == 1 or frame_idx % step_update_every == 0 or episode_over:
                    report("episode_step", episode=task_id, step=frame_idx)
            except StopIteration:
                episode_over = True
                print("Episode finished.")

        is_success = raw_env.unwrapped._success  # type: ignore[attr-defined]
        stats[task_id] = is_success
        episode_seconds = time.perf_counter() - episode_start_time
        _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
        report(
            "episode_end",
            episode=task_id,
            step=frame_idx,
            completed_episodes=len(stats),
            successes=sum(stats.values()),
            episode_seconds=episode_seconds,
            steps_per_second=(
                (frame_idx / episode_seconds) if episode_seconds > 0 else 0.0
            ),
        )

        if save_video and isinstance(env, VideoRecorder):
            env.release()

    persist_payload("ok", dict(stats))
    report("worker_status", status="closing")
    raw_env.close()
    persist_payload("ok", dict(stats))
    report(
        "worker_done",
        completed_episodes=len(stats),
        successes=sum(stats.values()),
    )
    return stats


def run_eval(
    config: EvalConfig,
    *,
    agent_factory: Callable[..., Any] | None = None,
    show_progress: bool = True,
) -> EvalResult:
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {config.num_workers}")
    if agent_factory is not None and config.num_workers != 1:
        raise ValueError("custom agent_factory is only supported with num_workers=1")

    eval_dir = config.eval_dir
    env_id = config.env_id
    policy = config.policy
    split = config.split
    host = config.host
    port = config.port
    data_format = config.data_format
    sim_mode = config.sim_mode
    headless = config.headless
    max_episode_steps = config.max_episode_steps
    num_episodes = config.num_episodes
    episode_start = config.episode_start
    data_dir = config.data_dir
    success_criteria = config.success_criteria
    save_video = config.save_video
    num_workers = config.num_workers

    if os.environ.get("SIMPLE_DISABLE_TUI", "").lower() in {"1", "true", "yes", "on"}:
        show_progress = False

    eval_dir_path = Path(config.eval_dir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = str(eval_dir_path / "eval_latest.log")
    Path(log_path).write_text("")
    worker_log_paths = {
        wid: str(eval_dir_path / f"eval_worker_{wid}.log") for wid in range(num_workers)
    }
    for worker_log_path in worker_log_paths.values():
        Path(worker_log_path).write_text("")

    terminal_stream = None
    if num_workers == 1 and show_progress:
        terminal_stream = os.fdopen(os.dup(2), "w", buffering=1)
    console = make_console(terminal_stream)
    worker_states = {wid: WorkerProgress() for wid in range(num_workers)}
    _append_eval_stats_line(eval_dir, "================\n")
    _append_eval_stats_line(eval_dir, f"run: {env_id} - {policy}\n")

    _check_server_health(host, port, policy)

    worker_kwargs = dict(
        env_id=env_id,
        policy=policy,
        split=split,
        host=host,
        port=port,
        data_format=data_format,
        sim_mode=sim_mode,
        headless=headless,
        eval_dir=eval_dir,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
        episode_start=episode_start,
        data_dir=data_dir,
        success_criteria=success_criteria,
        save_video=save_video,
    )

    console.print(f"Writing eval logs to [bold]{log_path}[/bold]")

    if num_workers == 1:
        stats: dict[str, bool]
        if show_progress:
            try:
                with (
                    Live(
                        render_progress(env_id, policy, worker_states, log_path),
                        console=console,
                        refresh_per_second=4,
                    ) as live,
                    _redirect_stdio_to_log(log_path),
                ):

                    def report(payload: dict[str, Any]) -> None:
                        update_progress(worker_states, 0, payload)
                        live.update(
                            render_progress(env_id, policy, worker_states, log_path),
                            refresh=False,
                        )

                    stats = _run_eval_worker(
                        **worker_kwargs,
                        worker_id=0,
                        num_workers=1,
                        progress_reporter=report,
                    )
            finally:
                restore_cursor(console)
        else:
            stats = _run_eval_worker(
                **worker_kwargs,
                worker_id=0,
                num_workers=1,
            )
    else:
        ctx = mp.get_context("spawn")
        progress_readers: dict[int, Any] = {}
        procs: list[mp.Process] = []
        result_dir = Path(tempfile.mkdtemp(prefix="simple_eval_workers_"))

        try:
            with Live(
                render_progress(env_id, policy, worker_states, log_path),
                console=console,
                refresh_per_second=4,
                auto_refresh=show_progress,
            ) as live:
                for wid in range(num_workers):
                    worker_result_path = str(result_dir / f"worker_{wid}.pkl")
                    recv_conn, send_conn = ctx.Pipe(duplex=False)
                    p = ctx.Process(
                        target=_run_eval_worker_entry,
                        args=(
                            worker_result_path,
                            wid,
                            num_workers,
                            worker_kwargs,
                            worker_log_paths[wid],
                            send_conn,
                        ),
                        name=f"eval-worker-{wid}",
                    )
                    p.start()
                    send_conn.close()
                    progress_readers[wid] = recv_conn
                    procs.append(p)

                while any(p.is_alive() for p in procs) or progress_readers:
                    ready = (
                        wait(list(progress_readers.values()), timeout=0.2)
                        if progress_readers
                        else []
                    )
                    for conn in ready:
                        try:
                            wid, payload = conn.recv()
                        except EOFError:
                            for key, value in list(progress_readers.items()):
                                if value is conn:
                                    value.close()
                                    del progress_readers[key]
                                    break
                            continue
                        update_progress(worker_states, wid, payload)
                        if show_progress:
                            live.update(
                                render_progress(
                                    env_id, policy, worker_states, log_path
                                ),
                                refresh=False,
                            )
        finally:
            restore_cursor(console)
            if terminal_stream is not None:
                terminal_stream.close()
            for conn in progress_readers.values():
                try:
                    conn.close()
                except Exception:
                    pass

        stats = {}
        results_by_worker: dict[int, dict[str, bool]] = {}
        errors_by_worker: dict[int, str] = {}
        failed_exit: list[tuple[int, int]] = []

        for wid, p in enumerate(procs):
            worker_result_path = result_dir / f"worker_{wid}.pkl"
            p.join()
            exit_code = p.exitcode if p.exitcode is not None else 999
            if exit_code != 0:
                failed_exit.append((wid, exit_code))
            if worker_result_path.exists():
                try:
                    with open(worker_result_path, "rb") as f:
                        kind, got_wid, payload = pickle.load(f)
                    if got_wid != wid:
                        errors_by_worker[wid] = (
                            f"mismatched worker payload id={got_wid}"
                        )
                    elif kind == "ok":
                        results_by_worker[wid] = payload
                    else:
                        errors_by_worker[wid] = payload
                except Exception:
                    errors_by_worker[wid] = traceback.format_exc()

        for wid in sorted(results_by_worker):
            worker_stats = results_by_worker[wid]
            overlap = set(worker_stats).intersection(stats)
            if overlap:
                raise RuntimeError(
                    f"Duplicate episode ids across workers: {sorted(overlap)}"
                )
            stats.update(worker_stats)

        if errors_by_worker or failed_exit:
            summary = []
            if errors_by_worker:
                summary.append(f"python_errors={sorted(errors_by_worker)}")
            if failed_exit:
                summary.append(f"nonzero_exit={failed_exit}")
            for wid in sorted(errors_by_worker):
                console.print(
                    f"[red]worker {wid} traceback[/red]\n{errors_by_worker[wid]}\n"
                    f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                )
            if failed_exit:
                console.print("[yellow]worker native exits[/yellow]")
                for wid, exit_code in failed_exit:
                    console.print(
                        f"[yellow]worker {wid} exit {exit_code}[/yellow] "
                        f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                    )
            console.print(
                "[red]Parallel eval worker failure[/red] "
                + "("
                + ", ".join(summary)
                + f"). See {log_path}"
            )
            raise typer.Exit(code=1)

        missing = sorted(
            set(range(num_workers)) - set(results_by_worker) - set(errors_by_worker)
        )
        if missing:
            console.print(
                "[red]Parallel eval worker missing result payload[/red] "
                f"(workers={missing}, exits={failed_exit}). See {log_path}"
            )
            raise typer.Exit(code=1)

    sr = sum(stats.values()) / len(stats) if stats else 0.0
    console.print(f"Success rate {env_id} - {policy}: {sr:.2%}")
    console.print(f"Eval log: {log_path}")

    _append_eval_stats_line(eval_dir, f"success rate: {sr:.2f} \n")
    return EvalResult(
        env_id=env_id,
        policy=policy,
        split=split,
        stats=dict(stats),
        success_rate=sr,
        log_path=log_path,
        eval_dir=eval_dir,
    )


def main(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "train",
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "rlds_numpy",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 100,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    success_criteria: Annotated[float, typer.Option()] = 0.9,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    num_workers: Annotated[int, typer.Option()] = 1,
):
    run_eval(
        EvalConfig(
            env_id=env_id,
            policy=policy,
            split=split,
            host=host,
            port=port,
            data_format=data_format,
            sim_mode=sim_mode,
            headless=headless,
            eval_dir=eval_dir,
            max_episode_steps=max_episode_steps,
            num_episodes=num_episodes,
            episode_start=episode_start,
            data_dir=data_dir,
            success_criteria=success_criteria,
            save_video=save_video,
            num_workers=num_workers,
        )
    )


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
