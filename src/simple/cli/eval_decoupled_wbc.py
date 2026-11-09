# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

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
import torch
import typer
import tyro
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
    ensure_cursor_restored_at_exit,
    restore_cursor,
    update_progress,
)
from simple.utils import snake_to_pascal


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


def _fetch_policy_info(host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
    """GET http://<host>:<port>/info from the policy server.

    Returns the parsed JSON (containing e.g. "policy" and "timestamp"). The
    endpoint is optional and not guaranteed to exist; if it is missing or the
    request fails for any reason, returns an empty dict so the caller falls back
    to the default output path.
    """
    import urllib.request

    url = f"http://{host}:{port}/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"No policy info from {url} ({e}); using default eval output path.")
        return {}


def _make_sonic_config() -> dict[str, Any]:
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(
        SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]
    )
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _set_render_isaac(sonic_env, agent, on_query: bool) -> int:
    """In on-query mode, render Isaac for the next env.step only if the agent will read the images
    of that step's observation (other steps get MuJoCo images). Returns 1 if Isaac renders."""
    if not on_query:
        return 0
    sonic_env.render_isaac = getattr(agent, "needs_render_next", lambda: True)()
    return int(sonic_env.render_isaac)


def _pace(step_start: float, control_dt: float) -> None:
    """Keep a control step from running faster than real time: the WBC interpolates its targets on
    the wall clock (target_time = now + control_dt), so faster steps would change its behavior."""
    sleep_time = control_dt - (time.monotonic() - step_start)
    if sleep_time > 0:
        time.sleep(sleep_time)


def _run_eval_worker_entry(
    worker_result_path: str,
    worker_id: int,
    num_workers: int,
    worker_kwargs: dict[str, Any],
    log_path: str | None,
    progress_conn: Any | None,
    episode_cursor: Any | None = None,
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
                    episode_cursor=episode_cursor,
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
    eval_dir: Annotated[str, typer.Option()] = "data/evals_decoupled_wbc",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 16,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    rollout_save_dir: Annotated[str | None, typer.Option()] = None,
    success_criteria: Annotated[float, typer.Option()] = 0.5,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    video_dir: Annotated[str, typer.Option()] = "data/evals",
    trailing_episode_steps: Annotated[int, typer.Option()] = 50,
    isaac_render: Annotated[str, typer.Option()] = "always",
    sonic_config: dict[str, Any] | None = None,
    worker_id: int = 0,
    num_workers: int = 1,
    worker_result_path: str | None = None,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    episode_cursor: Any | None = None,
):
    def persist_payload(kind: str, payload: Any):
        if not worker_result_path:
            return
        with open(worker_result_path, "wb") as f:
            pickle.dump((kind, worker_id, payload), f)

    def report(event: str, **payload: Any) -> None:
        if progress_reporter is not None:
            progress_reporter({"event": event, **payload})

    if sonic_config is None:
        sonic_config = _make_sonic_config()

    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt  # = 0.02 s (50 Hz)

    # eval videos -> <video_dir>/<policy>/<task>/<dr>, where <dr> is the trailing
    # component of --data-dir (data/SIMPLE-eval/<task>/<dr>). When the
    # policy server exposes /info, the <task> component is prefixed with the
    # exact checkpoint it serves: <video_dir>/<policy>/<server_policy>-<timestamp>.<task>/<dr>.
    dr = Path(data_dir).name
    task_component = env_id.split("/")[1]
    policy_info = _fetch_policy_info(host, port)
    server_policy = policy_info.get("policy")
    server_timestamp = policy_info.get("timestamp")
    if server_policy and server_timestamp:
        task_component = f"{server_policy}-{server_timestamp}.{task_component}"
    eval_output_dir = os.path.join(video_dir, policy, task_component, dr)
    os.makedirs(eval_output_dir, exist_ok=True)

    if rollout_save_dir and num_workers != 1:
        raise ValueError("LeRobot rollout recording only supports num_workers=1.")

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

        print("repoid+root", env_id, data_dir)
        dataset = LeRobotDataset(repo_id=env_id, root=data_dir, video_backend="pyav")
        dataset_size = dataset.num_episodes
        render_hz = dataset.meta.fps
        print(f"loaded dataset with {dataset_size} episodes.")

        def get_episode(dataset_obj, idx):
            return get_episode_lerobot(dataset_obj, idx)
    else:
        raise NotImplementedError(f"Data format {data_format} not supported YET.")

    global_episode_indices = list(range(episode_start, dataset_size))
    global_episode_indices = global_episode_indices[:num_episodes]

    # Episode durations vary wildly (an early success finishes in seconds, a
    # timeout runs the full step budget), so a fixed per-worker slice leaves fast
    # workers idle at the end while one straggler finishes. When the parent hands
    # us a shared cursor, pull the next episode on demand instead: every worker
    # stays busy until the global list is exhausted. Falls back to the static
    # round-robin slice when running solo or when no cursor was supplied.
    if episode_cursor is None:
        _static_iter = iter(global_episode_indices[worker_id::num_workers])

        def claim_next_episode() -> int | None:
            return next(_static_iter, None)
    else:

        def claim_next_episode() -> int | None:
            with episode_cursor.get_lock():
                cursor = episode_cursor.value
                if cursor >= len(global_episode_indices):
                    return None
                episode_cursor.value = cursor + 1
            return global_episode_indices[cursor]

    print(
        f"Evaluating up to {len(global_episode_indices)} episodes "
        f"(worker={worker_id}/{num_workers}, requested_total={num_episodes}, "
        f"dispatch={'dynamic' if episode_cursor is not None else 'static'})."
    )
    report(
        "worker_init",
        total_episodes=0,
        status="creating_env",
        global_total=len(global_episode_indices),
    )

    setup_start_time = time.perf_counter()
    print(f"Creating environment: {env_id}")
    make_kwargs = dict(
        sim_mode=sim_mode,
        render_hz=render_hz,
        headless=headless,
        sonic_config=sonic_config,
    )
    raw_env = gym.make(env_id, **make_kwargs)
    sonic_env = raw_env.unwrapped  # type: ignore
    if isaac_render not in ("always", "on-query"):
        raise ValueError(f"isaac_render must be 'always' or 'on-query', got {isaac_render!r}")
    on_query = isaac_render == "on-query" and sonic_env.isaac is not None
    if on_query:
        # Isaac only mirrors MuJoCo for the images, so sync it once per rendered step, not per substep
        sonic_env.sync_isaac_every_substep = False
    task = raw_env.unwrapped.task  # type: ignore[attr-defined]

    # Use provided max_episode_steps or fall back to task's metadata
    if max_episode_steps is None:
        max_episode_steps = task.metadata.get("max_episode_steps")

    # Apply TimeLimit wrapper if max_episode_steps is specified
    if max_episode_steps is not None:
        raw_env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)

    # Use provided success_criteria or fall back to task's metadata
    if success_criteria is not None:
        task.success_criteria = success_criteria
        task.metadata["success_criteria"] = success_criteria

    robot = task.robot

    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")
    agent = agent_clazz(task.robot, host, port, sonic_config=sonic_config)
    rollout_env = raw_env
    if rollout_save_dir:
        from simple.envs.lerobot import LerobotRecorder

        rollout_env = LerobotRecorder(
            env=raw_env,
            root_dir=rollout_save_dir,
            agent=agent,
        )
    setup_seconds = time.perf_counter() - setup_start_time
    report(
        "worker_init",
        total_episodes=0,
        status="ready",
        setup_seconds=setup_seconds,
        max_episode_steps=max_episode_steps,
        global_total=len(global_episode_indices),
    )

    step_update_every = 5
    stats = defaultdict(bool)
    claimed = 0

    while True:
        eps_idx = claim_next_episode()
        if eps_idx is None:
            break
        # "assigned" is only known as we go under dynamic dispatch, so grow the
        # worker's denominator each time it claims another episode.
        claimed += 1
        report("worker_init", total_episodes=claimed, status="running")

        env_conf, episode = get_episode(dataset, eps_idx)  # type: ignore[arg-type]
        task_id = f"episode_{eps_idx}"
        report("episode_start", episode=task_id)

        sonic_env.render_isaac = True
        if save_video:
            env = VideoRecorder(
                env=rollout_env,
                video_folder=eval_output_dir,
                name_prefix=task_id,
                framerate=render_hz,
                write_png=False,
            )
        else:
            env = rollout_env

        observation, info = env.reset(options={"state_dict": env_conf})

        # Reset the agent BEFORE stabilization so the WBC pipeline starts the
        # episode fresh and the upper body gets the smooth 2-second ramp to the
        # default pose on every episode (not just the first). 
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

        # engage RL policy immediately
        agent._wbc_policy.lower_body_policy.use_policy_action = True

        # --- Wait for robot to stabilize (velocity-based) ---
        sim_cnt = 0
        isaac_renders = 0
        # on-query: stabilization doesn't need images; Isaac renders the settled state once below
        sonic_env.render_isaac = not on_query
        while not robot.stabilized and sim_cnt < 300:
            step_start = time.monotonic()
            action = agent.get_stabilize_action(observation)
            observation, *_, info = sonic_env.step(action) # call the raw env step to avoid consuming the max_episode_steps limit in the TimeLimit wrapper

            sonic_env.update_viewer()
            sonic_env.update_reward()

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            sim_cnt += 1

        if on_query:
            sonic_env.render_isaac = True
            observation = sonic_env.refresh_obs()

        if isinstance(env, VideoRecorder):
            # empty the initial frames with stabilization steps
            env._init_writers(observation)
            
        frame_idx = 0
        episode_start_time = time.perf_counter()
        instruction = task.instruction

        print(
            f"Robot stabilized after {sim_cnt} simulation steps. Engaging Policy Now!"
        )

        # # TEST: stop here and move on to the next episode (skip policy eval)
        # if save_video and isinstance(env, VideoRecorder):
        #     env.release()
        # continue

        agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
            (1), dtype=torch.float32
        )

        episode_over = False
        trail = False  # only an episode that terminated (not truncated, not out of agent actions) trails
        while not episode_over:
            try:
                step_start = time.monotonic()
                action = agent.get_action(
                    observation, info=info, instruction=instruction
                )
                isaac_renders += _set_render_isaac(sonic_env, agent, on_query)
                observation, reward, terminated, truncated, info = env.step(action)
                if on_query:
                    _pace(step_start, control_dt)
                episode_over = terminated or truncated
                trail = terminated and not truncated
                frame_idx += 1
                if frame_idx == 1 or frame_idx % step_update_every == 0 or episode_over:
                    report("episode_step", episode=task_id, step=frame_idx)
            except StopIteration:
                episode_over, trail = True, False
                print("Episode finished.")

        is_success = raw_env.unwrapped._success  # type: ignore[attr-defined]
        # keep the policy running (and the videos recording) a little after the episode terminates
        # (never after a truncation, i.e. the time limit); the result is taken before, and restored
        # so the video name (_success/_failed) agrees with it
        for _ in range(trailing_episode_steps if trail else 0):
            step_start = time.monotonic()
            try:
                action = agent.get_action(observation, info=info, instruction=instruction)
            except StopIteration:
                break
            isaac_renders += _set_render_isaac(sonic_env, agent, on_query)
            observation, *_, info = env.step(action)
            if on_query:
                _pace(step_start, control_dt)
            frame_idx += 1
        raw_env.unwrapped._success = is_success  # type: ignore[attr-defined]
        stats[task_id] = is_success
        episode_seconds = time.perf_counter() - episode_start_time
        _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
        print(f"{task_id}: {frame_idx} steps in {episode_seconds:.1f} s ({frame_idx / max(episode_seconds, 1e-9):.2f} steps/s), "
              f"isaac_render={isaac_render}" + (f", {isaac_renders} Isaac-rendered steps" if on_query else ""), flush=True)
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
    sonic_config: dict[str, Any] | None = None,
    agent_factory: Callable[..., Any] | None = None,
    show_progress: bool = True,
) -> EvalResult:
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {config.num_workers}")
    if agent_factory is not None and config.num_workers != 1:
        raise ValueError("custom agent_factory is only supported with num_workers=1")

    if sonic_config is None:
        sonic_config = _make_sonic_config()

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
    rollout_save_dir = config.rollout_save_dir
    success_criteria = config.success_criteria
    save_video = config.save_video
    num_workers = config.num_workers
    video_dir = config.video_dir

    if rollout_save_dir and num_workers != 1:
        raise ValueError("LeRobot rollout recording only supports num_workers=1.")

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
    # Workers are joined after the Live display stops, and Isaac Sim writes to
    # this terminal while shutting down; make sure we restore the cursor last.
    ensure_cursor_restored_at_exit(console)
    worker_states = {wid: WorkerProgress() for wid in range(num_workers)}
    _append_eval_stats_line(eval_dir, "================\n")
    _append_eval_stats_line(eval_dir, f"run: {env_id} - {policy}\n")

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
        rollout_save_dir=rollout_save_dir,
        success_criteria=success_criteria,
        save_video=save_video,
        video_dir=video_dir,
        trailing_episode_steps=config.trailing_episode_steps,
        isaac_render=config.isaac_render,
        sonic_config=sonic_config,
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
                if terminal_stream is not None:
                    terminal_stream.close()
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
        # Shared hand-out point into the global episode list. Workers claim the
        # next index under its lock, so whoever frees up first takes the next
        # episode rather than sitting idle on an exhausted static slice.
        episode_cursor = ctx.Value("i", 0)
        # Total is only known once a worker has opened the dataset; it reports it.
        global_total: int | None = None

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
                            episode_cursor,
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
                        if payload.get("global_total") is not None:
                            global_total = int(payload["global_total"])
                        update_progress(worker_states, wid, payload)
                        if show_progress:
                            live.update(
                                render_progress(
                                    env_id,
                                    policy,
                                    worker_states,
                                    log_path,
                                    total_target=global_total,
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
    eval_dir: Annotated[str, typer.Option()] = "data/evals_decoupled_wbc",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 20,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    rollout_save_dir: Annotated[str | None, typer.Option()] = None,
    success_criteria: Annotated[float | None, typer.Option()] = None,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    num_workers: Annotated[int, typer.Option()] = 1,
    video_dir: Annotated[str, typer.Option(help="eval videos go to <video-dir>/<policy>/<task>/<dr>")] = "data/evals",
    trailing_episode_steps: Annotated[int, typer.Option(help="control steps (50 Hz) to keep running and recording after an episode terminates (not after a truncation)")] = 50,
    isaac_render: Annotated[str, typer.Option(help="always: Isaac renders every step; on-query: only steps whose images the agent reads (agent.needs_render_next), MuJoCo images fill the rest of the video")] = "always",
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
            rollout_save_dir=rollout_save_dir,
            success_criteria=success_criteria,
            save_video=save_video,
            num_workers=num_workers,
            video_dir=video_dir,
            trailing_episode_steps=trailing_episode_steps,
            isaac_render=isaac_render,
        ),
        show_progress=True if not os.environ.get("DEBUG", 0) else False,
    )


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
