#!/usr/bin/env python3
"""Merge the 17 per-task SIMPLE decoupled-WBC LeRobot v2.1 datasets (``data/data-fixed/<Task>``)
into one dataset (HF ``SIMPLE-org/SIMPLE-decoupled-wbc``).

Conventions (same as the released merge):
* sources appended in alphabetical order, episodes renumbered 0..N-1, ``index`` rewritten to a
  global 0..F-1, ``task_index`` = source position;
* ``episodes.jsonl``: ``dataset_from_index`` / ``dataset_to_index`` (inclusive) are global;
* ``episodes_stats.jsonl``: action + timestamp stats recomputed from each full episode;
* ``meta/stats.json`` (and identical ``stats_psi0.json``) recomputed over all frames;
* ``info.json`` features come from ``--info-template`` (joint names), totals rewritten.

    python scripts/merge_simple_decoupled_wbc.py --src data/data-fixed --out /hfm/data/simple/SIMPLE-decoupled-wbc \
        --info-template /path/to/old/meta/info.json
    python scripts/merge_simple_decoupled_wbc.py --check-stats /hfm/data/simple/SIMPLE-decoupled-wbc  # recompute+compare
    # subset, e.g. only the 5 motion-planning (AMO) tasks:
    python scripts/merge_simple_decoupled_wbc.py --src data/data-fixed --out /hfm/data/simple/SIMPLE-decoupled-wbc-MP \
        --info-template /path/to/old/meta/info.json --tasks G1WholebodyBendPickMP-v0 ...
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1000
NUMERIC = ["observation.hand_joints", "observation.arm_joints", "observation.leg_joints",
           "observation.prev_torso_rpy", "observation.prev_height", "states", "action",
           "timestamp", "frame_index", "episode_index", "index", "next.done", "task_index"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def col_array(table: pa.Table, name: str) -> np.ndarray:
    c = table.column(name)
    if pa.types.is_list(c.type) or pa.types.is_large_list(c.type):
        return np.asarray(c.combine_chunks().flatten().to_numpy(zero_copy_only=False),
                          dtype=np.float64).reshape(table.num_rows, -1)
    return np.asarray(c.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1)


def stats_of(x: np.ndarray, with_count: bool = False) -> dict:
    d = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0),
         "q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0)}
    d = {k: v.astype(np.float32).tolist() for k, v in d.items()}
    if with_count:
        d["count"] = [int(x.shape[0])]
    return d


def episode_paths(root: Path, ep: int) -> tuple[Path, Path]:
    ch = ep // CHUNK
    return (root / f"data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
            root / f"videos/chunk-{ch:03d}/egocentric/episode_{ep:06d}.mp4")


def full_stats(root: Path, n_eps: int) -> dict:
    cols: dict[str, list] = {k: [] for k in NUMERIC}
    for ep in range(n_eps):
        t = pq.read_table(episode_paths(root, ep)[0])
        for k in NUMERIC:
            cols[k].append(col_array(t, k))
    return {k: stats_of(np.concatenate(v)) for k, v in cols.items()}


def merge(src_root: Path, out: Path, info_template: Path, tasks: list[str] | None = None) -> None:
    srcs = sorted(p for p in src_root.iterdir()
                  if p.is_dir() and p.name.endswith("-v0") and (p / "meta/info.json").exists())
    if tasks:  # merge only the named tasks, e.g. a per-embodiment subset
        missing = sorted(set(tasks) - {s.name for s in srcs})
        assert not missing, f"not found under {src_root}: {missing}"
        srcs = [s for s in srcs if s.name in set(tasks)]
    else:
        assert len(srcs) == 17, [s.name for s in srcs]
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    (out / "meta").mkdir(parents=True)
    ref_meta = {f: (srcs[0] / "meta" / f).read_text() for f in ("modality.json", "lang_map.json", "relative_stats.json")}

    ep_rows, ep_stats, tasks, per_task = [], [], [], []
    new_ep, gidx = 0, 0
    for ti, src in enumerate(srcs):
        for f, txt in ref_meta.items():
            assert (src / "meta" / f).read_text() == txt, f"{src.name}: {f} differs"
        st = load_jsonl(src / "meta/tasks.jsonl")
        assert len(st) == 1, src
        task = st[0]["task"]
        tasks.append({"task_index": ti, "task": task, "category": "", "description": task})
        eps = sorted(load_jsonl(src / "meta/episodes.jsonl"), key=lambda r: r["episode_index"])
        n_frames = 0
        for row in eps:
            old = row["episode_index"]
            p_src, v_src = episode_paths(src, old)
            t = pq.read_table(p_src)
            L = t.num_rows
            assert L == row["length"], (src.name, old, L, row["length"])
            fi = t.column("frame_index").to_numpy()
            assert (fi == np.arange(L)).all(), (src.name, old)
            assert set(t.column("episode_index").to_pylist()) == {old}, (src.name, old)
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                             pa.array(np.full(L, new_ep, dtype=np.int64)))
            t = t.set_column(t.schema.get_field_index("index"), "index",
                             pa.array(np.arange(gidx, gidx + L, dtype=np.int64)))
            t = t.set_column(t.schema.get_field_index("task_index"), "task_index",
                             pa.array(np.full(L, ti, dtype=np.int64)))
            p_dst, v_dst = episode_paths(out, new_ep)
            p_dst.parent.mkdir(parents=True, exist_ok=True)
            v_dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, p_dst)
            shutil.copy2(v_src, v_dst)

            r = dict(row)
            r["episode_index"] = new_ep
            r["tasks"] = [ti]
            r["dataset_from_index"] = gidx
            r["dataset_to_index"] = gidx + L - 1
            if isinstance(r.get("instruction"), dict):
                r["instruction"] = {"task_index": ti, "task": task}
            ep_rows.append(r)
            ep_stats.append({"episode_index": new_ep, "stats": {
                "action": stats_of(col_array(t, "action"), True),
                "timestamp": stats_of(col_array(t, "timestamp"), True)}})
            new_ep += 1
            gidx += L
            n_frames += L
        per_task.append((ti, src.name, task, len(eps), n_frames))
        print(f"[{ti:2d}] {src.name}: {len(eps)} eps, {n_frames} frames")

    meta = out / "meta"
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_rows))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_stats))
    (meta / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
    for f, txt in ref_meta.items():
        (meta / f).write_text(txt)
    info = json.loads(info_template.read_text())
    info.update(total_episodes=new_ep, total_frames=gidx, total_tasks=len(tasks), total_videos=new_ep,
                total_chunks=(new_ep - 1) // CHUNK + 1, chunks_size=CHUNK)
    if "splits" in info:
        info["splits"] = {"train": f"0:{new_ep}"}
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    stats = json.dumps(full_stats(out, new_ep), indent=4)
    (meta / "stats.json").write_text(stats)
    (meta / "stats_psi0.json").write_text(stats)
    (out / "merge_summary.json").write_text(json.dumps(
        {"episodes": new_ep, "frames": gidx, "tasks": [dict(zip(["task_index", "source", "task", "episodes", "frames"], p)) for p in per_task]}, indent=2))
    print(f"done: {new_ep} episodes, {gidx} frames -> {out}")


def check_stats(root: Path) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    new = full_stats(root, info["total_episodes"])
    old = json.loads((root / "meta/stats.json").read_text())
    worst = 0.0
    for k in NUMERIC:
        for s in new[k]:
            a, b = np.array(new[k][s]), np.array(old[k][s])
            d = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))
            worst = max(worst, d)
            if d > 1e-4:
                print("MISMATCH", k, s, d)
    print("max rel diff", worst)
    rows = load_jsonl(root / "meta/episodes_stats.jsonl")
    for r in rows[:: max(1, len(rows) // 20)]:
        t = pq.read_table(episode_paths(root, r["episode_index"])[0])
        for k in ("action", "timestamp"):
            got = stats_of(col_array(t, k), True)
            for s, v in r["stats"][k].items():
                if not np.allclose(got[s], v, rtol=1e-5, atol=1e-6):
                    print("EP MISMATCH", r["episode_index"], k, s)
    print("episode stats checked")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--info-template", type=Path)
    ap.add_argument("--check-stats", type=Path)
    ap.add_argument("--tasks", nargs="+", help="only merge these task dirs under --src (default: all 17)")
    a = ap.parse_args()
    if a.check_stats:
        check_stats(a.check_stats)
    else:
        merge(a.src, a.out, a.info_template, a.tasks)
