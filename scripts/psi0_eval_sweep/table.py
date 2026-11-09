# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Rewrite LOG_ROOT/success_table.txt every 30 s from per-episode lines in eval_stats.txt.

Cell = successes/episodes-finished; suffix '*' running, '!' failed (nonzero exit or < expected
episodes), '-' queued. Exits once every job has finished.
usage: table.py LOG_ROOT TITLE EXPECTED_EPISODES
"""
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root, title, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
jobs = [tuple(l.split()) for l in (root / "jobs.txt").read_text().splitlines() if l.strip()]
LEVELS = list(dict.fromkeys(d for _, d in jobs))
tasks = sorted({t for t, _ in jobs})
EP = re.compile(r"^(?:episode_)?\d+: (True|False)\s*$")


def cell(task, dr):
    st = root / "status" / f"{task}__{dr}"
    if not st.exists():
        return None, 0, 0
    s = st.read_text()
    state = "running" if s.startswith("running") else "done"
    eval_dir = Path(re.search(r"eval_dir=(\S+)", s).group(1))
    f = (root.parent.parent / eval_dir / "eval_stats.txt") if not eval_dir.is_absolute() else eval_dir / "eval_stats.txt"
    res = [m.group(1) == "True" for l in (f.read_text().splitlines() if f.exists() else []) if (m := EP.match(l.strip()))]
    if state == "done" and ("rc=0" not in s or len(res) < expected):
        state = "failed"
    return state, sum(res), len(res)


while True:
    w = max(len(t) for t in tasks) + 2
    rows, lvl_tot = [], {d: [0, 0] for d in LEVELS}
    n_state = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for t in tasks:
        cells, ts, tn = [], 0, 0
        for d in LEVELS:
            state, s, n = cell(t, d)
            n_state[state or "queued"] += 1
            if state is None:
                cells.append("-")
                continue
            ts, tn = ts + s, tn + n
            lvl_tot[d][0] += s
            lvl_tot[d][1] += n
            cells.append(f"{s}/{n}" + {"running": "*", "failed": "!", "done": ""}[state])
        pct = f"{ts}/{tn} ({ts / tn:.0%})" if tn else "-"
        rows.append(f"{t:<{w}}" + "".join(f"{c:>15}" for c in cells) + f"{pct:>16}")
    S = sum(v[0] for v in lvl_tot.values())
    N = sum(v[1] for v in lvl_tot.values())
    lvl = "".join(f"{(f'{s}/{n} ({s / n:.0%})' if n else '-'):>15}" for s, n in lvl_tot.values())
    sep = "-" * (w + 15 * len(LEVELS) + 16)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = [
        f"{title}",
        f"updated {now} | jobs: {n_state['done']} done, {n_state['failed']} failed, "
        f"{n_state['running']} running, {n_state['queued']} queued (of {len(jobs)})",
        "cell = successes/episodes finished;  * running   ! failed   - queued",
        "",
        f"{'Task':<{w}}" + "".join(f"{d:>15}" for d in LEVELS) + f"{'Total':>16}",
        sep, *rows, sep,
        f"{'All tasks':<{w}}" + lvl + f"{(f'{S}/{N} ({S / N:.1%})' if N else '-'):>16}",
    ]
    tmp = root / "success_table.txt.tmp"
    tmp.write_text("\n".join(out) + "\n")
    tmp.replace(root / "success_table.txt")
    if n_state["queued"] == 0 and n_state["running"] == 0:
        break
    time.sleep(30)
