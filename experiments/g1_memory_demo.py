"""Tests A and B: home memory across a restart, and an object that moved.

    python experiments/g1_memory_demo.py experiments/configs/g1_memory_demo.json [--workers 3]

Per room (real MuJoCo G1 under the WBC policy, rpp navigator, Milestone B perception):
1. ``explore``: no memory. Confirmed handles are written to a fresh SQLite home memory.
2. restart: the memory is reopened from disk in a new HomeMemory one day later.
   ``recall``: same room, MemoryGuidedPlanner; ``recall_control``: same room, no memory.
3. ``moved``: the cabinet and a crate swap places; the memory still believes the old place.
   ``moved_control``: same moved room, no memory.
Measured: success (Milestone B scoring), robot time, path, and for ``moved`` whether the
old belief became "missing" and the new location was stored.

Frame alignment across sessions: ORACLE-ASSISTED odometry from an identical start pose, so
both sessions' odometry frames coincide by construction. This is declared, not relocalised:
a SLAM relocalisation backend must replace it before different start poses are supported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import wilson_interval  # noqa: E402
from dream.memory.home import HomeMemory  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.memory_search import (  # noqa: E402
    MemoryGuidedPlanner,
    remember_inspections,
    remembered_label,
)
from dream.runtime.search import RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import Inspect  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted, Monitor, score  # noqa: E402

DAY = 86400.0
GOAL = Inspect("cabinet")


def moved_room(room):
    """Swap the cabinet with the crate farthest from it (ground truth, scene setup only)."""
    crates = [f for f in room.furniture if f.kind == "crate"]
    if not crates:
        return None
    far = max(crates, key=lambda f: np.hypot(f.xy[0] - room.cabinet.xy[0],
                                             f.xy[1] - room.cabinet.xy[1]))  # fmt: skip
    cab = dataclasses.replace(room.cabinet, xy=far.xy, yaw=far.yaw)
    crate = dataclasses.replace(far, xy=room.cabinet.xy, yaw=room.cabinet.yaw)
    furniture = tuple(crate if f is far else f for f in room.furniture)
    return dataclasses.replace(room, cabinet=cab, furniture=furniture)


def run(
    room, seed, cfg, memory: HomeMemory | None, now: float, map_id: str, record: str | None = None
) -> tuple:
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*room.start)
    monitor = Monitor(env)
    rec = None
    if record:
        from experiments.mobile_recording import MobileRecorder

        rec = MobileRecorder(
            env,
            record,
            {
                "experiment": cfg["name"],
                "seed": seed,
                "memory": memory is not None,
                "day": now / DAY,
            },
        )

    def on_tick():
        monitor.tick()
        if rec:
            rec.sample()

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    tb = MobileToolbox(
        base,
        SpatialMemory(np.random.default_rng([seed, 2])),
        navigator="rpp",
        on_event=(lambda k, t, **_: rec.event(k, t)) if rec else None,
    )
    inner = Budgeted(RoomSearchPlanner("gain"), base, cfg["time_budget_s"])
    planner = MemoryGuidedPlanner(inner, memory, map_id, now) if memory else inner
    t0 = time.perf_counter()
    log = (lambda k, d: rec.event(k, str(d)[:60])) if rec else None
    ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(GOAL, log)
    out = {"claimed": ep.success_claimed, "reason": ep.reason, "sim_s": env.time,
           "wall_s": time.perf_counter() - t0, "path_m": monitor.path_m,
           "contact_steps": monitor.contact_steps, "fell": base.stop_reason is not None,
           "tool_calls": [s.call.name for s in ep.steps],
           "memory_log": list(getattr(planner, "log", []))}  # fmt: skip
    out.update(score(env, room, tb, ep.success_claimed))
    if rec:
        for state, entry in out["memory_log"]:
            rec.event("memory", f"{state}: {entry}"[:60])
        rec.save(out)
    return out, tb, env


def room_sessions(args) -> dict:
    seed, cfg, store, record = (*args, None)[:4]
    rp = (lambda name: f"{record}/memory_{seed}_{name}") if record else (lambda name: None)
    room = sample_room(np.random.default_rng([seed, 0]))
    moved = moved_room(room)
    path = pathlib.Path(store) / f"home_{seed}.sqlite"
    path.unlink(missing_ok=True)
    map_id = f"odom-start-aligned:room{seed}"
    res = {"seed": seed, "moved_available": moved is not None}
    mem = HomeMemory(path)
    res["explore"], tb, env = run(room, seed, cfg, None, 0.0, map_id, rp("1_explore"))
    stored = remember_inspections(tb, mem, map_id, GOAL, now=env.time) if res["explore"][
        "success"] else []  # fmt: skip
    res["stored"] = stored
    mem.close()  # restart: nothing survives but the file
    mem = HomeMemory(path)
    res["recall"], _, _ = run(room, seed, cfg, mem, DAY, map_id, rp("2_recall"))
    res["recall_control"], _, _ = run(room, seed, cfg, None, DAY, map_id)
    if moved is not None:
        res["moved"], tb, env = run(moved, seed, cfg, mem, 2 * DAY, map_id, rp("3_moved"))
        if res["moved"]["success"]:
            remember_inspections(tb, mem, map_id, GOAL, now=2 * DAY + env.time)
        res["moved_control"], _, _ = run(moved, seed, cfg, None, 2 * DAY, map_id)
        res["after_moved"] = [
            {"id": b.object_id, "state": b.state, "position": b.position.tolist()}
            for b in mem.where_is(remembered_label(GOAL), map_id, now=3 * DAY)
        ]
    mem.close()
    return res


def summarise(rows: list[dict]) -> dict:
    out = {}
    for cond in ("explore", "recall", "recall_control", "moved", "moved_control"):
        rs = [r[cond] for r in rows if cond in r]
        if not rs:
            continue
        k = sum(x["success"] for x in rs)
        out[cond] = {"n": len(rs), "success": k, "success_ci95": wilson_interval(k, len(rs)),
                     "false_claims": sum(x["false_claim"] for x in rs),
                     "falls": sum(x["fell"] for x in rs),
                     "median_sim_s": float(np.median([x["sim_s"] for x in rs])),
                     "median_path_m": float(np.median([x["path_m"] for x in rs]))}  # fmt: skip
    pairs = [(r["recall"]["sim_s"], r["recall_control"]["sim_s"]) for r in rows if r["stored"]]
    out["recall_time_saved_s_median"] = (
        float(np.median([b - a for a, b in pairs])) if pairs else None
    )
    mv = [r for r in rows if "moved" in r and r["stored"]]
    out["moved_old_belief_marked_missing"] = sum(
        any(s == "absent" for s, _ in r["moved"]["memory_log"]) for r in mv
    )
    out["moved_rooms_with_stored_memory"] = len(mv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--record", help="external directory for full-state recordings")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    from experiments.provenance import code_revision

    sha, dirty = code_revision()
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    store = pathlib.Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "DREAM/memory" / cfg["name"]
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(s, cfg, str(store), args.record) for s in range(*cfg["seeds"])]
    with ProcessPoolExecutor(args.workers) as pool, open(out / "rooms.jsonl", "w") as f:
        for r in pool.map(room_sessions, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            conds = ("explore", "recall", "recall_control", "moved", "moved_control")
            line = {c: (r[c]["success"], round(r[c]["sim_s"])) for c in conds if c in r}
            print(f"room {r['seed']}: {line}", flush=True)
    summary = summarise(rows)
    meta = {"config": cfg, "git_sha": sha, "dream_dirty_at_start": dirty,
            "frames": "oracle-assisted odometry, identical start pose"}  # fmt: skip
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
