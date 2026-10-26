"""Tests A/B again, with every session starting somewhere else: memory reuse needs relocalisation.

    python experiments/g1_relocalise_demo.py experiments/configs/g1_relocalise_demo.json \
        [--workers 2]

Per fresh room (real MuJoCo G1, WBC policy, rpp navigator, Milestone B perception):
1. ``explore`` from start A, no memory. Confirmed handles go to a new SQLite home memory in
   session A's odometry frame (map id ``room<seed>:A``); the occupancy grid is saved, and a
   Spark-DSG scene graph is exported from the memory.
2. restart. ``recall`` from a different start B: after its look-around the robot relocalises
   its new odometry frame against the stored grid (``dream.memory.relocalise``) from its own
   depth only; if accepted, the memory is read through that alignment and the remembered
   handle is checked first. ``recall_control``: same start B, no memory.
3. ``moved``: cabinet and a crate swap places; start C; relocalise, check, and (only with
   visible-but-undetected looks from two viewpoints) mark the old belief missing; the new
   sighting is stored in map frame A. ``moved_control``: start C, no memory.
The simulator's true poses are used only afterwards, to measure the alignment error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.memory.home import HomeMemory  # noqa: E402
from dream.memory.relocalise import AlignedMemory, relocalise  # noqa: E402
from dream.memory.scene_graph import SceneGraphView, build_scene_graph  # noqa: E402
from dream.perception.occupancy import OccupancyGrid  # noqa: E402
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
from dream.sim.room import _footprint_ok, add_room, sample_room  # noqa: E402
from dream.tools.goals import Inspect  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted, Monitor, score  # noqa: E402
from experiments.g1_memory_demo import moved_room, summarise  # noqa: E402

DAY = 86400.0
GOAL = Inspect("cabinet")


def other_start(room, rng, avoid, min_dist=1.5):
    """A free start (same footprint rule as the room generator) far from earlier starts."""
    placed = [(f.xy[0], f.xy[1], float(np.hypot(f.half[0], f.half[1]))) for f in room.furniture]
    c = room.cabinet
    placed.append((c.xy[0], c.xy[1], float(np.hypot(c.half[0], c.half[1])) + 0.8))
    for _ in range(2000):
        xy = (float(rng.uniform(-room.width / 2, room.width / 2)),
              float(rng.uniform(-room.depth / 2, room.depth / 2)))  # fmt: skip
        if _footprint_ok(xy, 0.45, placed, room.width, room.depth, margin=0.3) and all(
            math.hypot(xy[0] - a[0], xy[1] - a[1]) >= min_dist for a in avoid
        ):
            return (xy[0], xy[1], float(rng.uniform(-math.pi, math.pi)))
    return None


class RelocalisingPlanner:
    """After the first look-around, align this session to the stored map; then use memory."""

    def __init__(self, inner, tb, memory, map_id, now, stored_grid):
        self.inner, self.tb, self.memory, self.map_id, self.now = inner, tb, memory, map_id, now
        self.stored, self.delegate, self.alignment = stored_grid, None, None
        self.log = []

    def next_action(self, goal, scene, verdict, history):
        if self.delegate is None and any(s.call.name == "look_around" for s in history):
            self.alignment = relocalise(self.stored, self.tb.memory.grid)
            if self.alignment.accepted:
                self.delegate = MemoryGuidedPlanner(
                    self.inner, AlignedMemory(self.memory, self.alignment), self.map_id,
                    self.now, last_frame=lambda: self.tb.memory.last_frame)  # fmt: skip
                self.log = self.delegate.log
            else:
                self.delegate = self.inner  # no trustworthy frame: search without memory
        return (self.delegate or self.inner).next_action(goal, scene, verdict, history)


def run(room, start, seed, cfg, memory=None, map_id=None, now=0.0, stored_grid=None):
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*start)
    monitor = Monitor(env)
    base = G1MobileBase(env, WBCLocomotion.from_cache(), monitor.tick)
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([seed, 2])), navigator="rpp")
    inner = Budgeted(RoomSearchPlanner("gain"), base, cfg["time_budget_s"])
    planner = RelocalisingPlanner(inner, tb, memory, map_id, now, stored_grid) if memory else inner
    ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(GOAL)
    out = {"start": list(start), "claimed": ep.success_claimed, "reason": ep.reason,
           "sim_s": env.time, "path_m": monitor.path_m, "contact_steps": monitor.contact_steps,
           "fell": base.stop_reason is not None,
           "tool_calls": [s.call.name for s in ep.steps],
           "memory_log": list(getattr(planner, "log", []))}  # fmt: skip
    a = getattr(planner, "alignment", None)
    if a is not None:
        out["alignment"] = {"accepted": a.accepted, "reason": a.reason, "x": a.x, "y": a.y,
                            "yaw": a.yaw, "cost_m": a.cost_m, "inliers": a.inlier_fraction,
                            "ambiguity": a.ambiguity}  # fmt: skip
    out.update(score(env, room, tb, ep.success_claimed))
    return out, tb, env, planner


def alignment_error(a: dict, start_a, start_b) -> dict:
    """Evaluator only: the estimated pose of session B's frame in A's against the truth."""
    c, s = math.cos(start_a[2]), math.sin(start_a[2])
    dx, dy = start_b[0] - start_a[0], start_b[1] - start_a[1]
    tx, ty = c * dx + s * dy, -s * dx + c * dy
    dyaw = (a["yaw"] - (start_b[2] - start_a[2]) + math.pi) % (2 * math.pi) - math.pi
    return {"position_m": math.hypot(a["x"] - tx, a["y"] - ty), "yaw_deg": abs(math.degrees(dyaw))}


def room_sessions(args) -> dict:
    seed, cfg, store = args
    room = sample_room(np.random.default_rng([seed, 0]))
    rng = np.random.default_rng([seed, 9])
    start_a = room.start
    start_b = other_start(room, rng, [start_a])
    moved = moved_room(room)
    path = pathlib.Path(store) / f"home_{seed}.sqlite"
    path.unlink(missing_ok=True)
    map_id = f"room{seed}:A"
    res = {"seed": seed, "start_a": list(start_a), "start_b": start_b}
    mem = HomeMemory(path)
    res["explore"], tb, env, _ = run(room, start_a, seed, cfg)
    stored = []
    if res["explore"]["success"]:
        stored = remember_inspections(tb, mem, map_id, GOAL, now=env.time)
    res["stored"] = stored
    np.savez_compressed(pathlib.Path(store) / f"grid_{seed}.npz", logodds=tb.memory.grid.logodds)
    graph = pathlib.Path(store) / f"dsg_{seed}.json"
    build_scene_graph(mem, map_id, env.time).save(str(graph))
    mem.close()  # restart: only the files survive
    if not stored or start_b is None:
        return res
    mem = HomeMemory(path)
    grid = OccupancyGrid(logodds=np.load(pathlib.Path(store) / f"grid_{seed}.npz")["logodds"])
    view = SceneGraphView.load(str(graph))
    res["dsg_last_seen"] = view.last_seen(remembered_label(GOAL))
    res["recall"], *_ = run(room, start_b, seed, cfg, mem, map_id, DAY, grid)
    res["recall_control"], *_ = run(room, start_b, seed, cfg)
    if "alignment" in res["recall"]:
        res["recall"]["alignment_error"] = alignment_error(
            res["recall"]["alignment"], start_a, start_b)  # fmt: skip
    if moved is not None:
        start_c = other_start(moved, rng, [start_a, start_b], min_dist=1.0)
        res["start_c"] = start_c
        if start_c is not None:
            res["moved"], tb, env, planner = run(moved, start_c, seed, cfg, mem, map_id,
                                                 2 * DAY, grid)  # fmt: skip
            if "alignment" in res["moved"]:
                res["moved"]["alignment_error"] = alignment_error(
                    res["moved"]["alignment"], start_a, start_c)  # fmt: skip
            al = getattr(planner, "alignment", None)
            if res["moved"]["success"] and al is not None and al.accepted:
                remember_inspections(tb, AlignedMemory(mem, al), map_id, GOAL,
                                     now=2 * DAY + env.time)  # fmt: skip
            res["moved_control"], *_ = run(moved, start_c, seed, cfg)
            res["after_moved"] = [
                {"id": b.object_id, "state": b.state, "position": b.position.tolist()}
                for b in mem.where_is(remembered_label(GOAL), map_id, now=3 * DAY)
            ]
    mem.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    from experiments.provenance import code_revision

    sha, dirty = code_revision()
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    store = pathlib.Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "DREAM/memory" / cfg["name"]
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(s, cfg, str(store)) for s in range(*cfg["seeds"])]
    with ProcessPoolExecutor(args.workers) as pool, open(out / "rooms.jsonl", "w") as f:
        for r in pool.map(room_sessions, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            line = {c: (r[c]["success"], round(r[c]["sim_s"]), r[c].get("alignment", {}).get(
                "reason")) for c in ("explore", "recall", "recall_control", "moved",
                                     "moved_control") if c in r}  # fmt: skip
            print(f"room {r['seed']}: {line}", flush=True)
    summary = summarise(rows)
    for cond in ("recall", "moved"):
        al = [r[cond] for r in rows if cond in r and "alignment" in r[cond]]
        errs = [x["alignment_error"] for x in al if x["alignment"]["accepted"]]
        summary[f"{cond}_alignment"] = {
            "attempted": len(al), "accepted": len(errs),
            "refused": {x["alignment"]["reason"] for x in al if not x["alignment"]["accepted"]},
            "position_err_m": [e["position_m"] for e in errs],
            "yaw_err_deg": [e["yaw_deg"] for e in errs]}  # fmt: skip
        summary[f"{cond}_absences"] = sum(
            any(s == "absent" for s, _ in r[cond]["memory_log"]) for r in rows if cond in r
        )
    meta = {
        "config": cfg,
        "git_sha": sha,
        "dream_dirty_at_start": dirty,
        "frames": "different start poses; relocalised from depth; truth used only to score",
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2,
                                                 default=list))  # fmt: skip
    print(json.dumps(summary, indent=2, default=list))


if __name__ == "__main__":
    main()
