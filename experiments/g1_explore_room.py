"""Milestone B: a walking G1 enters an unfamiliar room, finds the cabinet and stands where it
can examine it, deciding what to look at next from its own head camera and odometry.

    python experiments/g1_explore_room.py experiments/configs/g1_explore_room.json [--workers 4]

Agent: TaskAgent + RoomSearchPlanner over MobileToolbox (dream/tools/mobile.py), on NVIDIA's
pretrained WBC walking policy. It never sees the room spec, poses, identities or mechanisms.
Odometry is ORACLE-ASSISTED (true increments with noise and bias; dream/sim/g1_mobile.py).

Scoring here is independent and physical:
- success: the agent claims it AND the robot truly stands 0.5-1.5 m from the true handle,
  on the cabinet's front side, facing it within 40 degrees, AND the handle the agent reports
  is the real one: its position relative to the robot (odometry frame, so drift cancels)
  is within 0.3 m of the true handle relative to the robot's true pose;
- false claim: a success claim that fails that check (any claim in a room with no cabinet);
- correct refusal: no claim in a room with no cabinet;
- falls, robot contacts with walls/furniture (physics steps in contact), time, path length.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import atan2, cos, sin

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import wilson_interval  # noqa: E402
from dream.navigation.follow import wrap  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import Finish, TaskAgent  # noqa: E402
from dream.runtime.search import RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import Inspect  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402


class Budgeted:
    """Stop the run when the robot-time budget is spent (an episode limit, not a strategy)."""

    def __init__(self, planner, base, seconds: float) -> None:
        self.planner, self.base, self.seconds = planner, base, seconds

    def next_action(self, goal, scene, verdict, history):
        if not verdict.get("satisfied") and self.base.time > self.seconds:
            return Finish(False, f"time budget ({self.seconds:.0f} s) spent")
        return self.planner.next_action(goal, scene, verdict, history)


class Monitor:
    """Evaluator: robot contacts with room geometry and the true path length."""

    def __init__(self, env: G1MobileEnv) -> None:
        m = env.model
        root = m.body("pelvis").id
        robot = {b for b in range(m.nbody) if _ancestor(m, b, root)}
        self.robot_geom = np.array([m.geom_bodyid[g] in robot for g in range(m.ngeom)])
        self.floor = m.geom("floor").id
        self.env, self.contact_steps, self.path_m = env, 0, 0.0
        self.last = env.true_pose()[:2]

    def tick(self) -> None:
        d = self.env.data
        for c in d.contact[: d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if self.floor in (g1, g2):
                continue
            if self.robot_geom[g1] != self.robot_geom[g2]:
                self.contact_steps += 1
                break
        p = self.env.true_pose()[:2]
        self.path_m += float(np.linalg.norm(p - self.last))
        self.last = p


def _ancestor(m, b, root) -> bool:
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def episode(args: tuple) -> dict:
    kind, seed, method, cfg, record_root, sha = args
    room = sample_room(
        np.random.default_rng([seed, 0]),
        shifted=kind == "shifted",
        with_cabinet=kind != "no_cabinet",
    )
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*room.start)
    monitor = Monitor(env)
    rec = None
    if record_root:
        from experiments.mobile_recording import MobileRecorder

        rec = MobileRecorder(
            env,
            pathlib.Path(record_root) / f"explore_{kind}_{seed}_{method}",
            {"experiment": "g1_explore_room", "kind": kind, "seed": seed, "method": method,
             "git_sha": sha},
        )  # fmt: skip

    def on_tick():
        monitor.tick()
        if rec:
            rec.sample()

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    events = []

    def log(kind_, text, **data):
        events.append({"time_s": env.time, "kind": kind_, "text": text})
        if rec:
            rec.event(kind_, text)

    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([seed, 2])), on_event=log)

    def on_event(k, d):
        if k == "action":
            log("decision", f"{d['tool']} {json.dumps(d['args'])}")
        elif k == "result":
            log("result", f"{d['tool']}: {d['status']} {d['message'][:60]}")
        elif k == "finish":
            log("finish", f"{'success' if d['success'] else 'stopped'}: {d['reason']}")

    t0 = time.perf_counter()
    planner = Budgeted(RoomSearchPlanner(method), base, cfg["time_budget_s"])
    ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(Inspect("cabinet"), on_event)
    wall = time.perf_counter() - t0
    out = {
        "kind": kind, "seed": seed, "method": method, "claimed": ep.success_claimed,
        "reason": ep.reason, "tool_calls": [s.call.name for s in ep.steps],
        "sim_s": env.time, "wall_s": wall, "fell": base.stop_reason is not None,
        "contact_steps": monitor.contact_steps, "path_m": monitor.path_m,
        "candidates": len(tb.memory.candidates), "frames": tb.memory.frames,
        "explored_m2": float(tb.memory.grid.free().sum()) * 0.05**2,
        "room": {"width": room.width, "depth": room.depth, "furniture": len(room.furniture),
                 "mechanism": room.cabinet.mechanism},
    }  # fmt: skip
    out.update(score(env, room, tb, ep.success_claimed))
    if rec:
        rec.save(out)
        out["recording"] = str(rec.root)
    return out


def score(env, room, tb, claimed: bool) -> dict:
    x, y, yaw = env.true_pose()
    if room.cabinet.mechanism == "none":
        return {
            "success": False,
            "false_claim": claimed,
            "correct_refusal": not claimed and not env.data.qpos[2] < 0.5,
        }
    h = env.data.site("cabinet_handle_site").xpos[:2]
    n = np.array([cos(room.cabinet.yaw), sin(room.cabinet.yaw)])
    d = float(np.hypot(h[0] - x, h[1] - y))
    facing = abs(wrap(atan2(h[1] - y, h[0] - x) - yaw))
    front = float((np.array([x, y]) - np.array(room.cabinet.xy)) @ n) > 0
    at_pose = 0.5 <= d <= 1.5 and facing < np.radians(40) and front
    # Identity, drift-free: the reported handle relative to the robot vs the truth.
    found = [c for c in tb.memory.candidates if c.handle is not None]
    ident_err = None
    if found:
        c = max(found, key=lambda c: c.handle_views)
        ox, oy, oyaw = env.odometry()
        rel = c.handle[:2] - [ox, oy]
        est = np.array([cos(-oyaw) * rel[0] - sin(-oyaw) * rel[1],
                        sin(-oyaw) * rel[0] + cos(-oyaw) * rel[1]])  # fmt: skip
        rt = h - [x, y]
        tru = np.array(
            [cos(-yaw) * rt[0] - sin(-yaw) * rt[1], sin(-yaw) * rt[0] + cos(-yaw) * rt[1]]
        )
        ident_err = float(np.linalg.norm(est - tru))
    identity = ident_err is not None and ident_err < 0.3
    ok = at_pose and identity
    return {
        "success": claimed and ok,
        "false_claim": claimed and not ok,
        "correct_refusal": None,
        "true_handle_distance_m": d,
        "facing_error_deg": float(np.degrees(facing)),
        "on_front_side": front,
        "identity_error_m": ident_err,
    }


def summarise(rows: list[dict], cfg: dict) -> dict:
    out = {}
    for kind in ("nominal", "shifted", "no_cabinet"):
        for method in cfg["methods"]:
            es = [r for r in rows if r["kind"] == kind and r["method"] == method]
            if not es:
                continue
            k = sum(r["success"] for r in es)
            out[f"{kind}/{method}"] = {
                "n": len(es),
                "success": k,
                "success_ci95": wilson_interval(k, len(es)),
                "false_claims": sum(r["false_claim"] for r in es),
                "correct_refusals": sum(bool(r.get("correct_refusal")) for r in es),
                "falls": sum(r["fell"] for r in es),
                "episodes_with_contact": sum(r["contact_steps"] > 0 for r in es),
                "median_sim_s": float(np.median([r["sim_s"] for r in es])),
                "median_path_m": float(np.median([r["path_m"] for r in es])),
                "median_success_sim_s": float(np.median([r["sim_s"] for r in es if r["success"]]))
                if k
                else None,
            }
    g = cfg["gates"]
    nom = out.get("nominal/gain", {})
    sh = out.get("shifted/gain", {})
    nc = out.get("no_cabinet/gain", {})
    gates = {
        "nominal_success": nom.get("success", 0) / max(nom.get("n", 1), 1)
        >= g["nominal_success_min"],
        "shifted_success": sh.get("success", 0) / max(sh.get("n", 1), 1)
        >= g["shifted_success_min"],
        "no_cabinet_refusal": nc.get("correct_refusals", 0) / max(nc.get("n", 1), 1)
        >= g["no_cabinet_refusal_min"],
        "false_claims": sum(r["false_claim"] for r in rows) <= g["max_false_claims"],
        "falls": sum(r["fell"] for r in rows) <= g["max_falls"],
    }
    return {"by_condition": out, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT))
    rec_root = os.environ.get("DREAM_RECORDINGS_ROOT", "/tmp/DREAM/recordings")
    record = {tuple(r) for r in cfg.get("record", [])}
    jobs = []
    for kind, seeds in cfg["seeds"].items():
        for seed in range(*seeds):
            for method in cfg["methods"] if kind != "no_cabinet" else cfg["methods"][:1]:
                rr = rec_root if (kind, seed, method) in record else None
                jobs.append((kind, seed, method, cfg, rr, sha))
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for r in pool.map(episode, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(
                f"[{len(rows)}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {r['kind']} "
                f"{r['seed']} {r['method']}: success={r['success']} claim={r['claimed']} "
                f"{r['reason'][:60]}",
                flush=True,
            )
    summary = summarise(rows, cfg)
    meta = {
        "config": cfg, "git_sha": sha, "dream_dirty_at_start": dirty,
        "mujoco": mujoco.__version__, "machine": platform.platform(),
        "wall_s": time.perf_counter() - t0,
        "odometry": "oracle-assisted (true increments + noise), dream/sim/g1_mobile.py",
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
