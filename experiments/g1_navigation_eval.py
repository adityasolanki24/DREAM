"""Natural navigation (Test C): the walking G1's old path follower vs Regulated Pure Pursuit.

    python experiments/g1_navigation_eval.py experiments/configs/g1_navigation_dev.json

Same episodes as Milestone B (TaskAgent + RoomSearchPlanner "gain" over MobileToolbox, goal
Inspect("cabinet"), NVIDIA's WBC walking policy in MuJoCo); the only difference between the
paired runs is MobileToolbox(navigator=...). Same rooms, seeds, budgets, perception and
planner (invariant 4). ODOMETRY IS ORACLE-ASSISTED, as in Milestone B.

Motion metrics come from the evaluator's own 50 Hz record (true pelvis state and the command
actually sent to the policy), split into navigation (go_to/approach) and perception turns
(look_around/inspect):
- path length, total rotation, turn-in-place time (|v| < 0.05 m/s and |w| > 0.1 rad/s),
- stop/start cycles while navigating (forward command falling below 0.02 m/s and resuming),
- angular jerk (RMS second derivative of the commanded yaw rate), command tracking error,
- repeated approaches to the same (candidate, side), contacts, falls, success, time.
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
from math import cos, sin

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import wilson_interval  # noqa: E402
from dream.navigation.follow import wrap  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.search import RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import Inspect  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted, Monitor, score  # noqa: E402

NAV_TOOLS = ("go_to", "approach")
DT_TICK = 0.02  # s, one policy tick


def motion_metrics(rows: list[dict]) -> dict:
    """Metrics over the 50 Hz evaluator record; pure, so it is tested on synthetic rows."""
    out = {}
    for part, keep in (("nav", lambda r: r["tool"] in NAV_TOOLS), ("all", lambda r: True)):
        rs = [r for r in rows if keep(r)]
        if len(rs) < 3:
            out[part] = None
            continue
        v = np.array([r["cmd"][0] for r in rs])
        w = np.array([r["cmd"][2] for r in rs])
        yaw = np.array([r["yaw"] for r in rs])
        xy = np.array([r["xy"] for r in rs])
        vfwd = np.array([r["v_body"] for r in rs])
        moving = np.abs(v) >= 0.05
        stops = int(np.sum(moving[:-1] & (np.abs(v[1:]) < 0.02)))
        jerk = np.diff(w, 2) / DT_TICK**2
        out[part] = {
            "time_s": len(rs) * DT_TICK,
            "path_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "rotation_rad": float(np.abs([wrap(a) for a in np.diff(yaw)]).sum()),
            "turn_in_place_s": float(np.sum(~moving & (np.abs(w) > 0.1)) * DT_TICK),
            "stopped_s": float(np.sum(~moving & (np.abs(w) <= 0.1)) * DT_TICK),
            "stop_start_cycles": stops,
            "angular_jerk_rms": float(np.sqrt(np.mean(jerk**2))),
            "forward_tracking_error_m_s": float(np.median(np.abs(vfwd[moving] - v[moving])))
            if moving.any()
            else None,
        }
    return out


def episode(args: tuple) -> dict:
    kind, seed, navigator, cfg, sha = args
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
    rows, current = [], {"tool": None}

    def on_tick():
        monitor.tick()
        d = env.data
        yaw = env.true_pose()[2]
        vb = cos(yaw) * d.qvel[0] + sin(yaw) * d.qvel[1]
        rows.append({"tool": current["tool"], "cmd": base.controller.command.tolist(),
                     "yaw": yaw, "xy": d.qpos[:2].tolist(), "v_body": float(vb)})  # fmt: skip

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([seed, 2])), navigator=navigator)
    calls = []

    def on_event(kind_, data):
        if kind_ == "action":
            current["tool"] = data["tool"]
            calls.append({"tool": data["tool"], "args": data["args"], "t": env.time})

    t0 = time.perf_counter()
    planner = Budgeted(RoomSearchPlanner("gain"), base, cfg["time_budget_s"])
    ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(Inspect("cabinet"), on_event)
    approaches = [(int(c["args"]["candidate"]), int(c["args"]["side"])) for c in calls
                  if c["tool"] == "approach"]  # fmt: skip
    out = {
        "kind": kind, "seed": seed, "navigator": navigator, "claimed": ep.success_claimed,
        "reason": ep.reason, "tool_calls": [c["tool"] for c in calls], "sim_s": env.time,
        "wall_s": time.perf_counter() - t0, "fell": base.stop_reason is not None,
        "contact_steps": monitor.contact_steps, "path_m": monitor.path_m,
        "approaches": len(approaches),
        "repeated_approaches": len(approaches) - len(set(approaches)),
        "motion": motion_metrics(rows),
    }  # fmt: skip
    out.update(score(env, room, tb, ep.success_claimed))
    return out


def summarise(rows: list[dict], cfg: dict) -> dict:
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        for nav in cfg["navigators"]:
            es = [r for r in rows if r["kind"] == kind and r["navigator"] == nav]
            if not es:
                continue
            k = sum(r["success"] for r in es)

            def med(key, part="nav", es=es):
                vals = [r["motion"][part][key] for r in es if r["motion"][part] is not None
                        and r["motion"][part][key] is not None]  # fmt: skip
                return float(np.median(vals)) if vals else None

            out[f"{kind}/{nav}"] = {
                "n": len(es), "success": k, "success_ci95": wilson_interval(k, len(es)),
                "false_claims": sum(r["false_claim"] for r in es),
                "falls": sum(r["fell"] for r in es),
                "episodes_with_contact": sum(r["contact_steps"] > 0 for r in es),
                "median_sim_s": float(np.median([r["sim_s"] for r in es])),
                "median_wall_s": float(np.median([r["wall_s"] for r in es])),
                "median_path_m": float(np.median([r["path_m"] for r in es])),
                "repeated_approaches": sum(r["repeated_approaches"] for r in es),
                "nav_median": {key: med(key) for key in (
                    "time_s", "path_m", "rotation_rad", "turn_in_place_s", "stopped_s",
                    "stop_start_cycles", "angular_jerk_rms", "forward_tracking_error_m_s")},
                "all_median_turn_in_place_s": med("turn_in_place_s", "all"),
            }  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT))
    jobs = [(kind, seed, nav, cfg, sha) for kind, seeds in cfg["seeds"].items()
            for seed in range(*seeds) for nav in cfg["navigators"]]  # fmt: skip
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for r in pool.map(episode, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            nav = r["motion"]["nav"] or {}
            print(f"[{len(rows)}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {r['kind']} "
                  f"{r['seed']} {r['navigator']}: success={r['success']} sim={r['sim_s']:.0f}s "
                  f"tip={nav.get('turn_in_place_s', 0):.1f}s stops={nav.get('stop_start_cycles')}"
                  f" contact={r['contact_steps']}", flush=True)  # fmt: skip
    summary = summarise(rows, cfg)
    meta = {"config": cfg, "git_sha": sha, "dream_dirty_at_start": dirty,
            "machine": platform.platform(), "wall_s": time.perf_counter() - t0,
            "odometry": "oracle-assisted (true increments + noise)"}  # fmt: skip
    (out / "summary.json").write_text(json.dumps({"meta": meta, "by_condition": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
