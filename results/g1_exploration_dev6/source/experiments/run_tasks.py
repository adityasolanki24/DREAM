"""Phase F evaluation: task agent vs baselines on held-out tabletop scenes.

    python experiments/run_tasks.py experiments/configs/tasks.json [--workers 4]

Methods share the scene, goal, controller, skills and sensing for each seed (paired design):
  agent           TaskAgent + RuleBasedPlanner (re-observes, verifies, retries, clears obstacles)
  agent_no_retry  the same, with one attempt per action (ablation: the value of recovery)
  fixed_sequence  open-loop script: every waypoint is planned once from the first observation
                  and run through the same controller, with no re-observation, verification or
                  retries
Conditions:
  nominal     3 cubes
  disturbed   an external lateral push (evaluation-side, physical force) knocks the held object
              out of the grasp the first time it is lifted above 10 cm
  cluttered   4 cubes (an arrangement never used during development)
Success is read from ground truth (dream.eval.manipulation), independently of agent claims.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dream.skills.manipulation as skills  # noqa: E402
from dream.eval.adequacy import mcnemar_p, wilson_interval  # noqa: E402
from dream.eval.manipulation import object_in_tray, object_on_top, safety_summary  # noqa: E402
from dream.runtime.agent import RuleBasedPlanner, TaskAgent  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.tools.goals import InRegion, OnTop  # noqa: E402
from dream.tools.toolbox import Toolbox  # noqa: E402


class KnockOff:
    """Knock ``name`` out of the hand the first time it is lifted above ``height``.

    A downward tug larger than the grip's friction capacity (~6 N) is applied only while the
    object is still within ``release_dist`` of the TCP, and stops as soon as it separates. The
    object slides out between the fingers and drops roughly where it is. Lateral pushes were
    tried first (25 N for 0.15 s, then 12 N until separation). They threw 50 g cubes 0.7-37 m off
    the table, an unrealistic and unrecoverable disturbance, and were replaced before any reported
    run.
    """

    def __init__(
        self,
        env: TabletopEnv,
        name: str,
        rng: np.random.Generator,
        force: float = 12.0,
        height: float = 0.10,
        release_dist: float = 0.03,
        max_duration: float = 0.3,
    ):
        self.body = env.model.body(name).id
        self.tcp = env.model.site("tcp").id
        self.force, self.height = force, height
        self.release_dist, self.max_duration = release_dist, max_duration
        self.start: float | None = None
        self.fired = False

    def __call__(self, model, data) -> None:
        if self.fired:
            return
        if self.start is None and data.xpos[self.body][2] > self.height:
            self.start = data.time
        if self.start is None:
            return
        in_hand = (
            np.linalg.norm(data.xpos[self.body] - data.site_xpos[self.tcp]) < self.release_dist
        )
        if in_hand and data.time - self.start < self.max_duration:
            data.xfrc_applied[self.body, 2] = -self.force
        else:
            data.xfrc_applied[self.body, :] = 0.0
            self.fired = True


def fixed_sequence(env: TabletopEnv, goal) -> dict:
    """Open-loop scripted pick-and-place, planned once from the first observation."""
    robot = env.embodiment()
    objs = {o.name: o for o in env.observe_objects()}
    obj = objs[goal.object]
    if isinstance(goal, InRegion):
        centre, _ = env.fixtures()[goal.region]
        target = centre + [0, 0, obj.size / 2]
    else:
        sup = objs[goal.support]
        target = sup.position + [0, 0, (sup.size + obj.size) / 2]
    yaw = skills.grasp_yaw(obj.yaw, skills.tcp_yaw(robot.state().tcp_quat))
    grasp_z = max(obj.position[2], 0.012)
    plan = [
        ("grip", skills.OPEN_WIDTH),
        ("move", [*obj.position[:2], skills.TRAVEL_HEIGHT]),
        ("move", [*obj.position[:2], grasp_z]),
        ("grip", 0.0),
        ("move", [*obj.position[:2], skills.TRAVEL_HEIGHT]),
        ("move", [*target[:2], skills.TRAVEL_HEIGHT]),
        ("move", [*target[:2], target[2] + 0.01]),
        ("grip", skills.OPEN_WIDTH),
        ("move", [*target[:2], skills.TRAVEL_HEIGHT]),
    ]
    failed = 0
    for kind, arg in plan:
        r = (
            skills.set_gripper(robot, arg)
            if kind == "grip"
            else skills.move_to(robot, np.array(arg, float), yaw)
        )
        failed += not r.ok
        if robot.state().protective_stop:
            break
    return {"tool_calls": len(plan), "failed_actions": failed, "claimed": None}


def run_agent(env: TabletopEnv, goal, max_attempts: int) -> dict:
    agent = TaskAgent(Toolbox(env.embodiment(), env), RuleBasedPlanner(max_attempts=max_attempts))
    ep = agent.run(goal)
    grasps = [s for s in ep.steps if s.call.name == "grasp"]
    return {
        "tool_calls": ep.tool_calls,
        "failed_actions": ep.failed_actions,
        "claimed": ep.success_claimed,
        "reason": ep.reason,
        "grasp_attempts": len(grasps),
        "grasp_successes": sum(s.result.ok for s in grasps),
        "trace": [(s.call.name, s.call.args.get("object"), s.result.status) for s in ep.steps],
    }


def episode(args: tuple) -> dict:
    method, task, condition, seed = args
    rng = np.random.default_rng([seed, 7])
    colors = (
        ("red", "blue", "green", "yellow") if condition == "cluttered" else ("red", "blue", "green")
    )
    spec = sample_tabletop(np.random.default_rng(seed), colors=colors)
    names = [o.name for o in spec.objects]
    picked = rng.choice(names, size=2, replace=False)
    goal = (
        InRegion(str(picked[0]), "tray")
        if task == "tray"
        else OnTop(str(picked[0]), str(picked[1]))
    )
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]))
    knock = KnockOff(env, goal.object, rng) if condition == "disturbed" else None
    env.step_hook = knock
    t0 = time.perf_counter()
    if method == "fixed_sequence":
        out = fixed_sequence(env, goal)
    else:
        out = run_agent(env, goal, max_attempts=1 if method == "agent_no_retry" else 3)
    env.step_hook = None
    env.step(500)  # settle before judging
    if isinstance(goal, InRegion):
        success = object_in_tray(env, goal.object)
        err = float(
            np.linalg.norm(env.privileged_object_pos(goal.object)[:2] - np.array(spec.tray.xy))
        )
    else:
        success = object_on_top(env, goal.object, goal.support)
        err = float(
            np.linalg.norm(
                env.privileged_object_pos(goal.object)[:2]
                - env.privileged_object_pos(goal.support)[:2]
            )
        )
    return {
        "method": method,
        "task": task,
        "condition": condition,
        "seed": seed,
        "goal": repr(goal),
        "success": success,
        "false_claim": bool(out.get("claimed") and not success),
        "disturbance_fired": bool(knock and knock.fired),
        "xy_error_m": err,
        "sim_time_s": env.time,
        "wall_time_s": time.perf_counter() - t0,
        **out,
        **safety_summary(env),
    }


def summarise(eps: list[dict]) -> dict:
    cells: dict = {}
    for e in eps:
        cells.setdefault((e["task"], e["condition"]), {}).setdefault(e["method"], []).append(e)
    out = {}
    for (task, cond), by_method in sorted(cells.items()):
        key = f"{task}/{cond}"
        out[key] = {}
        for m, lst in by_method.items():
            k = sum(e["success"] for e in lst)
            ok = [e for e in lst if e["success"]]
            att = sum(e.get("grasp_attempts", 0) for e in lst)
            fired = [e for e in lst if e["disturbance_fired"]]
            out[key][m] = {
                "n": len(lst),
                "success": k,
                "success_rate": k / len(lst),
                "success_ci95": wilson_interval(k, len(lst)),
                "false_claims": sum(e["false_claim"] for e in lst),
                "grasp_success_rate": (sum(e.get("grasp_successes", 0) for e in lst) / att)
                if att
                else None,
                "median_xy_error_mm": 1000 * float(np.median([e["xy_error_m"] for e in ok]))
                if ok
                else None,
                "mean_tool_calls": float(np.mean([e["tool_calls"] for e in lst])),
                "mean_failed_actions": float(np.mean([e["failed_actions"] for e in lst])),
                "recovered_after_disturbance": (sum(e["success"] for e in fired), len(fired))
                if fired
                else None,
                "mean_sim_time_s": float(np.mean([e["sim_time_s"] for e in lst])),
                "mean_wall_time_s": float(np.mean([e["wall_time_s"] for e in lst])),
                "protective_stops": sum(e["protective_stop"] for e in lst),
                "link_collisions": sum(e["link_collision"] for e in lst),
            }
        if "agent" in by_method:
            ref = {e["seed"]: e["success"] for e in by_method["agent"]}
            for m, lst in by_method.items():
                if m == "agent":
                    continue
                b = sum(ref[e["seed"]] and not e["success"] for e in lst)
                c = sum(e["success"] and not ref[e["seed"]] for e in lst)
                out[key][m]["vs_agent"] = {
                    "agent_only": b,
                    "this_only": c,
                    "mcnemar_p": mcnemar_p(b, c),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    seeds = range(cfg["seed0"], cfg["seed0"] + cfg["n"])
    jobs = [
        (m, t, c, s)
        for t in cfg["tasks"]
        for c in cfg["conditions"]
        for m in cfg["methods"]
        for s in seeds
    ]
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        eps = []
        for i, e in enumerate(pool.map(episode, jobs), 1):
            eps.append(e)
            f.write(json.dumps(e) + "\n")
            if i % 40 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s]", flush=True)
    summary = summarise(eps)
    meta = {
        "config": cfg,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "mujoco": mujoco.__version__,
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, "cells": summary}, indent=2))
    for key, by_method in summary.items():
        for m, r in by_method.items():
            shown = {
                k: r[k]
                for k in ("success", "n", "false_claims", "protective_stops", "link_collisions")
            }
            shown.update(
                ci=np.round(r["success_ci95"], 2).tolist(),
                calls=round(r["mean_tool_calls"], 1),
                failed=round(r["mean_failed_actions"], 2),
                recovered=r["recovered_after_disturbance"],
                vs_agent=r.get("vs_agent"),
            )
            print(f"{key:18s} {m:15s} {shown}")


if __name__ == "__main__":
    main()
