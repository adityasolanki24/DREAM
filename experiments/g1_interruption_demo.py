"""Test D: a phone request interrupts a running task, which then resumes after revalidation.

    python experiments/g1_interruption_demo.py experiments/configs/g1_interruption_demo.json

Real MuJoCo G1 under the WBC policy (rpp navigator). The scheduler runs "inspect cabinet"
(priority 0). When that task starts its second action, an authenticated HTTP request to the
phone API on localhost asks "goto entrance" with priority 5 (a stand-in for the phone; the
request travels over real HTTP). Expected, and scored independently:
1. the running walk stops safely (decelerates and stands, no fall), the task is suspended
   with its steps as the checkpoint;
2. the urgent task runs and the robot truly ends within 0.3 m of the entrance;
3. the inspection resumes from its checkpoint, observes afresh, and completes
   (Milestone B scoring), with no false claim.
"entrance" is the start pose, stored in home memory as a named place. ODOMETRY IS
ORACLE-ASSISTED.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.memory.home import HomeMemory  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.phone_api import PhoneAPI  # noqa: E402
from dream.runtime.scheduler import TaskScheduler  # noqa: E402
from dream.runtime.search import PlacePlanner, RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import GoToPlace, parse_goal  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted, Monitor, score  # noqa: E402


def entrance_error(true_pose, start) -> float:
    """True distance from the entrance, which is the robot's start pose (not the world origin:
    the first two Test D runs scored against the origin, a bug in this evaluator)."""
    return float(np.linalg.norm(np.asarray(true_pose)[:2] - np.asarray(start)[:2]))


def episode(seed: int, cfg: dict, store: pathlib.Path) -> dict:
    import secrets

    room = sample_room(np.random.default_rng([seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*room.start)
    monitor = Monitor(env)
    trigger = {}  # set when the inspection's chosen action starts; fires mid-motion

    def on_tick():
        monitor.tick()
        if "at" in trigger and "phone" not in truth and env.time >= trigger["at"]:
            truth["phone"] = phone_request("goto entrance", 5)  # during a walk, over HTTP

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([seed, 2])), navigator="rpp")
    home = HomeMemory(store / f"home_{seed}.sqlite")
    home.add_place("entrance", "place", f"odom-start-aligned:room{seed}", (0.0, 0.0))
    tb.places = {"entrance": np.zeros(2)}
    timeline, truth = [], {}

    def note(kind, data):
        p = env.true_pose()
        timeline.append({"t": env.time, "kind": kind, **data, "true_xy": p[:2].tolist(),
                         "speed": float(np.linalg.norm(env.data.qvel[:2]))})  # fmt: skip
        api.notify(kind, data)

    def run_task(task, history, preempt, cancel):
        tb.preempt, tb.cancel = preempt, cancel
        if isinstance(task.goal, GoToPlace):
            planner = PlacePlanner()
        else:
            planner = Budgeted(RoomSearchPlanner("gain"), base, cfg["time_budget_s"])
        calls = {"n": 0}

        def on_event(kind, data):
            note(f"agent_{kind}", {"task": task.text, **{k: str(v)[:80] for k, v in data.items()}})
            if kind == "action" and task.text == "inspect cabinet":
                calls["n"] += 1
                if calls["n"] == cfg["interrupt_at_action"] and "at" not in trigger:
                    trigger["at"] = env.time + cfg["interrupt_after_s"]

        ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(
            task.goal, on_event, history=history, preempt=preempt
        )
        if isinstance(task.goal, GoToPlace) and ep.success_claimed:
            truth["entrance_error_m"] = entrance_error(env.true_pose(), room.start)
        return ep

    sched = TaskScheduler(str(store / f"tasks_{seed}.sqlite"), run_task, parse_goal, notify=note)
    token = secrets.token_urlsafe(32)
    api = PhoneAPI(sched, token)
    server = api.serve(port=0)
    port = server.server_address[1]

    def phone_request(text, priority):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tasks",
            data=json.dumps({"text": text, "priority": priority}).encode(),
            headers={"Authorization": f"Bearer {token}"}, method="POST",
        )  # fmt: skip
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"status": r.status, "t": env.time, **json.loads(r.read())}

    try:
        inspect_id = sched.submit("inspect cabinet", 0)
        sched.run_until_idle()
    finally:
        server.shutdown()
    tasks = {t["text"]: t for t in sched.tasks()}
    inspect_score = score(env, room, tb, tasks["inspect cabinet"]["state"] == "succeeded")
    stop_rows = [r for r in timeline if r["kind"] == "agent_result" and "interrupted" in
                 r.get("status", "")]  # fmt: skip
    kinds = [r["kind"] for r in timeline]
    out = {
        "seed": seed, "tasks": sched.tasks(), "phone": truth.get("phone"),
        "interrupted_tool_stopped_safely": bool(stop_rows) and stop_rows[0]["speed"] < 0.15,
        "suspended": "suspended" in kinds,
        "urgent_succeeded": tasks.get("goto entrance", {}).get("state") == "succeeded",
        "entrance_error_m": truth.get("entrance_error_m"),
        "inspect_resumed": "resumed" in kinds, "inspect": inspect_score,
        "inspect_task_id": inspect_id, "fell": base.stop_reason is not None,
        "contact_steps": monitor.contact_steps, "sim_s": env.time,
    }  # fmt: skip
    out["pass"] = bool(
        out["interrupted_tool_stopped_safely"] and out["urgent_succeeded"]
        and (out["entrance_error_m"] or 1) <= 0.3 and out["inspect_resumed"]
        and inspect_score["success"] and not inspect_score["false_claim"] and not out["fell"]
    )  # fmt: skip
    home.close()
    return {**out, "timeline": timeline}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    from experiments.provenance import code_revision

    sha, _ = code_revision()
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    store = pathlib.Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "DREAM/interruption"
    store.mkdir(parents=True, exist_ok=True)
    for f in store.glob("*.sqlite"):
        f.unlink()
    rows = []
    for seed in range(*cfg["seeds"]):
        r = episode(seed, cfg, store)
        rows.append(r)
        print(
            f"room {seed}: pass={r['pass']} safe_stop={r['interrupted_tool_stopped_safely']}"
            f" urgent={r['urgent_succeeded']} err={r['entrance_error_m']}"
            f" resumed={r['inspect_resumed']} inspect={r['inspect']['success']}",
            flush=True,
        )
    with open(out / "episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    summary = {"git_sha": sha, "config": cfg, "passed": sum(r["pass"] for r in rows),
               "n": len(rows), "falls": sum(r["fell"] for r in rows),
               "false_claims": sum(r["inspect"]["false_claim"] for r in rows)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
