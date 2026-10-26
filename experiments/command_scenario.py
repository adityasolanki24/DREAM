"""End-to-end command scenario for a walking G1, independent of the simulator.

Every command arrives as an authenticated HTTP request to the phone API (a stand-in for the
phone; the request travels over real HTTP on localhost) and is run by the task scheduler:

1. "inspect cabinet" (priority 0): search the unfamiliar room/kitchen for a cabinet with a
   handle and stand in front of it;
2. while that task walks (its INTERRUPT_AT_ACTION-th action + INTERRUPT_AFTER_S), the phone
   sends "goto entrance" (priority 5): the walk yields, the inspection is suspended with its
   checkpoint, the robot goes to the entrance, then the inspection resumes and finishes;
3. confirmed handles are written to the persistent home memory;
4. the phone sends "goto entrance" again, then "inspect cabinet": this time the memory-guided
   planner checks the remembered handle first instead of searching.

Physical execution (simulator), model reasoning (classical planners here; no language
model), memory (SQLite), phone control (HTTP), recovery (suspend/resume) and the
independent verification (``scorer``, simulator truth, evaluator only) are logged
separately. The commands are structured strings; natural-language parsing is not used.
Odometry is ORACLE-ASSISTED in both simulators.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.request

import numpy as np

from dream.runtime.agent import TaskAgent
from dream.runtime.memory_search import (
    MemoryGuidedPlanner,
    remember_inspections,
    remembered_label,
)
from dream.runtime.phone_api import PhoneAPI
from dream.runtime.scheduler import TaskScheduler
from dream.runtime.search import PlacePlanner, RoomSearchPlanner
from dream.tools.goals import GoToPlace, Inspect, parse_goal

INTERRUPT_AT_ACTION = 2
INTERRUPT_AFTER_S = 2.0


def run_commands(env, base, tb, home, map_id: str, cfg: dict, scorer, log, store) -> dict:
    """Run the scenario. ``scorer(task_text, claimed)`` returns the independent verdict;
    ``log(kind, text, **data)`` receives every event (for captions and the timeline)."""
    from experiments.g1_explore_room import Budgeted

    home.add_place("entrance", "place", map_id, (0.0, 0.0))
    tb.places = {"entrance": np.zeros(2)}
    state = {"trigger": None, "phone": [], "use_memory": False, "verdicts": []}

    def note(kind, data):
        log(f"task_{kind}", " ".join(f"{k}={v}" for k, v in data.items())[:70])
        api.notify(kind, data)

    def run_task(task, history, preempt, cancel):
        tb.preempt, tb.cancel = preempt, cancel
        if isinstance(task.goal, GoToPlace):
            planner = PlacePlanner()
        else:
            planner = Budgeted(RoomSearchPlanner("gain"), base, cfg["time_budget_s"])
            if state["use_memory"]:
                planner = MemoryGuidedPlanner(planner, home, map_id, env.time,
                                              last_frame=lambda: tb.memory.last_frame)  # fmt: skip
                log(
                    "memory",
                    "remembered handles: "
                    + ", ".join(
                        f"{b.object_id[-8:]} ({b.state})"
                        for b in home.where_is(remembered_label(task.goal), map_id, now=env.time)
                    ),
                )
        calls = {"n": 0}

        def on_event(kind, data):
            text = {k: str(v)[:60] for k, v in data.items()}
            log(f"agent_{kind}", f"[{task.text}] " + " ".join(f"{k}={v}" for k, v in text.items()))
            if kind == "action" and task.text == "inspect cabinet" and not state["use_memory"]:
                calls["n"] += 1
                if calls["n"] == INTERRUPT_AT_ACTION and state["trigger"] is None:
                    state["trigger"] = env.time + INTERRUPT_AFTER_S

        ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(
            task.goal, on_event, history=history, preempt=preempt
        )
        if not ep.reason.startswith("interrupted"):
            v = scorer(task.text, ep.success_claimed)
            state["verdicts"].append({"task": task.text, "claimed": ep.success_claimed,
                                      "reason": ep.reason, "t": env.time, **v})  # fmt: skip
            log("verify", f"[{task.text}] claimed={ep.success_claimed} "
                f"verified={v.get('success')}")  # fmt: skip
        return ep

    sched = TaskScheduler(str(store / "tasks.sqlite"), run_task, parse_goal, notify=note)
    token = secrets.token_urlsafe(32)
    api = PhoneAPI(sched, token)
    server = api.serve(port=0)
    port = server.server_address[1]

    def phone(text, priority):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tasks",
            data=json.dumps({"text": text, "priority": priority}).encode(),
            headers={"Authorization": f"Bearer {token}"}, method="POST",
        )  # fmt: skip
        with urllib.request.urlopen(req, timeout=10) as r:
            row = {"text": text, "priority": priority, "status": r.status, "t": env.time,
                   **json.loads(r.read())}  # fmt: skip
        state["phone"].append(row)
        log("phone", f"'{text}' priority {priority} -> HTTP {row['status']}")
        return row

    tick = base.on_tick or (lambda: None)

    def on_tick():
        tick()
        if state["trigger"] is not None and env.time >= state["trigger"]:
            state["trigger"] = float("inf")
            phone("goto entrance", 5)  # mid-walk, over HTTP

    base.on_tick = on_tick
    t0 = time.perf_counter()
    try:
        phone("inspect cabinet", 0)
        sched.run_until_idle()
        stored = remember_inspections(tb, home, map_id, Inspect("cabinet"), now=env.time)
        log("memory", f"stored {len(stored)} confirmed handle(s) in home memory")
        state["use_memory"] = bool(stored)
        phone("goto entrance", 0)
        sched.run_until_idle()
        phone("inspect cabinet", 0)
        sched.run_until_idle()
    finally:
        server.shutdown()
    return {"tasks": sched.tasks(), "phone": state["phone"], "verdicts": state["verdicts"],
            "stored_handles": stored if "stored" in locals() else [],
            "wall_s": time.perf_counter() - t0}  # fmt: skip
