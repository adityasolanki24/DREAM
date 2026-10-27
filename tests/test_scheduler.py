"""Task scheduler contracts: safe preemption at action boundaries, checkpoint and resume with
fresh observation, cancellation acknowledgement, persistence across a restart."""

from __future__ import annotations

import threading

from dream.runtime.agent import Finish, TaskAgent, ToolCall
from dream.runtime.scheduler import TaskScheduler
from dream.tools.toolbox import ToolResult


class FakeTools:
    """Counts calls; 'work' is one physical sub-action that always completes."""

    def __init__(self, log):
        self.cancel, self.preempt, self.log = threading.Event(), threading.Event(), log
        self.on_work = None

    def call(self, name, args=None):
        self.log.append((name, dict(args or {})))
        if name == "work" and self.on_work:
            self.on_work(args)
        data = {"satisfied": False} if name == "verify_goal" else {"protective_stop": False}
        return ToolResult(name, "succeeded", "", data)


class CountPlanner:
    def next_action(self, goal, scene, verdict, history):
        done = sum(1 for s in history if s.call.name == "work")
        if done >= goal["n"]:
            return Finish(True, f"{goal['tag']} done")
        return ToolCall("work", {"tag": goal["tag"], "k": done})


def parse(text):
    tag, n = text.split(":")
    if not n.isdigit():
        raise ValueError("unsupported request")
    return {"tag": tag, "n": int(n)}


def make(tmp_path, log, notes):
    tools = FakeTools(log)

    def run_task(task, history, preempt, cancel):
        tools.preempt, tools.cancel = preempt, cancel
        return TaskAgent(tools, CountPlanner(), max_steps=20).run(
            task.goal, history=history, preempt=preempt
        )

    s = TaskScheduler(str(tmp_path / "tasks.sqlite"), run_task, parse,
                      notify=lambda k, d: notes.append((k, d.get("id"))))  # fmt: skip
    return s, tools


def test_higher_priority_request_preempts_at_a_boundary_and_the_old_task_resumes(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tidy = s.submit("tidy:4", priority=0)
    phone = {}

    def on_work(args):  # the phone request arrives while the second sub-action runs
        if args["tag"] == "tidy" and args["k"] == 1 and not phone:
            phone["id"] = s.submit("bottle:2", priority=5)

    tools.on_work = on_work
    s.run_until_idle()
    works = [a["tag"] + str(a["k"]) for n, a in log if n == "work"]
    # The interrupted sub-action completed; nothing ran twice; the old task resumed.
    assert works == ["tidy0", "tidy1", "bottle0", "bottle1", "tidy2", "tidy3"]
    states = {t["id"]: t for t in s.tasks()}
    assert states[tidy]["state"] == "succeeded" and states[tidy]["suspensions"] == 1
    assert states[phone["id"]]["state"] == "succeeded"
    # After resuming, the scene was observed before the next action (revalidation).
    i = next(i for i, (n, a) in enumerate(log) if n == "work" and a == {"tag": "tidy", "k": 2})
    assert log[i - 3][0] == "observe_scene"
    kinds = [k for k, _ in notes]
    assert kinds.index("suspended") < kinds.index("resumed")


def test_equal_or_lower_priority_does_not_preempt(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    s.submit("a:2", priority=3)
    tools.on_work = lambda args: s.submit("b:1", priority=3) if args["k"] == 0 and args[
        "tag"] == "a" else None  # fmt: skip
    s.run_until_idle()
    assert [a["tag"] for n, a in log if n == "work"] == ["a", "a", "b"]


def test_cancel_is_acknowledged_and_stops_at_the_next_boundary(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("long:5")
    queued = s.submit("later:1", priority=-1)
    assert s.cancel(queued) == "cancelled"
    acks = []
    tools.on_work = lambda args: acks.append(s.cancel(tid)) if args["k"] == 1 else None
    s.run_until_idle()
    assert acks == ["cancelling"]
    assert len([1 for n, _ in log if n == "work"]) == 2
    states = {t["id"]: t["state"] for t in s.tasks()}
    assert states == {tid: "cancelled", queued: "cancelled"}


def test_unsupported_requests_are_rejected_before_queueing(tmp_path):
    s, _ = make(tmp_path, [], [])
    try:
        s.submit("fly:away")
    except ValueError:
        pass
    assert s.tasks() == []


def test_checkpoint_survives_a_restart_and_ask_policy_holds_resumption(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("tidy:3", resume="ask")

    def interrupt(args):
        if args["k"] == 1:
            s.preempt.set()

    tools.on_work = interrupt
    s.step()
    assert s.get(tid).state == "suspended" and len(s.get(tid).steps) == 2
    s2, tools2 = make(tmp_path, [], [])
    assert s2.next_task() is None  # resume='ask': held until approved
    s2.resume(tid)
    s2.run_until_idle()
    t = s2.get(tid)
    assert t.state == "succeeded" and [x.call.args["k"] for x in t.steps] == [0, 1, 2]


def test_a_task_left_running_by_a_crash_restarts_as_suspended(tmp_path):
    s, _ = make(tmp_path, [], [])
    tid = s.submit("tidy:2")
    s.db.execute("UPDATE tasks SET state='running' WHERE id=?", (tid,))
    s.db.commit()
    s2, _ = make(tmp_path, [], [])
    assert s2.get(tid).state == "suspended"


def test_mobile_toolbox_verifies_a_goto_place_goal():
    """Regression (Test D, rooms 4100-4102): the tool-argument validator rejected GoToPlace,
    so every urgent 'goto entrance' task failed before moving."""
    import numpy as np

    from dream.tools.goals import GoToPlace, parse_goal
    from dream.tools.mobile import MobileToolbox

    class Base:
        time, stop_reason = 0.0, None

        def odometry(self):
            return np.array([0.1, 0.0, 0.0])

    tb = MobileToolbox(Base(), memory=None)
    tb.places = {"entrance": np.zeros(2)}
    r = tb.call("verify_goal", {"goal": parse_goal("goto entrance")})
    assert r.ok and r.data["satisfied"], r.message
    r = tb.call("verify_goal", {"goal": GoToPlace("kitchen")})
    assert r.ok and r.data.get("unknown_place")


def test_interruption_scoring_measures_from_the_start_pose():
    """Regression: Test D scored the entrance error from the world origin, so arrivals
    0.12-0.15 m from the true start were reported as 1.2-1.8 m."""
    from experiments.g1_interruption_demo import entrance_error

    assert entrance_error((1.78, -0.34, 0.2), (1.72, -0.44, 0.0)) < 0.13
    assert entrance_error((0.0, 0.0, 0.0), (1.72, -0.44, 0.0)) > 1.7


def test_pause_holds_task_at_safe_boundary_until_resume(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("tidy:3")
    tools.on_work = lambda args: s.pause(tid) if args["k"] == 0 else None
    s.step()
    assert s.get(tid).state == "suspended"
    assert s.next_task() is None
    assert len(s.get(tid).steps) == 1
    s.resume(tid)
    s.run_until_idle()
    assert s.get(tid).state == "succeeded"


def test_preemption_during_planning_never_dispatches_a_new_tool(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("tidy:1", resume="ask")

    class SlowPlanner:
        def next_action(self, goal, scene, verdict, history):
            s.preempt.set()
            return ToolCall("work", {})

    def run(task, history, preempt, cancel):
        tools.preempt, tools.cancel = preempt, cancel
        return TaskAgent(tools, SlowPlanner()).run(task.goal, history=history, preempt=preempt)

    s.run_task = run
    s.step()
    assert s.get(tid).state == "suspended"
    assert all(name != "work" for name, _ in log)


def test_cancel_wins_when_pause_and_cancel_arrive_together(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("tidy:3")

    def stop_both(args):
        s.pause(tid)
        s.cancel(tid)

    tools.on_work = stop_both
    s.step()
    assert s.get(tid).state == "cancelled"
    assert s.next_task() is None
    assert len([name for name, _ in log if name == "work"]) == 1


def test_cancel_arriving_after_suspended_episode_wins(tmp_path):
    from dream.runtime.agent import Episode

    s, _ = make(tmp_path, [], [])
    tid = s.submit("tidy:1")

    def run(task, history, preempt, cancel):
        preempt.set()
        cancel.set()
        return Episode(task.goal, False, "suspended", history, suspended=True)

    s.run_task = run
    s.step()
    assert s.get(tid).state == "cancelled"
    assert s.next_task() is None


def test_cancellation_during_planning_never_dispatches_a_tool_or_claims_success(tmp_path):
    log, notes = [], []
    s, tools = make(tmp_path, log, notes)
    tid = s.submit("tidy:1")

    class SlowPlanner:
        def next_action(self, goal, scene, verdict, history):
            s.cancel(tid)  # the stop request arrives while the planner is still thinking
            return ToolCall("work", {})

    def run(task, history, preempt, cancel):
        tools.preempt, tools.cancel = preempt, cancel
        return TaskAgent(tools, SlowPlanner()).run(task.goal, history=history, preempt=preempt)

    s.run_task = run
    s.step()
    task = s.get(tid)
    assert task.state == "cancelled" and task.result == "cancelled by user"
    assert all(name != "work" for name, _ in log)
