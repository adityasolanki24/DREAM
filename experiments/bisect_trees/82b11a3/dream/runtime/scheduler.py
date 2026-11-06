"""Priority task queue with safe preemption, cancellation and resumable checkpoints.

One robot, one running task. A newly submitted task with higher priority does not seize the
actuators: it sets the running task's preemption flag; the running tool brings the robot to a
standing stop (or finishes its safety-critical sub-action), TaskAgent suspends at the next
action boundary, and its steps are saved as the checkpoint. The higher-priority task runs;
then the suspended one resumes from its checkpoint, observing the scene afresh before its
next action (TaskAgent always does), so a changed world is revalidated, not assumed.

Tasks and checkpoints persist in SQLite, so a restart neither forgets queued work nor
resumes a stale plan blindly. Behaviour-tree style semantics (preempt, resume, cancel,
retry) live here at the task boundary; TaskAgent and the tools are unchanged otherwise.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time as _time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from dream.runtime.agent import Episode, Step, ToolCall
from dream.tools.toolbox import ToolResult

STATES = ("queued", "running", "suspended", "succeeded", "failed", "cancelled")


@dataclass
class Task:
    id: str
    text: str
    goal: object
    priority: int
    state: str = "queued"
    created: float = 0.0
    updated: float = 0.0
    steps: list[Step] = field(default_factory=list)
    result: str = ""
    suspensions: int = 0
    resume: str = "auto"  # "auto" | "ask": whether a suspended task restarts by itself


def _steps_json(steps: list[Step]) -> str:
    return json.dumps(
        [{"tool": s.call.name, "args": s.call.args, "status": s.result.status,
          "message": s.result.message, "data": s.result.data, "sim_time": s.result.sim_time,
          "wall_time": s.wall_time} for s in steps],
        default=str,
    )  # fmt: skip


def _steps_from(text: str) -> list[Step]:
    return [
        Step(ToolCall(r["tool"], r["args"]),
             ToolResult(r["tool"], r["status"], r["message"], r["data"], r["sim_time"]),
             r["wall_time"])
        for r in json.loads(text)
    ]  # fmt: skip


class TaskScheduler:
    """``run_task(task, history, preempt, on_event) -> Episode`` executes one task attempt;
    ``parse(text) -> goal`` turns a request into a structured goal (or raises ValueError)."""

    def __init__(self, path: str, run_task: Callable, parse: Callable, clock=_time.time,
                 notify: Callable[[str, dict], None] | None = None) -> None:  # fmt: skip
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, text TEXT, goal TEXT,"
            " priority INTEGER, state TEXT, created REAL, updated REAL, steps TEXT,"
            " result TEXT, suspensions INTEGER, resume TEXT)"
        )
        self.run_task, self.parse, self.clock = run_task, parse, clock
        self.notify = notify or (lambda kind, data: None)
        self.lock = threading.RLock()
        self.preempt = threading.Event()
        self.cancel_running = threading.Event()
        self.running: str | None = None
        # A task left 'running' by a crash cannot have a valid live plan: suspend it.
        self.db.execute("UPDATE tasks SET state='suspended' WHERE state='running'")
        self.db.commit()

    # -- persistence -------------------------------------------------------------------------
    def _save(self, t: Task) -> None:
        t.updated = self.clock()
        self.db.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (t.id, t.text, repr(t.goal), t.priority, t.state, t.created, t.updated,
             _steps_json(t.steps), t.result, t.suspensions, t.resume),
        )  # fmt: skip
        self.db.commit()

    def get(self, tid: str) -> Task | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if r is None:
            return None
        return Task(r[0], r[1], self.parse(r[1]), r[3], r[4], r[5], r[6], _steps_from(r[7]),
                    r[8], r[9], r[10])  # fmt: skip

    def tasks(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT id, text, priority, state, created, updated, result, suspensions, steps"
                " FROM tasks ORDER BY created"
            ).fetchall()
        return [{"id": r[0], "text": r[1], "priority": r[2], "state": r[3], "created": r[4],
                 "updated": r[5], "result": r[6], "suspensions": r[7],
                 "steps": len(json.loads(r[8]))} for r in rows]  # fmt: skip

    # -- requests (any thread) ------------------------------------------------------------------
    def submit(self, text: str, priority: int = 0, resume: str = "auto") -> str:
        goal = self.parse(text)  # rejects what the robot cannot represent, before queueing
        with self.lock:
            t = Task(uuid.uuid4().hex[:10], text, goal, int(priority), created=self.clock(),
                     resume=resume)  # fmt: skip
            self._save(t)
            running = self.get(self.running) if self.running else None
            if running is not None and t.priority > running.priority:
                self.preempt.set()  # honoured at the next safe boundary
            self.notify("accepted", {"id": t.id, "text": text, "preempting": self.preempt.is_set()})
            return t.id

    def cancel(self, tid: str) -> str:
        """Acknowledge immediately; a running task stops at its next safe boundary."""
        with self.lock:
            t = self.get(tid)
            if t is None or t.state in ("succeeded", "failed", "cancelled"):
                return "not_cancellable"
            if tid == self.running:
                self.cancel_running.set()
                return "cancelling"
            t.state, t.result = "cancelled", "cancelled before running"
            self._save(t)
            self.notify("cancelled", {"id": t.id, "text": t.text, "reason": t.result})
            return "cancelled"

    def set_priority(self, tid: str, priority: int) -> None:
        with self.lock:
            t = self.get(tid)
            if t is None:
                raise KeyError(tid)
            t.priority = int(priority)
            self._save(t)
            running = self.get(self.running) if self.running else None
            if running is not None and tid != self.running and t.priority > running.priority:
                if t.state in ("queued", "suspended"):
                    self.preempt.set()

    def resume(self, tid: str) -> None:
        """Approve resuming a task held with resume='ask'."""
        with self.lock:
            t = self.get(tid)
            if t is not None and t.state == "suspended":
                t.resume = "auto"
                self._save(t)

    def pause(self, tid: str) -> str:
        """Hold a queued task or request a safe-boundary stop of the running task."""
        with self.lock:
            t = self.get(tid)
            if t is None:
                raise KeyError(tid)
            if t.state not in ("queued", "running", "suspended"):
                return "not_pausable"
            t.resume = "ask"
            if tid == self.running:
                self._save(t)
                self.preempt.set()
                return "pausing"
            t.state = "suspended"
            self._save(t)
            return "paused"

    # -- execution (the robot's thread) ---------------------------------------------------------
    def next_task(self) -> Task | None:
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM tasks WHERE state='queued' OR (state='suspended' AND resume='auto')"
                " ORDER BY priority DESC, created ASC LIMIT 1"
            ).fetchone()
            return None if rows is None else self.get(rows[0])

    def step(self) -> Task | None:
        """Run the most urgent runnable task until it ends or is suspended."""
        with self.lock:
            t = self.next_task()
            if t is None:
                return None
            resumed = t.state == "suspended"
            t.state = "running"
            self._save(t)
            self.running = t.id
            self.preempt.clear()
            self.cancel_running.clear()
        self.notify("resumed" if resumed else "started", {"id": t.id, "text": t.text})
        try:
            ep: Episode = self.run_task(t, t.steps, self.preempt, self.cancel_running)
        except Exception as error:  # a crashed attempt is a failure, recorded, not a pass
            ep = Episode(t.goal, False, f"execution error: {error!r}", t.steps)
        with self.lock:
            t.steps = ep.steps
            # A concurrent pause may have changed the persisted resume policy.
            t.resume = self.get(t.id).resume
            if self.cancel_running.is_set():
                t.state = "cancelled"
            elif ep.suspended:
                t.state, t.suspensions = "suspended", t.suspensions + 1
            else:
                t.state = "succeeded" if ep.success_claimed else "failed"
            t.result = "cancelled by user" if t.state == "cancelled" else ep.reason
            self._save(t)
            self.running = None
        self.notify(t.state, {"id": t.id, "text": t.text, "reason": ep.reason})
        return t

    def run_until_idle(self, max_attempts: int = 50) -> None:
        for _ in range(max_attempts):
            if self.step() is None:
                return
