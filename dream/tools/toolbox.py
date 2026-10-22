"""Typed robot tools for a task-level agent (rule-based now, a language model later).

Each tool has a declared schema (``TOOL_SPECS``, in JSON-schema style for future function
calling). Every call goes through ``Toolbox.call``, which validates arguments, checks state
preconditions and returns a ``ToolResult``. Tools act only through the ``Embodiment``
(controller setpoints with safety limits enforced underneath) and read only ``Perception``.
They cannot move objects directly, write torques, or see simulator ground truth.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from dream.skills import manipulation as skills
from dream.tools.goals import InRegion, OnTop, parse_goal, resting_on, verify
from dream.types import Embodiment, Perception

HOLD_MIN_WIDTH = 0.01  # m: closed fingers stopped wider than this are holding something
HOLD_MAX_DISTANCE = 0.04  # m: an object this close to the TCP is the one being held
TABLE_REGION = ((0.35, 0.65), (-0.28, 0.28))  # m: where the arm can comfortably set objects down
# m beyond an object's size between neighbour centres: the closed fingers stick out ~1.5 cm
# past a held cube, so a tighter spacing makes them strike the neighbour while lowering.
FINGER_CLEARANCE = 0.025

TOOL_SPECS: list[dict] = [
    {
        "name": "observe_scene",
        "description": "List visible objects (pose estimates) and fixtures.",
        "parameters": {},
    },
    {
        "name": "get_robot_state",
        "description": "Joint state, TCP pose, gripper state, safety status, capabilities.",
        "parameters": {},
    },
    {
        "name": "move_end_effector",
        "description": "Move the gripper (pointing down) to a position and yaw.",
        "parameters": {
            "position": {"type": "vector3", "required": True},
            "yaw": {"type": "number", "required": False},
        },
    },
    {
        "name": "grasp",
        "description": "Pick up a named object from above. Requires an empty gripper.",
        "parameters": {"object": {"type": "string", "required": True}},
    },
    {"name": "release", "description": "Open the gripper.", "parameters": {}},
    {
        "name": "place",
        "description": "Put the held object in a region ('tray'), on an object, or on 'table'.",
        "parameters": {
            "object": {"type": "string", "required": True},
            "target": {"type": "string", "required": True},
        },
    },
    {
        "name": "verify_goal",
        "description": "Check a structured goal against observations, with margin.",
        "parameters": {"goal": {"type": "goal", "required": True}},
    },
]
SPECS = {s["name"]: s for s in TOOL_SPECS}


@dataclass
class ToolResult:
    tool: str
    status: str  # "succeeded" | "failed" | "rejected" | "timeout" | "stopped"
    message: str = ""
    data: dict = field(default_factory=dict)
    sim_time: float = 0.0  # s of robot time spent

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


class Cancelled(Exception):
    pass


class Toolbox:
    def __init__(
        self, robot: Embodiment, perception: Perception, cancel: threading.Event | None = None
    ):
        self.robot = robot
        self.perception = perception
        self.cancel = cancel or threading.Event()

    # ----- entry point ----------------------------------------------------------------

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        if name not in SPECS:
            return ToolResult(name, "rejected", f"unknown tool '{name}'")
        problem = _validate(SPECS[name]["parameters"], args)
        if problem:
            return ToolResult(name, "rejected", problem)
        if self.cancel.is_set():
            return ToolResult(name, "stopped", "cancelled before start")
        if self.robot.state().protective_stop and name not in (
            "observe_scene",
            "get_robot_state",
            "verify_goal",
        ):
            return ToolResult(
                name, "rejected", f"robot in protective stop: {self.robot.state().stop_reason}"
            )
        t0 = self.robot.state().time
        try:
            result = getattr(self, f"_{name}")(**args)
        except Cancelled:
            self.robot.hold()
            result = ToolResult(name, "stopped", "cancelled; robot holding position")
        result.sim_time = self.robot.state().time - t0
        return result

    # ----- tools ------------------------------------------------------------------------

    def _observe_scene(self) -> ToolResult:
        objects = [
            {
                "name": o.name,
                "color": o.color,
                "position": np.round(o.position, 4).tolist(),
                "yaw": round(o.yaw, 3),
                "size": o.size,
            }
            for o in self.perception.observe_objects()
        ]
        fixtures = {
            k: {"centre": np.round(c, 4).tolist(), "half_width": h}
            for k, (c, h) in self.perception.fixtures().items()
        }
        return ToolResult(
            "observe_scene",
            "succeeded",
            data={"objects": objects, "fixtures": fixtures, "holding": self.held_object()},
        )

    def _get_robot_state(self) -> ToolResult:
        s = self.robot.state()
        return ToolResult(
            "get_robot_state",
            "succeeded",
            data={
                "joint_pos": np.round(s.joint_pos, 4).tolist(),
                "tcp_pos": np.round(s.tcp_pos, 4).tolist(),
                "tcp_yaw": round(skills.tcp_yaw(s.tcp_quat), 3),
                "gripper_width": round(s.gripper_width, 4),
                "holding": self.held_object(),
                "protective_stop": s.protective_stop,
                "capabilities": [t["name"] for t in TOOL_SPECS],
            },
        )

    def _move_end_effector(self, position, yaw: float | None = None) -> ToolResult:
        y = skills.tcp_yaw(self.robot.state().tcp_quat) if yaw is None else float(yaw)
        r = self._guard(lambda rb: skills.move_to(rb, np.asarray(position, float), y))
        return self._from_skill("move_end_effector", r)

    def _grasp(self, object: str) -> ToolResult:
        if self.held_object() is not None:
            return ToolResult(
                "grasp", "rejected", f"precondition: gripper already holds '{self.held_object()}'"
            )
        if skills.find(self.perception, object) is None:
            return ToolResult("grasp", "rejected", f"precondition: '{object}' is not visible")
        blockers = resting_on(object, self.perception.observe_objects())
        if blockers:
            return ToolResult("grasp", "rejected", f"precondition: {blockers} rest on '{object}'")
        r = self._guard(lambda rb: skills.pick(rb, self.perception, object))
        if r.ok and self.held_object() != object:
            return ToolResult(
                "grasp",
                "failed",
                f"holding '{self.held_object()}' instead of '{object}'",
                r.details,
            )
        return self._from_skill("grasp", r)

    def _release(self) -> ToolResult:
        width = min(skills.OPEN_WIDTH, self.robot.state().gripper_width + skills.RELEASE_SLACK)
        r = self._guard(lambda rb: skills.set_gripper(rb, width))
        return self._from_skill("release", r)

    def _place(self, object: str, target: str) -> ToolResult:
        held = self.held_object()
        if held != object:
            return ToolResult(
                "place", "rejected", f"precondition: holding {held!r}, not '{object}'"
            )
        on_target = [
            b for b in resting_on(target, self.perception.observe_objects()) if b != object
        ]
        if on_target:
            return ToolResult("place", "rejected", f"precondition: {on_target} occupy '{target}'")
        point = self.target_point(object, target)
        if point is None:
            return ToolResult("place", "rejected", f"unknown or unseen target '{target}'")
        r = self._guard(lambda rb: skills.place(rb, self.perception, object, point))
        return self._from_skill("place", r)

    def _verify_goal(self, goal) -> ToolResult:
        try:
            g = goal if isinstance(goal, (InRegion, OnTop)) else parse_goal(goal)
        except ValueError as e:
            return ToolResult("verify_goal", "rejected", str(e))
        v = verify(g, self.perception.observe_objects(), self.perception.fixtures())
        return ToolResult(
            "verify_goal",
            "succeeded",
            v.reason,
            {"satisfied": v.satisfied, "margin_m": v.margin, "uncertain": v.uncertain},
        )

    # ----- helpers ------------------------------------------------------------------------

    def held_object(self) -> str | None:
        """Infer what is held from finger opening and the observed object nearest the TCP."""
        s = self.robot.state()
        if s.gripper_command > 0.01 or s.gripper_width < HOLD_MIN_WIDTH:
            return None
        objs = self.perception.observe_objects()
        if not objs:
            return None
        nearest = min(objs, key=lambda o: np.linalg.norm(o.position - s.tcp_pos))
        return (
            nearest.name
            if np.linalg.norm(nearest.position - s.tcp_pos) < HOLD_MAX_DISTANCE
            else None
        )

    def target_point(self, object: str, target: str) -> np.ndarray | None:
        """Where the object's centre should end up for a named target."""
        obj = skills.find(self.perception, object)
        fixtures = self.perception.fixtures()
        if obj is None:
            return None
        if target in fixtures:
            centre, half = fixtures[target]
            # Free spot inside the region, clear of objects already there and of the walls. Packing
            # from a corner leaves room for more objects than filling the centre first.
            inner = half - obj.size / 2 - 0.01
            lo = centre[:2] - inner
            spot = self.free_spot(
                obj,
                ((lo[0], centre[0] + inner), (lo[1], centre[1] + inner)),
                obj.size + FINGER_CLEARANCE,
                [],
                lo,
                step=0.005,
            )
            return None if spot is None else np.array([*spot, obj.size / 2 + 0.004])  # tray floor
        if target == "table":
            boxes = [(c[:2], h + 0.03) for c, h in fixtures.values()]
            spot = self.free_spot(obj, TABLE_REGION, 0.08, boxes, obj.position[:2])
            return None if spot is None else np.array([*spot, obj.size / 2])
        support = skills.find(self.perception, target)
        if support is None:
            return None
        return support.position + [0, 0, (support.size + obj.size) / 2]

    def free_spot(self, obj, region, clearance, boxes, prefer, step: float = 0.02):
        """Point in ``region`` at least ``clearance`` from other objects and outside ``boxes``,
        nearest to ``prefer``. Uses observations only. Returns None if nothing fits."""
        others = [o.position[:2] for o in self.perception.observe_objects() if o.name != obj.name]
        (x0, x1), (y0, y1) = region
        best, best_d = None, np.inf
        for x in np.arange(x0, x1 + 1e-9, step):
            for y in np.arange(y0, y1 + 1e-9, step):
                p = np.array([x, y])
                if any(np.linalg.norm(p - q) < clearance for q in others):
                    continue
                if any(np.max(np.abs(p - c)) < h + obj.size for c, h in boxes):
                    continue
                d = np.linalg.norm(p - prefer)
                if d < best_d:
                    best, best_d = p, d
        return best

    def _guard(self, fn):
        """Run a skill with an embodiment whose waits end early when cancel is requested."""
        return fn(_CancellableRobot(self.robot, self.cancel))

    @staticmethod
    def _from_skill(tool: str, r: skills.SkillResult) -> ToolResult:
        if r.ok:
            return ToolResult(tool, "succeeded", "", r.details)
        status = (
            "timeout"
            if r.reason.startswith("timeout")
            else "stopped"
            if "protective stop" in r.reason
            else "failed"
        )
        if "workspace" in r.reason:
            status = "rejected"
        return ToolResult(tool, status, r.reason, r.details)


class _CancellableRobot:
    """Embodiment wrapper: waits also end on cancel, then raise ``Cancelled``."""

    def __init__(self, robot: Embodiment, cancel: threading.Event) -> None:
        self._robot = robot
        self._cancel = cancel

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def wait_until(self, predicate, timeout: float) -> bool:
        ok = self._robot.wait_until(lambda: predicate() or self._cancel.is_set(), timeout)
        if self._cancel.is_set():
            raise Cancelled
        return ok


def _validate(params: dict, args: dict) -> str:
    for key in args:
        if key not in params:
            return f"unexpected argument '{key}'"
    for key, p in params.items():
        if key not in args:
            if p.get("required"):
                return f"missing required argument '{key}'"
            continue
        v = args[key]
        t = p["type"]
        if t == "string" and not (isinstance(v, str) and v):
            return f"'{key}' must be a non-empty string"
        if t == "number" and not (isinstance(v, (int, float)) and math.isfinite(v)):
            return f"'{key}' must be a finite number"
        if t == "vector3":
            try:
                arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                return f"'{key}' must be 3 numbers"
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                return f"'{key}' must be 3 finite numbers"
        if t == "goal" and not isinstance(v, (str, dict, InRegion, OnTop)):
            return f"'{key}' must be a structured goal"
    return ""
