"""Typed tools for a mobile robot exploring a room: the same call()/ToolResult contract as the
Panda toolbox, so TaskAgent (and later a model-driven agent) can drive either.

Everything here sees the robot only through ``MobileBase`` (odometry, head frames, velocity
commands to its own controller) and keeps what it learns in ``SpatialMemory`` for this
episode only. Movement is planned on the observed occupancy map, re-observed and re-planned
about once a second, and stopped when the map shows an obstacle right in front.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from dream.navigation.follow import PathFollower, wrap
from dream.navigation.planning import (
    ROBOT_RADIUS,
    distances,
    frontiers,
    path_to,
    traversable,
)
from dream.perception.occupancy import CELL
from dream.perception.spatial import CLOSE_VIEW, SIDES, SpatialMemory
from dream.tools.goals import Inspect, parse_goal
from dream.tools.toolbox import ToolResult, _validate
from dream.types import MobileBase

VIEW_DISTANCE = 1.0  # m beyond an object's extent to look for a handle: inside CLOSE_VIEW
REPLAN_S = 1.0  # s of walking between observations
STEP_S = 0.1  # s per velocity update of the path follower
GO_TO_TIMEOUT = 60.0  # s
FRONTIER_COST_WEIGHT = 1.0  # m^2 of expected new area per metre walked (score = gain - w*cost)
# Inspection pose: within this band of the handle, facing it.
INSPECT_RANGE = (0.6, CLOSE_VIEW)
INSPECT_ANGLE = math.radians(35)

MOBILE_TOOL_SPECS: list[dict] = [
    {
        "name": "observe_scene",
        "description": "Look with the head camera; report the map, candidate objects, "
        "unexplored frontiers and viewpoints for sides not yet seen up close.",
        "parameters": {},
    },
    {
        "name": "get_robot_state",
        "description": "Odometry pose, time and safety status.",
        "parameters": {},
    },
    {
        "name": "verify_goal",
        "description": "Check a structured goal against current observations.",
        "parameters": {"goal": {"type": "goal", "required": True}},
    },
    {
        "name": "look_around",
        "description": "Turn in place a full circle, looking every 45 degrees.",
        "parameters": {},
    },
    {
        "name": "go_to",
        "description": "Walk to an odometry-frame position over observed free space.",
        "parameters": {
            "x": {"type": "number", "required": True},
            "y": {"type": "number", "required": True},
            "heading": {"type": "number", "required": False},
        },
    },
    {
        "name": "approach",
        "description": "Walk to a viewpoint about 1 m from a candidate object on one side "
        "(side 0-7: bearing k*45 degrees from its centre in the odometry frame), facing it.",
        "parameters": {
            "candidate": {"type": "number", "required": True},
            "side": {"type": "number", "required": True},
        },
    },
    {
        "name": "inspect",
        "description": "Look closely at a candidate from where the robot stands; report what "
        "was seen and what remains unknown.",
        "parameters": {"candidate": {"type": "number", "required": True}},
    },
]
MOBILE_SPECS = {s["name"]: s for s in MOBILE_TOOL_SPECS}


@dataclass(frozen=True)
class Viewpoint:
    candidate: int
    side: int
    xy: np.ndarray
    heading: float
    cost: float  # m of walking


class MobileToolbox:
    def __init__(self, base: MobileBase, memory: SpatialMemory, on_event=None) -> None:
        self.base, self.memory = base, memory
        self.inspections: dict[int, dict] = {}  # candidate id -> last inspection report
        self.cancel = threading.Event()  # TaskAgent checks it between actions
        self.on_event = on_event or (lambda kind, text, **data: None)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        if name not in MOBILE_SPECS:
            return ToolResult(name, "rejected", f"unknown tool '{name}'")
        problem = _validate(MOBILE_SPECS[name]["parameters"], args)
        if problem:
            return ToolResult(name, "rejected", problem)
        if self.base.stop_reason and name not in ("observe_scene", "get_robot_state"):
            return ToolResult(name, "rejected", f"robot stopped: {self.base.stop_reason}")
        t0 = self.base.time
        result = getattr(self, f"_{name}")(**args)
        result.sim_time = self.base.time - t0
        return result

    # ----- perception ------------------------------------------------------------------
    def _look(self) -> None:
        self.memory.update(self.base.capture())

    def _passable(self) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        grid = self.memory.grid
        ok = traversable(grid)
        pose = self.base.odometry()
        i, j = grid.cell(pose[:2])
        r = int(0.3 / CELL)  # the robot's own footprint is where it stands: passable
        ii, jj = np.ogrid[-r : r + 1, -r : r + 1]
        disk = ii * ii + jj * jj <= r * r
        ok[i - r : i + r + 1, j - r : j + r + 1] |= (
            disk & ~grid.occupied()[i - r : i + r + 1, j - r : j + r + 1]
        )
        return ok, (int(i), int(j)), distances(ok, (int(i), int(j)))

    def viewpoints(self, passable, dist) -> list[Viewpoint]:
        """Reachable viewpoints on sides of candidates not yet seen up close (and always on
        the side a handle was seen on)."""
        out = []
        for c in self.memory.candidates:
            for k in range(SIDES):
                if k in c.sides_close and c.handle_side() != k:
                    continue
                chosen = self.side_point(c, k, passable, dist)
                if chosen is None:
                    continue
                xy, n, cost = chosen
                out.append(Viewpoint(c.id, k, xy, math.atan2(-n[1], -n[0]), cost))
        return out

    def side_point(self, c, k: int, passable, dist):
        """(where to stand, outward direction, walking cost) to look at side ``k``, or None.

        On the handle's side: VIEW_DISTANCE in front of the handle itself; elsewhere: the
        object's extent plus VIEW_DISTANCE along the bearing. The direction is rotated by up
        to 60 degrees (least first) if the straight one is not reachable: a face is legible
        obliquely, and the outward direction from a partial-view centre estimate is rough
        (development room 4 put it against a wall)."""
        grid = self.memory.grid
        if c.handle is not None and c.handle_side() == k:
            anchor, base = c.handle[:2], c.handle[:2] - c.position[:2]
            reach = VIEW_DISTANCE
        else:
            anchor, base, reach = c.position[:2], c.side_direction(k), c.radius + VIEW_DISTANCE
        base = base / max(float(np.linalg.norm(base)), 1e-6)
        for a in np.radians([0, 30, -30, 60, -60]):
            ca, sa = math.cos(a), math.sin(a)
            n = np.array([ca * base[0] - sa * base[1], sa * base[0] + ca * base[1]])
            xy = anchor + n * reach
            i, j = grid.cell(xy)
            if grid.inside(i, j) and passable[i, j] and np.isfinite(dist[i, j]):
                return xy, n, float(dist[i, j])
        return None

    def _observe_scene(self) -> ToolResult:
        self._look()
        passable, _, dist = self._passable()
        grid = self.memory.grid
        fr = frontiers(grid, passable, dist)
        fr.sort(key=lambda f: -(f.gain - FRONTIER_COST_WEIGHT * f.cost))
        data = {
            "pose": np.round(self.base.odometry(), 3).tolist(),
            "odometry_source": "robot odometry estimate",
            "explored_m2": float(grid.free().sum() * CELL * CELL),
            "candidates": [c.summary() for c in self.memory.candidates],
            "viewpoints": [
                {
                    "candidate": v.candidate,
                    "side": v.side,
                    "xy": np.round(v.xy, 3).tolist(),
                    "heading": round(v.heading, 3),
                    "cost_m": round(v.cost, 2),
                }
                for v in self.viewpoints(passable, dist)
            ],
            "frontiers": [
                {
                    "xy": np.round(grid.centre(*f.cell), 3).tolist(),
                    "gain_m2": round(f.gain, 2),
                    "cost_m": round(f.cost, 2),
                }
                for f in fr[:8]
            ],
            "inspected": {str(k): v for k, v in self.inspections.items()},
            "holding": None,
            "holding_handle": None,
        }
        return ToolResult("observe_scene", "succeeded", "", data)

    def _get_robot_state(self) -> ToolResult:
        return ToolResult(
            "get_robot_state",
            "succeeded",
            "",
            {
                "time": self.base.time,
                "odometry": np.round(self.base.odometry(), 3).tolist(),
                "protective_stop": self.base.stop_reason is not None,
                "stop_reason": self.base.stop_reason,
            },
        )

    def _verify_goal(self, goal) -> ToolResult:
        goal = parse_goal(goal) if isinstance(goal, (str, dict)) else goal
        if not isinstance(goal, Inspect):
            return ToolResult("verify_goal", "rejected", "this robot verifies 'inspect' goals")
        found = [c for c in self.memory.candidates if c.handle is not None]
        if not found:
            return ToolResult(
                "verify_goal",
                "succeeded",
                "",
                {"satisfied": False, "reason": "no box with a handle seen"},
            )
        c = max(found, key=lambda c: c.handle_views)
        at, why = self.at_inspection_pose(c)
        report = self.inspections.get(c.id, {})
        done = at and report.get("handle_confirmed_close", False)
        if done:
            reason = "handle re-detected while inspecting from an inspection pose"
        elif at and report:
            reason = "inspected, but the handle was not seen again up close"
        else:
            reason = why or "not inspected yet"
        return ToolResult(
            "verify_goal", "succeeded", "", {"satisfied": done, "reason": reason, "candidate": c.id}
        )

    def at_inspection_pose(self, c) -> tuple[bool, str]:
        """Within INSPECT_RANGE of the handle, on its side of the object, facing it."""
        if c.handle is None:
            return False, "no handle seen"
        pose = self.base.odometry()
        h = c.handle[:2]
        d = float(np.linalg.norm(pose[:2] - h))
        to_handle = math.atan2(h[1] - pose[1], h[0] - pose[0])
        outward = h - c.position[:2]
        if not INSPECT_RANGE[0] <= d <= INSPECT_RANGE[1]:
            return False, f"{d:.2f} m from the handle"
        if float((pose[:2] - h) @ outward) <= 0:
            return False, "not on the handle's side of the object"
        if abs(wrap(to_handle - pose[2])) > INSPECT_ANGLE:
            return False, "not facing the handle"
        return True, ""

    # ----- actions ---------------------------------------------------------------------
    def _look_around(self) -> ToolResult:
        self._look()
        for _ in range(8):
            if not self.base.walk(0.0, 0.0, 0.5, math.pi / 4 / 0.5) or not self.base.walk(
                0, 0, 0, 0.3
            ):
                return ToolResult("look_around", "stopped", self.base.stop_reason or "")
            self._look()
        return ToolResult("look_around", "succeeded", "", {"frames": self.memory.frames})

    def _blocked(self) -> bool:
        """An occupied cell closer than the planner's clearance ahead (60-degree cone): the map
        changed under the path. A stricter stop would block paths the planner rightly allows."""
        grid = self.memory.grid
        x, y, yaw = self.base.odometry()
        occ = np.argwhere(grid.occupied())
        if not len(occ):
            return False
        pts = grid.centre(occ[:, 0], occ[:, 1]) - [x, y]
        d = np.linalg.norm(pts, axis=1)
        ahead = np.abs(np.vectorize(wrap)(np.arctan2(pts[:, 1], pts[:, 0]) - yaw)) < math.pi / 3
        return bool(np.any((d < ROBOT_RADIUS - 0.03) & ahead))

    def _go_to(self, x: float, y: float, heading: float | None = None) -> ToolResult:
        goal = np.array([x, y], float)
        t0 = self.base.time
        follower = PathFollower()
        blocked_streak = 0
        while self.base.time - t0 < GO_TO_TIMEOUT:
            self._look()
            passable, _, dist = self._passable()
            grid = self.memory.grid
            gi, gj = grid.cell(goal)
            if not grid.inside(gi, gj):
                return ToolResult("go_to", "failed", "goal outside the mapped area")
            # The reachable cell nearest the goal, within 0.3 m of it (the goal itself may be
            # in an inflated or unobserved cell).
            r = int(0.3 / CELL)
            i0, j0 = max(0, gi - r), max(0, gj - r)
            win = dist[i0 : gi + r + 1, j0 : gj + r + 1]
            if not np.isfinite(win).any():
                return ToolResult("go_to", "failed", "no path over observed free space")
            ii, jj = np.nonzero(np.isfinite(win))
            k = int(np.argmin((ii + i0 - gi) ** 2 + (jj + j0 - gj) ** 2))
            cell = (int(ii[k] + i0), int(jj[k] + j0))
            cells = path_to(dist, cell)
            points = [grid.centre(i, j) for i, j in cells[3::4]] + [grid.centre(*cell)]
            follower.set_path(points, heading)
            for _ in range(round(REPLAN_S / STEP_S)):
                vx, wz, status = follower.update(self.base.odometry(), self._blocked())
                if status == "arrived":
                    self.base.walk(0, 0, 0, 0.3)
                    err = float(np.linalg.norm(self.base.odometry()[:2] - goal))
                    return ToolResult("go_to", "succeeded", "", {"error_m": err})
                if status == "blocked":
                    blocked_streak += 1
                    # Back off a little so the planner can find a way around.
                    if not self.base.walk(-0.15, 0, 0, 0.5):
                        return ToolResult("go_to", "stopped", self.base.stop_reason or "")
                    break
                blocked_streak = 0
                if not self.base.walk(vx, 0.0, wz, STEP_S):
                    return ToolResult("go_to", "stopped", self.base.stop_reason or "")
            if blocked_streak >= 5:
                return ToolResult("go_to", "failed", "path repeatedly blocked")
        self.base.walk(0, 0, 0, 0.3)
        return ToolResult("go_to", "timeout", f"not there after {GO_TO_TIMEOUT:.0f} s")

    def _candidate(self, cid: float):
        return next((c for c in self.memory.candidates if c.id == int(cid)), None)

    def _approach(self, candidate: float, side: float) -> ToolResult:
        c = self._candidate(candidate)
        if c is None or int(side) not in range(SIDES):
            return ToolResult("approach", "rejected", "unknown candidate or side")
        k = int(side)
        passable, _, dist = self._passable()
        chosen = self.side_point(c, k, passable, dist)
        if chosen is None:
            return ToolResult("approach", "failed", "no reachable viewpoint on that side")
        xy, n, _ = chosen
        self.on_event("action", f"approach candidate {c.id} ({c.color}) side {k}")
        r = self._go_to(float(xy[0]), float(xy[1]), math.atan2(-n[1], -n[0]))
        self._look()
        return ToolResult("approach", r.status, r.message, {**r.data, "candidate": c.summary()})

    def _inspect(self, candidate: float) -> ToolResult:
        c = self._candidate(candidate)
        if c is None:
            return ToolResult("inspect", "rejected", "unknown candidate")
        before = c.handle_views
        for turn in (0.0, 0.25, -0.5, 0.25):  # look straight, a little left, right, back
            if turn and not self.base.walk(0, 0, math.copysign(0.4, turn), abs(turn) / 0.4):
                return ToolResult("inspect", "stopped", self.base.stop_reason or "")
            self.base.walk(0, 0, 0, 0.3)
            self._look()
        at, why = self.at_inspection_pose(c) if c.handle is not None else (False, "")
        report = {
            # Seen again in this close-range look, not just inferred from earlier views.
            "handle_confirmed_close": c.handle_views > before,
            "candidate": c.summary(),
            "seen": "a box-like object"
            + (" with a handle on one face" if c.handle is not None else " with no handle seen"),
            "inferred": "probably opens (a handle suggests a door, drawer or lid)"
            if c.handle is not None
            else "no evidence it opens from the faces seen",
            "unknown": "how it moves: hinge side, sliding, lift or fixed; resistance; latches. "
            "Not observable without a physical interaction.",
            "would_resolve_it": "a force-limited probe of the handle along candidate directions",
            "at_inspection_pose": at,
            "pose_note": why,
        }
        self.inspections[c.id] = report
        return ToolResult("inspect", "succeeded", report["seen"], report)
