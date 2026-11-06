"""Use home memory to start a search where the object was last seen, then search as usual.

Wraps a search planner (``RoomSearchPlanner``): before exploring, it walks to a viewpoint of
each remembered location of the goal object (best belief first) and looks. If the object is
there, the inner planner takes over immediately (it now sees a candidate with a handle).
If the location was in clear view and nothing matching is there, the memory records an
absence: that belief becomes "missing" and is never trusted again without a new sighting.
Remembered is never treated as current: a belief only shortens the search.

The remembered position must be in the frame the robot navigates in; see
``dream.memory.home`` on map frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dream.memory.home import HomeMemory
from dream.runtime.agent import ToolCall

VIEW_RANGE = 1.0  # m: stand this far from the remembered spot (the inspection distance)
MATCH_RADIUS = 0.5  # m: a detected handle this close to the remembered one is a re-sighting
IN_VIEW_ANGLE = math.radians(25)  # within the head camera's 58 degree field of view
IN_VIEW_RANGE = 2.0  # m: inside reliable depth range (dream/navigation/planning.SENSE_RANGE)


def remembered_label(goal) -> str:
    return f"{goal.target}_handle"


@dataclass
class MemoryGuidedPlanner:
    inner: object
    memory: HomeMemory
    map_id: str
    now: float
    checked: set = field(default_factory=set)
    pending: str | None = None  # belief being checked by the last go_to
    log: list = field(default_factory=list)

    def next_action(self, goal, scene, verdict, history):
        if verdict.get("satisfied") or not any(s.call.name == "look_around" for s in history):
            return self.inner.next_action(goal, scene, verdict, history)  # map the start first
        handles = [np.array(c["handle"][:2]) for c in scene["candidates"] if c["handle"]]
        pose = np.array(scene["pose"])
        if self.pending is not None:  # just looked from the remembered location's viewpoint
            b = self._belief(goal, self.pending)
            self.checked.add(self.pending)
            seen = any(np.linalg.norm(h - b.position[:2]) < MATCH_RADIUS for h in handles)
            reached = history[-1].call.name == "go_to" and history[-1].result.ok
            if not seen and reached and self._in_view(pose, b.position[:2]):
                self.memory.record_absence(b.position[:2], MATCH_RADIUS, self.map_id,
                                           source="head_rgbd", time=self.now)  # fmt: skip
                self.log.append(("absent", b.object_id))
            else:
                self.log.append(("present" if seen else "not_in_view", b.object_id))
            self.pending = None
        for b in self.memory.where_is(remembered_label(goal), self.map_id, now=self.now):
            if b.state == "missing" or b.object_id in self.checked:
                continue
            if any(np.linalg.norm(h - b.position[:2]) < MATCH_RADIUS for h in handles):
                self.checked.add(b.object_id)  # already seen again: let the search use it
                self.log.append(("present", b.object_id))
                continue
            d = b.position[:2] - pose[:2]
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                continue
            stand = b.position[:2] - d / dist * min(VIEW_RANGE, dist)
            self.pending = b.object_id
            self.log.append(("check", b.object_id))
            return ToolCall("go_to", {"x": float(stand[0]), "y": float(stand[1]),
                                      "heading": math.atan2(d[1], d[0])})  # fmt: skip
        return self.inner.next_action(goal, scene, verdict, history)

    def _belief(self, goal, oid):
        return next(
            b for b in self.memory.where_is(remembered_label(goal), self.map_id, now=self.now)
            if b.object_id == oid
        )  # fmt: skip

    @staticmethod
    def _in_view(pose, xy) -> bool:
        d = np.asarray(xy) - pose[:2]
        bearing = math.atan2(d[1], d[0]) - pose[2]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return float(np.linalg.norm(d)) < IN_VIEW_RANGE and abs(bearing) < IN_VIEW_ANGLE


def remember_inspections(toolbox, memory: HomeMemory, map_id: str, goal, now: float) -> list:
    """Store handles that were re-detected up close while inspecting (verified sightings)."""
    stored = []
    for cid, report in toolbox.inspections.items():
        c = next((c for c in toolbox.memory.candidates if c.id == cid), None)
        if c is None or c.handle is None or not report.get("handle_confirmed_close"):
            continue
        oid = memory.observe(remembered_label(goal), c.handle, map_id, sigma=0.1,
                             source="head_rgbd inspection", time=now)  # fmt: skip
        stored.append(oid)
    return stored
