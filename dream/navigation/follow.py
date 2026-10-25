"""Follow a path of odometry-frame waypoints with velocity commands for a walking base.

A classical pure-pursuit-style follower: turn in place when the next point is far off the
heading, otherwise walk forward with a heading correction, slowing near the goal. It sees
only the robot's own odometry estimate and the path; obstacle clearance is the caller's
(``blocked`` stops it). Gains and limits are commissioning choices inside the locomotion
layer's envelope (dream/robot/g1_wbc.MAX_COMMAND), not tuned per scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi

import numpy as np


def wrap(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


@dataclass
class PathFollower:
    max_forward: float = 0.4  # m/s
    max_turn: float = 0.5  # rad/s
    lookahead: float = 0.35  # m
    goal_tolerance: float = 0.15  # m
    heading_tolerance: float = 0.15  # rad, for a final heading if one is given
    turn_in_place: float = 0.6  # rad: larger heading errors are corrected before walking
    path: list[np.ndarray] = field(default_factory=list)
    final_heading: float | None = None

    def set_path(self, points, final_heading: float | None = None) -> None:
        self.path = [np.asarray(p, float)[:2] for p in points]
        self.final_heading = final_heading

    def update(self, pose: np.ndarray, blocked: bool = False) -> tuple[float, float, str]:
        """(forward m/s, turn rad/s, status) for odometry pose (x, y, yaw)."""
        if not self.path:
            return 0.0, 0.0, "idle"
        if blocked:
            return 0.0, 0.0, "blocked"
        x, y, yaw = pose
        here = np.array([x, y])
        while len(self.path) > 1 and np.linalg.norm(self.path[0] - here) < self.lookahead:
            self.path.pop(0)
        goal = self.path[0]
        dist = float(np.linalg.norm(goal - here))
        if len(self.path) == 1 and dist < self.goal_tolerance:
            if self.final_heading is None:
                self.path = []
                return 0.0, 0.0, "arrived"
            err = wrap(self.final_heading - yaw)
            if abs(err) < self.heading_tolerance:
                self.path = []
                return 0.0, 0.0, "arrived"
            return 0.0, float(np.clip(1.5 * err, -self.max_turn, self.max_turn)), "aligning"
        err = wrap(atan2(goal[1] - y, goal[0] - x) - yaw)
        turn = float(np.clip(1.5 * err, -self.max_turn, self.max_turn))
        if abs(err) > self.turn_in_place:
            return 0.0, turn, "turning"
        remaining = dist + sum(
            hypot(*(b - a)) for a, b in zip(self.path, self.path[1:], strict=False)
        )
        forward = min(self.max_forward, 0.8 * remaining) * max(0.0, cos(err))
        return float(forward), turn, "walking"
