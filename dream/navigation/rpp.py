"""Smooth path tracking for a walking base: Nav2's Regulated Pure Pursuit, adapted.

Reimplements the control law of Nav2's ``nav2_regulated_pure_pursuit_controller``
(ros-navigation/navigation2, Apache-2.0; Macenski et al., "Regulated Pure Pursuit for Robot
Path Tracking", Autonomous Robots 2023), with the velocity limits of Nav2's
``nav2_velocity_smoother`` and the line-of-sight shortcutting and gradient smoothing of
``nav2_smoother``'s simple smoother. No ROS dependency: the same command interface as
``PathFollower`` (forward m/s, turn rad/s, status), inside the G1 policy's envelope.

Differences from the old follower that matter for a humanoid:
- it steers toward a lookahead point *on* the path (velocity-scaled distance), so heading
  errors are corrected while walking along a curve, not by stopping to turn;
- it turns in place only when the path direction is more than ``rotate_min_angle`` away
  (Nav2's rotate-to-heading), and aligns to a final heading only beyond its tolerance;
- speed is regulated down on tight curvature, near obstacles and near the goal;
- commands are rate-limited (linear and angular acceleration), so no step changes;
- the commanded arc is projected ahead and checked against the map before it is executed.

Parameters are Nav2 defaults unless noted; DREAM-specific values say why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, cos, hypot, sin

import numpy as np

from dream.navigation.follow import wrap


def line_of_sight(passable: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Every cell on the segment a-b (sampled at half a cell) is passable."""
    n = int(2 * max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    ii = np.rint(np.linspace(a[0], b[0], n + 1)).astype(int)
    jj = np.rint(np.linspace(a[1], b[1], n + 1)).astype(int)
    return bool(passable[ii, jj].all())


def shortcut(cells: list[tuple[int, int]], passable: np.ndarray, start: tuple[int, int]):
    """String-pull a grid path: keep only cells where line of sight to the anchor breaks."""
    if not cells:
        return []
    out, anchor = [], start
    for k in range(1, len(cells)):
        if not line_of_sight(passable, anchor, cells[k]):
            out.append(cells[k - 1])
            anchor = cells[k - 1]
    out.append(cells[-1])
    return out


def densify(points: list[np.ndarray], step: float = 0.05) -> list[np.ndarray]:
    out = [np.asarray(points[0], float)]
    for a, b in zip(points, points[1:], strict=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return out


def smooth(
    points: list[np.ndarray],
    is_free: Callable[[np.ndarray], bool],
    w_data: float = 0.2,
    w_smooth: float = 0.3,
    iterations: int = 200,
) -> list[np.ndarray]:
    """Nav2 simple smoother (w_data 0.2, w_smooth 0.3); endpoints fixed, moves that would
    leave free space are rejected per point."""
    p = np.array(points, float)
    if len(p) < 3:
        return list(p)
    orig = p.copy()
    for _ in range(iterations):
        change = 0.0
        for i in range(1, len(p) - 1):
            new = p[i] + w_data * (orig[i] - p[i]) + w_smooth * (p[i - 1] + p[i + 1] - 2 * p[i])
            if is_free(new):
                change += float(np.abs(new - p[i]).sum())
                p[i] = new
        if change < 1e-4:
            break
    return list(p)


@dataclass
class RegulatedPurePursuit:
    desired_speed: float = 0.4  # m/s: the old follower's cruise speed (G1 envelope 0.5)
    max_turn: float = 0.5  # rad/s: as the old follower (G1 envelope 0.6)
    lookahead_time: float = 1.5  # s
    min_lookahead: float = 0.3  # m
    max_lookahead: float = 0.9  # m
    rotate_min_angle: float = 0.785  # rad, Nav2 rotate_to_heading_min_angle
    rotate_speed: float = 0.5  # rad/s while rotating to heading
    min_radius: float = 0.9  # m, regulated_linear_scaling_min_radius
    min_speed: float = 0.15  # m/s floor of curvature/cost regulation (Nav2 0.25; G1 walks
    # slower in tight rooms, and 0.15 is inside the commissioned range)
    approach_dist: float = 0.6  # m, approach_velocity_scaling_dist
    min_approach_speed: float = 0.05  # m/s
    clearance_scaling_dist: float = 0.6  # m: slow down closer than this to obstacles
    goal_tolerance: float = 0.15  # m, as the old follower
    heading_tolerance: float = 0.25  # rad: the camera's 58 deg field tolerates this
    max_accel: float = 0.5  # m/s^2 (velocity smoother; the T1 prototype's command slew)
    max_decel: float = 0.8  # m/s^2
    max_angular_accel: float = 1.0  # rad/s^2: gentler than Nav2's 3.2 for a biped
    collision_time: float = 1.0  # s, max_allowed_time_to_collision_up_to_carrot
    dt: float = 0.1  # s between updates (the toolbox's STEP_S)
    path: list[np.ndarray] = field(default_factory=list)
    final_heading: float | None = None
    v: float = 0.0
    w: float = 0.0

    def set_path(self, points, final_heading: float | None = None) -> None:
        self.path = [np.asarray(p, float)[:2] for p in points]
        self.final_heading = final_heading

    def _smoothed(self, v: float, w: float) -> tuple[float, float]:
        dv = np.clip(v - self.v, -self.max_decel * self.dt, self.max_accel * self.dt)
        dw = np.clip(
            w - self.w, -self.max_angular_accel * self.dt, self.max_angular_accel * self.dt
        )
        self.v, self.w = float(self.v + dv), float(self.w + dw)
        return self.v, self.w

    def _carrot(self, here: np.ndarray, lookahead: float) -> tuple[np.ndarray, float]:
        """Prune passed points; return the path point at ``lookahead`` and path remaining."""
        d = [float(np.linalg.norm(p - here)) for p in self.path]
        k = int(np.argmin(d))
        self.path = self.path[k:]
        remaining = d[k] + sum(
            hypot(*(b - a)) for a, b in zip(self.path, self.path[1:], strict=False)
        )
        for p in self.path:
            if np.linalg.norm(p - here) >= lookahead:
                return p, remaining
        return self.path[-1], remaining

    def _collides(self, pose, v, w, clearance, min_clearance) -> bool:
        """The projected arc enters the clearance band *and* gets closer than the robot is
        now. Starting inside the band (after stopping near furniture at a viewpoint) must
        not block motion that leads away from the obstacle."""
        x, y, yaw = pose
        now = float(clearance(np.array([x, y])))
        steps = max(1, int(self.collision_time / 0.05))
        for _ in range(steps):
            x, y, yaw = x + v * cos(yaw) * 0.05, y + v * sin(yaw) * 0.05, yaw + w * 0.05
            c = float(clearance(np.array([x, y])))
            if c < min_clearance and c < now - 0.01:
                return True
        return False

    def update(
        self,
        pose: np.ndarray,
        blocked: bool = False,
        clearance: Callable[[np.ndarray], float] | None = None,
        min_clearance: float | None = None,
    ) -> tuple[float, float, str]:
        """``clearance(xy)``: distance to the nearest mapped obstacle; it regulates speed and,
        with ``min_clearance``, vetoes projected arcs that would come closer than that."""
        if not self.path:
            return (*self._smoothed(0.0, 0.0), "idle")
        if blocked:
            self.v = self.w = 0.0  # a safety stop is immediate, not smoothed
            return 0.0, 0.0, "blocked"
        x, y, yaw = pose
        here = np.array([x, y])
        goal = self.path[-1]
        dist_goal = float(np.linalg.norm(goal - here))
        if dist_goal < self.goal_tolerance:
            if self.final_heading is None or abs(wrap(self.final_heading - yaw)) < (
                self.heading_tolerance
            ):
                self.path = []
                return (*self._smoothed(0.0, 0.0), "arrived")
            err = wrap(self.final_heading - yaw)
            return (*self._smoothed(0.0, float(np.sign(err)) * self.rotate_speed), "aligning")
        lookahead = float(np.clip(abs(self.v) * self.lookahead_time, self.min_lookahead,
                                  self.max_lookahead))  # fmt: skip
        carrot, remaining = self._carrot(here, lookahead)
        dx, dy = carrot - here
        lx, ly = cos(yaw) * dx + sin(yaw) * dy, -sin(yaw) * dx + cos(yaw) * dy
        angle = atan2(ly, lx)
        if abs(angle) > self.rotate_min_angle:
            return (*self._smoothed(0.0, float(np.sign(angle)) * self.rotate_speed), "turning")
        d2 = lx * lx + ly * ly
        curvature = 2.0 * ly / d2 if d2 > 1e-9 else 0.0
        v = self.desired_speed
        if abs(curvature) > 1.0 / self.min_radius:  # regulate on curvature
            v *= 1.0 / (self.min_radius * abs(curvature))
        if clearance is not None:  # regulate near obstacles (Nav2: by costmap cost)
            c = float(clearance(here))
            if c < self.clearance_scaling_dist:
                v *= c / self.clearance_scaling_dist
        v = max(v, self.min_speed)
        if remaining < self.approach_dist:  # approach scaling
            v = min(v, max(self.min_approach_speed, v * remaining / self.approach_dist))
        w = v * curvature
        if abs(w) > self.max_turn:  # keep the arc, slow down along it
            v *= self.max_turn / abs(w)
            w = float(np.sign(w)) * self.max_turn
        if (
            clearance is not None
            and min_clearance is not None
            and self._collides(pose, v, w, clearance, min_clearance)
        ):
            self.v = self.w = 0.0
            return 0.0, 0.0, "blocked"
        return (*self._smoothed(v, w), "walking")
