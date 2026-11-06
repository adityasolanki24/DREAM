"""Navigation contracts: estimated local odometry, never simulator world coordinates.

NavGPT frame/scale references are pinned in experiments/configs/navigation_sources.json.
The Habitat TELEPORT command is intentionally not part of this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


def wrap(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


@dataclass(frozen=True)
class NavigationRequest:
    instruction: str
    observation_time_s: float
    authority: int


@dataclass(frozen=True)
class LocalWaypoint:
    forward_m: float
    left_m: float
    heading_rad: float
    observation_time_s: float
    authority: int


@dataclass(frozen=True)
class NavigationState:
    time_s: float
    # Dead-reckoned body velocity/gyro or a VIO estimate; initial origin arbitrary.
    odom_x_m: float
    odom_y_m: float
    odom_yaw_rad: float
    tilt_rad: float
    clearance_m: float  # minimum valid front-sector depth; unknown must be NaN
    source: str = "proprioceptive_dead_reckoning"
    depth_time_s: float = 0.0


@dataclass(frozen=True)
class NavigationOutcome:
    forward_m_s: float
    yaw_rad_s: float
    status: str
    remaining_m: float


def decode_waypoints(raw: np.ndarray, request: NavigationRequest, *, normalized: bool) -> tuple:
    """Eight cumulative targets, not increments. Scaling is explicit and applied once.

    Pinned NavGPTVLA.act already returns metres/radians; predict_actions returns normalized
    values. No waypoint includes a time duration or a humanoid joint target.
    """
    a = np.array(raw, dtype=float, copy=True)
    if a.shape != (8, 3) or not np.isfinite(a).all():
        raise ValueError("expected eight finite navigation waypoints")
    if not request.instruction.strip() or not np.isfinite(request.observation_time_s):
        raise ValueError("invalid navigation request")
    if normalized:
        a *= [1.0, 0.433, 2 * pi / 3]
    # Prototype authority envelope, independent of any evaluation outcome. Reject, never clip.
    if np.any(np.abs(a[:, :2]) > 1.5) or np.any(np.abs(a[:, 2]) > 2 * pi / 3 + 1e-6):
        raise ValueError("navigation target outside bounded local envelope")
    return tuple(LocalWaypoint(*row, request.observation_time_s, request.authority) for row in a)


class WaypointTracker:
    """Turn/forward tracker; no obstacle route planning or semantic arrival claim.

    Conservative prototype envelope: 0.2 m/s, 0.4 rad/s, 0.1 m positional and 0.1 rad
    heading tolerance, 0.4 m clearance, 25-degree tilt trip, 8 s waypoint lifetime.
    These are engineering guards, not calibrated humanoid performance limits.
    """

    def __init__(self) -> None:
        self.authority = 0
        self.target = None
        self._last_time = None

    def cancel(self) -> None:
        self.authority += 1
        self.target = None

    def submit(self, waypoint: LocalWaypoint, observed: NavigationState) -> None:
        self.target = None
        values = [waypoint.forward_m, waypoint.left_m, waypoint.heading_rad]
        if (
            not np.isfinite(values).all()
            or max(abs(v) for v in values[:2]) > 1.5
            or abs(waypoint.heading_rad) > 2 * pi / 3 + 1e-6
            or waypoint.authority != self.authority
            or not 0 <= observed.time_s - waypoint.observation_time_s <= 2
        ):
            raise ValueError("invalid, revoked or stale waypoint")
        self._validate_state(observed)
        c, s = cos(observed.odom_yaw_rad), sin(observed.odom_yaw_rad)
        self.target = (
            observed.odom_x_m + c * waypoint.forward_m - s * waypoint.left_m,
            observed.odom_y_m + s * waypoint.forward_m + c * waypoint.left_m,
            wrap(observed.odom_yaw_rad + waypoint.heading_rad),
            observed.time_s + 8,
        )
        self._last_time = observed.time_s

    @staticmethod
    def _validate_state(state: NavigationState) -> None:
        if (
            state.source not in ("proprioceptive_dead_reckoning", "visual_inertial_odometry")
            or not np.isfinite(
                [
                    state.time_s,
                    state.odom_x_m,
                    state.odom_y_m,
                    state.odom_yaw_rad,
                    state.tilt_rad,
                    state.clearance_m,
                ]
            ).all()
            or state.clearance_m < 0
            or state.tilt_rad < 0
            or not np.isfinite(state.depth_time_s)
            or not 0 <= state.time_s - state.depth_time_s <= 0.25
        ):
            raise ValueError("navigation needs finite estimated state and valid depth")

    def update(self, state: NavigationState) -> NavigationOutcome:
        try:
            self._validate_state(state)
        except ValueError:
            self.cancel()
            return NavigationOutcome(0, 0, "invalid observation", float("nan"))
        if self.target is None:
            return NavigationOutcome(0, 0, "idle", 0)
        x, y, yaw, expiry = self.target
        distance = hypot(x - state.odom_x_m, y - state.odom_y_m)
        if (
            state.time_s < self._last_time
            or state.time_s > expiry
            or state.tilt_rad > 25 * pi / 180
            or state.clearance_m < 0.4
        ):
            self.cancel()
            return NavigationOutcome(0, 0, "safety stop or timeout", distance)
        self._last_time = state.time_s
        error = wrap(
            (atan2(y - state.odom_y_m, x - state.odom_x_m) if distance > 0.1 else yaw)
            - state.odom_yaw_rad
        )
        if distance <= 0.1 and abs(error) <= 0.1:
            self.target = None
            return NavigationOutcome(0, 0, "waypoint reached", distance)
        forward = min(0.2, distance) if abs(error) < 0.2 and distance > 0.1 else 0.0
        return NavigationOutcome(forward, float(np.clip(error, -0.4, 0.4)), "tracking", distance)
