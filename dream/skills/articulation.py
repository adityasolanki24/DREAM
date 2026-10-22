"""Skills for operating articulated objects through a handle: grasp, probe, pull, release.

All motion with the handle in hand runs in the embodiment's compliant mode. The hand acts like
a force-capped spring toward a moving target and yields wherever the mechanism pushes back. The
mechanism therefore guides the motion, forces stay bounded, and a fixed object costs nothing
worse than a capped push. No skill knows or assumes what kind of joint is behind the handle.

``probe`` returns the raw measurements a discovery model needs. These are handle pose before and
after (from the handle tracker), TCP pose, measured external force, and finger width. Slip is the
change in the hand-handle offset, which is measured rather than assumed to be zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dream.skills.manipulation import (
    PRE_SQUEEZE,
    SkillResult,
    _done,
    move_to,
    move_via_travel_height,
    set_gripper,
    settle,
    tcp_yaw,
)
from dream.types import ArticulatedObservation, CompliantMode, Embodiment, Perception

HANDLE_OPENING = 0.037  # m: handle thickness + ~6 mm per side; the rear finger must fit the gap
APPROACH_ABOVE = 0.10  # m
TRACKER_FRAMES = 10  # averaged per snapshot: a still handle, so noise drops by ~sqrt(10)
PROBE_STIFFNESS = 300.0  # N/m


@dataclass
class ProbeOutcome:
    """Everything measured during one probe (all from robot sensors and the handle tracker)."""

    direction: np.ndarray
    distance: float
    force_limit: float
    handle_before: np.ndarray
    handle_after: np.ndarray
    handle_quat_before: np.ndarray
    handle_quat_after: np.ndarray
    tcp_before: np.ndarray
    tcp_after: np.ndarray
    force_after: np.ndarray  # N, measured external force on the hand at the end
    width_before: float
    width_after: float
    slip: float  # m, change in the hand-handle offset during the probe
    ok: bool = True
    reason: str = ""
    extra: dict = field(default_factory=dict)


def find_articulated(perception: Perception, name: str) -> ArticulatedObservation | None:
    return next((a for a in perception.observe_articulated() if a.name == name), None)


def handle_closing_yaw(obs: ArticulatedObservation, current_yaw: float) -> float:
    """Wrist yaw that closes the fingers across the handle's thickness (its local x axis)."""
    w, x, y, z = obs.handle_quat
    local_x = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    # _down(yaw) closes the fingers along (-sin yaw, cos yaw).
    yaw = float(np.arctan2(-local_x[0], local_x[1]))
    candidates = yaw + np.arange(-3, 4) * np.pi
    candidates = candidates[np.abs(candidates) <= np.pi * 0.75]
    return float(candidates[np.argmin(np.abs(candidates - current_yaw))])


def grasp_handle(
    robot: Embodiment, perception: Perception, name: str, handle_size: float = 0.025
) -> SkillResult:
    """Top-down grasp across the handle. Verified by finger width (closed on the handle)."""
    obs = find_articulated(perception, name)
    if obs is None:
        return SkillResult(False, f"no handle observed on '{name}'")
    t0 = robot.state().time
    robot.set_compliance(None)
    yaw = handle_closing_yaw(obs, tcp_yaw(robot.state().tcp_quat))
    set_gripper(robot, HANDLE_OPENING)
    r = move_via_travel_height(robot, obs.handle_pos + [0, 0, APPROACH_ABOVE], yaw)
    if not r.ok:
        return _done(r, robot, t0)
    obs = find_articulated(perception, name)  # re-observe before committing
    r = move_to(robot, obs.handle_pos, yaw)
    if not r.ok:
        return _done(r, robot, t0)
    set_gripper(robot, max(0.0, handle_size - PRE_SQUEEZE))
    width = set_gripper(robot, 0.0).details["width"]
    if not (0.6 * handle_size < width < handle_size + 0.01):
        set_gripper(robot, HANDLE_OPENING)
        return _done(
            SkillResult(False, f"handle grasp failed (finger width {width * 1000:.0f} mm)"),
            robot,
            t0,
        )
    return _done(SkillResult(True, "", details={"width": width, "yaw": yaw}), robot, t0)


def observe_handle_averaged(
    perception: Perception, name: str, frames: int = TRACKER_FRAMES
) -> ArticulatedObservation:
    """Average several tracker frames of a still handle (single frames have ~2 mm noise)."""
    obs = [find_articulated(perception, name) for _ in range(frames)]
    q_ref = obs[0].handle_quat
    quats = np.array([o.handle_quat * np.sign(np.dot(o.handle_quat, q_ref)) for o in obs])
    q = quats.mean(axis=0)
    return ArticulatedObservation(
        name,
        np.mean([o.handle_pos for o in obs], axis=0),
        q / np.linalg.norm(q),
        obs[-1].box_centre,
        obs[-1].box_half,
    )


def _snapshot(robot: Embodiment, perception: Perception, name: str):
    settle(robot, timeout=1.0, max_joint_speed=0.01)
    return robot.state(), observe_handle_averaged(perception, name)


def probe(
    robot: Embodiment,
    perception: Perception,
    name: str,
    direction: np.ndarray,
    distance: float,
    force_limit: float,
    stiffness: float = PROBE_STIFFNESS,
    ramp: float = 1.0,
    hold: float = 1.5,
) -> ProbeOutcome:
    """Drag the gripped handle toward ``distance`` along ``direction`` with a force cap, and settle.

    The target ramps out over ``ramp`` seconds and is held for ``hold`` seconds, then the
    outcome is read after the arm has settled (a quasi-static measurement, as the model assumes).
    """
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    robot.set_compliance(CompliantMode(stiffness, force_limit))
    st0, obs0 = _snapshot(robot, perception, name)
    start = st0.tcp_pos.copy()
    quat0 = st0.tcp_quat
    n = max(1, int(ramp / 0.05))
    for i in range(1, n + 1):
        robot.move_tcp(start + direction * distance * i / n, quat0)
        robot.wait_until(lambda: False, 0.05)
    robot.wait_until(lambda: False, hold)
    st1, obs1 = _snapshot(robot, perception, name)
    # Relax: put the target where the hand is now, so it stops pulling but keeps holding.
    robot.move_tcp(st1.tcp_pos, st1.tcp_quat)
    robot.wait_until(lambda: False, 0.3)
    stopped = st1.protective_stop
    return ProbeOutcome(
        direction=direction,
        distance=float(distance),
        force_limit=float(force_limit),
        handle_before=obs0.handle_pos,
        handle_after=obs1.handle_pos,
        handle_quat_before=obs0.handle_quat,
        handle_quat_after=obs1.handle_quat,
        tcp_before=st0.tcp_pos,
        tcp_after=st1.tcp_pos,
        force_after=st1.external_force,
        width_before=st0.gripper_width,
        width_after=st1.gripper_width,
        slip=float(
            np.linalg.norm((obs1.handle_pos - st1.tcp_pos) - (obs0.handle_pos - st0.tcp_pos))
        ),
        ok=not stopped,
        reason=st1.stop_reason if stopped else "",
    )


def pull_along(
    robot: Embodiment,
    waypoints: list[np.ndarray],
    force_limit: float,
    stiffness: float = PROBE_STIFFNESS,
    seconds_per_waypoint: float = 0.15,
) -> SkillResult:
    """Compliantly drag the gripped handle through ``waypoints`` (target positions for the TCP)."""
    t0 = robot.state().time
    robot.set_compliance(CompliantMode(stiffness, force_limit))
    quat = robot.state().tcp_quat
    for wp in waypoints:
        robot.move_tcp(np.asarray(wp, float), quat)
        robot.wait_until(lambda: robot.state().protective_stop, seconds_per_waypoint)
        if robot.state().protective_stop:
            return _done(
                SkillResult(False, f"protective stop: {robot.state().stop_reason}"), robot, t0
            )
    robot.wait_until(lambda: False, 1.0)
    st = robot.state()
    robot.move_tcp(st.tcp_pos, st.tcp_quat)  # stop pulling, keep holding
    return _done(SkillResult(True, "", details={"tcp": st.tcp_pos.tolist()}), robot, t0)


def release_handle(robot: Embodiment, back_off: float = 0.08) -> SkillResult:
    """Open the fingers, leave compliant mode, and back off upward."""
    t0 = robot.state().time
    set_gripper(robot, HANDLE_OPENING)
    robot.set_compliance(None)
    st = robot.state()
    r = move_to(robot, st.tcp_pos + [0, 0, back_off], tcp_yaw(st.tcp_quat))
    return _done(r, robot, t0)
