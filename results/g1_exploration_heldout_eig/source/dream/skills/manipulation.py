"""Reusable closed-loop manipulation skills over the ``Embodiment`` and ``Perception`` protocols.

Skills never touch the simulator. They command TCP and gripper setpoints, wait with timeouts,
and judge success from measured signals: finger opening (proprioception) and tracked object
poses (perception). A skill returns a ``SkillResult`` and never raises on physical failure, so
callers can replan.

The grasp is a top-down parallel-jaw grasp aligned with the cube's observed yaw. Holding is
inferred from the measured finger opening once the fingers stop. A fully closed gripper means a
miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dream.types import Embodiment, ObjectObservation, Perception

TRAVEL_HEIGHT = 0.18  # m, TCP height for transit above objects and fixtures
APPROACH_ABOVE = 0.08  # m above the grasp/place point before the final descent
POS_TOL = 0.004  # m
ROT_TOL = np.deg2rad(3.0)
OPEN_WIDTH = 0.08  # m, fully open
PRE_SQUEEZE = 0.008  # m below the expected width for the gentle first closing stage
GRASP_SLACK = 0.03  # m of total opening beyond the object's width before grasping
RELIABLE_STD = 0.006  # m: a held object's estimate above this is not trusted for placement
RELEASE_SLACK = 0.012  # m beyond the held width when releasing (wide openings hit walls/neighbours)


@dataclass
class SkillResult:
    ok: bool
    reason: str = ""
    sim_time: float = 0.0  # s spent in this skill
    details: dict = field(default_factory=dict)


def _down(yaw: float) -> np.ndarray:
    """TCP pointing at the table, closing direction rotated by ``yaw`` (quaternion w, x, y, z)."""
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def grasp_yaw(object_yaw: float, current_yaw: float = 0.0) -> float:
    """Grasp yaw closing across one face pair, folded to be closest to ``current_yaw``.

    The parallel gripper is 2-fold symmetric, so only multiples of pi are equivalent. The other
    face pair is ``object_yaw + pi/2`` (see ``choose_grasp_yaw``).
    """
    candidates = object_yaw + np.arange(-3, 4) * np.pi
    candidates = candidates[np.abs(candidates) <= np.pi * 0.75]  # stay inside the wrist range
    return float(candidates[np.argmin(np.abs(candidates - current_yaw))])


# m the open fingers' outer faces extend beyond the object along the closing axis:
# half the grasp slack plus the finger thickness (~12 mm).
FINGER_REACH = GRASP_SLACK / 2 + 0.012
WALL_BAND = 0.012  # m: fixture walls may extend this far outside the visible inner edge


def finger_clearance(obj: ObjectObservation, yaw: float, perception: Perception) -> float:
    """Free space (m) around both fingertips for a grasp at ``yaw``, from observations only."""
    close_dir = np.array([-np.sin(yaw), np.cos(yaw)])  # fingers close along this world axis
    tips = [obj.position[:2] + s * close_dir * (obj.size / 2 + FINGER_REACH) for s in (-1, 1)]
    gaps = []
    for tip in tips:
        for o in perception.observe_objects():
            if o.name != obj.name:
                gaps.append(np.linalg.norm(tip - o.position[:2]) - o.size / 2)
        for centre, half in perception.fixtures().values():
            cheb = np.max(np.abs(tip - centre[:2]))
            gaps.append(half - cheb if cheb < half else cheb - half - WALL_BAND)
    return float(min(gaps)) if gaps else np.inf


def choose_grasp_yaw(obj: ObjectObservation, current_yaw: float, perception: Perception) -> float:
    """Of the two face pairs of a cube, grasp across the one with more room for the fingers."""
    a = grasp_yaw(obj.yaw, current_yaw)
    b = grasp_yaw(obj.yaw + np.pi / 2, current_yaw)
    ca, cb = finger_clearance(obj, a, perception), finger_clearance(obj, b, perception)
    return a if ca >= cb - 0.005 else b  # prefer less wrist rotation unless clearly worse


def tcp_yaw(quat: np.ndarray) -> float:
    """Inverse of ``_down`` for a TCP that points down."""
    return float(2 * np.arctan2(quat[2], quat[1]))


def find(perception: Perception, name: str) -> ObjectObservation | None:
    return next((o for o in perception.observe_objects() if o.name == name), None)


def move_to(
    robot: Embodiment, pos: np.ndarray, yaw: float, timeout: float = 6.0, tol: float = POS_TOL
) -> SkillResult:
    """Bounded TCP motion with feedback: succeeds when tracking error is within tolerance."""
    t0 = robot.state().time
    goal = robot.move_tcp(np.asarray(pos, float), _down(yaw))
    if np.linalg.norm(goal - pos) > 1e-3:
        robot.hold()
        return SkillResult(
            False, "target outside safety workspace", 0.0, {"clamped_to": goal.tolist()}
        )

    def done() -> bool:
        pe, re = robot.tracking_error()
        return (pe < tol and re < ROT_TOL) or robot.state().protective_stop

    reached = robot.wait_until(done, timeout)
    st = robot.state()
    dt = st.time - t0
    if st.protective_stop:
        return SkillResult(False, f"protective stop: {st.stop_reason}", dt)
    if not reached:
        pe, _ = robot.tracking_error()
        return SkillResult(False, f"timeout ({pe * 1000:.0f} mm from target)", dt)
    return SkillResult(True, "", dt)


def set_gripper(robot: Embodiment, width: float, timeout: float = 1.5) -> SkillResult:
    """Command the fingers and wait until they stop moving (closing on an object stalls them)."""
    t0 = robot.state().time
    robot.set_gripper(width)
    last = [robot.state().gripper_width]

    def settled() -> bool:
        w = robot.state().gripper_width
        still = abs(w - last[0]) < 2e-5
        last[0] = w
        return still and (abs(w - width) < 2e-3 or width < 0.01)

    robot.wait_until(settled, timeout)
    return SkillResult(True, "", robot.state().time - t0, {"width": robot.state().gripper_width})


def settle(robot: Embodiment, timeout: float = 1.0, max_joint_speed: float = 0.02) -> bool:
    """Wait until the arm is (nearly) still.

    Readings combined from different sensors, such as an object pose and the TCP pose, are only
    consistent when the arm is not moving. With a live physics thread, time passes between the
    two reads.
    """
    return robot.wait_until(
        lambda: float(np.max(np.abs(robot.state().joint_vel))) < max_joint_speed, timeout
    )


def close_on(robot: Embodiment, expected_width: float) -> float:
    """Two-stage close: gentle squeeze to just under the expected width, then full grip.

    Closing straight to full force lets the first finger to touch shove the object against the
    table before the other finger arrives. With the stiff gripper this produced a 40 N
    one-sided load and a protective stop. Returns the measured finger opening.
    """
    set_gripper(robot, max(0.0, expected_width - PRE_SQUEEZE))
    return set_gripper(robot, 0.0).details["width"]


def move_via_travel_height(robot: Embodiment, pos: np.ndarray, yaw: float) -> SkillResult:
    """Rise, translate at travel height, then descend: avoids sweeping through other objects."""
    st = robot.state()
    total = 0.0
    for wp, y in (
        (np.array([*st.tcp_pos[:2], max(st.tcp_pos[2], TRAVEL_HEIGHT)]), tcp_yaw(st.tcp_quat)),
        (np.array([pos[0], pos[1], max(pos[2], TRAVEL_HEIGHT)]), yaw),
        (np.asarray(pos, float), yaw),
    ):
        r = move_to(robot, wp, y)
        total += r.sim_time
        if not r.ok:
            r.sim_time = total
            return r
    return SkillResult(True, "", total)


def pick(robot: Embodiment, perception: Perception, name: str) -> SkillResult:
    """Grasp object ``name`` from above and lift it. Verifies the lift by observation."""
    obj = find(perception, name)
    if obj is None:
        return SkillResult(False, f"object '{name}' not observed")
    t0 = robot.state().time
    yaw = choose_grasp_yaw(obj, tcp_yaw(robot.state().tcp_quat), perception)
    set_gripper(robot, min(OPEN_WIDTH, obj.size + GRASP_SLACK))
    above = obj.position + [0, 0, APPROACH_ABOVE]
    r = move_via_travel_height(robot, above, yaw)
    if not r.ok:
        return _done(r, robot, t0)
    obj = find(perception, name)  # re-observe before committing: the object may have moved
    if obj is None:
        return _done(SkillResult(False, f"object '{name}' lost before grasp"), robot, t0)
    yaw = choose_grasp_yaw(obj, yaw, perception)
    grasp_z = max(obj.position[2], 0.012)
    for wp, tol in (
        (obj.position + [0, 0, APPROACH_ABOVE], POS_TOL),
        ([*obj.position[:2], grasp_z], POS_TOL),
    ):
        r = move_to(robot, np.asarray(wp, float), yaw, tol=tol)
        if not r.ok:
            return _done(r, robot, t0)
    width = close_on(robot, obj.size)
    if width < 0.5 * obj.size:
        set_gripper(robot, OPEN_WIDTH)
        return _done(
            SkillResult(
                False, "grasp missed (fingers closed on nothing)", details={"width": width}
            ),
            robot,
            t0,
        )
    r = move_to(robot, np.array([*obj.position[:2], TRAVEL_HEIGHT]), yaw)
    if not r.ok:
        return _done(r, robot, t0)
    lifted = find(perception, name)
    width = robot.state().gripper_width
    # A camera may not see a lifted object behind the hand. Then the evidence is the grip: the
    # fingers stopped at about the object's width. A visible object still at its old height
    # means it was not lifted.
    hidden = lifted is None or not lifted.visible
    low = not hidden and lifted.position[2] < obj.position[2] + 0.05
    if low or width < 0.5 * obj.size:
        z = None if lifted is None else float(lifted.position[2])
        return _done(
            SkillResult(
                False,
                "object not lifted (slipped or dropped)",
                details={"object_z": z, "width": width},
            ),
            robot,
            t0,
        )
    details = {"width": width, "verified_by": "grip width (object hidden)" if hidden else "view"}
    if not hidden:
        details["object_z"] = float(lifted.position[2])
    return _done(SkillResult(True, "", details=details), robot, t0)


def place(
    robot: Embodiment,
    perception: Perception,
    name: str,
    target: np.ndarray,
    release_clearance: float = 0.01,
    yaw: float | None = None,
) -> SkillResult:
    """Lower the held object so its centre reaches ``target`` (+clearance), release, retreat.

    Success means the object was released near the target and stayed there. Whether the *task*
    goal holds is for a separate verifier to decide. ``yaw`` turns the hand (and the held
    object with it) on the way, e.g. to keep the wide hand clear of a container's walls.
    """
    t0 = robot.state().time
    settle(robot)  # the hand-object offset below needs both readings from a still arm
    held = find(perception, name)
    if held is None:
        return SkillResult(False, f"object '{name}' not observed")
    if robot.state().gripper_width < 0.5 * held.size:
        return SkillResult(False, "precondition failed: not holding an object")
    # The TCP-to-object offset is measured, not assumed: the grasp may not be centred. A camera
    # sees a held object partly hidden by the hand, and an estimate from its visible faces can be
    # off by half its size; then the grasp point is used (pick aimed the fingers at the centre).
    st = robot.state()
    reliable = held.visible and held.position_std <= RELIABLE_STD
    offset = held.position - st.tcp_pos if reliable else np.zeros(3)
    if yaw is None:
        yaw = tcp_yaw(st.tcp_quat)
    else:  # the held object turns with the hand, so its offset from the TCP turns too
        dyaw = yaw - tcp_yaw(st.tcp_quat)
        cz, sz = np.cos(dyaw), np.sin(dyaw)
        offset = np.array(
            [cz * offset[0] - sz * offset[1], sz * offset[0] + cz * offset[1], offset[2]]
        )
    tcp_goal = np.asarray(target, float) - offset + [0, 0, release_clearance]
    r = move_via_travel_height(robot, tcp_goal + [0, 0, APPROACH_ABOVE], yaw)
    if r.ok:
        r = move_to(robot, tcp_goal, yaw)
    if not r.ok:
        return _done(r, robot, t0)
    set_gripper(robot, min(OPEN_WIDTH, robot.state().gripper_width + RELEASE_SLACK))
    up = robot.state().tcp_pos + [0, 0, APPROACH_ABOVE]
    r = move_to(robot, up, yaw)
    if not r.ok:
        return _done(SkillResult(False, f"released, but retreat failed: {r.reason}"), robot, t0)
    final = find(perception, name)
    err = None if final is None else float(np.linalg.norm(final.position - target))
    return _done(SkillResult(True, "", details={"final_error": err}), robot, t0)


def _done(r: SkillResult, robot: Embodiment, t0: float) -> SkillResult:
    r.sim_time = robot.state().time - t0
    return r
