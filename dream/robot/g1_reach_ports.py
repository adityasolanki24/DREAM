"""The G1 behind the ``ReachBody`` port of the reach skill (dream/tools/reach.py).

Wires the pieces that already exist: the head camera and the occupancy map (``G1MobileEnv``,
``SpatialMemory``), the navigator (``MobileToolbox.go_to``), the posture planner, the rollout
gate on a planning copy, the Decoupled-WBC backend under the safety supervisor, and the
closed-loop correction. It reads only the robot's own sensors and the perceived obstacles it is
handed; it never touches the scene (the ``env`` object is the physics owner, used through its
sensor methods).

Gaze: the head camera rides on the torso, so torso pitch tilts it. A gaze is a posture request
(nominal height, arms down, torso pitch) executed through the same backend and supervisor, and
its range is the one MEASURED for looking (``ControllerCapabilities.gaze_pitch``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from dream.robot import g1_wbc
from dream.robot.g1_correction import CORRECTION_TOLERANCE, correct_reach
from dream.robot.g1_posture import PerceivedBox, PosturePlanner
from dream.robot.g1_rollout import Procedure, verify_by_rollout
from dream.robot.g1_supervisor import SafetyLimits, WholeBodySupervisor
from dream.robot.wholebody import StopEvent, WBCBackend, WholeBodyTarget
from dream.tools.reach import PRE_CONTACT_TOLERANCE, Sight

FOOT_FORCE_MIN = 1.0  # N of normal force for a foot to count as in contact
TOUCH_THRESHOLD = 0.1  # N on the hand's tactile sensors that counts as contact (== g1_mobile's)
TICK = 0.02  # s, one policy step
ALIGN_STEPS = 60  # closed-loop alignment steps of ALIGN_DT
ALIGN_DT = 0.2  # s
ALIGN_POS, ALIGN_YAW, ALIGN_GAIN = 0.03, 0.05, 0.8
ALIGN_MIN = (0.08, 0.08, 0.12)  # m/s, m/s, rad/s: smallest command that makes the policy step


def foot_contacts(env) -> tuple[bool, bool]:
    """Per-foot load from the feet's own contact forces (the robot's foot force sensing)."""
    m, d = env.model, env.data
    floor = m.geom("floor").id
    feet = (m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id)
    force = [0.0, 0.0]
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor else c.geom1
        b = int(m.geom_bodyid[other])
        if b in feet:
            mujoco.mj_contactForce(m, d, i, f6)
            force[feet.index(b)] += float(f6[0])
    return force[0] > FOOT_FORCE_MIN, force[1] > FOOT_FORCE_MIN


def perceived(boxes) -> list[PerceivedBox]:
    return [PerceivedBox(b.center, b.half_extents, b.yaw, bool(b.hand_may_touch)) for b in boxes]


class G1ReachBody:
    """``ReachBody`` for the walking G1. ``toolbox`` is a ``MobileToolbox`` over a
    ``G1MobileBase`` on ``env``; ``controller_factory`` builds the controller of a planning
    copy."""

    def __init__(
        self,
        env,
        controller,
        toolbox,
        planner: PosturePlanner,
        controller_factory=g1_wbc.WBCLocomotion.from_cache,
        rng: np.random.Generator | None = None,
        safety: SafetyLimits | None = None,
    ):
        self._env_cls = type(env)  # planning copies use the same physics owner class
        self.env, self.controller, self.tb, self.planner = env, controller, toolbox, planner
        self.caps = planner.caps
        self.rng = rng or np.random.default_rng(0)
        self.backend = WBCBackend(
            env, controller, self.caps, feet_contact=lambda: foot_contacts(env)
        )
        self.safety = safety
        self.supervisors: list[WholeBodySupervisor] = []  # every supervisor used, for the record
        self.supervisor = self._new_supervisor(None)
        self.controller_factory = controller_factory
        self.nominal = self.backend.command()
        self._gaze = 0.0
        self.last_rollouts: list = []

    def _new_supervisor(self, goal) -> WholeBodySupervisor:
        sup = WholeBodySupervisor(self.planner.pm, self.safety, goal)
        self.supervisors.append(sup)
        return sup

    # -- sensing -------------------------------------------------------------------------
    def pose(self) -> np.ndarray:
        return self.env.odometry()

    def now(self) -> float:
        return float(self.env.time)

    def gaze_range(self) -> tuple[float, float]:
        return self.caps.gaze_pitch

    def stopped(self) -> str | None:
        if self.controller.stopped:
            return str(self.controller.stopped)
        return None if self.backend.held is None else self.backend.held.reason

    def look(self, gaze: float | None) -> Sight:
        """One head-camera frame. ``gaze``: torso pitch to look with (0.0 nominal); None keeps
        the present posture (used while reaching)."""
        if gaze is not None and abs(gaze - self._gaze) > 1e-6 and self.backend.held is None:
            self._set_gaze(gaze)
        frame = self.env.capture_head()
        dets = self.tb.memory.update(frame)
        return Sight(dets, frame.calibration.position.copy(), float(frame.time), gaze)

    def _set_gaze(self, pitch: float) -> None:
        nominal = self.nominal
        q = self.planner.pm.q_nominal.copy()
        target = WholeBodyTarget(
            q, (0.0, 0.0, 0.0), (), float(nominal.height), (0.0, float(pitch), 0.0),
            np.zeros(14),
        )  # fmt: skip
        report = self.backend.execute(
            target, monitor=self.supervisor, capabilities=self.caps.gaze()
        )
        if report.completed:
            self._gaze = pitch

    # -- motion --------------------------------------------------------------------------
    def turn(self, angle: float) -> bool:
        base = self.tb.base
        ok = base.walk(0.0, 0.0, math.copysign(0.5, angle), abs(angle) / 0.5)
        return bool(ok and base.walk(0, 0, 0, 0.3))

    def go_to(self, x: float, y: float, heading: float) -> tuple[bool, str]:
        """The navigator walks to the stance over the observed map (it stops within its goal
        tolerance, 0.15 m); a short closed-loop alignment on odometry then places the feet
        where the posture was planned."""
        r = self.tb.call("go_to", {"x": float(x), "y": float(y), "heading": float(heading)})
        if r.status != "succeeded":
            return False, f"{r.status} {r.message}".strip()
        aligned = self._align(x, y, heading)
        # Alignment is best effort: the posture is re-planned from wherever the feet ended up.
        return not self.stopped(), "aligned" if aligned else "arrived; alignment incomplete"

    def _align(self, x: float, y: float, heading: float) -> bool:
        base = self.tb.base
        for _ in range(ALIGN_STEPS):
            px, py, yaw = self.pose()
            dx, dy = x - px, y - py
            ef = math.cos(yaw) * dx + math.sin(yaw) * dy
            el = -math.sin(yaw) * dx + math.cos(yaw) * dy
            eyaw = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
            if math.hypot(dx, dy) < ALIGN_POS and abs(eyaw) < ALIGN_YAW:
                return bool(base.walk(0, 0, 0, 0.5))
            vmax = g1_wbc.MAX_COMMAND
            cmd = np.array([ALIGN_GAIN * ef, ALIGN_GAIN * el, 1.5 * eyaw])
            # The policy only steps when the command norm is at least 0.05: below that it
            # stands. Keep the larger needed axis above the floor so the correction happens.
            for k, (e, tol) in enumerate(((ef, ALIGN_POS), (el, ALIGN_POS), (eyaw, ALIGN_YAW))):
                if abs(e) > tol and abs(cmd[k]) < ALIGN_MIN[k]:
                    cmd[k] = math.copysign(ALIGN_MIN[k], e)
                elif abs(e) <= tol:
                    cmd[k] = 0.0
            cmd = np.clip(cmd, -0.8 * vmax, 0.8 * vmax)
            if not base.walk(float(cmd[0]), float(cmd[1]), float(cmd[2]), ALIGN_DT):
                return False
        base.walk(0, 0, 0, 0.5)
        return math.hypot(*(np.array([x, y]) - self.pose()[:2])) < 2 * ALIGN_POS

    # -- planning and verification -------------------------------------------------------
    def plan(self, target, covariance, obstacles, stance=None, arm=None, approach=None):
        return self.planner.plan(
            target, covariance, tuple(self.pose()), perceived(obstacles),
            stances=None if stance is None else [tuple(float(v) for v in stance)],
            approach_axis=approach, arms=None if arm is None else (arm,),
        )  # fmt: skip

    def pre_contact(self, plan, point, obstacles, approach=None):
        """``plan``'s posture with the hand at ``point`` (a pre-contact pose), or None."""
        return self.planner.retarget(plan, point, perceived(obstacles), approach_axis=approach)

    def verify(self, plans, obstacles, pre=None, approach=None):
        """Closed-loop rollouts of ``plans`` on planning copies (see dream/robot/g1_rollout.py).
        With ``pre`` (a pre-contact plan of the same body) the copy runs the real procedure:
        execute ``pre``, correct it in free air, advance to the plan, correct; without it, the
        plan is executed and corrected. Returns (first passing plan or None, [(plan, result)])."""
        caps, boxes = self.caps, perceived(obstacles)

        def factory(model, stance):
            env = self._env_cls(model, np.random.default_rng(0), head_camera=None)
            env.reset(*stance)
            env.localise(stance)  # odometry frame = plan frame, as on the real robot
            return WBCBackend(env, self.controller_factory(), caps)

        chosen, tried = None, []
        for plan in plans:
            first = pre or plan
            stages = [(first, first.target.hands[0].position, PRE_CONTACT_TOLERANCE)]
            if pre is not None:
                stages.append((plan, plan.target.hands[0].position, CORRECTION_TOLERANCE))
            else:
                stages[0] = (plan, plan.target.hands[0].position, CORRECTION_TOLERANCE)
            procedure = Procedure(self.planner, stages, approach)
            r = verify_by_rollout(
                first.target, factory, boxes, self.planner.pm, self.safety, procedure, plan
            )
            tried.append((plan, r))
            if r.passed:
                chosen = plan
                break
        self.last_rollouts = tried
        return chosen, tried

    # -- execution -----------------------------------------------------------------------
    def execute(self, plan):
        goal = plan.target.hands[0]
        self.supervisor = self._new_supervisor((goal.arm, goal.position))
        return self.backend.execute(plan.target, monitor=self.supervisor)

    def correct(
        self, plan, target_provider, obstacles, executed=None, tolerance=None, approach=None
    ):
        kw = {} if tolerance is None else {"tolerance": tolerance}
        return correct_reach(
            self.backend, self.supervisor, self.planner, plan, target_provider,
            perceived(obstacles), executed=executed, approach_axis=approach, **kw,
        )  # fmt: skip

    def hand_position(self, arm: str) -> np.ndarray:
        row = self.supervisor.measure(self.backend.observe())
        return np.asarray(row[f"ach_hand_{arm}"], float)

    def touch(self, arm: str, seconds: float) -> tuple[float, float]:
        """Hold the present command still for ``seconds`` while reading the hand's own tactile
        sensors (simulated Dex3 touch sensors, ``G1MobileEnv.touch``). Returns the longest
        unbroken contact in seconds and the peak total force in N. The supervisor keeps
        monitoring: a stop ends the hold (and is held, as everywhere)."""
        c, cmd = self.controller, self.backend.command()
        run = best = 0
        peak = 0.0
        for _ in range(round(seconds / TICK)):
            c.set_command(0.0, 0.0, 0.0)
            c.set_posture(cmd.height, cmd.rpy)
            c.arm_target = np.array(cmd.arm, float)
            self.env.step(c)
            force = float(sum(self.env.touch(arm).values()))
            run = run + 1 if force >= TOUCH_THRESHOLD else 0
            best, peak = max(best, run), max(peak, force)
            event = self.supervisor(self.backend.observe())
            if event is not None:
                self.backend.hold(event)
                break
        return best * TICK, peak

    def recover(self) -> bool:
        """Return to the nominal posture. After a protective stop: the supervisor's validated
        recovery (reverses the recorded trajectory). Otherwise a monitored return to the
        nominal command, allowed only in a valid support state."""
        if self.controller.stopped:
            return False
        sup = self.supervisor
        now = self.backend.command()
        if self.backend.held is None and self._is_nominal(now):
            return True
        if self.backend.held is not None:
            result = sup.recover(self.backend)
            self._gaze = 0.0 if result.ok else self._gaze
            return bool(result.ok)
        ok, _ = sup.support_state(self.backend.observe())
        if not ok:
            return False
        sup.reset()
        report = self.backend.follow([self.nominal], monitor=sup, rate_scale=0.5)
        if report.completed:
            self._gaze = 0.0
        return bool(report.completed)

    def _is_nominal(self, c) -> bool:
        n = self.nominal
        return bool(
            abs(c.height - n.height) < 1e-3
            and np.allclose(c.rpy, n.rpy, atol=1e-3)
            and np.allclose(c.arm, n.arm, atol=1e-3)
        )

    def held_stop(self) -> StopEvent | None:
        return self.backend.held
