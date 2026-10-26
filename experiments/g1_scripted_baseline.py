"""SCRIPTED BASELINE for the unfamiliar-height reach test (labelled scripted_baseline everywhere).

The current upright approach, kept as the comparison for the posture-planning system:
- the same instruction parsing, perception, search, navigation and alignment (shared code);
- a FIXED 0.42 m stand-off in front of the perceived handle, along the perceived face normal;
- pelvis height at the nominal 0.74 m, torso pitch 0, the RIGHT arm only;
- position-only damped-least-squares IK of one arm, the algorithm of
  ``dream.skills.g1_reach.RightArm.solve`` with its contact point moved to the fingertip reach
  point (otherwise this arm's hand could never touch a surface); ``attach()`` is not used;
- a REFUSAL for perceived handle heights outside 0.55-1.15 m, as experiments/isaac_task.py does
  ("no crouch": 0.55 m is the bottom of the band that robot reaches without bending, 1.15 m the
  top);
- no balance check, no collision check, no rollout; it runs under the same supervisor and the same
  stop rules as the planning system, and verifies by the same forward-kinematics rule.

It exists to show what the planning system adds, not to be a strong baseline.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from dream.robot.wholebody import PostureCommand
from dream.skills.g1_reach import RIGHT_ARM_SLICE, RightArm
from dream.tools.reach import (
    PRE_CONTACT_OFFSET,
    PRE_CONTACT_TOLERANCE,
    ReachReport,
    ReachSkill,
    claim_threshold,
    parse_reach_instruction,
)

SYSTEM = "scripted_baseline"
STAND_OFF = 0.42  # m, target to pelvis
HEIGHT_BAND = (0.55, 1.15)  # m, perceived handle height the upright reach attempts
ARM = "right"
ITERATIONS = 3
TOLERANCE = 0.01  # m


class FingertipArm(RightArm):
    """``RightArm.solve`` (damped least squares, position only) for the reach point on the hand."""

    def __init__(self, model, data, controller, tip_in_wrist: np.ndarray) -> None:
        super().__init__(model, data, controller)
        self.wrist = model.body("right_wrist_yaw_link").id
        self.tip = np.asarray(tip_in_wrist, float)

    def tip_position(self, d=None) -> np.ndarray:
        d = self.d if d is None else d
        return d.xpos[self.wrist] + d.xmat[self.wrist].reshape(3, 3) @ self.tip

    def solve(self, target: np.ndarray, iters: int = 60) -> np.ndarray:
        s = self.scratch
        s.qpos[:] = self.d.qpos
        q = s.qpos[self.qadr].copy()
        jac = np.zeros((3, self.m.nv))
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            point = self.tip_position(s)
            err = target - point
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jac(self.m, s, jac, None, point, self.wrist)
            j = jac[:, self.dadr]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q


class ScriptedBaselineReach(ReachSkill):
    """Same front end as ``ReachSkill``; the back end is the fixed upright right-arm reach.

    ``body`` is a ``G1ReachBody`` (it provides the physics owner, the supervisor and the
    alignment); the baseline never calls its planner or rollout."""

    system = SYSTEM

    def run(self, instruction: str) -> ReachReport:
        try:
            goal = parse_reach_instruction(instruction)
        except ValueError as e:
            return ReachReport("refused", f"unsupported instruction: {e}",
                               parse_reach_instruction("touch the red handle"))  # fmt: skip
        rep = ReachReport("failed", "", goal)
        est = self.estimator_factory(goal.color)
        self._log(rep, "goal", f"[{SYSTEM}] {goal.verb} the {goal.color} {goal.noun}")
        if not self._observe(est, rep, search=True):
            rep.reason = f"the {goal.color} handle was not found"
            return rep
        target = est.estimate()
        from dream.tools.reach import _describe

        rep.target = _describe(target)
        z = float(target.position[2])
        if not HEIGHT_BAND[0] <= z <= HEIGHT_BAND[1]:
            rep.verdict = "refused"
            rep.reason = (
                f"perceived height {z:.2f} m is outside {HEIGHT_BAND[0]}-{HEIGHT_BAND[1]} m: "
                "the upright reach does not crouch or stretch"
            )
            self._log(rep, "result", rep.reason)
            return rep
        body = self.body
        n = target.face_normal
        if n is None:  # no fixture seen: stand on the robot's side of the target
            v = np.asarray(body.pose()[:2]) - target.position[:2]
            n = tuple(v / max(float(np.linalg.norm(v)), 1e-6))
        stand = target.position[:2] + STAND_OFF * np.asarray(n)
        heading = math.atan2(-n[1], -n[0])
        ok, msg = body.go_to(float(stand[0]), float(stand[1]), heading)
        self._log(rep, "go_to", f"stand-off {stand.round(2).tolist()}: {'ok' if ok else msg}")
        if not ok or body.stopped():
            rep.reason = "could not walk to the stand-off point"
            return rep
        target = self._reobserve(est, rep)
        if target is None:
            rep.reason = "handle lost after walking to the stand-off point"
            return rep
        rep.target = _describe(target)
        rep.posture = {
            "stance": [round(float(v), 3) for v in body.pose()], "arm": ARM,
            "pelvis_height": round(float(body.nominal.height), 3), "torso_rpy": [0.0, 0.0, 0.0],
        }  # fmt: skip
        pm = body.planner.pm
        arm = FingertipArm(
            pm.model, pm.data_at(pm.q_nominal), body.controller, pm.anatomy.grip[ARM]
        )
        body.supervisor = body._new_supervisor((ARM, tuple(target.position)))
        normal = np.array([n[0], n[1], 0.0])
        latest = {"target": target}
        stages = (
            ("pre_contact", PRE_CONTACT_OFFSET, PRE_CONTACT_TOLERANCE),
            ("contact", 0.0, TOLERANCE),
        )
        rep.execution = {"completed": True, "stops": [], "correction": []}
        for name, offset, tol in stages:
            for i in range(ITERATIONS + 1):
                s = body.look(None)
                est.reset()
                if est.ingest(s.detections, s.camera, s.time):
                    latest["target"] = est.estimate()
                goal_pt = latest["target"].position + offset * normal
                obs = body.backend.observe()
                row = body.supervisor.measure(obs)
                hand = np.asarray(row[f"ach_hand_{ARM}"])
                err = float(np.linalg.norm(hand - goal_pt))
                rep.execution["correction"].append({"stage": name, "iteration": i, "error": err})
                if err <= tol or i == ITERATIONS:
                    break
                # IK in the robot's own frame: measured configuration (IMU, encoders, odometry).
                q0 = body.planner.measured_qpos(obs.prop, obs.pose, row["ach_height"])
                arm.d = pm.data_at(q0)
                q = arm.solve(goal_pt)
                cmd = body.backend.command()
                arm14 = cmd.arm.copy()  # delta on the command: cancels the PD sag by feedback
                arm14[RIGHT_ARM_SLICE] = cmd.arm[RIGHT_ARM_SLICE] + (
                    q - np.asarray(obs.prop.q)[22:29]
                )
                run = body.backend.follow(
                    [PostureCommand(cmd.height, cmd.rpy, arm14)], monitor=body.supervisor
                )
                if run.stops or not run.completed:
                    why = run.stops[0].reason if run.stops else "incomplete"
                    body.recover()
                    rep.verdict, rep.reason = "failed", f"execution stopped: {why}"
                    rep.execution["stops"] = [(s_.t, s_.reason) for s_ in run.stops]
                    self._log(rep, "result", rep.reason)
                    return rep
        final = latest["target"]
        err = float(np.linalg.norm(body.hand_position(ARM) - final.position))
        thr = claim_threshold(goal, final)
        rep.measured_error, rep.claim_threshold = err, thr
        rep.target = _describe(final)
        rep.verdict = "reached" if err <= thr else "failed"
        rep.reason = (
            f"reach point {err * 1000:.1f} mm from the perceived handle"
            if rep.verdict == "reached"
            else f"hand {err * 1000:.1f} mm from the perceived handle, above {thr * 1000:.1f} mm"
        )
        self._log(rep, "result", rep.reason)
        return rep
