"""Closed-loop task-space correction after a posture has been executed (PLAN C5 follow-up).

The open-loop plan leaves a residual hand error (controller tracking, arm sag, odometry). After
the real execution, ``correct_reach`` measures the hand by forward kinematics from the robot's own
sensors, compares it with the target from the LATEST PERCEIVED observation (a callable supplied
by the caller), and while the error exceeds ``CORRECTION_TOLERANCE`` solves a small IK update
from the measured configuration (stance fixed, same capability limits and perceived obstacles).
The update is applied as a DELTA on the previous command (integral action: the controller's
steady-state bias is not modelled, it is cancelled by feedback) and re-executed through the
backend under the supervisor. At most ``MAX_ITERATIONS`` corrections; it stops when an iteration
does not reduce the error, when the IK or the capabilities refuse, or on a protective stop.
Every iteration is recorded in ``ExecutionReport.iterations``.

TORQUE-AWARE STEPS (W6). In W5 most protective stops were torque stops DURING correction (peaks
1.0-1.32 of the actuator limit against a static estimate of 0.27-0.45): each correction moved the
arm by the IK step without regard to the load the arm already carried. Now the measured ratio
(the largest requested-torque / actuator-limit over the joints, at this tick) is used: the
predicted ratio after a step is the measured one plus the change of the static gravity-torque
ratio between the measured and the new posture, and the step is scaled by the largest
``alpha`` in (0, 1] that keeps the prediction at or below ``TORQUE_SOFT``. If even the smallest
step would exceed it, the correction ends with action ``torque_limited`` (reported, not
executed) instead of running into the supervisor's stop.

Success is not decided here: the evaluator pre-registers its own threshold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dream.robot.g1_posture import PosturePlan, PosturePlanner, static_torque_ratio
from dream.robot.g1_supervisor import WholeBodySupervisor
from dream.robot.wholebody import ExecutionReport, WholeBodyTarget

# Assumptions (DREAM choices, not calibrated): correct when the hand is more than 1 cm from the
# perceived target, at most 3 times, and stop unless an iteration improves the error by 1 mm.
CORRECTION_TOLERANCE = 0.01  # m
MAX_ITERATIONS = 3
MIN_IMPROVEMENT = 0.001  # m
TORQUE_SOFT = 0.8  # fraction of the actuator limit a correction step may be predicted to reach
# (below the supervisor's stop at 1.0 and the rollout gate's 0.9; an assumption, not calibrated)
ALPHAS = (1.0, 0.7, 0.5, 0.3, 0.15)  # step scalings tried, largest first


def correct_reach(
    backend,
    supervisor: WholeBodySupervisor,
    planner: PosturePlanner,
    plan: PosturePlan,
    target_provider,
    obstacles=(),
    tolerance: float = CORRECTION_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    executed: WholeBodyTarget | None = None,
    approach_axis=None,
) -> ExecutionReport:
    """Correct the hand toward ``target_provider()`` (odometry frame) after ``plan`` was executed.

    ``executed`` is the target last sent to the backend (default ``plan.target``). The report's
    ``final_target`` is the last target this call sent (None if it sent nothing), so that a
    following stage can pass it as ``executed``."""
    arm = plan.arm
    prev = executed or plan.target
    report = ExecutionReport(True, {"point": None})
    last_error = None
    before_last = None  # the target in force before the last executed correction
    for i in range(max_iterations + 1):
        point = np.asarray(target_provider(), float)
        obs = backend.observe()
        row = supervisor.measure(obs)
        hand = np.asarray(row[f"ach_hand_{arm}"])
        error = float(np.linalg.norm(hand - point))
        it: dict[str, object] = {
            "iteration": i, "error": error, "hand": tuple(float(v) for v in hand),
            "target": tuple(float(v) for v in point), "action": None,
        }  # fmt: skip
        report.iterations.append(it)
        report.requested = {"point": tuple(float(v) for v in point)}
        report.achieved = {"hand_error": error}
        if error <= tolerance:
            it["action"] = "within_tolerance"
            break
        if last_error is not None and last_error - error < MIN_IMPROVEMENT:
            it["action"] = "no_improvement"
            if error > last_error and before_last is not None:  # undo the harmful correction
                undo = backend.execute(before_last, monitor=supervisor)
                report.trace += undo.trace
                report.stops += undo.stops
                it["action"] = "no_improvement_rolled_back"
                it["rollback_ok"] = undo.completed
                report.final_target = before_last
            break
        if i == max_iterations:
            it["action"] = "iteration_limit"
            break
        last_error = error
        q0 = planner.measured_qpos(obs.prop, obs.pose, row["ach_height"])
        q_new = planner.refine(q0, plan.stance, arm, point, obstacles, approach_axis)
        if q_new is None:
            it["action"] = "ik_failed"
            break
        qadr = planner.pm.qadr
        arm_meas = np.asarray(obs.prop.q)[15:29]
        measured_ratio = float(row.get("torque_ratio", 0.0))
        base_static = static_torque_ratio(planner.pm, q0)
        alpha = None
        for a in ALPHAS:
            q_a = q0.copy()
            q_a[qadr] = q0[qadr] + a * (q_new[qadr] - q0[qadr])
            predicted = measured_ratio + static_torque_ratio(planner.pm, q_a) - base_static
            if predicted <= TORQUE_SOFT or predicted <= measured_ratio:
                alpha = a
                break
        it["torque_measured"], it["torque_joint"] = measured_ratio, int(row.get("torque_joint", 0))
        if alpha is None:
            it["action"] = "torque_limited"
            break
        if alpha < 1.0:
            it["step_scale"] = alpha
            q_new = q0.copy()
            q_new[qadr] = q_a
            q_new[0:7] = q0[0:7]
        s0 = planner.pm.kinematic_state(obs.prop.quat, obs.prop.q)
        s1 = planner.pm.kinematic_state(q_new[3:7], q_new[7:36])
        # Legs are held at their measured values, so the pelvis height command stays as
        # executed; the torso rpy command and the arm move, as DELTAS on the previous command.
        caps = planner.caps
        rpy = np.asarray(prev.torso_rpy) + (s1["rpy"] - s0["rpy"])
        rpy = np.clip(rpy, caps.torso_rpy_min, caps.torso_rpy_max)  # IK stayed inside already
        nxt = replace(
            prev,
            q=q_new,
            hands=(replace(prev.hands[0], position=tuple(float(v) for v in point)),),
            torso_rpy=tuple(float(v) for v in rpy),
            arm_joints=prev.arm_joints + (q_new[qadr[15:29]] - arm_meas),
        )
        bad = caps.violations(nxt.pelvis_height, nxt.torso_rpy)
        if bad:
            it["action"] = "outside_capabilities: " + "; ".join(bad)
            break
        it["action"] = "executed"
        it["command_delta_arm"] = float(np.max(np.abs(nxt.arm_joints - prev.arm_joints)))
        run = backend.execute(nxt, monitor=supervisor)
        report.trace += run.trace
        report.stops += run.stops
        if run.refused or run.stops:
            it["action"] = "refused: " + run.refused if run.refused else "stopped"
            report.completed = False
            report.refused = run.refused
            break
        before_last, prev = prev, nxt
        report.final_target = nxt
    return report
