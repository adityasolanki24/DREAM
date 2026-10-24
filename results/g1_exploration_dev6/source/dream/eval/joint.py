"""Execution/evaluation wiring for the separate joint controller; no scene answers to policy."""

from __future__ import annotations

import threading

import numpy as np


def execute_target(
    env,
    joint_rad: np.ndarray,
    width_m: float,
    period_s: float,
    cancel: threading.Event | None = None,
) -> dict:
    """One bounded waypoint, measured settling, then a safe joint hold.

    Runs against synchronous stepping in tests or the existing live physics thread. A
    rejected target or cancellation cannot fall back to Cartesian/classical execution.
    """
    cancel = cancel or threading.Event()
    with env.lock:
        before = env.robot.state()
        if cancel.is_set():
            env.robot.hold()
            return {"executed": False, "reason": "cancelled before target"}
        try:
            receipt = env.robot.submit(joint_rad, width_m, period_s)
        except ValueError as error:
            env.robot.hold()
            return {"executed": False, "reason": str(error)}

    def done():
        with env.lock:
            state = env.robot.state()
            return (
                cancel.is_set()
                or state.protective_stop
                or (
                    not env.robot.moving
                    and np.max(np.abs(state.joint_vel)) < 0.02
                    and np.max(np.abs(state.joint_pos - joint_rad)) < 0.01
                )
            )

    try:
        reached = env.wait_until(done, receipt["duration_s"] + 1.0)
        with env.lock:
            after = env.robot.state()
            error = float(np.max(np.abs(after.joint_pos - joint_rad)))
            return {
                **receipt,
                "executed": True,
                "target_reached": bool(
                    reached and not cancel.is_set() and not after.protective_stop and error < 0.01
                ),
                "reason": "cancelled"
                if cancel.is_set()
                else (after.stop_reason or ("target reached" if reached else "tracking timeout")),
                "tracking_error_rad": error,
                "sim_start": before.time,
                "sim_end": after.time,
                "actual_displacement_rad": (after.joint_pos - before.joint_pos).tolist(),
            }
    finally:
        with env.lock:
            env.robot.hold()


def execute_chunk(
    env,
    joints_rad: np.ndarray,
    widths_m: np.ndarray,
    period_s: float,
    *,
    max_targets: int,
    max_sim_s: float,
    cancel: threading.Event | None = None,
) -> dict:
    """Bounded recorded/predicted prefix, with measured rest after every target.

    Deliberately retains conservative rest-to-rest timing and all original safety limits.
    This is not continuous chunk tracking at the model's training frequency. No inference
    or fallback occurs here; callers label provenance and log untouched predictions first.
    """
    q, w = np.asarray(joints_rad), np.asarray(widths_m)
    if (
        q.ndim != 2
        or q.shape[1] != 7
        or w.shape != (len(q),)
        or not np.isfinite(q).all()
        or not np.isfinite(w).all()
        or isinstance(max_targets, bool)
        or not isinstance(max_targets, int)
        or not 1 <= max_targets <= min(15, len(q))
        or not np.isfinite(max_sim_s)
        or not 0 < max_sim_s <= 30
    ):
        raise ValueError("invalid bounded joint chunk")
    cancel = cancel or threading.Event()
    start = env.robot_state().time
    records = []
    try:
        for target, width in zip(q[:max_targets], w[:max_targets], strict=True):
            if cancel.is_set() or env.robot_state().time - start >= max_sim_s:
                return {"steps": records, "completed": False, "reason": "cancelled or budget"}

            # Keep the budget as an active stop condition, including during a target.
            class Budget:
                def is_set(self):
                    return cancel.is_set() or env.robot_state().time - start >= max_sim_s

            record = execute_target(env, target, float(width), period_s, Budget())
            records.append(record)
            if not record.get("target_reached"):
                return {"steps": records, "completed": False, "reason": record["reason"]}
        return {"steps": records, "completed": True, "reason": "bounded prefix reached"}
    finally:
        with env.lock:
            env.robot.hold()
