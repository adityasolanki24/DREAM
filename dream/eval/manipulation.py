"""Ground-truth success conditions for tabletop tasks. Evaluation only.

These read true object poses from the simulator and are deliberately independent of the
agent's own ``verify_goal``, so a task counts as done only if the physical world says so.
"""

from __future__ import annotations

import numpy as np

from dream.sim.tabletop_env import TabletopEnv

REST_SPEED = 0.02  # m/s: an object moving faster than this is not "resting"


def _resting(env: TabletopEnv, name: str) -> bool:
    return float(np.linalg.norm(env.data.body(name).cvel[3:])) < REST_SPEED


def object_in_tray(env: TabletopEnv, name: str) -> bool:
    """Centre inside the tray's inner floor, below the wall tops, and at rest."""
    spec = env.privileged_spec()
    tray = spec.tray
    half = next(o.half_size for o in spec.objects if o.name == name)
    p = env.privileged_object_pos(name)
    inside = np.all(np.abs(p[:2] - np.array(tray.xy)) < tray.inner_half)
    return bool(inside and p[2] < tray.wall_height + half and _resting(env, name))


def object_on_top(env: TabletopEnv, top: str, bottom: str, xy_tol: float | None = None) -> bool:
    """``top`` rests on ``bottom``: centred within half an edge, one edge higher (+-1 cm)."""
    sizes = {o.name: 2 * o.half_size for o in env.privileged_spec().objects}
    pt, pb = env.privileged_object_pos(top), env.privileged_object_pos(bottom)
    xy_tol = xy_tol if xy_tol is not None else sizes[bottom] / 2
    stacked = np.linalg.norm(pt[:2] - pb[:2]) < xy_tol
    height_ok = abs((pt[2] - pb[2]) - (sizes[top] + sizes[bottom]) / 2) < 0.01
    return bool(stacked and height_ok and _resting(env, top) and _resting(env, bottom))


def placement_error(env: TabletopEnv, name: str, target: np.ndarray) -> float:
    """Horizontal distance (m) between the object's true centre and the intended target."""
    return float(np.linalg.norm(env.privileged_object_pos(name)[:2] - np.asarray(target)[:2]))


def safety_summary(env: TabletopEnv) -> dict:
    r = env.robot
    return {
        "protective_stop": bool(r.protective_stop),
        "stop_reason": r.stop_reason,
        "peak_link_contact_N": float(r.peak_contact_force),
        "peak_finger_contact_N": float(r.peak_finger_force),
        "link_collision": bool(r.peak_contact_force > 1.0),
    }
