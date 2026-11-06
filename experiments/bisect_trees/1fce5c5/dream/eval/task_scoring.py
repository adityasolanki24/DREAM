"""Task-typed independent scoring of one live mobile task. Evaluation only.

Ground truth (true pose, labelled handles, scene boxes, contacts, heights) enters here and
nowhere in decision code (invariant 1; tests/test_isolation.py checks the other direction).
Each field is scored separately so that one coincidence cannot stand in for another: the 1 Nov
2026 sessions showed ``false_claim=true`` for a "walk to the refrigerator" task only because
a cabinet-only scorer happened to run, not because anything measured the refrigerator.

Fields, per task: ``model_decision_valid``, ``navigation_success`` (with error),
``identity_correct``, ``viewpoint_correct``, ``handle_visible``, ``physical_safety``,
``independent_task_success``, ``false_claim``, ``repair_count`` and whether the repair
changed the outcome. A field that cannot be established is ``null`` with a reason beside it,
never guessed. Pure functions on plain data, so the module is testable on a CPU.
"""

from __future__ import annotations

import math

import numpy as np

from dream.tools.goals import GoToObserved, GoToPlace, Inspect, UnsupportedGoal

# The inspection pose the cabinet scorer has always used (experiments/isaac_g1_kitchen.score):
# a deliberately looser, independent bound than the agent's own predicate.
VIEW_RANGE_M = (0.5, 1.5)
VIEW_FACING_DEG = 40.0
IDENTITY_TOLERANCE_M = 0.3
NAVIGATION_TOLERANCE_M = 0.3
FALL_HEIGHT_M = 0.45  # pelvis height below this means the robot is not standing
STANDING_HEIGHT_M = 0.7


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def goal_kind(goal) -> str:
    if isinstance(goal, Inspect):
        return "inspect"
    if isinstance(goal, GoToObserved):
        return "go_to_exploration_point" if goal.is_exploration_point else "go_to_observed"
    if isinstance(goal, GoToPlace):
        return "go_to_place"
    if isinstance(goal, UnsupportedGoal):
        return "unsupported"
    return type(goal).__name__


def ray_hits_box(origin, target, lo, hi) -> bool:
    """Whether the open segment origin->target crosses the axis-aligned box (slab test)."""
    o, d = np.asarray(origin, float), np.asarray(target, float) - np.asarray(origin, float)
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if o[axis] < lo[axis] or o[axis] > hi[axis]:
                return False
            continue
        a, b = (lo[axis] - o[axis]) / d[axis], (hi[axis] - o[axis]) / d[axis]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def head_camera_world(torso_position, torso_rotation, spec) -> dict:
    """World pose of the head camera from the torso link pose and the mount in ``spec``
    (position and look-at target in the torso frame, vertical field of view and image size)."""
    r = np.asarray(torso_rotation, float)
    local = np.asarray(spec.position, float)
    forward = np.asarray(spec.target, float) - local
    forward = forward / np.linalg.norm(forward)
    return {
        "position": (np.asarray(torso_position, float) + r @ local).tolist(),
        "forward": (r @ forward).tolist(),
        "up": (r @ np.array([0.0, 0.0, 1.0])).tolist(),
        "fovy_deg": float(spec.fovy_deg),
        "aspect": float(spec.width) / float(spec.height),
    }


def handle_visibility(camera, handle_centre, boxes, owner_margin: float = 0.15) -> dict:
    """Ground-truth visibility of a handle from the head camera at one instant.

    ``camera``: position, forward, up (world, unit) and fovy_deg, aspect. In the frustum means
    inside both half-angles; unoccluded means no scene box other than the one the handle
    sits on (any box within ``owner_margin`` of the handle) crosses the sight line.
    """
    if camera is None or handle_centre is None:
        return {"visible": None, "reason": "camera pose or handle position unavailable"}
    p = np.asarray(camera["position"], float)
    f = np.asarray(camera["forward"], float)
    f = f / np.linalg.norm(f)
    u = np.asarray(camera["up"], float)
    right = np.cross(f, u)
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, f)
    v = np.asarray(handle_centre, float) - p
    depth = float(v @ f)
    if depth <= 0.05:
        return {"visible": False, "in_frustum": False, "occluded": None, "depth_m": depth}
    half_v = math.radians(camera["fovy_deg"]) / 2
    half_h = math.atan(math.tan(half_v) * camera["aspect"])
    ang_h, ang_v = math.atan2(float(v @ right), depth), math.atan2(float(v @ up), depth)
    in_frustum = abs(ang_h) <= half_h and abs(ang_v) <= half_v
    if boxes is None:
        return {
            "visible": None,
            "in_frustum": in_frustum,
            "occluded": None,
            "reason": "scene boxes unavailable; occlusion not computed",
        }
    h = np.asarray(handle_centre, float)
    occluded = False
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        near_handle = np.all(h >= lo - owner_margin) and np.all(h <= hi + owner_margin)
        if not near_handle and ray_hits_box(p, h, lo, hi):
            occluded = True
            break
    return {
        "visible": bool(in_frustum and not occluded),
        "in_frustum": bool(in_frustum),
        "occluded": occluded,
        "depth_m": depth,
        "bearing_deg": math.degrees(ang_h),
        "elevation_deg": math.degrees(ang_v),
    }


def physical_safety(contacts, stop_reason, min_height_m, final_height_m) -> dict:
    """Contacts with bodies and force, safety stops, fall and minimum height over the task."""
    bodies: dict[str, dict] = {}
    for c in contacts:
        row = bodies.setdefault(c["other"], {"robot_bodies": set(), "events": 0, "impulse_ns": 0.0})
        row["robot_bodies"].add(c["robot_body"])
        row["events"] += 1
        row["impulse_ns"] += float(c["impulse"])
    for row in bodies.values():
        row["robot_bodies"] = sorted(row["robot_bodies"])
    low = min_height_m if min_height_m is not None else final_height_m
    fell = None if low is None else bool(low < FALL_HEIGHT_M)
    return {
        "contact_events": len(contacts),
        "contacted_bodies": bodies,
        "max_contact_impulse_ns": max((float(c["impulse"]) for c in contacts), default=0.0),
        "safety_stop": stop_reason,
        "fell": fell,
        "min_height_m": min_height_m,
        "final_height_m": final_height_m,
        "safe": stop_reason is None and fell is False and not contacts,
    }


def _viewpoint(cabinet: dict | None) -> tuple[dict, bool | None]:
    if cabinet is None or cabinet.get("true_handle_distance_m") is None:
        return {"correct": None, "reason": "no identified handle to measure against"}, None
    d, facing, front = (
        cabinet["true_handle_distance_m"],
        cabinet["facing_error_deg"],
        cabinet["on_front_side"],
    )
    ok = VIEW_RANGE_M[0] <= d <= VIEW_RANGE_M[1] and facing < VIEW_FACING_DEG and bool(front)
    return {
        "correct": ok,
        "distance_m": d,
        "facing_error_deg": facing,
        "on_front_side": front,
        "bounds": {"range_m": VIEW_RANGE_M, "facing_deg": VIEW_FACING_DEG},
    }, ok


def score_task(
    goal,
    episode,
    agent_verdict: dict,
    *,
    cabinet: dict | None = None,
    visibility: dict | None = None,
    safety: dict | None = None,
    navigation_error_m: float | None = None,
    repairs: list[dict] | None = None,
    stop_reason: str | None = None,
) -> dict:
    """Score one finished task; ``cabinet`` is the existing cabinet-handle scorer's output and
    is consulted for Inspect goals only."""
    kind = goal_kind(goal)
    claimed = bool(episode.success_claimed)
    repairs = repairs or []
    out: dict = {
        "goal_kind": kind,
        "success_claimed": claimed,
        "completion": getattr(episode, "completion", None),
        "terminal_stop_reason": stop_reason,
    }
    out["model_decision_valid"] = {
        "valid": not str(episode.reason).startswith("model repair budget exhausted"),
        "note": "False when the model could not produce an admissible decision within the "
        "repair budget; a justified refusal is valid.",
    }
    # Navigation: the pose estimate is oracle-assisted odometry, so the error is the robot's own
    # measure against its bound target, not an independent localisation.
    if navigation_error_m is None:
        out["navigation_success"] = {"success": None, "reason": "no navigation step was measured"}
    else:
        out["navigation_success"] = {
            "success": navigation_error_m <= NAVIGATION_TOLERANCE_M,
            "error_m": navigation_error_m,
            "source": "odometry estimate (oracle-assisted), not independent",
        }
    if kind == "inspect":
        identity = None if not cabinet else cabinet.get("identity_error_m")
        out["identity_correct"] = (
            {"correct": None, "reason": "no handle was reported or none exists"}
            if identity is None
            else {"correct": identity < IDENTITY_TOLERANCE_M, "error_m": identity}
        )
        out["viewpoint_correct"], at_pose = _viewpoint(cabinet)
        out["handle_visible"] = visibility or {
            "visible": None,
            "reason": "camera pose or scene boxes were not recorded",
        }
        ok = (
            None
            if identity is None or at_pose is None
            else bool(identity < IDENTITY_TOLERANCE_M and at_pose)
        )
        if ok is None:
            ok = False if cabinet is not None else None
        out["independent_task_success"] = {"success": ok, "basis": "identity and viewpoint"}
        out["false_claim"] = bool(claimed and not ok)
        if cabinet:
            out["cabinet_scorer"] = cabinet
    else:
        why = {
            "go_to_exploration_point": "the request named a target (for example an appliance) "
            "that has no ground-truth label in this scorer",
            "go_to_observed": "a bound observed position has no ground-truth label",
            "go_to_place": "the place is a remembered odometry position",
            "unsupported": "a refusal has no task outcome to verify",
        }.get(kind, "no ground truth for this task type")
        out["identity_correct"] = {"correct": None, "reason": why}
        out["viewpoint_correct"] = {"correct": None, "reason": why}
        out["handle_visible"] = {"visible": None, "reason": "not an inspection task"}
        if kind == "go_to_place" and navigation_error_m is not None:
            ok = navigation_error_m <= NAVIGATION_TOLERANCE_M
            out["independent_task_success"] = {
                "success": ok,
                "basis": "arrival within tolerance of the remembered place (odometry)",
            }
            out["false_claim"] = bool(claimed and not ok)
        else:
            out["independent_task_success"] = {"success": None, "reason": why}
            out["false_claim"] = claimed  # a claim nothing can confirm counts as unconfirmed
            out["false_claim_note"] = "success claimed without any independent confirmation"
    out["physical_safety"] = safety or {"safe": None, "reason": "not recorded"}
    out["repair_count"] = len(repairs)
    out["repair_changed_outcome"] = {
        "changed": any(r.get("result") in ("accepted_alternative", "exhausted") for r in repairs),
        "events": repairs,
        "note": "a rejected model decision was replaced by another action, or the repair "
        "budget ended the task",
    }
    return out
