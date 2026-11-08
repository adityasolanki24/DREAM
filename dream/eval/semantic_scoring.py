"""Independent scoring of object-search episodes against the true simulation state.

Evaluation only (invariant 1): ground-truth instances, room types and true robot poses enter
here and nowhere in decision code. The ORIGINAL request is scored, not whatever goal was
bound: "Find the refrigerator" succeeds only if a real refrigerator was claimed.

Categories per episode (exactly one ``result``):
- ``success``: success claimed and the claimed object is a true instance of the requested
  class (for ``reached``, the robot also truly stands at it);
- ``false_success``: success claimed otherwise (wrong object, nothing there, wrong outcome);
- ``correct_refusal``: no claim, and the request is unsupported (refused as such) or the
  target is truly absent (reported exhausted or not found);
- ``false_absence``: the robot reported the search exhausted while a target existed;
- ``missed``: the target existed and the robot stopped without claiming (budget spent,
  unconfirmed or unreachable): an honest failure, not a false claim;
- ``wrong_refusal``: a supported request refused as unsupported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CLAIM_TOLERANCE = 0.75  # m from a true instance's footprint (object-scale, odometry drift)
REACH_TOLERANCE = 1.5  # m from the true footprint for a "reached" claim
REACH_FACING_DEG = 45.0
# Generator room types -> agent room categories counted as correct.
ROOM_MATCH = {
    "kitchen": {"kitchen"},
    "bedroom": {"bedroom"},
    "bathroom": {"bathroom"},
    "living_room": {"living_room"},
    "office": {"office"},
    "garage": {"garage", "workshop"},
    "laundry": {"laundry_room"},
    "storage": {"storage_room", "closet", "pantry", "basement"},
}


@dataclass(frozen=True)
class TrueInstance:
    id: str
    cls: str
    centre: np.ndarray  # (2,) world
    half: np.ndarray  # (2,) half extents in its own frame
    yaw: float
    room_type: str

    def distance(self, xy) -> float:
        """Distance from world ``xy`` to this instance's footprint rectangle (0 inside)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(xy, float)[:2] - self.centre
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        out = np.maximum(np.abs(local) - self.half, 0.0)
        return float(np.linalg.norm(out))


def odometry_to_world(true_pose, odom_pose):
    """Map odometry-frame points to world using the robot's true and estimated pose at the
    same instant (local drift compensation for scoring only)."""
    tx, ty, tyaw = map(float, true_pose)
    ox, oy, oyaw = map(float, odom_pose)
    dyaw = tyaw - oyaw
    c, s = math.cos(dyaw), math.sin(dyaw)

    def f(p):
        d = np.asarray(p, float)[:2] - (ox, oy)
        return np.array([tx + c * d[0] - s * d[1], ty + s * d[0] + c * d[1]])

    return f


def score_episode(
    *,
    target: str,
    outcome: str,
    supported_expected: bool,
    claimed: bool,
    completion: str | None,
    claim_xy_odom,
    instances: list[TrueInstance],
    to_world,
    true_pose_end,
) -> dict:
    """Score one episode; ``to_world`` maps odometry points to world at claim time."""
    targets = [i for i in instances if i.cls == target]
    out: dict = {
        "target": target,
        "outcome": outcome,
        "target_exists": bool(targets),
        "claimed": bool(claimed),
        "completion": completion,
    }
    claimed_inst = None
    if claimed and claim_xy_odom is not None:
        w = to_world(claim_xy_odom)
        nearest = min(instances, key=lambda i: i.distance(w)) if instances else None
        out["claim_world_xy"] = np.round(w, 3).tolist()
        if nearest is not None:
            out["claim_nearest"] = {"id": nearest.id, "cls": nearest.cls,
                                    "distance_m": round(nearest.distance(w), 3)}  # fmt: skip
        hits = [t for t in targets if t.distance(w) <= CLAIM_TOLERANCE]
        claimed_inst = min(hits, key=lambda t: t.distance(w)) if hits else None
        out["identity_error"] = (
            claimed_inst is None
            and nearest is not None
            and (nearest.distance(w) <= CLAIM_TOLERANCE)
        )
    reached_ok = None
    if claimed_inst is not None and outcome == "reached":
        p = np.asarray(true_pose_end, float)
        d = claimed_inst.distance(p[:2])
        bearing = math.atan2(*(claimed_inst.centre - p[:2])[::-1])
        facing = abs((bearing - p[2] + math.pi) % (2 * math.pi) - math.pi)
        reached_ok = d <= REACH_TOLERANCE and math.degrees(facing) <= REACH_FACING_DEG
        out["reach_true"] = {"distance_m": round(d, 3), "facing_deg": round(math.degrees(facing))}
    if claimed:
        ok = claimed_inst is not None and (outcome != "reached" or bool(reached_ok))
        result = "success" if ok and supported_expected else "false_success"
    elif not supported_expected:
        result = "correct_refusal" if completion == "goal_unsupported" else "missed"
    elif completion == "goal_unsupported":
        result = "wrong_refusal"
    elif not targets:
        result = "correct_refusal"
    elif completion == "search_exhausted":
        result = "false_absence"
    else:
        result = "missed"
    out["result"] = result
    out["original_goal_success"] = result == "success"
    out["false_success_claim"] = result == "false_success"
    return out


def score_rooms(regions: list[dict], to_world, room_at) -> dict:
    """Room classification of the robot's regions against true room types.

    ``regions``: summaries with ``centroid`` (odometry) and ``classification``;
    ``room_at(world_xy) -> true type or None``."""
    rows = []
    for r in regions:
        truth = room_at(to_world(r["centroid"]))
        label = r["classification"]
        if truth is None:
            continue
        top = r.get("top_category", "none")
        rows.append({"region": r["id"], "true": truth, "classified": label,
                     "correct": None if label == "ambiguous"
                     else label in ROOM_MATCH.get(truth, {truth}),
                     "top_category": top,
                     "top_correct": None if top == "none"
                     else top in ROOM_MATCH.get(truth, {truth})})  # fmt: skip
    decided = [r for r in rows if r["correct"] is not None]
    return {
        "regions": rows,
        "classified": len(decided),
        "ambiguous": len(rows) - len(decided),
        "correct": sum(r["correct"] for r in decided),
        "incorrect": sum(not r["correct"] for r in decided),
        "top_correct": sum(r["top_correct"] is True for r in rows),
        "top_with_evidence": sum(r["top_correct"] is not None for r in rows),
    }
