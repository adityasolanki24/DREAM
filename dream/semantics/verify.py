"""Verification of an object-search goal against the robot's own evidence.

Independent of whoever claims success (a deterministic planner or a language model): it
reads only the object tracks (dream/perception/semantic.py) and the robot's pose. A track
counts as the requested object only on accumulated, consistent evidence:

- ``min_support`` views labelled as the target,
- from at least ``min_viewpoints`` camera positions ``DISTINCT_VIEW`` apart (one lucky frame,
  or one viewpoint's systematic confusion, is not enough),
- label purity ``min_purity`` among all labels given to the same physical object (a detector
  that calls it a wardrobe as often as a refrigerator has not identified it),
- confidence that the target is the track's majority label (``purity_rule="wilson_lower"``,
  the default): the Wilson score lower bound (``purity_confidence_z``, 1.645 = 95% one-sided)
  on target labels / all labels must exceed ``purity_majority`` (0.5). A point purity from a
  few views says little when the detector confuses classes (bathtub and shower share a
  confusion group): 3 of 5 views give a lower bound of 0.27 and are refused, 3 of 3 give 0.53
  and 4 of 4 give 0.60 and are accepted. ``purity_rule="point"`` applies ``min_purity`` alone;
  it is the rule every result frozen before the verifier-v2 confirmation used
  (experiments/configs/semantic_search_verifier_v2.json, results/semantic_search_verifier_v2).
- seen as the target within ``close_range`` at least once,
- misses (plainly in view at close range and not detected) at most ``max_miss_ratio`` of its
  target views (hallucinations and moved objects stop being detected).

``reached`` also requires the robot to stand within ``reach_distance`` of the object's
footprint, facing it, with a target detection of that track since arriving.

Statuses: ``target_found_verified``; ``possible_target_unconfirmed`` (a track carries the
target label but fails a check); ``target_not_yet_found``. The thresholds were set before the
held-out evaluation (experiments/configs) and are reported with every result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from dream.perception.semantic import SemanticMap, Track

DISTINCT_VIEW = 0.5  # m between camera positions for two views to count as different


@dataclass(frozen=True)
class VerifierThresholds:
    min_support: int = 3
    min_viewpoints: int = 2
    min_purity: float = 0.6
    close_range: float = 2.5  # m
    max_miss_ratio: float = 0.34
    reach_distance: float = 1.2  # m from the object's footprint
    reach_facing_deg: float = 35.0
    fresh_s: float = 6.0  # s: a "since arriving" detection is at most this old
    purity_rule: str = "wilson_lower"  # also the bound below; "point": min_purity only
    purity_confidence_z: float = 1.645  # one-sided 95%
    purity_majority: float = 0.5  # the Wilson lower bound on purity must exceed this


def distinct_viewpoints(cams: list[np.ndarray]) -> int:
    """Greedy count of camera positions at least DISTINCT_VIEW apart."""
    kept: list[np.ndarray] = []
    for c in cams:
        if all(float(np.linalg.norm(c - k)) >= DISTINCT_VIEW for k in kept):
            kept.append(c)
    return len(kept)


def wilson_lower(successes: int, total: int, z: float) -> float:
    """Lower end of the Wilson score interval for a binomial proportion (0 when total is 0)."""
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    centre = p + z2 / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return max(0.0, (centre - spread) / (1 + z2 / total))


def check_track(track: Track, target: str, th: VerifierThresholds) -> dict:
    support = track.support(target)
    cams = track.label_views.get(target, [])
    lower = wilson_lower(support, sum(track.labels.values()), th.purity_confidence_z)
    checks = {
        "support": support >= th.min_support,
        "viewpoints": distinct_viewpoints(cams) >= th.min_viewpoints,
        "purity": track.purity(target) >= th.min_purity,
        "close": track.closest.get(target, math.inf) <= th.close_range,
        "misses": track.misses <= th.max_miss_ratio * max(support, 1),
    }
    if th.purity_rule == "wilson_lower":
        checks["purity_confident"] = lower > th.purity_majority
    elif th.purity_rule != "point":
        raise ValueError(f"unknown purity_rule {th.purity_rule!r}")
    return {
        "track": track.id,
        "verified": all(checks.values()),
        "checks": checks,
        "support": support,
        "viewpoints": distinct_viewpoints(cams),
        "purity": round(track.purity(target), 3),
        "purity_lower": round(lower, 3),
        "closest_m": round(track.closest.get(target, math.inf), 2),
        "misses": track.misses,
    }


def footprint_distance(track: Track, xy) -> float:
    """Distance (m) from ``xy`` to the track's footprint box (0 inside)."""
    p = np.asarray(xy, float)[:2]
    d = np.maximum(np.maximum(track.lo[:2] - p, p - track.hi[:2]), 0.0)
    return float(np.linalg.norm(d))


def verify_find(
    semantic: SemanticMap,
    target: str,
    outcome: str,
    pose: np.ndarray,
    now: float,
    th: VerifierThresholds | None = None,
) -> dict:
    """The verdict on a FindObject goal from the robot's evidence (no ground truth)."""
    th = th or VerifierThresholds()
    cands = semantic.candidates(target)
    reports = [check_track(t, target, th) for t in cands]
    verified = [t for t, r in zip(cands, reports, strict=True) if r["verified"]]
    base = {"target": target, "outcome": outcome, "thresholds": asdict(th), "tracks": reports}
    if not verified:
        status = "possible_target_unconfirmed" if cands else "target_not_yet_found"
        why = (
            f"{len(cands)} object(s) labelled {target} at least once, none verified"
            if cands
            else f"no {target} detected yet"
        )
        return {**base, "satisfied": False, "status": status, "reason": why}
    best = max(verified, key=lambda t: (t.support(target), -t.id))
    found = {
        **base,
        "status": "target_found_verified",
        "track": best.id,
        "position": np.round(best.position, 3).tolist(),
        "footprint": [np.round(best.lo[:2], 3).tolist(), np.round(best.hi[:2], 3).tolist()],
    }
    if outcome != "reached":
        return {**found, "satisfied": True, "reason": f"{target} verified as object {best.id}"}
    distance = footprint_distance(best, pose[:2])
    bearing = math.atan2(best.position[1] - pose[1], best.position[0] - pose[0])
    facing = abs((bearing - pose[2] + math.pi) % (2 * math.pi) - math.pi)
    fresh = now - best.last_labelled.get(target, -math.inf) <= th.fresh_s
    ok = distance <= th.reach_distance and math.degrees(facing) <= th.reach_facing_deg and fresh
    return {
        **found,
        "satisfied": bool(ok),
        "reach": {
            "distance_m": round(distance, 2),
            "facing_deg": round(math.degrees(facing), 1),
            "seen_since_arrival": bool(fresh),
        },
        "reason": f"standing at the verified {target}"
        if ok
        else f"{target} verified as object {best.id}; not yet standing at it "
        f"({distance:.2f} m, {math.degrees(facing):.0f} deg)",
    }
