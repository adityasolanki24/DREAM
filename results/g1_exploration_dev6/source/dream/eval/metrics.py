"""Scoring the agent's belief against ground truth. Evaluation only."""

from __future__ import annotations

import numpy as np

from dream.agent.belief import Belief
from dream.agent.model import FIXED, REVOLUTE, TYPE_NAMES
from dream.sim.mechanisms import MechanismSpec


def axis_error_deg(est: np.ndarray, true: np.ndarray) -> float:
    """Signed axis error. The sign encodes the opening direction, so it matters."""
    cos = np.dot(est, true) / (np.linalg.norm(est) * np.linalg.norm(true))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def point_to_line(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    r = point - origin
    return float(np.linalg.norm(r - np.dot(r, axis) * axis))


def score_belief(
    belief: Belief,
    spec: MechanismSpec,
    min_prob: float = 0.9,
    max_axis_deg: float = 10.0,
    max_line_m: float = 0.02,
) -> dict:
    """Is the mechanism identified? MAP type correct with confidence, plus accurate geometry.

    The thresholds are project engineering criteria, not published standards.
    """
    probs = belief.type_probs()
    fit = belief.fit_check()
    if spec.joint_type not in TYPE_NAMES:  # outside the agent's model class
        return {
            "p_true_type": None,
            "map_correct": False,
            "identified": False,
            "type_entropy": belief.type_entropy(),
            "max_type_prob": float(probs.max()),
            "axis_err_deg": None,
            "line_err_m": None,
            **fit,
        }
    true_kind = TYPE_NAMES.index(spec.joint_type)
    k = belief.map_type()
    out = {
        "p_true_type": float(probs[true_kind]),
        "map_correct": bool(k == true_kind),
        "type_entropy": belief.type_entropy(),
        "axis_err_deg": None,
        "line_err_m": None,
        "max_type_prob": float(probs.max()),
        **fit,
    }
    geometry_ok = True
    if true_kind != FIXED:
        est = belief.estimate(true_kind)
        out["axis_err_deg"] = axis_error_deg(est["axis"], spec.axis)
        geometry_ok = out["axis_err_deg"] < max_axis_deg
        if true_kind == REVOLUTE:
            out["line_err_m"] = point_to_line(est["origin"], spec.origin, spec.axis)
            geometry_ok &= out["line_err_m"] < max_line_m
    out["identified"] = bool(out["map_correct"] and probs[true_kind] >= min_prob and geometry_ok)
    return out
