"""The confidence-aware purity rule of the object-search verifier (dream/semantics/verify.py)."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from dream.perception.semantic import Track
from dream.semantics.verify import VerifierThresholds, check_track, wilson_lower

POINT = VerifierThresholds()
WILSON = VerifierThresholds(purity_rule="wilson_lower")
Z = 1.645


def track(labels: dict[str, int], target: str, cams: list[np.ndarray], closest: float) -> Track:
    t = Track(1, np.array([3.2, -4.5, 0.9]), np.array([3.0, -4.7, 0.0]), np.array([3.4, -4.3, 1.8]))
    t.labels = dict(labels)
    t.label_views[target] = list(cams[: labels[target]])
    t.closest[target] = closest
    t.views = sum(labels.values())
    return t


def test_wilson_lower_values():
    assert wilson_lower(3, 5, Z) == pytest.approx(0.27, abs=0.005)
    assert wilson_lower(3, 3, Z) == pytest.approx(0.526, abs=0.002)
    assert wilson_lower(4, 4, Z) == pytest.approx(0.597, abs=0.002)
    assert wilson_lower(0, 4, Z) == 0.0
    assert wilson_lower(0, 0, Z) == 0.0


def test_wilson_lower_increases_with_evidence_and_is_below_the_point_estimate():
    prev = 0.0
    for n in range(1, 30):
        lo = wilson_lower(n, n, Z)
        assert prev < lo < 1.0
        prev = lo
    assert wilson_lower(60, 100, Z) < 0.6 < wilson_lower(60, 100, 0.0) + 1e-12
    assert math.isclose(wilson_lower(6, 10, 0.0), 0.6)


def test_track_124_labels_verify_under_the_point_rule_only():
    """H2, scenario E seed 6001: a shower fragment labelled bathtub 3 times in 5 views."""
    cams = [np.array([0.0, 0.0]), np.array([0.7, 0.0]), np.array([1.1, 0.0])]
    t = track({"bathtub": 3, "shower": 2}, "bathtub", cams, closest=2.3)
    assert t.purity("bathtub") == pytest.approx(0.6)
    point = check_track(t, "bathtub", POINT)
    assert point["verified"] and point["viewpoints"] == 2
    wilson = check_track(t, "bathtub", WILSON)
    assert not wilson["verified"]
    assert [k for k, ok in wilson["checks"].items() if not ok] == ["purity_confident"]
    assert wilson["purity_lower"] == pytest.approx(0.27, abs=0.005)
    assert wilson["purity"] == point["purity"] == 0.6


def test_three_of_three_passes_both_rules():
    cams = [np.array([0.0, 0.0]), np.array([0.7, 0.0]), np.array([1.4, 0.0])]
    t = track({"bathtub": 3}, "bathtub", cams, closest=2.0)
    for th in (POINT, WILSON):
        assert check_track(t, "bathtub", th)["verified"]
    assert check_track(t, "bathtub", POINT)["purity_lower"] == pytest.approx(0.526, abs=0.002)


def test_point_rule_checks_are_unchanged():
    cams = [np.array([0.0, 0.0]), np.array([0.7, 0.0]), np.array([1.4, 0.0])]
    r = check_track(track({"bathtub": 3}, "bathtub", cams, 2.0), "bathtub", POINT)
    assert set(r["checks"]) == {"support", "viewpoints", "purity", "close", "misses"}


def test_unknown_rule_is_rejected():
    t = track({"bathtub": 3}, "bathtub", [np.array([0.0, 0.0])] * 3, 2.0)
    with pytest.raises(ValueError):
        check_track(t, "bathtub", VerifierThresholds(purity_rule="wilson"))


def test_default_thresholds_are_the_frozen_ones():
    d = {f.name: f.default for f in fields(VerifierThresholds)}
    assert d == {
        "min_support": 3, "min_viewpoints": 2, "min_purity": 0.6, "close_range": 2.5,
        "max_miss_ratio": 0.34, "reach_distance": 1.2, "reach_facing_deg": 35.0,
        "fresh_s": 6.0, "purity_rule": "point", "purity_confidence_z": 1.645,
        "purity_majority": 0.5,
    }  # fmt: skip
