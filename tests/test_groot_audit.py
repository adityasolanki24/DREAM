"""Conditional diagnostics are not an action adapter or model-inference substitute."""

import numpy as np
import pytest

from experiments.audit_groot_actions import joint_diagnostics


def test_decoded_absolute_targets_are_not_offset_twice_and_violations_are_preserved():
    report = joint_diagnostics(
        np.array([[0.7, -0.1], [1.1, -0.4]]),
        np.array([0.6, 0.0]),
        np.array([[-1, 1], [-0.2, 0.2]]),
        np.array([0.4, 1.0]),
    )
    assert report["first_target_delta_rad"] == pytest.approx([0.1, -0.1])
    assert report["first_target_slew_time_lower_bound_s"] == pytest.approx(0.25)
    assert report["samples_outside_mechanical_range"] == 2
    assert report["max_range_excess_rad"] == pytest.approx(0.2)
    assert not report["actuation_permitted"]


def test_diagnostic_rejects_nonfinite_or_unknown_speed():
    with pytest.raises(ValueError, match="contract"):
        joint_diagnostics(np.zeros((40, 1)), np.zeros(1), np.array([[-1, 1]]), np.zeros(1))
