"""A recovery must begin a fresh uninterrupted physical hold interval."""

import pytest

from dream.eval.g1 import verify_contact_hold


def rows(phases):
    return [
        {
            "phase": phase,
            "t": (i + 1) * 0.02,
            "table_clearance_m": 0.11,
            "hand_only_contact": True,
        }
        for i, phase in enumerate(phases)
    ]


def test_hold_cannot_join_samples_across_a_height_correction():
    trace = rows(["hold"] * 75 + ["hold_adjust"] + ["hold"] * 75)
    assert not verify_contact_hold(trace, 0.10, 3.0, 0.02)


def test_completed_recovery_requires_a_new_full_window():
    trace = rows(["hold"] * 75 + ["hold_adjust"] + ["hold"] * 150)
    trace[50]["table_clearance_m"] = 0.09
    assert verify_contact_hold(trace, 0.10, 3.0, 0.02)


@pytest.mark.parametrize("fault", ["clearance", "contact", "time_gap"])
def test_failed_sample_inside_final_window_is_rejected(fault):
    trace = rows(["hold"] * 150)
    if fault == "clearance":
        trace[90]["table_clearance_m"] = 0.099
    elif fault == "contact":
        trace[90]["hand_only_contact"] = False
    else:
        trace[90]["t"] += 0.01
    assert not verify_contact_hold(trace, 0.10, 3.0, 0.02)


@pytest.mark.parametrize(("hold_s", "tick_s"), [(0.0, 0.02), (-1.0, 0.02), (3.0, 0.0)])
def test_empty_or_invalid_duration_cannot_establish_a_hold(hold_s, tick_s):
    with pytest.raises(ValueError):
        verify_contact_hold([], 0.10, hold_s, tick_s)


def test_duration_is_never_rounded_down():
    assert not verify_contact_hold(rows(["hold"]), 0.10, 0.025, 0.02)
