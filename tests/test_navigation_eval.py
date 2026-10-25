"""Evaluator arithmetic of experiments/g1_navigation_eval.py on synthetic records."""

from __future__ import annotations

import numpy as np

from experiments.g1_navigation_eval import gates, motion_metrics


def rows(cmds, tool="go_to"):
    out, yaw, xy = [], 0.0, np.zeros(2)
    for v, w in cmds:
        out.append({"tool": tool, "cmd": [v, 0, w], "yaw": yaw, "xy": xy.tolist(), "v_body": v})
        yaw += w * 0.02
        xy = xy + v * 0.02 * np.array([np.cos(yaw), np.sin(yaw)])
    return out


def test_turn_in_place_stops_and_path_are_measured():
    cmds = [(0.0, 0.5)] * 50 + [(0.3, 0.0)] * 50 + [(0.0, 0.0)] * 10 + [(0.3, 0.0)] * 50
    m = motion_metrics(rows(cmds))["nav"]
    assert abs(m["turn_in_place_s"] - 1.0) < 1e-9
    assert m["stop_start_cycles"] == 1
    assert abs(m["path_m"] - 0.3 * 0.02 * 99) < 0.01
    assert abs(m["rotation_rad"] - 0.5) < 0.02


def test_perception_turns_are_not_navigation():
    m = motion_metrics(rows([(0.0, 0.5)] * 20, tool="look_around"))
    assert m["nav"] is None and m["all"]["turn_in_place_s"] > 0


def ep(nav, success=True, contact=0, jerk=100.0):
    return {"navigator": nav, "success": success, "contact_steps": contact, "false_claim": False,
            "fell": False, "motion": {"nav": {"angular_jerk_rms": jerk}}}  # fmt: skip


def test_gates_pass_and_fail_as_preregistered():
    cfg = {"gates": {"rpp_success_not_below_classic_minus": 1,
                     "rpp_contact_episodes_not_above_classic": True,
                     "rpp_nav_angular_jerk_median_reduction_min": 0.2,
                     "rpp_false_claims_max": 0, "rpp_falls_max": 0}}  # fmt: skip
    good = [ep("classic", jerk=150), ep("classic", contact=5, jerk=150), ep("rpp", jerk=100),
            ep("rpp", jerk=100)]  # fmt: skip
    assert gates(good, cfg)["all_pass"]
    smooth_but_worse = [ep("classic"), ep("classic"), ep("rpp", False, jerk=10),
                        ep("rpp", False, jerk=10)]  # fmt: skip
    g = gates(smooth_but_worse, cfg)
    assert not g["rpp_success_not_below_classic_minus"] and not g["all_pass"]
    rough = [ep("classic", jerk=100), ep("rpp", jerk=90)]
    assert not gates(rough, cfg)["rpp_nav_angular_jerk_median_reduction_min"]
