"""Automatic labels distinguish tool claims from independent physical evidence."""

from experiments.g1_skill_collection import skill_intervals


def test_failed_physical_gate_cannot_label_a_successful_tool_as_a_valid_lift():
    events = [{"tool": "grasp_object", "start_time_s": 0, "time_s": 1, "status": "succeeded"}]
    rows = [
        {
            "time_s": i / 100,
            "object_position_m": [0, 0, 0.9 if i == 0 else 0.95],
            "contacts": [
                {"body": f"left_hand_{finger}_0_link", "normal_force_n": 1.0}
                for finger in ("thumb", "index")
            ],
        }
        for i in range(101)
    ]
    summary = {"goal_success": False, "penetration_gate_passed": False, "safety_stop": False}
    label = skill_intervals(events, rows, summary)[0]
    assert label["independent_sustained_lift_contact_s"] == 1
    assert not label["contact_lift_verified"] and not label["placement_verified"]
    assert label["first_state_frame"] == 0 and label["last_state_frame_exclusive"] == 20
    summary["penetration_gate_passed"] = True
    assert skill_intervals(events, rows, summary)[0]["contact_lift_verified"]


def test_action_endpoint_cannot_cross_a_skill_boundary():
    event = {"tool": "reach_object", "start_time_s": 0.012, "time_s": 0.112, "status": "succeeded"}
    summary = {"goal_success": False, "penetration_gate_passed": True, "safety_stop": False}
    label = skill_intervals([event], [], summary)[0]
    # State at .05 -> action at .10 is inside; state at .10 -> .15 is outside.
    assert label["first_state_frame"] == 1
    assert label["last_state_frame_exclusive"] == 2
