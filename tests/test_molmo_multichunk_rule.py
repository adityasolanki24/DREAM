"""The pre-registered H4 rule, branch by branch."""

from experiments.analyse_molmo_multichunk import apply_rule


def test_rejected_needs_proximity_and_close_command_near_it():
    assert apply_rule(10, 0.04, 0.9, 0.8)["h4"] == "rejected"
    assert apply_rule(10, 0.04, 0.9, 0.2)["h4"] == "inconclusive"  # close only far from the object
    assert apply_rule(10, 0.06, 0.9, None)["h4"] == "inconclusive"


def test_stands_needs_far_tcp_and_gripper_never_above_point_one():
    assert apply_rule(10, 0.30, 0.1, None)["h4"] == "stands"
    assert apply_rule(10, 0.30, 0.2, None)["h4"] == "inconclusive"
    assert apply_rule(10, 0.20, 0.0, None)["h4"] == "inconclusive"  # intermediate distance


def test_fewer_than_ten_inferences_is_always_inconclusive():
    assert apply_rule(9, 0.01, 1.0, 1.0)["h4"] == "inconclusive"
    assert apply_rule(9, 0.9, 0.0, None)["h4"] == "inconclusive"
