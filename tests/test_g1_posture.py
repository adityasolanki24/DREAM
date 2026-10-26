"""Whole-body posture planner (dream/robot/g1_posture.py), rollout gate (g1_rollout.py).

The fast tests exercise the planner on the robot-only model with HYPOTHETICAL capabilities
(stated bounds, not measured), which is how planning is allowed to run without an envelope file.
The slow test runs the real pretrained controller: rollout on a planning copy, then the monitored
real execution, and reports the achieved hand error.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from dream.robot import g1_posture as gp
from dream.robot import g1_wbc as w
from dream.robot.wholebody import ENVELOPE_JSON, ControllerCapabilities, torso_rpy_from_matrices

WIDE = ControllerCapabilities.unmeasured("hypothetical", (0.2, 0.74), (-0.52,) * 3, (0.52,) * 3)
SMALL = gp.PlannerConfig(radii=(0.5,), n_angles=4, max_stances=2)  # the two nearest stances
SOURCE = Path(gp.__file__).read_text()


@pytest.fixture(scope="module")
def planner():
    return gp.PosturePlanner(WIDE, SMALL)


@pytest.fixture(scope="module")
def three_heights(planner):
    return {z: planner.plan([0.7, 0.0, z]) for z in (0.25, 0.75, 1.35)}


def test_same_code_path_crouches_low_targets_and_stands_for_high_ones(three_heights):
    """No height rule exists (see test below): the posture follows from the optimisation."""
    best = {z: r.plans[0] for z, r in three_heights.items()}
    for z, r in three_heights.items():
        assert r.feasible, f"target at {z} m: {r.reasons()}"
        assert not r.executable  # hypothetical capabilities: planning only
        assert r.plans[0].metrics["hand_residual"] < gp.HAND_TOLERANCE
    h = {z: p.pelvis_height for z, p in best.items()}
    print("pelvis height by target height:", {z: round(v, 3) for z, v in h.items()})
    assert h[0.25] < h[1.35] - 0.25  # significantly lower
    assert h[1.35] > 0.70  # the high target is not crouched
    assert h[0.25] < h[0.75] <= h[1.35] + 1e-6  # monotone: mid targets sit between
    for p in best.values():
        assert 0.2 - 1e-3 <= p.pelvis_height <= 0.74 + 1e-3
        assert np.all(np.abs(p.torso_rpy) <= 0.52 + 0.01)
        assert p.metrics["clearance"] > -1e-3 and p.metrics["com_margin"] > 0


def test_no_height_specific_rule_and_no_controller_constants_in_the_planner():
    assert not re.search(r"target\[2\]|z_target|target_z", SOURCE)
    for banned in ("HEIGHT_CMD", "HEIGHT_RANGE", "RPY_RANGE", "RPY_CMD", "WBCLocomotion"):
        assert banned not in SOURCE, f"planner references controller constant {banned}"


def test_plan_is_a_whole_body_target_with_derived_quantities(planner, three_heights):
    plan = three_heights[0.75].plans[0]
    t = plan.target
    assert t.q.shape == (36,) and t.hands[0].arm == plan.arm
    assert np.allclose(t.hands[0].position, [0.7, 0.0, 0.75])
    d = planner.pm.data_at(t.q)
    assert t.pelvis_height == pytest.approx(d.xpos[planner.pm.pelvis][2], abs=1e-9)
    rpy = torso_rpy_from_matrices(
        d.xmat[planner.pm.pelvis].reshape(3, 3), d.xmat[planner.pm.torso].reshape(3, 3)
    )
    assert np.allclose(t.torso_rpy, rpy)
    assert np.array_equal(planner.pm.qadr, np.arange(7, 36))  # q[7:36] in actuator order
    assert np.allclose(t.arm_joints, t.q[7 + w.N_ACTION : 7 + w.N_BODY])
    assert t.arm_joints.shape == (14,)
    # the achieved hand pose from encoders + IMU agrees with the planning FK
    s = planner.pm.kinematic_state(t.q[3:7], t.q[7:36])
    assert s["height"] == pytest.approx(t.pelvis_height, abs=1e-6)


def test_limits_come_only_from_the_capabilities(planner):
    tight = ControllerCapabilities.unmeasured(
        "tight", (0.6, 0.74), (0.0, 0.0, 0.0), (0.0, 0.2, 0.0)
    )
    p = gp.PosturePlanner(tight, SMALL)
    low = p.plan([0.7, 0.0, 0.25])
    assert not low.feasible  # the crouch the wide planner found is outside these limits
    assert low.reasons().get("hand_residual", 0) + low.reasons().get("outside_capabilities", 0)
    ok = p.plan([0.7, 0.0, 0.9])
    for plan in ok.plans:
        assert 0.6 - 1e-3 <= plan.pelvis_height <= 0.74 + 1e-3
        assert abs(plan.torso_rpy[0]) < 0.01 and abs(plan.torso_rpy[2]) < 0.01
        assert -0.01 <= plan.torso_rpy[1] <= 0.21


def test_unreachable_target_returns_reasons(planner):
    r = planner.plan([0.7, 0.0, 2.6])
    assert not r.feasible and r.reasons().get("hand_residual", 0) >= 1
    assert all(f.reasons for f in r.failures)


def test_footprint_radius_matches_the_navigation_radius():
    from dream.tools.mobile import COLLISION_RADIUS

    assert gp.FOOTPRINT_RADIUS == COLLISION_RADIUS


def test_stances_clear_perceived_obstacles(planner):
    box = gp.PerceivedBox((0.2, 0.0, 0.5), (0.1, 0.1, 0.5))
    assert gp.stance_blocked((0.2, 0.2), [box])  # inside the footprint radius of the box
    assert not gp.stance_blocked((0.2, 0.6), [box])
    low = gp.PerceivedBox((0.2, 0.2, 0.01), (0.1, 0.1, 0.01))
    assert not gp.stance_blocked((0.2, 0.2), [low])  # below the step height
    stances, blocked = planner.stances(np.array([0.7, 0.0, 0.8]), (0, 0, 0), [box])
    assert blocked >= 1
    assert all(not gp.stance_blocked(s[:2], [box]) for s in stances)


def test_perceived_obstacle_changes_the_plan_and_keeps_clearance(planner):
    # a post right next to the nominal stance: the plan must keep the body clear of it
    box = gp.PerceivedBox((0.45, 0.35, 0.6), (0.08, 0.08, 0.6))
    r = planner.plan([0.7, 0.0, 0.8], obstacles=[box])
    for p in r.plans:
        assert p.metrics["clearance"] > -1e-3
        assert not gp.stance_blocked(p.stance[:2], [box])
    assert r.feasible


def test_approach_axis_orients_the_fingers(planner):
    axis = (1.0, 0.0, 0.0)
    r = planner.plan([0.7, 0.0, 0.8], approach_axis=axis)
    plan = r.plans[0]
    d = planner.pm.data_at(plan.target.q)
    grip = d.site_xmat[planner.pm.model.site(f"{plan.arm}_grip").id].reshape(3, 3)
    assert grip[:, 0] @ np.array(axis) > 0.9
    assert plan.target.hands[0].approach_axis == axis


def test_static_torque_proxy_is_a_ranking_quantity_in_range(planner):
    ratio = gp.static_torque_ratio(planner.pm, planner.pm.q_nominal)
    assert 0.0 < ratio < 1.0
    assert "proxy" in gp.__doc__ or "RANK" in gp.__doc__


def test_ik_timing_is_reported(three_heights, planner):
    assert planner.last_timings and all(t > 0 for t in planner.last_timings)


def test_measured_capabilities_are_narrower_than_the_input_bound():
    if not ENVELOPE_JSON.exists():
        pytest.skip("no measured envelope")
    caps = ControllerCapabilities.from_envelope_json()
    assert caps.executable and caps.height_range[0] > w.HEIGHT_RANGE[0]


# -- slow: the real controller ------------------------------------------------------------------

HAVE_WEIGHTS = (w.policy_cache() / "GR00T-WholeBodyControl-Walk.onnx").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_WEIGHTS, reason="WBC weights not fetched")
@pytest.mark.skipif(not hasattr(w.WBCLocomotion, "set_posture"), reason="g1_wbc.set_posture absent")
@pytest.mark.skipif(not ENVELOPE_JSON.exists(), reason="no measured envelope: not executable")
def test_rollout_gate_then_monitored_real_execution_reaches_a_mid_height_target():
    from dream.robot.g1_rollout import verify_top_k
    from dream.robot.g1_supervisor import WholeBodySupervisor, summary
    from dream.robot.wholebody import WBCBackend
    from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1

    caps = ControllerCapabilities.from_envelope_json()
    planner = gp.PosturePlanner(
        caps, gp.PlannerConfig(radii=(0.45, 0.55), n_angles=4, max_stances=4)
    )
    target = np.array([0.5, 0.0, 0.8])  # mid height, in front, odometry frame == start frame
    result = planner.plan(target)
    assert result.feasible and result.executable, result.reasons()

    def factory(model, stance):
        env = G1MobileEnv(model, np.random.default_rng(0), head_camera=None)
        env.reset(*stance)
        return WBCBackend(env, w.WBCLocomotion.from_cache(), caps)

    chosen, tried = verify_top_k(result.plans, factory, [], planning_model=planner.pm)
    print("rollouts:", [(round(r.seconds, 1), r.passed, r.reasons) for _, r in tried])
    assert chosen is not None, [r.reasons for _, r in tried]

    env = G1MobileEnv(build_mobile_g1(head_camera=None), np.random.default_rng(1), head_camera=None)
    env.reset(*chosen.stance)
    backend = WBCBackend(env, w.WBCLocomotion.from_cache(), caps)
    goal = chosen.target.hands[0]
    sup = WholeBodySupervisor(planner.pm, hand_goal=(goal.arm, goal.position))
    report = backend.execute(chosen.target, monitor=sup)
    assert report.completed and not report.stops, report.stops
    d, wrist = env.data, env.model.body(f"{goal.arm}_wrist_yaw_link").id
    hand = d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ planner.pm.anatomy.grip[goal.arm]
    error = float(np.linalg.norm(hand - np.asarray(goal.position)))  # evaluator: true state
    print("real hand error (m):", round(error, 4), summary(sup.trace))
    assert error < 0.05
    assert math.isfinite(summary(sup.trace)["peak_torque_ratio"])
