"""Scene generator, config contract, evaluator and the general fixes made for the reach test.

Fast tests: no controller, no episode. The one physics episode is in test_reach_episode.py.
"""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dream.eval import reach as ev
from dream.robot import g1_posture as gp
from dream.robot.g1_rollout import COM_MARGIN_MAX_VIOLATION_S, longest_violation
from dream.robot.wholebody import TICK, ControllerCapabilities, torso_rpy_from_matrices
from dream.sim.reach_scenes import HANDLE_HALF, MIN_FIXTURE_ABOVE, sample_reach_scene

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "experiments/configs/g1_reach_heights.json").read_text())


# -- scenes --------------------------------------------------------------------------------------
@pytest.mark.parametrize("h", [0.10, 0.35, 0.75, 1.10, 1.45])
def test_scene_puts_the_handle_on_the_front_face_at_the_requested_height(h):
    s = sample_reach_scene(7, h, 1.7, -0.3, math.radians(15), n_crates=2, second_fixture=True)
    assert s.handle_center[2] == pytest.approx(h)
    top = 2 * s.fixture.half[2]
    assert top >= h + MIN_FIXTURE_ABOVE - 1e-9 and top >= 0.55
    n = np.array(s.face_normal)
    face = np.array(s.fixture.center[:2]) + n * s.fixture.half[0]
    hc = np.array(s.handle_center[:2])
    assert (hc - face) @ n == pytest.approx(HANDLE_HALF[0], abs=1e-9)  # protrudes from the face
    assert np.linalg.norm(face - [1.7, -0.3]) < 1e-9  # the face is where the config put it
    to_start = -np.array([1.7, -0.3]) / np.linalg.norm([1.7, -0.3])
    dyaw = math.atan2(n[1], n[0]) - math.atan2(to_start[1], to_start[0]) - math.radians(15)
    assert abs((dyaw + math.pi) % (2 * math.pi) - math.pi) < 1e-6


def test_scene_is_deterministic_and_varies_with_the_seed():
    a = sample_reach_scene(11, 0.5, 1.5, 0.0, 0.0)
    b = sample_reach_scene(11, 0.5, 1.5, 0.0, 0.0)
    c = sample_reach_scene(12, 0.5, 1.5, 0.0, 0.0)
    assert a == b and a.fixture.half != c.fixture.half


def test_distractors_keep_clear_of_the_target_face_and_the_start():
    for seed in range(20):
        s = sample_reach_scene(seed, 0.8, 1.6, 0.2, 0.1, n_crates=2, second_fixture=True)
        for d in s.distractors:
            if d.name == "distractor_handle":
                continue
            p = np.array(d.center[:2])
            assert np.linalg.norm(p - s.handle_center[:2]) > 0.9
            assert np.linalg.norm(p) > 0.7
        assert s.handle_color not in {
            tuple(d.rgba[:3]) for d in s.distractors if d.name == "distractor_handle"
        }


def test_scene_compiles_into_the_robot_model_with_hands():
    from dream.sim.g1_mobile import build_mobile_g1
    from dream.sim.reach_scenes import add_reach_scene

    s = sample_reach_scene(3, 0.9, 1.5, 0.1, 0.1, n_crates=1, second_fixture=True)
    m = build_mobile_g1(scene=lambda spec: add_reach_scene(spec, s), hands=True)
    assert m.geom("reach_handle").id >= 0 and m.geom("fixture_body").id >= 0
    assert m.nu == 43


# -- the frozen config ---------------------------------------------------------------------------
def test_config_is_pre_registered_and_splits_are_disjoint():
    dev, held = CFG["dev"], CFG["heldout"]
    assert set(dev["seeds"]).isdisjoint(held["seeds"])
    assert dev["heights"] == [0.35, 0.75, 1.10] and len(dev["layouts"]) == 2
    assert held["seeds"] == list(range(5000, 5024)) and held["height_m"] == [0.10, 1.45]
    assert held["distance_m"] == [1.2, 2.2] and held["lateral_m"] == [-0.4, 0.4]
    assert held["yaw_jitter_deg"] == 20.0
    g = CFG["gates"]
    assert (g["max_collapses"], g["max_false_success_claims"]) == (0, 0)
    assert (g["success_feasible_min"], g["success_feasible_off_band_min"]) == (0.80, 0.70)
    assert g["correct_refusal_min"] == 0.90 and CFG["success_radius_m"] == ev.SUCCESS_RADIUS == 0.03


def test_config_pins_the_v2_envelope_by_sha256():
    p = ROOT / CFG["envelope"]
    assert p.parts[-2] == "g1_wbc_envelope_v2"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CFG["envelope_sha256"]


def test_heldout_specs_are_drawn_from_the_stated_ranges_and_deterministic():
    import sys

    sys.path.insert(0, str(ROOT))
    from experiments.g1_reach_heights import dev_specs, heldout_specs

    h = heldout_specs(CFG)
    assert h == heldout_specs(CFG) and len(h) == 24
    d = CFG["heldout"]
    for s in h:
        assert d["height_m"][0] <= s["height"] <= d["height_m"][1]
        assert d["distance_m"][0] <= s["distance"] <= d["distance_m"][1]
        assert d["lateral_m"][0] <= s["lateral"] <= d["lateral_m"][1]
        assert abs(s["yaw_jitter"]) <= math.radians(d["yaw_jitter_deg"])
    hs = [s["height"] for s in h]
    assert min(hs) < 0.4 and max(hs) > 1.2  # the span is actually exercised
    dv = dev_specs(CFG)
    assert len(dv) == 36 and {s["height"] for s in dv} == {0.35, 0.75, 1.10}
    assert len({s["id"] for s in dv + h}) == 60


# -- measured capabilities -----------------------------------------------------------------------
def test_v2_envelope_loader_gives_the_measured_box_not_the_input_bound():
    c = ControllerCapabilities.from_envelope_v2()
    assert c.measured and c.executable and "g1_wbc_envelope_v2" in c.source
    assert c.height_range == (0.62, 0.74)  # 0.50 m failed in every trial of every arm pose
    assert c.torso_rpy_min == (0.0, 0.0, 0.0) and c.torso_rpy_max == (0.0, 0.4, 0.0)
    assert c.gaze_pitch[0] == -0.3 and c.gaze_pitch[1] == 0.4
    assert c.violations(0.55, (0, 0.1, 0)) and not c.violations(0.62, (0, 0.4, 0))
    assert c.violations(0.7, (0, -0.1, 0))  # backward lean is measured only for looking


# -- planner additions ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planner():
    caps = ControllerCapabilities.from_envelope_v2()
    return gp.PosturePlanner(caps, gp.PlannerConfig(radii=(0.45,), n_angles=4, max_stances=2))


def test_explicit_stance_and_arm_restrict_the_search(planner):
    r = planner.plan([0.6, 0.0, 0.8], stances=[(0.15, 0.0, 0.0)], arms=("left",))
    assert r.feasible and {p.arm for p in r.plans} == {"left"}
    assert {p.stance for p in r.plans} == {(0.15, 0.0, 0.0)} and r.n_stances_blocked == 0


def test_retarget_moves_only_the_arm_and_keeps_legs_height_and_torso(planner):
    plan = planner.plan([0.6, 0.0, 0.8], stances=[(0.15, 0.0, 0.0)], arms=("right",)).plans[0]
    pre = planner.retarget(plan, [0.54, 0.0, 0.8])
    assert pre is not None and pre.stance == plan.stance and pre.arm == plan.arm
    assert pre.pelvis_height == pytest.approx(plan.pelvis_height, abs=5e-3)
    assert np.allclose(pre.torso_rpy, plan.torso_rpy, atol=0.03)
    assert np.allclose(pre.target.q[7:19], plan.target.q[7:19], atol=0.05)  # legs held
    assert np.linalg.norm(pre.target.hands[0].position - np.array([0.54, 0.0, 0.8])) < 1e-9
    assert planner.retarget(plan, [3.0, 0.0, 0.8]) is None  # out of reach: refused, not forced
    d = planner.pm.data_at(pre.target.q)
    rpy = torso_rpy_from_matrices(
        d.xmat[planner.pm.pelvis].reshape(3, 3), d.xmat[planner.pm.torso].reshape(3, 3)
    )
    assert np.allclose(pre.torso_rpy, rpy)


def test_plans_keep_every_joint_clear_of_the_limits_the_supervisor_stops_at(planner):
    """Regression (dev_h1.10_L1_s101/_s102): plans ended AT a joint limit and the supervisor
    (stops 0.02 rad from a limit) stopped them on the way."""
    from dream.robot.g1_supervisor import JOINT_MARGIN

    assert gp.JOINT_PLAN_MARGIN > JOINT_MARGIN
    for z in (0.8, 1.1, 1.3):
        for p in planner.plan([0.6, 0.0, z], stances=[(0.2, 0.0, 0.0)]).plans:
            assert p.metrics["joint_clearance"] >= gp.JOINT_PLAN_MARGIN - 1e-3
            q = p.target.q[planner.pm.qadr]
            lo, hi = planner.pm.joint_range[:, 0], planner.pm.joint_range[:, 1]
            assert np.all(q - lo >= JOINT_MARGIN) and np.all(hi - q >= JOINT_MARGIN)


def test_reach_point_is_the_fingertip_not_a_link_origin(planner):
    """The point that touches must be at the end of the finger: the old mean of two link origins
    lay about 5 cm short of the fingertips, so no surface could be touched within 3 cm."""
    pm = planner.pm
    d = pm.data_at(pm.q_nominal)
    w = pm.model.body("right_wrist_yaw_link").id
    tip = d.xpos[w] + d.xmat[w].reshape(3, 3) @ pm.anatomy.grip["right"]
    m = pm.model
    body = m.body("right_hand_index_1_link").id
    origin = d.xpos[body]
    assert np.linalg.norm(tip - origin) > 0.04
    axis = origin - d.xpos[m.body("right_hand_index_0_link").id]
    axis /= np.linalg.norm(axis)
    reach = -np.inf
    for g in np.flatnonzero(m.geom_bodyid == body):
        if m.geom_type[g] != 7 or not m.geom_contype[g]:  # collision meshes
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        reach = max(reach, float(np.max((pts - tip) @ axis)))
    assert reach < 0.006  # no part of the finger lies more than 6 mm beyond the reach point


# -- rollout: a transient is not a loss of balance -----------------------------------------------
def test_com_margin_gate_tolerates_a_one_tick_transient_but_not_a_sustained_loss():
    ok = [0.05, 0.04, -0.04, 0.04, 0.05]  # the measured case: one tick at -0.04 m then settled
    bad = [0.05] + [-0.03] * 8 + [0.05]
    assert longest_violation(ok) * TICK <= COM_MARGIN_MAX_VIOLATION_S
    assert longest_violation(bad) * TICK > COM_MARGIN_MAX_VIOLATION_S
    assert longest_violation([]) == 0


# -- the evaluator -------------------------------------------------------------------------------
def test_wilson_interval_matches_known_values():
    lo, hi = ev.wilson(5, 5)
    assert 0.56 < lo < 0.58 and hi == pytest.approx(1.0)
    assert ev.wilson(0, 0) == [0.0, 1.0]
    lo, hi = ev.wilson(12, 24)
    assert 0.30 < lo < 0.32 and 0.68 < hi < 0.70


def fake_probe(dist=0.02, collapse=False, peak=None):
    hand = np.array([1.5, 0.0, 0.75]) + np.array([dist, 0.0, 0.0])
    p = SimpleNamespace(
        hand_points=lambda: {"left": hand + 5.0, "right": hand},
        collapse=lambda: {"collapsed": collapse, "joint_limit_total_s": 0.0},
        peak={
            "handle_hand": 3.0, "fixture_hand": 0.0, "fixture_other": 0.0, "distractor": 0.0,
            **(peak or {}),
        },
        touched={"handle_hand": 10, "fixture_hand": 0, "fixture_other": 0, "distractor": 0},
        min_dist={"left": 9.0, "right": dist},
    )  # fmt: skip
    return p


SCENE = SimpleNamespace(handle_center=(1.5, 0.0, 0.75))


def test_success_needs_distance_no_collapse_and_no_unresolved_stop():
    j = ev.judge(fake_probe(0.02), SCENE, "reached", None, [])
    assert j["physical_success"] and j["claimed_success"] and not j["false_success_claim"]
    assert j["true_distance_m"] == pytest.approx(0.02)  # the nearer hand counts
    assert not ev.judge(fake_probe(0.031), SCENE, "failed", None, [])["physical_success"]
    assert not ev.judge(fake_probe(0.02, collapse=True), SCENE, "reached", None, [])[
        "physical_success"
    ]
    held = SimpleNamespace(reason="x")
    j = ev.judge(fake_probe(0.02), SCENE, "failed", held, [(1.0, "pelvis tilt")])
    assert not j["physical_success"] and j["unresolved_protective_stop"]
    assert j["protective_stops"] == [{"t": 1.0, "reason": "pelvis tilt"}]


def test_false_success_claim_is_a_claim_the_truth_contradicts():
    j = ev.judge(fake_probe(0.05), SCENE, "reached", None, [])
    assert j["false_success_claim"] and not j["physical_success"]
    assert not ev.judge(fake_probe(0.05), SCENE, "failed", None, [])["false_success_claim"]
    assert not ev.judge(fake_probe(0.05), SCENE, "refused", None, [])["false_success_claim"]


def test_unintended_contacts_are_reported_apart_from_the_intended_one():
    j = ev.judge(fake_probe(0.02, peak={"fixture_other": 12.0}), SCENE, "reached", None, [])
    assert j["contacts"]["peak_force_n"]["handle_hand"] == 3.0
    probe = fake_probe(0.02)
    probe.touched["fixture_other"] = 4
    j = ev.judge(probe, SCENE, "reached", None, [])
    assert j["contacts"]["unintended_contact"]
    assert not ev.judge(fake_probe(0.02), SCENE, "reached", None, [])["contacts"][
        "unintended_contact"
    ]


def test_longest_run_is_in_seconds():
    assert ev.longest([False, True, True, True, False, True]) == pytest.approx(3 * ev.TICK)
    assert ev.longest([]) == 0


def test_summary_counts_refusals_against_the_reference_label():
    import sys

    sys.path.insert(0, str(ROOT))
    from experiments.g1_reach_heights import summarise

    def row(h, verdict, feasible, success, claim_ok=True, collapsed=False):
        return {
            "spec": {"id": f"h{h}{verdict}{feasible}", "height": h}, "error": None,
            "agent": {"verdict": verdict},
            "timing": {"sim_s": 10.0, "wall_s": 20.0},
            "evaluator": {
                "reference_feasible": feasible, "physical_success": success,
                "claimed_success": success and verdict == "reached",
                "collapse": {"collapsed": collapsed, "tilt_latch": False,
                             "undesired_ground_contact": False, "foot_support_loss": False,
                             "joint_limit_contact": False},
                "joint_limit_contact_s": 0.0,
                "false_success_claim": verdict == "reached" and not success,
                "unresolved_protective_stop": False, "protective_stops": [],
                "contacts": {"unintended_contact": False},
                "refusal_correct": verdict == "refused" and not feasible,
                "missed_opportunity": verdict == "refused" and feasible,
                "height_band": "low" if h < 0.55 else "high" if h > 1.15 else "mid",
            },
        }  # fmt: skip

    rows = [
        row(0.1, "refused", False, False), row(0.12, "refused", False, False),
        row(0.4, "reached", True, True), row(0.45, "refused", True, False),
        row(0.8, "reached", True, True), row(1.3, "reached", True, True),
    ]  # fmt: skip
    s = summarise(rows, CFG["gates"])
    assert s["correct_refusals_on_infeasible"]["k"] == 2 and s["missed_opportunities"] == 1
    assert s["success_reference_feasible"]["k"] == 3 and s["success_reference_feasible"]["n"] == 4
    assert s["success_reference_feasible_off_band"]["n"] == 3  # 0.4, 0.45 and 1.3
    assert s["gates"]["zero_collapses"]["pass"] and s["gates"]["zero_false_success_claims"]["pass"]
    assert not s["gates"]["success_on_reference_feasible_ge"]["pass"]  # 0.75 < 0.80
    rows[0] = row(0.1, "reached", False, False)
    assert not summarise(rows, CFG["gates"])["gates"]["zero_false_success_claims"]["pass"]
