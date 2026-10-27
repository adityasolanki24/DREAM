"""W6 corrections (each one a general defect found by the W5 failure attribution).

H3 (zero-candidate plans on reachable targets) was NOT the covariance or the stance grid: it was
the planner's own collision geometry. Regression tests, one per defect:
- the obstacle slots' bounding volumes followed the size, and a box has one canonical description;
- shoulder links are limited against the torso (the arm was planned INTO the torso);
- the approach axis is a preference that a candidate may relax;
- the rollout settles before the motion (it did not: 38 rollouts stopped at 0.1 s);
- torque-aware correction steps;
- anisotropic target covariance and an unseen box top.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dream.perception.detect import Detection
from dream.perception.reach_target import MIN_OCCLUDED_HEIGHT, HandleEstimator
from dream.robot import g1_correction as gc
from dream.robot import g1_posture as gp
from dream.robot import g1_rollout as gr
from dream.robot.wholebody import ControllerCapabilities

WIDE = ControllerCapabilities.unmeasured("hypothetical", (0.2, 0.74), (-0.52,) * 3, (0.52,) * 3)
ONE = gp.PlannerConfig(radii=(0.4,), n_angles=4, max_stances=4)


@pytest.fixture(scope="module")
def planner():
    return gp.PosturePlanner(WIDE, ONE)


# -- obstacle geometry ------------------------------------------------------------------------
def test_canonical_box_puts_the_shorter_horizontal_side_on_local_x_without_moving_the_box():
    half, yaw = gp.canonical_box((0.43, 0.2, 0.7), 1.1)
    assert half == (0.2, 0.43, 0.7) and yaw == pytest.approx(1.1 + math.pi / 2)
    assert gp.canonical_box((0.2, 0.43, 0.7), 2.68) == ((0.2, 0.43, 0.7), 2.68)


def test_one_box_in_two_descriptions_loads_identically_with_matching_bounding_volumes(planner):
    pm = planner.pm
    a = gp.PerceivedBox((1.7, -0.3, 0.6), (0.2, 0.43, 0.6), 2.68, True)
    b = gp.PerceivedBox((1.7, -0.3, 0.6), (0.43, 0.2, 0.6), 2.68 - math.pi / 2, True)
    g = pm.load_obstacles([a])[0]
    size_a, quat_a = pm.model.geom_size[g].copy(), pm.model.geom_quat[g].copy()
    pm.load_obstacles([b])
    assert np.allclose(pm.model.geom_size[g], size_a)
    assert np.allclose(pm.model.geom_quat[g], quat_a)
    # MuJoCo does not refresh bounding volumes when a size is edited at run time
    assert pm.model.geom_rbound[g] == pytest.approx(np.linalg.norm(pm.model.geom_size[g]))
    assert np.allclose(pm.model.geom_aabb[g][3:], pm.model.geom_size[g])
    pm.load_obstacles([])
    assert np.allclose(pm.model.geom_size[g], 0.1)  # a parked slot is a small cube again


def test_a_wide_perceived_fixture_plans_like_the_same_box_described_depth_first(planner):
    """H3: the width-first description of the s5008 fixture gave 0 candidates, depth-first 13+."""
    c, h, yaw = (1.72, -0.34, 0.676), (0.199, 0.436, 0.676), 2.68
    target, stance = [1.549, -0.255, 0.897], (1.15, -0.25, 0.0)
    counts = []
    for half, ang in ((h, yaw), ((h[1], h[0], h[2]), yaw - math.pi / 2)):
        box = gp.PerceivedBox(c, half, ang, True)
        r = planner.plan(target, None, (0, 0, 0), [box], stances=[stance])
        counts.append(len(r.plans))
    assert counts[0] == counts[1] and counts[0] > 0


# -- shoulders against the torso ----------------------------------------------------------------
def test_shoulder_links_are_limited_against_the_torso(planner):
    (links, torso), = planner.pm.shoulder_pairs()
    assert len(links) == 6 and torso
    # a target across the chest needs adduction into the torso: whatever is planned must clear it
    r = planner.plan([0.45, 0.05, 0.9], None, (0, 0, 0), (), stances=[(0.0, 0.0, 0.0)])
    pm, ft = planner.pm, np.zeros(6)
    for p in r.plans:
        d = pm.data_at(p.target.q)
        dist = min(
            mujoco.mj_geomDistance(pm.model, d, a, b, 0.1, ft) for a in links for b in torso
        )
        assert dist >= -1e-3
    # and the check is part of `evaluate`: a configuration with the arm pressed in is rejected
    q = pm.q_nominal.copy()
    q[pm.qadr[16]] = -0.9  # left shoulder roll adducted
    metrics, _, reasons = planner.evaluate(q, (0, 0, 0), "left", [0.0, 0.2, 0.8])
    assert metrics["clearance"] < 0 and "collision" in reasons


# -- approach axis ----------------------------------------------------------------------------
def test_the_approach_axis_is_a_preference_a_candidate_may_relax(planner, monkeypatch):
    real = planner._ik

    def no_axis_ik(q0, stance, arm, target, approach, *a, **k):
        if approach is not None:  # the axis makes this candidate fail (hand misses the target)
            return q0
        return real(q0, stance, arm, target, approach, *a, **k)

    monkeypatch.setattr(planner, "_ik", no_axis_ik)
    r = planner.plan(
        [1.4, 0.0, 0.8], None, (0, 0, 0), (), approach_axis=(1, 0, 0), stances=[(1.0, 0.0, 0.0)]
    )
    assert r.plans and all(p.metrics["axis_relaxed"] == 1.0 for p in r.plans)
    assert all("axis_error" in p.metrics for p in r.plans)


def test_a_plan_that_meets_the_axis_ranks_above_one_that_relaxes_it():
    base = {"deviation": 0.0, "com_margin": 1.0, "joint_margin": 1.0, "torque_ratio": 0.0,
            "hand_residual": 0.0}  # fmt: skip
    near = gp.PosturePlanner._score({**base, "axis_error": 0.0}, 0.0)
    far = gp.PosturePlanner._score({**base, "axis_error": 0.6}, 0.0)
    assert far - near == pytest.approx(gp.W_AXIS * 0.6)


# -- rollout harness ----------------------------------------------------------------------------
def test_the_rollout_settles_before_the_motion():
    steps = []
    cmd = SimpleNamespace(height=0.74, rpy=(0, 0, 0), arm=np.zeros(14))
    ctrl = SimpleNamespace(
        set_command=lambda *a: None, set_posture=lambda *a: None, arm_target=None
    )
    env = SimpleNamespace(step=lambda c: steps.append(c))
    gr.settle(SimpleNamespace(controller=ctrl, command=lambda: cmd, env=env), ticks=7)
    assert len(steps) == 7
    assert gr.SETTLE_TICKS * 0.02 >= 1.0  # at least a second of standing


def test_the_closed_loop_gate_judges_the_final_error_not_the_open_loop_one():
    assert gr.FINAL_HAND_ERROR_MAX <= 0.03 < gr.HAND_ERROR_MAX * 1.0 + 1  # the stated tolerance
    assert gr.FINAL_HAND_ERROR_MAX < gr.HAND_ERROR_MAX  # tighter, but of the corrected error


# -- torque-aware correction -----------------------------------------------------------------------
def _fake_correction(ratio, predicted_after):
    """A correction request with a measured torque ratio and a static-ratio model."""
    arm = np.zeros(14)
    plan_target = SimpleNamespace(
        q=np.zeros(36), torso_rpy=(0, 0, 0), arm_joints=arm, pelvis_height=0.7,
        hands=(SimpleNamespace(position=(0, 0, 0)),),
    )  # fmt: skip
    plan = SimpleNamespace(arm="left", stance=(0, 0, 0), target=plan_target)
    obs = SimpleNamespace(prop=SimpleNamespace(q=np.zeros(29), quat=(1, 0, 0, 0)), pose=(0, 0, 0))
    backend = SimpleNamespace(observe=lambda: obs, execute=lambda *a, **k: pytest.fail("moved"))
    sup = SimpleNamespace(
        measure=lambda o: {
            "ach_hand_left": (0.0, 0.0, 0.0), "ach_height": 0.7, "torque_ratio": ratio,
            "torque_joint": 16,
        }  # fmt: skip
    )
    q_new = np.zeros(36)
    q_new[7:36] = 0.3
    pm = SimpleNamespace(qadr=np.arange(7, 36), kinematic_state=lambda *a: {"rpy": np.zeros(3)})
    planner = SimpleNamespace(
        pm=pm, measured_qpos=lambda *a: np.zeros(36), refine=lambda *a, **k: q_new,
        caps=SimpleNamespace(torso_rpy_min=(-1,) * 3, torso_rpy_max=(1,) * 3),
    )  # fmt: skip
    return backend, sup, planner, plan, predicted_after


def test_a_correction_step_that_would_exceed_the_soft_torque_limit_is_not_executed(monkeypatch):
    backend, sup, planner, plan, _ = _fake_correction(ratio=0.7, predicted_after=None)
    # the static ratio rises with the step: any step is predicted above the soft limit
    monkeypatch.setattr(
        gc, "static_torque_ratio", lambda pm, q: 0.2 + 5.0 * float(np.abs(q[7:36]).max())
    )
    rep = gc.correct_reach(backend, sup, planner, plan, lambda: [0.1, 0.0, 0.0])
    assert [i["action"] for i in rep.iterations] == ["torque_limited"]
    assert rep.iterations[0]["torque_joint"] == 16 and rep.iterations[0]["torque_measured"] == 0.7


# -- perception -----------------------------------------------------------------------------------
def _handle(at=(1.5, 0.0, 0.75)):
    p = np.array(at, float)
    return Detection("handle", "red", p, 0.0, np.array([0.03, 0.09, 0.03]), 0.002, False, False,
                     pixels=60, top_centre=p)  # fmt: skip


def _fixture(occluded):
    return Detection("box", "grey", np.array([1.53, 0.0, 0.5]), 0.0, np.array([0.02, 0.6, 0.9]),
                     0.003, occluded, True, pixels=900)  # fmt: skip


def _estimate(occluded=True):
    est = HandleEstimator("red")
    assert est.ingest([_handle(), _fixture(occluded)], np.zeros(3))
    return est.estimate()


def test_target_covariance_is_larger_along_the_face_than_along_its_normal():
    t = _estimate()
    n = np.array([*t.face_normal, 0.0])
    lat = np.array([-n[1], n[0], 0.0])
    s_n, s_l = (float(v @ t.covariance @ v) ** 0.5 for v in (n, lat))
    assert s_l > 2 * s_n > 0.01  # lateral 15 mm against normal 6 mm (dev-fitted)


def test_a_box_top_at_the_image_border_is_extended_upward_but_a_seen_top_is_not():
    cut = _estimate(occluded=True).fixture
    seen = _estimate(occluded=False).fixture
    assert cut.center[2] + cut.half_extents[2] == pytest.approx(0.9 + MIN_OCCLUDED_HEIGHT)
    assert seen.center[2] + seen.half_extents[2] == pytest.approx(0.9)


# -- protocol v2: evaluator, sensors, config --------------------------------------------------
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from dream.eval import reach as ev  # noqa: E402
from dream.sim.g1_mobile import TOUCH_LINKS, TOUCH_THRESHOLD, build_mobile_g1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = json.loads((ROOT / "experiments/configs/g1_reach_heights_v2.json").read_text())
V1 = json.loads((ROOT / "experiments/configs/g1_reach_heights.json").read_text())
SCENE = SimpleNamespace(handle_center=(1.5, 0.0, 0.75))


def probe(contact_ticks, dist=0.1, force=3.0, collapse=False):
    hand = np.array(SCENE.handle_center) + [dist, 0.0, 0.0]
    return SimpleNamespace(
        hand_points=lambda: {"left": hand + 5.0, "right": hand},
        collapse=lambda: {"collapsed": collapse, "joint_limit_total_s": 0.0},
        peak={"handle_hand": force, "fixture_hand": 0.0, "fixture_other": 0.0, "distractor": 0.0},
        touched={"handle_hand": 1, "fixture_hand": 0, "fixture_other": 0, "distractor": 0},
        min_dist={"left": 9.0, "right": dist},
        handle_contact=[True] * contact_ticks,
    )  # fmt: skip


def test_v2_success_is_sustained_contact_not_distance_to_the_centre():
    near = ev.judge(probe(20, dist=0.02), SCENE, "reached", None, [], "contact")
    assert near["physical_success"] and near["handle_contact_s"] == pytest.approx(0.4)
    # touching 45 mm from the centre (the bar is 90 mm wide): v1 fails it, v2 passes it
    wide = ev.judge(probe(20, dist=0.045), SCENE, "reached", None, [], "contact")
    assert wide["physical_success"] and not wide["success_distance_v1"]
    # a probe 20 mm from the centre that never touched: v1 passes, v2 does not, the claim is false
    air = ev.judge(probe(0, dist=0.02), SCENE, "reached", None, [], "contact")
    assert not air["physical_success"] and air["success_distance_v1"] and air["false_success_claim"]
    short = ev.judge(probe(10), SCENE, "failed", None, [], "contact")  # 0.2 s
    assert not short["physical_success"]
    hard = ev.judge(probe(20, force=40.0), SCENE, "reached", None, [], "contact")
    assert not hard["physical_success"] and not hard["false_success_claim"]  # force limit
    assert not ev.judge(probe(20, collapse=True), SCENE, "reached", None, [], "contact")[
        "physical_success"
    ]
    assert ev.judge(probe(20, dist=0.02), SCENE, "reached", None, [])["success_mode"] == "distance"
    with pytest.raises(ValueError):
        ev.judge(probe(20), SCENE, "reached", None, [], "nonsense")


def test_touch_sensors_are_sites_only_and_leave_the_dynamics_untouched():
    plain, touch = build_mobile_g1(hands=True), build_mobile_g1(hands=True, touch=True)
    assert touch.nsensor == plain.nsensor + 2 * len(TOUCH_LINKS)
    assert touch.nsite == plain.nsite + 2 * len(TOUCH_LINKS)
    for attr in ("ngeom", "nbody", "nu", "nv"):
        assert getattr(touch, attr) == getattr(plain, attr)
    assert np.allclose(touch.body_mass, plain.body_mass)


def test_a_hand_link_pressed_on_a_surface_reports_a_touch_force():
    from dream.sim.g1_mobile import G1MobileEnv

    def at_rest(scene=None):
        m = build_mobile_g1(scene=scene, hands=True, touch=True)
        env = G1MobileEnv(m, np.random.default_rng(0), head_camera=None)
        env.reset(0.0, 0.0, 0.0)
        mujoco.mj_forward(m, env.data)
        return m, env

    m, env = at_rest()
    assert max(env.touch("right").values()) == 0.0
    tip = env.data.xpos[m.body("right_hand_index_1_link").id].copy()

    def box(spec):  # a thin plate through the fingertip link
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(tip), size=[0.05, 0.05, 0.004]
        )

    m2, env2 = at_rest(box)
    assert max(env2.touch("right").values()) > 0.0 and max(env2.touch("left").values()) == 0.0
    from dream.robot import g1_reach_ports as ports

    assert ports.TOUCH_THRESHOLD == TOUCH_THRESHOLD


def test_v2_config_changes_only_what_it_states_and_keeps_v1_gates():
    assert V2["backend"] == "wbc" and V2["success"]["mode"] == "contact" and V2["touch_sensors"]
    assert V2["success"]["contact_min_s"] == ev.CONTACT_MIN_S == 0.3
    assert V2["success"]["peak_contact_force_max_n"] == ev.CONTACT_FORCE_MAX
    assert V2["gates"] == V1["gates"]  # same gate values, new success definition
    assert V2["heldout"]["seeds"] == list(range(6000, 6024))
    for k in ("height_m", "distance_m", "lateral_m", "yaw_jitter_deg", "max_crates",
              "second_fixture_prob"):  # fmt: skip
        assert V2["heldout"][k] == V1["heldout"][k]  # same distributions as v1
    assert V2["dev"]["seeds"] == V1["dev"]["seeds"] and V2["dev"]["heights"] == V1["dev"]["heights"]
    assert V2["dev"]["v1_heldout_as_dev"]["seeds"] == V1["heldout"]["seeds"]
    assert set(V2["heldout"]["seeds"]).isdisjoint(V1["heldout"]["seeds"] + V1["dev"]["seeds"])
    assert hashlib.sha256((ROOT / V2["envelope"]).read_bytes()).hexdigest() == V2["envelope_sha256"]
    assert V2["status"] == "draft" and "changes_from_v1" in V2
    assert V2["dev"]["results_dir"] != V1["dev"]["results_dir"]


def test_the_v2_development_set_has_the_v1_dev_and_the_v1_heldout_scenes():
    import sys

    sys.path.insert(0, str(ROOT))
    from experiments.g1_reach_heights import heldout_specs, specs

    dev = specs(V2, "dev")
    assert len(dev) == 36 + 24
    v1ho = {s["seed"]: s for s in heldout_specs(V1)}
    ported = [s for s in dev if s["id"].startswith("dev_v1ho_")]
    assert len(ported) == 24
    for s in ported:
        t = v1ho[s["seed"]]
        keys = ("height", "distance", "lateral")
        assert [s[k] for k in keys] == [t[k] for k in keys]
    assert {s["seed"] for s in specs(V2, "heldout")} == set(range(6000, 6024))
