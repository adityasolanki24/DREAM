"""Physics regressions for the fixed-stance grasp using the cached WBC."""

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from dream.robot.g1_wbc import WBCLocomotion, policy_cache
from dream.sim.g1_humanoid import G1Humanoid
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1

pytest.importorskip("mink")
pytest.importorskip("daqp")
pytest.importorskip("onnxruntime")
from dream.robot.g1_posture import PosturePlanner
from dream.robot.g1_supervisor import WholeBodySupervisor
from dream.robot.wholebody import ControllerCapabilities, WBCBackend
from experiments.g1_grasp_bringup import HOLD_S, LIFT_M, TICK, _attempt, attempt

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(
            (policy_cache() / f"GR00T-WholeBodyControl-{mode}.onnx").exists()
            for mode in ("Balance", "Walk")
        ),
        reason="cached WBC weights required",
    ),
]


def test_tabletop_touch_does_not_trigger_a_full_lift():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_both_arms_clear.json").read_text())
    result = attempt(dict(cfg, trajectory_search=False))
    phases = result["phases"]
    assert "raise" in phases or "reach" in phases
    assert "lift" not in phases
    assert not result["grasp_verified"]
    assert not any(
        row["phase"] in {"close", "probe"} and "tabletop" in row["hand_contact_with"]
        for row in result["rows"]
    )


def test_clearance_checks_the_motion_to_a_feasible_endpoint():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_pinch.json").read_text())
    result = attempt(dict(cfg, trajectory_search=False))
    assert result["stop"]["reason"] == "hand clearance"
    assert result["phases"]["raise"]["stops"]
    assert "close" not in result["phases"]
    assert not any(
        "tabletop" in row["hand_contact_with"] or "tabletop" in row["left_hand_contact_with"]
        for row in result["rows"]
    )


@pytest.mark.parametrize("fixture", ["pinch", "lateral"])
def test_verified_free_standing_grasp_on_changed_box_positions(fixture):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(cfg)
    assert result["stop"] is None
    assert result["grasp_observed"], result["phases"]
    assert result["grasp_verified"], result["phases"]
    hold = result["rows"][-round(HOLD_S / TICK) :]
    assert len(hold) == round(HOLD_S / TICK)
    assert all(row["phase"] == "hold" for row in hold)
    assert min(row["lift_m"] for row in hold) >= LIFT_M
    assert min(row["table_clearance_m"] for row in hold) >= LIFT_M
    assert all(row["hand_only_contact"] for row in hold)


def test_geometry_ranked_pinch_retains_the_farther_box():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_fresh_far.json").read_text())
    result = attempt(dict(cfg, contact_geometry_search=True))
    assert result["phases"]["grasp_geometry"]["candidates"] > 1
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize("fixture", ["shift_out", "rectangular"])
def test_rehearsal_recovers_a_failed_geometric_pinch(fixture):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, contact_rehearsal_search=True))
    rank = result["phases"]["grasp_geometry"]["rank"]
    rehearsals = result["phases"]["rehearsal"]
    assert rehearsals[0]["minimum_hold_lift_m"] < LIFT_M
    assert rehearsals[rank]["minimum_hold_lift_m"] >= LIFT_M
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize(("fixture", "preview_holds"), [("pinch", False), ("shift_out", True)])
def test_whole_body_preview_filters_the_proposed_grasp(fixture, preview_holds):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, whole_body_grasp_preview=True))
    preview = result["phases"]["whole_body_grasp_preview"]
    assert preview["proposed_rank"] > 0
    assert preview["attempts"][0]["verified"] is preview_holds
    expected_rank = preview["attempts"][-1]["rank"] if preview["predicted_verified"] else 0
    assert result["phases"]["grasp_geometry"]["rank"] == expected_rank
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


def test_wrap_family_holds_the_bottle_from_geometric_and_physical_selection():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_cylinder.json").read_text())
    cfg.update(
        seed=53,
        object_shape="bottle",
        object_half_extents_m=[0.033, 0.033, 0.064],
        object_mass_kg=0.10,
        object_sliding_friction=0.85,
        table_top_m=0.74,
        object_center_xy_m=[0.465, -0.175],
        whole_body_grasp_preview=True,
        grasp_depth_search=True,
    )
    result = attempt(cfg)
    assert result["phases"]["whole_body_grasp_preview"]["family"] == "proximal_wrap"
    assert result["grasp_observed"]
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize(("fixture", "uses_thumb_bias"), [("cylinder", False), ("shift_in", True)])
def test_expanded_contact_search_retains_the_object(fixture, uses_thumb_bias):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, whole_body_grasp_preview=True, grasp_depth_search=True))
    geometry = result["phases"]["grasp_geometry"]
    assert geometry["offset_m"][0] != 0.0
    assert bool(geometry["thumb_bias_rad"]) is uses_thumb_bias
    assert result["phases"]["whole_body_grasp_preview"]["predicted_verified"]
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


def test_observed_object_loss_stops_despite_pressure_between_fingers():
    # Reproduce the bottle prediction whose tactile pressure persisted after
    # the object slipped to the table. Candidate rank comes from its recorded
    # geometric search, with the same independent planning material and noise.
    cfg = json.loads((CONFIG / "g1_grasp_bringup_cylinder.json").read_text())
    cfg.update(
        seed=np.random.SeedSequence(0, spawn_key=(1,)),
        object_shape="bottle",
        object_half_extents_m=[0.033, 0.033, 0.064],
        object_mass_kg=0.10,
        table_top_m=0.74,
        object_center_xy_m=[0.465, -0.175],
        contact_rehearsal_search=True,
        grasp_depth_search=True,
    )
    result = _attempt(cfg, planned_rank=414)
    assert not result["grasp_verified"]
    assert result["stop"]["reason"] == "object contact lost"


@pytest.mark.parametrize("rotated", [False, True])
def test_walk_to_grasp_reaches_a_selected_stance_and_holds(rotated):
    cfg = json.loads((CONFIG / "g1_walk_grasp.json").read_text())
    heading = 0.0
    if rotated:
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        translation = np.array([-0.4, 0.25])
        for key in ("table_center_xy_m", "object_center_xy_m"):
            cfg[key] = (rotation @ cfg[key] + translation).tolist()
        start = cfg["starting_stance_xy_yaw"]
        heading = np.pi / 2
        cfg["starting_stance_xy_yaw"] = [
            *(rotation @ start[:2] + translation),
            start[2] + heading,
        ]
        cfg["seed"] = 89
    result = attempt(cfg)
    phases = result["phases"]
    assert phases["stance"]["goal"] != tuple(cfg["starting_stance_xy_yaw"])
    assert phases["stance"]["heading"] == pytest.approx(heading)
    walk = phases["walk"]
    assert walk["status"] == "arrived"
    assert walk["true_displacement_m"] >= 1.0
    assert walk["table_contact_samples"] == 0
    assert result["grasp_observed"] and result["grasp_verified"], phases
    assert result["grasp_tool"]["status"] == "succeeded"
    assert result["grasp_tool"]["data"] == {"grasp_observed": True}
    rows = result["rows"]
    assert max(r["t"] for r in rows if r["phase"] == "walk") < min(
        r["t"] for r in rows if r["phase"] == "hold"
    )


def test_cancelled_walking_grasp_preserves_initial_state():
    cfg = json.loads((CONFIG / "g1_walk_grasp.json").read_text())
    cancel = Event()
    cancel.set()
    result = attempt(cfg, cancel=cancel)
    assert not result["grasp_observed"] and not result["grasp_verified"]
    assert result["grasp_tool"]["status"] == "stopped"
    assert result["grasp_tool"]["sim_time"] == 0.0
    assert not result["rows"]


def test_humanoid_moves_each_arm_and_hand_under_the_same_supervisor():
    model = build_mobile_g1(lambda spec: None, head_camera=None, hands=True, touch=True)
    env = G1MobileEnv(model, np.random.default_rng(4), head_camera=None)
    env.reset()
    controller = WBCLocomotion.from_cache()
    caps = ControllerCapabilities.from_envelope_json()
    backend = WBCBackend(env, controller, caps)
    supervisor = WholeBodySupervisor(PosturePlanner(caps).pm)
    robot = G1Humanoid(env, backend, supervisor)
    assert all(robot.step() for _ in range(round(1.0 / TICK)))

    def joint(name):
        state = robot.state()
        return state.joint_pos[state.joint_names.index(name)]

    for side in ("left", "right"):
        elbow = f"{side}_elbow_joint"
        before = joint(elbow)
        assert robot.move_arm(side, np.array([0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0]))
        assert joint(elbow) > before + 0.10
        assert robot.finger_forces(side)
        finger_goal = np.zeros_like(robot.finger_positions(side))
        finger_goal[0] = 0.20
        robot.set_hand(side, finger_goal)
        assert all(robot.step() for _ in range(round(0.5 / TICK)))
        assert robot.finger_positions(side)[0] > 0.10

    robot.hold("test stop")
    now = robot.state().time
    robot.set_velocity(0.15, 0.0, 0.0)
    assert not robot.step()
    assert robot.state().time == now
    assert robot.state().stop_reason == "test stop"


def test_large_tilted_box_cannot_claim_a_hold_from_centre_displacement():
    from experiments.g1_r1_development import episodes

    spec = json.loads((CONFIG / "g1_r1_development.json").read_text())
    cfg = episodes(spec)[0]
    assert cfg["object_shape"] == "box"
    result = attempt(cfg)
    assert not result["grasp_observed"] or result["grasp_verified"], result["phases"]
    for row in result["rows"]:
        if row["phase"] == "hold":
            assert row["observed_clearance_m"] <= row["table_clearance_m"] + 1e-12


@pytest.mark.parametrize("mobile", [False, True])
def test_live_trajectory_forecast_uses_independent_material_prior(monkeypatch, mobile):
    from experiments import g1_grasp_bringup as diagnostic

    cfg = json.loads((CONFIG / "g1_grasp_bringup_pinch.json").read_text())
    cfg.update(
        walk_to_grasp=mobile,
        object_mass_kg=0.27,
        object_sliding_friction=0.53,
        grasp_material_prior={"mass_kg": 0.10, "sliding_friction": 0.80},
    )
    built = []
    original = diagnostic.build_mobile_g1

    def capture_model(*args, **kwargs):
        model = original(*args, **kwargs)
        built.append((float(model.body("box").mass[0]), float(model.geom("box").friction[0])))
        return model

    monkeypatch.setattr(diagnostic, "build_mobile_g1", capture_model)
    cancel = Event()
    cancel.set()
    diagnostic._attempt(cfg, cancel=cancel)
    assert len(built) == 2
    assert built[0] == pytest.approx((0.27, 0.53))
    assert built[1] == pytest.approx((0.10, 0.80))
