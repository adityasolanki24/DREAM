"""Contracts for the compartment scene, the handle skills and the probe model on the Panda."""

import mujoco
import numpy as np
import pytest

import dream.skills.articulation as A
from dream.agent.model import ProbeController, predict_outcome, relative_rotvec
from dream.eval.truth import compartment_to_hypothesis
from dream.sim.tabletop import sample_compartment_scene
from dream.sim.tabletop_env import TabletopEnv

OUT_FORCE = 12.0  # N


def _env(kind, seed=0):
    spec = sample_compartment_scene(kind, np.random.default_rng(seed))
    return spec, TabletopEnv(spec, np.random.default_rng(seed + 1))


def _joint_q(env):
    return float(env.data.joint("compartment_joint").qpos[0])


def _front(env):
    """The visible front direction, from where the handle sits relative to the box (no truth)."""
    obs = A.find_articulated(env, "compartment")
    d = obs.handle_pos - obs.box_centre
    d[2] = 0.0
    return d / np.linalg.norm(d)


@pytest.mark.parametrize("kind", ["drawer", "lid", "fixed"])
def test_compartment_adds_no_constraints_and_closed_cavity_is_not_visible(kind):
    _, env = _env(kind)
    assert env.model.neq == 1  # only Menagerie's finger mimic
    assert "compartment" not in env.fixtures()
    has_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "compartment_joint") >= 0
    assert has_joint == (kind != "fixed")


def test_closed_roof_physically_blocks_a_dropped_cube():
    """A cube dropped onto the closed drawer rests on the roof, not in the cavity."""
    spec, env = _env("drawer")
    c = spec.compartment
    j = env.model.joint("red_free")
    adr = j.qposadr[0]
    env.data.qpos[adr : adr + 3] = [*c.xy, c.height + 0.05]  # scene setup, before any agent acts
    mujoco.mj_forward(env.model, env.data)
    env.step(1000)
    assert env.data.qpos[adr + 2] > c.height  # on top of the roof


@pytest.mark.slow
def test_drawer_grasp_and_probes_follow_the_hidden_slide():
    """Pulling out moves the handle along the slide; sideways is blocked; force is capped."""
    _, env = _env("drawer")
    robot = env.embodiment()
    r = A.grasp_handle(robot, env, "compartment")
    assert r.ok, r.reason
    out = A.probe(robot, env, "compartment", _front(env), 0.06, OUT_FORCE)
    assert out.ok, out.reason
    assert np.linalg.norm(out.handle_after - out.handle_before) > 0.03
    assert _joint_q(env) > 0.03
    side = np.cross([0, 0, 1.0], _front(env))
    blocked = A.probe(robot, env, "compartment", side, 0.06, OUT_FORCE)
    assert blocked.ok
    assert np.linalg.norm(blocked.handle_after - blocked.handle_before) < 0.006
    # The admittance caps the push: measured hand force stays near the limit, far below 40 N.
    assert np.linalg.norm(blocked.force_after) < OUT_FORCE * 1.3
    assert blocked.slip < 0.02
    assert A.release_handle(robot).ok
    assert robot.state().gripper_width > 0.03


@pytest.mark.slow
def test_fixed_box_does_not_move_under_capped_probes():
    _, env = _env("fixed", seed=2)
    robot = env.embodiment()
    assert A.grasp_handle(robot, env, "compartment").ok
    up_out = _front(env) + [0, 0, 1.0]
    for d in (_front(env), up_out / np.linalg.norm(up_out)):
        o = A.probe(robot, env, "compartment", d, 0.08, 15.0)
        assert o.ok, o.reason
        assert np.linalg.norm(o.handle_after - o.handle_before) < 0.006
        assert np.linalg.norm(o.force_after) < 15.0 * 1.3


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["drawer", "lid"])
def test_quasi_static_model_predicts_panda_probes_with_true_parameters(kind):
    """The gantry's probe model, with the Panda's norm-capped spring, predicts real probes.

    Measured over 32 random probes per kind (4 scenes): median 2 mm, max 10 mm position error,
    median 1.3 deg rotation error. The gate here is median < 5 mm and max < 15 mm.
    """
    spec, env = _env(kind, seed=1)
    robot = env.embodiment()
    ref = A.observe_handle_averaged(env, "compartment")
    assert A.grasp_handle(robot, env, "compartment").ok
    truth = compartment_to_hypothesis(spec.compartment)
    ctrl = ProbeController(A.PROBE_STIFFNESS, norm_clip=True)
    up = np.array([0, 0, 1.0])
    rng = np.random.default_rng(7)
    dirs = [_front(env), up, -_front(env)] + list(rng.normal(size=(3, 3)))
    errs, moved = [], []
    for d in dirs:
        d = np.asarray(d) / np.linalg.norm(d)
        o = A.probe(robot, env, "compartment", d, 0.06, 12.0)
        assert o.ok, o.reason
        y, _, _ = predict_outcome(
            truth, ref.handle_pos, o.handle_before, o.handle_before, d, np.array(0.06), 12.0, ctrl
        )
        errs.append(np.linalg.norm(y[0, :3] - o.handle_after))
        moved.append(np.linalg.norm(o.handle_after - o.handle_before))
        rot = relative_rotvec(ref.handle_quat, o.handle_quat_after)
        assert np.linalg.norm(y[0, 3:] - rot) < np.deg2rad(6)
    assert max(moved) > 0.02  # the set includes probes that really move the mechanism
    assert np.median(errs) < 0.005 and max(errs) < 0.015


def _run_task(kind, seed):
    from dream.runtime.agent import TaskAgent
    from dream.tools.goals import InRegion
    from dream.tools.toolbox import Toolbox

    spec, env = _env(kind, seed)
    tb = Toolbox(env.embodiment(), env, rng=np.random.default_rng(seed))
    ep = TaskAgent(tb).run(InRegion("red", "compartment"))
    env.step(250)
    return env, ep


@pytest.mark.slow
def test_drawer_task_investigates_opens_places_and_is_true_in_the_world():
    from dream.eval.manipulation import object_in_compartment

    env, ep = _run_task("drawer", 3)
    tools = [s.call.name for s in ep.steps]
    assert tools[0] == "investigate_physics" and "place" in tools
    assert ep.success_claimed and object_in_compartment(env, "red")
    assert not env.robot.protective_stop


@pytest.mark.slow
def test_fixed_box_is_declined_without_any_attempt_to_force_it_open():
    env, ep = _run_task("fixed", 3)
    assert not ep.success_claimed and "not forcing it" in ep.reason
    assert [s.call.name for s in ep.steps] == ["investigate_physics"]
    assert env.robot.peak_finger_force < 40.0
