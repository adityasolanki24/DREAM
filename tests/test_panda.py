"""Contracts for the Panda embodiment, its safety layer and the contact-based skills."""

import mujoco
import numpy as np
import pytest

import dream.skills.manipulation as M
from dream.robot.panda import SafetyLimits
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv


def _env(seed=0, **kw):
    return TabletopEnv(
        sample_tabletop(np.random.default_rng(seed)), np.random.default_rng(seed + 1), **kw
    )


def test_scene_has_no_attachment_constraints():
    """Grasps must come from contact: the only equality is Menagerie's finger mimic joint."""
    env = _env()
    m = env.model
    assert m.neq == 1
    assert m.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    names = {m.joint(m.eq_obj1id[0]).name, m.joint(m.eq_obj2id[0]).name}
    assert names == {"finger_joint1", "finger_joint2"}


@pytest.mark.parametrize(
    "target,yaw", [((0.5, 0.0, 0.3), 0.0), ((0.4, -0.25, 0.05), -1.2), ((0.35, 0.25, 0.18), 1.5)]
)
def test_reach_converges_including_near_shoulder_singularity(target, yaw):
    env = _env()
    r = M.move_to(env.embodiment(), np.array(target), yaw)
    assert r.ok, r.reason
    pe, re = env.robot.tracking_error()
    assert pe < M.POS_TOL and re < M.ROT_TOL


def test_targets_outside_workspace_are_refused():
    env = _env()
    r = M.move_to(env.embodiment(), np.array([0.5, 0.0, -0.05]), 0.0)
    assert not r.ok and "workspace" in r.reason


def test_pressing_into_the_table_latches_a_protective_stop():
    env = _env(limits=SafetyLimits(workspace_lo=(0.2, -0.45, -0.1)))
    r = M.move_to(env.embodiment(), np.array([0.5, 0.0, -0.05]), 0.0)
    assert not r.ok and "protective stop" in r.reason
    assert env.robot_state().protective_stop
    q = env.robot_state().joint_pos
    env.step(500)  # the stopped controller holds still
    assert np.max(np.abs(env.robot_state().joint_pos - q)) < 1e-2


def test_pick_and_place_by_contact():
    env = _env(seed=3)
    robot = env.embodiment()
    r = M.pick(robot, env, "red")
    assert r.ok, r.reason
    assert 0.035 < robot.state().gripper_width < 0.045  # fingers stopped on the 4 cm cube
    assert env.privileged_object_pos("red")[2] > 0.15  # really lifted, by friction alone
    centre, half = env.tray_region()
    r = M.place(robot, env, "red", centre + [0, 0, 0.02])
    assert r.ok, r.reason
    final = env.privileged_object_pos("red")
    assert np.all(np.abs(final[:2] - centre[:2]) < half)


class _MisleadingPerception:
    """Reports an object 6 cm from where it really is (e.g. a perception error)."""

    def __init__(self, env):
        self.env = env

    def observe_objects(self):
        return [
            type(o)(o.name, o.color, o.position + [0.06, 0, 0], o.yaw, o.size)
            for o in self.env.observe_objects()
        ]

    def fixtures(self):
        return self.env.fixtures()


def test_grasp_on_nothing_is_reported_as_a_miss():
    env = _env(seed=3)
    r = M.pick(env.embodiment(), _MisleadingPerception(env), "red")
    assert not r.ok and "missed" in r.reason


def test_grip_is_firm_and_cancels_in_the_external_force_estimate():
    """The stiffer grip squeezes ~20 N, yet the wrist-force estimate only sees the cube's weight
    (the opposing finger forces cancel), so grasping is not mistaken for a collision."""
    env = _env(seed=3)
    assert M.pick(env.embodiment(), env, "red").ok
    assert abs(env.data.actuator_force[7]) > 15.0
    f = env.robot_state().external_force
    assert abs(f[2] + 0.05 * 9.81) < 0.15 and np.linalg.norm(f[:2]) < 0.3


def test_off_centre_close_does_not_trip_the_safety_stop():
    """Regression (seed 31): closing straight to full force shoved the cube with one finger."""
    env = TabletopEnv(sample_tabletop(np.random.default_rng(31)), np.random.default_rng(1031))
    r = M.pick(env.embodiment(), env, "blue")
    assert r.ok, r.reason
