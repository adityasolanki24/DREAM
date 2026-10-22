import numpy as np
import pytest

from dream.sim.env import MechanismEnv, SensorNoise
from dream.sim.mechanisms import FAMILIES, sample_mechanism
from dream.types import Probe


def _grasped_env(family: str, seed: int = 0, noise: SensorNoise | None = None) -> MechanismEnv:
    rng = np.random.default_rng(seed)
    env = MechanismEnv(sample_mechanism(family, rng), rng, noise=noise)
    env.move_ee_to(env.geometry().handle_pos)
    assert env.grasp()
    return env


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_compiles_and_grasps(family):
    _grasped_env(family)


def test_grasp_fails_when_end_effector_is_far():
    rng = np.random.default_rng(0)
    env = MechanismEnv(sample_mechanism("drawer", rng), rng)
    assert not env.grasp()
    with pytest.raises(RuntimeError):
        env.execute_probe(Probe(np.array([0.0, -1.0, 0.0]), 0.05, 10.0))


def test_drawer_moves_along_its_axis_only():
    env = _grasped_env("drawer")
    axis = env.privileged_spec().axis
    r = env.execute_probe(Probe(axis, 0.06, 15.0))
    q = env.privileged_joint_position()
    assert 0.045 < q < 0.065, "compliant drag should get close to the commanded 6 cm"
    dp = r.after.handle_pos - r.before.handle_pos
    assert np.dot(dp, axis) > 0.04
    # Pulling sideways must not move a drawer.
    env.execute_probe(Probe(np.array([1.0, 0.0, 0.0]), 0.06, 15.0))
    assert abs(env.privileged_joint_position() - q) < 2e-3


def test_drawer_cannot_be_pushed_past_closed_stop():
    env = _grasped_env("drawer")
    env.execute_probe(Probe(-env.privileged_spec().axis, 0.05, 15.0))
    assert env.privileged_joint_position() < 2e-3


def test_fixed_panel_does_not_move_and_robot_force_saturates():
    env = _grasped_env("fixed", noise=SensorNoise(0, 0, 0, 0))
    r = env.execute_probe(Probe(np.array([0.0, -1.0, 0.0]), 0.08, 12.0))
    assert np.linalg.norm(r.after.handle_pos - r.before.handle_pos) < 3e-3
    assert r.after.ee_force[1] == pytest.approx(-12.0, abs=0.2)


@pytest.mark.parametrize("family", ["left_hinge", "right_hinge", "bottom_hinge", "top_hinge"])
def test_hinge_opens_when_pulled_out_and_respects_limits(family):
    env = _grasped_env(family)
    env.execute_probe(Probe(np.array([0.0, -1.0, 0.0]), 0.08, 15.0))
    q = env.privileged_joint_position()
    assert q > 0.05
    assert q <= env.privileged_spec().upper_limit + 0.02


def test_weak_probe_below_friction_does_not_move_drawer():
    rng = np.random.default_rng(3)
    spec = sample_mechanism("drawer", rng, friction_range=(3.0, 3.0))
    env = MechanismEnv(spec, rng)
    env.move_ee_to(env.geometry().handle_pos)
    assert env.grasp()
    env.execute_probe(Probe(spec.axis, 0.05, 2.0))
    assert env.privileged_joint_position() < 2e-3


def test_observation_noise_matches_configuration():
    rng = np.random.default_rng(1)
    env = MechanismEnv(sample_mechanism("fixed", rng), rng, noise=SensorNoise(handle_pos_std=0.002))
    samples = np.array([env.observe().handle_pos for _ in range(2000)])
    assert samples.std(axis=0) == pytest.approx([0.002] * 3, rel=0.1)
    assert np.allclose(samples.mean(axis=0), env.geometry().handle_pos, atol=3e-4)
