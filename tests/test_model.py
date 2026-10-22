import numpy as np
import pytest

from dream.agent.model import (
    FIXED,
    PRISMATIC,
    REVOLUTE,
    Hypotheses,
    handle_at,
    predict_outcome,
    project,
    relative_rotvec,
    rotate,
)
from dream.eval.truth import spec_to_hypothesis
from dream.sim.env import MechanismEnv, SensorNoise
from dream.sim.mechanisms import FAMILIES, sample_mechanism
from dream.types import Probe

H0 = np.array([0.1, -0.05, 0.6])


def _hyp(kind, axis, origin=(0.0, 0.0, 0.0), upper=1.0, friction=1.0) -> Hypotheses:
    return Hypotheses(
        kind=np.array([kind]),
        axis=np.array([axis], dtype=float) / np.linalg.norm(axis),
        origin=np.array([origin], dtype=float),
        upper=np.array([upper]),
        friction=np.array([friction]),
    )


def _predict(h, direction, distance=0.05, force=15.0):
    y, q, _ = predict_outcome(h, H0, H0, H0, np.array(direction, float), np.array(distance), force)
    return y[0], q[0]


def test_rotate_quarter_turn():
    out = rotate(np.array([[0, 0, 1.0]]), np.array([np.pi / 2]), np.array([[1.0, 0, 0]]))
    assert np.allclose(out, [[0, 1, 0]], atol=1e-12)


def test_project_inverts_handle_at():
    rng = np.random.default_rng(0)
    for kind in (PRISMATIC, REVOLUTE):
        axis = rng.normal(size=3)
        h = _hyp(kind, axis, origin=(-0.1, -0.03, 0.6), upper=2.0)
        q = np.array([0.37])
        assert project(h, handle_at(h, q, H0), H0) == pytest.approx(q, abs=1e-9)


def test_fixed_never_moves():
    y, q = _predict(_hyp(FIXED, (1, 0, 0)), (0, -1, 0))
    assert np.allclose(y[:3], H0) and q == 0


def test_prismatic_moves_along_axis_and_stops_short_by_friction():
    y, q = _predict(_hyp(PRISMATIC, (0, -1, 0), friction=2.0), (0, -1, 0), distance=0.05)
    # The spring stretches until it carries the 2 N friction: 2 / 400 = 5 mm short.
    assert q == pytest.approx(0.05 - 2.0 / 400, abs=1e-4)
    assert np.allclose(y[3:], 0)


def test_prismatic_ignores_perpendicular_pull_and_closed_stop():
    h = _hyp(PRISMATIC, (0, -1, 0))
    assert _predict(h, (1, 0, 0))[1] == 0
    assert _predict(h, (0, 1, 0))[1] == 0


def test_weak_probe_below_friction_predicts_no_motion():
    assert _predict(_hyp(PRISMATIC, (0, -1, 0), friction=3.0), (0, -1, 0), force=2.0)[1] == 0


def test_revolute_pull_parallel_to_axis_predicts_no_motion():
    h = _hyp(REVOLUTE, (0, 0, -1), origin=(-0.2, -0.05, 0.6))
    assert _predict(h, (0, 0, 1))[1] == 0
    y, q = _predict(h, (0, -1, 0), distance=0.05)
    assert q > 0.1
    assert np.allclose(y[3:], np.array([0, 0, -1]) * q)


def test_relative_rotvec_roundtrip():
    angle, axis = 0.7, np.array([0.0, 0.6, 0.8])
    q_ref = np.array([1.0, 0, 0, 0])
    q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    assert np.allclose(relative_rotvec(q_ref, q), axis * angle)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_predictor_matches_mujoco_given_true_parameters(family):
    """Model validity: with the true hypothesis, settled predictions match the simulator.

    Measured mismatch is about 0.2-2 mm median. Hinges overshoot the quasi-static equilibrium
    slightly because of inertia during the ramp. The belief's likelihood noise is sized to cover it.
    """
    rng = np.random.default_rng(42)
    pos_err, rot_err = [], []
    for _ in range(3):
        spec = sample_mechanism(family, rng)
        env = MechanismEnv(spec, rng, noise=SensorNoise(0, 0, 0, 0))
        geo = env.geometry()
        env.move_ee_to(geo.handle_pos)
        assert env.grasp()
        q_init = env.observe().handle_quat
        hyp = spec_to_hypothesis(spec)
        for _ in range(4):
            d = rng.normal(size=3)
            probe = Probe(d / np.linalg.norm(d), rng.choice([0.03, 0.08]), rng.choice([6.0, 15.0]))
            r = env.execute_probe(probe)
            y, _, _ = predict_outcome(
                hyp,
                geo.handle_pos,
                r.before.handle_pos,
                r.before.ee_pos,
                probe.direction,
                np.array(probe.distance),
                probe.max_force,
            )
            pos_err.append(np.linalg.norm(y[0, :3] - r.after.handle_pos))
            rot_err.append(np.linalg.norm(y[0, 3:] - relative_rotvec(q_init, r.after.handle_quat)))
    assert np.median(pos_err) < 2e-3
    assert np.max(pos_err) < 1e-2
    assert np.max(rot_err) < np.deg2rad(5)
