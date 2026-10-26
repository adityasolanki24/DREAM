"""Contracts for the 43-actuator G1 (``build_mobile_g1(hands=True)``): the Dex3 finger hinges
re-enabled on the WBC's own model. The fast tests check that the default model is unchanged,
that body inertias and the 29-joint policy order are untouched and that the fingers carry
Unitree's limits. The slow test sweeps all fingers over their full range, at the WBC's
0.005 s step and default integrator, with the robot standing under the pretrained policy.
"""

import hashlib

import mujoco
import numpy as np
import pytest

from dream.robot import g1_wbc as w
from dream.sim import g1_mobile as gm
from dream.sim.g1_mobile import HAND_ASSET, HAND_JOINTS, MESHES, G1MobileEnv, build_mobile_g1

HAVE_WEIGHTS = (w.policy_cache() / "GR00T-WholeBodyControl-Walk.onnx").exists()
try:
    import onnxruntime  # noqa: F401
except ImportError:  # optional "locomotion" extra
    HAVE_WEIGHTS = False
needs_policy = pytest.mark.skipif(not HAVE_WEIGHTS, reason="WBC weights not fetched")

KEY_ARRAYS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "jnt_range", "jnt_axis", "dof_armature", "dof_damping", "dof_frictionloss",
    "actuator_gainprm", "actuator_biasprm", "geom_size",
)  # fmt: skip


def _digest(model: mujoco.MjModel) -> str:
    h = hashlib.sha256()
    for name in KEY_ARRAYS:
        h.update(np.ascontiguousarray(getattr(model, name)).tobytes())
    h.update(np.array([model.nq, model.nv, model.nu, model.nbody, model.ngeom]).tobytes())
    return h.hexdigest()


def _vendored_model() -> mujoco.MjModel:
    """The unedited NVIDIA model with only the build-time directory and step the env sets."""
    spec = mujoco.MjSpec.from_file(str(gm.ASSET))
    spec.meshdir = str(MESHES)
    spec.option.timestep = w.DT_PHYSICS
    return spec.compile()


@pytest.fixture(scope="module")
def plain():
    return build_mobile_g1(head_camera=None)


@pytest.fixture(scope="module")
def hand():
    return build_mobile_g1(head_camera=None, hands=True)


def test_default_model_is_the_vendored_one(plain):
    ref = _vendored_model()
    assert (plain.nq, plain.nv, plain.nu) == (ref.nq, ref.nv, ref.nu) == (36, 35, 29)
    assert _digest(plain) == _digest(ref)
    assert plain.opt.integrator == ref.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER


def test_body_masses_and_inertias_are_identical(plain, hand):
    assert hand.body_mass.sum() == pytest.approx(plain.body_mass.sum(), abs=1e-12)
    assert hand.nbody == plain.nbody
    for i in range(plain.nbody):
        j = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY, plain.body(i).name)
        assert j >= 0
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            assert np.array_equal(getattr(plain, field)[i], getattr(hand, field)[j]), (
                field,
                plain.body(i).name,
            )


def test_first_29_joints_and_actuators_keep_the_policy_order(plain, hand):
    body_names = [plain.joint(i).name for i in range(1, plain.njnt)]
    assert len(body_names) == w.N_BODY
    assert [hand.actuator(i).name for i in range(w.N_BODY)] == body_names
    assert [plain.actuator(i).name for i in range(plain.nu)] == body_names
    for i, name in enumerate(body_names):  # same joint, same limits, same motor
        a, b = plain.joint(name), hand.joint(name)
        assert np.array_equal(a.range, b.range) and a.armature == b.armature
        assert np.array_equal(plain.actuator_gainprm[i], hand.actuator_gainprm[i])
        assert hand.actuator(i).trnid[0] == b.id
    # the finger joints are exactly the added ones
    extra = {hand.joint(i).name for i in range(1, hand.njnt)} - set(body_names)
    assert extra == set(HAND_JOINTS) and hand.nu == w.N_BODY + len(HAND_JOINTS) == 43
    assert [hand.actuator(w.N_BODY + k).name for k in range(14)] == list(HAND_JOINTS)


def test_finger_limits_match_the_unitree_model(hand):
    spec = mujoco.MjSpec.from_file(str(HAND_ASSET))
    spec.meshdir = str(MESHES)
    unitree = spec.compile()
    for k, name in enumerate(HAND_JOINTS):
        u, h = unitree.joint(name), hand.joint(name)
        assert np.array_equal(u.range, h.range) and np.array_equal(u.axis, h.axis)
        assert np.array_equal(unitree.jnt_pos[u.id], hand.jnt_pos[h.id])
        assert np.array_equal(unitree.jnt_actfrcrange[u.id], hand.jnt_actfrcrange[h.id])
        assert unitree.body(unitree.jnt_bodyid[u.id]).name == hand.body(hand.jnt_bodyid[h.id]).name
        act = hand.actuator(w.N_BODY + k)
        assert np.array_equal(act.forcerange, unitree.jnt_actfrcrange[u.id])
        assert np.array_equal(act.ctrlrange, u.range)
        assert act.gainprm[0] == gm.FINGER_KP and act.biasprm[1] == -gm.FINGER_KP
        assert hand.dof_damping[h.dofadr[0]] == gm.FINGER_KD
    assert hand.opt.timestep == w.DT_PHYSICS
    assert hand.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER


def test_env_accepts_both_models_and_only_hands_take_finger_targets(plain, hand):
    rng = np.random.default_rng(0)
    env = G1MobileEnv(plain, rng, head_camera=None)
    with pytest.raises(ValueError):
        env.set_hand(np.zeros(14))
    env = G1MobileEnv(hand, rng, head_camera=None)
    env.reset()
    assert env.hands and env.hand_positions().shape == (14,)
    body = [hand.jnt_qposadr[hand.actuator(i).trnid[0]] for i in range(29)]
    assert np.array_equal(env.qadr, body)
    with pytest.raises(ValueError):
        env.set_hand(np.zeros(13))
    env.set_hand(np.full(14, 99.0))  # clipped to the joint range, never beyond it
    assert np.allclose(env.hand_targets, env.hand_range[:, 1])


class _Torques:
    """Stands in for the WBC: constant body torques, so the test sees where ctrl goes."""

    stopped = None

    def tick(self, prop):
        pass

    def torques(self, prop):
        return np.ones(w.N_BODY)


def test_env_step_writes_torques_to_the_body_and_targets_to_the_fingers(hand):
    env = G1MobileEnv(hand, np.random.default_rng(0), head_camera=None)
    env.reset()
    target = env.hand_range[:, 1] * 0.5
    env.set_hand(target)
    env.step(_Torques())
    assert np.array_equal(env.data.ctrl[: w.N_BODY], np.ones(w.N_BODY))
    assert np.array_equal(env.data.ctrl[w.N_BODY :], target)


@pytest.mark.slow
@needs_policy
def test_finger_sweep_is_stable_at_the_wbc_step_under_the_policy(hand):
    """Smooth full-range sweeps and abrupt end-to-end switching, robot balancing throughout."""
    env = G1MobileEnv(hand, np.random.default_rng(0), head_camera=None)
    env.reset()
    ctrl = w.WBCLocomotion.from_cache()
    lo, hi = env.hand_range.T
    dofs = [hand.jnt_dofadr[hand.actuator(w.N_BODY + k).trnid[0]] for k in range(14)]
    tol_smooth, tol_abrupt = 0.05, 0.15  # rad of soft-limit penetration measured at the stops
    vmax_by_phase = []
    for phase, abrupt in enumerate([False, True]):
        over = vmax = 0.0
        for k in range(500):  # 10 s
            if abrupt:
                env.set_hand(lo if (k // 10) % 2 else hi)
            else:
                env.set_hand(lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * 0.5 * k * 0.02)))
            env.step(ctrl)
            q = env.hand_positions()
            assert np.isfinite(env.data.qpos).all() and not ctrl.stopped, (phase, k)
            over = max(over, float(np.max(np.maximum(lo - q, q - hi))))
            vmax = max(vmax, float(np.abs(env.data.qvel[dofs]).max()))
        assert over < (tol_abrupt if abrupt else tol_smooth), (phase, over)
        assert env.data.qpos[2] > 0.7  # still standing
        vmax_by_phase.append(vmax)
    assert max(vmax_by_phase) < 100.0  # torque-limited servos: bounded speed, no growth
    env.set_hand(hi)
    for _ in range(100):
        env.step(ctrl)
    assert np.abs(env.data.qvel[dofs]).max() < 1.0  # settled at the stop, no residual ringing


# -- explicit 29 + 14 actuator bridge: everything by joint name -----------------------------


def _shuffle_actuators(spec: mujoco.MjSpec) -> None:
    """Test scene hook: re-create every actuator in reversed order (same parameters)."""
    old = list(spec.actuators)
    params = [
        {
            k: getattr(a, k)
            for k in (
                "name",
                "target",
                "trntype",
                "gaintype",
                "gainprm",
                "biastype",
                "biasprm",
                "ctrllimited",
                "ctrlrange",
                "forcelimited",
                "forcerange",
                "gear",
            )
        }  # fmt: skip
        for a in old
    ]
    for a in old:
        spec.delete(a)
    for kw in reversed(params):
        spec.add_actuator(**kw)


class _Ramp:
    """Distinct torque per policy-order joint, so a mix-up cannot go unnoticed."""

    stopped = None

    def tick(self, prop):
        pass

    def torques(self, prop):
        return 1.0 + np.arange(w.N_BODY)


@pytest.fixture(scope="module")
def shuffled():
    return build_mobile_g1(head_camera=None, hands=True, scene=_shuffle_actuators)


def test_shuffled_actuator_order_is_handled_by_name(hand, shuffled):
    assert [shuffled.actuator(i).name for i in range(shuffled.nu)][0] == HAND_JOINTS[-1]
    env = G1MobileEnv(shuffled, np.random.default_rng(0), head_camera=None)
    ref = G1MobileEnv(hand, np.random.default_rng(0), head_camera=None)
    assert [shuffled.actuator(int(i)).name for i in env.body_actuators] == list(gm.BODY_JOINTS)
    assert [shuffled.actuator(int(i)).name for i in env.finger_actuators] == list(HAND_JOINTS)
    assert np.array_equal(env.qadr, ref.qadr) and np.array_equal(env.vadr, ref.vadr)
    assert np.array_equal(env.hand_qadr, ref.hand_qadr)
    assert np.array_equal(env.hand_range, ref.hand_range)


def test_finger_and_body_commands_never_cross(hand, shuffled):
    for model in (hand, shuffled):
        env = G1MobileEnv(model, np.random.default_rng(0), head_camera=None)
        env.reset()
        target = env.hand_range.mean(axis=1) + 0.01  # strictly inside every range
        env.set_hand(target)
        env.step(_Ramp())
        ctrl = env.data.ctrl
        assert np.array_equal(ctrl[env.body_actuators], 1.0 + np.arange(w.N_BODY))
        assert np.array_equal(ctrl[env.finger_actuators], target)
        # body torque goes to the named body actuator, whatever the model's index order
        for k, name in enumerate(gm.BODY_JOINTS):
            assert ctrl[model.actuator(name).id] == 1.0 + k
        for k, name in enumerate(HAND_JOINTS):
            assert ctrl[model.actuator(name).id] == target[k]
        assert env.proprioception().q.shape == (w.N_BODY,)
        names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in env.body_actuators]
        assert names == list(gm.BODY_JOINTS)


def test_missing_or_duplicated_body_actuator_is_rejected(plain):
    def drop_one(spec):
        spec.delete(spec.actuator("left_knee_joint"))

    with pytest.raises(ValueError, match="left_knee_joint"):
        G1MobileEnv(build_mobile_g1(head_camera=None, scene=drop_one), np.random.default_rng(0))


def _legacy_step(env, controller):
    """The pre-bridge env.step, verbatim: full-array ctrl write, 29-actuator model."""
    controller.tick(env.proprioception())
    for _ in range(w.DECIMATION):
        env.data.ctrl[:] = controller.torques(env.proprioception())
        mujoco.mj_step(env.model, env.data)
    env._integrate_odometry()


def test_default_model_ctrl_writes_are_the_legacy_ones(plain):
    env = G1MobileEnv(plain, np.random.default_rng(0), head_camera=None)
    assert np.array_equal(env.body_actuators, np.arange(w.N_BODY)) and not env.hands
    env.reset()
    env.step(_Ramp())
    assert np.array_equal(env.data.ctrl, 1.0 + np.arange(w.N_BODY))


@needs_policy
def test_default_model_trajectory_is_byte_identical_to_the_legacy_step(plain):
    final = []
    for stepper in (lambda e, c: e.step(c), _legacy_step):
        env = G1MobileEnv(plain, np.random.default_rng(3), head_camera=None)
        env.reset(0.0, 0.0, 0.7)
        ctrl = w.WBCLocomotion.from_cache()
        ctrl.set_command(0.3, 0.0, 0.2)
        for _ in range(150):  # 3 s of walking
            stepper(env, ctrl)
        final.append((env.data.qpos.copy(), env.data.qvel.copy(), env.odometry()))
    for a, b in zip(final[0], final[1], strict=True):
        assert np.array_equal(a, b)
