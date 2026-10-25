"""Contracts for the pretrained G1 locomotion layer (dream/robot/g1_wbc.py, dream/sim/g1_mobile.py).

The fast tests check the observation/action contract against NVIDIA's vendored config and
MJCF. The slow tests run the released ONNX policies through MuJoCo physics; they need the
weights in the local cache (experiments/fetch_wbc_policy.py) and are skipped otherwise.
"""

import ast
import re
from math import cos, sin

import numpy as np
import pytest

from dream.robot import g1_wbc as w
from dream.sim.g1_mobile import ASSET, G1MobileEnv, build_mobile_g1

HAVE_WEIGHTS = (w.policy_cache() / "GR00T-WholeBodyControl-Walk.onnx").exists()
try:
    import onnxruntime  # noqa: F401
except ImportError:  # optional "locomotion" extra
    HAVE_WEIGHTS = False
needs_policy = pytest.mark.skipif(not HAVE_WEIGHTS, reason="WBC weights not fetched")


def _config(path) -> dict:
    """The vendored YAML is flat ``key: value`` with bracketed lists and ``#`` comments."""
    text = re.sub(r"#[^\n]*", "", path.read_text())
    pairs = re.findall(r"^(\w+):(.*?)(?=^\w+:|\Z)", text, flags=re.M | re.S)
    out = {}
    for key, value in pairs:
        value = " ".join(value.split())
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value.strip('"')
    return out


def test_constants_match_nvidias_config():
    cfg = _config(ASSET.parent / "g1_gear_wbc.yaml")
    assert cfg["simulation_dt"] == w.DT_PHYSICS and cfg["control_decimation"] == w.DECIMATION
    assert np.allclose(cfg["kps"], w.KP) and np.allclose(cfg["kds"], w.KD)
    assert np.allclose(cfg["default_angles"], w.DEFAULT)
    assert np.allclose(cfg["cmd_scale"], w.CMD_SCALE) and cfg["height_cmd"] == w.HEIGHT_CMD
    assert cfg["ang_vel_scale"] == w.ANG_VEL_SCALE and cfg["dof_vel_scale"] == w.DOF_VEL_SCALE
    assert cfg["action_scale"] == w.ACTION_SCALE and cfg["num_actions"] == w.N_ACTION
    assert cfg["num_obs"] == w.HISTORY * (13 + 2 * w.N_BODY + w.N_ACTION) == 516


def test_joint_order_is_legs_waist_arms_as_the_policy_expects():
    m = build_mobile_g1(head_camera=None)
    names = [m.joint(i).name for i in range(1, m.njnt)]
    assert len(names) == w.N_BODY and names[0] == "left_hip_pitch_joint"
    assert names[11] == "right_ankle_roll_joint" and names[12:15] == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    assert [m.actuator(i).name for i in range(m.nu)] == names  # ctrl index == joint index


def test_gravity_and_tilt_from_the_imu_quaternion():
    assert np.allclose(w.gravity_in_body(np.array([1.0, 0, 0, 0])), [0, 0, -1])
    a = 0.3  # pitched nose-down about +y: gravity tips toward +x in the body frame
    q = np.array([cos(a / 2), 0, sin(a / 2), 0])
    assert np.allclose(w.gravity_in_body(q), [sin(a), 0, -cos(a)])
    assert np.isclose(w.tilt(q), a)
    yawed = np.array([cos(0.7), 0, 0, sin(0.7)])  # yaw does not change gravity
    assert np.allclose(w.gravity_in_body(yawed), [0, 0, -1])


def test_observation_layout():
    prop = w.Proprioception(
        np.array([1.0, 0, 0, 0]), np.array([0.1, 0.2, 0.3]), np.arange(29) * 0.01, np.ones(29)
    )
    o = w.build_observation(prop, np.array([0.4, 0.0, 0.5]), np.full(15, 0.5, np.float32))
    assert o.shape == (86,)
    assert np.allclose(o[0:3], [0.8, 0.0, 0.25]) and o[3] == pytest.approx(0.74)
    assert np.allclose(o[7:10], [0.05, 0.1, 0.15]) and np.allclose(o[10:13], [0, 0, -1])
    assert np.isclose(o[13], 0.0 - w.DEFAULT[0]) and np.isclose(o[13 + 16], 0.16)  # arm
    assert np.allclose(o[42:71], 0.05) and np.allclose(o[71:], 0.5)


@needs_policy
def test_commands_outside_the_envelope_are_rejected_not_clipped():
    c = w.WBCLocomotion.from_cache()
    c.set_command(0.5, 0.0, -0.6)
    for bad in ((0.6, 0, 0), (0, 0.4, 0), (0, 0, np.nan)):
        with pytest.raises(ValueError):
            c.set_command(*bad)
    assert np.allclose(c.command, [0.5, 0.0, -0.6])


@pytest.mark.slow
@needs_policy
def test_pretrained_policy_stands_walks_turns_and_stops_in_physics():
    env = G1MobileEnv(build_mobile_g1(head_camera=None), np.random.default_rng(0))
    env.reset()
    c = w.WBCLocomotion.from_cache()
    for _ in range(100):  # 2 s standing
        env.step(c)
    start = env.true_pose()
    c.set_command(0.4, 0, 0)
    for _ in range(250):
        env.step(c)
    walked = env.true_pose()
    c.set_command(0, 0, 0.5)
    for _ in range(200):
        env.step(c)
    turned = env.true_pose()
    c.set_command(0, 0, 0)
    for _ in range(100):
        env.step(c)
    end = env.true_pose()
    assert c.stopped is None and 0.70 < env.data.qpos[2] < 0.80
    assert 1.2 < walked[0] - start[0] < 2.2  # commanded 2.0 m over 5 s
    assert 1.4 < (turned[2] - walked[2]) % (2 * np.pi) < 2.4  # commanded 2.0 rad
    assert np.linalg.norm(end[:2] - turned[:2]) < 0.15  # stopping keeps it in place
