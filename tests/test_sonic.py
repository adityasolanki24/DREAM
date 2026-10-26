"""Contracts for the GEAR-SONIC backend (dream/robot/sonic.py), an unofficial port.

Fast tests check the port's constants against the upstream files vendored in
third_party/gear_sonic, the request validation and the capability refusals, with stub ONNX
sessions. The ONNX shape test and the physics test need the weights
(experiments/fetch_sonic_policy.py) and are skipped otherwise.
"""

import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from dream.robot import sonic as S
from dream.robot.wholebody import ControllerCapabilities, WholeBodyTarget

VENDOR = Path(__file__).resolve().parents[1] / "third_party/gear_sonic"
HAVE_WEIGHTS = all(
    (S.policy_cache() / f).exists()
    for f in ("model_encoder.onnx", "model_decoder.onnx", "planner_sonic.onnx")
)
try:
    import onnxruntime  # noqa: F401
except ImportError:
    HAVE_WEIGHTS = False
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="SONIC weights not fetched")
HPP = (VENDOR / "policy_parameters.hpp").read_text()


def _hpp_ints(name: str) -> np.ndarray:
    m = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", HPP)
    return np.array([int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()])


def _target(height=0.79, rpy=(0.0, 0.0, 0.0), ref=None, names=()):
    return WholeBodyTarget(
        np.zeros(36), (0.0, 0.0, 0.0), (), height, rpy, np.zeros(14),
        joint_reference=ref, joint_names=names,
    )  # fmt: skip


def _ref(n=20, dt=0.1, amp=0.0):
    out = []
    for k in range(n):
        q = S.DEFAULT_ANGLES.copy()
        q[15] += amp * k / n
        out.append((k * dt, q))
    return tuple(out)


def _bare_backend(caps, arms="report"):
    """A backend with no env and no weights: enough for the checks that run before physics."""
    b = S.SonicBackend.__new__(S.SonicBackend)
    b._caps, b.held, b.arms = caps, None, arms
    return b


def test_vendored_files_match_their_checksums():
    for line in (VENDOR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((VENDOR / name.lstrip("*")).read_bytes()).hexdigest() == digest


def test_permutation_tables_match_vendored_header_and_round_trip():
    i2m, m2i = _hpp_ints("isaaclab_to_mujoco"), _hpp_ints("mujoco_to_isaaclab")
    assert np.array_equal(i2m, S.ISAACLAB_TO_MUJOCO) and np.array_equal(m2i, S.MUJOCO_TO_ISAACLAB)
    v = np.random.default_rng(0).normal(size=29)
    assert np.allclose(v[m2i][i2m], v) and np.allclose(v[i2m][m2i], v)
    assert sorted(i2m) == list(range(29))


def test_gains_scales_and_limits_match_vendored_files():
    yml = re.sub(r"#[^\n]*", "", (VENDOR / "g1_29dof_sonic_model12.yaml").read_text())
    body = re.search(r"motor_effort_limit_list:\s*(\[[^\]]*\])", yml)
    limits = np.array(ast.literal_eval(body.group(1)))
    assert limits.shape == (43,)
    assert np.allclose(limits[:22], S.YML_EFFORT_BODY29[:22])  # legs, waist, left arm
    assert np.allclose(limits[29:36][:5], S.YML_EFFORT_BODY29[22:27])  # right arm after left hand
    for c, v in (("5020", S.ARM_5020), ("7520_14", S.ARM_7520_14), ("7520_22", S.ARM_7520_22),
                 ("4010", S.ARM_4010)):  # fmt: skip
        assert float(re.search(rf"ARMATURE_{c}\s*=\s*([\d.]+)", HPP).group(1)) == v
    assert S.ACTION_SCALE.shape == S.KPS.shape == S.KDS.shape == (29,)
    assert np.all(S.KPS > 0) and np.all(S.KDS > 0)
    assert S.BODY_JOINT_NAMES[0] == "left_hip_pitch_joint" and len(S.BODY_JOINT_NAMES) == 29


def test_mode_ids_and_height_ranges_match_vendored_upstream_files():
    kp = (VENDOR / "localmotion_kplanner.hpp").read_text()
    for name, value in (("IDLE", S.IDLE), ("IDEL_SQUAT", S.SQUAT),
                        ("IDEL_KNEEL_TWO_LEGS", S.KNEEL_TWO)):  # fmt: skip
        assert int(re.search(rf"\b{name}\s*=\s*(\d+)", kp).group(1)) == value
    doc = (VENDOR / "planner_onnx.md").read_text()
    assert re.search(r"`squat`.*range ~0\.4.{1,3}0\.8m", doc)
    assert re.search(r"`kneelTwoLeg`.*0\.2m-0\.4m", doc)


def test_plan_posture_uses_the_documented_height_ranges():
    assert S.plan_posture(0.79) == (S.IDLE, -1.0)
    assert S.plan_posture(0.45) == (S.SQUAT, 0.45)
    assert S.plan_posture(0.40) == (S.SQUAT, 0.40)
    assert S.plan_posture(0.30) == (S.KNEEL_TWO, 0.30)
    for bad in (0.19, 0.81, float("nan")):
        with pytest.raises(ValueError):
            S.plan_posture(bad)


class _Dec:
    """Decoder stub: returns a fixed raw action in IsaacLab order."""

    def __init__(self, action):
        self.action = np.asarray(action, np.float32)

    def run(self, _names, _feeds):
        return [self.action[None]]


class _Enc:
    def run(self, _names, feeds):
        assert feeds["obs_dict"].shape == (1, 1762)
        return [np.zeros((1, 64), np.float32)]


def test_action_decoding_uses_permuted_index_and_last_action_stores_raw():
    raw = np.arange(29, dtype=np.float64) / 10.0
    p = S.SonicPolicy(_Enc(), _Dec(raw), None)
    p.hist = [
        {
            "q": np.zeros(29),
            "dq": np.zeros(29),
            "act": np.zeros(29),
            "quat": np.array([1.0, 0, 0, 0]),
            "gyro": np.zeros(3),
        }  # fmt: skip
    ] * 10
    p.motion.T = 1
    q_t, _, obs = p.policy()
    a32 = raw.astype(np.float32).astype(np.float64)
    expected = S.DEFAULT_ANGLES + a32[S.ISAACLAB_TO_MUJOCO] * S.ACTION_SCALE
    assert np.allclose(q_t, expected, atol=1e-6)
    assert not np.allclose(q_t, S.DEFAULT_ANGLES + a32 * S.ACTION_SCALE)  # unpermuted differs
    assert np.allclose(p.last_action, a32)  # raw, not permuted
    assert obs.shape == (994,)


def test_state_logger_permutes_joint_state_to_isaaclab_order():
    p = S.SonicPolicy(_Enc(), _Dec(np.zeros(29)), None)
    q = S.DEFAULT_ANGLES + np.arange(29) * 0.01
    p.log_state(q, np.arange(29.0), [1, 0, 0, 0], [0, 0, 0])
    e = p.hist[-1]
    assert np.allclose(e["q"], (q - S.DEFAULT_ANGLES)[S.MUJOCO_TO_ISAACLAB], atol=1e-6)
    assert np.allclose(e["dq"], np.arange(29.0)[S.MUJOCO_TO_ISAACLAB])


def test_unmeasured_capabilities_refuse_execution():
    caps = S.sonic_capabilities(Path("missing/envelope.json"))
    assert not caps.executable and caps.source == "unmeasured: not executable"
    assert caps.supports("posture", "joint_reference")
    b = _bare_backend(caps)
    for t in (_target(), _target(ref=_ref())):
        rep = b.execute(t)
        assert not rep.completed and rep.refused == "unmeasured: not executable"


def test_measured_envelope_file_sets_capabilities(tmp_path):
    f = tmp_path / "envelope.json"
    f.write_text('{"height_m": [0.4, 0.79]}')
    caps = S.sonic_capabilities(f)
    assert caps.executable and caps.height_range == (0.4, 0.79)
    assert caps.request_kinds == {"posture"}  # joint_reference only once measured
    rep = _bare_backend(caps).execute(_target(ref=_ref()))
    assert rep.refused and "joint_reference unsupported" in rep.refused


def test_posture_request_validation():
    caps = ControllerCapabilities("gear_sonic", S.SONIC_KINDS, (0.3, 0.8), (-0.02,) * 3,
                                  (0.02,) * 3, measured=True, source="test")  # fmt: skip
    b = _bare_backend(caps)
    assert "height" in b.execute(_target(height=0.25)).refused
    assert "torso rpy" in b.execute(_target(rpy=(0.0, 0.3, 0.0))).refused
    assert "arm joints" in _bare_backend(caps, "refuse").execute(_target(height=0.5)).refused


def test_joint_reference_validation():
    t, q, err = S.validate_joint_reference(_target(ref=_ref(), names=S.BODY_JOINT_NAMES))
    assert err is None and q.shape == (20, 29)
    assert S.validate_joint_reference(_target(ref=_ref(n=1)))[2]  # too short
    assert "joint_names" in S.validate_joint_reference(_target(ref=_ref(), names=("a",) * 29))[2]
    twice = ((0.0, np.zeros(29)), (0.0, np.zeros(29)))
    assert "increasing" in S.validate_joint_reference(_target(ref=twice))[2]
    short = ((0.0, np.zeros(5)), (1.0, np.zeros(5)))
    assert "29" in S.validate_joint_reference(_target(ref=short))[2]
    nan = S.DEFAULT_ANGLES.copy()
    nan[3] = np.nan
    assert "non-finite" in S.validate_joint_reference(_target(ref=((0.0, nan), (1.0, nan))))[2]
    fast = ((0.0, S.DEFAULT_ANGLES), (0.1, S.DEFAULT_ANGLES + 1.0))
    assert "speed" in S.validate_joint_reference(_target(ref=fast))[2]
    lo, hi = S.DEFAULT_ANGLES - 0.1, S.DEFAULT_ANGLES + 0.1
    assert "limits" in S.validate_joint_reference(_target(ref=_ref(amp=0.5)), lo, hi)[2]


def test_reference_resampling_and_motion_buffer_follow_the_g1_encoder_layout():
    t, q, _ = S.validate_joint_reference(_target(ref=_ref(n=11, dt=0.1, amp=0.5)))
    frames, blend = S.resample_reference(t, q, S.DEFAULT_ANGLES)
    assert blend == 0.0 and frames.shape == (51, 29)  # 1.0 s at 50 Hz, first frame included
    p = S.SonicPolicy(_Enc(), _Dec(np.zeros(29)), None)
    p.load_reference(frames)
    assert p.reference_mode and not p.planner.enabled and p.play and p.cf == 0
    assert p.motion.T == 51 + S.REFERENCE_WINDOW and p.reference_end == 50
    assert np.allclose(p.motion.jp[10], frames[10][S.MUJOCO_TO_ISAACLAB])
    assert np.allclose(p.motion.jv[10], (p.motion.jp[11] - p.motion.jp[10]) * 50.0)
    assert np.allclose(p.motion.jv[50:], 0.0) and np.allclose(p.motion.quat[5], [1, 0, 0, 0])
    for _ in range(80):
        p.advance_reference()
    assert p.cf == 50 and p.reference_done()
    _, blend = S.resample_reference(t, q, S.DEFAULT_ANGLES - 0.3)
    assert abs(blend - 0.3 / S.REFERENCE_BLEND_SPEED) < 0.05


@needs_weights
def test_onnx_shapes():
    p = S.SonicPolicy.from_cache(threads=1)
    assert p.enc.get_inputs()[0].shape == [1, 1762] and p.enc.get_outputs()[0].shape == [1, 64]
    assert p.dec.get_inputs()[0].shape == [1, 994] and p.dec.get_outputs()[0].shape == [1, 29]
    assert p.pl_sess.get_inputs()[0].shape == [1, 4, 36]
    assert p.pl_sess.get_outputs()[0].shape == [1, 64, 36]


def _c3_contacts(env) -> tuple[list[str], int]:
    """Ground truth for PLAN C3 (evaluator only): non-foot robot geoms touching the floor, and
    the number of feet in contact."""
    m, d = env.model, env.data
    feet = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
    floor = m.geom("floor").id
    bad, support = set(), set()
    for c in d.contact[: d.ncon]:
        g = (c.geom1, c.geom2)
        if floor not in g:
            continue
        body = int(m.geom_bodyid[g[1] if g[0] == floor else g[0]])
        (support if body in feet else bad).add(body)
    return sorted(m.body(b).name for b in bad), len(support)


@pytest.mark.slow
@needs_weights
def test_stand_then_squat_on_dream_body_does_not_fall():
    """Stand 3 s, then squat to 0.45 m. No fall by the C3 criteria (tilt latch, non-foot floor
    contact, loss of foot support); the achieved pelvis height is recorded (the port is known to
    sit above the commanded squat height). No tight assertion."""
    from dream.robot.g1_wbc import tilt
    from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1

    model = build_mobile_g1(head_camera=None, hands=True)
    env = G1MobileEnv(model, np.random.default_rng(0), head_camera=None)
    env.reset()
    caps = ControllerCapabilities(
        "gear_sonic", S.SONIC_KINDS, (0.2, 0.8), (-0.02,) * 3, (0.02,) * 3,
        measured=True, source="test: stated bounds",
    )  # fmt: skip
    backend = S.SonicBackend(env, S.SonicPolicy.from_cache(), caps)
    pelvis = model.body("pelvis").id
    log = {"max_tilt": 0.0, "non_foot": set(), "lost_support_s": 0.0, "z": []}

    def probe(obs):
        log["max_tilt"] = max(log["max_tilt"], tilt(obs.prop.quat))
        bad, feet = _c3_contacts(env)
        log["non_foot"] |= set(bad)
        log["lost_support_s"] += 0.02 * (feet == 0)
        log["z"].append((obs.t, float(env.data.xpos[pelvis][2])))

    stand = backend.execute(_target(0.79), probe)
    while env.time < 3.0:  # stand for 3 s in total
        assert backend._step(stand, probe)
    squat = backend.execute(_target(0.45), probe)
    assert stand.completed and squat.completed and not squat.stops, squat.stops
    assert log["max_tilt"] < 0.6 and not log["non_foot"] and log["lost_support_s"] < 0.2
    z_final = float(np.mean([z for t, z in log["z"] if t > env.time - 0.5]))
    print(f"SONIC squat 0.45 m: pelvis {z_final:.3f} m, max tilt {log['max_tilt']:.3f} rad")
    assert 0.3 < z_final < 0.8  # sanity only, not a tracking claim
    assert squat.achieved["unmet"] == ["arm_joints"]
