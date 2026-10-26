"""GEAR-SONIC whole-body controller as a second backend of the controller-independent interface.

unofficial Python/onnxruntime port of gear_sonic_deploy;
not verified against the reference C++ binary.
Upstream: NVlabs/GR00T-WholeBodyControl, commit 45c03facbf848e0c743091b7d9c17952096ff908.

Licences: the port is a derivative of Apache-2.0 source; the ONNX weights (encoder, decoder,
planner) are NVIDIA Open Model License and are NOT in Git (experiments/fetch_sonic_policy.py).
Upstream files this module reads constants from are vendored unmodified in third_party/gear_sonic.

Citation key (paths relative to the upstream commit above; tests/test_sonic.py checks the
constants against the vendored copies):
  CPP = gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
  PP  = .../include/policy_parameters.hpp      (vendored)
  KP  = .../include/localmotion_kplanner.hpp   (vendored)
  KPT = .../include/localmotion_kplanner_tensorrt.hpp
  MU  = .../include/math_utils.hpp
  SLc = .../src/state_logger.cpp
  BS  = gear_sonic/utils/mujoco_sim/base_sim.py
  YML = gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml   (vendored)
  OBS = gear_sonic_deploy/policy/release/observation_config.yaml              (vendored)
  DOC = docs/source/references/planner_onnx.md                                (vendored)
Deviations from upstream are marked ``# DEVIATION``.

Design (PLAN C2, C7):
- ``SonicPolicy`` is the control-side logic of the C++ deploy (encoder, decoder, planner, motion
  buffer), one call per 50 Hz tick. The planner runs SYNCHRONOUSLY on the control tick, every
  fifth tick as the C++ planner thread's 10 Hz cadence (a planner call takes ~170 ms of wall
  time on a CPU; its latency is not modelled in simulation time). This is the documented
  backend choice, a DEVIATION from the C++ planner thread.
- ``SonicController`` is the ``tick(prop)`` / ``torques(prop)`` object ``G1MobileEnv.step``
  expects. SONIC's PD gains and torque clips are applied to the 29 body joints only; the env
  writes them through its name-mapped body actuator indices, never a full-array ctrl write.
  The 14 finger actuators stay with ``G1MobileEnv.set_hand`` (SONIC's Dex3 handling is not
  ported).
- ``SonicBackend`` implements ``WholeBodyBackend``. Request kinds: ``posture`` (pelvis height
  and facing to the planner's idle / squat / kneelTwoLeg modes) and ``joint_reference`` (a timed
  body-joint trajectory fed to the encoder in its "g1" reference-motion mode).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dream.robot.g1_wbc import DECIMATION, DT_PHYSICS, N_BODY, Proprioception, tilt
from dream.robot.wholebody import (
    ControllerCapabilities,
    ExecutionReport,
    Monitor,
    Observation,
    PostureCommand,
    StopEvent,
    WholeBodyTarget,
)

LABEL = (
    "unofficial Python/onnxruntime port of gear_sonic_deploy; "
    "not verified against the reference C++ binary"
)
ROOT = Path(__file__).resolve().parents[2]
ENVELOPE_JSON = ROOT / "results/sonic_envelope/envelope.json"
SONIC_KINDS = frozenset({"posture", "joint_reference"})


def policy_cache() -> Path:
    """Where fetched weights live (outside Git): $DREAM_CACHE/gear_sonic or ~/.cache/dream."""
    base = os.environ.get("DREAM_CACHE") or Path.home() / ".cache" / "dream"
    return Path(base) / "gear_sonic"


# ----------------------------------------------------------------------------------------------
# Constants: PP:40-43 armatures, PP:46-47, PP:50-59, PP:62-65, PP:100-105 permutations,
# PP:109-141 action scale, PP:143-208 kp/kd, PP:210-240 default angles
ARM_5020, ARM_7520_14, ARM_7520_22, ARM_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
NAT_FREQ = 10 * 2.0 * 3.1415926535
DAMP_RATIO = 2
_ARM = {"m5020": ARM_5020, "m7520_14": ARM_7520_14, "m7520_22": ARM_7520_22, "m4010": ARM_4010}
ST = {k: a * NAT_FREQ * NAT_FREQ for k, a in _ARM.items()}
DP = {k: 2.0 * DAMP_RATIO * a * NAT_FREQ for k, a in _ARM.items()}
EF = {"m5020": 25.0, "m7520_14": 88.0, "m7520_22": 139.0, "m4010": 5.0}
# Motor type per hardware (MuJoCo) order joint (PP:109-141); kp/kd multipliers PP:143-208.
_T = ["m7520_22", "m7520_22", "m7520_14", "m7520_22", "m5020", "m5020",
      "m7520_22", "m7520_22", "m7520_14", "m7520_22", "m5020", "m5020",
      "m7520_14", "m5020", "m5020",
      "m5020", "m5020", "m5020", "m5020", "m5020", "m4010", "m4010",
      "m5020", "m5020", "m5020", "m5020", "m5020", "m4010", "m4010"]  # fmt: skip
_KMUL = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2] + [1] * 14
ACTION_SCALE = np.array([0.25 * EF[t] / ST[t] for t in _T])
# std::array<float> in the C++, hence the float32 round trip.
KPS = np.array([_KMUL[i] * ST[t] for i, t in enumerate(_T)], np.float32).astype(np.float64)
KDS = np.array([_KMUL[i] * DP[t] for i, t in enumerate(_T)], np.float32).astype(np.float64)
DEFAULT_ANGLES = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
     0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]
)  # fmt: skip
# PP:100-105. Names follow the C++: ISAACLAB_TO_MUJOCO[j] is the IsaacLab index of MuJoCo joint j;
# MUJOCO_TO_ISAACLAB[i] is the MuJoCo index of IsaacLab joint i.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20,
     22, 24, 26, 28]
)  # fmt: skip
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
     20, 27, 21, 28]
)  # fmt: skip
LOWER_BODY_ISAAC = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]  # PP:96
# Torque clip of the sim: YML motor_effort_limit_list, body entries (the Dex3 entries are dropped).
YML_EFFORT_BODY29 = np.array(
    [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50,
     25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], float
)  # fmt: skip
BODY_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)  # fmt: skip

CONTROL_DT = DT_PHYSICS * DECIMATION  # 0.02 s, CPP:2186
assert abs(CONTROL_DT - 0.02) < 1e-12
PLANNER_DT = 0.1  # CPP:2187
INIT_DURATION = 3.0  # CPP:2189
WAIT_TICKS = 25  # operator ']' press: 0.5 s hold at the default pose (DEVIATION: fixed delay)
REPLAN_INTERVAL = {"run": 0.1, "crawl": 0.2, "box": 1.0, "other": 1.0}  # CPP:231-234
LOOK_AHEAD, DEFAULT_HEIGHT, INITIAL_SEED = 2, 0.788740, 1234  # KP:215-217
K_ADAPT_TRIGGER, K_ADAPT_STOP, K_RECOVER_TRIGGER = 0.10, 0.05, 0.045  # CPP:244-246
# KP:78-87 LocomotionMode.
IDLE, SLOW_WALK, WALK, RUN, SQUAT, KNEEL_TWO, KNEEL_ONE, LYING, CRAWLING = range(9)
MODE_NAMES = {IDLE: "idle", SQUAT: "squat", KNEEL_TWO: "kneelTwoLeg"}
# KP:108 is_static_motion_mode: IDLE, SQUAT, KNEEL_TWO, KNEEL, LYING, IDLE_BOXING.
STATIC_MODES = {0, 4, 5, 6, 7, 9}
CRAWL_MODES = {8}
RUN_MODES = {3}
BOX_MODES = {11, 12, 13, 15, 16}  # CPP:3678-3690 (CRAWLING only; ELBOW_CRAWLING 14 is not in it)

# Planner height request ranges, DOC:118-119 (docs/source/references/planner_onnx.md): squat
# "range ~0.4-0.8 m", kneelTwoLeg "0.2 m-0.4 m". The 0.4 m boundary belongs to squat.
SQUAT_HEIGHT_RANGE = (0.4, 0.8)
KNEEL_HEIGHT_RANGE = (0.2, 0.4)
# DREAM choice (an assumption, not upstream): a requested pelvis height at or above this is a
# standing request and uses idle (mode default height, SONIC's standing pelvis is 0.789 m).
STAND_THRESHOLD = 0.70
# Torso rpy cannot be commanded in planner mode; this tolerance (rad) only absorbs noise in the
# rpy derived from a nominal q. A larger request is refused. Assumption.
RPY_TOLERANCE = 0.02
# DREAM sanity bounds for joint references (assumptions, not SONIC limits).
REFERENCE_MAX_JOINT_SPEED = 6.0  # rad/s, finite-difference sanity bound on the request
REFERENCE_BLEND_SPEED = 0.5  # rad/s, ramp from the measured pose to the first reference frame
REFERENCE_WINDOW = 46  # frames the encoder reads ahead: (10 - 1) * 5 + 1 (CPP:436)
MAX_EXECUTE_S = 12.0  # a planner motion is at most 64 frames at 30 Hz (~2.1 s) per replan
SETTLE_S = 1.0  # monitored hold after the planner motion has finished
FALL_TILT_DEG = 60.0  # base_sim.py check_fall tilt used by the port's runs


# ----------------------------------------------------------------------------------------------
# math (MU)
def quat_rotate(q, v):  # MU:132 quat_rotate_d
    qw = q[0]
    qv = np.array(q[1:4], float)
    v = np.asarray(v, float)
    return v * (2.0 * qw * qw - 1.0) + np.cross(qv, v) * qw * 2.0 + qv * np.dot(qv, v) * 2.0


def quat_mul(a, b):  # MU:174 quat_mul_d
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    return np.array([
        qq - ww + (z1 - y1) * (y2 - z2), qq - xx + (x1 + w1) * (x2 + w2),
        qq - yy + (w1 - x1) * (y2 + z2), qq - zz + (z1 + y1) * (w2 - x2),
    ])  # fmt: skip


def quat_unit(q, eps=1e-12):  # MU:255
    return np.asarray(q, float) / max(float(np.linalg.norm(q)), eps)


def quat_from_angle_z(angle):  # MU:278 with axis (0, 0, 1)
    return quat_unit(np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)]))


def calc_heading(q):  # MU:206
    d = quat_rotate(q, [1, 0, 0])
    return math.atan2(d[1], d[0])


def heading_quat(q):  # MU:300
    return quat_from_angle_z(calc_heading(q))


def heading_quat_inv(q):  # MU:314
    return quat_from_angle_z(-calc_heading(q))


def quat_conj(q):  # MU:338
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_slerp(q0, q1, t):  # MU:343
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    q1a = q1
    if dot < 0.0:
        q1a = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_unit(q0 + t * (q1a - q0))
    th = math.acos(abs(dot))
    s = math.sin(th)
    return (math.sin((1 - t) * th) / s) * q0 + (math.sin(t * th) / s) * q1a


def quat_to_R(q):  # MU:410
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


# ----------------------------------------------------------------------------------------------
class Session(Protocol):
    """The slice of ``onnxruntime.InferenceSession`` the port uses (tests inject stubs)."""

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass
class Timer:
    t: dict[str, list[float]]

    @classmethod
    def new(cls) -> Timer:
        return cls({"encoder": [], "decoder": [], "planner": []})

    def add(self, k: str, dt: float) -> None:
        self.t[k].append(dt * 1e3)

    def summary(self) -> dict[str, dict[str, float] | None]:
        return {
            k: (
                {"n": len(v), "mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95)),
                 "max_ms": float(np.max(v))}
                if v else None
            )
            for k, v in self.t.items()
        }  # fmt: skip


class Motion:
    """MotionSequence; joint arrays are IsaacLab order (CPP:772, KP:515)."""

    def __init__(self, cap: int = 1500):
        self.cap = cap
        self.jp = np.zeros((cap, 29))
        self.jv = np.zeros((cap, 29))
        self.pos = np.zeros((cap, 3))
        self.quat = np.tile([1.0, 0, 0, 0], (cap, 1))
        self.T = 0


@dataclass
class Move:
    """MovementState (KP:60-72)."""

    mode: int = 0
    mdir: tuple = (0.0, 0.0, 0.0)
    fdir: tuple = (1.0, 0.0, 0.0)
    speed: float = 0.0
    height: float = 0.0


class Planner:
    """LocalMotionPlannerTensorRT semantics (KPT) on onnxruntime. Version 2 (planner path
    contains 'V2', CPP:2438-2442)."""

    def __init__(self, sess: Session):
        self.s = sess
        self.ctx = np.zeros((1, 4, 36), np.float32)
        self.mode = np.zeros(1, np.int64)
        self.vel = np.full(1, -1, np.float32)
        self.mdir = np.zeros((1, 3), np.float32)
        self.fdir = np.zeros((1, 3), np.float32)
        self.seed = np.zeros(1, np.int64)
        self.height = np.full(1, -1, np.float32)  # KPT:79,90
        self.has_tgt = np.zeros((1, 1), np.int64)
        self.tpos = np.zeros((1, 4, 3), np.float32)
        self.thead = np.zeros((1, 4), np.float32)  # KPT:91-93
        self.allowed = np.zeros((1, 11), np.int64)
        self.allowed[0, 3:6] = 1  # KPT:97-102 -> tokens 9,10,11
        self.motion50 = Motion()
        self.gen_frame = 0
        self.available = False
        self.enabled = False
        self.initialized = False
        self.first_raw = np.zeros((0, 36))

    def _update_inputs(self, mode, vel, height, mdir, fdir, seed):  # KPT:376-413
        self.mode[0] = mode if mode < 27 else 0
        self.vel[0] = vel
        self.height[0] = height  # version 2: height always written (KPT:392-396)
        self.mdir[0] = mdir
        self.fdir[0] = fdir
        if seed != -1:
            self.seed[0] = seed

    def _infer(self, timer: Timer):
        t = time.perf_counter()
        feeds = {
            "context_mujoco_qpos": self.ctx, "target_vel": self.vel, "mode": self.mode,
            "movement_direction": self.mdir, "facing_direction": self.fdir,
            "random_seed": self.seed, "has_specific_target": self.has_tgt,
            "specific_target_positions": self.tpos, "specific_target_headings": self.thead,
            "allowed_pred_num_tokens": self.allowed, "height": self.height,
        }  # fmt: skip
        q, n = self.s.run(None, feeds)
        timer.add("planner", time.perf_counter() - t)
        return q[0], int(np.asarray(n).reshape(-1)[0])

    def initialize(self, motor_q, timer: Timer):
        """KP:332-393 + InitializeContext KP:591-625."""
        self.initialized = False
        self._update_inputs(0, -1.0, -1.0, [0, 0, 0], [1, 0, 0], INITIAL_SEED)
        self.gen_frame = 0
        for n in range(4):  # x=y=0, z=default_height, identity quaternion, measured raw joint q
            self.ctx[0, n, :3] = [0, 0, DEFAULT_HEIGHT]
            self.ctx[0, n, 3:7] = [1, 0, 0, 0]
            self.ctx[0, n, 7:] = motor_q
        q, n = self._infer(timer)
        self.first_raw = q[:n].copy()
        self._resample(q, n)
        self.initialized = True

    def update(self, current_frame, motion, mode, vel, height, mdir, fdir, timer: Timer):
        """KP:533-589."""
        self._update_inputs(mode, vel, height, mdir, fdir, -1)
        self.gen_frame = current_frame + LOOK_AHEAD
        self._update_context(motion)
        q, n = self._infer(timer)
        self._resample(q, n)

    def _update_context(self, m: Motion):  # KP:628-680
        gen_time = self.gen_frame / 50.0
        for n in range(4):
            f50 = (gen_time + n / 30.0) * 50.0
            f0 = min(int(math.floor(f50)), m.T - 1)
            f1 = min(f0 + 1, m.T - 1)
            w0 = 1.0 - (f50 - f0)
            w1 = 1.0 - w0
            self.ctx[0, n, 3:7] = quat_slerp(m.quat[f0], m.quat[f1], f50 - f0)
            self.ctx[0, n, 0:3] = w0 * m.pos[f0] + w1 * m.pos[f1]
            self.ctx[0, n, 7 + MUJOCO_TO_ISAACLAB] = w0 * m.jp[f0] + w1 * m.jp[f1]

    def _resample(self, q, n):
        """KP:395-531 (30 Hz -> 50 Hz)."""
        if n > 64 or np.isnan(q[:n]).any():
            raise RuntimeError("planner output invalid")
        T = int(math.floor(n / 30.0 * 50))
        m = self.motion50
        m.T = T
        qj = q[:, 7:][:, MUJOCO_TO_ISAACLAB]  # jp[joint] = qpos[7 + mujoco_to_isaaclab[joint]]
        for f in range(T):
            f30 = (f / 50.0) * 30
            f0 = int(math.floor(f30))
            f1 = min(f0 + 1, n - 1)
            w0 = 1.0 - (f30 - f0)
            w1 = 1.0 - w0
            m.pos[f] = w0 * q[f0, 0:3] + w1 * q[f1, 0:3]
            m.quat[f] = quat_slerp(q[f0, 3:7], q[f1, 3:7], f30 - f0)
            m.jp[f] = w0 * qj[f0] + w1 * qj[f1]
        m.jv[: T - 1] = (m.jp[1:T] - m.jp[: T - 1]) * 50.0
        m.jv[T - 1] = m.jv[T - 2]
        self.available = True


# ----------------------------------------------------------------------------------------------
class SonicPolicy:
    """Control-side logic of g1_deploy_onnx_ref (CPP), one call per 50 Hz tick.

    ``encoder``, ``decoder`` and ``planner`` are onnxruntime sessions (or stubs with ``run``)."""

    def __init__(self, encoder: Session, decoder: Session, planner: Session):
        self.enc, self.dec, self.pl_sess = encoder, decoder, planner
        self.timer = Timer.new()
        self.planner = Planner(planner)
        self.hist: list[dict[str, np.ndarray]] = []
        self.last_action = np.zeros(29)
        self.motion = Motion()
        self.cf = 0
        self.play = False
        self.reinit = False
        self.reference_mode = False
        self.reference_end = 0  # last real reference frame (frames after it are padding)
        self.heading_init_quat = np.array([1.0, 0, 0, 0])
        self.delta_heading = 0.0
        self.init_ref_rot = np.array([1.0, 0, 0, 0])
        self.idle_stored = False
        self.idle_state = "IDLE"
        self.idle_orig = np.zeros(29)
        self.movement_buf = Move(IDLE, (0, 0, 0), (1.0, 0, 0), -1.0, -1.0)  # CPP:2273
        self.last_move = Move(0, (0, 0, 0), (1, 0, 0), -1.0, -1.0)  # CPP:230
        self.replan_counter = 0.0
        self.token = np.zeros(64)
        self.q_raw_latest = DEFAULT_ANGLES.copy()
        # Encoder observation layout from observation_config.yaml (OBS, encoder_observations).
        enc_dims = [
            ("encoder_mode_4", 4), ("motion_joint_positions_10frame_step5", 290),
            ("motion_joint_velocities_10frame_step5", 290),
            ("motion_root_z_position_10frame_step5", 10), ("motion_root_z_position", 1),
            ("motion_anchor_orientation", 6), ("motion_anchor_orientation_10frame_step5", 60),
            ("motion_joint_positions_lowerbody_10frame_step5", 120),
            ("motion_joint_velocities_lowerbody_10frame_step5", 120),
            ("vr_3point_local_target", 9), ("vr_3point_local_orn_target", 12),
            ("smpl_joints_10frame_step1", 720), ("smpl_anchor_orientation_10frame_step1", 60),
            ("motion_joint_positions_wrists_10frame_step1", 60),
        ]  # fmt: skip
        self.enc_off: dict[str, int] = {}
        o = 0
        for n, d in enc_dims:
            self.enc_off[n] = o
            o += d
        assert o == 1762

    @classmethod
    def from_cache(cls, directory: str | Path | None = None, threads: int = 2) -> SonicPolicy:
        import onnxruntime as ort  # optional dependency: the "locomotion" extra

        d = Path(directory) if directory else policy_cache()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads

        def mk(name: str):
            return ort.InferenceSession(str(d / name), so, providers=["CPUExecutionProvider"])

        return cls(mk("model_encoder.onnx"), mk("model_decoder.onnx"), mk("planner_sonic.onnx"))

    # State logger (CPP:2842 GatherRobotStateToLogger; floats from DDS).
    def log_state(self, q_hw, dq_hw, quat, gyro):
        q_hw = np.asarray(q_hw, np.float32).astype(np.float64)
        dq_hw = np.asarray(dq_hw, np.float32).astype(np.float64)
        quat = np.asarray(quat, np.float32).astype(np.float64)
        gyro = np.asarray(gyro, np.float32).astype(np.float64)
        body_q = (q_hw - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB]  # CPP:2862
        body_dq = dq_hw[MUJOCO_TO_ISAACLAB]  # CPP:2864
        self.hist.append({"q": body_q, "dq": body_dq, "act": self.last_action.copy(),
                          "quat": quat, "gyro": gyro})  # fmt: skip
        self.hist = self.hist[-64:]  # only the newest 10 are ever read
        self.q_raw_latest = q_hw

    @staticmethod
    def _zero():  # SLc:474-504
        return {"q": np.zeros(29), "dq": np.zeros(29), "act": np.zeros(29),
                "quat": np.zeros(4), "gyro": np.zeros(3)}  # fmt: skip

    def get_latest(self, n):
        """SLc:199-234 stride-1 branch, newest_first=false: oldest..newest, zero padded."""
        out = self.hist[::-1][:n]
        while len(out) < n:
            out.append(self._zero())
        return out[::-1]

    def update_heading(self):  # CPP:559-591
        if self.reinit:
            self.heading_init_quat = self.hist[-1]["quat"].copy()
            self.delta_heading = 0.0
            self.reinit = False
            self.init_ref_rot = self.motion.quat[self.cf].copy()
        if self.cf == 0:
            self.init_ref_rot = self.motion.quat[0].copy()

    def apply_delta_heading(self):  # CPP:593-607
        a = quat_mul(heading_quat(self.heading_init_quat), heading_quat_inv(self.init_ref_rot))
        if self.delta_heading != 0.0:
            a = quat_mul(quat_from_angle_z(self.delta_heading), a)
        return a

    def _frames_hold(self, n, step):
        """CPP:762-770, 640-650: play ? clamp(cf + k * step) : cf."""
        T = self.motion.T
        return [min(self.cf + k * step, T - 1) if self.play else self.cf for k in range(n)]

    def obs_motion_pos(self, n, step):  # CPP:740
        return np.concatenate([self.motion.jp[f] for f in self._frames_hold(n, step)])

    def obs_motion_vel(self, n, step):
        """CPP:819-908: frame = clamp(cf + k * step); zeros unless play."""
        T = self.motion.T
        return np.concatenate([
            self.motion.jv[min(self.cf + k * step, T - 1)] if self.play else np.zeros(29)
            for k in range(n)
        ])  # fmt: skip

    def obs_anchor(self, n, step):
        """CPP:616-701 orientation_mode 0."""
        base = self.hist[-1]["quat"]
        ad = self.apply_delta_heading()
        out = []
        for f in self._frames_hold(n, step):
            R = quat_to_R(quat_mul(quat_conj(base), quat_mul(ad, self.motion.quat[f])))
            out.append([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])  # first two columns
        return np.concatenate(out)

    def encode(self):
        buf = np.zeros(1762)
        o = self.enc_off
        buf[o["encoder_mode_4"] :][:4] = [0.0, 0, 0, 0]  # CPP:1691: GetEncodeMode()=0 (CPP:2402)
        buf[o["motion_joint_positions_10frame_step5"] :][:290] = self.obs_motion_pos(10, 5)
        buf[o["motion_joint_velocities_10frame_step5"] :][:290] = self.obs_motion_vel(10, 5)
        # Required set of encoder mode "g1" (OBS).
        buf[o["motion_anchor_orientation_10frame_step5"] :][:60] = self.obs_anchor(10, 5)
        t = time.perf_counter()
        tok = self.enc.run(None, {"obs_dict": buf.astype(np.float32)[None]})[0][0]
        self.timer.add("encoder", time.perf_counter() - t)
        self.token = tok.astype(np.float64)
        return self.token

    def decoder_obs(self):
        """OBS observations order: token, ang vel, joint pos, joint vel, last actions, gravity
        (all 10frame_step1)."""
        h = self.get_latest(10)
        tok = self.encode()
        gdir = [quat_rotate(quat_conj(e["quat"]), [0.0, 0.0, -1.0]) for e in h]  # CPP:1613-1630
        obs = np.concatenate([
            tok, np.concatenate([e["gyro"] for e in h]), np.concatenate([e["q"] for e in h]),
            np.concatenate([e["dq"] for e in h]), np.concatenate([e["act"] for e in h]),
            np.concatenate(gdir),
        ])  # fmt: skip
        assert obs.shape == (994,)
        return obs

    def policy(self):
        """CPP:3137-3170 CreatePolicyCommand. Returns (joint targets, raw action, obs)."""
        obs = self.decoder_obs()
        t = time.perf_counter()
        a = self.dec.run(None, {"obs_dict": obs.astype(np.float32)[None]})[0][0].astype(np.float64)
        self.timer.add("decoder", time.perf_counter() - t)
        # CPP:3153: floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i] (PERMUTED index).
        q_t = DEFAULT_ANGLES + a[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        self.last_action = a.copy()  # CPP:3154: last_action[i] = floatarr[i] (RAW value)
        return q_t.astype(np.float32).astype(np.float64), a, obs  # q_target is float (CPP:3155)

    def planner_tick(self):
        """CPP:3595-3830. Returns True when the planner ran."""
        pl = self.planner
        if not (pl.enabled and pl.initialized) or self.motion.T == 0:
            return False
        cur, last = self.movement_buf, self.last_move
        under_static = cur.mode in STATIC_MODES
        facing_changed = tuple(cur.fdir) != tuple(last.fdir)
        height_changed = cur.height != last.height
        mode_changed = cur.mode != last.mode
        speed_changed = cur.speed != last.speed
        dir_changed = tuple(cur.mdir) != tuple(last.mdir)
        self.replan_counter += PLANNER_DT
        key = (
            "run" if cur.mode in RUN_MODES else "crawl" if cur.mode in CRAWL_MODES
            else "box" if cur.mode in BOX_MODES else "other"
        )  # fmt: skip
        time_to_replan = False
        # DEVIATION (cosmetic): epsilon for float accumulation of 0.1 steps.
        if self.replan_counter >= REPLAN_INTERVAL[key] - 1e-9:
            self.replan_counter = 0.0
            time_to_replan = True
        need = False
        if mode_changed or facing_changed or height_changed:
            need = True
        elif (not under_static) and (
            speed_changed or dir_changed or (time_to_replan and cur.speed != 0)
        ):
            need = True
        if not need:
            return False
        self.last_move = Move(cur.mode, tuple(cur.mdir), tuple(cur.fdir), cur.speed, cur.height)
        pl.update(self.cf, self.motion, cur.mode, cur.speed, cur.height, cur.mdir, cur.fdir,
                  self.timer)  # fmt: skip
        return True

    def advance(self):
        """CPP:3184-3425 CurrentFrameAdvancement (planner branch)."""
        pl, m, motor_q = self.planner, self.motion, self.q_raw_latest
        if not (pl.enabled and pl.initialized):
            return
        success = first = False
        if pl.available and pl.motion50.T > 0:
            g = pl.motion50
            pl.available = False
            if m.T == 0:
                m.jp[: g.T] = g.jp[: g.T]
                m.jv[: g.T] = g.jv[: g.T]
                m.pos[: g.T] = g.pos[: g.T]
                m.quat[: g.T] = g.quat[: g.T]
                m.T = g.T
                success = first = True
            else:
                fgen, cf = pl.gen_frame, self.cf
                new_len = fgen - cf + g.T
                if new_len > 0:
                    blend_start = max(0, fgen - cf)
                    for f in range(new_len):  # in-place sequential loop exactly as CPP:3262-3300
                        f_old_raw = f + cf
                        # f_new from the UNclamped f_old (CPP:3276-3283)
                        f_new = int(np.clip(f_old_raw - fgen, 0, g.T - 1))
                        f_old = int(np.clip(f_old_raw, 0, m.T - 1))
                        w_new = float(np.clip((f - blend_start) / 8.0, 0.0, 1.0))
                        w_old = 1.0 - w_new
                        q0 = m.quat[f_old].copy()
                        m.jp[f] = w_old * m.jp[f_old] + w_new * g.jp[f_new]
                        m.jv[f] = w_old * m.jv[f_old] + w_new * g.jv[f_new]
                        m.pos[f] = w_old * m.pos[f_old] + w_new * g.pos[f_new]
                        m.quat[f] = quat_slerp(q0, g.quat[f_new], w_new)
                    m.T = new_len
                    success = True
            if success:
                self.cf = 0
                self.idle_stored = False
                if first:
                    self.reinit = True
        if m.T > 0 and self.play:  # CPP:3335-3410
            nf = self.cf + 1
            if nf >= m.T:
                nf = m.T - 1
                if self.movement_buf.mode == IDLE:
                    self._idle_readapt(nf, motor_q)
            self.cf = nf

    def _idle_readapt(self, nf, motor_q):
        m = self.motion
        if not self.idle_stored:
            for idx in LOWER_BODY_ISAAC:
                self.idle_orig[idx] = m.jp[nf][idx]
            self.idle_stored = True
            self.idle_state = "IDLE"
        err = np.mean([abs(m.jp[nf][i] - motor_q[MUJOCO_TO_ISAACLAB[i]]) for i in LOWER_BODY_ISAAC])
        if self.idle_state == "IDLE":
            if err > K_ADAPT_TRIGGER:
                self.idle_state = "ADAPTING"
            elif err < K_RECOVER_TRIGGER:
                self.idle_state = "RECOVERING"
        elif self.idle_state == "ADAPTING":
            if err < K_ADAPT_STOP:
                self.idle_state = "IDLE"
        elif self.idle_state == "RECOVERING":
            if err > K_ADAPT_TRIGGER:
                self.idle_state = "ADAPTING"
        if self.idle_state == "ADAPTING":
            for i in LOWER_BODY_ISAAC:
                m.jp[nf][i] = 0.98 * m.jp[nf][i] + 0.02 * motor_q[MUJOCO_TO_ISAACLAB[i]]
        elif self.idle_state == "RECOVERING":
            for i in LOWER_BODY_ISAAC:
                m.jp[nf][i] = 0.98 * m.jp[nf][i] + 0.02 * self.idle_orig[i]

    # -- reference-motion mode (encoder "g1": joint positions, velocities, anchor orientation)
    def load_reference(self, q_body: np.ndarray) -> None:
        """Replace the motion buffer by a 50 Hz body-joint reference ``q_body`` (n, 29), MuJoCo
        joint order, and play it from frame 0 with the planner off.

        The C++ plays a loaded motion through the same MotionSequence the planner fills
        (current_motion_, CPP:3411-3432, observation gathers CPP:616-908), so the encoder input
        is built by the unchanged ``encode``. What a joint-only reference does not carry is
        filled as follows (DEVIATIONs): the anchor (root) orientation is the identity, an upright
        root whose heading follows the robot (reinit heading, CPP:559-591); the joint velocities
        are finite differences of the positions (as the planner resample, KP:395-531) because
        the motion CSVs' velocity files are not available; the real frames are followed by
        ``REFERENCE_WINDOW`` copies of the last frame with zero velocity so the encoder's look
        ahead (CPP:436, streamed motions stop at timesteps minus that window) stays inside the
        buffer."""
        q = np.asarray(q_body, float)
        n = len(q)
        if q.ndim != 2 or q.shape[1] != 29 or n < 2:
            raise ValueError("reference must be (n >= 2, 29)")
        if n + REFERENCE_WINDOW > self.motion.cap:
            raise ValueError(
                f"reference too long: {n} frames (cap {self.motion.cap - REFERENCE_WINDOW})"
            )
        m = Motion(self.motion.cap)
        jp = q[:, MUJOCO_TO_ISAACLAB]  # jp[i] = q_mj[M2I[i]] as in Planner._resample
        m.jp[:n] = jp
        m.jp[n : n + REFERENCE_WINDOW] = jp[-1]
        m.jv[: n - 1] = (jp[1:] - jp[:-1]) * 50.0
        m.jv[n - 1] = m.jv[n - 2]
        m.jv[n - 1 :] = 0.0
        m.pos[:] = [0.0, 0.0, DEFAULT_HEIGHT]
        m.T = n + REFERENCE_WINDOW
        self.motion = m
        self.reference_end = n - 1
        self.reference_mode = True
        self.planner.enabled = False
        self.cf = 0
        self.play = True
        self.reinit = True
        self.idle_stored = False

    def advance_reference(self) -> None:
        """Frame advance for a non-planner motion (CPP:3411-3432). DEVIATION: a finished
        reference holds its last frame (the C++ file motions reset to frame 0 with play off)."""
        if self.reference_mode and self.play:
            self.cf = min(self.cf + 1, self.reference_end)

    def reference_done(self) -> bool:
        return self.reference_mode and self.cf >= self.reference_end

    def freeze_reference(self) -> None:
        """Hold: the encoder keeps reading the current frame with zero velocity (play off)."""
        self.play = False

    def end_reference(self, motor_q: np.ndarray) -> None:
        """Return to planner mode: re-initialise the planner context from the measured pose
        (KP:332-393) and idle. DEVIATION: the C++ does this through the operator's planner key."""
        self.reference_mode = False
        self.motion = Motion()
        self.cf = 0
        self.movement_buf = Move(IDLE, (0, 0, 0), (1, 0, 0), -1.0, -1.0)
        self.last_move = Move(IDLE, (0, 0, 0), (1, 0, 0), -1.0, -1.0)
        self.planner.enabled = True
        self.planner.initialize(motor_q, self.timer)
        self.advance()  # first-time copy of the planner motion (CPP:3199-3225)
        self.play = True


# ----------------------------------------------------------------------------------------------
class SonicController:
    """The ``tick(prop)`` / ``torques(prop)`` pair ``G1MobileEnv.step`` drives.

    Start-up follows the C++ states: INIT (3 s linear ramp from the measured pose to the default
    angles, CPP:2762-2796), a short WAIT at the default pose, then CONTROL with the planner
    initialised from the measured pose. DEVIATION: the sim's elastic band is not used (the robot
    stands on the floor during the ramp), and the operator's key presses are programmatic.

    Torques: BS:259-285 ``tau = kp (q_des - q) + kd (0 - dq)`` with SONIC's gains, clipped to
    the YML effort limits (BS:423), on the 29 body joints only.
    """

    def __init__(self, policy: SonicPolicy, init_s: float = 0.0):
        """``init_s`` is the INIT ramp length. 0 (default) is a DIRECT start from the measured
        standing pose: the robot is free-standing, with no gantry to hold it while it ramps
        (the C++ ramps for 3 s, INIT_DURATION, with the robot hanging). DEVIATION."""
        self.policy = policy
        self.init_s = init_s
        self.phase = "init" if init_s > 0 else "start"
        self.t_init = 0.0
        self.wait = 0
        self.n = 0  # control ticks since CONTROL started
        self.target = DEFAULT_ANGLES.copy()
        self.stopped: str | None = None

    @property
    def started(self) -> bool:
        return self.phase == "control"

    def tick(self, prop: Proprioception) -> np.ndarray:
        p = self.policy
        q = np.asarray(prop.q, float)
        if self.phase == "init":
            self.t_init += CONTROL_DT
            if self.t_init < self.init_s:
                r = min(max(self.t_init / self.init_s, 0.0), 1.0)
                cmd = q.astype(np.float32).astype(np.float64) * (1 - r) + DEFAULT_ANGLES * r
                self.target = cmd.astype(np.float32).astype(np.float64)
            else:
                self.target = DEFAULT_ANGLES.copy()
                self.phase = "wait"
            return self.target
        if self.phase == "wait":
            self.wait += 1
            self.target = DEFAULT_ANGLES.copy()
            if self.wait >= WAIT_TICKS:
                self.phase = "start"
            return self.target
        p.log_state(q, prop.dq, prop.quat, prop.gyro)
        if self.phase == "start":
            # DEVIATION: the C++ has 3.5 s of logged INIT states; a direct start replicates the
            # measured state so the 10-frame history window is not zero padded.
            while len(p.hist) < 10:
                p.hist.insert(0, {**p.hist[0], "act": np.zeros(29)})
            p.planner.enabled = True
            p.planner.initialize(p.q_raw_latest, p.timer)
            p.advance()  # first-time copy of planner motion (CPP:3199-3225)
            p.play = True  # operator planner key (KB:522)
            self.phase = "control"
            self.target = DEFAULT_ANGLES.copy()  # this tick holds the INIT target
            return self.target
        if self.n % 5 == 0 and not p.reference_mode:
            # DEVIATION: planner executed synchronously on the control tick (module docstring).
            p.planner_tick()
        p.update_heading()
        q_t, _, _ = p.policy()
        if p.reference_mode:
            p.advance_reference()
        else:
            p.advance()
        self.n += 1
        self.target = q_t
        return q_t

    def torques(self, prop: Proprioception) -> np.ndarray:
        tau = KPS * (self.target - np.asarray(prop.q)) + KDS * (0.0 - np.asarray(prop.dq))
        return np.clip(tau, -YML_EFFORT_BODY29, YML_EFFORT_BODY29)

    def raw_torques(self, prop: Proprioception) -> np.ndarray:
        return KPS * (self.target - np.asarray(prop.q)) + KDS * (0.0 - np.asarray(prop.dq))

    # -- commands (the operator's movement buffer, CPP:2273)
    def command_planner(self, mode: int, height: float, facing_yaw: float = 0.0) -> None:
        if self.policy.reference_mode:
            raise RuntimeError("reference active: call end_reference first")
        f = (math.cos(facing_yaw), math.sin(facing_yaw), 0.0)
        self.policy.movement_buf = Move(mode, (0.0, 0.0, 0.0), f, -1.0, height)

    def planner_settled(self) -> bool:
        """The last command has been planned and its motion has played to the end."""
        p = self.policy
        if p.reference_mode or not p.planner.initialized or p.motion.T == 0:
            return False
        b, last = p.movement_buf, p.last_move
        same = (b.mode, b.height, tuple(b.fdir)) == (last.mode, last.height, tuple(last.fdir))
        return same and not p.planner.available and p.cf >= p.motion.T - 1


# ----------------------------------------------------------------------------------------------
def plan_posture(height: float) -> tuple[int, float]:
    """Planner (mode, height request) for a derived pelvis height, from the ranges the planner
    documents (DOC:118-119): squat 0.4-0.8 m, kneelTwoLeg 0.2-0.4 m. Heights at or above
    STAND_THRESHOLD are standing (idle, mode-default height). Raises ValueError outside 0.2-0.8."""
    h = float(height)
    if not math.isfinite(h) or not KNEEL_HEIGHT_RANGE[0] <= h <= SQUAT_HEIGHT_RANGE[1]:
        raise ValueError(f"pelvis height {height!r} outside the planner's 0.2-0.8 m")
    if h >= STAND_THRESHOLD:
        return IDLE, -1.0
    if h >= SQUAT_HEIGHT_RANGE[0]:
        return SQUAT, h
    return KNEEL_TWO, h


def sonic_capabilities(path: str | Path = ENVELOPE_JSON) -> ControllerCapabilities:
    """Measured limits from results/sonic_envelope/envelope.json, else "unmeasured: not executable".

    Minimal envelope schema (the file is written by a future SONIC envelope experiment):
    ``{"height_m": [lo, hi], "kinds": ["posture", "joint_reference"]}``; ``kinds`` defaults to
    the postures only, so joint references stay refused until they are measured."""
    p = Path(path)
    stated = (KNEEL_HEIGHT_RANGE[0], SQUAT_HEIGHT_RANGE[1])
    rpy = (RPY_TOLERANCE,) * 3
    neg = (-RPY_TOLERANCE,) * 3
    if not p.exists():
        return ControllerCapabilities.unmeasured(
            "gear_sonic", stated, neg, rpy, kinds=SONIC_KINDS,
            note="no envelope file: stated planner ranges only. " + LABEL,
        )  # fmt: skip
    raw = json.loads(p.read_text())
    kinds = frozenset(raw.get("kinds", ["posture"]))
    return ControllerCapabilities(
        "gear_sonic", kinds, tuple(raw["height_m"]), neg, rpy,
        measured=True, source=str(p), note=LABEL,
    )  # fmt: skip


def validate_joint_reference(
    target: WholeBodyTarget, jlo: np.ndarray | None = None, jhi: np.ndarray | None = None
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Check a request's ``joint_reference``. Returns (times, q (n, 29), error)."""
    ref = target.joint_reference
    if ref is None or len(ref) < 2:
        return None, None, "joint_reference needs at least 2 frames"
    if target.joint_names and tuple(target.joint_names) != BODY_JOINT_NAMES:
        return None, None, "joint_names must be the 29 body joints in the SONIC/MJCF order"
    try:
        t = np.array([float(a) for a, _ in ref])
        q = np.array([np.asarray(b, float) for _, b in ref])
    except (TypeError, ValueError):
        return None, None, "joint_reference frames must be (t, q_body of 29)"
    if q.ndim != 2 or q.shape[1] != N_BODY:
        return None, None, f"joint_reference q must have {N_BODY} values per frame"
    if not (np.isfinite(t).all() and np.isfinite(q).all()):
        return None, None, "joint_reference contains non-finite values"
    if np.any(np.diff(t) <= 0):
        return None, None, "joint_reference times must be strictly increasing"
    if jlo is not None and (np.any(q < jlo - 1e-6) or np.any(q > jhi + 1e-6)):
        return None, None, "joint_reference outside the joint limits"
    speed = float(np.max(np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]))
    if speed > REFERENCE_MAX_JOINT_SPEED:
        return None, None, f"joint_reference speed {speed:.1f} rad/s > {REFERENCE_MAX_JOINT_SPEED}"
    return t, q, None


def resample_reference(
    t: np.ndarray, q: np.ndarray, measured: np.ndarray, blend_speed: float = REFERENCE_BLEND_SPEED
) -> tuple[np.ndarray, float]:
    """50 Hz frames: a linear blend from the measured pose to the first frame (slowest joint at
    ``blend_speed``), then the request linearly interpolated. Returns (frames, blend seconds)."""
    blend = float(np.max(np.abs(q[0] - measured))) / blend_speed
    nb = int(math.ceil(blend / CONTROL_DT))
    head = [measured + (k / nb) * (q[0] - measured) for k in range(1, nb)] if nb > 1 else []
    grid = np.arange(0.0, t[-1] - t[0] + 1e-9, CONTROL_DT)
    body = np.stack([np.interp(grid + t[0], t, q[:, j]) for j in range(N_BODY)], axis=1)
    frames = np.vstack([np.array(head).reshape(-1, N_BODY), body]) if head else body
    return frames, nb * CONTROL_DT if nb > 1 else 0.0


class SonicBackend:
    """``WholeBodyBackend`` over a ``G1MobileEnv``-like physics owner (``step(controller)``,
    ``proprioception()``, ``model``, ``data``; ``set_hand`` for the finger actuators).

    Request kinds ``posture`` and ``joint_reference``; ``capabilities()`` is "unmeasured: not
    executable" until results/sonic_envelope/envelope.json exists, and ``execute`` then refuses.
    Planner-mode postures cannot set arm joints: ``arms="report"`` (default) executes the posture
    and reports ``unmet: ["arm_joints"]`` with the arm error; ``arms="refuse"`` refuses instead.
    Achieved pelvis height needs ground truth, so it is read from the monitor's ``measure``
    (``ach_*`` keys), like ``WBCBackend``; the backend itself reads proprioception only.
    """

    def __init__(
        self,
        env,
        policy: SonicPolicy | None = None,
        capabilities: ControllerCapabilities | None = None,
        arms: str = "report",
        settle_s: float = SETTLE_S,
        init_s: float = 0.0,
    ):
        if arms not in ("report", "refuse"):
            raise ValueError("arms must be 'report' or 'refuse'")
        self.env = env
        self.policy = policy or SonicPolicy.from_cache()
        self.controller = SonicController(self.policy, init_s)
        self._caps = capabilities or sonic_capabilities()
        self.arms, self.settle_s = arms, settle_s
        self.held: StopEvent | None = None
        self._odom_yaw0: float | None = None

    def capabilities(self) -> ControllerCapabilities:
        return self._caps

    def set_hand(self, targets: np.ndarray) -> None:
        """Finger targets go to the env's name-mapped finger actuators (not SONIC's Dex3 logic)."""
        self.env.set_hand(targets)

    # -- observation / holding -------------------------------------------------------------
    def command(self) -> PostureCommand:
        b = self.policy.movement_buf
        arm = np.asarray(self.env.proprioception().q)[15:].copy()
        height = DEFAULT_HEIGHT if b.height < 0 else float(b.height)
        return PostureCommand(height, (0.0, 0.0, 0.0), arm)

    def observe(self) -> Observation:
        prop = self.env.proprioception()
        return Observation(
            self.env.time, prop, self.controller.torques(prop), self.env.odometry(),
            self.command(),
        )  # fmt: skip

    def hold(self, event: StopEvent) -> None:
        """Protective stop (PLAN C4): stop issuing new references. A joint reference freezes at its
        current frame; a planner posture keeps its last (static) mode. The policy keeps running."""
        if self.policy.reference_mode:
            self.policy.freeze_reference()
        self.held = event

    def release_hold(self) -> None:
        if self.policy.reference_mode:
            self.policy.play = True
        self.held = None

    # -- execution -------------------------------------------------------------------------
    def execute(
        self,
        target: WholeBodyTarget,
        monitor: Monitor | None = None,
        capabilities: ControllerCapabilities | None = None,
    ) -> ExecutionReport:
        caps = capabilities or self._caps
        req: dict[str, object] = {
            "height": target.pelvis_height, "rpy": target.torso_rpy, "arm": target.arm_joints,
            "kind": "joint_reference" if target.joint_reference is not None else "posture",
        }  # fmt: skip
        if not caps.executable:
            return ExecutionReport(False, req, refused=caps.source)
        if self.held is not None:
            return ExecutionReport(False, req, refused=f"held: {self.held.reason}")
        if target.joint_reference is not None:
            return self._execute_reference(target, req, caps, monitor)
        return self._execute_posture(target, req, caps, monitor)

    def _execute_posture(self, target, req, caps, monitor) -> ExecutionReport:
        if not caps.supports("posture"):
            return ExecutionReport(False, req, refused="posture request unsupported")
        bad = caps.violations(target.pelvis_height, target.torso_rpy)
        if bad:
            return ExecutionReport(False, req, refused="; ".join(bad))
        try:
            mode, height = plan_posture(target.pelvis_height)
        except ValueError as e:
            return ExecutionReport(False, req, refused=str(e))
        if self.arms == "refuse":
            return ExecutionReport(False, req, refused="arm joints cannot be set in planner mode")
        req.update(mode=MODE_NAMES[mode], planner_height=height)
        report = ExecutionReport(False, req)
        if not self._start(report, monitor):
            return report
        if self.policy.reference_mode:
            self.policy.end_reference(self.policy.q_raw_latest)
        if self._odom_yaw0 is None:
            self._odom_yaw0 = float(self.env.odometry()[2])
        facing = _wrap(float(target.stance[2]) - self._odom_yaw0)
        # Facing is sent only for a real turn request: a DEVIATION-free default (1, 0, 0) is the
        # planner's initial heading, and a small stance yaw would otherwise force a replan.
        self.controller.command_planner(mode, height, facing if abs(facing) > 0.1 else 0.0)
        t0 = self.env.time
        settled_at: float | None = None
        while self.env.time - t0 < MAX_EXECUTE_S:
            if not self._step(report, monitor):
                break
            if self.controller.planner_settled():
                settled_at = settled_at if settled_at is not None else self.env.time
                if self.env.time - settled_at >= self.settle_s:
                    report.completed = True
                    break
        else:
            report.stops.append(StopEvent(self.env.time, "execution_timeout"))
        self._finish(report, target)
        return report

    def _execute_reference(self, target, req, caps, monitor) -> ExecutionReport:
        if not caps.supports("joint_reference"):
            kinds = sorted(caps.request_kinds)
            why = f"joint_reference unsupported by {caps.name} (supports {kinds})"
            return ExecutionReport(False, req, refused=why)
        m = self.env.model
        joints = m.actuator_trnid[self.env.body_actuators, 0]
        t, q, err = validate_joint_reference(target, m.jnt_range[joints, 0], m.jnt_range[joints, 1])
        if err:
            return ExecutionReport(False, req, refused=err)
        report = ExecutionReport(False, req)
        if not self._start(report, monitor):
            return report
        measured = np.asarray(self.env.proprioception().q, float)
        frames, blend = resample_reference(t, q, measured)
        req.update(frames=len(frames), blend_s=blend)
        self.policy.load_reference(frames)
        t0 = self.env.time
        limit = len(frames) * CONTROL_DT + 2.0
        done_at: float | None = None
        while self.env.time - t0 < limit:
            if not self._step(report, monitor):
                break
            if self.policy.reference_done():
                done_at = done_at if done_at is not None else self.env.time
                if self.env.time - done_at >= self.settle_s:
                    report.completed = True
                    break
        else:
            report.stops.append(StopEvent(self.env.time, "execution_timeout"))
        self._finish(report, target)
        return report

    def _start(self, report: ExecutionReport, monitor) -> bool:
        """Run the C++ INIT / WAIT / planner-start sequence once. Returns False on a stop."""
        while not self.controller.started:
            if not self._step(report, monitor):
                return False
        return True

    def _step(self, report: ExecutionReport, monitor) -> bool:
        """One 50 Hz tick. Returns False after a stop event (the backend then holds)."""
        self.env.step(self.controller)
        obs = self.observe()
        row: dict[str, object] = {
            "t": obs.t,
            "cmd_height": obs.command.height,
            "tilt_rad": tilt(obs.prop.quat),
            "phase": self.controller.phase,
        }
        measure = getattr(monitor, "measure", None)
        if measure is not None:
            row.update(measure(obs))
        report.trace.append(row)
        event = monitor(obs) if monitor is not None else None
        if event is None and math.degrees(float(row["tilt_rad"])) > FALL_TILT_DEG:
            event = StopEvent(obs.t, f"fall: tilt {math.degrees(float(row['tilt_rad'])):.0f} deg")
        if event is not None:
            report.stops.append(event)
            self.hold(event)
            return False
        return True

    def _finish(self, report: ExecutionReport, target: WholeBodyTarget) -> None:
        if report.trace:
            report.achieved = {k: v for k, v in report.trace[-1].items() if k.startswith("ach_")}
        if target.joint_reference is None:
            arms = np.asarray(self.env.proprioception().q)[15:]
            err = float(np.max(np.abs(arms - np.asarray(target.arm_joints, float))))
            report.achieved["unmet"] = ["arm_joints"]
            report.achieved["arm_error_max_rad"] = err
        report.achieved["controller"] = LABEL
        report.achieved["latency_ms"] = self.policy.timer.summary()


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
