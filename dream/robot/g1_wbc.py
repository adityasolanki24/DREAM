"""NVIDIA's released Decoupled WBC lower-body policies for the Unitree G1, as a locomotion layer.

Pretrained, not DREAM's: two ONNX actors from GR00T-WholeBodyControl (commit pinned in
experiments/configs/wbc_policy.json), "Balance" for standing and "Walk" for velocity commands.
The contract is copied from the maintained policy class
decoupled_wbc/control/policy/g1_gear_wbc_policy.py and its config g1_gear_wbc.yaml
(vendored in third_party/gr00t_wbc_g1), and checked by tests/test_g1_wbc.py:
- 50 Hz policy over a 200 Hz simulation (0.005 s physics, decimation 4).
- Observation, 86 values per frame, six frames, oldest first: velocity command
  [vx, vy, wz] x [2, 2, 0.5]; base-height command 0.74 m; torso roll/pitch/yaw command;
  pelvis gyro x 0.5 (body frame); gravity in the pelvis frame; 29 body joint positions minus
  the defaults (legs, waist, arms; arm defaults 0); joint velocities x 0.05; the previous
  15 actions.
- Action: 15 targets (12 legs, 3 waist) = action x 0.25 + default angles, tracked by joint PD
  with the config's gains. The Balance actor runs when the command norm is below 0.05.
- Arms are not controlled by this policy. Here they are held by PD at a neutral pose, as in
  the official sim2mujoco runner (kp 100, kd 0.5); an upper-body skill may replace that.
Posture commands (``WBCLocomotion.set_posture``), upstream contract at commit
45c03facbf848e0c743091b7d9c17952096ff908 of NVlabs/GR00T-WholeBodyControl:
- obs[3] = height_cmd and obs[4:7] = (roll, pitch, yaw) torso command are copied unscaled and
  unclipped into the observation (g1_gear_wbc_policy.py:121-122, set in get_action :205-218;
  defaults from g1_gear_wbc.yaml height_cmd 0.74, rpy_cmd 0).
- The torso rpy is the orientation of torso_link relative to the YAW-ONLY pelvis frame:
  R_yaw(pelvis)^T @ R_torso, converted with Pinocchio matrixToRpy
  (g1_decoupled_whole_body_policy.py:125-135). That FK is evaluated with the lower-body joints
  at zero, so it is the waist chain's own orientation: roll and pitch cannot exceed the waist
  joint range (+-0.52 rad in the vendored MJCF).
- Height: the teleop streamers clip the command to [0.2, 0.74] m
  (teleop/streamers/pico_streamer.py:132 and joycon_streamer.py:426, step 0.01 m per call);
  the keyboard path uses 0.1 m and 10 deg steps without a clip (g1_gear_wbc_policy.py:265-285).
  Upstream has no explicit rpy clip. The trained range of the actor is not published; the
  usable region is measured by experiments/g1_wbc_envelope.py, not assumed.
Inputs are the robot's own sensors only (IMU orientation and gyro, joint encoders). The
velocity envelope is DREAM's commissioning limit, not the policy's trained range, and a tilt
beyond FALL_TILT latches a stop with joint damping only.
"""

from __future__ import annotations

import collections
import os
import pathlib
from dataclasses import dataclass

import numpy as np

N_BODY = 29  # legs 12, waist 3, arms 14: the order of the vendored MJCF and the policy
N_ACTION = 15
DT_PHYSICS = 0.005  # s, g1_gear_wbc.yaml simulation_dt
DECIMATION = 4  # physics steps per policy step (50 Hz)
DEFAULT = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0], np.float32
)
KP = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], float)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], float)
ARM_KP, ARM_KD = 100.0, 0.5  # official sim2mujoco runner's arm hold
CMD_SCALE = np.array([2.0, 2.0, 0.5], np.float32)
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 0.05, 0.25
HEIGHT_CMD = 0.74  # m, g1_gear_wbc.yaml height_cmd; the default posture
RPY_CMD = (0.0, 0.0, 0.0)  # rad, g1_gear_wbc.yaml rpy_cmd
# Hard safety bounds: out-of-range posture commands are REJECTED, never clipped.
HEIGHT_RANGE = (
    0.20,
    0.74,
)  # m, upstream teleop clip (pico_streamer.py:132, joycon_streamer.py:426)
# Roll/pitch: the waist joint range of the vendored MJCF (+-0.52 rad), the most the upstream FK
# definition of the command can ever produce. Yaw: DREAM's conservative limit (the waist yaw
# range is +-2.618 rad, but nothing beyond ~1 rad was ever exercised upstream: keyboard steps
# are 10 deg).
RPY_RANGE = np.array([[-0.52, 0.52], [-0.52, 0.52], [-1.0, 1.0]])  # rad
HISTORY = 6
# DREAM commissioning envelope (m/s, m/s, rad/s): the range exercised in
# experiments/g1_locomotion.py, not a statement of what the policy was trained on.
MAX_COMMAND = np.array([0.5, 0.3, 0.6])
FALL_TILT = 0.6  # rad between pelvis z and vertical: a stand-up is not within this policy


def policy_cache() -> pathlib.Path:
    """Where fetched weights live (outside Git): $DREAM_CACHE/gr00t_wbc or ~/.cache/dream."""
    base = os.environ.get("DREAM_CACHE") or pathlib.Path.home() / ".cache" / "dream"
    return pathlib.Path(base) / "gr00t_wbc"


@dataclass(frozen=True)
class Proprioception:
    quat: np.ndarray  # pelvis orientation (w, x, y, z) from the IMU
    gyro: np.ndarray  # pelvis angular velocity, pelvis frame, rad/s
    q: np.ndarray  # 29 body joint positions, rad
    dq: np.ndarray  # 29 body joint velocities, rad/s


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame of ``quat``."""
    w, x, y, z = quat
    return np.array([2 * (w * y - x * z), -2 * (w * x + y * z), -(1 - 2 * (x * x + y * y))])


def tilt(quat: np.ndarray) -> float:
    """Angle (rad) between the body z axis and vertical."""
    return float(np.arccos(np.clip(-gravity_in_body(quat)[2], -1.0, 1.0)))


def build_observation(
    prop: Proprioception,
    command: np.ndarray,
    last_action: np.ndarray,
    height: float = HEIGHT_CMD,
    rpy: tuple[float, float, float] | np.ndarray = RPY_CMD,
) -> np.ndarray:
    o = np.zeros(13 + 2 * N_BODY + N_ACTION, np.float32)
    o[0:3] = np.asarray(command, np.float32) * CMD_SCALE
    o[3] = height
    o[4:7] = rpy  # torso roll, pitch, yaw command
    o[7:10] = np.asarray(prop.gyro) * ANG_VEL_SCALE
    o[10:13] = gravity_in_body(np.asarray(prop.quat))
    defaults = np.zeros(N_BODY, np.float32)
    defaults[:N_ACTION] = DEFAULT
    o[13 : 13 + N_BODY] = np.asarray(prop.q) - defaults
    o[13 + N_BODY : 13 + 2 * N_BODY] = np.asarray(prop.dq) * DOF_VEL_SCALE
    o[13 + 2 * N_BODY :] = last_action
    return o


class WBCLocomotion:
    """Velocity-commanded walking and standing from the two released actors.

    ``tick(prop)`` runs one 50 Hz policy step and returns the joint targets; ``torques(prop)``
    computes the PD torques at the physics rate. Commands outside MAX_COMMAND are rejected,
    not clipped. After a fall (tilt > FALL_TILT) only damping is applied.

    Posture (base height, torso rpy) defaults to the validated constants (0.74 m, 0, 0,
    observation byte-identical to before). ``set_posture`` sets a new target; with the optional
    rate limits the applied command ramps toward it at 50 Hz, otherwise it jumps.
    """

    def __init__(self, balance_path: str | pathlib.Path, walk_path: str | pathlib.Path):
        import onnxruntime as ort  # optional dependency: the "locomotion" extra

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # deterministic, and small enough for one core
        # Explicit CPU provider: the validated runs used the CPU build, and GPU builds (Isaac
        # Sim ships onnxruntime-gpu) refuse a session without an explicit provider list.
        cpu = ["CPUExecutionProvider"]
        self._balance = ort.InferenceSession(str(balance_path), opts, providers=cpu)
        self._walk = ort.InferenceSession(str(walk_path), opts, providers=cpu)
        self.command = np.zeros(3)
        self.action = np.zeros(N_ACTION, np.float32)
        self.target = DEFAULT.astype(float).copy()
        self.arm_target = np.zeros(N_BODY - N_ACTION)
        self.history = collections.deque(
            [np.zeros(13 + 2 * N_BODY + N_ACTION, np.float32)] * HISTORY, maxlen=HISTORY
        )
        self.stopped: str | None = None
        self.active = "balance"
        self.height = HEIGHT_CMD  # applied posture command
        self.rpy = np.array(RPY_CMD, float)
        self._height_goal, self._rpy_goal = self.height, self.rpy.copy()
        self.max_height_rate: float | None = None  # m/s; None = no rate limit
        self.max_rpy_rate: float | None = None  # rad/s

    @classmethod
    def from_cache(cls) -> WBCLocomotion:
        d = policy_cache()
        return cls(
            d / "GR00T-WholeBodyControl-Balance.onnx", d / "GR00T-WholeBodyControl-Walk.onnx"
        )

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        c = np.array([vx, vy, wz], float)
        if not np.isfinite(c).all() or np.any(np.abs(c) > MAX_COMMAND + 1e-9):
            raise ValueError(f"velocity command {c.tolist()} outside {MAX_COMMAND.tolist()}")
        self.command = c

    def set_posture(
        self,
        height: float,
        rpy: tuple[float, float, float] | np.ndarray = RPY_CMD,
        max_height_rate: float | None = None,
        max_rpy_rate: float | None = None,
    ) -> None:
        """Command pelvis height (m) and torso_link roll/pitch/yaw (rad) relative to the
        yaw-only pelvis frame (module docstring). Values outside HEIGHT_RANGE / RPY_RANGE or
        non-finite are rejected with ValueError and leave the posture unchanged. The rates, when
        given, bound how fast the applied command moves toward the goal (None leaves the
        previous setting); upstream streamers move 0.01 m per call, DREAM's envelope runs use
        0.1 m/s and 0.3 rad/s, chosen slow enough for the 50 Hz policy to track (measured).
        """
        r = np.asarray(rpy, float)
        h = float(height)
        if r.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(h):
            raise ValueError(f"posture height {height!r} rpy {rpy!r} must be finite, rpy of 3")
        if not HEIGHT_RANGE[0] <= h <= HEIGHT_RANGE[1]:
            raise ValueError(f"height command {h} outside {HEIGHT_RANGE}")
        if np.any(r < RPY_RANGE[:, 0]) or np.any(r > RPY_RANGE[:, 1]):
            raise ValueError(f"torso rpy command {r.tolist()} outside {RPY_RANGE.tolist()}")
        for name, rate in (("max_height_rate", max_height_rate), ("max_rpy_rate", max_rpy_rate)):
            if rate is not None and not (np.isfinite(rate) and rate > 0):
                raise ValueError(f"{name} must be positive, got {rate}")
        if max_height_rate is not None:
            self.max_height_rate = max_height_rate
        if max_rpy_rate is not None:
            self.max_rpy_rate = max_rpy_rate
        self._height_goal, self._rpy_goal = h, r
        if self.max_height_rate is None:
            self.height = h
        if self.max_rpy_rate is None:
            self.rpy = r.copy()

    def _advance_posture(self) -> None:
        dt = DT_PHYSICS * DECIMATION
        if self.max_height_rate is not None:
            step = self.max_height_rate * dt
            self.height += float(np.clip(self._height_goal - self.height, -step, step))
        if self.max_rpy_rate is not None:
            step = self.max_rpy_rate * dt
            self.rpy = self.rpy + np.clip(self._rpy_goal - self.rpy, -step, step)

    def tick(self, prop: Proprioception) -> np.ndarray:
        if self.stopped is None and tilt(prop.quat) > FALL_TILT:
            self.stopped = f"fall: pelvis tilt {np.degrees(tilt(prop.quat)):.0f} deg"
        if self.stopped:
            return self.target
        self._advance_posture()
        self.history.append(
            build_observation(prop, self.command, self.action, self.height, self.rpy)
        )
        x = np.concatenate(self.history)[None]
        walking = np.linalg.norm(self.command) >= 0.05
        session = self._walk if walking else self._balance
        self.active = "walk" if walking else "balance"
        out = session.run(None, {session.get_inputs()[0].name: x})[0][0]
        self.action = np.asarray(out, np.float32)
        self.target = self.action.astype(float) * ACTION_SCALE + DEFAULT
        return self.target

    def torques(self, prop: Proprioception) -> np.ndarray:
        q, dq = np.asarray(prop.q), np.asarray(prop.dq)
        tau = np.zeros(N_BODY)
        if self.stopped:  # no position authority after a fall: damping only
            tau[:N_ACTION] = -KD * dq[:N_ACTION]
            tau[N_ACTION:] = -ARM_KD * dq[N_ACTION:]
            return tau
        tau[:N_ACTION] = KP * (self.target - q[:N_ACTION]) - KD * dq[:N_ACTION]
        tau[N_ACTION:] = ARM_KP * (self.arm_target - q[N_ACTION:]) - ARM_KD * dq[N_ACTION:]
        return tau
