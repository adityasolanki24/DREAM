"""Data passed between the simulator and the agent.

Everything here is information the robot could plausibly measure or command on real hardware.
Ground-truth mechanism parameters live in ``dream.sim.mechanisms.MechanismSpec`` and must never
appear in these types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Robot controller constants. These belong to the robot, not the mechanism, so the agent may use
# them. The simulator builds its actuators from the same values.
GANTRY_KP = 400.0  # N/m, per-axis position-servo stiffness
GANTRY_KV = 40.0  # N s/m


@dataclass(frozen=True)
class PartGeometry:
    """Visible geometry of the movable part, as a perfect part segmenter would report it.

    This is the report's "oracle geometry, unknown joint" rung: the agent knows where the panel
    and handle are, but not how (or whether) the panel is jointed.
    """

    panel_center: np.ndarray  # (3,) m, world frame
    panel_half_extents: np.ndarray  # (3,) m, axis-aligned with world in the closed state
    handle_pos: np.ndarray  # (3,) m, grasp point in the initial state


@dataclass(frozen=True)
class Probe:
    """One compliant interaction: drag the gripped handle along ``direction``.

    The end effector target moves ``distance`` metres along ``direction`` over ``duration``
    seconds. Each actuator axis is force-limited to ``max_force`` newtons, so the robot cannot
    exert more than that, which also makes a fixed part safe to probe.
    """

    direction: np.ndarray  # (3,) unit vector, world frame
    distance: float  # m
    max_force: float  # N, per actuator axis
    duration: float = 1.0  # s

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=float)
        norm = np.linalg.norm(d)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"probe direction must be a unit vector, got norm {norm:.4f}")
        object.__setattr__(self, "direction", d)


@dataclass(frozen=True)
class Observation:
    """Noisy sensor readings at one instant."""

    time: float  # s
    handle_pos: np.ndarray  # (3,) m, tracked handle position
    handle_quat: np.ndarray  # (4,) tracked handle orientation, (w, x, y, z)
    ee_pos: np.ndarray  # (3,) m, end-effector position from robot proprioception
    ee_force: np.ndarray  # (3,) N, actuator forces commanded by the robot's controller


@dataclass(frozen=True)
class ProbeResult:
    """What the robot recorded while executing one probe."""

    probe: Probe
    before: Observation
    after: Observation  # at the end of the loaded hold, before relaxing
    trajectory: list[Observation] = field(default_factory=list)


@dataclass(frozen=True)
class CompliantMode:
    """Request for a compliant end effector, the same for every embodiment.

    The hand acts like a spring of ``stiffness`` toward its target whose force is capped at
    ``force_limit`` by *norm* (not per axis, unlike the gantry's servos). Each embodiment maps
    this onto its own controller (the Panda: Cartesian admittance).
    """

    stiffness: float = 300.0  # N/m
    force_limit: float = 15.0  # N


@dataclass(frozen=True)
class RobotState:
    """Robot proprioception: what any arm reports about itself, independent of the scene."""

    time: float  # s
    joint_pos: np.ndarray  # rad
    joint_vel: np.ndarray  # rad/s
    tcp_pos: np.ndarray  # (3,) m, tool centre point in the world frame
    tcp_quat: np.ndarray  # (4,) (w, x, y, z)
    gripper_width: float  # m, measured finger opening
    gripper_command: float  # m, commanded opening
    protective_stop: bool
    stop_reason: str = ""
    external_force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # N, world frame


@dataclass(frozen=True)
class ObjectObservation:
    """One tracked object as a perception system reports it (pose with noise, no physics)."""

    name: str
    color: str
    position: np.ndarray  # (3,) m, centre
    yaw: float  # rad, about world z
    size: float  # m, edge length (from the visible extent)
    position_std: float = 0.0  # m, the backend's 1-sigma position uncertainty (0: not reported)
    visible: bool = True  # False: last-known estimate of an object not seen in the latest frame
    last_seen: float | None = None  # s, time of the last frame the object was detected in


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole RGB-D camera, MuJoCo convention: looks along its -z axis, x right, y up.

    ``rotation`` maps camera-frame vectors to the world frame; ``position`` is the optical
    centre in the world. This is the camera's (known) calibration, not scene knowledge.
    """

    width: int
    height: int
    fovy_deg: float
    position: np.ndarray  # (3,) m
    rotation: np.ndarray  # (3, 3)

    @property
    def focal(self) -> float:
        """Focal length in pixels (square pixels)."""
        return 0.5 * self.height / np.tan(np.deg2rad(self.fovy_deg) / 2)


@dataclass(frozen=True)
class RGBDFrame:
    """One synchronised RGB-D image pair, as a depth camera driver delivers it.

    ``depth`` is the distance along the optical axis (m), NaN where there is no return.
    ``self_mask`` marks pixels on the robot's own body, predicted from its kinematic model and
    joint encoders (the standard robot self-filter). It carries no information about the scene.
    """

    time: float  # s
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, m
    self_mask: np.ndarray  # (H, W) bool
    calibration: CameraCalibration


class Embodiment(Protocol):
    """What skills may ask of a robot. Implemented per robot (see dream/sim/tabletop_env.py).

    Commands are setpoints for a deterministic controller that enforces safety limits below
    this interface; there is no way to send raw torques through it.
    """

    def state(self) -> RobotState: ...
    def move_tcp(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray: ...  # clamped target
    def set_gripper(self, width: float) -> None: ...
    def tracking_error(self) -> tuple[float, float]: ...
    def wait_until(self, predicate: Callable[[], bool], timeout: float) -> bool: ...
    def hold(self) -> None: ...
    def set_compliance(self, mode: CompliantMode | None) -> None: ...  # None: stiff tracking


@dataclass(frozen=True)
class DexterousState:
    """Named humanoid proprioception, SI units; root quaternion is wxyz in world frame."""

    time: float
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    protective_stop: bool
    stop_reason: str = ""


class DexterousEmbodiment(Protocol):
    """Multi-finger capability without pretending fingers are a parallel gripper."""

    def state(self) -> DexterousState: ...
    def palm_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]: ...
    def grasp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]: ...
    def finger_forces(self, side: str) -> dict[str, float]: ...
    def reach(self, side: str, position: np.ndarray, quaternion: np.ndarray) -> None: ...
    def set_hand(self, side: str, closure: float) -> None: ...
    def move_arm(self, side: str, positions: np.ndarray) -> None: ...
    def hold(self) -> None: ...
    def hold_for_reasoning(self) -> None: ...


class Perception(Protocol):
    def observe_objects(self) -> list[ObjectObservation]: ...
    def fixtures(self) -> dict[str, tuple[np.ndarray, float]]: ...  # name -> (centre, half width)
    def observe_articulated(self) -> list[ArticulatedObservation]: ...


@dataclass(frozen=True)
class ArticulatedObservation:
    """What perception reports about an object that has a handle (mechanism unknown).

    ``handle_pos``/``handle_quat`` come from a handle tracker: oracle-assisted perception with
    noise, the same convention as the cube tracker. ``box_centre``/``box_half`` are the
    object's visible outer extent (world axis-aligned) in its current state.
    """

    name: str
    handle_pos: np.ndarray  # (3,) m
    handle_quat: np.ndarray  # (4,) (w, x, y, z)
    box_centre: np.ndarray  # (3,) m
    box_half: np.ndarray  # (3,) m
    # (3,) m, half extent of the handle in its own frame (x out of the face); None if unknown
    handle_half: np.ndarray | None = None
    # False: ``handle_quat`` is only the face direction seen while closed, not a measurement of
    # the handle's orientation (e.g. a camera tracking a round knob); rotations are unobserved.
    orientation_observed: bool = True
    # False: the handle is not seen in the latest frame; ``handle_pos`` is its last-known place
    # and must not be used as a measurement.
    handle_visible: bool = True
    position_std: float = 0.0  # m, 1-sigma uncertainty of handle_pos (0: not reported)
    handle_id: int | None = None  # perception track identity, never a simulator identifier
    last_seen: float | None = None  # s, last actual handle detection
