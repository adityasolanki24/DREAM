"""Data passed between the simulator and the agent.

Everything here is information the robot could plausibly measure or command on real hardware.
Ground-truth mechanism parameters live in ``dream.sim.mechanisms.MechanismSpec`` and must never
appear in these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
