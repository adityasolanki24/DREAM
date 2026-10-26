"""Whole-body backend interface and safety supervisor, tested with a fake controller and plant.

No physics here: the fakes honour the contract the real objects have (WBCLocomotion's
``set_posture`` / ``arm_target`` / ``set_command`` / ``stopped`` / ``torques`` and a plant with
``step(controller)`` / ``proprioception()`` / ``odometry()``), so what is checked is the backend's
trajectory and hold logic and the supervisor's stop and recovery decisions.
"""

import numpy as np
import pytest

from dream.robot import g1_supervisor as sup
from dream.robot.g1_posture import PlanningModel
from dream.robot.g1_wbc import DEFAULT, N_ACTION, N_BODY, Proprioception
from dream.robot.wholebody import (
    ENVELOPE_JSON,
    TICK,
    ControllerCapabilities,
    HandGoal,
    PostureCommand,
    StopEvent,
    WBCBackend,
    WholeBodyTarget,
    synchronized_path,
)

CAPS = ControllerCapabilities(
    "fake", frozenset({"posture"}),
    (0.3, 0.74), (-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), measured=True, source="test",
)  # fmt: skip


@pytest.fixture(scope="module")
def pm():
    return PlanningModel()


class FakeController:
    def __init__(self):
        self.height, self.rpy = 0.74, np.zeros(3)
        self.arm_target = np.zeros(N_BODY - N_ACTION)
        self.command = np.zeros(3)
        self.stopped = None
        self.posture_calls = []

    def set_command(self, vx, vy, wz):
        self.command = np.array([vx, vy, wz], float)

    def set_posture(self, height, rpy=(0, 0, 0), **_):
        self.height, self.rpy = float(height), np.asarray(rpy, float)
        self.posture_calls.append((self.height, tuple(self.rpy)))

    def torques(self, prop):
        return np.zeros(N_BODY)


class FakePlant:
    """Standing robot: joints at the policy defaults, arms follow their target exactly."""

    def __init__(self, controller, script=None):
        self.c, self.t, self.script = controller, 0.0, script or (lambda tick, plant: None)
        self.ticks = 0
        self.q = np.concatenate([DEFAULT, np.zeros(N_BODY - N_ACTION)]).astype(float)
        self.dq = np.zeros(N_BODY)
        self.gyro = np.zeros(3)
        self.quat = np.array([1.0, 0, 0, 0])
        self.feet = (True, True)
        self.force = 0.0
        self.tau = np.zeros(N_BODY)

    @property
    def time(self):
        return self.t

    def step(self, controller):
        self.ticks += 1
        self.t += TICK
        self.q[N_ACTION:] = controller.arm_target
        self.script(self.ticks, self)

    def proprioception(self):
        return Proprioception(self.quat.copy(), self.gyro.copy(), self.q.copy(), self.dq.copy())

    def odometry(self):
        return np.zeros(3)


class TorquedController(FakeController):
    def __init__(self, plant_ref):
        super().__init__()
        self.plant_ref = plant_ref

    def torques(self, prop):
        return self.plant_ref[0].tau if self.plant_ref else np.zeros(N_BODY)


def make(pm, script=None, caps=CAPS, **kw):
    ref = []
    c = TorquedController(ref)
    plant = FakePlant(c, script)
    ref.append(plant)
    backend = WBCBackend(
        plant, c, caps, contact_force=lambda: plant.force, feet_contact=lambda: plant.feet
    )
    return backend, plant, sup.WholeBodySupervisor(pm, **kw)


def target(height=0.6, rpy=(0.0, 0.2, 0.0), arm_shift=0.4):
    arm = np.zeros(14)
    arm[0] = -arm_shift
    return WholeBodyTarget(
        np.zeros(36), (0, 0, 0), (HandGoal("left", (0.5, 0.1, 0.9)),), height, rpy, arm
    )


def test_trajectory_is_rate_limited_and_channels_arrive_together():
    a = PostureCommand(0.74, (0, 0, 0), np.zeros(14))
    b = PostureCommand(0.54, (0, 0.3, 0), np.full(14, 0.1))
    path = synchronized_path(a, b, CAPS)
    assert path[-1].height == pytest.approx(0.54) and path[-1].rpy[1] == pytest.approx(0.3)
    steps = np.diff([a.height] + [p.height for p in path])
    assert np.abs(steps).max() <= CAPS.height_rate * TICK + 1e-9
    assert len(path) == pytest.approx(0.2 / CAPS.height_rate / TICK, abs=1)
    slow = synchronized_path(a, b, CAPS, scale=0.5)
    assert len(slow) == pytest.approx(2 * len(path), abs=2)


def test_unmeasured_capabilities_refuse_execution(pm):
    caps = ControllerCapabilities.unmeasured("x", (0.2, 0.74), (-0.5,) * 3, (0.5,) * 3)
    assert not caps.executable and caps.source == "unmeasured: not executable"
    backend, plant, _ = make(pm, caps=caps)
    report = backend.execute(target())
    assert report.refused == "unmeasured: not executable" and not report.completed
    assert plant.ticks == 0


def test_targets_outside_measured_limits_are_refused_not_clipped(pm):
    backend, plant, _ = make(pm)
    report = backend.execute(target(height=0.2))
    assert report.refused and "height" in report.refused and plant.ticks == 0
    report = backend.execute(target(rpy=(0.0, 0.5, 0.0)))
    assert report.refused and "rpy" in report.refused


def test_measured_envelope_file_gives_executable_capabilities():
    if not ENVELOPE_JSON.exists():
        pytest.skip("results/g1_wbc_envelope/envelope.json not present")
    caps = ControllerCapabilities.from_envelope_json()
    assert caps.executable and caps.source.endswith("envelope.json")
    assert caps.height_range[1] == pytest.approx(0.74) and caps.height_range[0] > 0.2
    missing = ControllerCapabilities.from_envelope_json("no/such/file.json")
    assert not missing.executable and "unmeasured" in missing.source


def test_clean_execution_completes_and_traces_commanded_vs_achieved(pm):
    backend, plant, sv = make(pm)
    report = backend.execute(target(arm_shift=0.2), monitor=sv)
    assert report.completed and not report.stops and not report.refused
    row = report.trace[-1]
    assert row["cmd_height"] == pytest.approx(0.6)
    assert {"ach_height", "ach_rpy", "ach_hand_left", "com_margin", "torque_ratio"} <= set(row)
    assert plant.c.height == pytest.approx(0.6)
    assert np.allclose(plant.c.arm_target[0], -0.2)


@pytest.mark.parametrize(
    "name,script,reason",
    [
        ("joint", lambda t, p: p.q.__setitem__(3, 2.95) if t == 3 else None, "joint position"),
        ("speed", lambda t, p: p.dq.__setitem__(1, 12.0) if t == 3 else None, "joint velocity"),
        ("tilt", lambda t, p: setattr(p, "quat", _tilted(0.4)) if t == 3 else None, "pelvis tilt"),
        ("rate", lambda t, p: setattr(p, "gyro", np.array([2.0, 0, 0])) if t == 3 else None,
         "tilt rate"),
        ("force", lambda t, p: setattr(p, "force", 80.0) if t == 3 else None, "contact force"),
        ("foot", lambda t, p: setattr(p, "feet", (True, False)) if t == 3 else None,
         "foot support"),
        ("torque", lambda t, p: p.tau.__setitem__(slice(None), 1e3) if t == 3 else None,
         "torque limit"),
    ],
)  # fmt: skip
def test_each_violation_latches_and_holds(pm, name, script, reason):
    backend, plant, sv = make(pm, script, limits=sup.SafetyLimits(contact_force=50.0))
    report = backend.execute(target(arm_shift=0.5), monitor=sv)
    assert not report.completed and report.stops and reason in report.stops[0].reason
    assert sv.latched is not None
    # HOLD: velocity zero, references frozen at the applied command / measured arm joints,
    # policy not damped (the controller's fall latch is untouched).
    c = plant.c
    assert np.allclose(c.command, 0) and c.stopped is None
    assert backend.held is not None
    assert c.height == pytest.approx(c.posture_calls[-1][0])
    assert np.allclose(c.arm_target, plant.q[N_ACTION:])
    n = plant.ticks
    again = backend.execute(target())
    assert again.refused and "held" in again.refused and plant.ticks == n


def _tilted(angle):
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def _stopped_backend(pm):
    backend, plant, sv = make(
        pm, lambda t, p: setattr(p, "gyro", np.array([2.0, 0, 0])) if t == 12 else None
    )
    backend.execute(target(arm_shift=0.5), monitor=sv)
    plant.gyro[:] = 0.0
    return backend, plant, sv


def test_recover_reverses_the_recorded_trajectory_in_a_valid_support_state(pm):
    backend, plant, sv = _stopped_backend(pm)
    assert backend.held
    result = sv.recover(backend)
    assert result.ok, result.refused
    assert backend.held is None and sv.latched is None
    assert plant.c.height == pytest.approx(0.74) and np.allclose(plant.c.arm_target, 0)
    # the replay is at half the rate limit
    h = [h for h, _ in plant.c.posture_calls[-40:]]
    assert np.abs(np.diff(h)).max() <= 0.5 * CAPS.height_rate * TICK + 1e-9


def test_recover_refuses_in_invalid_support_states(pm):
    backend, plant, sv = _stopped_backend(pm)
    plant.feet = (True, False)
    assert "foot" in sv.recover(backend).refused
    plant.feet = (True, True)
    plant.quat = _tilted(0.3)
    assert "tilt" in sv.recover(backend).refused
    plant.quat = np.array([1.0, 0, 0, 0])
    plant.gyro = np.array([0.0, 0.5, 0.0])
    assert "tilt" in sv.recover(backend).refused
    plant.gyro[:] = 0.0
    assert backend.held is not None  # a refusal leaves the hold in place
    # without a foot-contact signal support is unverifiable
    backend._feet_contact = None
    assert "unverifiable" in sv.recover(backend).refused
    backend._feet_contact = lambda: plant.feet
    # CoM outside the support polygon
    real = pm.kinematic_state
    pm.kinematic_state = lambda *a, **k: {**real(*a, **k), "com_margin": -0.02}
    try:
        assert "CoM" in sv.recover(backend).refused
    finally:
        pm.kinematic_state = real
    assert sv.recover(backend).ok


def test_recover_refuses_after_a_controller_fall_latch(pm):
    backend, plant, sv = _stopped_backend(pm)
    plant.c.stopped = "fall: pelvis tilt 40 deg"
    assert "fall" in sv.recover(backend).refused


def test_controller_fall_is_reported_as_a_stop_without_hold(pm):
    backend, plant, sv = make(pm, lambda t, p: setattr(p.c, "stopped", "fall") if t == 4 else None)
    report = backend.execute(target(), monitor=sv)
    assert report.stops and report.stops[0].reason == "fall" and backend.held is None


def test_stop_event_is_data():
    e = StopEvent(1.0, "x", {"a": 1.0})
    assert e.reason == "x" and e.detail["a"] == 1.0


def test_joint_reference_targets_are_refused_by_the_posture_only_backend(pm):
    backend, plant, _ = make(pm)
    assert backend.capabilities().request_kinds == frozenset({"posture"})
    ref = tuple((0.1 * k, np.zeros(29)) for k in range(5))
    t = target()
    t = WholeBodyTarget(
        t.q, t.stance, t.hands, t.pelvis_height, t.torso_rpy, t.arm_joints, ref, ("j",) * 29
    )
    report = backend.execute(t)
    assert report.refused and "joint_reference unsupported" in report.refused
    assert plant.ticks == 0
    both = ControllerCapabilities(
        "sonic-like", frozenset({"posture", "joint_reference"}), (0.3, 0.74), (-1,) * 3, (1,) * 3,
        measured=True, source="test",
    )  # fmt: skip
    assert both.supports("joint_reference") and not CAPS.supports("joint_reference")


def test_arm_feedforward_is_a_backend_option_and_the_controller_is_untouched(pm):
    backend, _, _ = make(pm)
    assert backend.arm_feedforward is True
    t = target()
    plain = WBCBackend(backend.env, backend.controller, CAPS, arm_feedforward=False)
    assert np.array_equal(plain._arm_feedforward(t), t.arm_joints)
    import dream.robot.g1_wbc as w

    src = open(w.__file__).read()
    assert "feedforward" not in src.lower()  # WBCLocomotion itself has no such term
