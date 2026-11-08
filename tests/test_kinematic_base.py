"""The kinematic surrogate base: integration, collisions, odometry frame, camera pose vs the G1."""

import time
from dataclasses import replace

import numpy as np
import pytest

import dream.sim.home as H
from dream.robot.g1_wbc import MAX_COMMAND
from dream.sim.g1_mobile import OdometryNoise
from dream.sim.kinematic_base import (
    DT,
    HEAD_CAMERA,
    LAG_TAU,
    KinematicBase,
    head_camera_pose,
)

NO_NOISE = OdometryNoise(0.0, 0.0, 0.0)
HALF_W, HALF_D = 4.0, 3.0


def _empty_home(start=(0.0, 0.0, 0.0)):
    t = 0.05
    walls = (
        H.WallSeg("px", (HALF_W + t, 0), (t, HALF_D + 2 * t)),
        H.WallSeg("nx", (-HALF_W - t, 0), (t, HALF_D + 2 * t)),
        H.WallSeg("py", (0, HALF_D + t), (HALF_W, t)),
        H.WallSeg("ny", (0, -HALF_D - t), (HALF_W, t)),
    )
    room = H.RoomBox(0, "storage", (-HALF_W, -HALF_D), (HALF_W, HALF_D))
    return H.HomeSpec(8.0, 6.0, (room,), (), (), start, "E", "refrigerator", "", walls=walls)


def _base(start=(0.0, 0.0, 0.0), noise=NO_NOISE, **kw):
    return KinematicBase(_empty_home(start), np.random.default_rng(0), noise, **kw)


def test_straight_walk_matches_the_lagged_integral():
    base = _base()
    assert base.walk(0.4, 0.0, 0.0, 5.0)
    n = round(5.0 / DT)
    k = np.arange(1, n + 1)
    expected = 0.4 * np.sum(1 - np.exp(-k * DT / LAG_TAU)) * DT
    assert base.true_pose()[0] == pytest.approx(expected, abs=1e-6)
    assert base.true_pose()[1] == pytest.approx(0.0, abs=1e-9)
    assert base.odometry()[0] == pytest.approx(expected, abs=1e-6)  # noise-free odometry


def test_turn_and_strafe_integrate_in_the_body_frame():
    base = _base((0.0, 0.0, np.pi / 2))
    base.walk(0.0, 0.2, 0.0, 2.0)  # left of a robot facing +y is -x
    x, y, yaw = base.true_pose()
    assert x < -0.2 and abs(y) < 1e-9 and yaw == pytest.approx(np.pi / 2)
    base = _base()
    base.walk(0.0, 0.0, 0.5, 4.0)
    k = np.arange(1, round(4.0 / DT) + 1)
    assert base.true_pose()[2] == pytest.approx(
        0.5 * np.sum(1 - np.exp(-k * DT / LAG_TAU)) * DT, abs=1e-6
    )


def test_commands_outside_the_g1_limits_are_rejected_like_the_controller():
    base = _base()
    for bad in ([0.6, 0, 0], [0, 0.31, 0], [0, 0, -0.7], [np.nan, 0, 0]):
        with pytest.raises(ValueError):
            base.walk(*bad, 1.0)
    assert base.time == 0.0
    assert base.walk(*MAX_COMMAND, 0.1)


def test_time_advances_by_commanded_seconds_and_no_stop():
    base = _base()
    base.walk(0.1, 0, 0, 1.0)
    base.walk(0.0, 0, 0, 0.5)
    assert base.time == pytest.approx(1.5)
    assert base.stop_reason is None


def test_wall_blocks_the_capsule_and_contacts_are_logged():
    base = _base((2.0, 0.0, 0.0))
    ok = base.walk(0.5, 0.0, 0.0, 8.0)
    assert ok and base.stop_reason is None
    x = base.true_pose()[0]
    assert HALF_W - 0.25 - 0.03 < x <= HALF_W - 0.25 + 1e-6  # stopped one radius short
    log = base.contact_log()
    assert log and all(name == "wall:px" for _, name, _ in log)
    assert base.time == pytest.approx(8.0)
    base.walk(0.0, 0.0, 0.5, 1.0)  # can still turn in place against the wall
    assert base.true_pose()[2] > 0.1


def test_furniture_blocks_too():
    home = _empty_home((0.0, 0.0, 0.0))
    half, parts = H.make_object(np.random.default_rng(0), "refrigerator", "refrigerator_0")
    fridge = H.ObjectSpec("refrigerator_0", "refrigerator", 0, (2.0, 0.0), np.pi, half, parts)
    base = KinematicBase(replace(home, objects=(fridge,)), np.random.default_rng(0), NO_NOISE)
    base.walk(0.5, 0, 0, 8.0)
    assert base.true_pose()[0] < 2.0 - half[0] - 0.2
    assert base.contact_log()[0][1].startswith("sem:refrigerator_0:")


def test_odometry_noise_is_applied_and_seeded():
    runs = []
    for _ in range(2):
        base = KinematicBase(_empty_home(), np.random.default_rng(3), OdometryNoise())
        base.walk(0.3, 0, 0, 6.0)
        runs.append(base.odometry())
    assert np.allclose(runs[0], runs[1])
    truth = KinematicBase(_empty_home(), np.random.default_rng(3), NO_NOISE)
    truth.walk(0.3, 0, 0, 6.0)
    assert not np.allclose(runs[0], truth.odometry())
    assert np.hypot(*(runs[0][:2] - truth.odometry()[:2])) < 0.3


@pytest.mark.parametrize("start", [(1.5, 0.5, 0.4), (-1.0, -1.0, -0.3)])
def test_odometry_frame_calibration_lands_on_the_true_wall(start):
    """Back-project depth with the frame's calibration (odometry frame), map odometry to the
    world with the true start pose, and the wall pixels land on the wall plane."""
    base = _base(start)
    base.walk(0.3, 0.0, 0.0, 1.0)
    frame = base.capture()
    cal = frame.calibration
    v, u = np.nonzero(np.isfinite(frame.depth))
    d = frame.depth[v, u].astype(float)
    cam = np.stack(
        [
            (u + 0.5 - cal.width / 2) / cal.focal * d,
            -(v + 0.5 - cal.height / 2) / cal.focal * d,
            -d,
        ],
        1,
    )
    p_odom = cal.position + cam @ cal.rotation.T
    c, s = np.cos(start[2]), np.sin(start[2])
    world = p_odom[:, :2] @ np.array([[c, s], [-s, c]]) + start[:2]
    wall = p_odom[:, 2] > 0.4  # above the floor
    assert wall.sum() > 2000
    dist = np.min(
        [np.abs(world[wall, 0] - HALF_W), np.abs(world[wall, 0] + HALF_W),
         np.abs(world[wall, 1] - HALF_D), np.abs(world[wall, 1] + HALF_D)], axis=0,
    )  # fmt: skip
    assert np.percentile(dist, 99) < 0.05


def test_head_camera_pose_matches_the_g1_forward_kinematics():
    from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1

    env = G1MobileEnv(build_mobile_g1(), np.random.default_rng(0))
    for x, y, yaw in [(0.0, 0.0, 0.0), (1.2, -0.7, 0.9)]:
        env.reset(x, y, yaw)
        cam = env.data.cam(HEAD_CAMERA.name)
        pos, rot = head_camera_pose(x, y, yaw)
        assert np.linalg.norm(pos - cam.xpos) < 0.02
        r = rot.T @ cam.xmat.reshape(3, 3)
        assert np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))) < 1.0
        assert env.model.cam(HEAD_CAMERA.name).fovy[0] == pytest.approx(HEAD_CAMERA.fovy_deg)


def test_frames_have_the_g1_frame_contract_and_no_detections_by_default():
    base = _base()
    frame = base.capture()
    assert frame.rgb.shape == (240, 320, 3) and frame.depth.shape == (240, 320)
    assert frame.self_mask.shape == (240, 320) and not frame.self_mask.any()
    assert frame.detections is None  # no detector ran
    assert np.nanmax(frame.depth) <= 3.1  # range limit plus depth noise (sigma 15 mm at 3 m)


def test_capture_is_fast_enough_for_large_sample_runs():
    base = _base()
    base.capture()
    times = []
    for _ in range(30):
        t = time.perf_counter()
        base.capture()
        times.append(time.perf_counter() - t)
    # target is < 15 ms on the development laptop; the bound allows for a loaded CI machine
    assert np.median(times) < 0.06
