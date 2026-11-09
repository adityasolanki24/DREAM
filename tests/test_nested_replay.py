"""Pure helpers of the nested-object replay (no capture data needed)."""

import math

import numpy as np
import pytest

from experiments import nested_replay as nr

CAMERA = {
    "position": [1.0, 2.0, 1.2],
    "forward": [0.6, 0.5, -0.6],
    "up": [0.04, 0.04, 1.0],
    "fovy_deg": 58.0,
}


def test_odometry_camera_equals_world_camera_when_odometry_is_exact():
    pose = [1.06, 0.68, 0.73]
    same = nr.camera_in_odometry(CAMERA, pose, pose)
    assert same["position"] == pytest.approx(CAMERA["position"])
    assert same["forward"] == pytest.approx(CAMERA["forward"])


def test_odometry_camera_is_rigid_with_the_base():
    """Move the odometry pose by a known rigid motion: the camera moves with it."""
    true_pose, odo = [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]
    cam = nr.camera_in_odometry(CAMERA, true_pose, odo)
    # base frame position of the camera is (rotate by -yaw) of the offset from the base
    off = np.array([0.0, 0.0, CAMERA["position"][2]])
    c, s = math.cos(-0.5), math.sin(-0.5)
    expected = np.array([c * 0.0 - s * 0.0, s * 0.0 + c * 0.0, off[2]])
    assert cam["position"] == pytest.approx(expected, abs=1e-9)  # camera sits over the base
    assert np.linalg.norm(cam["forward"]) == pytest.approx(np.linalg.norm(CAMERA["forward"]))
    turned = nr.camera_in_odometry(CAMERA, true_pose, [0.0, 0.0, math.pi / 2])
    assert turned["forward"][2] == pytest.approx(CAMERA["forward"][2])  # yaw leaves pitch alone


def test_span_boxes_enclose_what_is_mounted_in_them_and_tight_boxes_do_not():
    truth = [
        {"cls": "countertop", "box": [0, 40, 100, 60], "pixels": 2000},
        {"cls": "sink", "box": [30, 20, 60, 70], "pixels": 1200},
        {"cls": "kitchen_cabinet", "box": [0, 60, 100, 120], "pixels": 5000},
        {"cls": "refrigerator", "box": [200, 0, 240, 100], "pixels": 100},  # under 400 px
    ]
    tight = dict(nr.surrogate_boxes(truth, "tight"))
    assert tight["countertop"] == [0, 40, 100, 60] and "refrigerator" not in tight
    span = dict(nr.surrogate_boxes(truth, "span"))
    assert span["countertop"] == [0, 20, 100, 70]  # countertop plus the sink
    assert span["kitchen_cabinet"] == [0, 40, 100, 120]  # cabinet plus the countertop
    assert span["sink"] == [30, 20, 60, 70]


def test_box_mask_is_the_pixel_rectangle():
    m = nr.box_mask([2.5, 1.0, 5.0, 3.2], 10, 10)
    assert m.sum() == (5 - 2) * (4 - 1) and m[1, 2] and not m[0, 2]
