"""Sensed clearance must include the observed orientation of the whole object."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dream.perception.geometry import observed_lower_bound_z


def test_a_tilted_cube_centre_does_not_establish_full_clearance():
    half = 0.04
    points = np.array(list(product((-half, half), repeat=3)))
    table_top = 0.75
    centre = np.array([0.0, 0.0, table_top + half + 0.115])
    rotation = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    bottom = observed_lower_bound_z(centre, rotation, points)
    assert centre[2] - (table_top + half) > 0.10
    assert bottom - table_top < 0.10
    assert bottom == pytest.approx(centre[2] - np.sqrt(2) * half)


def test_clearance_includes_an_offset_part_under_arbitrary_rotation():
    points = np.array([[0.0, 0.0, -0.03], [0.0, 0.12, 0.04], [0.02, 0.10, 0.04]])
    pose = np.array([-0.4, 0.7, 0.9])
    rotation = Rotation.from_euler("xyz", (1.0, -0.3, 1.7)).as_matrix()
    expected = min((rotation @ point + pose)[2] for point in points)
    assert observed_lower_bound_z(pose, rotation, points) == pytest.approx(expected)
