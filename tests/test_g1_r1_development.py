"""Protect the R1 cohort from inadvertent changes to its breadth or balance."""

import json
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("mink")
pytest.importorskip("daqp")
from experiments.g1_r1_development import BASE, episodes


def test_r1_cohort_is_repeatable_and_stratified():
    spec = json.loads(BASE.with_name("g1_r1_development.json").read_text(encoding="utf-8"))
    cases = episodes(spec)
    assert cases == episodes(spec)
    assert len(cases) == spec["gate"]["episode_count"] == 50
    assert spec["gate"]["minimum_verified_fraction"] == 0.70
    assert len({case["seed"] for case in cases}) == len(cases)
    assert Counter(case["object_shape"] for case in cases) == dict.fromkeys(spec["shapes"], 10)
    assert set(case["object_shape"] for case in cases[:5]) == set(spec["shapes"])
    for shape in spec["shapes"]:
        family = [case for case in cases if case["object_shape"] == shape]
        widths = np.array([2 * case["object_half_extents_m"][1] for case in family])
        mass = np.array([case["object_mass_kg"] for case in family])
        height = np.array([case["table_top_m"] for case in family])
        friction = np.array([case["object_sliding_friction"] for case in family])
        distance = np.array(
            [
                np.max(
                    np.abs(
                        np.subtract(case["starting_stance_xy_yaw"][:2], case["table_center_xy_m"])
                    )
                    - case["table_half_size_m"][:2]
                )
                for case in family
            ]
        )
        for key, values in zip(
            ("width_m", "mass_kg", "table_top_m", "sliding_friction", "start_distance_m"),
            (widths, mass, height, friction, distance),
            strict=True,
        ):
            low, high = spec[key]
            cuts = (values - low) * 10 / (high - low)
            assert sorted(np.floor(cuts).astype(int)) == list(range(10))


def test_measured_approach_follows_translation_and_shortest_wrist_rotation_together():
    from scipy.spatial.transform import Rotation

    from experiments.g1_grasp_bringup import GRASP_OFFSET, _hand_pose_step

    start = np.array([0.15, -0.10, 0.85])
    goal = np.array([0.23, -0.04, 0.87])
    for angles in ((178, -178), (0, 105), (15, 15)):
        current, desired = (Rotation.from_euler("z", a, degrees=True).as_matrix() for a in angles)
        point, rotation, distance = _hand_pose_step(start, current, goal, desired, 0.035)
        angle = Rotation.from_matrix(rotation @ current.T).magnitude()
        assert np.linalg.norm(point - start) <= 0.035 + 1e-12
        assert angle * np.linalg.norm(GRASP_OFFSET) <= 0.035 + 1e-12
        assert np.linalg.norm(point - goal) < np.linalg.norm(start - goal)
        _, _, remaining = _hand_pose_step(point, rotation, goal, desired, 0.035)
        assert remaining == pytest.approx(max(0, distance - 0.035))
    point, rotation, distance = _hand_pose_step(goal, current, goal, current, 0.035)
    assert distance == pytest.approx(0)
    np.testing.assert_allclose(point, goal)
    np.testing.assert_allclose(rotation, current)
