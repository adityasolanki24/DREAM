"""Physical contracts for the procedural R1 object fixtures."""

import mujoco
import numpy as np
import pytest

from dream.eval.g1 import lowest_collision_point_z
from dream.perception.geometry import observed_lower_bound_z
from dream.sim.g1_grasp_fixtures import add_grasp_object, oracle_object_bounds


@pytest.mark.parametrize(("shape", "parts"), [("bottle", 2), ("mug", 16)])
def test_compound_fixture_has_one_rigid_body_and_the_requested_mass(shape, parts):
    spec = mujoco.MjSpec()
    add_grasp_object(spec, shape, (0.0, 0.0, 0.10), np.array([0.03, 0.03, 0.04]), 0.15)
    model = spec.compile()
    body = model.body("box")
    assert model.njnt == 1
    assert np.count_nonzero(model.geom_bodyid == body.id) == parts
    assert body.mass[0] == pytest.approx(0.15)


def test_configured_surface_friction_is_applied_to_the_contact():
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.1],
        friction=[1.0, 0.005, 0.0001],
    )
    add_grasp_object(spec, "sphere", (0.0, 0.0, 0.029), np.full(3, 0.03), 0.10, 0.7)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    assert data.contact[0].friction[:2] == pytest.approx([0.7, 0.7])


def test_lowest_point_includes_the_handle_of_a_tilted_mug():
    spec = mujoco.MjSpec()
    add_grasp_object(spec, "mug", (0.0, 0.0, 0.2), np.array([0.03, 0.03, 0.04]), 0.15)
    model = spec.compile()
    data = mujoco.MjData(model)
    # Rotate +y to -z, putting the outside of the handle below the mug wall.
    data.qpos[3:7] = [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    geoms = np.flatnonzero(model.geom_bodyid == model.body("box").id)
    assert lowest_collision_point_z(model, data, geoms) == pytest.approx(0.2 - 1.72 * 0.03)


@pytest.mark.parametrize("shape", ["box", "cylinder", "sphere", "bottle", "mug"])
def test_oracle_observation_bounds_enclose_all_rotated_fixture_parts(shape):
    spec = mujoco.MjSpec()
    add_grasp_object(spec, shape, (0.0, 0.0, 0.2), np.full(3, 0.03), 0.15)
    model = spec.compile()
    data = mujoco.MjData(model)
    body = model.body("box").id
    points = oracle_object_bounds(model, body)
    geoms = np.flatnonzero(model.geom_bodyid == body)
    for angle in (0.0, 0.7, 1.5, 2.5):
        data.qpos[3:7] = [np.cos(angle / 2), -np.sin(angle / 2), 0.0, 0.0]
        mujoco.mj_forward(model, data)
        lower = observed_lower_bound_z(data.xpos[body], data.xmat[body].reshape(3, 3), points)
        assert lower <= lowest_collision_point_z(model, data, geoms) + 1e-12
