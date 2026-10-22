"""Contracts for the RGB-D camera and the vision perception backend.

Ground truth appears only here, as the evaluator; dream/perception never sees it
(tests/test_isolation.py).
"""

import numpy as np
import pytest

from dream.perception.detect import color_name, detect
from dream.perception.geometry import backproject, project
from dream.perception.vision import VisionPerception
from dream.sim.tabletop import TabletopSpec, sample_tabletop
from dream.sim.tabletop_env import TabletopEnv


def _env(seed=3, spec=None, **kw):
    spec = spec or sample_tabletop(np.random.default_rng(seed))
    return spec, TabletopEnv(spec, np.random.default_rng(seed + 1), **kw)


def _vision(env, seed=0):
    return VisionPerception(env.capture_frame, lambda: env.time, np.random.default_rng(seed))


def test_frame_is_a_calibrated_rgbd_image_and_reconstructs_a_known_point():
    spec, env = _env()
    frame = env.capture_frame()
    cal = frame.calibration
    assert frame.rgb.shape == (cal.height, cal.width, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (cal.height, cal.width) and np.isfinite(frame.depth).all()
    pts = backproject(frame)
    for obj in spec.objects:  # the centre of each cube's top face, projected and reconstructed
        top = env.data.body(obj.name).xpos + [0, 0, obj.half_size]
        u, v = project(cal, top).astype(int)
        assert np.linalg.norm(pts[v, u] - top) < 0.008  # depth noise ~2.5 mm, 1 px ~3 mm
    table = pts[np.abs(pts[..., 2]) < 0.01]
    assert len(table) > 0.3 * pts[..., 0].size  # most of the image is the table, at z = 0


def test_capturing_frames_does_not_change_the_oracle_tracker_or_the_physics():
    _, a = _env()
    _, b = _env()
    a.capture_frame()
    a.step(50)
    b.step(50)
    assert np.array_equal(a.data.qpos, b.data.qpos)
    oa, ob = a.observe_objects(), b.observe_objects()
    assert all(np.array_equal(x.position, y.position) for x, y in zip(oa, ob, strict=True))


@pytest.mark.parametrize(
    "seed", [3, pytest.param(5, marks=pytest.mark.slow), pytest.param(8, marks=pytest.mark.slow)]
)
def test_detection_finds_each_cube_and_the_tray_from_the_image(seed):
    spec, env = _env(seed)
    dets, _ = detect(env.capture_frame(), np.random.default_rng(0))
    objects = {d.color: d for d in dets if d.kind == "object"}
    assert sorted(objects) == sorted(o.name for o in spec.objects)  # colours named from pixels
    for o in spec.objects:
        d = objects[o.name]
        assert np.linalg.norm(d.position - env.data.body(o.name).xpos) < 0.008
        assert abs(d.size - 2 * o.half_size) < 0.008
    (tray,) = [d for d in dets if d.kind == "container"]
    assert np.linalg.norm(tray.position[:2] - np.array(spec.tray.xy)) < 0.008
    assert abs(tray.inner_half - spec.tray.inner_half) < 0.008


def test_empty_table_gives_no_detections():
    _, env = _env(spec=TabletopSpec(objects=(), tray=None))
    dets, _ = detect(env.capture_frame(), np.random.default_rng(0))
    assert dets == []


def test_color_names_come_from_hue_not_from_the_scene():
    assert color_name(np.array([200, 30, 30])) == "red"
    assert color_name(np.array([30, 160, 50])) == "green"
    assert color_name(np.array([40, 60, 200])) == "blue"
    assert color_name(np.array([220, 200, 40])) == "yellow"
    assert color_name(np.array([120, 120, 125])) == "grey"


def test_world_model_keeps_a_hidden_object_as_last_known_not_as_seen():
    """When the camera loses an object, it is reported as not visible at its last-known place;
    it is never moved to where it really is."""
    spec, env = _env()
    vis = _vision(env)
    before = {o.name: o for o in vis.observe_objects()}
    assert all(o.visible for o in before.values())
    red = env.model.body("red")
    adr = env.model.jnt_qposadr[red.jntadr[0]]
    env.data.qpos[adr : adr + 3] = [3.0, 3.0, 0.02]  # scene change outside the camera's view
    env.step(1)
    after = {o.name: o for o in vis.observe_objects()}
    assert not after["red"].visible
    assert np.allclose(after["red"].position, before["red"].position)
    assert after["red"].last_seen < env.time
    assert after["blue"].visible


@pytest.mark.parametrize("kind", ["drawer", "lid"])
def test_vision_finds_the_closed_compartment_and_measures_its_handle(kind):
    """The handle position is measured from the image (near, not equal, to the truth), the
    orientation is declared unobserved, and a closed compartment offers no opening."""
    from dream.sim.tabletop import sample_compartment_scene

    spec = sample_compartment_scene(kind, np.random.default_rng([3, 0]))
    env = TabletopEnv(spec, np.random.default_rng(4))
    vis = _vision(env)
    (a,) = vis.observe_articulated()
    assert a.name == "compartment" and a.handle_visible and not a.orientation_observed
    true = env.data.site("handle").xpos
    assert 0 < np.linalg.norm(a.handle_pos - true) < 0.02
    front = a.handle_pos[:2] - a.box_centre[:2]
    assert np.dot(front, spec.compartment.front_dir[:2]) > 0  # the handle side is the front
    assert np.allclose(np.abs(a.box_half[:2]), 0.098, atol=0.02)
    assert "compartment" not in vis.fixtures()


@pytest.mark.slow
@pytest.mark.parametrize("task", ["tray", "stack"])
def test_camera_driven_pick_and_place_is_true_in_the_world(task):
    from dream.eval.manipulation import object_in_tray, object_on_top
    from dream.runtime.agent import TaskAgent
    from dream.tools.goals import InRegion, OnTop
    from dream.tools.toolbox import Toolbox

    _, env = _env(7)
    vis = _vision(env)
    goal = InRegion("red", "tray") if task == "tray" else OnTop("red", "blue")
    ep = TaskAgent(Toolbox(env.embodiment(), vis)).run(goal)
    env.step(250)
    truth = object_in_tray(env, "red") if task == "tray" else object_on_top(env, "red", "blue")
    assert ep.success_claimed and truth
    assert vis.frames >= 3  # decisions came from several camera frames


def test_averaging_handle_observations_keeps_what_perception_cannot_measure():
    """Regression: averaging rebuilt the observation with defaults, so a camera's
    'orientation not observed' and 'handle not visible' were silently lost, and lid rotations
    were treated as measured zeros."""
    import dream.skills.articulation as art
    from dream.types import ArticulatedObservation

    class Fixed:
        def observe_articulated(self):
            return [
                ArticulatedObservation(
                    "c",
                    np.zeros(3),
                    np.array([1.0, 0, 0, 0]),
                    np.zeros(3),
                    np.ones(3),
                    handle_half=np.ones(3),
                    orientation_observed=False,
                    handle_visible=False,
                    position_std=0.025,
                )
            ]

    a = art.observe_handle_averaged(Fixed(), "c")
    assert not a.orientation_observed and not a.handle_visible
    assert a.handle_half is not None and a.position_std == 0.025  # identical frames: no gain
