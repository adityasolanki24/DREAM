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
from dream.tools.toolbox import WALL_CLEARANCE


def test_an_occluded_view_never_shrinks_an_opening_but_can_widen_it():
    from dataclasses import replace

    from dream.perception.detect import Detection
    from dream.perception.world import WorldModel

    world = WorldModel()
    d = Detection(
        "box",
        "grey",
        np.array([0.5, 0.0, 0.05]),
        0.0,
        np.array([0.2, 0.2, 0.1]),
        0.003,
        False,
        True,
        opening=(np.array([0.5, 0.0, 0.01]), 0.02),
    )
    world.update([d], 0.0)
    for time, width in enumerate([0.06, 0.018, 0.08], 1):
        world.update([replace(d, occluded=True, opening=(d.opening[0], width))], time)
        assert world.tracks[0].detection.opening[1] == max(0.02, 0.06, width)


def test_a_clear_view_can_show_a_compartment_closing_again():
    from dataclasses import replace

    from dream.perception.detect import Detection
    from dream.perception.world import WorldModel

    world = WorldModel()
    d = Detection(
        "box",
        "grey",
        np.array([0.5, 0.0, 0.05]),
        0.0,
        np.array([0.2, 0.2, 0.1]),
        0.003,
        False,
        True,
        opening=(np.array([0.5, 0.0, 0.01]), 0.08),
    )
    world.update([d], 0.0)
    world.update([replace(d, opening=(d.opening[0], 0.02))], 1.0)
    assert world.tracks[0].detection.opening[1] == 0.02
    world.update([replace(d, opening=None)], 2.0)
    assert world.tracks[0].detection.opening is None


def test_old_open_space_is_not_an_indefinite_fixture_under_occlusion():
    from dataclasses import replace

    from dream.perception.detect import Detection
    from dream.perception.world import OPENING_MAX_AGE_S

    vis = VisionPerception(lambda: None, lambda: 0.0, np.random.default_rng(0))
    vis.refresh = lambda: None
    vis._clock = lambda: vis.world.time
    d = Detection(
        "box",
        "grey",
        np.array([0.5, 0.0, 0.05]),
        0.0,
        np.array([0.2, 0.2, 0.1]),
        0.003,
        False,
        True,
        opening=(np.array([0.5, 0.0, 0.01]), 0.08),
    )
    vis.world.update([d], 0.0)
    vis.world.tracks[0] = replace(vis.world.tracks[0], articulated=True)
    vis.world.update([replace(d, occluded=True, opening=None)], 1.0)
    assert "compartment" in vis.fixtures()
    vis.world.update([replace(d, occluded=True, opening=None)], OPENING_MAX_AGE_S + 1.0)
    assert "compartment" not in vis.fixtures()
    assert vis.world.tracks[0].detection.opening is not None  # retained as historical evidence


def test_opening_motion_does_not_swap_two_handles_between_receptacles():
    from dataclasses import replace

    from dream.perception.detect import Detection
    from dream.perception.world import WorldModel

    world = WorldModel()
    box = Detection(
        "box",
        "grey",
        np.array([0.4, 0.0, 0.05]),
        0.0,
        np.array([0.2, 0.2, 0.1]),
        0.003,
        False,
        True,
    )
    handle = replace(box, kind="handle", color="blue", position=np.array([0.4, -0.1, 0.1]))
    other_box = replace(box, position=np.array([0.7, 0.0, 0.05]))
    other_handle = replace(handle, color="red", position=np.array([0.7, -0.1, 0.1]))
    world.update([box, other_box, handle, other_handle], 0.0)
    receptacles = [t for t in world.tracks if t.articulated]
    assert len(receptacles) == 2
    ids = [world.handle_of(t).id for t in receptacles]
    world.update(
        [
            box,
            other_box,
            replace(handle, position=other_handle.position + [0, 0, 0.2]),
            other_handle,
        ],
        1.0,
    )
    assert [world.handle_of(t).id for t in world.tracks if t.articulated] == ids


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


def _compartment_with_cameras(kind):
    from dream.sim.camera import SIDE_CAMERA, CameraSpec
    from dream.sim.tabletop import sample_compartment_scene

    spec = sample_compartment_scene(kind, np.random.default_rng([3, 0]))
    env = TabletopEnv(spec, np.random.default_rng(4), extra_cameras=(CameraSpec(**SIDE_CAMERA),))
    vis = VisionPerception(env.frame_sources(), lambda: env.time, np.random.default_rng(0))
    return env, vis


def _put(env, name, xyz):
    import mujoco

    q = env.model.joint(f"{name}_free").qposadr[0]
    env.data.qpos[q : q + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    env.step(200)


@pytest.mark.parametrize("kind,q", [("drawer", 0.18), ("lid", 1.45)])
def test_an_object_inside_an_open_compartment_does_not_shrink_its_opening(kind, q):
    """Contents are not walls. A cube resting in the open cavity stays inside the opening the
    camera reports (the oracle tracker counts such cells as open too), so verification can find
    it there. Lid seed 1 in development: a placed cube read as "outside region" because its top
    made its own cells look closed and the square moved off it."""
    import mujoco

    env, vis = _compartment_with_cameras(kind)
    assert "compartment" not in vis.fixtures()  # seen closed first, as in a task
    j = env.model.joint("compartment_joint")
    env.data.qpos[j.qposadr[0]] = q  # scene setup: open it partly, as an opening would
    mujoco.mj_forward(env.model, env.data)
    env.step(5)
    assert "compartment" in vis.fixtures()
    centre, _ = env._visible_cavity()  # scene setup only: where to put the cube
    _put(env, "red", [centre[0], centre[1], centre[2] + 0.021])
    c, half = vis.fixtures()["compartment"]
    red = env.privileged_object_pos("red")
    # Still accessible for this cube (so the planner does not reopen it), and the cube is in
    # the region verification uses.
    assert half >= 0.02 + WALL_CLEARANCE
    assert np.all(np.abs(red[:2] - c[:2]) <= half)


def test_a_cube_on_a_closed_lid_is_not_an_opening():
    """The other side of the same rule: only contents *below* the rim count as open."""
    env, vis = _compartment_with_cameras("lid")
    lo, hi = env._compartment_aabb()
    _put(env, "red", [*((lo + hi) / 2)[:2], hi[2] + 0.021])
    assert "compartment" not in vis.fixtures()
