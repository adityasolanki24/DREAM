"""The oracle-assisted semantic detector: pixel fidelity and the configured error rates."""

import numpy as np
import pytest

import dream.sim.home as H
from dream.sim.camera import MAX_RANGE
from dream.sim.kinematic_base import HEAD_CAMERA, KinematicBase
from dream.sim.semantic_sensor import (
    CONFUSION_GROUPS,
    DEGRADED,
    NOMINAL,
    SOURCE,
    SemanticCamera,
    SemanticG1Base,
    SemanticNoise,
    confusable_with,
)
from dream.types import SemanticDetection

CERTAIN = SemanticNoise(
    a50=1.0, slope=0.1, confusion_near=0.0, confusion_far=0.0, false_positive=0.0
)


def _home(scenario="A", seed=4):
    return H.sample_home(np.random.default_rng(seed), scenario)


def _camera(home, noise, seed=0):
    base = KinematicBase(home, np.random.default_rng(0))
    index = H.semantic_index(base.model, home)
    return base, SemanticCamera(base.model, HEAD_CAMERA, index, np.random.default_rng(seed), noise)


def _synthetic(base, cam, home, area, range_m=2.0, cls="refrigerator"):
    """Segmentation with one instance occupying ``area`` pixels at ``range_m``, no other scene."""
    geom = next(g for g, (_, c) in H.semantic_index(base.model, home).items() if c == cls)
    seg = np.full((240, 320), -1, dtype=np.int64)
    seg.flat[:area] = geom
    depth = np.full((240, 320), np.nan, dtype=np.float32)
    depth.flat[:area] = range_m
    return seg, depth


def test_probability_curves_match_the_configuration():
    assert NOMINAL.detection_probability(NOMINAL.a50) == pytest.approx(0.5)
    assert NOMINAL.detection_probability(0) == 0.0 and NOMINAL.detection_probability(29) == 0.0
    assert NOMINAL.detection_probability(1000) > 0.99
    assert NOMINAL.confusion_probability(1.0) == pytest.approx(0.05)
    assert NOMINAL.confusion_probability(3.0) == pytest.approx(0.2)
    assert DEGRADED.confusion_probability(3.0) == pytest.approx(2 * 0.2)
    assert DEGRADED.false_positive > NOMINAL.false_positive and DEGRADED.a50 > NOMINAL.a50


def test_confusion_groups_are_symmetric_and_in_vocabulary():
    for g in CONFUSION_GROUPS:
        assert g <= set(H.CLASSES)
        for c in g:
            assert g - {c} <= set(confusable_with(c))
    assert "wardrobe" in confusable_with("refrigerator") and "dryer" in confusable_with(
        "chest_freezer"
    )
    assert "bathtub" not in confusable_with("refrigerator")


def test_synthetic_detection_rate_confusion_rate_and_labels():
    home = _home()
    base, cam = _camera(home, NOMINAL, seed=1)
    seg, depth = _synthetic(base, cam, home, 250, range_m=2.25)  # a50 pixels, mid-range
    n, hits, wrong, labels = 4000, 0, 0, set()
    for _ in range(n):
        dets = [d for d in cam.detect_from_arrays(seg, depth) if d.mask.sum() == 250]
        hits += bool(dets)
        for d in dets:
            labels.add(d.label)
            wrong += d.label != "refrigerator"
    p = 0.5
    assert abs(hits / n - p) < 4 * np.sqrt(p * (1 - p) / n)
    q = NOMINAL.confusion_probability(2.25)
    assert abs(wrong / hits - q) < 4 * np.sqrt(q * (1 - q) / hits)
    assert labels <= {"refrigerator", *confusable_with("refrigerator")}
    assert len(labels) == 4  # all three look-alikes occur


@pytest.mark.slow
def test_false_positive_rate_and_chunk_size():
    home = _home()
    for noise in (NOMINAL, DEGRADED):
        base, cam = _camera(home, noise, seed=2)
        wall = base.model.geom("wall:outer_px").id
        seg = np.full((240, 320), wall, dtype=np.int64)
        depth = np.full((240, 320), 2.0, dtype=np.float32)
        n, fp, sizes = 5000, 0, []
        for _ in range(n):
            dets = cam.detect_from_arrays(seg, depth)
            fp += len(dets)
            sizes += [int(d.mask.sum()) for d in dets]
            assert all(d.source == SOURCE and d.label in noise.vocabulary for d in dets)
        p = noise.false_positive
        assert abs(fp / n - p) < 4 * np.sqrt(p * (1 - p) / n)
        assert min(sizes) >= 300


def test_pixels_outside_range_or_without_depth_are_never_detected():
    home = _home()
    base, cam = _camera(home, CERTAIN, seed=3)
    seg, depth = _synthetic(base, cam, home, 4000, range_m=MAX_RANGE + 0.5)
    assert cam.detect_from_arrays(seg, depth) == ()
    depth[:] = np.nan
    assert cam.detect_from_arrays(seg, depth) == ()


def test_rendered_mask_equals_visible_pixels_and_agrees_with_ray_casting():
    home = _home("A", 4)
    base, cam = _camera(home, CERTAIN)
    base._place(np.array(home.start))
    frame = base.capture()
    dets = cam.detect(base.data, frame.depth)
    fridges = [d for d in dets if d.label == "refrigerator"]
    assert len(fridges) == 1 and isinstance(fridges[0], SemanticDetection)
    mask = fridges[0].mask
    assert mask.shape == (240, 320) and mask.dtype == bool
    seg = cam.segmentation(base.data)
    ids = [o.instance_id for o in H.instances_of(home, "refrigerator")]
    geoms = [g for g, (iid, _) in H.semantic_index(base.model, home).items() if iid in ids]
    valid = np.isfinite(frame.depth) & (frame.depth <= MAX_RANGE)
    assert (mask == (np.isin(seg, geoms) & valid)).all()
    rays = H.GroundTruthView(home).pixels(ids, *home.start, stride=1)
    assert abs(int(mask.sum()) - rays) < 0.08 * rays + 20


def test_no_detection_of_an_instance_hidden_behind_a_wall():
    home = _home("B", 2)  # the target is not visible at the start by construction
    base, cam = _camera(home, CERTAIN)
    ids = [o.instance_id for o in H.instances_of(home, "refrigerator")]
    assert H.GroundTruthView(home).pixels(ids, *home.start, stride=1) == 0
    frame = base.capture()
    for _ in range(20):
        assert not [d for d in cam.detect(base.data, frame.depth) if d.label == "refrigerator"]


def test_kinematic_base_attaches_a_tuple_only_when_a_detector_runs():
    home = _home("A", 4)
    base = KinematicBase(home, np.random.default_rng(0), semantic=CERTAIN)
    frame = base.capture()
    assert isinstance(frame.detections, tuple) and frame.detections
    plain = KinematicBase(home, np.random.default_rng(0)).capture()
    assert plain.detections is None
    assert np.array_equal(frame.rgb, plain.rgb)  # the detector's rng does not touch the camera's
    empty = KinematicBase(_home("B", 2), np.random.default_rng(0), semantic=CERTAIN)
    assert empty.capture().detections == () or all(
        d.label != "refrigerator" for d in empty.capture().detections
    )


@pytest.mark.slow
def test_g1_wrapper_adds_detections_without_changing_the_g1_streams():
    from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1
    from dream.sim.semantic_sensor import SemanticCamera as Cam

    home = _home("A", 4)
    model = build_mobile_g1(scene=lambda s: H.add_home(s, home))
    frames = []
    for wrap in (False, True):
        env = G1MobileEnv(model, np.random.default_rng(7))
        env.reset(*home.start)
        base = G1MobileBase(env, controller=None)
        if wrap:
            cam = Cam(
                model, HEAD_CAMERA, H.semantic_index(model, home), np.random.default_rng(1), CERTAIN
            )
            base = SemanticG1Base(base, cam)
        frames.append(base.capture())
    plain, semantic = frames
    assert plain.detections is None and isinstance(semantic.detections, tuple)
    assert np.array_equal(plain.rgb, semantic.rgb) and np.array_equal(
        plain.depth, semantic.depth, equal_nan=True
    )
    assert any(d.label == "refrigerator" and d.mask.sum() > 300 for d in semantic.detections)
