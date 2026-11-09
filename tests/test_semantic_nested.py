"""Nested-object handling in SemanticMap on synthetic depth and boxes (no simulator).

Scene: a camera at 1.4 m looks at a cabinet (floor to 0.85 m), a thin countertop on top of it
and a sink-sized object on the countertop. The detector's "countertop" box spans the counter
and therefore contains the sink's pixels, as Qwen's boxes do on the Isaac kitchen."""

import numpy as np
import pytest

from dream.perception.geometry import calibration_from_look, project
from dream.perception.occupancy import OccupancyGrid
from dream.perception.semantic import NestedOptions, SemanticMap
from dream.types import RGBDFrame, SemanticDetection

W, H = 160, 120
CABINET = ([2.0, -1.5, 0.0], [2.6, 1.5, 0.85])
COUNTER = ([1.95, -1.5, 0.85], [2.6, 1.5, 0.9])
SINK = ([2.1, -0.3, 0.9], [2.5, 0.3, 1.0])


def render(cal, boxes):
    """Optical-axis depth and box index per pixel of axis-aligned boxes (slab method)."""
    u = np.arange(W) + 0.5 - W / 2
    v = np.arange(H) + 0.5 - H / 2
    rays = np.stack(
        np.broadcast_arrays(u[None, :] / cal.focal, -v[:, None] / cal.focal, -np.ones((H, W))), -1
    )
    dirs = rays @ cal.rotation.T
    ids = np.zeros((H, W), int)
    best = np.full((H, W), np.inf)
    for k, (lo, hi) in enumerate(boxes, 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (np.asarray(lo, float) - cal.position) / dirs
            t2 = (np.asarray(hi, float) - cal.position) / dirs
        tmin, tmax = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < best)
        best, ids = np.where(hit, tmin, best), np.where(hit, k, ids)
    depth = np.where(np.isfinite(best), best * (dirs @ -cal.rotation[:, 2]), np.nan)
    return depth.astype(np.float32), ids


def bounding_mask(selected):
    ys, xs = np.nonzero(selected)
    m = np.zeros((H, W), bool)
    m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return m


def counter_frame(t=0.0, camera_x=0.0):
    cal = calibration_from_look([camera_x, 0, 1.4], [1, 0, -0.35], [0, 0, 1], 58.0, W, H)
    depth, ids = render(cal, [CABINET, COUNTER, SINK])
    dets = (
        SemanticDetection("kitchen_cabinet", 0.9, bounding_mask(ids == 1), "t"),
        SemanticDetection("countertop", 0.9, bounding_mask((ids == 2) | (ids == 3)), "t"),
        SemanticDetection("sink", 0.9, bounding_mask(ids == 3), "t"),
    )
    return RGBDFrame(t, np.zeros((H, W, 3), np.uint8), depth, np.zeros((H, W), bool), cal, dets)


def tracks_of(options, frames):
    sm = SemanticMap(OccupancyGrid(), options)
    for f in frames:
        sm.update(f)
    return sm.tracks


def test_calibration_from_look_puts_the_forward_point_at_the_image_centre():
    cal = calibration_from_look([1, 2, 1.5], [1, 1, -0.5], [0, 0, 1], 58.0, W, H)
    ahead = cal.position + 3 * np.array([1, 1, -0.5]) / np.linalg.norm([1, 1, -0.5])
    assert project(cal, ahead) == pytest.approx([W / 2, H / 2], abs=1e-6)
    right = ahead + cal.rotation[:, 0]
    assert project(cal, right)[0] > W / 2  # +x of the camera is image right


def test_default_map_fuses_cabinet_countertop_and_sink_into_one_track():
    (track,) = tracks_of(None, [counter_frame()])
    assert set(track.labels) == {"kitchen_cabinet", "countertop", "sink"}
    assert track.purity("sink") == pytest.approx(1 / 3)


def test_nested_option_separates_them_and_cleans_the_countertop_points():
    tracks = tracks_of(NestedOptions(), [counter_frame()])
    assert sorted(tuple(t.labels) for t in tracks) == [
        ("countertop",),
        ("kitchen_cabinet",),
        ("sink",),
    ]
    assert all(t.purity(t.top_label()) == 1.0 for t in tracks)
    counter = next(t for t in tracks if t.top_label() == "countertop")
    assert counter.hi[2] - counter.lo[2] < 0.06  # sink points no longer stretch it upward
    sink = next(t for t in tracks if t.top_label() == "sink")
    assert sink.position[2] == pytest.approx(0.95, abs=0.06) and sink.hi[1] - sink.lo[1] < 0.8


def test_each_half_of_the_option_is_needed():
    gate_only = tracks_of(NestedOptions(exclude_contained=False), [counter_frame()])
    exclude_only = tracks_of(NestedOptions(gate_extent=False), [counter_frame()])
    assert len(gate_only) == 3 and len(exclude_only) == 1
    counter = next(t for t in gate_only if t.top_label() == "countertop")
    assert counter.hi[2] - counter.lo[2] > 0.1  # gating alone leaves sink points in the countertop


def test_purity_holds_over_several_views():
    frames = [counter_frame(t, x) for t, x in enumerate((0.0, 0.2, -0.2, 0.4))]
    tracks = tracks_of(NestedOptions(), frames)
    assert len(tracks) == 3 and sorted(t.views for t in tracks) == [4, 4, 4]
    assert all(t.purity(t.top_label()) == 1.0 for t in tracks)
    (fused,) = tracks_of(None, frames)
    assert fused.purity("sink") == pytest.approx(1 / 3)


def test_association_stays_label_agnostic_for_one_object_seen_twice():
    cal = calibration_from_look([0, 0, 1.4], [1, 0, -0.35], [0, 0, 1], 58.0, W, H)
    depth, ids = render(cal, [SINK])
    mask = bounding_mask(ids == 1)
    frames = [
        RGBDFrame(
            t,
            np.zeros((H, W, 3), np.uint8),
            depth,
            np.zeros((H, W), bool),
            cal,
            (SemanticDetection(label, 0.9, mask, "t"),),
        )  # fmt: skip
        for t, label in enumerate(("sink", "bathtub"))
    ]
    (track,) = tracks_of(NestedOptions(), frames)
    assert track.labels == {"sink": 1, "bathtub": 1}  # a changed label does not split a track


def test_equal_sized_overlapping_detections_are_not_nested():
    cal = calibration_from_look([0, 0, 1.4], [1, 0, -0.35], [0, 0, 1], 58.0, W, H)
    depth, ids = render(cal, [SINK])
    mask = bounding_mask(ids == 1)
    dets = (SemanticDetection("sink", 0.9, mask, "t"), SemanticDetection("basin", 0.8, mask, "t"))
    frame = RGBDFrame(0.0, np.zeros((H, W, 3), np.uint8), depth, np.zeros((H, W), bool), cal, dets)
    sm = SemanticMap(OccupancyGrid(), NestedOptions())
    masks = sm._region_masks(dets, np.isfinite(depth))
    assert masks[0].sum() == masks[1].sum() == (mask & np.isfinite(depth)).sum()
    assert len(tracks_of(NestedOptions(), [frame])) == 1


def test_a_partial_view_of_a_large_object_is_a_known_cost_of_the_footprint_gate():
    """A track built from a small fragment of a large surface refuses the later full view.
    This fragmentation is why the option is off by default pending a fresh-seed check."""
    cal = calibration_from_look([0, 0, 1.4], [1, 0, -0.35], [0, 0, 1], 58.0, W, H)
    depth, ids = render(cal, [COUNTER])
    full = bounding_mask(ids == 1)
    part = full.copy()
    part[:, :70] = False  # a central sliver of the counter
    part[:, 90:] = False
    frames = [
        RGBDFrame(
            t,
            np.zeros((H, W, 3), np.uint8),
            depth,
            np.zeros((H, W), bool),
            cal,
            (SemanticDetection("countertop", 0.9, m, "t"),),
        )  # fmt: skip
        for t, m in enumerate((part, full))
    ]
    assert len(tracks_of(None, frames)) == 1
    assert len(tracks_of(NestedOptions(), frames)) == 2
