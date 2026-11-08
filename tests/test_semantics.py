"""Contracts of the semantic layers on synthetic maps and tracks (no simulator)."""

import json

import numpy as np
import pytest

from dream.perception.occupancy import CELL, OccupancyGrid
from dream.perception.semantic import SemanticMap, Track, detection_quality
from dream.semantics.belief import BeliefConfig, TargetBelief
from dream.semantics.graph import SemanticGraph
from dream.semantics.knowledge import (
    PSEUDO,
    UNKNOWN_ROOM,
    SemanticKnowledge,
    load_knowledge,
    uniform_knowledge,
)
from dream.semantics.rooms import RoomSegmenter, build_regions, category_weights
from dream.semantics.verify import VerifierThresholds, check_track, verify_find
from dream.types import CameraCalibration, RGBDFrame, SemanticDetection

CATS = ("kitchen", "bedroom", "bathroom", "garage")
KNOW = SemanticKnowledge(
    CATS,
    {
        "refrigerator": {"kitchen": 0.9, "garage": 0.3},
        "sink": {"kitchen": 0.8, "bathroom": 0.6},
        "countertop": {"kitchen": 0.8},
        "bed": {"bedroom": 0.95},
        "toilet": {"bathroom": 0.95},
    },
    {"refrigerator": ("fridge",)},
    "test",
)


def two_rooms() -> OccupancyGrid:
    """Two 4 x 4 m rooms side by side, joined by a 1 m doorway, fully observed."""
    g = OccupancyGrid(size=16.0)
    g.logodds[:] = 0.0

    def box(x0, x1, y0, y1, value):
        i0, j0 = g.cell(np.array([x0, y0]))
        i1, j1 = g.cell(np.array([x1, y1]))
        g.logodds[i0:i1, j0:j1] = value

    box(-4.0, 4.0, -2.0, 2.0, -2.0)  # floor
    for x0, x1, y0, y1 in ((-4.1, 4.1, 2.0, 2.1), (-4.1, 4.1, -2.1, -2.0),
                           (-4.1, -4.0, -2.1, 2.1), (4.0, 4.1, -2.1, 2.1)):  # fmt: skip
        box(x0, x1, y0, y1, 2.0)
    box(-0.05, 0.05, -2.0, -0.5, 2.0)  # dividing wall with a doorway at y in (-0.5, 0.5)
    box(-0.05, 0.05, 0.5, 2.0, 2.0)
    return g


def track(tid, xy, labels, cams=None, misses=0, closest=1.5) -> Track:
    t = Track(tid, np.array([*xy, 0.9]), np.array([xy[0] - 0.3, xy[1] - 0.3, 0.0]),
              np.array([xy[0] + 0.3, xy[1] + 0.3, 1.8]))  # fmt: skip
    for label, n in labels.items():
        t.labels[label] = n
        t.label_views[label] = list(cams or [np.array([xy[0] - 1.5 + 0.6 * k, xy[1]])
                                             for k in range(n)])  # fmt: skip
        t.closest[label] = closest
        t.last_labelled[label] = 10.0
    t.views, t.misses = sum(labels.values()), misses
    return t


# ----- knowledge ---------------------------------------------------------------------------
def test_association_keeps_every_room_possible_and_uniform_without_knowledge():
    a = KNOW.association("refrigerator")
    assert a.argmax() == 0 and (a > 0).all() and np.isclose(a.sum(), 1)
    assert np.allclose(KNOW.association("piano"), 1 / len(CATS))
    assert np.allclose(uniform_knowledge(CATS).likelihood_ratio("refrigerator"), 1)
    assert PSEUDO > 0


def test_loader_reads_only_location_edges(tmp_path):
    data = {"sources": [{"name": "ConceptNet 5.7"}], "objects": {"refrigerator": {
        "places": {"kitchen": [{"relation": "/r/AtLocation", "weight": 2.0}],
                   "garage": [{"relation": "/r/LocatedNear", "weight": 1.0}],
                   "store": [{"relation": "/r/AtLocation", "weight": 9.0}],
                   "office": [{"relation": "/r/MadeOf", "weight": 5.0}]},
        "synonyms": [{"text": "fridge", "source": "wordnet", "grammar_safe": True},
                     {"text": "coolchest", "source": "wiktionary"}]}}}  # fmt: skip
    path = tmp_path / "k.json"
    path.write_text(json.dumps(data))
    k = load_knowledge(path, categories=("kitchen", "garage", "office"))
    assert k.strength["refrigerator"]["kitchen"] == pytest.approx(2.0)
    assert k.strength["refrigerator"]["garage"] == pytest.approx(0.5)
    assert "office" not in k.strength["refrigerator"] and k.synonyms["refrigerator"] == ("fridge",)


# ----- rooms -------------------------------------------------------------------------------
def test_segmentation_cuts_the_doorway():
    g = two_rooms()
    seg = RoomSegmenter()
    labels = seg.update(g)
    left = labels[g.cell(np.array([-2.0, 0.0]))]
    right = labels[g.cell(np.array([2.0, 0.0]))]
    assert left > 0 and right > 0 and left != right
    again = seg.update(g)  # stable ids on an unchanged map
    assert again[g.cell(np.array([-2.0, 0.0]))] == left


def test_room_weights_without_evidence_are_neutral_and_labels_count_once():
    w = category_weights({}, KNOW)
    assert len(set(np.round(list(w.values()), 9))) == 1 and UNKNOWN_ROOM in w
    one = category_weights({"sink": 1.0}, KNOW)
    two = category_weights({"sink": 1.0, "countertop": 1.0}, KNOW)
    assert two["kitchen"] > one["kitchen"] > w["kitchen"]
    # A sink alone leaves the bathroom plausible; it does not decide the room.
    assert one["bathroom"] > w["bathroom"] and one["kitchen"] < 0.5


def test_regions_take_objects_from_the_side_they_were_seen():
    g = two_rooms()
    seg = RoomSegmenter()
    seg.update(g)
    # A sink against the dividing wall, seen from the left room.
    sink = track(1, (-0.4, 1.0), {"sink": 3}, cams=[np.array([-2.0, 1.0])] * 3)
    counter = track(2, (-1.5, 1.6), {"countertop": 3}, cams=[np.array([-1.5, 0.0])] * 3)
    regions = build_regions(g, seg, [sink, counter], KNOW)
    left = seg.labels[g.cell(np.array([-2.0, 0.0]))]
    assert set(regions[left].tracks) == {1, 2}
    assert regions[left].leading()[0] == "kitchen"


# ----- belief ------------------------------------------------------------------------------
def test_belief_prefers_the_kitchen_and_coverage_removes_mass():
    g = two_rooms()
    sm = SemanticMap(g)
    sm.tracks = [
        track(1, (-1.0, 1.6), {"sink": 3}, cams=[np.array([-1.0, 0.0])] * 3),
        track(2, (-2.5, 1.6), {"countertop": 3}, cams=[np.array([-2.5, 0.0])] * 3),
    ]
    seg = RoomSegmenter()
    seg.update(g)
    regions = build_regions(g, seg, sm.tracks, KNOW)
    b = TargetBelief("refrigerator", KNOW)
    b.update(g, sm, seg, regions)
    left, right = (int(seg.labels[g.cell(np.array([x, 0.0]))]) for x in (-2.0, 2.0))
    assert b.region_factor[left] > 1.0 > b.region_factor[right] - 1e-9
    assert not b.candidate[g.free()].any()  # free floor is never a hypothesis
    kitchen_view = b.view(g, sm, np.array([-2.0, 0.0]))
    other_view = b.view(g, sm, np.array([2.0, 0.0]))
    assert kitchen_view.detection > other_view.detection
    total = b.total()
    sm.coverage[g.occupied()] = 0.85
    b.update(g, sm, seg, regions)
    assert b.total() == pytest.approx(0.15 * total, rel=1e-3)
    flat = TargetBelief("refrigerator", KNOW, BeliefConfig(semantic=False))
    flat.update(g, sm, seg, regions)
    assert np.allclose(flat.prior[flat.candidate], 1.0)


def test_memory_hint_raises_the_prior_where_it_was_seen():
    g = two_rooms()
    sm, seg = SemanticMap(g), RoomSegmenter()
    seg.update(g)
    hint = {"xy": [3.9, 1.5], "confidence": 1.0}
    b = TargetBelief("refrigerator", KNOW, memory_hints=[hint])
    b.update(g, sm, seg, {})
    near = b.prior[g.cell(np.array([4.02, 1.5]))]
    far = b.prior[g.cell(np.array([4.02, -1.5]))]
    assert near > 3 * far
    off = TargetBelief("refrigerator", KNOW, BeliefConfig(memory=False), [hint])
    off.update(g, sm, seg, {})
    assert off.prior[g.cell(np.array([4.02, 1.5]))] == pytest.approx(far)


# ----- verification ------------------------------------------------------------------------
TH = VerifierThresholds()


@pytest.mark.parametrize(
    "labels, misses, closest, cams, verified, failing",
    [
        ({"refrigerator": 3}, 0, 1.5, None, True, None),
        ({"refrigerator": 2}, 0, 1.5, None, False, "support"),
        ({"refrigerator": 3}, 0, 1.5, [np.array([0.0, 0.0])] * 3, False, "viewpoints"),
        ({"refrigerator": 3, "wardrobe": 3}, 0, 1.5, None, False, "purity"),
        ({"refrigerator": 3}, 0, 2.9, None, False, "close"),
        ({"refrigerator": 3}, 2, 1.5, None, False, "misses"),
    ],
)
def test_track_checks(labels, misses, closest, cams, verified, failing):
    t = track(1, (2.0, 0.0), labels, cams=cams, misses=misses, closest=closest)
    r = check_track(t, "refrigerator", TH)
    assert r["verified"] is verified
    if failing:
        assert not r["checks"][failing]


def test_reached_needs_standing_at_it_facing_it_with_a_fresh_detection():
    sm = SemanticMap(OccupancyGrid())
    sm.tracks = [track(1, (2.0, 0.0), {"refrigerator": 4})]
    far = verify_find(sm, "refrigerator", "reached", np.array([-1.0, 0.0, 0.0]), 11.0)
    assert far["status"] == "target_found_verified" and not far["satisfied"]
    at = verify_find(sm, "refrigerator", "reached", np.array([1.0, 0.0, 0.0]), 11.0)
    assert at["satisfied"]
    away = verify_find(sm, "refrigerator", "reached", np.array([1.0, 0.0, np.pi]), 11.0)
    assert not away["satisfied"]
    stale = verify_find(sm, "refrigerator", "reached", np.array([1.0, 0.0, 0.0]), 30.0)
    assert not stale["satisfied"]
    found = verify_find(sm, "refrigerator", "visually_localized", np.zeros(3), 99.0)
    assert found["satisfied"]
    none = verify_find(sm, "toilet", "visually_localized", np.zeros(3), 99.0)
    assert none["status"] == "target_not_yet_found"


# ----- semantic map from a frame -------------------------------------------------------------
def test_detection_becomes_a_track_at_the_measured_position():
    h, w = 60, 80
    rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)  # looks along +x
    cal = CameraCalibration(w, h, 58.0, np.array([0.0, 0.0, 1.0]), rotation)
    depth = np.full((h, w), np.nan, np.float32)
    mask = np.zeros((h, w), bool)
    mask[10:50, 30:50] = True
    depth[mask] = 2.0
    det = SemanticDetection("refrigerator", 0.8, mask, "test")
    frame = RGBDFrame(1.0, np.zeros((h, w, 3), np.uint8), depth, np.zeros((h, w), bool), cal,
                      (det,))  # fmt: skip
    sm = SemanticMap(OccupancyGrid())
    assert sm.update(frame) == [1]
    t = sm.tracks[0]
    assert t.position[0] == pytest.approx(2.0, abs=0.05) and abs(t.position[1]) < 0.1
    assert t.support("refrigerator") == 1 and t.closest["refrigerator"] < 2.2
    assert sm.coverage.max() == pytest.approx(float(detection_quality(2.0)), abs=0.05)
    blind = RGBDFrame(2.0, frame.rgb, depth, frame.self_mask, cal, None)
    assert sm.update(blind) == [] and sm.frames == 1  # no detector ran: no evidence


# ----- graph ---------------------------------------------------------------------------------
def test_graph_keeps_hypotheses_off_object_nodes():
    g = SemanticGraph("odometry:test")
    g.node("region:1", "region")
    g.node("class:refrigerator", "class")
    g.node("object:3", "object", labels={"sink": 2})
    g.edge("region:1", "contains", "object:3", "observed", "perception")
    g.edge("region:1", "may_contain", "class:refrigerator", "hypothesis", "belief", score=0.4,
           score_kind="share_of_search_mass")  # fmt: skip
    assert g.validate() == [] and g.hypotheses("refrigerator") == [("region:1", 0.4)]
    g.node("object:9", "object")
    g.edge("region:1", "may_contain", "object:9", "hypothesis", "belief")
    problems = g.validate()
    assert any("no observation" in p for p in problems)
    assert any("points at an object" in p for p in problems)
    with pytest.raises(ValueError):
        g.edge("region:1", "is", "class:refrigerator", "fact", "x")
    assert CELL == 0.05


def test_toolbox_runs_a_slow_detector_only_on_standing_looks():
    import threading

    from dream.perception.spatial import SpatialMemory
    from dream.tools.mobile import MobileToolbox

    class Base:
        time, stop_reason = 0.0, None

        def odometry(self):
            return np.zeros(3)

        def capture(self):
            h, w = 24, 32
            rot = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
            cal = CameraCalibration(w, h, 58.0, np.array([0.0, 0.0, 1.0]), rot)
            depth = np.full((h, w), 2.0, np.float32)
            return RGBDFrame(self.time, np.zeros((h, w, 3), np.uint8), depth,
                             np.zeros((h, w), bool), cal)  # fmt: skip

        def walk(self, *a):
            self.time += a[-1]
            return True

    seen = []

    def detector(frame):
        seen.append(frame.time)
        return frame

    tb = MobileToolbox(Base(), SpatialMemory(np.random.default_rng(0), boxes=False),
                       detector=detector, detect_when="stationary")  # fmt: skip
    tb.cancel = threading.Event()
    tb.call("go_to", {"x": 0.0, "y": 0.0})
    assert not seen
    tb.call("look_at", {"x": 1.0, "y": 0.0})
    assert len(seen) == 4 and not tb._standing_look
