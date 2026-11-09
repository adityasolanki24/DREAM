"""Room evidence for unexplored space (BeliefConfig.unexplored_rooms), on synthetic maps.

The left room is fully mapped (it has a core); the right room is mostly unknown, with a strip
of floor seen through the doorway. Objects seen across the doorway have fronts that are far
from every core: they are unplaced, and only they inform the factor of nearby candidate cells."""

import numpy as np

from dream.perception.semantic import SemanticMap
from dream.semantics.belief import BeliefConfig, TargetBelief
from dream.semantics.rooms import RoomSegmenter, build_regions
from tests.test_semantics import KNOW, track, two_rooms

FAR_CAM = [np.array([-1.0, 0.0])]


def through_the_door():
    g = two_rooms()
    i0, j0 = g.cell(np.array([0.3, -2.2]))
    i1, j1 = g.cell(np.array([4.2, 2.2]))
    g.logodds[i0:i1, j0:j1] = 0.0  # the right room: unknown ...
    a0, b0 = g.cell(np.array([0.06, -0.5]))
    a1, b1 = g.cell(np.array([1.6, 0.5]))
    g.logodds[a0:a1, b0:b1] = -2.0  # ... except a strip of floor seen through the doorway
    return g


def belief(tracks, config, g=None):
    g = g or through_the_door()
    sm, seg = SemanticMap(g), RoomSegmenter()
    sm.tracks = tracks
    seg.update(g)
    regions = build_regions(g, seg, sm.tracks, KNOW)
    b = TargetBelief("refrigerator", KNOW, config)
    b.update(g, sm, seg, regions)
    return g, b


def own(tid=9, label="toilet"):
    """An object inside the mapped left room: evidence for that room (a bathroom), not for the
    unexplored space across the doorway."""
    return track(tid, (-3.2, 1.0), {label: 3}, cams=[np.array([-2.0, 1.0])] * 3)


def across(tid, label, views=3, xy=(1.9, 0.9)):
    return track(tid, xy, {label: views}, cams=FAR_CAM * views)


OFF, ON = BeliefConfig(), BeliefConfig(unexplored_rooms=True)
AT = np.array([2.3, 0.9])  # unexplored space next to the objects seen across the doorway


def prior_at(b, g, xy=AT):
    return float(b.prior[g.cell(xy)])


def test_it_is_off_by_default():
    assert BeliefConfig().unexplored_rooms is False
    g, b = belief([across(1, "sink"), across(2, "countertop")], OFF)
    assert b.unplaced == {}


def test_kitchen_objects_across_the_doorway_raise_the_prior_of_the_space_around_them():
    # Without the option the left room's region also absorbs the kitchen objects seen across
    # its doorway, so the unexplored cells next to them get a bathroom-and-kitchen mixture.
    tracks = [own(), across(1, "sink"), across(2, "countertop", xy=(1.9, 1.5))]
    g, off = belief(tracks, OFF)
    _, on = belief(tracks, ON)
    assert on.unplaced["far_tracks"] == 2 and on.unplaced["cells"] > 0
    assert prior_at(on, g) > 1.1 * prior_at(off, g) and prior_at(on, g) > 1.0


def test_bedroom_objects_lower_it():
    g, on = belief([across(1, "bed")], ON)
    assert on.unplaced["far_tracks"] == 1
    assert prior_at(on, g) < 0.95


def test_a_weak_cue_moves_the_prior_less_than_a_strong_one():
    g, one = belief([across(1, "sink", views=1)], ON)
    _, many = belief([across(1, "sink", views=4), across(2, "countertop", views=4)], ON)
    shift = lambda b: abs(np.log(prior_at(b, g)))  # noqa: E731
    assert 0 < shift(one) < shift(many)


def test_objects_in_a_mapped_room_do_not_act_through_this_option():
    inside = [track(1, (-1.0, 1.6), {"sink": 3}, cams=[np.array([-1.0, 0.0])] * 3)]
    g, off = belief(inside, OFF)
    _, on = belief(inside, ON)
    assert on.unplaced["far_tracks"] == 0
    assert np.array_equal(off.prior, on.prior)


def test_it_rescales_existing_hypotheses_and_adds_none():
    tracks = [own(), across(1, "sink"), across(2, "countertop", xy=(1.9, 1.5))]
    g, off = belief(tracks, OFF)
    _, on = belief(tracks, ON)
    assert np.array_equal(off.candidate, on.candidate)  # same hypothesis support
    assert (on.prior[~on.candidate] == 0).all()
    # nothing observed was written: tracks and their labels are exactly what went in
    sm = SemanticMap(g)
    sm.tracks = tracks
    assert [t.labels for t in sm.tracks] == [{"toilet": 3}, {"sink": 3}, {"countertop": 3}]


def test_unknown_labels_and_no_tracks_leave_the_prior_alone():
    g, off = belief([across(1, "piano")], OFF)
    _, on = belief([across(1, "piano")], ON)
    assert np.allclose(off.prior, on.prior)
    _, none = belief([], ON)
    assert none.unplaced["far_tracks"] == 0


def test_no_knowledge_means_no_change():
    from dream.semantics.knowledge import uniform_knowledge

    g = through_the_door()
    sm, seg = SemanticMap(g), RoomSegmenter()
    sm.tracks = [across(1, "sink"), across(2, "countertop", xy=(1.9, 1.5))]
    seg.update(g)
    uniform = uniform_knowledge(KNOW.categories)
    regions = build_regions(g, seg, sm.tracks, uniform)
    priors = []
    for config in (OFF, ON):
        b = TargetBelief("refrigerator", uniform, config)
        b.update(g, sm, seg, regions)
        priors.append(b.prior)
    assert np.allclose(*priors)
