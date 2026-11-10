"""The planner's decision log is read-only: it never changes the chosen action (synthetic map)."""

from types import SimpleNamespace

import numpy as np

from dream.navigation.planning import distances, traversable
from dream.runtime.semantic_search import SemanticSearchPlanner, UtilityWeights
from dream.semantics.world import SemanticWorld
from tests.test_semantics import KNOW, track, two_rooms


def half_explored():
    """Two rooms; the right one is unobserved except for the doorway."""
    g = two_rooms()
    i0, j0 = g.cell(np.array([0.3, -2.2]))
    i1, j1 = g.cell(np.array([4.2, 2.2]))
    g.logodds[i0:i1, j0:j1] = 0.0  # unknown again
    world = SemanticWorld(g, KNOW)
    world.map.tracks = [
        track(1, (-1.0, 1.6), {"sink": 3}, cams=[np.array([-1.0, 0.0])] * 3),
        track(2, (-2.5, 1.6), {"countertop": 3}, cams=[np.array([-2.5, 0.0])] * 3),
    ]
    world.set_target("refrigerator")
    world.refresh()
    return g, world


def decide(diagnose: bool):
    g, world = half_explored()
    pose = np.array([-2.0, 0.0, 0.0])
    base = SimpleNamespace(time=12.0, odometry=lambda: pose)
    planner = SemanticSearchPlanner(
        SimpleNamespace(semantic=world, base=base),
        "utility",
        UtilityWeights(kappa=3.0),
        diagnose=diagnose,
    )
    ok = traversable(g)
    ok[g.cell(pose[:2])] = True
    dist = distances(ok, tuple(int(v) for v in g.cell(pose[:2])))
    action = planner._explore(world, "refrigerator", ok, dist, pose, [])
    return planner, action


def test_logging_does_not_change_the_action():
    _, plain = decide(False)
    logged_planner, logged = decide(True)
    assert (plain.name, plain.args) == (logged.name, logged.args)
    assert logged_planner.trace[-1]["purpose"] in ("explore", "explore_look")


def test_the_log_holds_terms_and_both_counterfactuals():
    off, _ = decide(False)
    assert off.diagnostics == []
    on, _ = decide(True)
    (rec,) = on.diagnostics
    for key in (
        "chosen",
        "no_detection",
        "uniform",
        "top",
        "regions",
        "n_views",
        "changed_uniform",
        "changed_no_detection",
        "gap_to_second",
    ):
        assert key in rec
    c = rec["chosen"]
    assert abs(c["sem_term"] - 3.0 * c["detection"]) < 0.01
    # The log is consistent with itself: "changed" means the picks differ.
    assert rec["changed_uniform"] == (rec["uniform"]["i"] != c["i"])
    assert rec["changed_no_detection"] == (rec["no_detection"]["i"] != c["i"])
    assert rec["top"][0]["i"] == c["i"]
