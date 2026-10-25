"""Contracts for mobile exploration: occupancy mapping, candidate/handle perception, the mobile
toolbox and the room-search planner. Ground truth appears only here, as the evaluator or to
place the robot (scene setup).
"""

import numpy as np
import pytest

from dream.navigation.planning import distances, path_to, traversable
from dream.perception.occupancy import CELL, OccupancyGrid
from dream.robot import g1_wbc
from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.search import RoomSearchPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult

HAVE_WEIGHTS = (g1_wbc.policy_cache() / "GR00T-WholeBodyControl-Walk.onnx").exists()
needs_policy = pytest.mark.skipif(not HAVE_WEIGHTS, reason="WBC weights not fetched")


def test_planner_never_walks_through_obstacles_or_unknown_space():
    g = OccupancyGrid(size=4.0)
    g.logodds[:] = -2.0  # all free
    g.logodds[0:70, 38:42] = 2.0  # a wall along x at y = 0 (i is x, j is y), gap at x > 1.5
    ok = traversable(g)
    start, goal = g.cell([0.0, -1.0]), g.cell([0.0, 1.0])
    dist = distances(ok, (int(start[0]), int(start[1])))
    path = path_to(dist, (int(goal[0]), int(goal[1])))
    assert path and all(ok[c] for c in path)
    assert max(i for i, _ in path) >= 70  # it goes around the end of the wall
    assert len(path) * CELL > 2.0  # longer than the blocked straight line
    g.logodds[72:, :] = 0.0  # make the gap unknown: no path any more
    dist = distances(traversable(g), (int(start[0]), int(start[1])))
    assert not np.isfinite(dist[int(goal[0]), int(goal[1])])


def _scene(**kw):
    base = {
        "candidates": [],
        "viewpoints": [],
        "frontiers": [],
        "inspected": {},
    }
    return base | kw


def _step(name, ok=True, **args):
    return Step(ToolCall(name, args), ToolResult(name, "succeeded" if ok else "failed"), 0.0)


def test_planner_checks_candidates_before_exploring_and_explores_by_gain():
    p = RoomSearchPlanner("gain")
    hist = [_step("look_around")]
    vp = {"candidate": 3, "side": 0, "xy": [1, 0], "heading": 0, "cost_m": 2.0}
    fr = [
        {"xy": [5, 5], "gain_m2": 9.0, "cost_m": 4.0},
        {"xy": [1, 1], "gain_m2": 1.0, "cost_m": 1.0},
    ]
    a = p.next_action(Inspect("cabinet"), _scene(viewpoints=[vp], frontiers=fr), {}, hist)
    assert a == ToolCall("approach", {"candidate": 3, "side": 0})
    a = p.next_action(Inspect("cabinet"), _scene(frontiers=fr), {}, hist)
    assert a == ToolCall("go_to", {"x": 5, "y": 5})
    a = RoomSearchPlanner("nearest").next_action(Inspect("cabinet"), _scene(frontiers=fr), {}, hist)
    assert a == ToolCall("go_to", {"x": 1, "y": 1})
    a = p.next_action(Inspect("cabinet"), _scene(), {}, hist)
    assert isinstance(a, Finish) and not a.success


def test_planner_drops_targets_that_keep_failing():
    p = RoomSearchPlanner()
    vp = {"candidate": 3, "side": 0, "xy": [1, 0], "heading": 0, "cost_m": 2.0}
    hist = [_step("look_around")] + [_step("approach", False, candidate=3, side=0)] * 2
    fr = [{"xy": [5, 5], "gain_m2": 9.0, "cost_m": 4.0}]
    a = p.next_action(Inspect("cabinet"), _scene(viewpoints=[vp], frontiers=fr), {}, hist)
    assert a.name == "go_to"


def _room_env(seed, pose):
    from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1
    from dream.sim.room import add_room, sample_room

    room = sample_room(np.random.default_rng(seed))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng(seed)
    )
    env.reset(*pose)
    ctrl = g1_wbc.WBCLocomotion.from_cache()
    for _ in range(40):
        env.step(ctrl)
    return room, env


@pytest.mark.slow
@needs_policy
def test_wall_fragments_cut_by_the_image_border_are_not_handles():
    """Development room seed 1: from here, wall pieces cut by the top border sat beside the
    table and were taken as its handle, and the agent claimed the table was the cabinet."""
    from dream.perception.spatial import SpatialMemory

    _, env = _room_env(1, (0.52, 1.44, 3.08))
    mem = SpatialMemory(np.random.default_rng(0))
    for _ in range(3):
        mem.update(env.capture_head())
    assert mem.candidates and all(c.handle is None for c in mem.candidates)


@pytest.mark.slow
@needs_policy
def test_the_cabinet_handle_is_found_from_one_metre_in_front():
    from dream.perception.spatial import SpatialMemory
    from dream.sim.room import cabinet_front

    room = __import__("dream.sim.room", fromlist=["sample_room"]).sample_room(
        np.random.default_rng(1)
    )
    front, n = cabinet_front(room.cabinet)
    stand = front[:2] + n * 1.0
    yaw = float(np.arctan2(-n[1], -n[0]))
    _, env = _room_env(1, (stand[0], stand[1], yaw))
    mem = SpatialMemory(np.random.default_rng(0))
    for _ in range(3):
        mem.update(env.capture_head())
    with_handle = [c for c in mem.candidates if c.handle is not None]
    assert len(with_handle) == 1 and with_handle[0].color == room.cabinet.color
    # odometry starts at the reset pose: express the true handle in that frame to compare
    true = env.data.site("cabinet_handle_site").xpos
    c, s = np.cos(yaw), np.sin(yaw)
    rel = true[:2] - stand
    local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
    assert np.linalg.norm(with_handle[0].handle[:2] - local) < 0.06


def test_viewpoints_follow_perception_not_where_the_robot_stood():
    """Regression (held-out rooms 1105, 1119, 2105): an rpp filter dropped viewpoints near
    poses already stood at, hiding faces perception had not seen up close; the search then
    ended without finding the handle. Viewpoints must depend on the candidates' observed
    sides only, for both navigators."""
    import inspect

    from dream.tools.mobile import MobileToolbox

    src = inspect.getsource(MobileToolbox.viewpoints)
    assert "visited" not in src and "sides_close" in src
