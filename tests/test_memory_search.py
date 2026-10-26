"""Memory-guided search: remembered locations are checked first, never trusted blindly."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dream.memory.home import HomeMemory
from dream.perception.geometry import visibility
from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.memory_search import MemoryGuidedPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult
from dream.types import CameraCalibration, RGBDFrame

MAP = "m"
GOAL = Inspect("cabinet")


class Inner:
    def next_action(self, goal, scene, verdict, history):
        if not any(s.call.name == "look_around" for s in history):
            return ToolCall("look_around")
        return Finish(False, "inner search")


def step(name, ok=True, **args):
    return Step(ToolCall(name, args), ToolResult(name, "succeeded" if ok else "failed"), 0.0)


def scene(pose, handles=()):
    return {"pose": list(pose), "candidates": [{"handle": list(h)} for h in handles]}


@pytest.fixture
def mem(tmp_path):
    m = HomeMemory(tmp_path / "h.sqlite")
    yield m
    m.close()


def test_maps_first_then_goes_to_the_remembered_handle_facing_it(mem):
    mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    assert p.next_action(GOAL, scene((0, 0, 0)), {}, []).name == "look_around"
    a = p.next_action(GOAL, scene((0, 0, 0)), {}, [step("look_around")])
    assert a.name == "go_to"
    assert a.args["x"] == pytest.approx(2.0) and a.args["heading"] == pytest.approx(0.0)


def camera_frame(pose, depth=5.0, self_mask=False):
    """Head camera 1.2 m up at ``pose`` looking along its heading; uniform depth (m)."""
    x, y, h = pose
    fwd, right = np.array([np.cos(h), np.sin(h), 0.0]), np.array([np.sin(h), -np.cos(h), 0.0])
    rot = np.stack([right, [0.0, 0.0, 1.0], -fwd], axis=1)  # camera axes in the world
    cal = CameraCalibration(320, 240, 58.0, np.array([x, y, 0.8]), rot)
    d = np.full((240, 320), depth, np.float32)
    return RGBDFrame(0.0, np.zeros((240, 320, 3), np.uint8), d,
                     np.full((240, 320), self_mask), cal)  # fmt: skip


def drive(p, depths):
    """Run the planner as the agent loop does: each go_to is executed (the robot ends at the
    requested viewpoint), the head camera sees a uniform ``depth``, and the planner decides
    the next action. Returns the viewpoints visited and the final action."""
    h = [step("look_around")]
    a = p.next_action(GOAL, scene((0, 0, 0)), {}, h)
    stands = []
    for depth in depths:
        assert a.name == "go_to"
        stand = (a.args["x"], a.args["y"], a.args["heading"])
        stands.append(stand)
        h.append(step("go_to", x=stand[0], y=stand[1]))
        p.last_frame = lambda stand=stand, depth=depth: camera_frame(stand, depth)
        a = p.next_action(GOAL, scene(stand), {}, h)
    return stands, a


def test_absence_needs_two_visible_misses_from_different_viewpoints(mem):
    oid = mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    (first,), a = drive(p, [5.0])  # a wall 5 m away: the spot itself is in clear view
    assert ("look_visible", oid) in p.log and a.name == "go_to"  # one look is not enough
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state != "missing"
    p2 = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    (v1, v2), a = drive(p2, [5.0, 5.0])
    assert np.hypot(v2[0] - v1[0], v2[1] - v1[1]) > 0.3  # a different viewpoint
    assert ("absent", oid) in p2.log and isinstance(a, Finish)
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state == "missing"


def test_occluded_looks_are_never_evidence_of_absence(mem):
    oid = mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    _, a = drive(p, [0.4, 0.4, 0.4])  # something 0.4 m in front of the camera every time
    assert ("unverified", oid) in p.log and ("absent", oid) not in p.log
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state != "missing"
    assert isinstance(a, Finish)


def test_without_frames_no_absence_is_recorded(mem):
    mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    h = [step("look_around")]
    for _ in range(3):
        p.next_action(GOAL, scene((0, 0, 0)), {}, h)
        h.append(step("go_to", x=2.0, y=0.0))
        p.next_action(GOAL, scene((2.0, 0.0, 0.0)), {}, h)
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state != "missing"


def test_visibility_reasons():
    point = np.array([2.0, 0.0, 0.8])
    assert visibility(camera_frame((0, 0, 0), 5.0), point) == "visible"
    assert visibility(camera_frame((0, 0, 0), 1.0), point) == "occluded"
    assert visibility(camera_frame((0, 0, np.pi), 5.0), point) == "behind"
    assert visibility(camera_frame((0, 0, 1.2), 5.0), point) == "out_of_image"
    assert visibility(camera_frame((0, 0, 0), 5.0, self_mask=True), point) == "self_occluded"
    assert visibility(camera_frame((-1, 0, 0), 5.0), point) == "too_far"
    nan = camera_frame((0, 0, 0), np.nan)
    assert visibility(nan, point) == "no_return"


def test_present_hands_over_to_the_search_without_recording_absence(mem):
    oid = mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    h = [step("look_around")]
    p.next_action(GOAL, scene((0, 0, 0)), {}, h)
    h.append(step("go_to", x=2.0, y=0.0))
    p.next_action(GOAL, scene((2.0, 0.0, 0.0), [(3.1, 0.1)]), {}, h)
    assert ("present", oid) in p.log
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state != "missing"


def test_unreached_viewpoint_or_out_of_view_is_not_evidence_of_absence(mem):
    mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    h = [step("look_around")]
    p.next_action(GOAL, scene((0, 0, 0)), {}, h)
    h.append(step("go_to", ok=False, x=2.0, y=0.0))
    p.next_action(GOAL, scene((0.5, 0.0, 0.0)), {}, h)
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state != "missing"


def test_without_memory_it_is_the_inner_planner(mem):
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=0)
    h = [step("look_around")]
    assert p.next_action(GOAL, scene((0, 0, 0)), {}, h) == Finish(False, "inner search")


def test_remember_inspections_stores_only_confirmed_handles(mem):
    from dream.runtime.memory_search import remember_inspections

    c1 = SimpleNamespace(id=1, handle=(1.0, 2.0, 0.8))
    c2 = SimpleNamespace(id=2, handle=(4.0, 2.0, 0.8))
    tb = SimpleNamespace(memory=SimpleNamespace(candidates=[c1, c2]),
                         inspections={1: {"handle_confirmed_close": True},
                                      2: {"handle_confirmed_close": False}})  # fmt: skip
    assert len(remember_inspections(tb, mem, MAP, GOAL, now=5)) == 1
    (b,) = mem.where_is("cabinet_handle", MAP, now=6)
    assert tuple(b.position) == (1.0, 2.0, 0.8)
