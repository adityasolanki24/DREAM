"""Memory-guided search: remembered locations are checked first, never trusted blindly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dream.memory.home import HomeMemory
from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.memory_search import MemoryGuidedPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult

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


def test_absent_when_looked_at_and_not_seen_then_search_continues(mem):
    oid = mem.observe("cabinet_handle", (3.0, 0.0, 0.8), MAP, time=0)
    p = MemoryGuidedPlanner(Inner(), mem, MAP, now=100)
    h = [step("look_around")]
    p.next_action(GOAL, scene((0, 0, 0)), {}, h)
    h.append(step("go_to", x=2.0, y=0.0))
    a = p.next_action(GOAL, scene((2.0, 0.0, 0.0)), {}, h)
    assert isinstance(a, Finish) and ("absent", oid) in p.log
    assert mem.where_is("cabinet_handle", MAP, now=101)[0].state == "missing"


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
