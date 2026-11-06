"""Evaluation-side instrumentation of experiments/g1_explore_room.py: contact classes, frontier
decision log, refusal categories and phantom handles. None of it may change what the agent
does; the recorder only calls stateless planners on the same inputs."""

from types import SimpleNamespace

import mujoco
import numpy as np

from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.search import RoomSearchPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult
from experiments.g1_explore_room import (
    Budgeted,
    Monitor,
    contact_class,
    phantom_handles,
    refusal_category,
)


def test_contact_classes():
    assert (
        contact_class("left_ankle_roll_link", "furniture_0", "furniture_0_leg1") == "foot_furniture"
    )
    assert contact_class("right_ankle_roll_link", "cabinet", "cabinet_crate") == "foot_furniture"
    assert contact_class("right_hand_index_1_link", "cabinet", "cabinet_handle") == "hand_handle"
    assert (
        contact_class("left_wrist_yaw_link", "cabinet_handle_post0", "cabinet_handle_post0")
        == "hand_handle"
    )
    assert (
        contact_class("left_hand_thumb_2_link", "cabinet_door", "cabinet_door") == "hand_furniture"
    )
    assert contact_class("left_hand_thumb_2_link", "furniture_2", "furniture_2") == "hand_furniture"
    assert contact_class("left_hand_thumb_2_link", "wall_px", "wall_px") == "other"
    assert contact_class("torso_link", "cabinet", "cabinet_crate") == "other"
    assert contact_class("left_ankle_roll_link", "wall_nx", "wall_nx") == "other"


def test_refusal_categories():
    ex = "explored the reachable space; 3 candidate object(s) checked, none with a handle"
    ph = "candidate 2 has a handle, but no reachable viewpoint"
    assert refusal_category(False, ex) == "exhausted_exploration"
    assert refusal_category(False, ph) == "phantom_or_unreachable_candidate"
    assert refusal_category(False, "time budget (240 s) spent") == "other"
    assert refusal_category(True, "found and inspected candidate 1: a box with a handle") is None


XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <geom name="wall_px" type="box" pos="0 -1 .5" size=".5 .05 .5"/>
    <body name="cabinet" pos="1 0 .5">
      <geom name="cabinet_crate" type="box" size=".3 .3 .3"/>
      <geom name="cabinet_handle" type="box" pos="-.3 0 0" size=".05 .05 .05"/>
    </body>
    <body name="pelvis" pos="0 0 .5">
      <freejoint/>
      <geom name="pelvis_geom" type="sphere" size=".05" pos="0 -10 0" contype="0" conaffinity="0"/>
      <body name="right_hand_index_1_link" pos="0 0 0">
        <geom type="sphere" size=".06" pos="0.67 0 0"/>
      </body>
      <body name="left_ankle_roll_link" pos="0 0 0">
        <geom type="sphere" size=".06" pos="1.18 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_monitor_aggregates_contacts_by_body_pair_with_force_and_time():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon >= 2  # hand-vs-handle and ankle-vs-crate both overlap
    env = SimpleNamespace(model=m, data=d, time=0.0, true_pose=lambda: np.array([0.0, 0.0, 0.0]))
    mon = Monitor(env)
    for i in range(5):
        env.time = 10.0 + i
        mon.tick()
    rows = {(r["robot_body"], r["other"]): r for r in mon.summary(end_t=14.0)}
    hand = rows[("right_hand_index_1_link", "cabinet_handle")]
    assert hand["class"] == "hand_handle" and hand["ticks"] == 5
    assert hand["first_t"] == 10.0 and hand["last_t"] == 14.0
    assert hand["ticks_in_last_3s"] == 4 and hand["in_last_3s"]
    assert mon.contact_steps == 5  # legacy counter: once per tick regardless of pair count
    foot = [r for r in rows.values() if r["class"] == "foot_furniture"]
    assert foot and foot[0]["ticks"] == 5
    assert all(r["peak_force_n"] >= 0 for r in rows.values())
    late = Monitor(env).summary(end_t=100.0)
    assert late == []  # nothing recorded before any tick


def _step(name, ok=True, **args):
    return Step(ToolCall(name, args), ToolResult(name, "succeeded" if ok else "failed"), 0.0)


def test_frontier_log_records_both_policies_and_divergence_without_changing_the_action():
    scene = {
        "candidates": [],
        "viewpoints": [],
        "inspected": {},
        "frontiers": [
            {"xy": [5, 5], "gain_m2": 9.0, "cost_m": 4.0},
            {"xy": [1, 1], "gain_m2": 1.0, "cost_m": 1.0},
        ],
    }
    hist = [_step("look_around")]
    base = SimpleNamespace(time=12.0)
    for method, expected in (("gain", [5, 5]), ("nearest", [1, 1])):
        planned = RoomSearchPlanner(method).next_action(Inspect("cabinet"), scene, {}, hist)
        b = Budgeted(RoomSearchPlanner(method), base, 240.0, method)
        a = b.next_action(Inspect("cabinet"), scene, {}, hist)
        assert a == planned == ToolCall("go_to", {"x": expected[0], "y": expected[1]})
        (dec,) = b.decisions
        assert dec["chosen_xy"] == expected and dec["method"] == method
        assert dec["gain_policy_xy"] == [5, 5] and dec["nearest_policy_xy"] == [1, 1]
        assert dec["diverges"] is True
        assert (dec["cost_m"], dec["gain_m2"]) == ((4.0, 9.0) if method == "gain" else (1.0, 1.0))
        assert b.frontier_steps == [[12.0, [[5, 5, 9.0, 4.0], [1, 1, 1.0, 1.0]]]]


def test_frontier_log_agreement_and_budget_stop():
    scene = {
        "candidates": [],
        "viewpoints": [],
        "inspected": {},
        "frontiers": [{"xy": [2, 2], "gain_m2": 5.0, "cost_m": 1.0}],
    }
    b = Budgeted(RoomSearchPlanner("gain"), SimpleNamespace(time=1.0), 240.0)
    b.next_action(Inspect("cabinet"), scene, {}, [_step("look_around")])
    assert b.decisions[0]["diverges"] is False and b.method == "gain"
    late = Budgeted(RoomSearchPlanner("gain"), SimpleNamespace(time=300.0), 240.0)
    a = late.next_action(Inspect("cabinet"), scene, {}, [_step("look_around")])
    assert isinstance(a, Finish) and late.decisions == []


def _env_tb(handle_xyz, cabinet=True):
    site = SimpleNamespace(xpos=np.array([2.0, 0.0, 0.75]))
    env = SimpleNamespace(
        true_pose=lambda: np.array([0.0, 0.0, 0.0]),
        odometry=lambda: np.array([0.0, 0.0, 0.0]),
        data=SimpleNamespace(site=lambda name: site),
    )
    room = SimpleNamespace(cabinet=SimpleNamespace(mechanism="hinge" if cabinet else "none"))
    cands = [SimpleNamespace(handle=h) for h in handle_xyz]
    return env, room, SimpleNamespace(memory=SimpleNamespace(candidates=cands))


def test_phantom_handles_use_the_kitchen_definition():
    env, room, tb = _env_tb([np.array([2.1, 0.0, 0.75]), np.array([2.0, 0.6, 0.75]), None])
    assert phantom_handles(env, room, tb) == 1  # first within 0.3 m, second 0.6 m off
    env, room, tb = _env_tb([np.array([2.0, 0.0, 0.75])], cabinet=False)
    assert phantom_handles(env, room, tb) == 1  # nothing to be near in a cabinet-less room
    env, room, tb = _env_tb([None])
    assert phantom_handles(env, room, tb) == 0
