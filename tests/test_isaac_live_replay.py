"""Replay of the saved 1 Nov 2026 genuine-Qwen session 150528 (development evidence).

R3-a: "walk to the refrigerator" was bound to frontier:0, 0.09 m from the robot, verified at
sim time 0 and reported as success. R3-b: the viewpoint (candidate 1, side 5) was chosen
1.37 m from the handle against a 1.40 m limit. CPU only; reads results/isaac_qwen_live_20261101.
"""

import io
import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import dream.tools.mobile as mobile
from dream.perception.spatial import Candidate, SpatialMemory
from dream.runtime.agent import TaskAgent
from dream.runtime.mobile_model import MobileModelPlanner, interpret_mobile_goal
from dream.tools.goals import GoToObserved, UnsupportedGoal
from dream.tools.mobile import MobileToolbox
from dream.tools.toolbox import ToolResult

SESSION = (
    Path(__file__).resolve().parents[1]
    / "results/isaac_qwen_live_20261101/dream-qwen-isaac-20261102-001940-150528"
)


def saved_scene(index: int) -> dict:
    record = json.loads((SESSION / f"qwen_decisions/{index:04d}_decision.json").read_text())
    message = json.loads(next(m["text"] for m in record["messages"] if m["role"] == "user"))
    return message.get("observed_scene") or message["observation"]


class Replay:
    """Returns the saved model output; the model itself is not run."""

    def __init__(self, index):
        record = json.loads((SESSION / f"qwen_decisions/{index:04d}_decision.json").read_text())
        self.output, self.calls = record["output"], 0

    def generate(self, messages, tools):
        self.calls += 1
        return self.output


def test_r3a_saved_refrigerator_request_is_no_longer_bound_to_a_reached_frontier():
    scene = saved_scene(0)
    assert scene["frontiers"][0]["cost_m"] == 0.07  # the frontier under the robot
    goal = interpret_mobile_goal(Replay(0), "walk to the refrigerator", {}, scene, lambda: None)
    assert isinstance(goal, UnsupportedGoal) and "already reached" in goal.reason


def test_r3a_saved_verdict_and_saved_finish_true_cannot_become_success():
    scene = saved_scene(4)  # the decision at which Qwen answered finish(success=true)
    saved_finish = Replay(4)
    assert '"success": true' in saved_finish.output
    verdict = {"satisfied": True, "reason": "0.09 m away", "reference": "frontier:0"}

    class Tools:
        cancel = threading.Event()

        def call(self, name, args=None):
            data = {
                "observe_scene": scene,
                "verify_goal": verdict,
                "get_robot_state": {"protective_stop": False},
            }[name]
            return ToolResult(name, "succeeded", "", data)

    planner = MobileModelPlanner(saved_finish, "walk to the refrigerator", lambda: None)
    episode = TaskAgent(Tools(), planner).run(GoToObserved(-0.025, -0.075, "frontier:0"))
    assert not episode.success_claimed
    assert episode.completion == "exploration_point_reached_target_unverified"
    assert "refrigerator" not in episode.reason
    assert saved_finish.calls == 0  # the typed outcome needs no model decision


def test_r3b_saved_map_no_longer_offers_the_viewpoint_at_the_old_boundary(monkeypatch):
    scene = saved_scene(9)
    with zipfile.ZipFile(SESSION / "raw.zip") as z:
        occupancy = np.load(io.BytesIO(z.read("out/jobs/1/live/sensor_0009.npz")))["occupancy"]
    memory = SpatialMemory(np.random.default_rng(0))
    memory.grid.logodds[:] = occupancy
    for c in scene["candidates"]:
        memory.candidates.append(
            Candidate(
                c["id"],
                c["color"],
                np.array(c["position"]),
                0.0,
                np.array(c["extent"]),
                views=c["views"],
                handle=None if c["handle"] is None else np.array(c["handle"]),
                handle_views=c["handle_views"],
                sides_close=set(c["sides_seen_close"]),
            )
        )
    pose = np.array(scene["pose"], float)
    base = SimpleNamespace(odometry=lambda: pose.copy(), stop_reason=None, time=0.0)
    toolbox = MobileToolbox(base, memory, navigator="rpp")
    handle = memory.candidates[0].handle[:2]

    def offered():
        passable, _, dist = toolbox._passable()
        return {
            (v.candidate, v.side): v for v in toolbox.viewpoints(passable, dist) if v.candidate == 1
        }

    monkeypatch.setattr(mobile, "ARRIVAL_MARGIN", 0.0)  # the behaviour of the saved session
    old = offered()[(1, 5)]
    assert np.allclose(old.xy, [-0.275, -0.625]) and 1.36 < np.linalg.norm(old.xy - handle) < 1.4
    monkeypatch.undo()
    assert (1, 5) not in offered()
    for v in offered().values():
        assert np.linalg.norm(v.xy - handle) <= mobile.INSPECT_RANGE[1] - mobile.ARRIVAL_MARGIN
