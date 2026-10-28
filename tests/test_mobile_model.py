import json

from dream.runtime.agent import Finish, ToolCall
from dream.runtime.mobile_model import MobileModelPlanner, interpret_mobile_goal
from dream.tools.goals import GoToPlace, Inspect


class Model:
    def __init__(self, name, args):
        self.output = "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"

    def generate(self, messages, tools):
        return self.output


def decide(name, args, scene=None, verdict=None):
    return MobileModelPlanner(Model(name, args), "inspect cabinet", lambda: None).next_action(
        object(), scene or {}, verdict or {}, []
    )


def test_mobile_model_only_uses_observed_targets():
    scene = {
        "frontiers": [{"xy": [1.2, 0.3]}],
        "candidates": [{"id": 4}],
        "viewpoints": [{"candidate": 4, "side": 2}],
    }
    assert decide("explore_frontier", {"index": 0}, scene) == ToolCall(
        "go_to", {"x": 1.2, "y": 0.3}
    )
    assert isinstance(decide("explore_frontier", {"index": 1}, scene), Finish)
    assert isinstance(decide("approach", {"candidate": 4, "side": 3}, scene), Finish)
    assert isinstance(decide("inspect", {"candidate": 8}, scene), Finish)
    assert isinstance(decide("go_to", {"x": 1, "y": 2}, scene), Finish)


def test_model_cannot_claim_success_without_bound_goal_verification():
    action = decide("finish", {"success": True, "reason": "done"})
    assert isinstance(action, Finish) and not action.success
    assert decide(
        "finish", {"success": True, "reason": "done"}, verdict={"satisfied": True}
    ).success


def test_language_model_maps_request_to_only_supported_typed_goals():
    assert interpret_mobile_goal(
        Model("inspect_cabinet", {}),
        "go look for the cupboard",
        {},
        {"candidates": []},
        lambda: None,
    ) == Inspect("cabinet")
    assert interpret_mobile_goal(
        Model("go_to_place", {"place": "entrance"}),
        "take me to the door",
        {"entrance": [0, 0]},
        {"candidates": []},
        lambda: None,
    ) == GoToPlace("entrance")
    try:
        interpret_mobile_goal(
            Model("go_to_place", {"place": "bedroom"}),
            "go to my bedroom",
            {"entrance": [0, 0]},
            {},
            lambda: None,
        )
    except ValueError as error:
        assert "not in home memory" in str(error)
    else:
        raise AssertionError("unobserved place must be refused")


def test_home_memory_is_read_only_and_reports_sightings(tmp_path):
    from dream.memory.home import HomeMemory
    from experiments.phase1_interface import read_home_memory

    path = tmp_path / "memory.sqlite"
    store = HomeMemory(path, clock=lambda: 100.0)
    store.observe("bottle", [1, 2, 3], "room", source="camera")
    store.close()
    before = path.read_bytes()
    data = read_home_memory(path)
    assert data["objects"][0]["label"] == "bottle"
    assert data["objects"][0]["position"] == [1.0, 2.0, 3.0]
    assert "current state not assessed" in data["source"]
    assert path.read_bytes() == before
