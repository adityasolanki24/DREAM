"""Advertised goal arguments must match the physical verifier's capabilities."""

from dream.runtime.vlm import tool_schemas
from dream.tools.object_exploration import ExplorationToolbox


def test_exploration_goal_schema_offers_only_supported_region_relation():
    schemas = {s["function"]["name"]: s["function"] for s in tool_schemas(ExplorationToolbox.specs)}
    for name in ("verify_goal", "finish"):
        relation = schemas[name]["parameters"]["properties"]["goal"]["properties"]["type"]
        assert relation["enum"] == ["in"]


def test_general_manipulation_still_offers_both_existing_relations():
    schema = next(s["function"] for s in tool_schemas() if s["function"]["name"] == "verify_goal")
    relation = schema["parameters"]["properties"]["goal"]["properties"]["type"]
    assert relation["enum"] == ["in", "on"]
