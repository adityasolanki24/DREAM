"""Contract tests for the non-actuating VoLoAgent adapter.

The message examples in ``tests/fixtures/volo_upstream_messages.json`` were recorded from the
real upstream code (``experiments/record_volo_messages.py``, commit b4e6230). VoLo itself is
optional: the tests that compare against it are skipped when it is not installed.
"""

import ast
import json
import pathlib

import numpy as np
import pytest

from dream.runtime import volo_contract as vc

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "volo_upstream_messages.json").read_text("utf-8")
)
DECISIONS = FIXTURE["tool_chain_decisions"]


def _wire_obs():
    rng = np.random.default_rng(0)
    obs = {"prompt": "pick up the red block"}
    for key, spec in FIXTURE["robolab_wire_observation"].items():
        if isinstance(spec, dict):
            obs[key] = np.zeros(spec["shape"], dtype=spec["dtype"]) + (0.5 if "depth" in key else 0)
        else:
            obs[key] = spec
    obs["observation/exterior_image_1_left"] = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    return obs


@pytest.mark.parametrize("name", sorted(DECISIONS))
def test_decision_validation_agrees_with_recorded_upstream_result(name):
    rec = DECISIONS[name]
    if rec["upstream_ok"]:
        d = vc.parse_volo_decision(rec["raw"])
        got = (d.subgoal_action, d.tool, d.args)
        assert got == (rec["subgoal_action"], rec["tool"], rec["args"])
    else:
        with pytest.raises(vc.VoLoContractError):
            vc.parse_volo_decision(rec["raw"])


def test_dream_is_stricter_than_upstream_only_where_documented():
    with pytest.raises(vc.VoLoContractError):
        vc.parse_volo_decision({"subgoal_action": "continue", "tool": "noop"})
    with pytest.raises(vc.VoLoContractError):
        vc.parse_volo_decision(
            {"subgoal_action": "continue", "tool": "grasp", "args": {"target": "x", "stack": "yes"}}
        )


def test_grasp_and_place_map_to_existing_dream_toolbox_signatures():
    from dream.tools.toolbox import SPECS

    grasp = vc.to_dream_call(vc.parse_volo_decision(DECISIONS["grasp_first"]["raw"]))
    assert grasp.name == "grasp" and grasp.args == {"object": "red block"} and not grasp.unresolved
    assert set(grasp.args) <= set(SPECS["grasp"]["parameters"])

    table = vc.parse_volo_decision(
        {
            "subgoal_action": "continue",
            "tool": "place",
            "args": {"destination": "on the table", "held_object_hint": "cube"},
        }
    )
    call = vc.to_dream_call(table)
    assert call.args == {"object": "cube", "target": "table"} and not call.unresolved
    assert set(call.args) <= set(SPECS["place"]["parameters"])

    bowl = vc.to_dream_call(vc.parse_volo_decision(DECISIONS["place_in_container"]["raw"]))
    assert bowl.unresolved == ("destination",)  # free-form phrase needs perception grounding
    assert vc.to_dream_call(vc.parse_volo_decision(DECISIONS["advance_noop"]["raw"])) is None


def test_recorded_wire_observation_is_valid_and_corruptions_are_reported():
    obs = _wire_obs()
    assert vc.validate_wire_observation(obs, need_grasp_keys=True) == []
    bad = dict(obs)
    bad["observation/joint_position"] = np.zeros(14)  # e.g. a two-armed robot
    assert any("joint_position" in p for p in vc.validate_wire_observation(bad))
    del bad["observation/camera_K"]
    assert any("camera_K" in p for p in vc.validate_wire_observation(bad, need_grasp_keys=True))


def test_action_chunks_validated_and_g1_dex3_mismatch_reported():
    assert vc.validate_wire_actions(np.zeros((8, 8), np.float32)) == []
    assert vc.validate_wire_actions(np.zeros((8, 29), np.float32))
    over = np.zeros((8, 8), np.float32)
    over[:, 7] = 2.0
    assert vc.validate_wire_actions(over)
    assert FIXTURE["passthrough_dry_run_observed"]["response_actions"]["shape"] == [8, 8]
    assert any("28" in r for r in vc.embodiment_mismatches())  # 2 * (7 arm + 7 hand)


def test_gr00t_native_message_facts_match_constants():
    assert tuple(FIXTURE["gr00t_resolution_hw"]) == vc.GR00T_IMAGE_HW
    native = FIXTURE["gr00t_native_observation"]
    assert native["state.joint_position"]["shape"] == [1, 1, vc.VOLO_ARM_DOF]
    assert native["video.exterior_image_1_left"]["shape"] == [1, 1, 180, 320, 3]
    act = FIXTURE["gr00t_native_action_for_8x8_chunk"]
    assert act["action.joint_position"]["shape"] == [1, FIXTURE["gr00t_min_chunk_len"], 7]
    assert act["action.gripper_position"]["shape"][-1] == 1


@pytest.mark.parametrize("name", ["grasp", "place"])
def test_skill_tool_never_executes_and_returns_dry_run_plan(name):
    tool = vc.VoLoSkillTool()
    args = (
        {"target": "red block"}
        if name == "grasp"
        else {"destination": "in the blue bowl", "held_object_hint": "red block"}
    )
    result = tool.call(name, **args)
    assert result.status == "rejected" and not result.ok
    assert result.message.startswith("not executable: ")
    plan = result.data["dry_run"]
    assert plan["executable"] is False and result.data["executable"] is False
    assert plan["volo_phases"][-1] == "done" and plan["requires_services"]
    assert plan["embodiment_mismatches"]


def test_skill_tool_rejects_invalid_or_unknown_calls():
    tool = vc.VoLoSkillTool()
    assert "invalid arguments" in tool.call("grasp").message
    assert "unknown VoLo skill" in tool.call("push", target="x").message
    assert tool.call("grasp", target="x", stack="maybe").status == "rejected"
    assert {s["name"] for s in tool.specs()} == {"volo_grasp", "volo_place"}


def test_original_passive_status_inspection_is_unchanged():
    assert vc.inspect_volo_contract({"status": "running", "completed": 1, "total": 4}).supported
    assert not vc.inspect_volo_contract({"actions": [[0.0] * 8]}).supported


def test_module_has_no_actuator_path_or_heavy_imports():
    path = pathlib.Path(vc.__file__)
    tree = ast.parse(path.read_text("utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    banned = (
        "mujoco",
        "dream.robot",
        "dream.sim",
        "dream.eval",
        "socket",
        "zmq",
        "websockets",
        "requests",
        "vlm_orchestrator",
    )
    assert not [n for n in names if n.startswith(banned)]
    assert not (set(dir(vc.VoLoSkillTool)) & {"execute", "actuate", "send", "run", "step"})


def test_skill_phase_constants_match_upstream_when_installed():
    pytest.importorskip("vlm_orchestrator")
    from vlm_orchestrator.grasp.tool import GraspPhase
    from vlm_orchestrator.place.tool import PlacePhase

    grasp = {p.value for p in GraspPhase} - {"idle", "failed"}
    place = {p.value for p in PlacePhase} - {"idle", "failed"}
    assert set(vc.GRASP_PHASES) == grasp and set(vc.PLACE_PHASES) == place
