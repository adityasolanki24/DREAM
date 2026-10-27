# ruff: noqa: E501
"""Record synthetic parser cases and wire-shape examples from pinned VoLoAgent code.

Run this in the SEPARATE VoLo environment (never DREAM's .venv); it imports upstream code:

    <voloenv>/Scripts/python.exe experiments/record_volo_messages.py <VoLoAgent clone> <out.json>

Nothing here actuates a robot. The upstream clone is pinned by the commit recorded in the output.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DECISIONS = {
    "grasp_first": '{"subgoal_action": "continue", "tool": "grasp", "args": {"target": "red block"}, "reason": "Subgoal 0 needs the red block; grasping it."}',
    "place_in_container": '{"subgoal_action": "continue", "tool": "place", "args": {"destination": "in the blue bowl", "held_object_hint": "red block"}, "reason": "Block is in gripper; placing into the bowl now."}',
    "place_on_table_stack_false": '{"subgoal_action": "continue", "tool": "place", "args": {"destination": "empty space on the table", "held_object_hint": "yellow lemon", "stack": false}, "reason": "Lemon is grasped, setting it down on the table."}',
    "grasp_stack_true": '{"subgoal_action": "continue", "tool": "grasp", "args": {"target": "blue block", "stack": true}, "reason": "Grasp for stacking."}',
    "advance_noop": '{"subgoal_action": "advance", "tool": "noop", "args": {}, "reason": "Red block visible in the blue bowl; subgoal 0 done."}',
    "replan_noop": '{"subgoal_action": "replan", "tool": "noop", "args": {}, "reason": "Blue bowl tipped over; re-decompose from current scene."}',
    "abort_noop": '{"subgoal_action": "abort", "tool": "noop", "args": {}, "reason": "Unrecoverable."}',
    "bad_advance_with_tool": '{"subgoal_action": "advance", "tool": "place", "args": {"destination": "on the table"}, "reason": "x"}',
    "bad_grasp_without_target": '{"subgoal_action": "continue", "tool": "grasp", "args": {}, "reason": "x"}',
    "bad_unknown_tool": '{"subgoal_action": "continue", "tool": "push", "args": {}, "reason": "x"}',
    "bad_not_json": "I will grasp the red block now.",
}


def evidence_provenance() -> dict[str, str | bool]:
    """Describe the evidence this recorder can produce; it does not run a VLM or proxy."""
    return {
        "decision_inputs": "synthetic_strings_passed_to_real_upstream_parser",
        "vlm_outputs_observed": False,
        "wire_observations": "synthetic_schema_examples_generated_by_upstream_helper",
        "passthrough_proxy_run": False,
    }


def main(clone: Path, out: Path) -> None:
    sys.path.insert(0, str(clone))
    from vlm_orchestrator.protocols import gr00t_zmq
    from vlm_orchestrator.strategies.tool_chain import parse_tool_chain_response

    decisions = {}
    for name, raw in DECISIONS.items():
        try:
            d = parse_tool_chain_response(raw)
            decisions[name] = {
                "raw": raw,
                "upstream_ok": True,
                "subgoal_action": d.subgoal_action,
                "tool": d.tool,
                "args": d.args,
            }
        except ValueError as exc:
            decisions[name] = {"raw": raw, "upstream_ok": False, "error": str(exc)[:200]}

    spec = importlib.util.spec_from_file_location(
        "smoke", clone / "examples" / "schema_smoke_test.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    wire = smoke._make_wire_obs(0, "ep0", "pick up the red block", True)
    wire_desc = {
        k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if isinstance(v, np.ndarray) else v)
        for k, v in wire.items()
    }

    canon = dict(wire)
    canon["observation/ee_pos"] = np.array([0.4, 0.0, 0.3], np.float32)
    g_obs = gr00t_zmq.canonical_to_gr00t(canon)
    g_obs_desc = {
        k: ({"shape": list(v.shape), "dtype": str(v.dtype)} if isinstance(v, np.ndarray) else v)
        for k, v in g_obs.items()
    }
    g_act = gr00t_zmq.canonical_action_to_gr00t({"actions": np.zeros((8, 8), np.float32)})
    g_act_desc = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in g_act.items()}

    sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    payload = {
        "upstream": "NVlabs/VoLoAgent",
        "commit": sha,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "vlm_orchestrator": importlib.metadata.version("vlm-orchestrator"),
        },
        "evidence_provenance": evidence_provenance(),
        "gr00t_resolution_hw": list(gr00t_zmq.GR00T_RESOLUTION),
        "gr00t_min_chunk_len": gr00t_zmq.GR00T_MIN_CHUNK_LEN,
        "tool_chain_decisions": decisions,
        "robolab_wire_observation": wire_desc,
        "gr00t_native_observation": g_obs_desc,
        "gr00t_native_action_for_8x8_chunk": g_act_desc,
    }
    with out.open("x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} (commit {sha})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
