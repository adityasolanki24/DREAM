# ruff: noqa: E501
"""Record real VoLoAgent message examples as a fixture for DREAM's contract tests.

Run this in the SEPARATE VoLo environment (never DREAM's .venv); it imports upstream code:

    <voloenv>/Scripts/python.exe experiments/record_volo_messages.py <VoLoAgent clone> <out.json>

Nothing here actuates a robot. The upstream clone is pinned by the commit recorded in the output.
"""

from __future__ import annotations

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
        "gr00t_resolution_hw": list(gr00t_zmq.GR00T_RESOLUTION),
        "gr00t_min_chunk_len": gr00t_zmq.GR00T_MIN_CHUNK_LEN,
        "tool_chain_decisions": decisions,
        "robolab_wire_observation": wire_desc,
        "gr00t_native_observation": g_obs_desc,
        "gr00t_native_action_for_8x8_chunk": g_act_desc,
        "passthrough_dry_run_observed": {
            "command": "mock openpi VLA :8000 + vlm-orchestrator --mode passthrough :8001 + examples/schema_smoke_test.py --grasp --steps 3",
            "metadata_keys": ["model", "orchestrator"],
            "response_actions": {"shape": [8, 8], "dtype": "float32"},
            "result": "PASS",
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} (commit {sha})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
