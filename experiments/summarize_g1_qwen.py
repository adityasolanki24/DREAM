"""Summarize saved paired episodes without changing controller, gates or raw evidence."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def summarize(root: Path) -> dict:
    comparison = json.loads((root / "comparison.json").read_text())
    results = []
    for record in comparison["episodes"]:
        folder = root / record["backend"] / record["case"]
        episode = json.loads((folder / "episode.json").read_text())
        steps = episode.get("steps", [])
        failed = [i for i, s in enumerate(steps) if s.get("status") in ("failed", "stopped")]
        attempted_claims = [
            s
            for s in steps
            if (s.get("call") or {}).get("name") == "finish"
            and s["call"]["arguments"].get("success") is True
        ]
        rejected_claims = sum(bool(s.get("message")) for s in attempted_claims)
        outputs = [
            json.loads(p.read_text()) for p in sorted((folder / "inference").glob("*_output.json"))
        ]
        tools = json.loads((folder / "tools.json").read_text())
        belief = json.loads((folder / "belief.json").read_text())
        results.append(
            {
                **record,
                "raw_finish_success_attempts": len(attempted_claims),
                "rejected_success_attempts": rejected_claims,
                "inference_outputs": sum("text" in o for o in outputs),
                "inference_errors": [o for o in outputs if "error" in o],
                "total_inference_wall_s": sum(o["latency_s"] for o in outputs),
                "model_tool_order": [(s.get("call") or {}).get("name") for s in steps]
                if record["backend"] == "qwen"
                else [],
                "failure_followups": [
                    {
                        "failed_step": i,
                        "next_call": steps[i + 1].get("call"),
                        "next_tool": steps[i + 1].get("tool"),
                    }
                    for i in failed
                    if i + 1 < len(steps)
                ],
                "accepted_belief_updates": sum(e.get("update_accepted", False) for e in belief),
                "tool_failures": [
                    {k: e.get(k) for k in ("tool", "status", "message")}
                    for e in tools
                    if e["status"] in ("failed", "stopped")
                ],
                "recovered_to_goal_after_failure": bool(failed and record["goal_success"]),
            }
        )
    groups = {}
    for backend in ("qwen", "deterministic"):
        rows = [r for r in results if r["backend"] == backend]
        groups[backend] = {
            "n": len(rows),
            "independent_goal_successes": sum(r["goal_success"] for r in rows),
            "accepted_false_claims": sum(r["false_success_claim"] for r in rows),
            "rejected_success_attempts": sum(r["rejected_success_attempts"] for r in rows),
            "penetration_failures": sum(not r["penetration_gate_passed"] for r in rows),
            "safety_stops": sum(r["safety_stop"] for r in rows),
            "episodes_with_action_failures": sum(bool(r["tool_failures"]) for r in rows),
            "recovered_to_goal_after_failure": sum(
                r["recovered_to_goal_after_failure"] for r in rows
            ),
        }
    return {
        "scope": comparison["scope"],
        "groups": groups,
        "episodes": results,
        "limitations": "All cases previously observed; constrained case initially overlaps. "
        "Synchronous physics during tools only. Recovery-to-goal is measured after failed "
        "actions, not inferred from generated prose. No generalisation inference.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, help="Optional byte-preserving archive of guest out/"
    )
    args = parser.parse_args()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(summarize(args.root), stream, indent=2)
        stream.write("\n")
    if args.archive is not None:
        source = args.root.parent.resolve()
        if args.archive.resolve().is_relative_to(source):
            raise ValueError("archive must be outside guest output")
        with tarfile.open(args.archive, "x:gz") as archive:
            archive.add(source, arcname="out")
