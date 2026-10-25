"""Summarize the frozen paired evaluation from original inference and physics logs.

Missing attempts stay in denominators. Rejected model success assertions are separate
from accepted false claims. Camera-observed grasp loss is not labelled a confirmed slip;
original contact/pose traces and both videos remain available for that diagnosis.
"""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.cloud_g1_qwen import validate_evidence


def read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def report(out: Path, config: Path) -> dict:
    cfg = read(config, {})
    frozen = read(out / "freeze.json", {})
    model = read(out / "model.json", {})
    published = {
        (r["backend"], r["case"]): r
        for r in read(out / "comparison.json", {"episodes": []})["episodes"]
    }
    expected_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if frozen and frozen["config_sha256"] != expected_digest:
        raise ValueError("evaluation configuration differs from its original freeze")
    result = {
        "source_commit": frozen.get("base_git_sha"),
        "config_sha256": expected_digest,
        "model": {k: model.get(k) for k in ("checkpoint", "revision", "device", "gpu", "torch")},
        "evidence_gate_passed": False,
        "episodes": [],
        "totals": {},
        "learned_motor_calls": 0,
        "scope": "synchronous fixed-pelvis sphere/platform task; classical motor tools; "
        "within-family held-out cases, not open-world generalisation or real-time control",
    }
    try:
        validate_evidence(out, config)
        result["evidence_gate_passed"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["evidence_gate_error"] = str(error)
    for case in cfg["cases"]:
        for backend in ("qwen", "deterministic"):
            folder = out / backend / case["name"]
            summary = read(folder / "summary.json", {})
            tools = read(folder / "tools.json", [])
            episode = read(folder / "episode.json", {})
            outputs = [read(p, {}) for p in sorted((folder / "inference").glob("*_output.json"))]
            success_attempts = [
                s
                for s in episode.get("steps", [])
                if (s.get("call") or {}).get("name") == "finish"
                and s["call"].get("arguments", {}).get("success") is True
            ]
            physics = folder / "physics.jsonl"
            peak_hand_force = 0.0 if physics.exists() else None
            if physics.exists():
                with physics.open() as trace:
                    for line in trace:
                        row = json.loads(line)
                        peak_hand_force = max(
                            peak_hand_force,
                            sum(
                                c["normal_force_n"]
                                for c in row["contacts"]
                                if "_hand_" in c["body"]
                            ),
                        )
            failures = {"failed", "stopped", "rejected"}
            actions = [e for e in tools if e["tool"] not in ("observe_scene", "verify_goal")]
            row = {
                "case": case["name"],
                "seed": case["seed"],
                "backend": backend,
                "split": case.get("split", "development"),
                "record_present": bool(summary),
                **summary,
                "official_outcome_available": bool(summary)
                and (backend, case["name"]) in published,
                "agent_verified_task_completion": bool(summary.get("goal_success"))
                and bool(episode.get("success_claimed")),
                "model_calls_attempted": len(outputs),
                "model_outputs_completed": sum(bool(o.get("text")) for o in outputs),
                "model_latency_s": sum(o.get("latency_s", 0) for o in outputs),
                "rejected_success_assertions": sum(
                    "success rejected" in s.get("message", "") for s in success_attempts
                ),
                "observed_grasp_tracking_loss_events": sum(
                    (e.get("data") or {}).get("grasp_observation") == "object no longer tracks hand"
                    for e in tools
                ),
                "inspection_after_failure": sum(
                    e["tool"] == "inspect_object" and previous["status"] in failures
                    for previous, e in zip(actions, actions[1:], strict=False)
                ),
                "peak_scored_hand_object_normal_force_n": peak_hand_force,
                "tool_failures": [
                    {k: e.get(k) for k in ("tool", "status", "message", "time_s")}
                    for e in tools
                    if e["status"] in failures
                ],
            }
            result["episodes"].append(row)
    result["planned_attempts"] = len(result["episodes"])
    result["finalized_attempts"] = sum(r["official_outcome_available"] for r in result["episodes"])
    result["incomplete_attempts"] = result["planned_attempts"] - result["finalized_attempts"]
    result["missing_physical_records"] = sum(not r["record_present"] for r in result["episodes"])
    result["evaluation_complete"] = (
        result["incomplete_attempts"] == 0 and result["evidence_gate_passed"]
    )
    result["partial_outcome_policy"] = (
        "Terminal summaries from unpublished attempts remain diagnostics only. Offline video "
        "recovery cannot finalize the original comparison or change its outcome denominator."
    )
    for backend in ("qwen", "deterministic"):
        for split in sorted({c.get("split", "development") for c in cfg["cases"]}):
            rows = [
                r for r in result["episodes"] if r["backend"] == backend and r["split"] == split
            ]
            result["totals"][f"{backend}/{split}"] = {
                "attempts": len(rows),
                "record_present": sum(r["record_present"] for r in rows),
                "completed_attempts": sum(r["official_outcome_available"] for r in rows),
                "incomplete_attempts": sum(not r["official_outcome_available"] for r in rows),
                **{
                    key: sum(bool(r.get(key)) and r["official_outcome_available"] for r in rows)
                    for key in (
                        "goal_success",
                        "agent_verified_task_completion",
                        "correct_safe_refusal",
                        "conditional_task_success",
                        "false_success_claim",
                        "safety_stop",
                    )
                },
                "rejected_success_assertions": sum(r["rejected_success_assertions"] for r in rows),
                "model_calls": sum(r["model_calls_attempted"] for r in rows),
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/g1_autonomy_frozen.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.run, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["totals"], indent=2))
