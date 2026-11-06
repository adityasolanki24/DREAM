"""Summarise the retrieved live Isaac G1 + Qwen sessions into one per-task table.

Reads only the retrieved files under ``results/isaac_qwen_live_20261101/<session>/`` (and each
session's ``raw.zip``); writes ``summary.json`` beside them. Nothing is rescored: the
original ``*_evaluation.json`` fields are copied verbatim as ``original_scorer`` and anything
a session did not record is ``null``.

    python -m experiments.summarize_isaac_live [--dir results/isaac_qwen_live_20261101]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path

LABELS = {
    "evidence_class": "development, interactive, not predeclared; not a held-out evaluation",
    "environment": "simulation (NVIDIA Isaac Sim 6.2 / PhysX, Unitree G1 under the unchanged "
    "NVIDIA Decoupled WBC)",
    "model": "genuine Qwen3-VL-4B-Instruct, revision ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "odometry": "oracle-assisted (true increments plus noise); not perception or SLAM",
    "scoring": "original evaluation JSONs are verbatim; this table does not rescore anything",
    "scorer_caveat": "original_scorer is the cabinet-handle scorer of that commit. For a task "
    "that was not a cabinet inspection its false_claim reflects that scorer, not a measurement "
    "of the requested object.",
}
SCORER_KEYS = (
    "success",
    "false_claim",
    "identity_error_m",
    "true_handle_distance_m",
    "facing_error_deg",
    "on_front_side",
    "phantom_handles",
    "candidates",
    "candidates_with_handle",
)


def jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def failure_stage(state: str, result: str | None) -> str | None:
    if state == "succeeded":
        return None
    result = result or ""
    for prefix, stage in (
        ("execution error", "runtime exception before the agent finished"),
        ("verify_goal failed", "goal verification rejected the bound goal"),
        ("model repair budget exhausted", "model repair budget exhausted"),
        ("step budget", "step budget exhausted"),
        ("robot in protective stop", "protective stop"),
    ):
        if result.startswith(prefix):
            return stage
    return "agent declined or goal not verified" if state == "failed" else state


def session_summary(directory: Path) -> dict:
    job = json.loads((directory / "job.json").read_text())
    stages = jsonl((directory / "stages.jsonl").read_text())
    events = jsonl((directory / "tool_events.jsonl").read_text())
    evaluations = {
        re.match(r"task_(.+)_evaluation\.json", p.name)[1]: json.loads(p.read_text())
        for p in directory.glob("task_*_evaluation.json")
    }
    physical = (
        json.loads((directory / "physical_outcome.json").read_text())
        if (directory / "physical_outcome.json").is_file()
        else None
    )
    decisions = [
        json.loads(p.read_text()) for p in sorted((directory / "qwen_decisions").glob("*.json"))
    ]
    with zipfile.ZipFile(directory / "raw.zip") as z:
        trajectory = jsonl(z.read("out/jobs/1/live/trajectory.jsonl").decode())
        exit_code = (
            z.read("out/jobs/1/EXIT_CODE").decode().strip()
            if "out/jobs/1/EXIT_CODE" in z.namelist()
            else None
        )
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            db.write_bytes(z.read("out/jobs/1/live/tasks.sqlite"))
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            tasks = [dict(r) for r in conn.execute("select * from tasks order by created")]
            conn.close()
    table = []
    for t in tasks:
        mine = [e for e in events if e.get("id") == t["id"]]
        goal_event = next((e for e in mine if e["kind"] == "goal"), None)
        finish = next((e for e in mine if e["kind"] == "finish"), None)
        started = goal_event["sim_time_s"] if goal_event else None
        ended = finish["sim_time_s"] if finish else None
        window = (
            [r for r in trajectory if started <= r["sim_time_s"] <= ended]
            if started is not None and ended is not None
            else []
        )
        results = [e for e in mine if e["kind"] == "result"]
        arrival = [
            e["data"]["error_m"]
            for e in results
            if isinstance(e.get("data"), dict) and "error_m" in e["data"]
        ]
        ev = evaluations.get(t["id"])
        scorer = (
            None if ev is None else {k: ev["independent_verification"].get(k) for k in SCORER_KEYS}
        )
        reference = re.search(r"reference='([^']+)'", t["goal"] or "")
        table.append(
            {
                "task_id": t["id"],
                "request": t["text"],
                "goal": t["goal"],
                "bound_reference": reference[1] if reference else None,
                "scheduler_state": t["state"],
                "success_claimed": (
                    finish["success"]
                    if finish and "success" in finish
                    else (ev["episode"]["success_claimed"] if ev else None)
                ),
                "final_reason": (finish or {}).get("reason", t["result"]),
                "failure_stage": failure_stage(t["state"], t["result"]),
                "sim_start_s": started,
                "sim_end_s": ended,
                "sim_duration_s": None
                if started is None or ended is None
                else round(ended - started, 3),
                "tool_steps": [
                    {
                        "tool": e["tool"],
                        "status": e["status"],
                        "error_m": (e.get("data") or {}).get("error_m"),
                    }
                    for e in results
                ],
                "arrival_errors_m": arrival or None,
                "agent_verdict": None if ev is None else ev["agent_verdict"],
                "original_scorer": scorer,
                "original_scorer_file": None if ev is None else f"task_{t['id']}_evaluation.json",
                "min_pelvis_height_m": min(
                    (r["height_evaluation_only"] for r in window), default=None
                ),
                "contacts_in_session": None if physical is None else len(physical["contacts"]),
            }
        )
    inference = [
        d["inference_s"] for d in decisions if isinstance(d.get("inference_s"), (int, float))
    ]
    gpu = json.loads((directory / "gpu_stats.json").read_text())
    return {
        "session": directory.name,
        "git_sha": job["git_sha"],
        "job_args": job["args"],
        "exit_code": exit_code,
        "final_stage": stages[-1]["stage"] if stages else None,
        "wall_s_to_last_stage": stages[-1].get("wall_s") if stages else None,
        "qwen_decisions": len(decisions),
        "qwen_inference_s_mean": round(sum(inference) / len(inference), 2) if inference else None,
        "physical_outcome_recorded": physical is not None,
        "final_height_m": None
        if physical is None
        else physical["final_height_m_for_evaluation_only"],
        "gpu_whole_run": gpu["whole_run"],
        "tasks": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results/isaac_qwen_live_20261101"))
    args = parser.parse_args()
    sessions = [session_summary(d) for d in sorted(args.dir.iterdir()) if (d / "raw.zip").is_file()]
    summary = {"labels": LABELS, "sessions": sessions}
    (args.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{sum(len(s['tasks']) for s in sessions)} tasks in {len(sessions)} sessions")


if __name__ == "__main__":
    main()
