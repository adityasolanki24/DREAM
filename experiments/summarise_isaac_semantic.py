"""Merge the retrieved jobs of the Isaac semantic evaluation into results/isaac_semantic_eval.

    python experiments/summarise_isaac_semantic.py --out results/isaac_semantic_eval \
        --run <session>:<job>:<path to the job's output folder holding summary.json> ...

Reads each job's ``summary.json`` (written by experiments/isaac_semantic_eval.py), merges them
(a case run in several jobs keeps its latest non-deferred record), applies the config's gates
and writes ``summary.json`` with the gates, the per-case table, versions, SHAs and the cost
ledger estimate, plus a compact per-case evidence copy (evaluation, grounding and detector
JSON; no images or sensor arrays, which stay in the raw archives outside Git).
Cost figures are estimates from list prices and lifetime, not billed amounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.isaac_semantic_eval import CONFIG, load_config, merge_summaries  # noqa: E402

CLOUD_RUNS = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "DREAM/cloud_runs"
EVIDENCE_GLOBS = (
    "task_*_evaluation.json",
    "isaac_session.json",
    "scoring_instances.json",
    "physical_outcome_summary.json",
)


def table_row(case: dict) -> dict:
    keys = (
        "case_id", "split", "kitchen", "start", "arm", "task", "text", "original_result",
        "success_claimed", "false_success", "completion", "completion_source", "goal",
        "goal_kind", "tool_calls", "episode_reason", "fell", "protective_stop",
        "evaluation_present", "scheduler_state", "timed_out", "process_died", "refusal",
    )  # fmt: skip
    return {k: case.get(k) for k in keys}


def copy_evidence(run_dir: pathlib.Path, case_id: str, out: pathlib.Path) -> int:
    """Compact evidence of one case. Returns the bytes copied."""
    src = run_dir / case_id
    if not src.is_dir():
        return 0
    dst = out / "cases" / case_id
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    live = src / "live"
    sources = [src / "phone_tasks.json", src / "phone_events.json"]
    for pattern in EVIDENCE_GLOBS:
        sources += sorted(live.glob(pattern))
    for decisions in ("qwen_decisions", "qwen_detector"):
        sources += sorted((live / decisions).glob("*.json"))
    sources += [live / "tool_events.jsonl", live / "physical_outcome.json"]
    for path in sources:
        if path.is_file():
            target = dst / (
                path.parent.name + "__" + path.name
                if path.parent.name.startswith("qwen")
                else path.name
            )
            shutil.copyfile(path, target)
            total += target.stat().st_size
    log = src / "driver_live_stdout.log"
    if log.is_file():  # the tail only: the head is Isaac start-up warnings
        lines = [
            ln for ln in log.read_text(errors="replace").splitlines() if "PhysX warning" not in ln
        ]
        (dst / "driver_live_stdout_tail.txt").write_text("\n".join(lines[-80:]), encoding="utf-8")
    return total


def session_row(session: str) -> dict:
    run = CLOUD_RUNS / session
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    ledger = {}
    for path in [
        pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "DREAM/cloud/isaac_ledger.json"
    ]:
        if path.is_file():
            for entry in json.loads(path.read_text(encoding="utf-8")):
                if entry["name"] == session:
                    ledger = entry
    post = run / "post_audit.json"
    errors = run / "cleanup_errors.json"
    return {
        "session": session,
        "started_utc": manifest.get("started_utc"),
        "ended_utc": ledger.get("ended_utc"),
        "max_minutes": manifest.get("max_minutes"),
        "estimated_usd": manifest.get("estimated_usd"),
        "reconciled_usd": ledger.get("reconciled_usd"),
        "git_sha_at_start": manifest.get("git_sha"),
        "cleanup_errors": json.loads(errors.read_text()) if errors.is_file() else None,
        "post_audit_present": post.is_file(),
        "retrieved": ledger.get("retrieved"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--run", action="append", required=True, help="session:job:folder")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--extra-sessions", nargs="*", default=[], help="non-matrix sessions")
    args = parser.parse_args()
    cfg = load_config(args.config, args.allow_draft)
    summaries, jobs, copied = [], [], 0
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.run:
        session, job, folder = spec.split(":", 2)
        run_dir = pathlib.Path(folder)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        job_json = json.loads((CLOUD_RUNS / session / "jobs" / job / "job.json").read_text())
        ran = [c for c in summary["cases"] if not c.get("deferred")]
        jobs.append(
            {
                "session": session,
                "job": job,
                "git_sha": job_json["git_sha"],
                "archive_sha256": job_json["archive_sha256"],
                "args": job_json["args"],
                "model_server": summary.get("model_server"),
                "cases": len(ran),
            }
        )
        for case in summary["cases"]:
            if not case.get("deferred"):
                copied += copy_evidence(run_dir, case["case_id"], args.out)
    merged = merge_summaries(cfg, summaries)
    sessions = [session_row(s) for s in dict.fromkeys(j["session"] for j in jobs)]
    sessions += [session_row(s) for s in args.extra_sessions]
    estimated = round(sum(float(s["estimated_usd"] or 0) for s in sessions), 2)
    reconciled = round(
        sum(float(s["reconciled_usd"] or s["estimated_usd"] or 0) for s in sessions), 2
    )
    result = {
        "config": cfg["name"],
        "config_status": cfg["status"],
        "config_frozen_at_commit": cfg.get("frozen_at_commit"),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": cfg["model"],
        "model_revision": cfg["model_revision"],
        "environment": cfg["environment"],
        "jobs": jobs,
        "sessions": sessions,
        "cost_estimate_usd": {
            "sum_of_full_estimates": estimated,
            "sum_reconciled_by_lifetime": reconciled,
            "note": "estimates from list prices and VM lifetime; billed cost not read",
        },  # fmt: skip
        "gates": merged["gates"],
        "reports": merged["reports"],
        "limitations": cfg["limitations"],
        "table": [table_row(c) for c in merged["cases"]],
        "cases_planned": len(merged["cases"]),
        "cases_deferred": [c["case_id"] for c in merged["cases"] if c.get("deferred")],
        "evidence_bytes_copied": copied,
    }
    (args.out / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (args.out / "merged_cases.json").write_text(
        json.dumps(merged["cases"], indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"gates": merged["gates"], "cost": result["cost_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    main()
