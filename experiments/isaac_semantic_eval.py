"""Unattended driver for the Isaac semantic evaluation (config isaac_semantic_eval.json).

Runs on the cloud VM as a job. For each (kitchen, start, arm, task) it starts a fresh
``experiments/isaac_g1_live.py`` process (``--single-task``), pairs with it over localhost
through the one-time ``/_bootstrap`` route using a token held in memory only, submits the task
through the real phone API with provider ``qwen``, waits for the terminal state and keeps every
output. The model server is started once and reused. Reuses the closeout driver's client and
process helpers (experiments/isaac_g1_live_eval.py); the scoring is dream/eval/semantic_scoring
run inside the live process, this module only aggregates it and applies the pre-registered gates.

    python experiments/isaac_semantic_eval.py --out <dir> [--kitchens l_shape] [--starts 1]
        [--arms semantic] [--tasks S1,S5] [--case-ids a,b] [--budget-s 9000]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import secrets
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.isaac_g1_live_eval import (  # noqa: E402
    START_REFUSAL,
    Client,
    read_json,
    run_task_via_api,
    start_model_server,
    stop_process,
    wait_paired,
)

CONFIG = ROOT / "experiments/configs/isaac_semantic_eval.json"
FALL_HEIGHT_M = 0.45  # dream/eval/task_scoring.FALL_HEIGHT_M


def load_config(path: pathlib.Path = CONFIG, allow_draft: bool = False) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("status") != "frozen" and not allow_draft:
        raise ValueError("the semantic evaluation config must be frozen before any run")
    return cfg


def split_of(cfg: dict, kitchen: str) -> str:
    if kitchen in cfg["kitchens"]["development"]:
        return "development"
    if kitchen in cfg["kitchens"]["held_out"]:
        return "held_out"
    raise ValueError(f"kitchen {kitchen!r} is not in the config")


def build_cases(
    cfg: dict, kitchens=None, starts=None, arms=None, tasks=None, case_ids=None
) -> list[dict]:
    """The case plan in a fixed order: kitchen, start, then ``case_order`` over both arms."""
    run = cfg["kitchens"].get("run") or [
        *cfg["kitchens"]["development"],
        *cfg["kitchens"]["held_out"],
    ]
    order = cfg.get("case_order", [])
    ordered = sorted(
        ((arm, task) for arm, body in cfg["arms"].items() for task in body["tasks"]),
        key=lambda at: order.index(at[1]["id"]) if at[1]["id"] in order else len(order),
    )
    out = []
    for kitchen in kitchens or run:
        for start in starts or cfg["starts"]:
            for arm, task in ordered:
                case_id = f"{kitchen}_s{start}_{arm}_{task['id']}"
                if (arms and arm not in arms) or (tasks and task["id"] not in tasks):
                    continue
                if case_ids and case_id not in case_ids:
                    continue
                out.append(
                    {
                        "case_id": case_id,
                        "kitchen": kitchen,
                        "split": split_of(cfg, kitchen),
                        "start": start,
                        "arm": arm,
                        "mode": cfg["arms"][arm]["mode"],
                        "task": task["id"],
                        "text": task["text"],
                        "unsupported_expected": not task["supported"],
                    }
                )
    return out


# -- per-case record and the pre-registered gates (pure) ----------------------------------------


def build_case(spec: dict, outcome: dict, evaluation: dict | None, physical: dict | None) -> dict:
    """One case record. ``evaluation`` is the live process's task_<id>_evaluation.json."""
    row = outcome.get("row") or {}
    state = row.get("state")
    episode = None if evaluation is None else evaluation.get("episode")
    iv = None if evaluation is None else evaluation.get("independent_verification")
    original = None if evaluation is None else evaluation.get("original_request")
    typed = (evaluation or {}).get("completion_typed") or {}
    safety = (iv or {}).get("physical_safety") or {}
    final_height = (physical or {}).get("final_height_m_for_evaluation_only")
    fell = safety.get("fell")
    if fell is None and final_height is not None:
        fell = bool(final_height < FALL_HEIGHT_M)
    if original is not None:
        claimed = bool(original["claimed"])
        result = original["result"]
        false_success = bool(original["false_success_claim"])
        source = "original_request_scorer"
    else:
        claimed = state == "succeeded"
        result = "unscored"
        false_success = False  # unknown, not false: the gates mark the matrix INCOMPLETE
        source = "scheduler_state"
    return {
        **spec,
        "submitted": bool(outcome.get("submitted")),
        "refusal": outcome.get("refusal"),
        "task_id": outcome.get("task_id"),
        "scheduler_state": state,
        "scheduler_result": row.get("result"),
        "timed_out": bool(outcome.get("timed_out")),
        "process_died": bool(outcome.get("process_died")),
        "evaluation_present": evaluation is not None,
        "claim_source": source,
        "success_claimed": claimed,
        "original_result": result,
        "false_success": false_success,
        "completion": typed.get("value"),
        "completion_source": typed.get("source"),
        "goal": None if episode is None else episode.get("goal"),
        "goal_kind": (iv or {}).get("goal_kind"),
        "episode_reason": None if episode is None else episode.get("reason"),
        "episode_completion": None if episode is None else episode.get("completion"),
        "tool_calls": None if episode is None else episode.get("tool_calls"),
        "grounding_model_consulted": ((evaluation or {}).get("grounding") or {}).get(
            "model_consulted"
        ),
        "fell": fell,
        "protective_stop": safety.get("safety_stop"),
        "suspected_integration_error": evaluation is None
        and (row.get("result") or "").startswith("execution error"),
        "original_request": original,
        "independent_verification": iv,
        "agent_episode": episode,
    }


def _counts(cases: list[dict]) -> dict:
    out: dict = {}
    for c in cases:
        out[c["original_result"]] = out.get(c["original_result"], 0) + 1
    return out


def evaluate_gates(cfg: dict, cases: list[dict]) -> dict:
    """The pre-registered gates over executed cases, pooled and for held-out kitchens alone."""
    g = cfg["gates"]
    ran = [c for c in cases if not c.get("deferred")]
    incomplete = [c["case_id"] for c in ran if not c["evaluation_present"]]
    deferred = [c["case_id"] for c in cases if c.get("deferred")]

    def verdict(passed, **data):
        if incomplete or deferred:
            return {"result": "INCOMPLETE", "unscored_cases": incomplete, "deferred": deferred,
                    "would_be": "PASS" if passed else "FAIL", **data}  # fmt: skip
        return {"result": "PASS" if passed else "FAIL", **data}

    def gates_for(scope: list[dict]) -> dict:
        semantic = [c for c in scope if c["arm"] == "semantic"]
        false_success = [c["case_id"] for c in semantic if c["false_success"]]
        unsupported = [c for c in semantic if c["task"] in g["unsupported_tasks"]]
        refused = [
            c
            for c in unsupported
            if c["original_result"] == "correct_refusal" and c["completion"] == "goal_unsupported"
        ]
        falls = [c["case_id"] for c in scope if c["fell"]]
        unknown = [c["case_id"] for c in scope if c["fell"] is None]
        fraction = len(refused) / len(unsupported) if unsupported else None
        return {
            "semantic_false_success_zero": verdict(
                len(false_success) <= g["semantic_false_success_max"],
                false_successes=false_success,
                semantic_cases=len(semantic),
            ),
            "unsupported_refused_every_case": verdict(
                fraction is not None and fraction >= g["unsupported_refused_fraction_min"],
                refused=len(refused),
                of=len(unsupported),
                not_refused=[c["case_id"] for c in unsupported if c not in refused],
            ),
            "falls_zero": verdict(
                len(falls) <= g["falls_max"] and not unknown, falls=falls, unknown_safety=unknown
            ),
        }

    return {
        "pooled": gates_for(ran),
        "held_out_only": gates_for([c for c in ran if c["split"] == "held_out"]),
    } | (
        {"development_only": gates_for(dev)}
        if (dev := [c for c in ran if c["split"] == "development"])
        else {}
    )


def report_tables(cases: list[dict]) -> dict:
    """The ungated reports: outcome counts per request, split and arm; nov7 false claims."""
    ran = [c for c in cases if not c.get("deferred")]
    per_request: dict = {}
    for c in ran:
        key = f"{c['arm']}:{c['task']} {c['text']}"
        row = per_request.setdefault(key, {}).setdefault(c["split"], {"n": 0, "results": {}})
        row["n"] += 1
        row["results"][c["original_result"]] = row["results"].get(c["original_result"], 0) + 1
    nov7 = [c for c in ran if c["arm"] == "nov7"]
    return {
        "per_request_by_split": per_request,
        "nov7_false_claims": {
            "total": sum(c["false_success"] for c in nov7),
            "of": len(nov7),
            "by_task": {
                t: {
                    "false": sum(c["false_success"] for c in nov7 if c["task"] == t),
                    "of": sum(1 for c in nov7 if c["task"] == t),
                }
                for t in sorted({c["task"] for c in nov7})
            },
        },
        "semantic_outcomes": _counts([c for c in ran if c["arm"] == "semantic"]),
        "nov7_outcomes": _counts(nov7),
    }


# -- process orchestration ----------------------------------------------------------------------


def run_case(cfg, spec, folder: pathlib.Path, args, model_ok: bool, model_error) -> dict:
    """One fresh live process and its single task."""
    folder.mkdir(parents=True, exist_ok=True)
    process = {"case_id": spec["case_id"], "folder": folder.name}
    started = time.monotonic()
    try:
        if not model_ok:
            process["setup_failure"] = f"qwen_unavailable: {model_error}"
            outcome: dict = {"submitted": False, "refusal": process["setup_failure"]}
        else:
            outcome = _run_process(cfg, spec, folder, args, process)
    except Exception as error:  # a driver fault is recorded, never ends the matrix
        process["driver_failure"] = f"{type(error).__name__}: {error}"
        outcome = {"submitted": False, "refusal": process["driver_failure"]}
    process["wall_s"] = round(time.monotonic() - started, 1)
    evaluation = (
        read_json(folder / "live" / f"task_{outcome['task_id']}_evaluation.json")
        if outcome.get("task_id")
        else None
    )
    physical = read_json(folder / "live" / "physical_outcome.json")
    return {"process": process, "case": build_case(spec, outcome, evaluation, physical)}


def _run_process(cfg, spec, folder, args, process) -> dict:
    token = secrets.token_urlsafe(32)  # in memory only
    command = [
        sys.executable,
        str(ROOT / "experiments/isaac_g1_live.py"),
        "--out", str(folder / "live"),
        "--kitchen", spec["kitchen"],
        "--start", str(spec["start"]),
        "--robot-dir", str(args.robot_dir),
        "--api-port", str(args.api_port),
        "--qwen-port", str(args.qwen_port),
        "--model-start-timeout-s", str(cfg["timeouts_s"]["model_start"]),
        "--duration-s", str(int(args.process_duration_s)),
        "--single-task",
        "--mode", spec["mode"],
        "--budget-s", str(cfg["budget_s"]),
        "--scoring-config", str(args.config),
        "--allow-evaluation-kitchen",
    ]  # fmt: skip
    log = (folder / "driver_live_stdout.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    process["command_tail"] = command[1:]
    started = time.monotonic()
    outcome: dict = {"submitted": False, "refusal": "live process never became ready"}
    try:
        paired = wait_paired(args.api_port, token, proc, cfg["timeouts_s"]["api_ready"])
        process["boot_s"] = round(time.monotonic() - started, 1)
        process["paired"] = paired
        if paired:
            client = Client(args.api_port, token)
            outcome = run_task_via_api(
                client, spec["text"], cfg["timeouts_s"]["task_terminal"], proc
            )
            _, events = client.call("GET", "/events?after=0")
            (folder / "phone_events.json").write_text(json.dumps(events), encoding="utf-8")
            _, tasks = client.call("GET", "/tasks")
            (folder / "phone_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    finally:
        process["exit_code"] = stop_process(
            proc, cfg["timeouts_s"]["process_exit_grace"], term=False
        )
        log.close()
    stdout = (folder / "driver_live_stdout.log").read_text(encoding="utf-8", errors="replace")
    process["start_refused"] = START_REFUSAL in stdout
    return outcome


def run_matrix(cfg: dict, args, out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    plan = build_cases(
        cfg,
        kitchens=args.kitchens.split(",") if args.kitchens else None,
        starts=[int(s) for s in args.starts.split(",")] if args.starts else None,
        arms=args.arms.split(",") if args.arms else None,
        tasks=args.tasks.split(",") if args.tasks else None,
        case_ids=set(args.case_ids.split(",")) if args.case_ids else None,
    )
    model_proc = model_log = None
    model_ok, model_error = True, None
    try:
        model_proc, model_log = start_model_server(cfg, out, args.qwen_port)
    except Exception as error:  # a model that cannot load is a failed case, never a substitution
        model_ok, model_error = False, repr(error)
    deadline = time.monotonic() + args.budget_s
    processes, cases = [], []
    try:
        for spec in plan:
            if deadline - time.monotonic() < cfg["timeouts_s"]["min_remaining_to_start_process_s"]:
                processes.append({"case_id": spec["case_id"], "deferred_time_budget": True})
                cases.append({**spec, "deferred": True})
                continue
            result = run_case(cfg, spec, out / spec["case_id"], args, model_ok, model_error)
            processes.append(result["process"])
            cases.append(result["case"])
            (out / "progress.json").write_text(
                json.dumps({"processes": processes, "cases": cases}, indent=2, default=str),
                encoding="utf-8",
            )
    finally:
        stop_process(model_proc, 60, term=True)
        if model_log is not None:
            model_log.close()
    summary = {
        "config": cfg["name"],
        "status": cfg["status"],
        "model_server": {"ok": model_ok, "error": model_error},
        "planned": [s["case_id"] for s in plan],
        "processes": processes,
        "gates": evaluate_gates(cfg, cases),
        "reports": report_tables(cases),
        "cases": cases,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def merge_summaries(cfg: dict, summaries: list[dict]) -> dict:
    """Combine per-job summaries: a case run in several jobs keeps its latest non-deferred
    record. The plan is the full matrix, so cases nobody ran are listed as deferred."""
    latest: dict[str, dict] = {}
    for summary in summaries:
        for case in summary["cases"]:
            if case.get("deferred") and case["case_id"] in latest:
                continue
            latest[case["case_id"]] = case
    plan = build_cases(cfg)
    order = {s["case_id"]: i for i, s in enumerate(plan)}
    for spec in plan:
        latest.setdefault(spec["case_id"], {**spec, "deferred": True})
    cases = sorted(latest.values(), key=lambda c: order.get(c["case_id"], 9999))
    return {
        "config": cfg["name"],
        "gates": evaluate_gates(cfg, cases),
        "reports": report_tables(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--allow-draft", action="store_true", help="smoke/development runs only")
    parser.add_argument("--kitchens")
    parser.add_argument("--starts")
    parser.add_argument("--arms")
    parser.add_argument("--tasks")
    parser.add_argument("--case-ids")
    parser.add_argument("--robot-dir", type=pathlib.Path, default=pathlib.Path("/work/g1_usd"))
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--qwen-port", type=int, default=8767)
    parser.add_argument("--budget-s", type=float, default=9000.0, help="job time budget")
    parser.add_argument("--process-duration-s", type=float, default=3600.0)
    args = parser.parse_args()
    cfg = load_config(args.config, args.allow_draft)
    if args.merge:
        merged = merge_summaries(cfg, [json.loads(p.read_text("utf-8")) for p in args.merge])
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "merged_summary.json").write_text(json.dumps(merged, indent=2), "utf-8")
        print(json.dumps({"gates": merged["gates"]}, indent=2))
        return
    summary = run_matrix(cfg, args, args.out)
    print(json.dumps({"gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
