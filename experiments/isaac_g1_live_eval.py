"""Unattended driver for the frozen Qwen closeout evaluation of the live Isaac G1 session.

Runs on the cloud VM as a job. For every (start, process group) of the frozen config it starts
a fresh ``experiments/isaac_g1_live.py`` process, pairs with it over localhost through the
one-time ``/_bootstrap`` route using a token generated here (held in memory, never printed or
written), submits each task through the real phone API with provider ``qwen``, waits for the
terminal state and keeps every output. The model server is started once and reused by the
live processes. There is no scripted controller and no mock inference: if the model cannot
load, each case is recorded as a failure.

    python experiments/isaac_g1_live_eval.py --out <dir> [--groups s7_T1_T2,s8_T3 ...]

The scoring itself is dream/eval/task_scoring.py, run inside the live process; this module
only aggregates what it wrote into per-case records and the frozen gates.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "experiments/configs/isaac_g1_qwen_closeout.json"
TERMINAL = ("succeeded", "failed", "cancelled")
START_REFUSAL = "no free start cell at the configured clearance"
DIMENSIONS = (
    "model_decision_valid",
    "navigation_success",
    "identity_correct",
    "viewpoint_correct",
    "handle_visible",
    "physical_safety",
    "independent_task_success",
    "recovery_effectiveness",
)


def load_config(path: pathlib.Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("status") != "frozen":
        raise ValueError("the closeout config must be frozen before any run")
    return cfg


def build_groups(cfg: dict, starts: list[int] | None = None) -> list[dict]:
    """The process plan: for each start, one group per entry of ``process_groups``."""
    groups = []
    for start in starts or cfg["starts"]:
        for tasks in cfg["process_groups"]:
            groups.append({"id": group_id(start, tasks), "start": start, "tasks": list(tasks)})
    return groups


def group_id(start: int, tasks: list[str]) -> str:
    return f"s{start}_" + "_".join(tasks)


def pick_candidate(objects: list[dict]) -> int | None:
    """Frozen T2 rule: the lowest candidate id whose handle has been observed."""
    ids = [o["id"] for o in objects if isinstance(o, dict) and o.get("handle") is not None]
    return min(ids) if ids else None


# -- phone API client (localhost, bearer token held in memory only) ---------------------------


class Client:
    def __init__(self, port: int, token: str, host: str = "127.0.0.1") -> None:
        self.base, self._token = f"http://{host}:{port}", token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                return error.code, json.loads(raw or b"null")
            except ValueError:
                return error.code, {"error": raw[:200].decode("utf-8", "replace")}


def pair(port: int, token: str, host: str = "127.0.0.1", timeout: float = 10.0):
    """POST the one-time pairing token; the response never contains it."""
    request = urllib.request.Request(
        f"http://{host}:{port}/_bootstrap",
        data=json.dumps({"token": token}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def wait_paired(port, token, proc, timeout_s, sleep=time.sleep, clock=time.monotonic) -> bool:
    """Retry pairing until the live process serves its API; False on timeout or early exit."""
    deadline = clock() + timeout_s
    while clock() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            status, body = pair(port, token)
            if status == 200 and body == {"ready": True}:
                return True
            if status != 200:
                return False
        except (OSError, ValueError):
            pass
        sleep(5)
    return False


def run_task_via_api(client: Client, text: str, timeout_s: float, proc=None, sleep=time.sleep):
    """Submit one task as the phone does and wait for a terminal state. Returns a dict with
    ``submitted``, ``task_id``, ``row`` (final scheduler row), ``timed_out``, ``refusal``."""
    status, body = client.call("POST", "/tasks", {"text": text, "provider": "qwen"})
    if status != 201:
        return {"submitted": False, "refusal": {"http_status": status, "body": body}}
    task_id = body["id"]
    deadline = time.monotonic() + timeout_s
    row = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return {
                "submitted": True,
                "task_id": task_id,
                "row": row,
                "timed_out": False,
                "process_died": True,
            }
        try:
            _, tasks = client.call("GET", "/tasks")
            row = next((t for t in tasks if t["id"] == task_id), row)
        except (OSError, ValueError):
            pass
        if row is not None and row["state"] in TERMINAL:
            return {"submitted": True, "task_id": task_id, "row": row, "timed_out": False}
        sleep(3)
    try:
        client.call("POST", f"/tasks/{task_id}/cancel", {})
    except (OSError, ValueError):
        pass
    return {"submitted": True, "task_id": task_id, "row": row, "timed_out": True}


# -- per-case record and the frozen gates (pure) -----------------------------------------------


def _recovery(episode: dict | None, iv: dict | None) -> dict:
    if not episode or not iv:
        return {"effective": None, "reason": "no evaluation was produced"}
    failed = int(episode.get("failed_actions") or 0)
    repairs = int(iv.get("repair_count") or 0)
    needed = failed > 0 or repairs > 0
    success = (iv.get("independent_task_success") or {}).get("success")
    return {
        "needed": needed,
        "failed_actions": failed,
        "repair_count": repairs,
        "repair_changed_outcome": (iv.get("repair_changed_outcome") or {}).get("changed"),
        "effective": (success if needed else None),
        "reason": None if needed else "no failed action or repair occurred",
    }


def build_case(spec: dict, outcome: dict, evaluation: dict | None, physical: dict | None) -> dict:
    """One case's record: raw scorer fields copied verbatim and the eight reported dimensions.

    ``outcome`` is run_task_via_api's dict (or a not-submitted marker); ``evaluation`` is the
    live process's task_<id>_evaluation.json; ``physical`` its physical_outcome.json."""
    iv = None if evaluation is None else evaluation.get("independent_verification")
    episode = None if evaluation is None else evaluation.get("episode")
    row = outcome.get("row") or {}
    state = row.get("state")
    if iv is not None:
        claimed = bool(iv["success_claimed"])
        false_claim = bool(iv["false_claim"])
        claim_source = "evaluation"
    else:
        claimed = state == "succeeded"
        false_claim = claimed  # nothing independent was produced to confirm it
        claim_source = "scheduler_state"
    safety = (iv or {}).get("physical_safety") or {}
    final_height = (physical or {}).get("final_height_m_for_evaluation_only")
    fell = safety.get("fell")
    if fell is None and final_height is not None:
        fell = bool(final_height < 0.45)
    dims = {
        "model_decision_valid": (iv or {}).get("model_decision_valid"),
        "navigation_success": (iv or {}).get("navigation_success"),
        "identity_correct": (iv or {}).get("identity_correct"),
        "viewpoint_correct": (iv or {}).get("viewpoint_correct"),
        "handle_visible": (iv or {}).get("handle_visible"),
        "physical_safety": safety or None,
        "independent_task_success": (iv or {}).get("independent_task_success"),
        "recovery_effectiveness": _recovery(episode, iv),
    }
    result = row.get("result") or ""
    return {
        **spec,
        "submitted": bool(outcome.get("submitted")),
        "refusal": outcome.get("refusal"),
        "task_id": outcome.get("task_id"),
        "scheduler_state": state,
        "scheduler_result": row.get("result"),
        "timed_out": bool(outcome.get("timed_out")),
        "process_died": bool(outcome.get("process_died")),
        "goal": None if episode is None else episode.get("goal"),
        "goal_kind": (iv or {}).get("goal_kind"),
        "success_claimed": claimed,
        "claim_source": claim_source,
        "false_claim": false_claim,
        "fell": fell,
        "protective_stop": safety.get("safety_stop") or (iv or {}).get("terminal_stop_reason"),
        "evaluation_present": evaluation is not None,
        "suspected_integration_error": evaluation is None and result.startswith("execution error"),
        "dimensions": dims,
        "independent_verification": iv,
        "agent_episode": episode,
    }


def evaluate_gates(cfg: dict, cases: list[dict]) -> dict:
    """The frozen gates over every recorded case. A gate with missing cases is INCOMPLETE."""
    g, want = cfg["gates"], cfg["gates"]["cases_expected_per_task"]
    ran = [c for c in cases if not c.get("deferred")]

    def of(task):
        return [c for c in ran if c["task"] == task]

    def complete(task):
        return len(of(task)) >= want

    def row(passed, **data):
        return {"result": "PASS" if passed else "FAIL", **data}

    gates = {}
    # T3 and T4 have no ground truth: a success claim there is a false claim by the frozen
    # decision, even where the scorer's own task-typed check (e.g. arrival at a remembered
    # place the model substituted for the refrigerator) reports no false claim.
    no_truth = {"T3", "T4"}
    claims = [
        c for c in ran if c["false_claim"] or (c["task"] in no_truth and c["success_claimed"])
    ]
    gates["false_claims_zero"] = (
        row(len(claims) <= g["false_claims_max"], false_claims=len(claims), cases=len(ran))
        if len(ran) == len(cases)
        else {"result": "INCOMPLETE", "cases_run": len(ran), "cases_planned": len(cases)}
    )
    unknown = [c["case_id"] for c in ran if c["fell"] is None]
    falls = [c["case_id"] for c in ran if c["fell"]]
    stops = [c["case_id"] for c in ran if c["protective_stop"]]
    gates["falls_zero"] = {
        **row(len(falls) <= g["falls_max"] and not unknown, falls=falls, unknown_safety=unknown),
        "protective_stops_recorded": stops,
    }
    for task, key in (
        ("T4", "T4_no_success_claim_min"),
        ("T3", "T3_no_verified_success_claim_min"),
    ):
        n = sum(1 for c in of(task) if not c["success_claimed"])
        gates[f"{task}_no_success_claim"] = (
            row(n >= g[key], count=n, of=len(of(task)))
            if complete(task)
            else {"result": "INCOMPLETE", "count": n, "of": len(of(task)), "expected": want}
        )
    for task, key in (("T1", "T1_independent_success_min"), ("T2", "T2_independent_success_min")):
        n = sum(
            1
            for c in of(task)
            if ((c["dimensions"]["independent_task_success"] or {}).get("success")) is True
        )
        gates[f"{task}_independent_success"] = (
            row(n >= g[key], count=n, of=len(of(task)))
            if complete(task)
            else {"result": "INCOMPLETE", "count": n, "of": len(of(task)), "expected": want}
        )
    checkable = [
        c
        for c in of("T5")
        if (c["dimensions"]["independent_task_success"] or {}).get("success") is not None
    ]
    wins = sum(1 for c in checkable if c["dimensions"]["independent_task_success"]["success"])
    if not complete("T5"):
        gates["T5_success"] = {"result": "INCOMPLETE", "of": len(of("T5")), "expected": want}
    elif not checkable and g["T5_unverifiable_if_no_case_has_an_independent_check"]:
        gates["T5_success"] = {
            "result": "UNVERIFIABLE",
            "reason": "no T5 case has an independent check (the scorer returns no ground truth)",
            "goal_kinds": [c["goal_kind"] for c in of("T5")],
        }
    else:
        gates["T5_success"] = row(
            wins / len(checkable) >= g["T5_success_fraction_min_where_checkable"],
            successes=wins,
            checkable=len(checkable),
            of=len(of("T5")),
            goal_kinds=[c["goal_kind"] for c in of("T5")],
        )
    return gates


def dimension_table(cases: list[dict]) -> list[dict]:
    """Compact per-case, per-dimension values (the verbatim scorer fields stay in the cases)."""

    def short(name, value):
        if value is None:
            return None
        if name == "model_decision_valid":
            return value.get("valid")
        if name == "navigation_success":
            return value.get("success")
        if name == "identity_correct":
            return value.get("correct")
        if name == "viewpoint_correct":
            return value.get("correct")
        if name == "handle_visible":
            return value.get("visible")
        if name == "physical_safety":
            return value.get("safe")
        if name == "independent_task_success":
            return value.get("success")
        return value.get("effective")

    return [
        {
            "case": c["case_id"],
            "start": c["start"],
            "task": c["task"],
            "text": c["text"],
            "goal_kind": c.get("goal_kind"),
            "scheduler_state": c.get("scheduler_state"),
            "success_claimed": c["success_claimed"],
            "false_claim": c["false_claim"],
            **{d: short(d, (c["dimensions"] or {}).get(d)) for d in DIMENSIONS},
        }
        for c in cases
    ]


# -- process orchestration ---------------------------------------------------------------------


def start_model_server(cfg: dict, out: pathlib.Path, port: int):
    """Start the pinned Qwen server once; returns (process, log) or (None, None) if reusable."""
    from experiments.isaac_g1_live import QWEN_REVISION

    if cfg["model_revision"] != QWEN_REVISION:
        raise RuntimeError("config model revision differs from isaac_g1_live.QWEN_REVISION")
    health_url = f"http://127.0.0.1:{port}/healthz"

    def healthy() -> bool:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                health = json.loads(response.read())
            return bool(health.get("ready") and health.get("revision") == QWEN_REVISION)
        except (OSError, ValueError):
            return False

    if healthy():
        return None, None
    log = (out / "qwen_server.log").open("w", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    command = [
        os.environ.get("DREAM_QWEN_PYTHON", "/opt/dream-runtime/bin/python"),
        "-m",
        "experiments.qwen_tool_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--revision",
        QWEN_REVISION,
    ]
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + cfg["timeouts_s"]["model_start"]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError(f"Qwen server exited during load; inspect {out / 'qwen_server.log'}")
        if healthy():
            return proc, log
        time.sleep(3)
    proc.terminate()
    raise TimeoutError("Qwen3-VL did not become ready before the frozen deadline")


def stop_process(proc, grace_s: float, *, term: bool) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None and term:
        proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=60)


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_group(cfg, group, start, folder, args, model_ok: bool, model_error: str | None) -> dict:
    """_run_group, with a driver exception recorded as failed cases instead of ending the matrix."""
    try:
        return _run_group(cfg, group, start, folder, args, model_ok, model_error)
    except Exception as error:
        failure = f"driver_exception: {error!r}"
        specs = [
            {"case_id": f"s{group['start']}_{t}", "group": group["id"], "start": group["start"],
             "task": t, "text": cfg["tasks"][t].get("text")}
            for t in group["tasks"]
        ]  # fmt: skip
        return {
            "process": {"group": group["id"], "start": start, "driver_failure": failure},
            "cases": [
                build_case(s, {"submitted": False, "refusal": failure}, None, None) for s in specs
            ],
        }


def _run_group(cfg, group, start, folder, args, model_ok: bool, model_error: str | None) -> dict:
    """One fresh live process and its tasks. Returns case dicts and process facts."""
    folder.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "case_id": f"s{group['start']}_{t}",
            "group": group["id"],
            "start": group["start"],
            "effective_start": start,
            "task": t,
            "text": cfg["tasks"][t].get("text"),
        }
        for t in group["tasks"]
    ]
    process = {"group": group["id"], "start": start, "folder": folder.name}
    if not model_ok:
        process["setup_failure"] = f"qwen_unavailable: {model_error}"
        return {
            "process": process,
            "cases": [
                build_case(s, {"submitted": False, "refusal": process["setup_failure"]}, None, None)
                for s in specs
            ],
        }
    token = secrets.token_urlsafe(32)  # in memory only
    single = len(group["tasks"]) == 1
    command = [
        sys.executable,
        str(ROOT / "experiments/isaac_g1_live.py"),
        "--out",
        str(folder / "live"),
        "--kitchen",
        cfg["kitchen"],
        "--start",
        str(start),
        "--robot-dir",
        str(args.robot_dir),
        "--api-port",
        str(args.api_port),
        "--qwen-port",
        str(args.qwen_port),
        "--model-start-timeout-s",
        str(cfg["timeouts_s"]["model_start"]),
        "--duration-s",
        str(int(args.process_duration_s)),
    ]
    if single:
        command.append("--single-task")
    log = (folder / "driver_live_stdout.log").open("w", encoding="utf-8")
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    process["command_tail"] = command[1:]
    cases: list[dict] = []
    outcomes: list[tuple[dict, dict]] = []
    try:
        paired = wait_paired(args.api_port, token, proc, cfg["timeouts_s"]["api_ready"])
        process["boot_s"] = round(time.monotonic() - started, 1)
        process["paired"] = paired
        if not paired:
            outcome = {"submitted": False, "refusal": "live process never became ready"}
            outcomes = [(s, outcome) for s in specs]
        else:
            client = Client(args.api_port, token)
            for spec in specs:
                spec = dict(spec)
                if spec["task"] == "T2":
                    _, snapshot = client.call("GET", "/snapshot")
                    (folder / "snapshot_before_T2.json").write_text(
                        json.dumps(snapshot, indent=2), encoding="utf-8"
                    )
                    cid = pick_candidate((snapshot or {}).get("objects") or [])
                    spec["candidate"] = cid
                    if cid is None:
                        outcomes.append(
                            (spec, {"submitted": False, "refusal": "no_candidate_with_handle"})
                        )
                        continue
                    spec["text"] = cfg["tasks"]["T2"]["text_template"].format(candidate=cid)
                outcome = run_task_via_api(
                    client, spec["text"], cfg["timeouts_s"]["task_terminal"], proc
                )
                outcomes.append((spec, outcome))
                if outcome.get("timed_out") or outcome.get("process_died"):
                    break
            for spec in specs[len(outcomes) :]:
                outcomes.append((spec, {"submitted": False, "refusal": "earlier task did not end"}))
            _, events = client.call("GET", "/events?after=0")
            (folder / "phone_events.json").write_text(json.dumps(events), encoding="utf-8")
            _, final_tasks = client.call("GET", "/tasks")
            (folder / "phone_tasks.json").write_text(json.dumps(final_tasks), encoding="utf-8")
    finally:
        code = stop_process(proc, cfg["timeouts_s"]["process_exit_grace"], term=not single)
        log.close()
        process["exit_code"] = code
        process["wall_s"] = round(time.monotonic() - started, 1)
    stdout = (folder / "driver_live_stdout.log").read_text(encoding="utf-8", errors="replace")
    process["start_refused"] = START_REFUSAL in stdout
    physical = read_json(folder / "live" / "physical_outcome.json")
    for spec, outcome in outcomes:
        evaluation = (
            read_json(folder / "live" / f"task_{outcome['task_id']}_evaluation.json")
            if outcome.get("task_id")
            else None
        )
        cases.append(build_case(spec, outcome, evaluation, physical))
    return {"process": process, "cases": cases}


def run_matrix(cfg, args, out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    groups = build_groups(cfg)
    if args.groups:
        wanted = set(args.groups.split(","))
        unknown = wanted - {g["id"] for g in groups}
        if unknown:
            raise ValueError(f"unknown groups: {sorted(unknown)}")
        groups = [g for g in groups if g["id"] in wanted]
    model_proc = model_log = None
    model_ok, model_error = True, None
    try:
        model_proc, model_log = start_model_server(cfg, out, args.qwen_port)
    except Exception as error:  # a model that cannot load is a failed case, never a substitution
        model_ok, model_error = False, repr(error)
    deadline = time.monotonic() + args.budget_s
    replacements = list(cfg["replacement_starts"])
    replaced: dict[int, int] = {}
    processes, cases = [], []
    try:
        for group in groups:
            if deadline - time.monotonic() < cfg["timeouts_s"]["min_remaining_to_start_process_s"]:
                processes.append({"group": group["id"], "deferred_time_budget": True})
                for t in group["tasks"]:
                    cases.append(
                        {
                            "case_id": f"s{group['start']}_{t}",
                            "group": group["id"],
                            "start": group["start"],
                            "task": t,
                            "text": cfg["tasks"][t].get("text"),
                            "deferred": True,
                        }
                    )
                continue
            start = replaced.get(group["start"], group["start"])
            result = run_group(cfg, group, start, out / group["id"], args, model_ok, model_error)
            proc = result["process"]
            if proc.get("start_refused") and not proc.get("paired") and replacements:
                replaced[group["start"]] = replacements.pop(0)
                proc["start_replaced_by"] = replaced[group["start"]]
                processes.append(proc)
                result = run_group(
                    cfg,
                    group,
                    replaced[group["start"]],
                    out / (group["id"] + "_replacement"),
                    args,
                    model_ok,
                    model_error,
                )
                proc = result["process"]
            processes.append(proc)
            cases.extend(result["cases"])
            (out / "progress.json").write_text(
                json.dumps({"processes": processes, "cases": cases}, indent=2, default=str),
                encoding="utf-8",
            )
    finally:
        stop_process(model_proc, 60, term=True)
        if model_log is not None:
            model_log.close()
    ran = [c for c in cases if not c.get("deferred")]
    summary = {
        "config": cfg["name"],
        "status": cfg["status"],
        "evidence_class": cfg["evidence_class"],
        "model_server": {"ok": model_ok, "error": model_error},
        "start_replacements": replaced,
        "processes": processes,
        "gates": evaluate_gates(cfg, cases) if not any(c.get("deferred") for c in cases) else None,
        "partial": any(c.get("deferred") for c in cases) or bool(args.groups),
        "table": dimension_table(ran),
        "cases": cases,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def merge_summaries(cfg: dict, summaries: list[dict]) -> dict:
    """Combine per-job summaries of one attempt (a later job may run groups an earlier one
    deferred). A group run in several jobs keeps its latest non-deferred record."""
    latest: dict[str, dict] = {}
    for summary in summaries:
        for case in summary["cases"]:
            if case.get("deferred") and case["case_id"] in latest:
                continue
            latest[case["case_id"]] = case
    order = {
        f"s{s}_{t}": i
        for i, (s, t) in enumerate((s, t) for s in cfg["starts"] for t in cfg["tasks"])
    }
    cases = sorted(latest.values(), key=lambda c: order.get(c["case_id"], 999))
    ran = [c for c in cases if not c.get("deferred")]
    return {
        "config": cfg["name"],
        "partial": len(cases) != len(order) or len(ran) != len(cases),
        "gates": evaluate_gates(cfg, cases) if len(cases) == len(order) else None,
        "table": dimension_table(ran),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", nargs="+", type=pathlib.Path, help="summary.json files to merge")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--groups", help="comma-separated group ids, e.g. s7_T1_T2,s8_T3")
    parser.add_argument("--robot-dir", type=pathlib.Path, default=pathlib.Path("/work/g1_usd"))
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--qwen-port", type=int, default=8767)
    parser.add_argument("--budget-s", type=float, default=9000.0, help="job time budget")
    parser.add_argument("--process-duration-s", type=float, default=3600.0)
    args = parser.parse_args()
    if args.merge:
        merged = merge_summaries(
            load_config(args.config), [json.loads(p.read_text("utf-8")) for p in args.merge]
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "merged_summary.json").write_text(json.dumps(merged, indent=2), "utf-8")
        print(json.dumps({"gates": merged["gates"], "partial": merged["partial"]}, indent=2))
        return
    summary = run_matrix(load_config(args.config), args, args.out)
    print(json.dumps({"gates": summary["gates"], "partial": summary["partial"]}, indent=2))


if __name__ == "__main__":
    main()
