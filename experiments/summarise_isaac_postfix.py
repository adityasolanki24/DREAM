"""Combine the two parts of the post-fix Qwen matrix and compare it with the frozen run.

    python experiments/summarise_isaac_postfix.py

Reads results/isaac_qwen_postfix_closeout/{postfix,postfix_rest}/summary.json, applies the frozen
gates (experiments.isaac_g1_live_eval.evaluate_gates) to the 15 ran cases and compares each case
with results/isaac_qwen_closeout_20261107. Nothing is rescored and the frozen result is not
replaced. Contact counts come from the physical_safety dimension each case recorded.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.isaac_g1_live_eval import evaluate_gates, load_config  # noqa: E402


def row(case: dict) -> dict:
    safety = (case.get("dimensions") or {}).get("physical_safety") or {}
    truth = (case.get("dimensions") or {}).get("independent_task_success") or {}
    return {
        "goal": case.get("goal"),
        "scheduler_state": case.get("scheduler_state"),
        "success_claimed": case.get("success_claimed"),
        "independent_success": truth.get("success"),
        "result": case.get("scheduler_result"),
        "contact_events": safety.get("contact_events"),
        "contacted_bodies": sorted(safety.get("contacted_bodies") or {}),
        "max_contact_impulse_ns": safety.get("max_contact_impulse_ns"),
        "scorer_safe": safety.get("safe"),
        "fell": case.get("fell"),
    }


def main() -> None:
    results = ROOT / "results"
    out = results / "isaac_qwen_postfix_closeout"
    first = json.loads((out / "postfix/summary.json").read_text(encoding="utf-8"))
    rest = json.loads((out / "postfix_rest/summary.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (results / "isaac_qwen_closeout_20261107/summary.json").read_text(encoding="utf-8")
    )
    cases = [c for c in first["cases"] if not c.get("deferred")] + rest["cases"]
    if len(cases) != 15 or len({c["case_id"] for c in cases}) != 15:
        raise ValueError("expected the 15 predeclared cases exactly once")
    gates = evaluate_gates(load_config(), cases)
    base = {c["case_id"]: row(c) for c in frozen["cases"]}
    per_case = {c["case_id"]: {"frozen": base[c["case_id"]], "postfix": row(c)} for c in cases}
    changed = sorted(
        k
        for k, v in per_case.items()
        if v["frozen"]["success_claimed"] != v["postfix"]["success_claimed"]
        or v["frozen"]["independent_success"] != v["postfix"]["independent_success"]
    )
    summary = {
        "label": "post-fix development check; development kitchen, not held-out; does not replace "
        "results/isaac_qwen_closeout_20261107",
        "frozen_config": "experiments/configs/isaac_g1_qwen_closeout.json",
        "code": "aba05d1 (postfix), e881d0a (postfix_rest; adds only result files)",
        "gates": gates,
        "frozen_gates": frozen["gates"],
        "outcome_changed_from_frozen": changed,
        "contact_cases": {
            "frozen": sorted(k for k, v in per_case.items() if v["frozen"]["contact_events"]),
            "postfix": sorted(k for k, v in per_case.items() if v["postfix"]["contact_events"]),
        },
        "per_case": per_case,
        "disclosures": [
            "Run in two sessions. The first used --budget-s 4800, too short for the matrix, so the "
            "driver deferred 6 cases; they ran unchanged in a second session with --groups. This "
            "was an operator error, not a result-driven rerun: no case was repeated.",
            "A case's goal can differ between runs because Qwen chose it; a difference is not "
            "attributed to the fix by this comparison. One run per case.",
        ],
    }
    text = json.dumps(summary, indent=2, default=float)
    (out / "summary.json").write_text(text, encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("gates", "outcome_changed_from_frozen",
                                              "contact_cases")}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
