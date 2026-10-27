"""Failure attribution of reach episodes, v1 against a v2 development rerun (W6 step 5).

Each agent episode is assigned ONE cause from its recorded verdict, reason and stop details:

    success                  physically successful (the set's own success definition)
    refused: planner         the planner produced no candidate (refusal before any rollout)
    refused: rollout         candidates existed but every rollout failed the gate
    stop: <reason>           a protective stop (torque / joint / support / tilt ...)
    perception/navigation    handle never found or lost, or the stance could not be reached
    no claim                 executed, but no certified contact (no tactile claim in v2; the
                             distance claim in v1)
    false claim              claimed, evaluator disagrees
    other

Reported over the episodes that are reference-feasible under the v1 label (the 41 of the W5
attribution when run on the v1 dev + held-out results) and, separately, over all episodes. v1 and
v2 episode ids are matched by scene: ``heldout_sNNNN`` is ``dev_v1ho_sNNNN`` in the v2 development
set.

    python experiments/g1_reach_attribute.py --v2 results/g1_reach_heights_v2_dev
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
V1_DIRS = ("g1_reach_heights_dev", "g1_reach_heights_heldout")


def cause(r: dict) -> str:
    """Heuristic labels from recorded evidence; unresolved cases need manual review."""
    a, e = r["agent"], r["evaluator"]
    reason = a.get("reason", "").lower()
    if e["false_success_claim"]:
        return "false claim"
    if e["physical_success"]:
        return "success" if a["verdict"] == "reached" else "physical success without claim"
    stops = r.get("stop_details") or e.get("protective_stops") or []
    if stops:
        first = stops[0]
        what = first.get("reason", "unknown")
        joint = first.get("joint_name")
        return f"stop: {what}" + (f" ({joint})" if joint else "")
    if a["verdict"] == "refused":
        if "rollout" in reason:
            return "refused: rollout"
        if any(word in reason for word in ("planner", "candidate", "plan")):
            return "refused: planner"
        return "unresolved refusal"
    if any(word in reason for word in ("not found", "lost", "could not walk")):
        return "perception/navigation"
    if any(word in reason for word in ("no claim", "not certain", "claim threshold")):
        return "no claim"
    return "unresolved"


def load_v1() -> dict[str, dict]:
    out = {}
    for d in V1_DIRS:
        for f in glob.glob(str(ROOT / "results" / d / "agent" / "*.json")):
            r = json.loads(pathlib.Path(f).read_text())
            if r.get("error") is None:
                out[r["spec"]["id"].replace("heldout_", "dev_v1ho_")] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True, help="v2 development results directory")
    ap.add_argument("--out")
    args = ap.parse_args()
    v1 = load_v1()
    v2 = {}
    for f in glob.glob(str(ROOT / args.v2 / "agent" / "*.json")):
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("error") is None:
            v2[r["spec"]["id"]] = r
    ids = sorted(set(v1) & set(v2))
    feas_v1 = [i for i in ids if v1[i]["evaluator"]["reference_feasible"]]
    table = {}
    for name, subset in (("reference_feasible_v1", feas_v1), ("all", ids)):
        c1 = collections.Counter(cause(v1[i]) for i in subset)
        c2 = collections.Counter(cause(v2[i]) for i in subset)
        joints = collections.Counter(
            (stop.get("joint_name"), stop.get("reason", "unknown"))
            for i in subset for stop in (v2[i].get("stop_details") or [])
        )
        table[name] = {
            "n": len(subset), "v1": dict(c1), "v2": dict(c2),
            "v2_stop_joints": {f"{j}|{why}": n for (j, why), n in joints.items()},
            "transitions": dict(collections.Counter(
                f"{cause(v1[i])} -> {cause(v2[i])}" for i in subset)),
        }  # fmt: skip
    table["v2_reference_label_i"] = {
        "feasible": sum(v2[i]["evaluator"]["reference_feasible"] for i in v2),
        "n": len(v2),
        "v2_success_on_label_i_feasible": sum(
            v2[i]["evaluator"]["physical_success"] and v2[i]["evaluator"]["reference_feasible"]
            for i in v2
        ),
    }  # fmt: skip
    truth = {}
    for f in (ROOT / args.v2 / "agent_truth_inputs").glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("error") is None and r.get("evaluator"):
            truth[r["spec"]["id"]] = r
    success = lambda r: r["evaluator"]["physical_success"]  # noqa: E731
    covered = {i for i, r in v2.items() if success(r) or i in truth}
    label_ii = {i for i, r in v2.items() if success(r) or (i in truth and success(truth[i]))}
    label_i = {i for i, r in v2.items() if r["evaluator"]["reference_feasible"]}
    table["reference_comparison"] = {
        "note": "Finite searches; missing truth inputs are unresolved, not infeasible.",
        "coverage_n": len(covered), "agent_n": len(v2),
        "uncovered": sorted(set(v2) - covered),
        "i_only_among_covered": sorted((label_i & covered) - label_ii),
        "ii_only": sorted(label_ii - label_i),
    }
    table["matching"] = {
        "v1_only": sorted(set(v1) - set(v2)), "v2_only": sorted(set(v2) - set(v1)),
        "note": "Cause categories are heuristics based on recorded evidence, not root causes.",
    }
    text = json.dumps(table, indent=1)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
