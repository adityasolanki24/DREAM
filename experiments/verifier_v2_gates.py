"""Evaluate the pre-registered gates of experiments/configs/semantic_search_verifier_v2.json.

    python experiments/verifier_v2_gates.py results/semantic_search_verifier_v2/V1 \
        [--config CONFIG] [--out gates.json]

Read-only analysis of episodes.jsonl. Counts are over valid episodes (a crash or a scene
failure is reported and never counted as a pass).
- V1: the candidate arm makes no false success claim; with 0 claims, the one-sided 95% upper
  bound on the rate is reported (summarise_semantic_search.zero_upper).
- V2: candidate successes on target-present homes >= reference successes - margin, paired on
  identical homes (summarise_semantic_search.paired).
- V3 (descriptive, no threshold): shadow acceptances per purity rule and arm, split into
  those on a decoy (the eval-side nearest true object is not the target class, and the
  subset within the claim tolerance) and on the target; the reference arm's false claims;
  the paired time to verified find and McNemar test.
The default verifier rule is not changed by this script or by its outcome.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.summarise_semantic_search import (  # noqa: E402
    arm_table,
    load,
    paired,
    zero_upper,
)

CLAIM_TOLERANCE = 0.75  # m, dream/eval/semantic_scoring.CLAIM_TOLERANCE
RULES = ("point", "wilson_lower")


def shadow_summary(rows: list[dict], arm: str) -> dict:
    out = {}
    valid = [r for r in rows if r["arm_name"] == arm and "shadow_acceptances" in r]
    for rule in RULES:
        acc = [(r, a) for r in valid for a in r["shadow_acceptances"].get(rule, [])]
        decoy = [(r, a) for r, a in acc if a["nearest_cls"] != r["target"]]
        near = [(r, a) for r, a in decoy if a["nearest_m"] <= CLAIM_TOLERANCE]
        out[rule] = {
            "acceptances": len(acc),
            "on_target": len(acc) - len(decoy),
            "decoy": len(decoy),
            "decoy_within_tolerance": len(near),
            "episodes_with_decoy": len({id(r) for r, _ in decoy}),
            "episodes_with_decoy_within_tolerance": len({id(r) for r, _ in near}),
            "decoy_by_nearest_class": dict(Counter(a["nearest_cls"] for _, a in decoy)),
            "decoy_episodes": [
                {
                    "scenario": r["scenario"], "target": r["target"], "generator": r["generator"],
                    "seed": r["seed"], "track": a["track"], "sim_s": a["sim_s"],
                    "nearest_cls": a["nearest_cls"], "nearest_m": a["nearest_m"],
                }
                for r, a in near
            ],
        }  # fmt: skip
    return {"episodes": len(valid), **out}


def evaluate(rows: list[dict], candidate: str, reference: str, budget: float, margin: int) -> dict:
    table = arm_table(rows)
    cand = [r for r in rows if r["arm_name"] == candidate]
    valid = [r for r in cand if r.get("result") not in ("crash", "scene_failure")]
    false_c = [r for r in valid if r["result"] == "false_success"]
    pair = paired(rows, candidate, reference, budget)
    n_pairs = pair.get("pairs", 0)
    s_c, s_r = pair.get("success", [0, 0])
    v1 = {
        "gate": f"{candidate} makes 0 false success claims",
        "false_success_claims": len(false_c),
        "valid_episodes": len(valid),
        "zero_upper_95": zero_upper(len(valid)) if not false_c else None,
        "false_claim_episodes": [
            {k: r.get(k) for k in ("scenario", "target", "generator", "seed", "sim_s")}
            for r in false_c
        ],
        "crashes": table[candidate]["crashes"],
        "scene_failures": table[candidate]["scene_failures"],
        "passed": len(false_c) == 0,
    }
    v2 = {
        "gate": f"{candidate} target-present successes >= {reference} - {margin}, paired",
        "pairs": n_pairs,
        "successes": {candidate: s_c, reference: s_r},
        "threshold": s_r - margin,
        "passed": bool(n_pairs > 0 and s_c >= s_r - margin),
        "pairs_expected": 96,
    }
    ref_valid = [r for r in rows if r["arm_name"] == reference]
    v3 = {
        "descriptive": True,
        "shadow_acceptances": {
            candidate: shadow_summary(rows, candidate),
            reference: shadow_summary(rows, reference),
        },
        "reference_false_claims": [
            {k: r.get(k) for k in ("scenario", "target", "generator", "seed", "sim_s")}
            | {"claim_nearest": r.get("claim_nearest")}
            for r in ref_valid
            if r.get("result") == "false_success"
        ],
        "paired": pair,
        "arms": {
            a: {
                "success_target_present": table[a]["success_target_present"],
                "false_success_claims": table[a]["false_success_claims"][:2],
                "correct_refusal_absent": table[a]["correct_refusal_absent"],
                "median_success_sim_s": table[a]["median_success_sim_s"],
                "mean_path_m": table[a]["navigation"]["mean_path_m"],
                "crashes": table[a]["crashes"],
            }
            for a in (reference, candidate)
        },
    }
    decision = v1["passed"] and v2["passed"]
    return {
        "V1": v1, "V2": v2, "V3": v3,
        "decision": {
            "rule": "wilson_lower becomes the default only if V1 and V2 both pass",
            "V1_passed": v1["passed"], "V2_passed": v2["passed"],
            "criteria_met": decision,
            "note": "reported only; the default is changed by the main session, not by this script",
        },
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/"
                                            "semantic_search_verifier_v2.json"))  # fmt: skip
    ap.add_argument("--candidate", default="proposed_v2")
    ap.add_argument("--reference", default="proposed")
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    rows = load([args.dir])
    result = {
        "config": cfg["name"], "frozen_commit": cfg.get("frozen_commit"), "episodes": len(rows),
        **evaluate(rows, args.candidate, args.reference, cfg["time_budget_s"], args.margin),
    }  # fmt: skip
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
