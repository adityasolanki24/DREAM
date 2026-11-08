"""Evaluate the gates pre-registered in experiments/configs/semantic_search_heldout.json.

    python experiments/semantic_search_gates.py results/semantic_search_heldout [--out FILE]

Reads each split's episodes.jsonl (H1_main, H1_ablations, H2, H3, H4; missing splits are
reported as not run) and applies the gate text of the frozen config literally. Read-only:
nothing here feeds back into the agent or the config. Crashes and scene failures never count
as passes; the fault-injection arms are excluded from G0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.summarise_semantic_search import paired, wilson, zero_upper  # noqa: E402

INVALID = ("crash", "scene_failure")
FAULT_ARMS = ("abl_substitution_gated", "abl_substitution_ungated")
TYPICAL = ["B", "C", "F", "G", "K"]


def read(root: pathlib.Path, split: str) -> list[dict] | None:
    p = root / split / "episodes.jsonl"
    if not p.exists():
        return None
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def valid(rows):
    return [r for r in rows if r.get("result") not in INVALID]


def count(rows, arm, result, scenarios=None):
    rs = [r for r in valid(rows) if r["arm_name"] == arm]
    if scenarios:
        rs = [r for r in rs if r["scenario"] in scenarios]
    return sum(r["result"] == result for r in rs), len(rs)


def invalid_counts(rows):
    out = defaultdict(int)
    for r in rows:
        if r.get("result") in INVALID:
            out[r["arm_name"]] += 1
    return dict(out)


def gates(root: pathlib.Path, budget: float) -> dict:
    splits = {s: read(root, s) for s in ("H1_main", "H1_ablations", "H2", "H3", "H4")}
    h1 = (splits["H1_main"] or []) + (splits["H1_ablations"] or [])
    out: dict = {"splits_run": {k: (len(v) if v is not None else None) for k, v in splits.items()}}
    out["invalid_episodes"] = {k: invalid_counts(v) for k, v in splits.items() if v}

    g0 = {}
    for name, rows in (("H1", h1), ("H2", splits["H2"]), ("H3", splits["H3"]),
                       ("H4", splits["H4"])):  # fmt: skip
        if not rows:
            g0[name] = "not run"
            continue
        arms = sorted({r["arm_name"] for r in rows} - set(FAULT_ARMS))
        g0[name] = {a: count(rows, a, "false_success") for a in arms}
    run = [v for v in g0.values() if v != "not run"]
    out["G0_no_false_success"] = {
        "per_split": g0,
        "proposed_upper95": {
            k: zero_upper(v["proposed"][1])
            for k, v in g0.items()
            if v != "not run" and v["proposed"][0] == 0
        },  # fmt: skip
        "pass": all(v["proposed"][0] == 0 for v in run) if run else None,
        "complete": len(run) == 4,
    }

    lrows = [r for r in valid(h1) if r["arm_name"] == "proposed" and r["scenario"] == "L"]
    ok = [r["completion"] == "goal_unsupported" and r["path_m"] == 0.0 and r["tool_calls"] == 0
          for r in lrows]  # fmt: skip
    out["G1_unsupported_refused"] = {"refused_without_motion": [sum(ok), len(ok)],
                                     "pass": bool(lrows) and all(ok)}  # fmt: skip

    present = [r for r in valid(h1) if r["arm_name"] == "proposed" and r.get("target_exists")
               and r["scenario"] != "L"]  # fmt: skip
    fa = sum(r["result"] == "false_absence" for r in present)
    out["G2_false_absence"] = {
        "false_absence": [fa, len(present), wilson(fa, len(present))],
        "pass": bool(present) and fa <= 0.05 * len(present),
    }

    main = splits["H1_main"] or []
    p = paired(main, "proposed", "A_gain_frontier", budget, TYPICAL)
    improved = (p.get("wilcoxon_p") is not None and p["wilcoxon_p"] < 0.05
                and p["median_time_diff_s"] < 0)  # fmt: skip
    out["G3_primary_efficiency"] = {
        "paired": p,
        "verdict": "improvement" if improved else "no demonstrated improvement",
    }

    sp, n_p = count(main, "proposed", "success", ["D", "J"])
    sa, n_a = count(main, "A_gain_frontier", "success", ["D", "J"])
    out["G4_atypical_non_inferiority"] = {
        "proposed": [sp, n_p], "A_gain_frontier": [sa, n_a],
        "paired": paired(main, "proposed", "A_gain_frontier", budget, ["D", "J"]),
        "pass": (sp >= sa - 2) if n_p and n_a else None,
    }  # fmt: skip

    abl = splits["H1_ablations"] or []
    ung = count(abl, "abl_substitution_ungated", "false_success")
    gat = count(abl, "abl_substitution_gated", "false_success")
    out["G5_fault_injection_validity"] = {
        "ungated_false_success": ung, "gated_false_success": gat,
        "gated_rejections": sum("not accepted" in (r.get("reason") or "") for r in abl
                                if r["arm_name"] == "abl_substitution_gated"),
        "pass": (ung[0] >= 1 and gat[0] == 0) if ung[1] and gat[1] else None,
    }  # fmt: skip

    g6 = {}
    for name, rows in (("H1", h1), ("H3", splits["H3"])):
        if rows:
            sv = sorted({r["scenario"] for r in rows if r["arm_name"] == "abl_single_view_verify"})
            g6[name] = {
                "scenarios": sv,
                "abl_single_view_verify": count(rows, "abl_single_view_verify", "false_success"),
                "proposed_same_scenarios": count(rows, "proposed", "false_success", sv),
            }  # fmt: skip
    out["G6_verification_ablation"] = g6 or "not run"

    h2 = splits["H2"]
    if h2:
        per = {}
        for t in sorted({r["target"] for r in h2}):
            rs = [r for r in h2 if r["target"] == t]
            per[t] = {a: count(rs, a, "success", ["B", "D"]) for a in
                      sorted({r["arm_name"] for r in rs})}  # fmt: skip
        out["G7_generalisation"] = {
            "proposed_false_success": count(h2, "proposed", "false_success"),
            "success_target_present_by_class": per,
            "pass": count(h2, "proposed", "false_success")[0] == 0,
        }
    else:
        out["G7_generalisation"] = "not run"

    h4 = splits["H4"]
    if h4:
        falls = {a: sum(bool(r.get("fell")) for r in h4 if r["arm_name"] == a)
                 for a in sorted({r["arm_name"] for r in h4})}  # fmt: skip
        out["G8_embodiment"] = {
            "proposed_false_success": count(h4, "proposed", "false_success"),
            "falls": falls,
            "success_target_present": {
                a: [
                    sum(
                        r["result"] == "success"
                        for r in valid(h4)
                        if r["arm_name"] == a and r.get("target_exists") and r["scenario"] != "L"
                    ),
                    sum(
                        1
                        for r in valid(h4)
                        if r["arm_name"] == a and r.get("target_exists") and r["scenario"] != "L"
                    ),
                ]
                for a in sorted({r["arm_name"] for r in h4})
            },  # fmt: skip
            "pass": count(h4, "proposed", "false_success")[0] == 0 and falls.get("proposed") == 0,
        }
    else:
        out["G8_embodiment"] = "not run"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    text = json.dumps(gates(pathlib.Path(args.root), args.budget), indent=2, default=float)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
