"""Evaluate the pre-registered gates of experiments/configs/semantic_knowledge_xl.json (K1).

    python experiments/semantic_knowledge_gates.py results/semantic_knowledge_xl/K1 \
        [--config CONFIG] [--out gates.json]

Read-only analysis of episodes.jsonl. Counts are over valid episodes (a crash or a scene
failure is reported and never counted as a pass). Pairs are identical homes (scenario,
target, generator, noise, seed) where the target exists; time to verified find is censored at
the budget for misses (the censoring of experiments/summarise_semantic_search.paired).

- K-G1 (safety): every arm using the current verifier has 0 false success claims; each arm's
  count and zero-event one-sided 95% upper bound are reported.
- K-G2 (primary): each semantic arm against A_frontier on B and D: one-sided paired Wilcoxon
  signed-rank test (alternative: the semantic arm is faster), Holm-Bonferroni across the three
  comparisons at family alpha 0.05. Passes if adjusted p < 0.05 and the median paired
  difference is negative.
- K-G3: each semantic arm's target-present successes >= A_frontier's - margin (3), paired.
- K-G4 (descriptive): arm-versus-arm comparisons, B vs D, path length, refusals on E, the old
  verifier's false claims, per-target results, room-inference accuracy per knowledge source.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.summarise_semantic_search import (  # noqa: E402
    arm_table,
    key,
    load,
    paired,
    zero_upper,
)

SEMANTIC = ("sem_conceptnet", "sem_qwen", "sem_combined")
REFERENCE = "A_frontier"
OLD_VERIFIER = "A_frontier_point"
ALPHA = 0.05
MARGIN = 3


def pair_arrays(rows, a: str, b: str, budget: float, scenarios=None):
    by = defaultdict(dict)
    for r in rows:
        if r.get("result") in ("crash", "scene_failure") or not r.get("target_exists"):
            continue
        if scenarios and r["scenario"] not in scenarios:
            continue
        by[key(r)][r["arm_name"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    sa = np.array([x["result"] == "success" for x, _ in pairs], bool)
    sb = np.array([y["result"] == "success" for _, y in pairs], bool)
    ta = np.array([x["sim_s"] if s else budget for (x, _), s in zip(pairs, sa, strict=True)])
    tb = np.array([y["sim_s"] if s else budget for (_, y), s in zip(pairs, sb, strict=True)])
    return ta, tb, sa, sb


def one_sided_wilcoxon(diff: np.ndarray) -> float:
    """p-value that the paired difference (a - b) is negative; 1.0 when every pair ties."""
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(stats.wilcoxon(nonzero, alternative="less").pvalue)


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm-Bonferroni adjusted p-values (monotone) and rejections."""
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[name]))
        out[name] = {"p": pvals[name], "p_holm": running, "reject": running < alpha}
    return out


def evaluate(rows: list[dict], budget: float) -> dict:
    table = arm_table(rows)
    arms = sorted(table)
    g1_arms = [a for a in arms if a != OLD_VERIFIER]
    g1 = {"gate": "every arm using the current verifier makes 0 false success claims", "arms": {}}
    for a in g1_arms:
        valid = [r for r in rows if r["arm_name"] == a
                 and r.get("result") not in ("crash", "scene_failure")]  # fmt: skip
        false = [r for r in valid if r["result"] == "false_success"]
        g1["arms"][a] = {
            "false_success_claims": len(false), "valid_episodes": len(valid),
            "zero_upper_95": zero_upper(len(valid)) if not false else None,
            "false_claim_episodes": [{k: r.get(k) for k in ("scenario", "target", "seed", "sim_s")}
                                     for r in false],
        }  # fmt: skip
    g1["passed"] = all(v["false_success_claims"] == 0 for v in g1["arms"].values())

    present = ("B", "D")
    raw, detail = {}, {}
    for a in SEMANTIC:
        ta, tb, sa, sb = pair_arrays(rows, a, REFERENCE, budget, present)
        diff = ta - tb
        raw[a] = one_sided_wilcoxon(diff) if len(diff) else 1.0
        detail[a] = {
            "pairs": len(diff), "found": [int(sa.sum()), int(sb.sum())],
            "median_paired_diff_s": round(float(np.median(diff)), 1) if len(diff) else None,
            "mean_paired_diff_s": round(float(np.mean(diff)), 1) if len(diff) else None,
            "only_sem_found": int((sa & ~sb).sum()), "only_ref_found": int((~sa & sb).sum()),
        }  # fmt: skip
    adj = holm(raw, ALPHA)
    g2 = {"gate": "semantic arm vs A_frontier, B+D, one-sided paired Wilcoxon, Holm family 0.05",
          "comparisons": {}}  # fmt: skip
    for a in SEMANTIC:
        d = detail[a]
        passed = bool(adj[a]["reject"] and d["median_paired_diff_s"] is not None
                      and d["median_paired_diff_s"] < 0)  # fmt: skip
        g2["comparisons"][a] = {**d, **adj[a], "passed": passed}
    g2["passed_arms"] = [a for a in SEMANTIC if g2["comparisons"][a]["passed"]]
    g2["passed"] = len(g2["passed_arms"]) == len(SEMANTIC)

    g3 = {"gate": f"semantic successes >= {REFERENCE} successes - {MARGIN}, B+D paired",
          "arms": {}}  # fmt: skip
    for a in SEMANTIC:
        _, _, sa, sb = pair_arrays(rows, a, REFERENCE, budget, present)
        g3["arms"][a] = {"pairs": len(sa), "successes": [int(sa.sum()), int(sb.sum())],
                         "threshold": int(sb.sum()) - MARGIN,
                         "passed": bool(len(sa) and sa.sum() >= sb.sum() - MARGIN)}  # fmt: skip
    g3["passed"] = all(v["passed"] for v in g3["arms"].values())

    def cmp(a, b, scen=None):
        return paired(rows, a, b, budget, scen)

    per_target = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("target_exists") and r.get("result") not in ("crash", "scene_failure"):
            c = per_target[r["target"]][r["arm_name"]]
            c[0] += r["result"] == "success"
            c[1] += 1
    g4 = {
        "descriptive": True,
        "qwen_vs_conceptnet": {"all": cmp("sem_qwen", "sem_conceptnet", present)},
        "combined_vs_qwen": {"all": cmp("sem_combined", "sem_qwen", present)},
        "each_vs_none": {a: cmp(a, "sem_none", present) for a in SEMANTIC},
        "sem_conceptnet_vs_A_frontier": None,
        "by_scenario_vs_A_frontier": {
            a: {s: cmp(a, REFERENCE, (s,)) for s in present} for a in (*SEMANTIC, "sem_none")
        },
        "by_scenario_qwen_vs_conceptnet": {s: cmp("sem_qwen", "sem_conceptnet", (s,))
                                           for s in present},  # fmt: skip
        "A_frontier_point_vs_A_frontier": {
            "false_claims": {a: table[a]["false_success_claims"][:2] for a in (OLD_VERIFIER,
                                                                              REFERENCE)
                             if a in table},
            "paired": cmp(OLD_VERIFIER, REFERENCE, present),
        },
        "per_arm": {
            a: {
                "found_target_present": table[a]["success_target_present"],
                "false_success_claims": table[a]["false_success_claims"],
                "correct_refusal_absent": table[a]["correct_refusal_absent"],
                "false_absence": table[a]["false_absence"],
                "median_success_sim_s": table[a]["median_success_sim_s"],
                "mean_path_m": table[a]["navigation"]["mean_path_m"],
                "mean_sim_s": table[a]["navigation"]["mean_sim_s"],
                "episodes_with_contact": table[a]["navigation"]["episodes_with_contact"],
                "crashes": table[a]["crashes"], "scene_failures": table[a]["scene_failures"],
            }
            for a in arms
        },
        "per_arm_by_scenario": {
            a: {
                s: arm_table([r for r in rows if r["arm_name"] == a and r["scenario"] == s])
                .get(a, {})
                .get("success_target_present" if s != "E" else "correct_refusal_absent")
                for s in ("B", "D", "E")
            }
            for a in arms
        },
        "per_target_found": {t: dict(v) for t, v in sorted(per_target.items())},
        "room_inference_by_knowledge": {
            a: {
                "top1_accuracy": table[a]["rooms"]["top1_accuracy"],
                "top_correct": table[a]["rooms"]["top_correct"],
                "top_with_evidence": table[a]["rooms"]["top_with_evidence"],
                "accuracy_when_decided": table[a]["rooms"]["accuracy_when_decided"],
                "by_true_type": table[a]["rooms"]["by_true_type"],
            }
            for a in (*SEMANTIC, "sem_none") if a in table
        },
    }  # fmt: skip
    del g4["sem_conceptnet_vs_A_frontier"]
    return {"K-G1": g1, "K-G2": g2, "K-G3": g3, "K-G4": g4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/"
                                            "semantic_knowledge_xl.json"))  # fmt: skip
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    rows = load([args.dir])
    result = {"config": cfg["name"], "frozen_commit": cfg.get("frozen_commit"),
              "episodes": len(rows), **evaluate(rows, cfg["time_budget_s"])}  # fmt: skip
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
