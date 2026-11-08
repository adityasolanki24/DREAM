"""Summarise object-search runs: per-arm outcomes with intervals, paired comparisons against
the proposed method, and navigation reported apart from semantic correctness.

    python experiments/summarise_semantic_search.py RESULT_DIR [RESULT_DIR ...] --out DIR

Read-only analysis of episodes.jsonl files. Statistics:
- proportions with Wilson 95% intervals; zero false claims also get the one-sided 95% upper
  bound (rule of three ~ 3/n), so "0 of n" is never presented as a rate of zero;
- paired comparison on identical homes (scenario, target, generator, noise, seed): exact
  McNemar test on success, and time to verified success censored at the budget (an
  unsuccessful episode counts as the full budget) with a Wilcoxon signed-rank test and a
  bootstrap 95% interval of the median paired difference (2000 resamples, seed 0).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

GROUPS = {
    "visible": ["A"],
    "typical_location": ["B", "C", "F", "G", "K"],
    "atypical_location": ["D", "J"],
    "target_absent": ["E", "H"],
    "memory": ["I"],
    "unsupported": ["L"],
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def zero_upper(n: int) -> float | None:
    """One-sided 95% upper bound on a rate when 0 of n events were seen."""
    return None if n == 0 else round(1 - 0.05 ** (1 / n), 4)


def load(dirs: list[str]) -> list[dict]:
    rows = []
    for d in dirs:
        for line in (pathlib.Path(d) / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r: dict) -> tuple:
    return (r["scenario"], r["target"], r["generator"], r["noise"], r["seed"])


def arm_table(rows: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm_name"] for r in rows}):
        rs = [r for r in rows if r["arm_name"] == arm]
        res = Counter(r.get("result") for r in rs)
        n = len(rs)
        valid = [r for r in rs if r.get("result") not in ("crash", "scene_failure")]
        present = [r for r in valid if r.get("target_exists") and r["scenario"] != "L"]
        absent = [r for r in valid if not r.get("target_exists") and r["scenario"] != "L"]
        unsup = [r for r in valid if r["scenario"] == "L"]
        succ = sum(r["result"] == "success" for r in present)
        false = sum(r["result"] == "false_success" for r in valid)
        rooms = Counter()
        for r in valid:
            for k in ("correct", "incorrect", "ambiguous"):
                rooms[k] += (r.get("rooms") or {}).get(k, 0)
        decided = rooms["correct"] + rooms["incorrect"]
        times = [r["sim_s"] for r in present if r["result"] == "success"]
        out[arm] = {
            "episodes": n,
            "results": dict(res),
            "success_target_present": [succ, len(present), wilson(succ, len(present))],
            "false_success_claims": [
                false,
                len(valid),
                wilson(false, len(valid)),
                zero_upper(len(valid)) if false == 0 else None,
            ],
            "correct_refusal_absent": _rate(absent, "correct_refusal"),
            "correct_refusal_unsupported": _rate(unsup, "correct_refusal"),
            "false_absence": _rate(present, "false_absence"),
            "identity_errors": sum(bool(r.get("identity_error")) for r in valid),
            "median_success_sim_s": float(np.median(times)) if times else None,
            "navigation": {
                "mean_path_m": _mean(valid, "path_m"),
                "mean_sim_s": _mean(valid, "sim_s"),
                "failed_go_to": sum(r.get("failed_go_to", 0) for r in valid),
                "episodes_with_contact": sum(bool(r.get("contact_substeps")) for r in valid),
                "falls": sum(bool(r.get("fell")) for r in valid),
                "mean_frames": _mean(valid, "frames"),
                "mean_look_arounds": _mean(valid, "look_arounds"),
                "mean_explored_m2": _mean(valid, "explored_m2"),
                "model_calls": sum(r.get("model_calls", 0) for r in valid),
            },
            "rooms": {
                **rooms,
                "accuracy_when_decided": round(rooms["correct"] / decided, 3) if decided else None,
            },  # fmt: skip
            "crashes": res.get("crash", 0),
            "scene_failures": res.get("scene_failure", 0),
        }
    return out


def _rate(rs, result):
    k = sum(r["result"] == result for r in rs)
    return [k, len(rs), wilson(k, len(rs))]


def _mean(rs, field):
    vals = [r.get(field) for r in rs if isinstance(r.get(field), (int, float))]
    return round(float(np.mean(vals)), 2) if vals else None


def paired(rows: list[dict], a: str, b: str, budget: float, scenarios=None) -> dict:
    """Arm ``a`` against ``b`` on identical homes where the target exists."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("result") in ("crash", "scene_failure") or not r.get("target_exists"):
            continue
        if scenarios and r["scenario"] not in scenarios:
            continue
        by[key(r)][r["arm_name"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    if not pairs:
        return {"pairs": 0}
    sa = np.array([x["result"] == "success" for x, _ in pairs])
    sb = np.array([y["result"] == "success" for _, y in pairs])
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    mcnemar = (
        float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    )
    ta = np.array([x["sim_s"] if s else budget for (x, _), s in zip(pairs, sa, strict=True)])
    tb = np.array([y["sim_s"] if s else budget for (_, y), s in zip(pairs, sb, strict=True)])
    diff = ta - tb
    rng = np.random.default_rng(0)
    boots = [float(np.median(rng.choice(diff, len(diff)))) for _ in range(2000)]
    nonzero = diff[diff != 0]
    wil = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else None
    return {
        "pairs": len(pairs),
        "success": [int(sa.sum()), int(sb.sum())],
        "discordant": {"only_" + a: only_a, "only_" + b: only_b},
        "mcnemar_exact_p": round(mcnemar, 4),
        "median_time_diff_s": round(float(np.median(diff)), 1),
        "median_time_diff_ci95": [
            round(float(np.percentile(boots, 2.5)), 1),
            round(float(np.percentile(boots, 97.5)), 1),
        ],  # fmt: skip
        "wilcoxon_p": None if wil is None else round(wil, 4),
        "note": "time censored at the budget for unsuccessful episodes; negative = a faster",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--reference", default="proposed")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = load(args.dirs)
    arms = sorted({r["arm_name"] for r in rows})
    out = {
        "episodes": len(rows),
        "by_arm": arm_table(rows),
        "by_group": {
            g: arm_table([r for r in rows if r["scenario"] in s]) for g, s in GROUPS.items()
        },
        "paired": {
            f"{args.reference}_vs_{b}": {
                "all": paired(rows, args.reference, b, args.budget),
                **{
                    g: paired(rows, args.reference, b, args.budget, s)
                    for g, s in GROUPS.items()
                    if g not in ("target_absent", "unsupported")
                },
            }
            for b in arms
            if b != args.reference
        },  # fmt: skip
    }
    text = json.dumps(out, indent=2, default=float)
    if args.out:
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(args.out) / "analysis.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
