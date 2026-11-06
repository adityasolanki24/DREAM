"""Gates and summaries for the closeout evaluations, computed from saved episodes only.

    python experiments/summarise_closeout.py room results/g1_explore_room_closeout
    python experiments/summarise_closeout.py compartment results/compartment_vision_closeout

Writes closeout_summary.json next to the episodes. No re-running, no thresholds of its own:
every gate and the frontier-comparison rules are read from the frozen config recorded in the
output directory's summary.json (or the committed config of the same name).
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ("foot_furniture", "hand_handle", "hand_furniture", "other")


def load_rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def bootstrap_mean_ci(x, n: int = 10000, seed: int = 0) -> list[float] | None:
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()


def rate(k: int, n: int) -> float | None:
    return k / n if n else None


def phantom_driven(r: dict) -> bool:
    return r.get("refusal_category") == "phantom_or_unreachable_candidate" or (
        r.get("phantom_handles", 0) > 0
    )


def room_summary(rows: list[dict], cfg: dict) -> dict:
    g = cfg["gates"]
    gain = lambda kind: [r for r in rows if r["kind"] == kind and r["method"] == "gain"]  # noqa: E731
    nom, sh, nc = gain("nominal"), gain("shifted"), gain("no_cabinet")
    nc_refuse = sum(bool(r.get("correct_refusal")) for r in nc)
    strict_refuse = sum(bool(r.get("correct_refusal")) and not phantom_driven(r) for r in nc)
    non_phantom = [r for r in nc if not phantom_driven(r)]
    gates = {
        "nominal_success": {
            "value": rate(sum(r["success"] for r in nom), len(nom)),
            "n": len(nom), "gate": g["nominal_success_min"],
        },
        "shifted_success": {
            "value": rate(sum(r["success"] for r in sh), len(sh)),
            "n": len(sh), "gate": g["shifted_success_min"],
        },
        "no_cabinet_correct_refusal_original_scorer": {
            "value": rate(nc_refuse, len(nc)), "n": len(nc), "gate": g["no_cabinet_refusal_min"],
        },
        "false_claims_all_rows": {
            "value": sum(r["false_claim"] for r in rows), "n": len(rows),
            "gate": g["max_false_claims"],
        },
        "falls_all_rows": {
            "value": sum(r["fell"] for r in rows), "n": len(rows), "gate": g["max_falls"],
        },
    }  # fmt: skip
    for name, v in gates.items():
        v["pass"] = (
            v["value"] is not None
            and (v["value"] <= v["gate"] if name.endswith("_all_rows") else v["value"] >= v["gate"])
        )  # fmt: skip

    # Contacts by body class, over every episode of every arm.
    contacts = {}
    for cls in CLASSES:
        eps = [r for r in rows if cls in r.get("contact_classes", [])]
        cs = [c for r in rows for c in r.get("contacts", []) if c["class"] == cls]
        contacts[cls] = {
            "episodes": len(eps), "of": len(rows),
            "ticks": sum(c["ticks"] for c in cs),
            "peak_force_n": max((c["peak_force_n"] for c in cs), default=0.0),
            "episodes_in_last_3s": len({(r["kind"], r["seed"], r["method"]) for r in rows
                                        for c in r.get("contacts", [])
                                        if c["class"] == cls and c["in_last_3s"]}),
        }  # fmt: skip
    pairs_ticks: dict = defaultdict(lambda: [0, 0.0])
    for r in rows:
        for c in r.get("contacts", []):
            k = f"{c['robot_body']} | {c['other']}"
            pairs_ticks[k][0] += c["ticks"]
            pairs_ticks[k][1] = max(pairs_ticks[k][1], c["peak_force_n"])
    succ = [r for r in rows if r["success"]]
    strict = {
        "successes": len(succ),
        "strict_successes": sum(bool(r.get("strict_success")) for r in succ),
        "successes_with_non_hand_handle_contact": sum(not r.get("strict_success") for r in succ),
        "gain_nominal_strict": rate(sum(bool(r.get("strict_success")) for r in nom), len(nom)),
        "gain_shifted_strict": rate(sum(bool(r.get("strict_success")) for r in sh), len(sh)),
    }
    no_cab = {
        "n": len(nc), "correct_refusals_original": nc_refuse,
        "phantom_driven_refusals": sum(phantom_driven(r) for r in nc if not r["claimed"]),
        "strict_correct_refusals_excluding_phantom_driven": strict_refuse,
        "strict_rate_over_all": rate(strict_refuse, len(nc)),
        "strict_rate_over_non_phantom_episodes": rate(strict_refuse, len(non_phantom)),
        "n_non_phantom_episodes": len(non_phantom),
    }  # fmt: skip
    phantom = {
        "episodes_with_phantom": sum(r.get("phantom_handles", 0) > 0 for r in rows),
        "phantom_handles_total": sum(r.get("phantom_handles", 0) for r in rows),
        "by_kind_method": {
            f"{k}/{m}": sum(r.get("phantom_handles", 0) for r in rows
                            if r["kind"] == k and r["method"] == m)
            for k, m in sorted({(r["kind"], r["method"]) for r in rows})
        },
    }  # fmt: skip
    refusals = Counter(
        f"{r['kind']}/{r['method']}: {r['refusal_category']}"
        for r in rows
        if r.get("refusal_category")
    )

    # Paired frontier comparison.
    fc = cfg["frontier_comparison"]
    by = {(r["kind"], r["seed"], r["method"]): r for r in rows}
    pairs = [
        (by[(k, s, "gain")], by[(k, s, "nearest")])
        for k, s, m in sorted(by)
        if k in ("shifted", "no_cabinet") and m == "gain" and (k, s, "nearest") in by
    ]
    d_time = [a["sim_s"] - b["sim_s"] for a, b in pairs]
    d_succ = [float(a["success"]) - float(b["success"]) for a, b in pairs]
    div_gain = sum(a.get("frontier_divergences", 0) for a, _ in pairs)
    div_near = sum(b.get("frontier_divergences", 0) for _, b in pairs)
    ci = bootstrap_mean_ci(d_time)
    if div_gain < fc["min_divergent_decisions"]:
        verdict = "insufficient divergence, no claim"
    elif ci is not None and ci[1] < 0:
        verdict = "gain faster than nearest: interval excludes 0 in gain's favour"
    else:
        verdict = "no claim: interval does not exclude 0 in gain's favour"
    comparison = {
        "n_pairs": len(pairs),
        "mean_sim_s_diff_gain_minus_nearest": float(np.mean(d_time)) if d_time else None,
        "sim_s_diff_ci95": ci,
        "mean_success_diff_gain_minus_nearest": float(np.mean(d_succ)) if d_succ else None,
        "success_diff_ci95": bootstrap_mean_ci(d_succ),
        "divergent_decisions_gain_arm": div_gain,
        "divergent_decisions_nearest_arm": div_near,
        "pairs_with_any_divergence": sum(a.get("frontier_divergences", 0) > 0 for a, _ in pairs),
        "decisions_gain_arm": sum(len(a.get("frontier_decisions", [])) for a, _ in pairs),
        "min_divergent_decisions": fc["min_divergent_decisions"],
        "conclusion": verdict,
        "by_kind": {
            k: {
                "n": len([1 for a, _ in pairs if a["kind"] == k]),
                "mean_sim_s_diff": float(np.mean([a["sim_s"] - b["sim_s"] for a, b in pairs
                                                  if a["kind"] == k] or [np.nan])),
            }
            for k in ("shifted", "no_cabinet")
        },
    }  # fmt: skip

    failures = [
        {
            "kind": r["kind"], "seed": r["seed"], "method": r["method"], "claimed": r["claimed"],
            "success": r["success"], "false_claim": r["false_claim"], "fell": r["fell"],
            "refusal_category": r.get("refusal_category"), "reason": r["reason"],
            "phantom_handles": r.get("phantom_handles"), "sim_s": r["sim_s"],
            "contact_classes": r.get("contact_classes"),
            "identity_error_m": r.get("identity_error_m"),
            "true_handle_distance_m": r.get("true_handle_distance_m"),
            "on_front_side": r.get("on_front_side"),
            "facing_error_deg": r.get("facing_error_deg"),
        }
        for r in rows
        if (r["kind"] != "no_cabinet" and not r["success"])
        or (r["kind"] == "no_cabinet" and not r.get("correct_refusal"))
        or r["fell"]
    ]  # fmt: skip
    return {
        "gates": gates, "all_gates_pass": all(v["pass"] for v in gates.values()),
        "no_cabinet_strict": no_cab, "contacts_by_class": contacts,
        "contacts_by_pair_ticks_peak_force": {k: v for k, v in sorted(pairs_ticks.items(),
                                                                      key=lambda kv: -kv[1][0])},
        "strict_success": strict, "phantoms": phantom,
        "refusal_categories": dict(refusals), "frontier_comparison": comparison,
        "failures": failures,
    }  # fmt: skip


def compartment_category(r: dict) -> str:
    if r["kind"] == "fixed":
        if r["false_claim"]:
            return "false_claim"
        return "fixed_not_declined" if not r["correct_rejection"] else "ok"
    if r["success"]:
        return "ok"
    if r["false_claim"]:
        return "false_claim"
    inv = r.get("investigation") or {}
    if not inv:
        return "no_investigation"
    if not r.get("identified_correctly"):
        return f"not_identified_correctly(status={inv.get('status')})"
    return "identified_but_not_opened_or_placed"


def compartment_summary(rows: list[dict], cfg: dict) -> dict:
    g = cfg["gates"]
    op = [r for r in rows if r["kind"] != "fixed"]
    fx = [r for r in rows if r["kind"] == "fixed"]
    gates = {
        "dream_openable_success": {
            "value": rate(sum(r["success"] for r in op), len(op)), "n": len(op),
            "gate": g["dream_openable_success_min"],
        },
        "dream_fixed_declined": {
            "value": rate(sum(r["correct_rejection"] for r in fx), len(fx)), "n": len(fx),
            "gate": g["dream_correct_rejection_min"],
        },
        "false_claims": {
            "value": sum(r["false_claim"] for r in rows), "n": len(rows),
            "gate": g["max_false_claims"],
        },
    }  # fmt: skip
    for n, v in gates.items():
        v["pass"] = v["value"] is not None and (
            v["value"] <= v["gate"] if n == "false_claims" else v["value"] >= v["gate"]
        )
    cats = [(r, compartment_category(r)) for r in rows]
    return {
        "gates": gates, "all_gates_pass": all(v["pass"] for v in gates.values()),
        "by_kind": {k: {"n": sum(r["kind"] == k for r in rows),
                        "success": sum(r["success"] for r in rows if r["kind"] == k),
                        "correct_rejection": sum(r["correct_rejection"] for r in rows
                                                 if r["kind"] == k)}
                    for k in ("drawer", "lid", "fixed")},
        "failures": [
            {"kind": r["kind"], "seed": r["seed"], "category": c, "claimed": r["claimed"],
             "reason": r["reason"], "open_calls": r["open_calls"], "tool_calls": r["tool_calls"],
             "failed_actions": r["failed_actions"],
             "investigation_status": (r.get("investigation") or {}).get("status"),
             "investigation_reason": (r.get("investigation") or {}).get("reason"),
             "identified_correctly": r.get("identified_correctly"),
             "protective_stop": r.get("protective_stop"), "link_collision": r.get("link_collision")}
            for r, c in cats if c != "ok"
        ],
        "failure_categories": dict(Counter(c for _, c in cats if c != "ok")),
    }  # fmt: skip


def main() -> None:
    kind, d = sys.argv[1], pathlib.Path(sys.argv[2])
    d = d if d.is_absolute() else ROOT / d
    cfg = json.loads((ROOT / "experiments/configs" / (d.name + ".json")).read_text())
    rows = load_rows(d / "episodes.jsonl")
    fn = {"room": room_summary, "compartment": compartment_summary}[kind]
    out = {"config": d.name, "status": cfg.get("status"), "n_episodes": len(rows),
           "python": platform.python_version(), "numpy": np.__version__,
           **fn(rows, cfg)}  # fmt: skip
    (d / "closeout_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("gates", "all_gates_pass")}, indent=2, default=float))


if __name__ == "__main__":
    main()
