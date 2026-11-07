"""Read results/g1_explore_room_postfix_closeout under the rule declared in its frozen config.

    python experiments/summarise_room_postfix.py [results_root]

Reads R1, R2_fresh, R2_hist and R3 (episodes.jsonl) and the baselines named in the config
(results/g1_explore_room_closeout, results/g1_explore_room_regression_closeout/A1,
results/g1_explore_room, gain rows only). Writes diagnostic_summary.json and tables.md next to the
arms. Nothing is rescored: pass flags are the scorer's own, as recorded.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAND_FOOT = {"hand_handle", "hand_furniture", "foot_furniture"}
KINDS = ("nominal", "shifted", "no_cabinet")


def load(path: pathlib.Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["kind"], r["seed"]): r for r in rows if r.get("method", "gain") == "gain"}


def passed(r: dict) -> bool:
    if r["kind"] == "no_cabinet":
        return bool(r.get("correct_refusal"))
    return bool(r["success"])


def hand_foot(r: dict) -> bool | None:
    if "contact_classes" not in r:
        return None
    return bool(HAND_FOOT & set(r["contact_classes"]))


def compare(new: dict, base: dict, base_name: str) -> dict:
    seeds = sorted(new, key=lambda k: (KINDS.index(k[0]), k[1]))
    per_seed, flips = [], {"fail_to_pass": [], "pass_to_fail": []}
    for key in seeds:
        n, b = new[key], base.get(key)
        row = {
            "kind": key[0], "seed": key[1], "new_pass": passed(n),
            "base_pass": None if b is None else passed(b),
            "new_strict": n.get("strict_success"),
            "new_claimed": n["claimed"], "new_false_claim": n["false_claim"], "new_fell": n["fell"],
            "new_contact_classes": n.get("contact_classes"),
            "base_contact_classes": None if b is None else b.get("contact_classes"),
            "new_sim_s": round(n["sim_s"], 1),
            "base_sim_s": None if b is None else round(b["sim_s"], 1),
            "new_reason": n["reason"][:90],
            "renderer": n.get("renderer"),
        }  # fmt: skip
        if b is not None and passed(n) != passed(b):
            flips["fail_to_pass" if passed(n) else "pass_to_fail"].append(key)
        per_seed.append(row)
    by_kind = {}
    for kind in KINDS:
        ks = [k for k in seeds if k[0] == kind]
        if not ks:
            continue
        by_kind[kind] = {
            "n": len(ks),
            "new_pass": sum(passed(new[k]) for k in ks),
            "base_pass": sum(passed(base[k]) for k in ks if k in base),
            "base_n": sum(k in base for k in ks),
            "new_strict_success": sum(bool(new[k].get("strict_success")) for k in ks),
            "flips": sum((passed(new[k]) != passed(base[k])) for k in ks if k in base),
        }
    hf_new = [hand_foot(new[k]) for k in seeds]
    hf_base = [hand_foot(base[k]) for k in seeds if k in base]
    return {
        "baseline": base_name,
        "n": len(seeds),
        "new_pass": sum(passed(new[k]) for k in seeds),
        "base_pass": sum(passed(base[k]) for k in seeds if k in base),
        "flips_total": len(flips["fail_to_pass"]) + len(flips["pass_to_fail"]),
        "fail_to_pass": [list(k) for k in flips["fail_to_pass"]],
        "pass_to_fail": [list(k) for k in flips["pass_to_fail"]],
        "false_claims": sum(bool(new[k]["false_claim"]) for k in seeds),
        "falls": sum(bool(new[k]["fell"]) for k in seeds),
        "hand_foot_contact_episodes": sum(bool(x) for x in hf_new if x is not None),
        "base_hand_foot_contact_episodes": (
            None if None in hf_base or not hf_base else sum(bool(x) for x in hf_base)
        ),
        "episodes_with_any_contact": sum(bool(new[k].get("contact_classes")) for k in seeds),
        "by_kind": by_kind,
        "per_seed": per_seed,
    }


def table(c: dict, title: str) -> str:
    out = [f"### {title}", "", f"Baseline: {c['baseline']}", ""]
    out.append("| kind | seed | base | new | strict | claim | contacts (new) | contacts (base) | "
               "sim_s new/base | reason |")  # fmt: skip
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in c["per_seed"]:
        mark = "" if r["base_pass"] in (None, r["new_pass"]) else " **flip**"
        out.append(
            f"| {r['kind']} | {r['seed']} | {r['base_pass']} | {r['new_pass']}{mark} | "
            f"{r['new_strict']} | {r['new_claimed']}{'/FALSE' if r['new_false_claim'] else ''} | "
            f"{','.join(r['new_contact_classes'] or []) or '-'} | "
            f"{','.join(r['base_contact_classes'] or []) or '-'} | "
            f"{r['new_sim_s']}/{r['base_sim_s']} | {r['new_reason']} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    post = root / "g1_explore_room_postfix_closeout"
    frozen = load(root / "g1_explore_room_closeout/episodes.jsonl")
    a1 = load(root / "g1_explore_room_regression_closeout/A1/episodes.jsonl")
    hist = load(root / "g1_explore_room/episodes.jsonl")
    arms = {a: load(post / a / "episodes.jsonl") for a in ("R1", "R2_fresh", "R2_hist", "R3")
            if (post / a / "episodes.jsonl").exists()}  # fmt: skip
    bases = {
        "R1": (frozen, "frozen EGL result (results/g1_explore_room_closeout, gain rows)"),
        "R2_fresh": (frozen, "frozen EGL result (results/g1_explore_room_closeout, gain rows)"),
        "R2_hist": (a1, "A1 (results/g1_explore_room_regression_closeout/A1)"),
        "R3": (hist, "historical OSMesa record (results/g1_explore_room, gain rows)"),
    }
    comp = {a: compare(rows, *bases[a]) for a, rows in arms.items()}
    if "R3" in comp:  # the historical rows carry no contact classes: compare contacts with A1
        c = comp["R3"]
        hf = [hand_foot(a1[k]) for k in arms["R3"] if k in a1]
        c["contact_baseline"] = "A1 (historical rows carry no contact classes)"
        c["base_hand_foot_contact_episodes"] = sum(bool(x) for x in hf)
    post_arms = [a for a in ("R2_fresh", "R2_hist", "R3") if a in comp]
    r1 = comp.get("R1")
    cond = {
        "complete": all(a in comp for a in ("R1", "R2_fresh", "R2_hist", "R3"))
        and all(comp[a]["n"] == n for a, n in (("R1", 40), ("R2_fresh", 40), ("R2_hist", 36),
                                               ("R3", 36))),
        "1_no_false_claim": all(comp[a]["false_claims"] == 0 for a in post_arms),
        "2_no_fall": all(comp[a]["falls"] == 0 for a in post_arms),
        "3_contacts_up_at_most_2": {
            a: comp[a]["hand_foot_contact_episodes"] - comp[a]["base_hand_foot_contact_episodes"]
            for a in post_arms
        },
        "4_seed_1013_passes_under_osmesa": (
            bool(arms.get("R3", {}).get(("nominal", 1013)))
            and passed(arms["R3"][("nominal", 1013)])
        ),
        "5_pass_count_change_per_arm": {a: comp[a]["new_pass"] - comp[a]["base_pass"]
                                        for a in post_arms},
        "5_per_kind_pass_change": {
            a: {k: v["new_pass"] - v["base_pass"] for k, v in comp[a]["by_kind"].items()}
            for a in post_arms
        },
    }  # fmt: skip
    cond["3_ok"] = all(v <= 2 for v in cond["3_contacts_up_at_most_2"].values())
    cond["5_ok_arm_pooled"] = all(v >= 0 for v in cond["5_pass_count_change_per_arm"].values())
    cond["5_ok_per_kind"] = all(
        v >= 0 for d in cond["5_per_kind_pass_change"].values() for v in d.values()
    )
    cond["retain_arm_pooled"] = bool(
        cond["complete"] and cond["1_no_false_claim"] and cond["2_no_fall"] and cond["3_ok"]
        and cond["4_seed_1013_passes_under_osmesa"] and cond["5_ok_arm_pooled"]
    )  # fmt: skip
    cond["retain_arm_per_kind"] = bool(cond["retain_arm_pooled"] and cond["5_ok_per_kind"])
    summary = {
        "label": "post-fix development and renderer diagnostic; not held-out",
        "R1_renderer_flips": None if r1 is None else {
            "of": r1["n"], "flipped": r1["flips_total"],
            "fail_to_pass_under_osmesa": r1["fail_to_pass"],
            "pass_to_fail_under_osmesa": r1["pass_to_fail"],
            "by_kind": {k: v["flips"] for k, v in r1["by_kind"].items()},
        },  # fmt: skip
        "retention": cond,
        "arms": {a: {k: v for k, v in c.items() if k != "per_seed"} for a, c in comp.items()},
        "per_seed": {a: c["per_seed"] for a, c in comp.items()},
    }
    (post / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    titles = {"R1": "R1 pre-fix, OSMesa, fresh seeds", "R2_fresh": "R2 post-fix, EGL, fresh seeds",
              "R2_hist": "R2 post-fix, EGL, historical seeds",
              "R3": "R3 post-fix, OSMesa, historical seeds"}  # fmt: skip
    (post / "tables.md").write_text("\n".join(table(comp[a], titles[a]) for a in comp))
    print(json.dumps({k: summary[k] for k in ("R1_renderer_flips", "retention")}, indent=2,
                     default=float))  # fmt: skip


if __name__ == "__main__":
    main()
