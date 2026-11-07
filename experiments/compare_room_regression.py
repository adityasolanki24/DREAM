"""Per-seed comparison of a regression-diagnostic arm with its frozen or historical baseline.

    python experiments/compare_room_regression.py ARM_DIR BASELINE_DIR OUT.json

Reads saved episodes only (gain rows). A seed 'passes' under the pre-declared definition in
experiments/configs/g1_explore_room_regression_closeout.json: success for nominal and shifted,
correct refusal for no_cabinet. Nothing is rescored; baseline files are only read.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

NO_VIEWPOINT = "no reachable viewpoint"


def load(directory: pathlib.Path) -> dict[tuple[str, int], dict]:
    path = directory / "episodes.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {(r["kind"], r["seed"]): r for r in rows if r["method"] == "gain"}


def passed(r: dict) -> bool:
    return bool(r["correct_refusal"]) if r["kind"] == "no_cabinet" else bool(r["success"])


def contact_classes(r: dict) -> list[str]:
    return sorted(set(r.get("contact_classes") or []))


def digest(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {
        "pass": passed(r),
        "claimed": r["claimed"],
        "false_claim": r["false_claim"],
        "fell": r["fell"],
        "contact_steps": r.get("contact_steps"),
        "contact_classes": contact_classes(r) if "contact_classes" in r else None,
        "strict_success": r.get("strict_success"),
        "refusal_category": r.get("refusal_category"),
        "no_viewpoint": NO_VIEWPOINT in r["reason"],
        "sim_s": round(r["sim_s"], 1),
        "reason": r["reason"][:90],
    }


def compare(arm: dict, base: dict) -> dict:
    keys = sorted(set(arm) | set(base))
    seeds = [
        {
            "kind": k[0], "seed": k[1], "baseline": digest(base.get(k)), "arm": digest(arm.get(k)),
            "changed": bool(k in arm and k in base and passed(arm[k]) != passed(base[k])),
        }
        for k in keys
    ]  # fmt: skip
    out = {"per_seed": seeds, "by_kind": {}}
    for kind in ("nominal", "shifted", "no_cabinet"):
        rows = [s for s in seeds if s["kind"] == kind and s["arm"] and s["baseline"]]
        out["by_kind"][kind] = {
            "n": len(rows),
            "baseline_pass": sum(s["baseline"]["pass"] for s in rows),
            "arm_pass": sum(s["arm"]["pass"] for s in rows),
            "seeds_changed": [s["seed"] for s in rows if s["changed"]],
            "fixed": [s["seed"] for s in rows if s["arm"]["pass"] and not s["baseline"]["pass"]],
            "broken": [s["seed"] for s in rows if s["baseline"]["pass"] and not s["arm"]["pass"]],
            "false_claims_baseline": sum(s["baseline"]["false_claim"] for s in rows),
            "false_claims_arm": sum(s["arm"]["false_claim"] for s in rows),
            "falls_baseline": sum(s["baseline"]["fell"] for s in rows),
            "falls_arm": sum(s["arm"]["fell"] for s in rows),
            "no_viewpoint_baseline": sum(s["baseline"]["no_viewpoint"] for s in rows),
            "no_viewpoint_arm": sum(s["arm"]["no_viewpoint"] for s in rows),
            "contact_episodes_baseline": sum(bool(s["baseline"]["contact_steps"]) for s in rows),
            "contact_episodes_arm": sum(bool(s["arm"]["contact_steps"]) for s in rows),
            "strict_success_arm": sum(bool(s["arm"]["strict_success"]) for s in rows),
            "strict_success_baseline": (
                sum(bool(s["baseline"]["strict_success"]) for s in rows)
                if all(s["baseline"]["strict_success"] is not None for s in rows)
                else None
            ),
            "arm_contact_class_counts": dict(
                Counter(c for s in rows for c in (s["arm"]["contact_classes"] or []))
            ),
        }
    out["seeds_changed_total"] = sum(len(v["seeds_changed"]) for v in out["by_kind"].values())
    return out


def main() -> None:
    arm_dir, base_dir, out = map(pathlib.Path, sys.argv[1:4])
    result = compare(load(arm_dir), load(base_dir))
    result["arm_dir"], result["baseline_dir"] = str(arm_dir), str(base_dir)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for s in result["per_seed"]:
        b, a = s["baseline"], s["arm"]
        print(
            f"{s['kind']:10} {s['seed']}  base={'P' if b and b['pass'] else 'f'}"
            f"  arm={'P' if a and a['pass'] else 'f'}{'  CHANGED' if s['changed'] else ''}"
            f"  | {(a or {}).get('reason', 'missing')[:60]}"
        )
    print(json.dumps(result["by_kind"], indent=1))


if __name__ == "__main__":
    main()
