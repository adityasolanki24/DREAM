"""Which generator cases can the Isaac composer build? A per-case table of omissions and refusals.

CPU only. For each (variant, seed, scenario, target) it samples the home, applies the exclusion
list and records what is omitted, or why the case is refused (excluded target, scenario depending
on an excluded class, generator failure, a class with no validated asset). Written to
results/isaac_env_validation/home_refusals.json so refusals are listed, never silent.

    python experiments/isaac_home_refusals.py --seeds 100-103 --out <json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.sim.home import HomeGenerationError, sample_home  # noqa: E402
from experiments import isaac_home_compose as comp  # noqa: E402

# The Isaac target set (owner decision of 9 Nov 2026; no bathtub).
TARGETS_ISAAC = ("refrigerator", "bed", "toilet", "stove", "washing_machine", "workbench", "sofa")
SCENARIOS_ISAAC = ("A", "B", "C", "D", "E", "F", "G", "J")


def case_row(variant: str, seed: int, scenario: str, target: str, manifest: dict) -> dict:
    row = {"variant": variant, "seed": seed, "scenario": scenario, "target": target}
    try:
        home = sample_home(np.random.default_rng(seed), scenario, target, variant, seed)
    except HomeGenerationError as error:
        return {**row, "status": "generator_failed", "why": str(error)[:160]}
    row.update(rooms=[r.type for r in home.rooms], objects=len(home.objects))
    try:
        comp.check_case(home, manifest)
    except comp.CaseRefused as refusal:
        return {**row, "status": "refused", "why": refusal.reasons}
    gone = comp.excluded_classes(manifest)
    classes = manifest["classes"]
    missing = sorted(
        {o.cls for o in home.objects if o.cls not in gone and classes.get(o.cls, {}).get(
            "status") not in ("ok", "pending_front_read")}
    )  # fmt: skip
    om = comp.omissions(home, manifest)
    if missing:
        return {**row, "status": "refused", "why": [f"no validated asset for {missing}"]}
    try:
        placements = comp.plan_placements(home, manifest)
    except comp.MissingAssetError as refusal:
        return {**row, "status": "refused", "why": refusal.classes}
    row["native_anchored"] = [p.cls for p in placements if p.mode == "native_anchored"]
    return {**row, "status": "composable", "omitted": [{"class": o["class"],
            "room_id": o["room_id"]} for o in om]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-103")
    ap.add_argument("--variant", default="nominal")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--budget-s", type=float, default=900.0)
    args = ap.parse_args()
    a, b = (int(v) for v in args.seeds.split("-"))
    manifest = comp.load_manifest()
    rows, t0 = [], time.monotonic()
    for seed in range(a, b + 1):
        for scenario in SCENARIOS_ISAAC:
            for target in TARGETS_ISAAC:
                if time.monotonic() - t0 > args.budget_s:
                    break
                rows.append(case_row(args.variant, seed, scenario, target, manifest))
    summary: dict[str, int] = {}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    args.out.write_text(json.dumps({"exclusion_list": manifest.get("excluded"),
                                    "summary": summary, "cases": rows}, indent=1))  # fmt: skip
    print(summary)


if __name__ == "__main__":
    main()
