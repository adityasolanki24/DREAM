"""Compare every deterministic development attempt; retain rejected hypotheses explicitly."""

import argparse
import json
from pathlib import Path

from experiments.g1_dexterous import write_json


def summarize(roots: list[Path]) -> dict:
    stages = []
    for root in roots:
        metadata = json.loads((root / "metadata.json").read_text())
        config = metadata["config"]
        records = json.loads((root / "summary.json").read_text())
        if any(c.get("split") != "development" for c in config["cases"]):
            raise ValueError("reliability development must not relabel held-out data")
        stages.append(
            {
                "stage": root.name,
                "source_metadata": str(root / "metadata.json"),
                "planned": len(config["cases"]),
                "completed": len(records),
                "physical_successes": sum(r["goal_success"] for r in records.values()),
                "false_success_claims": sum(r["false_success_claim"] for r in records.values()),
                "cases": {
                    c["name"]: {
                        "seed": c["seed"],
                        **{
                            k: records.get(c["name"], {}).get(k)
                            for k in (
                                "goal_success",
                                "goal_geometric",
                                "success_claimed",
                                "false_success_claim",
                                "maximum_contact_penetration_m",
                                "penetration_gate_passed",
                                "safety_stop",
                                "maximum_sustained_lift_contact_s",
                                "failed_actions",
                                "reason",
                                "recording",
                            )
                        },
                    }
                    for c in config["cases"]
                },
            }
        )
    return {
        "scope": "Matched deterministic development experiments only; no new model inference, "
        "no frozen evaluation change, no claim of held-out transfer.",
        "stages": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stages)
    write_json(args.out, result)
    print([(s["stage"], s["physical_successes"], s["completed"]) for s in result["stages"]])
