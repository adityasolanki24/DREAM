"""Offline phase diagnostics from original 100 Hz traces; no physics execution.

Contact loss and relative displacement are measurements, not a diagnosis of friction.
The independent 1 kHz peak score remains authoritative; sampled phase peaks can be lower.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.g1_dexterous import write_json


def audit(folder: Path) -> list[dict]:
    with (folder / "physics.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    events = json.loads((folder / "tools.json").read_text())
    results = []
    for event in events:
        if event["time_s"] <= event.get("start_time_s", event["time_s"]):
            continue
        phase = [r for r in rows if event["start_time_s"] < r["time_s"] <= event["time_s"] + 1e-8]
        if not phase:
            continue
        obj = np.array([r["object_position_m"] for r in phase])
        hand = np.array([r["grasp_frame_position_m"] for r in phase])
        relative = obj - hand
        contact = [
            {
                c["body"].split("_hand_")[1].split("_")[0]
                for c in r["contacts"]
                if "_hand_" in c["body"] and c["normal_force_n"] > 0.1
            }
            for r in phase
        ]
        peak = (
            max((-c["distance_m"], r["time_s"], c["body"]) for r in phase for c in r["contacts"])
            if any(r["contacts"] for r in phase)
            else (0, None, None)
        )
        results.append(
            {
                "tool": event["tool"],
                "start_s": event["start_time_s"],
                "end_s": event["time_s"],
                "status": event["status"],
                "message": event["message"],
                "object_start_m": obj[0].tolist(),
                "object_end_m": obj[-1].tolist(),
                "object_min_height_m": float(obj[:, 2].min()),
                "relative_object_drift_max_m": float(
                    np.linalg.norm(relative - relative[0], axis=1).max()
                ),
                "last_two_chain_contact_s": next(
                    (
                        r["time_s"]
                        for r, chains in reversed(list(zip(phase, contact, strict=True)))
                        if len(chains) >= 2
                    ),
                    None,
                ),
                "sampled_peak_penetration_m": peak[0],
                "peak_time_s": peak[1],
                "peak_body": peak[2],
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        str(path.parent.relative_to(args.root)): audit(path.parent)
        for path in args.root.rglob("physics.jsonl")
        if (path.parent / "tools.json").is_file()
    }
    write_json(args.out, result)
    for name, phases in result.items():
        print(
            name,
            [
                (
                    p["tool"],
                    p["status"],
                    round(p["sampled_peak_penetration_m"] * 1000, 2),
                    round(p["relative_object_drift_max_m"] * 1000, 1),
                    round(p["object_end_m"][2], 3),
                )
                for p in phases
            ],
        )
