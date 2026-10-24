"""Export ordinary physics recordings, retaining failures and excluding held-out training.

Tool status alone never makes a training demonstration. Placement episodes must pass
the unchanged independent physics gates. Lift labels additionally require sustained
measured object lift with multiple contacting finger chains. These are classical
demonstrations; exporting them neither trains nor validates a learned motor skill.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.g1_dataset import export
from experiments.g1_recording import FPS, external_recording_root


def skill_intervals(events: list[dict], rows: list[dict], summary: dict) -> list[dict]:
    labels = []
    previous = 0.0
    initial_z = rows[0]["object_position_m"][2] if rows else 0.0
    for event in events:
        start, end = event.get("start_time_s", previous), event["time_s"]
        previous = end
        if end <= start:
            continue
        samples = [r for r in rows if start < r["time_s"] <= end + 1e-8]
        consecutive, longest = 0, 0
        for row in samples:
            fingers = {
                c["body"].split("_hand_")[1].split("_")[0]
                for c in row["contacts"]
                if "_hand_" in c["body"] and c["normal_force_n"] > 0.1
            }
            lifted = row["object_position_m"][2] > initial_z + 0.02
            consecutive = consecutive + 1 if lifted and len(fingers) >= 2 else 0
            longest = max(longest, consecutive)
        physical_ok = summary["penetration_gate_passed"] and not summary["safety_stop"]
        labels.append(
            {
                "skill": event["tool"],
                "start_time_s": start,
                "end_time_s": end,
                "first_state_frame": int(np.ceil(start * FPS - 1e-7)),
                # Each row's command is at t+50 ms; never label a command from
                # the next tool as part of the preceding skill.
                "last_state_frame_exclusive": max(
                    int(np.ceil(start * FPS - 1e-7)), int(np.floor(end * FPS + 1e-7))
                ),
                "tool_status": event["status"],
                "independent_sustained_lift_contact_s": longest * 0.01,
                "contact_lift_verified": physical_ok and longest * 0.01 >= 0.3,
                "placement_verified": event["tool"] == "release_object" and summary["goal_success"],
                "motor_source": "classical controller, not VLA",
                "interval_origin": "logged start/end"
                if "start_time_s" in event
                else "sequential tool endpoints from legacy recording",
            }
        )
    return labels


def collect(recordings: list[Path], out: Path) -> dict:
    out = external_recording_root(out)
    out.mkdir(parents=True, exist_ok=False)
    episodes = []
    for index, root in enumerate(recordings):
        metadata = json.loads((root / "metadata.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        events = json.loads((root / "tools.json").read_text())
        rows = [json.loads(line) for line in (root / "physics.jsonl").read_text().splitlines()]
        held_out = metadata.get("case", {}).get("split") == "held_out"
        eligible = bool(
            summary["goal_success"]
            and summary["penetration_gate_passed"]
            and not summary["safety_stop"]
            and not held_out
        )
        destination = out / ("candidates" if eligible else "diagnostics") / f"episode_{index:06}"
        destination.parent.mkdir(exist_ok=True)
        dataset = export(root, destination)
        labels = skill_intervals(events, rows, summary)
        (destination / "meta/skill_labels.json").write_text(json.dumps(labels, indent=2))
        episodes.append(
            {
                "dataset": str(destination),
                "source": str(root),
                "case": metadata.get("case"),
                "source_commit": dataset["source_commit"],
                "eligible_imitation_candidate": eligible,
                "held_out": held_out,
                "goal_success": summary["goal_success"],
                "skill_intervals": len(labels),
            }
        )
    result = {
        "episodes": episodes,
        "training_ready": False,
        "candidate_datasets": [e["dataset"] for e in episodes if e["eligible_imitation_candidate"]],
        "readiness_reason": "Export does not establish sufficient coverage or training approval.",
        "held_out_training_excluded": True,
        "normalization_scope": "each candidate's own rows; no cross-split aggregation",
    }
    (out / "collection.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.recording, args.out), indent=2))
