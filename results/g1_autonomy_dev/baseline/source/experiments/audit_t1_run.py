"""Summarize an already retrieved Isaac run; no inference or cloud resource creation."""

import argparse
import json
from pathlib import Path

from cloud.launch import verify_results


def audit(out: Path) -> dict:
    verify_results(out)
    cases = {}
    for name in ("stand", "forward", "left"):
        folder = out / name
        preflight = json.loads((folder / "preflight.json").read_text())
        trace = json.loads((folder / "partial_trace.json").read_text())
        cases[name] = {
            "code_sha": preflight["code_sha"],
            "recorded_steps": len(trace),
            "episode_present": (folder / "episode.json").is_file(),
            "calibration_present": (folder / "sensor_calibration.json").is_file(),
            "failure_present": (folder / "failure.json").is_file(),
            "stage_log_present": (folder / "stages.jsonl").is_file(),
        }
    return {
        "stable_checksums_verified": True,
        "guest_exit_code": (out / "EXIT_CODE").read_text().strip(),
        "started_utc": (out / "started_utc.txt").read_text().strip(),
        "finished_utc": (out / "finished_utc.txt").read_text().strip(),
        "cases": cases,
        "walking_validated": False,
        "interpretation": (
            "No episode or recorded steps establishes no walking evidence; an empty trace "
            "cannot locate failure within initialization/warmup/capture. "
            "Guest exit zero alone is insufficient."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    args.summary.write_text(json.dumps(audit(args.out), indent=2) + "\n")
