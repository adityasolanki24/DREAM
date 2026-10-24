"""Read-only 200 Hz joint-stop audit; no actor, simulator, network or cloud access.

The derivative of sampled positions and the reported solver velocity are different
measurements. Inward instantaneous velocity does not by itself establish recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inspect(stand: Path, joint: str = "Left_Knee_Pitch") -> dict:
    articulation = json.loads((stand / "articulation.json").read_text())
    index = articulation["joint_names"].index(joint)
    lo, hi = articulation["limits_rad"][index]
    rows = [json.loads(s) for s in (stand / "physics_substeps.jsonl").read_text().splitlines()]
    samples = []
    for row in rows:
        q = row["proprioception"]["q"][index]
        dq = row["proprioception"]["dq"][index]
        residual = max(lo - q, q - hi, 0)
        inward = (q < lo and dq > 0) or (q > hi and dq < 0)
        samples.append(
            {
                "time_s": row["time_s"],
                "q_rad": q,
                "dq_rad_s": dq,
                "excess_rad": residual,
                "velocity_points_inward": inward,
                "target_rad": row["joint_target_rad"][index],
            }
        )
    violations = [r for r in samples if r["excess_rad"] > 0]
    if len(samples) >= 2:
        dt = samples[-1]["time_s"] - samples[-2]["time_s"]
        final_slope = (samples[-1]["q_rad"] - samples[-2]["q_rad"]) / dt
    else:
        final_slope = None
    return {
        "joint": joint,
        "limits_rad": [lo, hi],
        "samples": samples,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (stand / "articulation.json", stand / "physics_substeps.jsonl")
        },
        "violation_samples": len(violations),
        "maximum_excess_rad": max((r["excess_rad"] for r in violations), default=0),
        "last_sample_velocity_inward": samples[-1]["velocity_points_inward"],
        "final_position_difference_velocity_rad_s": final_slope,
        "residual_grew_on_last_sample": len(samples) > 1
        and samples[-1]["excess_rad"] > samples[-2]["excess_rad"],
        "contact_force_and_realized_motor_effort_available": False,
        "correlation_limit": "Targets are recorded; contact forces and realized effort are absent, "
        "so causal contact/drive attribution is unsupported.",
        "conclusion": "Mixed residual trend followed by largest excursion; "
        "recording ends at rejection. "
        "No evidence of subsequent recovery or stable walking.",
        "runtime_change": "none; exact measured-state guard retained",
        "proposed_policy": {
            "status": "not enabled; no independently calibrated nonzero residual budget",
            "requirements": [
                "Preserve raw q/dq and mechanical limits; never clip measurements "
                "or expand the simulated joint travel.",
                "Stop immediately on nonfinite state, overspeed, excessive tilt/effort "
                "or an excursion beyond an independently justified physical envelope.",
                "Within a calibrated residual envelope, monitor penetration growth, "
                "outward velocity "
                "and elapsed dwell every physics step; inward velocity alone never authorizes "
                "continuation.",
                "Calibrate envelope and dwell against pinned solver settings, contact/effort "
                "telemetry and independent recovery tests before permitting any nonzero residual.",
                "Keep host scoring independent and use the same declared contract; "
                "logic tests are not evidence of walking.",
            ],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stand", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.stand)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
