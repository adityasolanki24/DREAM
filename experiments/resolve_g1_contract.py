"""Use public named dataset evidence to choose an adaptation, without authorizing REAL_G1.

The dataset is authoritative for itself, not proof of the N1.7 training/deployment contract.
Recorded trajectory checks below are range/FK diagnostics only, never dynamic execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from dream.sim.g1 import G1Env
from experiments.g1_dexterous import write_json
from experiments.g1_recording import digest

EVIDENCE = Path("results/g1_contract_resolution")


def resolve(trajectory: Path, out: Path) -> dict:
    import pyarrow.parquet as pq

    info = json.loads((EVIDENCE / "nvidia_dataset_info.json").read_text())
    modality = json.loads((EVIDENCE / "nvidia_dataset_modality.json").read_text())
    sources = json.loads((EVIDENCE / "sources.json").read_text())
    names = info["features"]["action"]["names"]
    if names != info["features"]["observation.state"]["names"]:
        raise ValueError("dataset state/action joint identities differ")
    env = G1Env()
    if set(names) != set(env.robot.names):
        raise ValueError("dataset names differ from current embodiment")
    mapping = [env.robot.names.index(name) for name in names]
    limits = env.robot.limits[mapping]
    table = pq.read_table(trajectory)
    recorded = np.array(table["action"].to_pylist())
    measured = np.array(table["observation.state"].to_pylist())
    if recorded.shape != measured.shape or recorded.shape[1] != 43:
        raise ValueError("invalid public trajectory")
    recorded_bad = (recorded < limits[:, 0]) | (recorded > limits[:, 1])
    state_bad = (measured < limits[:, 0]) | (measured > limits[:, 1])
    valid = recorded[~recorded_bad.any(axis=1)]
    wrists = []
    # FK only, on a separate diagnostic MjData; never used as controller observations.
    data = mujoco.MjData(env.model)
    for q in valid:
        data.qpos[env.robot.qadr[mapping]] = q
        mujoco.mj_forward(env.model, data)
        wrists.append(
            [data.body(f"{side}_wrist_yaw_link").xpos.copy() for side in ("left", "right")]
        )
    conditional = {}
    raw = Path("results/groot_cuda_openloop/raw/groot/inference/decoded_actions.npz")
    with np.load(raw, allow_pickle=False) as prediction:
        for side in ("left", "right"):
            key = f"{side}_hand"
            start, end = (modality["action"][key][k] for k in ("start", "end"))
            bounds = limits[start:end]
            values = prediction[key][0]
            bad = (values < bounds[:, 0]) | (values > bounds[:, 1])
            conditional[key] = {
                "public_dataset_names": names[start:end],
                "out_of_range_values": int(bad.sum()),
                "scope": "Hypothesis that N1.7 uses this older dataset's slots; unverified",
                "actuation_permitted": False,
            }
    report = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": digest(Path(__file__)),
        "sources": sources,
        "dataset_license": "CC-BY-4.0, NVIDIA GEAR; numerical metadata only in Git",
        "dataset_verified": {
            "joint_names": names,
            "sampling_hz": info["fps"],
            "camera": "ego RealSense RGB 640x480 at 20 FPS per dataset card/info",
            "morphology": "Unitree G1 tri-finger, 29 body + 14 hand coordinates",
            "missing": [
                "exact robot/hand asset revision",
                "camera intrinsics/extrinsics",
                "wrist EEF origin/frame",
                "proof this is the N1.7 REAL_G1 contract",
            ],
        },
        "dream_asset": {
            "commit": "15ebe53f4cd77e7d745ae83b9d7ff90e685e8998",
            "file": "g1_29dof_with_hand_rev_1_0.xml",
            "wrist": "4010, rev_1_0",
            "all_dataset_names_present": True,
            "exact_training_asset_match_verified": False,
            "support": "fixed pelvis; no locomotion or whole-body base command adapter",
            "joints": [
                {
                    "name": name,
                    "axis": env.model.jnt_axis[j].tolist(),
                    "limits_rad": env.robot.limits[i].tolist(),
                    "body": env.model.body(env.model.jnt_bodyid[j]).name,
                }
                for i, (name, j) in enumerate(zip(env.robot.names, env.robot.joints, strict=True))
            ],
        },
        "public_trajectory_diagnostic": {
            "path": str(trajectory),
            "sha256": digest(trajectory),
            "rows": len(recorded),
            "action_values_outside_dream_ranges": int(recorded_bad.sum()),
            "measured_values_outside_dream_ranges": int(state_bad.sum()),
            "range_valid_action_frames_with_finite_fk": len(wrists),
            "finite_wrist_positions": bool(wrists) and bool(np.isfinite(wrists).all()),
            "physics_actions_executed": 0,
            "note": "Ranges and finite FK cannot verify real training frames, dynamics or cameras.",
        },
        "original_predictions_under_dataset_order": conditional,
        "original_predictions_sha256": digest(raw),
        "real_g1_actuation_permitted": False,
        "alternatives": [
            {
                "path": "GR00T NEW_EMBODIMENT on DREAM's supported G1",
                "chosen": True,
                "reason": "Official custom-embodiment recipe, existing model runtime/weights, "
                "current robot/safety and typed tools retained. Own named 31D absolute "
                "joint references, external camera and 20 Hz recordings remove unknown "
                "EEF/hand/base contracts. Requires diverse successful demos and fine-tuning.",
            },
            {
                "path": "UNITREE_G1_SONIC",
                "chosen": False,
                "reason": "Different 64D motion-token/WBC deployment, separate gripper "
                "and finetuned policy. Replaces fixed-pelvis control and needs locomotion "
                "validation; much larger integration than the supported manipulation slice.",
            },
            {
                "path": "SmolVLA",
                "chosen": False,
                "reason": "Official docs require custom data/fine-tuning; no verified "
                "drop-in G1 Dex3 checkpoint identified. New runtime and weights add effort "
                "without resolving the need for a named physical dataset.",
            },
        ],
        "next_gates": [
            "Validate exported LeRobot schema and action/state/video alignment on real episodes",
            "Collect varied independently successful motor-skill demonstrations, retain failures",
            "Freeze split/gates; obtain a fresh itemised GPU estimate and explicit cost approval",
            "Fine-tune existing checkpoint in isolated pinned environment, no silent fallback",
            "Fresh held-out offline action checks, then bounded safety-limited recorded execution",
            "Only then expose the validated bounded motor skill to existing Qwen tool interface",
        ],
        "model_calls": 0,
        "learned_physical_milestone_achieved": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    write_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    resolve(args.trajectory, args.out)
