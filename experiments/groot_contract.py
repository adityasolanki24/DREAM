"""Reconstruct saved REAL_G1 decoding offline; this is not an actuator adapter.

Equations follow the pinned NVIDIA sources recorded in the executable JSON contract.
The only clipping below reproduces the upstream decoder, never physical limit clipping.
Dataset slot semantics cannot be inferred from statistical ranges or dimension matches.
No new inference, simulation episode, controller command or physical transfer is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.audit_groot_actions import audit
from experiments.g1_dexterous import write_json
from experiments.groot_openloop import ACTION_DIMS, validate_actions

CONTRACT = Path(__file__).with_name("configs") / "groot_action_contract.json"


def require_actuation_contract() -> None:
    """Refuse the current, unverified physical mapping before any actuator is called.

    Deliberately accepts no caller-supplied 'verified' booleans. Removing this refusal
    requires a reviewed named physical contract and new evidence, not a CLI override.
    """
    contract = json.loads(CONTRACT.read_text())
    raise ValueError("REAL_G1 actuation blocked: " + "; ".join(contract["blockers"]))


def rotation_rows(values: np.ndarray) -> np.ndarray:
    """Reproduce pinned pose.py's row-based ROT6D Gram-Schmidt convention."""
    rows = np.asarray(values, np.float64).reshape(2, 3).copy()
    norms = np.linalg.norm(rows[0])
    if norms <= 0:
        raise ValueError("degenerate ROT6D")
    rows[0] /= norms
    rows[1] -= np.dot(rows[0], rows[1]) * rows[0]
    norms = np.linalg.norm(rows[1])
    if norms <= 0:
        raise ValueError("degenerate ROT6D")
    rows[1] /= norms
    return np.vstack((rows, np.cross(*rows)))


def decode_for_audit(normalized: np.ndarray, state: dict, contract: dict) -> dict:
    """Reconstruct absolute streams from untouched normalized predictions and raw state.

    This reproduces the current pinned min/max decoder (with percentile bounds for
    absolute actions and relative min/max overrides). It does not repair the input.
    Relative samples all reference the final observation, never the preceding sample.
    """
    if normalized.shape != (1, 40, 132) or not np.isfinite(normalized).all():
        raise ValueError("invalid normalized REAL_G1 tensor")
    options = contract["processor_options"]
    if not (
        options["use_percentiles"]
        and options["use_relative_action"]
        and not options["use_mean_std"]
    ):
        raise ValueError("unsupported pinned normalization options")
    result = {}
    start = 0
    modality = contract["modality"]["action"]
    for key, cfg in zip(modality["modality_keys"], modality["action_configs"], strict=True):
        dimension = ACTION_DIMS[key]
        relative = cfg["rep"] == "RELATIVE"
        stats = contract["statistics"]["relative_action" if relative else "action"][key]
        low, high = ("min", "max") if relative else ("q01", "q99")
        lower, upper = np.asarray(stats[low]), np.asarray(stats[high])
        values = normalized[..., start : start + dimension]
        # NVIDIA utils.py unnormalize_values_minmax clips normalized space unconditionally.
        decoded = (np.clip(values, -1, 1) + 1) / 2 * (upper - lower) + lower
        if relative:
            reference = np.asarray(state[cfg["state_key"]])[0, -1]
            if cfg["type"] == "EEF":
                rotation = rotation_rows(reference[3:])
                absolute = []
                for pose in decoded[0]:
                    translation = reference[:3] + rotation @ pose[:3]
                    orientation = rotation @ rotation_rows(pose[3:])
                    absolute.append(np.concatenate((translation, orientation[:2].ravel())))
                decoded = np.array(absolute)[None]
            else:
                decoded = decoded + reference
        result[key] = decoded.astype(np.float32)
        start += dimension
    validate_actions(result)
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_audit(run: Path, out: Path) -> dict:
    """Save exhaustive conditional violations and representation/physical evidence separately."""
    contract = json.loads(CONTRACT.read_text())
    raw_paths = [
        run / "inference/normalized_actions.npy",
        run / "inference/decoded_actions.npz",
        run / "sensors/observation.npz",
        run / "sensors/contract.json",
    ]
    before = {str(p): sha256(p) for p in raw_paths}
    manifest = run.parent / "MANIFEST.sha256"
    verified = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (manifest.parent / name).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or sha256(path) != digest:
            raise ValueError("original raw manifest mismatch: " + name)
        verified[name] = digest
    normalized = np.load(run / "inference/normalized_actions.npy", allow_pickle=False)
    with np.load(run / "sensors/observation.npz", allow_pickle=False) as observation:
        state = dict(observation)
    with np.load(run / "inference/decoded_actions.npz", allow_pickle=False) as saved:
        actions = dict(saved)
    reconstructed = decode_for_audit(normalized, state, contract)
    mechanical = audit(run)
    errors = {key: float(np.max(np.abs(reconstructed[key] - actions[key]))) for key in actions}
    streams = {}
    start = 0
    modality = contract["modality"]["action"]
    for key, cfg in zip(modality["modality_keys"], modality["action_configs"], strict=True):
        end = start + ACTION_DIMS[key]
        streams[key] = {
            "normalized_slice_start_inclusive": start,
            "normalized_slice_end_exclusive": end,
            "representation_before_decode": cfg,
            "decoded_representation": "absolute in supplied state's convention",
            "normalized_values_clipped_by_upstream": int(
                np.count_nonzero(np.abs(normalized[..., start:end]) > 1)
            ),
            "normalization": "relative min/max" if cfg["rep"] == "RELATIVE" else "action q01/q99",
            "physical_slot_mapping_verified": False,
        }
        start = end
    violations = []
    for key in ("left_hand", "right_hand"):
        diagnostic = mechanical["conditional_joint_diagnostics"][key]
        limits = np.asarray(diagnostic["mechanical_limits_rad"])
        values = actions[key][0]
        for step, slot in np.argwhere((values < limits[:, 0]) | (values > limits[:, 1])):
            violations.append(
                {
                    "stream": key,
                    "step": int(step),
                    "slot": int(slot),
                    "provisional_dream_joint": diagnostic["provisional_joint_names"][slot],
                    "decoded_value": float(values[step, slot]),
                    "conditional_limit_rad": limits[slot].tolist(),
                    "normalized_value": float(
                        normalized[0, step, streams[key]["normalized_slice_start_inclusive"] + slot]
                    ),
                }
            )
    state_ranges = {}
    for key, values in state.items():
        if key == "video":
            continue
        stats = contract["statistics"]["state"][key]
        reference = values[0, -1]
        state_ranges[key] = {
            "outside_training_min_max_slots": np.flatnonzero(
                (reference < stats["min"]) | (reference > stats["max"])
            ).tolist(),
            "outside_training_q01_q99_slots": np.flatnonzero(
                (reference < stats["q01"]) | (reference > stats["q99"])
            ).tolist(),
            "scope": "supplied provisional order; state percentile normalization clips outliers",
        }
    report = {
        "scope": "offline representation audit only; original inference remains provisional",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_sources_sha256": {str(p): sha256(p) for p in (Path(__file__), CONTRACT)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_prediction_seed": json.loads((run / "inference/input.json").read_text())["seed"],
        "contract_sha256": sha256(CONTRACT),
        "raw_sha256": before,
        "original_manifest_files_verified": len(verified),
        "upstream_metadata_sha256": contract["metadata_sha256"],
        "streams": streams,
        "padding_dimensions_ignored_by_upstream": 132 - start,
        "decoder_max_abs_errors": errors,
        "state_range_diagnostics": state_ranges,
        "conditional_hand_violations": violations,
        "conditional_hand_violation_count": len(violations),
        "hand_action_quantiles": {
            key: {q: contract["statistics"]["action"][key][q] for q in ("q01", "q99")}
            for key in ("left_hand", "right_hand")
        },
        "blockers": contract["blockers"],
        "units": contract["physical_units"],
        "hand_hypothesis": contract["hand_hypothesis"],
        "recording": contract["recording"],
        "gates": {
            "decoder_reconstruction": max(errors.values())
            <= contract["acceptance"]["decoder_max_abs_error"],
            "original_bytes_preserved": before == {str(p): sha256(p) for p in raw_paths},
            "hand_mapping_verified": False,
            "physical_execution_authorized": False,
        },
        "actuation_permitted": False,
        "new_model_calls": 0,
        "physical_actions_executed": 0,
        "cloud_resources_allocated": False,
        "next_required_evidence": (
            "Pinned REAL_G1 training/deployment metadata with named slots, robot/hand asset, "
            "units/signs/zeros, EEF frames, camera calibration, rate and command arbitration. "
            "A different embodiment's joint list is not evidence for this checkpoint. "
            "Correctly encoded observations require fresh inference before physical validation."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "report.json", report)
    write_json(out / "conditional_ranges.json", mechanical)
    write_json(out / "contract.json", contract)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.run, args.out)
    print(json.dumps({"gates": report["gates"], "actuation_permitted": False}))
