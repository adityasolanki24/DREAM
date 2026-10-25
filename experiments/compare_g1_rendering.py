"""Reprocess saved RGB-D only to isolate rendering inputs from perception arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np

from dream.perception.sphere import detect_sphere
from dream.types import CameraCalibration, RGBDFrame
from experiments.g1_dexterous import write_json


def estimate(path: Path):
    with np.load(path, allow_pickle=False) as a:
        rgb = a["rgb"]
        cal = CameraCalibration(
            rgb.shape[1], rgb.shape[0], float(a["fovy_deg"]), a["position"], a["rotation"]
        )
        return detect_sphere(RGBDFrame(float(a["time_s"]), rgb, a["depth"], a["self_mask"], cal))


def compare(cloud: Path, local: Path) -> dict:
    result = {}
    for case in ("free_shift", "free_heavier"):
        c, local_case = cloud / "deterministic" / case, local / case
        deviations = []
        for saved in json.loads((c / "frames.json").read_text()):
            actual = estimate(c / "frames" / saved["file"])
            if actual is not None and saved["estimate"] is not None:
                deviations.append(
                    float(np.linalg.norm(actual.position - saved["estimate"]["position_m"]))
                )
        cp, lp = c / "frames/0000.npz", local_case / "frames/0000.npz"
        with np.load(cp) as a, np.load(lp) as b:
            rgb_difference = np.any(a["rgb"] != b["rgb"], axis=-1)
            valid = np.isfinite(a["depth"]) & np.isfinite(b["depth"])
            depth_difference = a["depth"][valid] - b["depth"][valid]
        cloud_rows = [json.loads(s) for s in (c / "physics.jsonl").read_text().splitlines()]
        local_rows = [
            json.loads(s) for s in (local_case / "physics.jsonl").read_text().splitlines()
        ]
        first_divergence = next(
            (
                {"time_s": a["time_s"], "tool": a["tool"], "object_difference_m": float(delta)}
                for a, b in zip(cloud_rows, local_rows, strict=False)
                if a["physics_step"] == b["physics_step"]
                and (
                    delta := np.linalg.norm(
                        np.array(a["object_position_m"]) - b["object_position_m"]
                    )
                )
                > 1e-8
            ),
            None,
        )
        result[case] = {
            "saved_cloud_frames_reprocessed": len(deviations),
            "max_local_reprocessing_position_difference_m": max(deviations, default=None),
            "initial_rgb_pixels_differing": int(np.count_nonzero(rgb_difference)),
            "shared_valid_depth_pixels": int(np.count_nonzero(valid)),
            "initial_depth_rms_difference_m": float(np.sqrt(np.mean(depth_difference**2))),
            "initial_estimated_centre_difference_m": float(
                np.linalg.norm(estimate(cp).position - estimate(lp).position)
            ),
            "first_object_difference_above_10nm": first_divergence,
            "scope": "Same initial scene/seed; different renderer/platform inputs. "
            "Not proof of a sole cause or GPU physics defect.",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.cloud, args.local)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
