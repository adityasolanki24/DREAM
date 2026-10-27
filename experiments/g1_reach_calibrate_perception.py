"""Fit the handle-target covariance model on DEVELOPMENT episodes (W6 step 2).

Procedure: for every episode of the v1 development set (results/g1_reach_heights_dev/agent) that
produced a target estimate, take the perceived handle position (the estimate the plan was made
from) and the true handle centre (scene truth; evaluation data, used here to FIT the model offline,
never by the agent). Express the error in the frame of the fixture face (normal, lateral, vertical)
and set the systematic term of each axis to its rms error less the mean sensor-model variance on
that axis (the sensor-model part is already in the stored covariance as `cov - bias^2 I`).

The held-out-v1 episodes are reported as a CHECK ONLY (they are development evidence for v2 since
their results have been seen); they never enter the fit.

    python experiments/g1_reach_calibrate_perception.py            # writes the JSON below
"""

from __future__ import annotations

import glob
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/g1_reach_heights_dev/perception_calibration_v2.json"
OLD_BIAS = 0.004  # the isotropic term the v1 estimator added (HANDLE_BIAS_STD)


def load(set_name: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / f"results/g1_reach_heights_{set_name}/agent/*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        tg = d["agent"].get("target") or {}
        if not tg.get("position") or tg.get("fixture") is None:
            continue
        fx = d["truth"]["fixture"]
        # outward normal of the true front face, toward the robot's start
        n = np.array([math.cos(fx["yaw"]), math.sin(fx["yaw"])])
        if n @ -np.array(fx["center"][:2]) < 0:
            n = -n
        frame = np.array([[*n, 0.0], [-n[1], n[0], 0.0], [0.0, 0.0, 1.0]])  # rows: normal, lat, z
        err = frame @ (np.array(tg["position"]) - np.array(d["truth"]["handle_center"]))
        # the stored covariance is sensor + isotropic bias: remove the bias to get the sensor part
        sensor = np.array(tg["covariance"]) - OLD_BIAS**2 * np.eye(3)
        out.append({"id": d["spec"]["id"], "err": err, "frame": frame,
                    "sensor": frame @ sensor @ frame.T})  # fmt: skip
    return out


def main() -> None:
    dev, held = load("dev"), load("heldout")
    err = np.array([r["err"] for r in dev])
    rms = np.sqrt((err**2).mean(0))
    sensor_var = np.mean([np.diag(r["sensor"]) for r in dev], axis=0)
    bias = np.sqrt(np.maximum(rms**2 - sensor_var, 0.0))
    old = np.full(3, OLD_BIAS)
    # coverage of "error within 2 sigma along each axis" per axis, old vs new model
    cover = lambda rows, b: float(  # noqa: E731
        np.mean([all(abs(r["err"][i]) <= 2 * math.sqrt(r["sensor"][i, i] + b[i] ** 2)
                     for i in range(3)) for r in rows])  # fmt: skip
    )
    res = {
        "note": "fitted on v1 DEVELOPMENT episodes only; v1 held-out is a check, not part of it",
        "n_dev": len(dev), "n_check": len(held),
        "frame": "fixture face: [normal (outward), lateral, vertical]",
        "dev_error_mean_m": err.mean(0).tolist(), "dev_error_rms_m": rms.tolist(),
        "mean_sensor_std_m": np.sqrt(sensor_var).tolist(),
        "bias_std_m": bias.tolist(), "old_isotropic_bias_std_m": OLD_BIAS,
        "coverage_2sigma_all_axes": {
            "dev_old": cover(dev, old), "dev_new": cover(dev, bias),
            "check_old": cover(held, old), "check_new": cover(held, bias),
        },
    }  # fmt: skip
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
