"""Numerical contact sensitivity, not hardware calibration; preserve upstream baseline.

Only a commissioning comparison: known scene placement is allowed here, not in the
exploration controller. Positive solref = (time constant, critical damping). The 5 ms
candidate exceeds MuJoCo's documented 2 dt stability floor at both tested resolutions.
"""

import argparse
import json
from pathlib import Path

from dream.eval.g1 import object_evidence, score_episode
from dream.sim.g1 import G1Env, build_g1
from experiments.g1_dexterous import write_json


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cfg = {
        "cases": [[0.025, 0.05], [0.020, 0.03]],
        "dt_s": [0.001, 0.0005],
        "timeconst_s": [0.02, 0.005],
        "penetration_gate_m": 0.001,
        "source": "https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters",
        "classification": "numerical resolution study; no measured material calibration",
    }
    write_json(out / "config.json", cfg)
    gates = json.loads(Path("experiments/configs/g1_dex3_dev.json").read_text())["gates"]
    summaries = {}
    for radius, mass in cfg["cases"]:
        for dt in cfg["dt_s"]:
            for tau in cfg["timeconst_s"]:
                name = f"radius{radius}_dt{dt}_tau{tau}"
                model = build_g1(
                    (0.33127486, 0.10665347, 0.88823068), radius, mass, contact_timeconst=tau
                )
                model.opt.timestep = dt  # episode setup, before controller or physics
                env = G1Env(model)
                rows = []
                for phase, duration in (("open", 0.5), ("close", 2), ("lift", 2), ("hold", 1)):
                    if phase == "close":
                        env.robot.set_hand("left", 1)
                    if phase == "lift":
                        env.robot.move_arm("left", [0, 0, 0, -0.35, 0, 0, 0])
                    for i in range(round(duration / dt)):
                        env.step()
                        if i % round(0.01 / dt) == 0 or env.robot.stop_reason:
                            rows.append(
                                {
                                    "phase": phase,
                                    "time_s": env.data.time,
                                    "stop_reason": env.robot.stop_reason,
                                    **object_evidence(model, env.data),
                                }
                            )
                        if env.robot.stop_reason:
                            break
                    if env.robot.stop_reason:
                        break
                summary = score_episode(rows, gates)
                summary["penetration_gate_passed"] = (
                    summary["maximum_hand_contact_penetration_m"] <= cfg["penetration_gate_m"]
                )
                summaries[name] = summary
                write_json(out / f"{name}.json", {"summary": summary, "trace": rows})
    write_json(out / "summary.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, summary in run(args.out).items():
        print(
            name,
            "penetration_mm",
            round(summary["maximum_hand_contact_penetration_m"] * 1000, 3),
            "lift",
            summary["grasp_success"],
        )
