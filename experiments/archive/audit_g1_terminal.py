"""Inspect saved terminal contacts and joint margins without stepping physics."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from experiments.g1_dexterous import write_json


def audit(root: Path) -> dict:
    model = mujoco.MjModel.from_binary_path(str(root / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(root / "terminal.npz", allow_pickle=False) as arrays:
        mujoco.mj_setState(
            model, data, arrays["integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
    mujoco.mj_forward(model, data)
    contacts = []
    for index, contact in enumerate(data.contact):
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        if any(name.startswith("left_") for name in bodies):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            contacts.append(
                {
                    "bodies": bodies,
                    "geoms": [model.geom(int(g)).name for g in contact.geom],
                    "distance_m": float(contact.dist),
                    "normal_force_n": float(force[0]),
                }
            )
    joints = []
    for index in range(model.njnt):
        name = model.joint(index).name
        if name.startswith("left_") and model.jnt_limited[index]:
            position = float(data.qpos[model.jnt_qposadr[index]])
            lower, upper = model.jnt_range[index]
            joints.append(
                {
                    "joint": name,
                    "position_rad": position,
                    "limit_margin_rad": float(min(position - lower, upper - position)),
                }
            )
    return {
        "physics_reexecuted": False,
        "terminal_s": float(data.time),
        "contacts": contacts,
        "joints": joints,
        "scope": "Terminal evidence only; does not identify the entire trajectory's cause.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        str(path.parent.relative_to(args.root)): audit(path.parent)
        for path in args.root.rglob("terminal.npz")
    }
    write_json(args.out, result)
