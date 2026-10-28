"""Plot independent saved physical evidence; no new simulation or model invocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(root: Path, out: Path, case: str = "free_heavier") -> dict:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    diagnostic = {}
    for backend, color in (("qwen", "#b54733"), ("deterministic", "#22634d")):
        folder = root / backend / case
        rows = [json.loads(line) for line in (folder / "physics.jsonl").read_text().splitlines()]
        t = np.array([r["time_s"] for r in rows])
        position = np.array([r["object_position_m"] for r in rows])
        hand = np.array([r["grasp_frame_position_m"] for r in rows])
        distance = np.linalg.norm(position - hand, axis=1)
        chains = [
            len(
                {
                    c["body"].split("_hand_")[1].split("_")[0]
                    for c in r["contacts"]
                    if "_hand_" in c["body"] and c["normal_force_n"] > 0.1
                }
            )
            for r in rows
        ]
        label = "Qwen: failed" if backend == "qwen" else "Deterministic: accepted placement"
        axes[0].plot(t, position[:, 2], color=color, label=label)
        axes[1].plot(t, distance, color=color)
        axes[2].step(t, chains, where="post", color=color)
        phases = []
        previous = None
        for r in rows:
            if r["tool"] != previous:
                phases.append({"tool": r["tool"], "time_s": r["time_s"]})
                previous = r["tool"]
        moving = np.array([r["tool"] == "move_object" for r in rows])
        selected = np.flatnonzero(moving & (np.array(chains) == 0))
        diagnostic[backend] = {
            "phase_starts": phases,
            "final_object_position_m": position[-1].tolist(),
            "final_hand_object_distance_m": float(distance[-1]),
            "first_transfer_sample_without_finger_contact_s": float(t[selected[0]])
            if len(selected)
            else None,
            "minimum_object_height_during_transfer_m": float(position[moving, 2].min())
            if moving.any()
            else None,
            "scope": "Independent 100 Hz saved physics samples; no control or perception input",
        }
    axes[0].set_ylabel("Object centre height (m)")
    axes[1].set_ylabel("Hand–object distance (m)")
    axes[2].set_ylabel("Contacting finger chains")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_xlabel("Simulation time (s); physics pauses during model inference")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Same heavier-object scene: real Qwen decisions vs deterministic planner")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {
        "case": case,
        "traces": diagnostic,
        "limitation": "Transfer contact loss is observed, but its initiating mechanical cause "
        "is not established. A tool-local grasp report is not an accepted whole-episode grasp.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    with args.diagnostic.open("x", encoding="utf-8") as stream:
        json.dump(plot(args.root, args.out), stream, indent=2)
        stream.write("\n")
