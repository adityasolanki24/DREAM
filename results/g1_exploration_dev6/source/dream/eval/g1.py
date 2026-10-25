"""Independent G1 contact/object scoring. Ground truth never flows to the controller."""

from __future__ import annotations

import mujoco
import numpy as np


def object_evidence(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    geom = model.geom("test_object_geom").id
    contacts = []
    for i, contact in enumerate(data.contact):
        if geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == geom else contact.geom1
        body = model.body(model.geom_bodyid[other]).name
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append(
            {"body": body, "normal_force_n": float(force[0]), "distance_m": float(contact.dist)}
        )
    return {
        "object_position_m": data.body("test_object").xpos.copy().tolist(),
        "object_quaternion_wxyz": data.body("test_object").xquat.copy().tolist(),
        "contacts": contacts,
    }


def score_episode(rows: list[dict], gates: dict) -> dict:
    """Score unsupported lift separately from actuation/contact, including failed cases."""
    start = np.array(rows[0]["object_position_m"])

    def chains(row):
        return {
            c["body"].split("_hand_")[1].split("_")[0]
            for c in row["contacts"]
            if "_hand_" in c["body"] and c["normal_force_n"] > gates["hand_contact_force_n"]
        }

    hold = [r for r in rows if r["phase"] == "hold"]
    fraction = (
        float(
            np.mean(
                [
                    r["object_position_m"][2] - start[2] >= gates["lift_above_start_m"]
                    and len(chains(r)) >= gates["hold_min_distinct_finger_chains"]
                    and not any(
                        c["body"] == "world" and c["normal_force_n"] > gates["hand_contact_force_n"]
                        for c in r["contacts"]
                    )
                    for r in hold
                ]
            )
        )
        if hold
        else 0.0
    )
    displacement = max(
        float(np.linalg.norm(np.array(r["object_position_m"]) - start)) for r in rows
    )
    held_height = (
        float(np.mean([r["object_position_m"][2] for r in hold])) if hold else float(start[2])
    )
    release_drop = held_height - rows[-1]["object_position_m"][2]
    stopped = any(r["stop_reason"] for r in rows)
    return {
        "physics_executed": len(rows) > 1 and rows[-1]["time_s"] > rows[0]["time_s"],
        "contact_observed": any(chains(r) for r in rows),
        "max_object_displacement_m": displacement,
        "object_moved": displacement > gates["object_displacement_m"],
        "hold_fraction_lifted_with_two_fingers": fraction,
        "minimum_hold_lift_m": min((r["object_position_m"][2] - start[2] for r in hold), default=0),
        "maximum_hand_contact_penetration_m": max(
            (
                -c.get("distance_m", 0)
                for r in rows
                for c in r["contacts"]
                if "_hand_" in c["body"] or "wrist" in c["body"]
            ),
            default=0,
        ),
        "grasp_success": fraction >= gates["hold_min_fraction"] and not stopped,
        "release_drop_m": float(release_drop),
        "release_drop_observed": release_drop >= gates["release_drop_m"],
        "termination": rows[-1]["stop_reason"] or "completed bounded experiment",
        "simulated_s": rows[-1]["time_s"],
        "scope": "supported-base scripted commissioning; "
        "no perception, autonomous reaching or balance; uncalibrated soft mesh contacts",
    }
