"""Rank tabletop pinches from predicted Dex3 contacts and support clearance.

The isolated hand sweep is geometric. It does not establish that the arm
can reach a pose or that a physical grasp will hold under load.
"""

from dataclasses import dataclass
from itertools import product

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PinchCandidate:
    offset: np.ndarray
    rotation: np.ndarray
    score: float
    clearance_m: float
    support_links: int
    closure_fraction: float


def rank_pinches(
    model: mujoco.MjModel,
    reference: np.ndarray,
    object_pose: np.ndarray,
    half_extents: np.ndarray,
    hand_joints: np.ndarray,
    opening: np.ndarray,
    closing: np.ndarray,
    tool_offset: np.ndarray,
    clearance_m: float,
    *,
    object_geom: int,
    support_geom: int,
) -> list[PinchCandidate]:
    """Use the observed object pose and mesh geometry in a separate kinematic state."""
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    wrist = model.body("right_wrist_yaw_link").id
    object_body = model.body(model.geom_bodyid[object_geom])
    object_qpos = int(model.jnt_qposadr[object_body.jntadr[0]])
    hand = {
        g: model.body(model.geom_bodyid[g]).name
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.body(model.geom_bodyid[g]).name.startswith(("right_hand_", "right_wrist_yaw"))
    }
    mujoco.mj_kinematics(model, data)
    wrist_to_world = data.xmat[wrist].reshape(3, 3).copy()
    root_to_wrist = data.xpos[wrist].copy() - reference[:3]
    root_rot = Rotation.from_quat(reference[[4, 5, 6, 3]]).as_matrix()
    centre = object_pose[:3]
    width, height_extent = half_extents[1:]
    candidates = []

    def evaluate(rot, offset):
        change = rot @ wrist_to_world.T
        root_quat = Rotation.from_matrix(change @ root_rot).as_quat()
        data.qpos[:] = reference
        data.qpos[:3] = centre + offset - rot @ tool_offset - change @ root_to_wrist
        data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
        data.qpos[object_qpos : object_qpos + 7] = object_pose
        closest_table = 0.10
        best = None
        for fraction in np.linspace(0.0, 1.0, 17):
            data.qpos[hand_joints] = opening + fraction * (closing - opening)
            mujoco.mj_forward(model, data)
            closest_table = min(
                closest_table,
                *(mujoco.mj_geomDistance(model, data, g, support_geom, 0.1, None) for g in hand),
            )
            if closest_table < clearance_m:
                best = None
                break
            touches = []
            for contact in data.contact:
                other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
                if (
                    object_geom in (contact.geom1, contact.geom2)
                    and other in hand
                    and contact.dist < 0
                ):
                    normal = contact.frame[:3] * (1 if contact.geom2 == object_geom else -1)
                    touches.append((hand[other], contact.pos.copy(), normal))
            opposing = [
                entry
                for entry in touches
                if any(digit in entry[0] for digit in ("hand_index_", "hand_middle_"))
            ]
            support = len({name for name, _, _ in opposing})
            for name, thumb_pos, thumb_normal in touches:
                if "hand_thumb_" not in name:
                    continue
                for _, pad_pos, pad_normal in opposing:
                    separation = pad_pos - thumb_pos
                    length = np.linalg.norm(separation)
                    if length < 1e-6:
                        continue
                    axis = separation / length
                    opposition = min(thumb_normal @ axis, -pad_normal @ axis)
                    if opposition <= 0:
                        continue
                    moment = np.cross(centre - (thumb_pos + pad_pos) / 2, [0.0, 0.0, -1.0])
                    # Prefer opposition, a small gravity moment around the
                    # pinch axis, additional contact links, and closure reserve.
                    score = (
                        opposition
                        - abs(axis @ moment) / width
                        + 0.1 * min(support, 3)
                        - 0.1 * fraction
                    )
                    if best is None or score > best[0]:
                        best = (float(score), support, float(fraction))
        if best is None:
            return None
        score, support, fraction = best
        return PinchCandidate(offset, rot, score, closest_table, support, fraction)

    # Search around the observed box. The signed offsets scale with its width.
    for roll, pitch, height, lateral in product(
        (-15, 0, 15), (0, 15, 30), (0.5, 0.75, 1.0, 1.25), (-0.2, 0.0, 0.2)
    ):
        rot = Rotation.from_euler("xy", [roll, pitch], degrees=True).as_matrix()
        offset = np.array([0.0, lateral * width, height * height_extent])
        candidate = evaluate(rot, offset)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
