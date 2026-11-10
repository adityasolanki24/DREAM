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
    thumb_bias_rad: float


@dataclass(frozen=True)
class Rehearsal:
    rank: int
    opposed: bool
    minimum_hold_lift_m: float
    maximum_slip_m: float | None


def regulate_closure(
    fraction: float, force: float, displacement: float, width: float, dt: float, grip_force: float
) -> tuple[float, float]:
    """Use slip to request more preload, bounded by the live hand force guard."""
    goal = grip_force * (1.0 + min(2.0, displacement / (0.1 * width)))
    if force < goal:
        fraction = min(1.0, fraction + 0.5 * dt)
    elif force > 2 * goal:
        fraction = max(0.0, fraction - 0.25 * dt)
    return fraction, goal


def rehearse_pinches(
    model: mujoco.MjModel,
    object_pose: np.ndarray,
    candidates: list[tuple[int, PinchCandidate]],
    hand_joints: tuple[str, ...],
    opening: np.ndarray,
    closing: np.ndarray,
    tool_offset: np.ndarray,
    *,
    object_width: float,
    control_period: float,
    grip_force: float,
    grip_dwell: float,
    lift_height: float,
    hold_duration: float,
) -> list[Rehearsal]:
    """Screen under gravity with a separate stage and nominal material properties."""
    data = mujoco.MjData(model)
    stage = model.body("rehearsal_stage").id
    wrist = model.body("right_wrist_yaw_link").id
    lift_actuator = model.actuator("rehearsal_lift").id
    lift_qpos = int(model.joint("rehearsal_lift").qposadr[0])
    object_geom = model.geom("box").id
    object_joint = model.joint(model.body("box").jntadr[0])
    object_qpos = int(object_joint.qposadr[0])
    actuators = [model.actuator(name).id for name in hand_joints]
    finger_positions = [int(model.joint(name).qposadr[0]) for name in hand_joints]
    step_count = round(control_period / model.opt.timestep)
    wrench = np.zeros(6)
    results = []
    model.actuator_biasprm[lift_actuator, 0] = -model.body_subtreemass[stage] * model.opt.gravity[2]

    def step(fraction, lift, end):
        data.ctrl[actuators] = opening + fraction * (end - opening)
        data.ctrl[lift_actuator] = lift
        mujoco.mj_step(model, data, nstep=step_count)
        thumb_force = opposing_force = 0.0
        for index, contact in enumerate(data.contact):
            if object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
            link = model.body(model.geom_bodyid[other]).name
            mujoco.mj_contactForce(model, data, index, wrench)
            if "hand_thumb_2" in link:
                thumb_force += wrench[0]
            elif "hand_index_" in link or "hand_middle_" in link:
                opposing_force += wrench[0]
        return min(thumb_force, opposing_force)

    for rank, candidate in candidates:
        mujoco.mj_resetData(model, data)
        model.body_pos[stage] = (
            object_pose[:3] + candidate.offset - candidate.rotation @ tool_offset
        )
        quat = Rotation.from_matrix(candidate.rotation).as_quat()
        model.body_quat[wrist] = quat[[3, 0, 1, 2]]
        mujoco.mj_setConst(model, data)
        data.qpos[finger_positions] = opening
        data.qpos[object_qpos : object_qpos + 7] = object_pose
        fraction = 0.0
        dwell = 0
        end = closing.copy()
        end[0] += candidate.thumb_bias_rad
        for _ in range(round(1.5 / control_period)):
            if step(fraction, 0.0, end) >= grip_force:
                dwell += 1
                if dwell >= round(grip_dwell / control_period):
                    break
            else:
                dwell = 0
                fraction = min(1.0, fraction + control_period)
        else:
            results.append(Rehearsal(rank, False, 0.0, None))
            continue

        grasped_position = data.qpos[object_qpos : object_qpos + 3].copy()
        start_slide = data.qpos[lift_qpos]
        # Average lift rate of the live probe on the commissioning controller.
        lift_steps = round(lift_height / (0.03 * control_period))
        min_height = np.inf
        max_slip = 0.0
        for tick in range(lift_steps + round(hold_duration / control_period)):
            lift = min((tick + 1) / lift_steps, 1.0) * lift_height
            weakest = step(fraction, lift, end)
            position = data.qpos[object_qpos : object_qpos + 3]
            stage_rise = data.qpos[lift_qpos] - start_slide
            relative = position - grasped_position - [0.0, 0.0, stage_rise]
            slip = float(np.linalg.norm(relative))
            max_slip = max(max_slip, slip)
            fraction, _ = regulate_closure(
                fraction, weakest, slip, object_width, control_period, grip_force
            )
            if tick >= lift_steps:
                min_height = min(min_height, float(position[2] - object_pose[2]))
        results.append(Rehearsal(rank, True, min_height, max_slip))
    return results


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
    depth_search: bool = False,
    thumb_biases: tuple[float, ...] = (0.0,),
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
    depth_extent, width, height_extent = half_extents
    candidates = []

    def evaluate(rot, offset, thumb_bias):
        change = rot @ wrist_to_world.T
        root_quat = Rotation.from_matrix(change @ root_rot).as_quat()
        data.qpos[:] = reference
        data.qpos[:3] = centre + offset - rot @ tool_offset - change @ root_to_wrist
        data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
        data.qpos[object_qpos : object_qpos + 7] = object_pose
        closest_table = 0.10
        best = None
        end = closing.copy()
        end[0] += thumb_bias
        for fraction in np.linspace(0.0, 1.0, 17):
            data.qpos[hand_joints] = opening + fraction * (end - opening)
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
        return PinchCandidate(offset, rot, score, closest_table, support, fraction, thumb_bias)

    # Sample between the centre and the front and back surfaces. Changing how
    # far the fingers wrap around an object changes which links carry the load.
    depths = (-0.8, -0.4, 0.0, 0.4, 0.8) if depth_search else (0.0,)
    for thumb_bias, depth, roll, pitch, height, lateral in product(
        thumb_biases, depths, (-15, 0, 15), (0, 15, 30), (0.5, 0.75, 1.0, 1.25), (-0.2, 0.0, 0.2)
    ):
        rot = Rotation.from_euler("xy", [roll, pitch], degrees=True).as_matrix()
        offset = np.array([depth * depth_extent, lateral * width, height * height_extent])
        candidate = evaluate(rot, offset, thumb_bias)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
