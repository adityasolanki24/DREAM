"""Fixed-stance G1 grasp diagnostic using a labelled oracle-assisted object pose."""

import argparse
import json
import subprocess
from copy import copy
from dataclasses import asdict, replace
from itertools import chain
from pathlib import Path

import mujoco
import numpy as np

from dream.robot.g1_grasp_geometry import rank_pinches
from dream.robot.g1_posture import PerceivedBox, PosturePlanner
from dream.robot.g1_supervisor import SafetyLimits, WholeBodySupervisor
from dream.robot.g1_wbc import WBCLocomotion
from dream.robot.wholebody import (
    TICK,
    ControllerCapabilities,
    PostureCommand,
    StopEvent,
    WBCBackend,
)
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1

# Right-hand targets mirror the supported G1's seven-joint closure in dream/robot/g1.py.
OPEN = np.array([0, 0, -0.08, 0.08, 0.08, 0.08, 0.08])
CLOSED = np.array([0, -0.7, -1.2, 1.1, 1.2, 1.1, 1.2])
RIGHT = slice(7, 14)
LIFT_M = 0.10
HOLD_S = 3.0
# Approximate overlap of the opposing Dex3 pads near the end of the commissioned closure.
# The open fingertips lie much farther forward than this grasp centre.
GRASP_OFFSET = np.array([0.13, 0.045, 0.0])
# Sample the actual articulated hand sweep. The 3 mm allowance covers mm-scale
# foot-height/odometry differences seen in the WBC reach diagnostic.
TABLE_CLEARANCE_M = 0.003
PROBE_M = 0.03
MIN_RESPONSE_M = 0.01
# Commissioning grip floor: 2 * 3 N at friction 0.5 supports about 0.30 kg.
# It must persist for 0.10 s and is still subject to the 35 N hand force guard.
GRIP_NORMAL_N = 3.0
GRIP_DWELL_S = 0.10
LIFT_STEP_M = 0.03


def attempt(cfg: dict) -> dict:
    top = cfg["table_top_m"]
    size = np.asarray(
        cfg["object_half_extents_m"]
        if "object_half_extents_m" in cfg
        else [cfg["object_half_size_m"]] * 3,
        dtype=float,
    )
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("object half extents must be three positive finite lengths")
    _, half, half_height = size
    shape = cfg.get("object_shape", "box")
    if shape == "box":
        geom_type, geom_size = mujoco.mjtGeom.mjGEOM_BOX, size
    elif shape == "cylinder" and size[0] == size[1]:
        geom_type, geom_size = mujoco.mjtGeom.mjGEOM_CYLINDER, [size[0], half_height, 0.0]
    elif shape == "sphere" and np.all(size == size[0]):
        geom_type, geom_size = mujoco.mjtGeom.mjGEOM_SPHERE, [size[0], 0.0, 0.0]
    else:
        raise ValueError("unsupported shape or dimensions for the fixture")
    table_half = cfg["table_half_size_m"]
    table = (*cfg["table_center_xy_m"], top - table_half[2])
    center = (*cfg["object_center_xy_m"], top + half_height)

    def scene(spec):
        spec.worldbody.add_geom(
            name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX, pos=table, size=table_half
        )
        box = spec.worldbody.add_body(name="box", pos=center)
        box.add_freejoint()
        box.add_geom(
            name="box", type=geom_type, size=geom_size,
            mass=cfg["object_mass_kg"],
        )  # fmt: skip

    model = build_mobile_g1(scene, head_camera=None, hands=True, touch=True)
    if model.neq:
        raise RuntimeError("grasp diagnostic requires a scene without equality constraints")
    env = G1MobileEnv(model, np.random.default_rng(cfg["seed"]), head_camera=None)
    stance = cfg.get("starting_stance_xy_yaw", [0.0, 0.0, 0.0])
    env.reset(*stance)
    env.localise(stance)
    ctrl = WBCLocomotion.from_cache()
    if cfg.get("start_arms_down", False):
        # Fixture initialisation only, using the tucked posture from isaac_task.py.
        arm = np.array([0.1, 0.1, 0, 1.5, 0, 0, 0, 0.1, -0.1, 0, 1.5, 0, 0, 0])
        # Test the hip-clearance hypothesis with a different starting shoulder roll.
        arm[1] = cfg.get("start_left_shoulder_roll_rad", arm[1])
        arm[8] = cfg.get("start_right_shoulder_roll_rad", arm[8])
        env.data.qpos[env.qadr[-arm.size :]] = arm
        ctrl.arm_target[:] = arm
        mujoco.mj_forward(model, env.data)
    caps = ControllerCapabilities.from_envelope_json()
    planner = PosturePlanner(caps, grip_offsets={"right": GRASP_OFFSET})
    # Engineering commissioning limit for this diagnostic, not a sensor calibration.
    backend_force = lambda: sum(env.touch("right").values())  # noqa: E731
    monitor = WholeBodySupervisor(planner.pm, SafetyLimits(contact_force=35.0))
    backend = WBCBackend(env, ctrl, caps, contact_force=backend_force)
    fingers = env.hand_targets.copy()
    fingers[RIGHT] = OPEN
    env.set_hand(fingers)
    rows, phases = [], {}
    box_id = model.body("box").id
    wrist = model.joint("right_wrist_pitch_joint")
    wrist_dof = int(wrist.dofadr[0])
    wrist_channel = int(np.flatnonzero(env.vadr == wrist_dof)[0])
    wrist_actuator = model.actuator("right_wrist_pitch_joint").id
    hand_geoms = {
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith(("right_hand_", "right_wrist_yaw"))
    }
    left_hand_geoms = {
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith(("left_hand_", "left_wrist_yaw"))
    }
    reference_z = env.data.xpos[box_id, 2]
    scratch = mujoco.MjData(model)
    table_geom = model.geom("tabletop").id
    collision_geoms = sorted(g for g in hand_geoms | left_hand_geoms if model.geom_contype[g])
    # Keep the generic obstacle guard on the body. For the hands this runner
    # checks the articulated meshes against the known fixture dimensions.
    obstacle = PerceivedBox(table, tuple(table_half), hand_may_touch=True)
    wrist_body = model.body("right_wrist_yaw_link").id
    box_geom = model.geom("box").id
    box_pose_qpos = int(model.jnt_qposadr[model.body("box").jntadr[0]])
    thumb = [
        g
        for g in collision_geoms
        if g in hand_geoms and "thumb_2" in model.body(model.geom_bodyid[g]).name
    ]
    fingers_against_thumb = [
        g
        for g in collision_geoms
        if g in hand_geoms
        and any(
            f"hand_{finger}" in model.body(model.geom_bodyid[g]).name
            for finger in ("index", "middle")
        )
    ]

    def set_scratch(obs, planned=None, hand=None):
        # Use robot proprioception and the labelled table fixture. Do not copy the
        # live scene or inspect what the tactile sensor touched.
        if planned is None:
            sensed = monitor.pm.kinematic_state(obs.prop.quat, obs.prop.q)
            scratch.qpos[:3] = [*obs.pose[:2], sensed["height"]]
            scratch.qpos[3:7] = obs.prop.quat
            scratch.qpos[env.qadr] = obs.prop.q
        else:
            scratch.qpos[:7] = planned[:7]
            scratch.qpos[env.qadr] = planned[planner.pm.qadr]
        scratch.qpos[env.hand_qadr] = env.hand_positions() if hand is None else hand
        mujoco.mj_kinematics(model, scratch)

    def table_distance():
        return min(
            mujoco.mj_geomDistance(model, scratch, g, table_geom, 0.10, None)
            for g in collision_geoms
        )

    def check_closure():
        set_scratch(backend.observe())
        rot = scratch.xmat[wrist_body].reshape(3, 3)
        grasp = scratch.xpos[wrist_body] + rot @ GRASP_OFFSET
        clearance = table_distance()
        # Estimate whether the observed box fits between the opposing
        # fingers. The pose adapter is oracle-assisted for this fixture.
        scratch.qpos[box_pose_qpos : box_pose_qpos + 7] = observe_box_pose()
        opposing_gap = 0.10
        for alpha in np.linspace(0.0, 1.0, 17):
            scratch.qpos[env.hand_qadr[RIGHT]] = OPEN + alpha * (CLOSED - OPEN)
            mujoco.mj_kinematics(model, scratch)
            clearance = min(clearance, table_distance())
            gaps = [
                min(mujoco.mj_geomDistance(model, scratch, g, box_geom, 0.10, None) for g in side)
                for side in (thumb, fingers_against_thumb)
            ]
            opposing_gap = min(opposing_gap, max(gaps))
        return {
            "grasp_xyz_m": grasp.tolist(), "wrist_rotation": rot.tolist(),
            "table_clearance_m": clearance, "opposing_gap_m": opposing_gap,
        }  # fmt: skip

    def observe_box_pose():
        # Labelled oracle-assisted perception for this development fixture.
        return np.r_[env.data.xpos[box_id], env.data.xquat[box_id]]

    def observe_box():
        return observe_box_pose()[:3]

    def sample(phase):
        # Evaluation only. Decisions below use the touch sensors and labelled pose estimate.
        others = set()
        hand_contact = set()
        left_hand_contact = set()
        for c in env.data.contact:
            for hand, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if hand in hand_geoms or hand in left_hand_geoms:
                    contact = hand_contact if hand in hand_geoms else left_hand_contact
                    contact.add(model.geom(other).name or model.body(model.geom_bodyid[other]).name)
            if model.geom_bodyid[c.geom1] == box_id:
                others.add(int(c.geom2))
            elif model.geom_bodyid[c.geom2] == box_id:
                others.add(int(c.geom1))
        rows.append({
            "phase": phase, "t": env.time, "lift_m": float(env.data.xpos[box_id, 2] - reference_z),
            "odom": env.odometry().tolist(),
            "hand_only_contact": bool(others) and others <= hand_geoms,
            "hand_contact_with": sorted(hand_contact),
            "left_hand_contact_with": sorted(left_hand_contact),
            "wrist_requested_nm": float(backend.observe().tau[wrist_channel]),
            "wrist_commanded_nm": float(env.data.ctrl[wrist_actuator]),
            "wrist_applied_nm": float(env.data.qfrc_actuator[wrist_dof]),
            "wrist_angle_rad": float(env.data.qpos[wrist.qposadr[0]]),
            "touch_n": env.touch("right"), "fingers_rad": env.hand_positions()[RIGHT].tolist(),
        })  # fmt: skip

    def tick(phase):
        env.step(ctrl)
        sample(phase)
        obs = backend.observe()
        stop = monitor(obs)
        if stop is None and phase != "settle":
            stop = clearance_stop(obs)
        if stop and not ctrl.stopped:
            backend.hold(stop)
        return stop is None and not ctrl.stopped

    def clearance_stop(obs):
        set_scratch(obs)
        distance = table_distance()
        if distance < TABLE_CLEARANCE_M:
            return StopEvent(obs.t, "hand clearance", {"table_clearance_m": distance})
        return None

    def plan_at(point, stance, orientation):
        return planner.plan(
            point,
            stances=[stance],
            arms=["right"],
            obstacles=[obstacle],
            orientation=orientation,
        )

    def command_candidates(target):
        start = backend.command()
        end = PostureCommand(
            target.pelvis_height, target.torso_rpy, backend._arm_feedforward(target)
        )
        yield "direct", [end]
        # Stage joints by anatomical groups. The command values come from
        # measured posture and IK; the rollout decides which order is feasible.
        groups = {
            "upper arm first": [7, 8, 9, 10],
            "wrist first": [11, 12, 13],
            "shoulder first": [7, 8, 9],
            "elbow first": [10],
            "shoulder pitch first": [7],
            "shoulder roll first": [8],
        }
        for name, channels in groups.items():
            for fraction in (0.5, 1.0):
                arm = start.arm.copy()
                arm[channels] += fraction * (end.arm[channels] - arm[channels])
                yield f"{name} {fraction:g}", [replace(start, arm=arm), end]
        yield "body first", [replace(end, arm=start.arm), end]

    def clearance_paths(target):
        obs = backend.observe()
        set_scratch(obs)
        base = scratch.qpos.copy()
        start = backend.command()
        end = PostureCommand(
            target.pelvis_height, target.torso_rpy, backend._arm_feedforward(target)
        )
        indices = env.qadr[-7:]
        limits = model.jnt_range[model.actuator_trnid[env.body_actuators[-7:], 0]]
        start_q = base[indices]
        end_q = target.arm_joints[-7:]
        # Local search samples around the path and retracts or swings the
        # elbow/wrist in either direction. Commands come from model geometry.
        rng = np.random.default_rng(0)
        delta = end_q - start_q
        ranges = limits[:, 1] - limits[:, 0]
        centres = start_q + np.arange(0.25, 1.0, 0.25)[:, None] * delta
        candidates = np.concatenate(
            [centre + rng.normal(size=(96, 7)) * ranges * 0.15 for centre in centres]
        )
        candidates = np.clip(candidates, limits[:, 0] + 0.05, limits[:, 1] - 0.05)

        def minimum(a, b):
            smallest = 0.10
            for q in np.linspace(a, b, 19):
                scratch.qpos[:] = base
                scratch.qpos[indices] = q
                mujoco.mj_kinematics(model, scratch)
                smallest = min(smallest, table_distance())
                if smallest < TABLE_CLEARANCE_M:
                    break
            return smallest

        feasible = []
        for q in candidates:
            d = min(minimum(start_q, q), minimum(q, end_q))
            if d >= TABLE_CLEARANCE_M:
                scratch.qpos[:] = base
                scratch.qpos[indices] = q
                mujoco.mj_kinematics(model, scratch)
                # Prefer raising the whole hand before extending it past the edge.
                # Geometry and the current joint state determine the sampled motion.
                elevation = min(scratch.geom_xpos[g, 2] for g in collision_geoms if g in hand_geoms)
                length = np.linalg.norm(q - start_q) + np.linalg.norm(end_q - q)
                feasible.append((float(-elevation + 0.03 * length), q))
        feasible.sort(key=lambda candidate: candidate[0])
        for i, (_, q) in enumerate(feasible[:32]):
            arm = start.arm.copy()
            arm[-7:] = q
            yield f"geometric waypoint {i}", [replace(start, arm=arm), end]

    def preview(commands, rate_scale=1.0):
        # Restart a separate physics model from the robot observation and the
        # labelled fixture. No live object state or evaluator contacts are copied.
        obs = backend.observe()
        set_scratch(obs)
        trial = G1MobileEnv(model, np.random.default_rng(0), noise=env.noise, head_camera=None)
        trial.reset(*obs.pose)
        trial.data.qpos[:] = scratch.qpos
        trial.data.qvel[3:6] = obs.prop.gyro
        trial.data.qvel[trial.vadr] = obs.prop.dq
        if len(rows) > 1:
            trial.data.qvel[:2] = (np.array(rows[-1]["odom"][:2]) - rows[-2]["odom"][:2]) / TICK
        trial.set_hand(env.hand_targets)
        mujoco.mj_forward(model, trial.data)
        trial.localise(obs.pose)
        policy = copy(ctrl)
        policy.history = copy(ctrl.history)
        policy.arm_target = ctrl.arm_target.copy()
        trial_backend = WBCBackend(
            trial, policy, caps, contact_force=lambda: sum(trial.touch("right").values())
        )
        supervisor = WholeBodySupervisor(planner.pm, monitor.limits)

        def guard(state):
            stop = supervisor(state)
            if stop:
                return stop
            set_scratch(state, hand=trial.hand_positions())
            d = table_distance()
            if d < TABLE_CLEARANCE_M:
                return StopEvent(state.t, "hand clearance", {"table_clearance_m": d})
            return None

        report = trial_backend.follow(commands, monitor=guard, rate_scale=rate_scale)
        if report.completed:
            set_scratch(trial_backend.observe(), hand=trial.hand_positions())
            rot = scratch.xmat[wrist_body].reshape(3, 3)
            report.achieved["grasp_xyz_m"] = scratch.xpos[wrist_body] + rot @ GRASP_OFFSET
        return report

    def goals(phase, point):
        if phase != "raise":
            yield "nominal", point, pinch_rotation
            return
        # Lift behind the table before rotating into the pinch. Build the
        # waypoint from the observed wrist, not from a stored arm pose.
        set_scratch(backend.observe())
        wrist_rot = scratch.xmat[wrist_body].reshape(3, 3).copy()
        hand = scratch.xpos[wrist_body].copy() + wrist_rot @ GRASP_OFFSET
        for z in (point[2], point[2] + 2 * half):
            raised = np.array([hand[0], hand[1], z])
            yield "vertical raise in measured orientation", raised, wrist_rot
            raised[0] -= 2 * half
            yield "vertical raise with retreat", raised.copy(), wrist_rot
        # A raised wrist angle retracts the fingertips in x during ascent.
        yield "nominal", point, pinch_rotation
        width = 2.0 * half
        for up, back in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (1, 2)):
            waypoint = point + [-back * width, 0.0, up * width]
            for pitch in (-np.pi / 4, -np.pi / 2, 0.0):
                c, sn = np.cos(pitch), np.sin(pitch)
                rotation = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
                if up or back or pitch:
                    yield f"up {up}, retreat {back}, pitch {pitch:.2f}", waypoint, rotation

    def reach(phase, point):
        attempts = []
        use_search = cfg.get("trajectory_search", True) and (
            phase in {"raise", "pregrasp", "reach"} or phase.startswith("align")
        )

        def watch(obs):
            sample(phase)
            return monitor(obs) or clearance_stop(obs)

        candidates = goals(phase, point) if use_search else [("nominal", point, pinch_rotation)]
        for label, waypoint, rotation in candidates:
            result = plan_at(waypoint, env.odometry(), rotation)
            if not result.plans:
                attempts.append({"goal": label, "plan_failure": result.reasons()})
                continue
            target = result.plans[0].target
            set_scratch(backend.observe(), planned=target.q)
            distance = table_distance()
            if distance < TABLE_CLEARANCE_M:
                attempts.append({"goal": label, "planned_table_clearance_m": distance})
                continue

            if use_search:
                paths = command_candidates(target)
                if label == "nominal":
                    paths = chain(paths, clearance_paths(target))
                for name, commands in paths:
                    if label != "nominal" and name not in {"direct", "upper arm first 1"}:
                        continue
                    prediction = preview(commands)
                    attempts.append({
                        "goal": label, "order": name, "passed": prediction.completed,
                        "stop": asdict(prediction.stops[0]) if prediction.stops else None,
                    })  # fmt: skip
                    if prediction.completed:
                        run = backend.follow(commands, monitor=watch)
                        break
                else:
                    continue
            else:
                run = backend.execute(target, monitor=watch)

            phases[phase] = {
                "predictions": attempts, "completed": run.completed, "refused": run.refused,
                "stops": [asdict(stop) for stop in run.stops],
            }  # fmt: skip
            return run.completed
        phases[phase] = {"refused": "no verified trajectory", "predictions": attempts}
        return False

    def opposing_forces():
        touch = env.touch("right")
        return (
            touch.get("hand_thumb_2", 0.0),
            sum(
                value
                for name, value in touch.items()
                if name.startswith(("hand_index_", "hand_middle_"))
            ),
        )

    fraction = 0.0
    lost_ticks = 0
    held_box = None

    def box_in_wrist():
        rot = scratch.xmat[wrist_body].reshape(3, 3)
        return rot.T @ (observe_box() - scratch.xpos[wrist_body])

    def maintain_grip(obs):
        nonlocal fraction, lost_ticks
        weakest = min(opposing_forces())
        # Avoid a contact loss false positive from one 50 Hz sensor sample.
        lost_ticks = lost_ticks + 1 if weakest < 0.1 else 0
        if lost_ticks >= round(GRIP_DWELL_S / TICK):
            return StopEvent(obs.t, "grip lost", {"weakest_normal_n": weakest})
        # Normal force may stay steady while the box creeps down the pads.
        # The measured relative pose raises the requested preload on slip.
        displacement = float(np.linalg.norm(box_in_wrist() - held_box))
        normal_goal = GRIP_NORMAL_N * (1.0 + min(2.0, displacement / (0.2 * half)))
        rows[-1]["relative_box_motion_m"] = displacement
        rows[-1]["grip_target_n"] = normal_goal
        if weakest < normal_goal:
            fraction = min(1.0, fraction + 0.5 * TICK)
        elif weakest > 2 * normal_goal:
            fraction = max(0.0, fraction - 0.25 * TICK)
        fingers[RIGHT] = OPEN + fraction * (CLOSED - OPEN)
        env.set_hand(fingers)
        return None

    def relative_command(point, rot):
        obs = backend.observe()
        sensed = planner.pm.kinematic_state(obs.prop.quat, obs.prop.q)
        q0 = planner.measured_qpos(obs.prop, obs.pose, sensed["height"])
        q = planner.refine(
            q0, obs.pose, "right", point, [obstacle], hold_waist=True, orientation=rot
        )
        if q is None:
            return None
        cmd = backend.command()
        arm = cmd.arm.copy()
        arm[RIGHT] += (q[planner.pm.qadr] - obs.prop.q)[-7:]
        limits = planner.pm.joint_range[planner.pm.qadr.size - 7 :]
        if np.any(arm[RIGHT] < limits[:, 0]) or np.any(arm[RIGHT] > limits[:, 1]):
            return None
        return replace(cmd, arm=arm)

    def vertical_move(phase, height, grasp=False):
        # Avoid sweeping the box sideways through a newly planned posture.
        # Re-solve small vertical increments from the measured hand state,
        # and hold the whole-body balance reference during each increment.
        obs = backend.observe()
        set_scratch(obs)
        rot = scratch.xmat[wrist_body].reshape(3, 3).copy()
        origin = scratch.xpos[wrist_body].copy() + rot @ GRASP_OFFSET
        count = max(1, int(np.ceil(abs(height) / LIFT_STEP_M)))
        for step in range(1, count + 1):
            point = origin + [0.0, 0.0, height * step / count]
            next_cmd = relative_command(point, rot)
            if next_cmd is None:
                phases[phase] = {"completed": False, "refused": "relative lift IK"}
                return False

            def watch(state):
                sample(phase)
                stop = monitor(state) or clearance_stop(state)
                if stop is None and grasp:
                    stop = maintain_grip(state)
                return stop

            run = backend.follow([next_cmd], monitor=watch, rate_scale=0.5)
            if not run.completed:
                phases[phase] = {"completed": False, "stops": [asdict(stop) for stop in run.stops]}
                return False
        phases[phase] = {"completed": True}
        return True

    def descend_to_grip():
        attempts = []
        for _ in range(10):
            geometry = check_closure()
            position = np.asarray(geometry["grasp_xyz_m"])
            goal = observe_box() + pinch_offset
            error = goal - position
            if (
                geometry["table_clearance_m"] >= TABLE_CLEARANCE_M
                and geometry["opposing_gap_m"] <= 0.0
                and np.abs(error[:2]).max() <= 0.5 * half
                and abs(error[2]) <= 0.05 * half_height
            ):
                phases["descend"] = {"completed": True, "corrections": attempts}
                return True, geometry, error

            # Move in box-scaled increments. Compare candidate progress from
            # the model and reobserve after every executed step.
            delta = np.clip(error, -half / 2, half / 2)
            rot = np.asarray(geometry["wrist_rotation"])
            for scale in (1.0, 0.5, 0.25):
                command = relative_command(position + scale * delta, rot)
                if command is None:
                    continue
                prediction = preview([command], rate_scale=0.5)
                if not prediction.completed:
                    continue
                predicted = prediction.achieved["grasp_xyz_m"]
                if np.linalg.norm(goal - predicted) >= np.linalg.norm(error) - 0.0005:
                    continue

                def watch(state):
                    sample("descend")
                    return monitor(state) or clearance_stop(state)

                run = backend.follow([command], monitor=watch, rate_scale=0.5)
                attempts.append({
                    "before_error_m": error.tolist(), "prediction_m": predicted.tolist(),
                    "stops": [asdict(stop) for stop in run.stops],
                })  # fmt: skip
                if not run.completed:
                    phases["descend"] = {"completed": False, "corrections": attempts}
                    return False, geometry, error
                break
            else:
                break
        phases["descend"] = {"completed": False, "corrections": attempts}
        return False, geometry, error

    # Keep the pinch near the upper part of the box. The wrist must be high
    # enough to clear the table with the wider palm and fingers.
    pinch_offset = np.array([0.0, 0.0, half_height])
    pinch_rotation = np.eye(3)
    if cfg.get("contact_geometry_search", False):
        set_scratch(backend.observe())
        pinches = rank_pinches(
            model,
            scratch.qpos,
            observe_box_pose(),
            size,
            env.hand_qadr[RIGHT],
            OPEN,
            CLOSED,
            GRASP_OFFSET,
            TABLE_CLEARANCE_M,
            object_geom=box_geom,
            support_geom=table_geom,
        )
        if pinches:
            selected = pinches[0]
            pinch_offset, pinch_rotation = selected.offset, selected.rotation
            phases["grasp_geometry"] = {
                "candidates": len(pinches), "offset_m": pinch_offset.tolist(),
                "rotation": pinch_rotation.tolist(), "score": selected.score,
                "support_links": selected.support_links,
                "predicted_closure_fraction": selected.closure_fraction,
            }  # fmt: skip
        else:
            phases["grasp_geometry"] = {"candidates": 0}
    grip_point = observe_box() + pinch_offset
    # Raise on the near side of the table before moving over it. The open
    # fingertip extends about 87 mm beyond the grasp centre along +x.
    approach = grip_point + [-0.06, 0.0, 0.04]
    raise_point = approach.copy()
    raise_point[0] = min(approach[0], table[0] - table_half[0] - 0.11)
    settled = all(tick("settle") for _ in range(round(1.0 / TICK)))
    if settled and cfg.get("trajectory_search", True):
        # Balance can keep moving the pelvis after the initial hold. This
        # window needs to resolve drift below the table clearance allowance.
        window = round(0.4 / TICK)
        for _ in range(round(3.0 / TICK)):
            if not tick("settle"):
                settled = False
                break
            xy = np.asarray([row["odom"][:2] for row in rows[-window:]])
            if np.linalg.norm(np.ptp(xy, axis=0)) < TABLE_CLEARANCE_M:
                break
        else:
            settled = False
        phases["settle"] = {"stable": settled}

    can_close = False
    above = grip_point + [0.0, 0.0, 2 * half_height]
    if (
        settled
        and reach("raise", raise_point)
        and reach("pregrasp", approach)
        and reach("reach", above)
    ):
        command_point = above.copy()
        for i in range(4):
            clearance = check_closure()
            grip_point = observe_box() + pinch_offset
            measured = np.asarray(clearance["grasp_xyz_m"])
            error = grip_point - measured
            if np.abs(error[:2]).max() <= half / 2:
                can_close, clearance, error = descend_to_grip()
                break
            if i == 3:
                break
            command_point[:2] += np.clip(error[:2], -2 * half, 2 * half)
            if not reach(f"align_{i + 1}", command_point):
                break
        phases["preclose"] = {**clearance, "position_error_m": error.tolist(), "clear": can_close}
    if can_close:
        stable_ticks = 0
        for _ in range(round(1.5 / TICK)):
            fingers[RIGHT] = OPEN + fraction * (CLOSED - OPEN)
            env.set_hand(fingers)
            if not tick("close"):
                break
            if min(opposing_forces()) >= GRIP_NORMAL_N:
                stable_ticks += 1
                if stable_ticks >= round(GRIP_DWELL_S / TICK):
                    break
            else:
                stable_ticks = 0
                fraction = min(1.0, fraction + TICK)
        opposed = stable_ticks >= round(GRIP_DWELL_S / TICK)
        phases["close"] = {"opposed_touch": opposed, "closure_fraction": fraction}
        if opposed:
            set_scratch(backend.observe())
            held_box = box_in_wrist()
            before = float(observe_box()[2])
            if vertical_move("probe", PROBE_M, grasp=True):
                response = float(observe_box()[2]) - before
                phases["probe"]["object_rise_m"] = response
                if response >= MIN_RESPONSE_M:
                    # Stop on observed height or lack of progress. Each
                    # request is still a bounded vertical correction.
                    for _ in range(6):
                        z = float(observe_box()[2])
                        if z - before >= LIFT_M + 0.02:
                            break
                        if not vertical_move("lift", LIFT_STEP_M, grasp=True):
                            break
                        if float(observe_box()[2]) - z < MIN_RESPONSE_M:
                            break
                    if not backend.held and float(observe_box()[2]) - before >= LIFT_M:
                        for _ in range(round(HOLD_S / TICK)):
                            if not tick("hold"):
                                break
                            obs = backend.observe()
                            stop = maintain_grip(obs)
                            if stop:
                                backend.hold(stop)
                                break

    hold = [r for r in rows if r["phase"] == "hold"]
    return {
        "phases": phases, "rows": rows,
        "stop": asdict(backend.held) if backend.held else ctrl.stopped,
        "peak_lift_m": max((r["lift_m"] for r in rows), default=0.0),
        "grasp_verified": not backend.held and not ctrl.stopped
        and len(hold) == round(HOLD_S / TICK)
        and all(r["lift_m"] >= LIFT_M and r["hand_only_contact"] for r in hold),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {"config": cfg, "git_sha": revision, "mujoco": mujoco.__version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = attempt(cfg)
    rows = result.pop("rows")
    (args.out / "trace.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
