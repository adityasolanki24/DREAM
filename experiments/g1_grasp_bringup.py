"""Fixed-stance G1 grasp diagnostic using a labelled oracle-assisted object pose."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from dream.robot.g1_posture import PerceivedBox, PosturePlanner
from dream.robot.g1_supervisor import SafetyLimits, WholeBodySupervisor
from dream.robot.g1_wbc import WBCLocomotion
from dream.robot.wholebody import TICK, ControllerCapabilities, StopEvent, WBCBackend
from dream.sim.g1_mobile import TOUCH_THRESHOLD, G1MobileEnv, build_mobile_g1

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


def attempt(cfg: dict) -> dict:
    top = cfg["table_top_m"]
    half = cfg["object_half_size_m"]
    table_half = cfg["table_half_size_m"]
    table = (*cfg["table_center_xy_m"], top - table_half[2])
    center = (*cfg["object_center_xy_m"], top + half)

    def scene(spec):
        spec.worldbody.add_geom(
            name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX, pos=table, size=table_half
        )
        box = spec.worldbody.add_body(name="box", pos=center)
        box.add_freejoint()
        box.add_geom(
            name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3,
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

    def set_scratch(obs, planned=None):
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
        scratch.qpos[env.hand_qadr] = env.hand_positions()
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
        for alpha in np.linspace(0.0, 1.0, 17):
            scratch.qpos[env.hand_qadr[RIGHT]] = OPEN + alpha * (CLOSED - OPEN)
            mujoco.mj_kinematics(model, scratch)
            clearance = min(clearance, table_distance())
        return {
            "grasp_xyz_m": grasp.tolist(), "wrist_rotation": rot.tolist(),
            "table_clearance_m": clearance,
        }  # fmt: skip

    def observe_box_z():
        # Labelled oracle-assisted perception for this development fixture.
        return float(env.data.xpos[box_id, 2])

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

    def plan_at(point, stance):
        return planner.plan(
            point,
            stances=[stance],
            arms=["right"],
            obstacles=[obstacle],
            orientation=np.eye(3),
        )

    def reach(phase, point):
        result = plan_at(point, env.odometry())
        if not result.plans:
            phases[phase] = {
                "plan_failure": result.reasons(),
                "candidates": [asdict(failure) for failure in result.failures],
            }
            return False

        target = result.plans[0].target
        set_scratch(backend.observe(), planned=target.q)
        distance = table_distance()
        if distance < TABLE_CLEARANCE_M:
            phases[phase] = {"refused": "planned hand clearance", "table_clearance_m": distance}
            return False

        def watch(obs):
            sample(phase)
            return monitor(obs) or clearance_stop(obs)

        run = backend.execute(target, monitor=watch)
        phases[phase] = {
            "completed": run.completed, "refused": run.refused,
            "stops": [asdict(stop) for stop in run.stops],
        }  # fmt: skip
        return run.completed

    # Keep the pinch near the upper part of the box. The wrist must be high
    # enough to clear the table with the wider palm and fingers.
    grip_point = np.array(center) + [0.0, 0.0, half]
    # Raise on the near side of the table before moving over it. The open
    # fingertip extends about 87 mm beyond the grasp centre along +x.
    approach = grip_point + [-0.06, 0.0, 0.04]
    raise_point = approach.copy()
    raise_point[0] = min(approach[0], table[0] - table_half[0] - 0.11)
    settled = all(tick("settle") for _ in range(round(1.0 / TICK)))

    can_close = False
    if (
        settled
        and reach("raise", raise_point)
        and reach("pregrasp", approach)
        and reach("reach", grip_point)
    ):
        clearance = check_closure()
        error = np.abs(np.asarray(clearance["grasp_xyz_m"]) - grip_point)
        can_close = clearance["table_clearance_m"] >= TABLE_CLEARANCE_M and error.max() <= half
        phases["preclose"] = {**clearance, "position_error_m": error.tolist(), "clear": can_close}
    if can_close:
        opposed = False
        for k in range(1, round(1.0 / TICK) + 1):
            fingers[RIGHT] = OPEN + k * TICK * (CLOSED - OPEN)
            env.set_hand(fingers)
            if not tick("close"):
                break
            touch = env.touch("right")
            opposed = touch.get("hand_thumb_2", 0.0) >= TOUCH_THRESHOLD and any(
                value >= TOUCH_THRESHOLD
                for name, value in touch.items()
                if name.startswith(("hand_index_", "hand_middle_"))
            )
            if opposed:
                break
        phases["close"] = {"opposed_touch": opposed}
        if opposed:
            before = observe_box_z()
            if reach("probe", grip_point + [0.0, 0.0, PROBE_M]):
                response = observe_box_z() - before
                phases["probe"]["object_rise_m"] = response
                if response >= MIN_RESPONSE_M and reach(
                    "lift", grip_point + [0.0, 0.0, LIFT_M + 0.02]
                ):
                    all(tick("hold") for _ in range(round(HOLD_S / TICK)))

    hold = [r for r in rows if r["phase"] == "hold"]
    return {
        "phases": phases, "rows": rows,
        "stop": asdict(backend.held) if backend.held else ctrl.stopped,
        "peak_lift_m": max((r["lift_m"] for r in rows), default=0.0),
        "grasp_verified": len(hold) == round(HOLD_S / TICK)
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
