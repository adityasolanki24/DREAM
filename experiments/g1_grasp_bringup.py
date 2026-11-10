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
from dream.robot.wholebody import TICK, ControllerCapabilities, WBCBackend
from dream.sim.g1_mobile import TOUCH_THRESHOLD, G1MobileEnv, build_mobile_g1

# Right-hand targets mirror the supported G1's seven-joint closure in dream/robot/g1.py.
OPEN = np.array([0, 0, -0.08, 0.08, 0.08, 0.08, 0.08])
CLOSED = np.array([0, -0.7, -1.2, 1.1, 1.2, 1.1, 1.2])
RIGHT = slice(7, 14)
LIFT_M = 0.10
HOLD_S = 3.0


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
        env.data.qpos[env.qadr[-arm.size :]] = arm
        ctrl.arm_target[:] = arm
        mujoco.mj_forward(model, env.data)
    caps = ControllerCapabilities.from_envelope_json()
    planner = PosturePlanner(caps)
    # Engineering commissioning limit for this diagnostic, not a sensor calibration.
    backend_force = lambda: sum(env.touch("right").values())  # noqa: E731
    monitor = WholeBodySupervisor(planner.pm, SafetyLimits(contact_force=35.0))
    backend = WBCBackend(env, ctrl, caps, contact_force=backend_force)
    fingers = env.hand_targets.copy()
    fingers[RIGHT] = OPEN
    env.set_hand(fingers)
    rows, phases = [], {}
    box_id = model.body("box").id
    hand_geoms = {
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith(("right_hand_", "right_wrist_yaw"))
    }
    reference_z = env.data.xpos[box_id, 2]

    def sample(phase):
        # Evaluation only. Decisions below use the touch sensors and labelled pose estimate.
        others = set()
        hand_contact = set()
        for c in env.data.contact:
            for hand, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if hand in hand_geoms:
                    hand_contact.add(
                        model.geom(other).name or model.body(model.geom_bodyid[other]).name
                    )
            if model.geom_bodyid[c.geom1] == box_id:
                others.add(int(c.geom2))
            elif model.geom_bodyid[c.geom2] == box_id:
                others.add(int(c.geom1))
        rows.append({
            "phase": phase, "t": env.time, "lift_m": float(env.data.xpos[box_id, 2] - reference_z),
            "hand_only_contact": bool(others) and others <= hand_geoms,
            "hand_contact_with": sorted(hand_contact),
            "touch_n": env.touch("right"), "fingers_rad": env.hand_positions()[RIGHT].tolist(),
        })  # fmt: skip

    def tick(phase):
        env.step(ctrl)
        sample(phase)
        obs = backend.observe()
        stop = monitor(obs)
        if stop and not ctrl.stopped:
            backend.hold(stop)
        return stop is None and not ctrl.stopped

    def reach(phase, point):
        # The table and starting object pose are supplied as oracle-assisted fixture inputs.
        obstacle = PerceivedBox(table, tuple(table_half))
        result = planner.plan(
            point,
            stances=[env.odometry()],
            arms=["right"],
            obstacles=[obstacle],
            approach_axis=(1.0, 0.0, 0.0),
        )
        if not result.plans:
            phases[phase] = {
                "plan_failure": result.reasons(),
                "candidates": [asdict(failure) for failure in result.failures],
            }
            return False

        def watch(obs):
            sample(phase)
            return monitor(obs)

        run = backend.execute(result.plans[0].target, monitor=watch)
        phases[phase] = {
            "completed": run.completed, "refused": run.refused,
            "stops": [asdict(stop) for stop in run.stops],
        }  # fmt: skip
        return run.completed

    settled = all(tick("settle") for _ in range(round(1.0 / TICK)))
    if settled and reach("reach", center):
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
        if opposed and reach("lift", np.array(center) + [0.0, 0.0, LIFT_M + 0.02]):
            all(tick("hold") for _ in range(round(HOLD_S / TICK)))

    hold = [r for r in rows if r["phase"] == "hold"]
    return {
        "phases": phases, "rows": rows,
        "stop": asdict(monitor.latched) if monitor.latched else ctrl.stopped,
        "peak_lift_m": max(r["lift_m"] for r in rows),
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
