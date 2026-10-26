"""Walk to an object in an NVIDIA Isaac Sim kitchen and act on it: open a cabinet, fetch food.

Called from ``experiments/isaac_g1_kitchen.py --mode task --task open_cabinet|fetch_banana``.

- See: the head camera's instance segmentation, mapped to the scene's semantic class labels,
  is the object detector (a perfect-detector stand-in, declared); where the object is comes
  from the robot's own depth image, in its odometry frame.
- Go: turn on the spot only until the target is in view, then one planned walk
  (Regulated Pure Pursuit: it turns while walking) to a stand point facing the target.
- Reach: right-arm damped-least-squares IK on the identical G1 MJCF (mirrored from the
  Isaac joint state); the arm PD tracks it while the pretrained WBC policy balances.
- Grasp: ASSISTED, as in BEHAVIOR/OmniGibson: these hands have no actuated fingers, so on
  contact a physics joint pins the fingertip link to the object's rigid body. Opening and
  carrying are then the scene's own joint and contact physics under the robot's motion.
- Record: a chase camera rendered live at 10 Hz of robot time (evidence only).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

HAND_LINK = "right_hand_index_0_link"
STAND_OFF = {"cabinet_pull": 0.42, "banana": 0.40}  # m from target to pelvis
GRASP_JOINT = "/World/DreamAssistedGrasp"


class ChaseCamera:
    def __init__(self, world, out: pathlib.Path, size=(960, 540)):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.world, self.out = world, out
        cam = UsdGeom.Camera.Define(world.stage, "/World/DreamChaseCamera")
        cam.CreateFocalLengthAttr(13.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * size[1] / size[0])
        cam.CreateClippingRangeAttr((0.05, 100.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        self.op = xf.AddTransformOp()
        rp = rep.create.render_product("/World/DreamChaseCamera", size)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([rp])
        (out / "chase").mkdir(parents=True, exist_ok=True)
        self.log = open(out / "chase_frames.jsonl", "w")
        self.k, self.heading = 0, None

    def shoot(self, t: float, pose) -> None:
        from PIL import Image
        from pxr import Gf

        px, py, yaw = pose
        if self.heading is None:
            self.heading = yaw
        self.heading += 0.1 * math.atan2(math.sin(yaw - self.heading), math.cos(yaw - self.heading))
        a = self.heading + 0.7  # behind and to the right: arm and target stay visible
        eye = Gf.Vec3d(px - 1.6 * math.cos(a), py - 1.6 * math.sin(a), 2.3)
        view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(px, py, 0.7), Gf.Vec3d(0, 0, 1))
        self.op.Set(view.GetInverse())
        self.world.render()
        self.world.render()
        img = np.asarray(self.rgb.get_data())
        if img.size:
            Image.fromarray(img[..., :3]).save(self.out / "chase" / f"{self.k:05d}.jpg", quality=88)
            self.log.write(json.dumps({"index": self.k, "time_s": t}) + "\n")
            self.log.flush()
            self.k += 1


def labelled(stage) -> dict:
    """prim path -> semantic class, from the scene author's labels."""
    from dream.eval.isaac_scene import labelled_objects

    return {o.path: o.label for o in labelled_objects(stage)}


def locate(env, frame, labels: dict, want: str):
    """Instances of class ``want`` in the last head frame: [(prim, odom xyz, pixels)]."""
    from dream.perception.geometry import backproject

    if env.last_segmentation is None:
        return []
    ids, info = env.last_segmentation
    pts = backproject(frame)
    out = []
    for key, path in info.items():
        path = path if isinstance(path, str) else str(path)
        owner = max((p for p in labels if path == p or path.startswith(p + "/")), key=len,
                    default=None)  # fmt: skip
        if owner is None or labels[owner] != want:
            continue
        mask = (ids == int(key)) & np.isfinite(frame.depth)
        if mask.sum() < 15:
            continue
        out.append((owner, np.median(pts[mask], axis=0), int(mask.sum())))
    return sorted(out, key=lambda o: -o[2])


def rigid_ancestor(stage, path: str) -> str | None:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    while prim and prim.IsValid() and str(prim.GetPath()) != "/":
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def make_grasp(stage, env, body1: str) -> dict:
    """Assisted grasp: a fixed joint from the fingertip link to ``body1`` where they are."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    hand = next(b for b in env.bodies if b.endswith("/" + HAND_LINK))
    p_hand, r_hand = env.link_pose(HAND_LINK)
    m1 = UsdGeom.Xformable(stage.GetPrimAtPath(body1)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())  # fmt: skip
    t1 = np.array(m1.ExtractTranslation())
    r1 = np.array(m1.ExtractRotationMatrix()).T  # USD row-vector convention -> column matrix
    j = UsdPhysics.FixedJoint.Define(stage, GRASP_JOINT)
    j.CreateBody0Rel().SetTargets([hand])
    j.CreateBody1Rel().SetTargets([body1])
    local1 = r1.T @ (p_hand - t1)
    rel = r1.T @ r_hand  # hand orientation in body1's frame
    q = Gf.Quatf(Gf.Matrix3d(*rel.T.flatten()).ExtractRotation().GetQuat())
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*local1))
    j.CreateLocalRot1Attr(q)
    return {"hand": hand, "body": body1, "contact_world": p_hand.tolist()}


def release(stage) -> None:
    if stage.GetPrimAtPath(GRASP_JOINT):
        stage.RemovePrim(GRASP_JOINT)


class ArmIK:
    """Right-arm IK on the MuJoCo G1 (same MJCF), mirrored from the Isaac state."""

    def __init__(self, env, controller):
        import mujoco

        from dream.sim.g1_mobile import build_mobile_g1
        from dream.skills.g1_reach import RightArm

        self.env, self.mj = env, mujoco
        self.model = build_mobile_g1(head_camera=None)
        self.data = mujoco.MjData(self.model)
        self.arm = RightArm(self.model, self.data, controller)

    def mirror(self) -> None:
        p, q = self.env._root()
        self.data.qpos[0:3] = p
        self.data.qpos[3:7] = q
        self.data.qpos[7 : 7 + 29] = self.env.proprioception().q
        self.mj.mj_kinematics(self.model, self.data)

    def solve(self, target_world: np.ndarray) -> np.ndarray:
        self.mirror()
        return self.arm.solve(np.asarray(target_world, float))


def odom_to_world(start, p):
    x0, y0, yaw0 = start
    c, s = math.cos(yaw0), math.sin(yaw0)
    return np.array([x0 + c * p[0] - s * p[1], y0 + s * p[0] + c * p[1], p[2]])


def world_to_odom(start, p):
    x0, y0, yaw0 = start
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = p[0] - x0, p[1] - y0
    return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])


def stand_point(tb, target_xy, robot_xy, reach: float) -> np.ndarray:
    """The walkable cell nearest the target, no closer than 0.25 m and within ``reach``
    + 0.25 m of it, preferring the robot's side (never a point inside furniture)."""
    from dream.navigation.planning import traversable

    grid = tb.memory.grid
    ok = traversable(grid)
    i, j = np.nonzero(ok)
    pts = grid.centre(i, j)
    dist = np.linalg.norm(pts - target_xy, axis=1)
    near = (dist >= 0.25) & (dist <= reach + 0.25)
    if not near.any():
        u = (target_xy - robot_xy) / max(np.linalg.norm(target_xy - robot_xy), 1e-6)
        return target_xy - u * reach
    cost = np.abs(dist - reach) + 0.2 * np.linalg.norm(pts - robot_xy, axis=1) / 5.0
    cost[~near] = np.inf
    return pts[int(np.argmin(cost))]


def face(base, target_xy, tol: float = 0.06) -> None:
    """Short in-place turn so the target is straight ahead (the only on-the-spot turn)."""
    for _ in range(40):
        x, y, yaw = base.odometry()
        err = math.atan2(target_xy[1] - y, target_xy[0] - x) - yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        if abs(err) < tol:
            break
        base.walk(0.0, 0.0, float(np.clip(1.5 * err, -0.5, 0.5)), 0.2)
    base.walk(0.0, 0.0, 0.0, 0.4)


def place_on_counter_edge(stage, label: str, start_xy) -> dict:
    """Scene setup: put one ``label`` item 12 cm in from the front edge of the countertop
    nearest the robot's start (where a person leaves something), avoiding sinks."""
    from pxr import Gf, UsdGeom

    from dream.eval.isaac_scene import labelled_objects

    objs = labelled_objects(stage)
    item = next(o for o in objs if o.label == label)
    sinks = [o for o in objs if o.label in ("sink", "sink_basin", "sink_rim")]
    centre = np.mean([[o.centre[0], o.centre[1]] for o in objs if o.label == "countertop"], 0)
    best = None
    for c in (o for o in objs if o.label == "countertop"):
        long_x = (c.hi[0] - c.lo[0]) >= (c.hi[1] - c.lo[1])
        if long_x:  # front edge: the long side facing the kitchen's centre
            ey = c.lo[1] if abs(c.lo[1] - centre[1]) < abs(c.hi[1] - centre[1]) else c.hi[1]
            inward = 0.12 if ey == c.lo[1] else -0.12
            xs = np.linspace(c.lo[0] + 0.15, c.hi[0] - 0.15, 9)
            pts = [(x, ey + inward) for x in xs]
        else:
            ex = c.lo[0] if abs(c.lo[0] - centre[0]) < abs(c.hi[0] - centre[0]) else c.hi[0]
            inward = 0.12 if ex == c.lo[0] else -0.12
            ys = np.linspace(c.lo[1] + 0.15, c.hi[1] - 0.15, 9)
            pts = [(ex + inward, y) for y in ys]
        for p in pts:
            if any(s.lo[0] - 0.1 < p[0] < s.hi[0] + 0.1 and s.lo[1] - 0.1 < p[1] < s.hi[1] + 0.1
                   for s in sinks):  # fmt: skip
                continue
            d = float(np.hypot(p[0] - start_xy[0], p[1] - start_xy[1]))
            if best is None or d < best[0]:
                best = (d, p, float(c.hi[2]))
    _, (px, py), top = best
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(item.path))
    offset = np.array([px, py, top + 0.03]) - item.centre
    t, *_ = xf.GetXformVectors(0)
    xf.SetTranslate(Gf.Vec3d(*(np.array(t) + offset)))
    return {"item": item.path, "placed_at": [px, py, top + 0.03]}


def torso_forward(env) -> float:
    """How far the torso is ahead of the pelvis along the heading (m), from link poses."""
    p_t, _ = env.link_pose("torso_link")
    p_p, _ = env._root()
    yaw = env.true_pose()[2]
    return float((p_t[0] - p_p[0]) * math.cos(yaw) + (p_t[1] - p_p[1]) * math.sin(yaw))


def lean_forward(env, base, log, pitch: float, height: float) -> dict:
    """Lean over the counter with the policy's own torso-pitch/height commands, ramped over
    1 s. The sign convention is checked from the measured torso position and flipped once."""
    ctrl = base.controller
    before = torso_forward(env)
    for sign in (1.0, -1.0):
        for k in range(1, 26):
            ctrl.rpy_cmd = np.array([0.0, sign * pitch * k / 25, 0.0])
            ctrl.height_cmd = 0.74 + (height - 0.74) * k / 25
            base.walk(0, 0, 0, 0.04)
        base.walk(0, 0, 0, 0.6)
        gained = torso_forward(env) - before
        log("lean", f"torso pitch command {sign * pitch:+.2f} rad, height {height:.2f} m: "
            f"torso moved {gained * 100:+.0f} cm forward")  # fmt: skip
        if gained > 0.03:
            return {"pitch_cmd": sign * pitch, "height_cmd": height, "torso_forward_m": gained}
        ctrl.rpy_cmd = np.zeros(3)
        base.walk(0, 0, 0, 0.8)
    return {"pitch_cmd": 0.0, "height_cmd": height, "torso_forward_m": 0.0}


ARMS_DOWN = np.array([0.1, 0.1, 0.0, 1.5, 0.0, 0.0, 0.0,  # left arm
                      0.1, -0.1, 0.0, 1.5, 0.0, 0.0, 0.0])  # right arm  # fmt: skip


def tuck_arms(base) -> None:
    """Hands beside the hips (FK on the G1 MJCF: 0.6 m high, 0.2 m out), ramped over 1 s."""
    ctrl = base.controller
    a0 = ctrl.arm_target.copy()
    for k in range(1, 26):
        ctrl.arm_target[:] = a0 + (ARMS_DOWN - a0) * k / 25
        base.walk(0, 0, 0, 0.04)


def upright(base) -> None:
    """Ramp the posture commands back to the validated defaults (before walking far)."""
    ctrl = base.controller
    r0, h0 = ctrl.rpy_cmd.copy(), ctrl.height_cmd
    for k in range(1, 26):
        ctrl.rpy_cmd = r0 * (1 - k / 25)
        ctrl.height_cmd = h0 + (0.74 - h0) * k / 25
        base.walk(0, 0, 0, 0.04)
    base.walk(0, 0, 0, 0.4)


def make_dynamic(stage, label: str) -> list[str]:
    """Scene setup: make the labelled food items rigid bodies (in the asset they are static
    props), with convex colliders and a light mass, so they can be picked up."""
    from pxr import Usd, UsdPhysics

    from dream.eval.isaac_scene import labelled_objects

    done = []
    for o in labelled_objects(stage):
        if o.label != label:
            continue
        prim = stage.GetPrimAtPath(o.path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.15)
        for child in Usd.PrimRange(prim):
            if child.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(child)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
        done.append(o.path)
    return done


def run_task(world, env, base, cfg, ev, start, task: str) -> dict:
    from dream.perception.spatial import SpatialMemory
    from dream.tools.mobile import MobileToolbox

    stage = world.stage
    want = {"open_cabinet": "cabinet_pull", "fetch_banana": "banana"}[task]
    labels = labelled(stage)
    events = []

    def log(kind, text, **data):
        row = {"time_s": env.time, "kind": kind, "text": text, **data}
        events.append(row)
        ev.line("events.jsonl", row)

    chase = ChaseCamera(world, ev.out)
    tick = base.on_tick or (lambda: None)
    counter = {"n": 0}

    def on_tick():
        tick()
        counter["n"] += 1
        if counter["n"] % 5 == 0:  # 10 Hz of robot time
            chase.shoot(env.time, env.true_pose())

    base.on_tick = on_tick
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([2])), navigator="rpp",
                       on_event=lambda k, t, **_: log(k, t))  # fmt: skip
    tb.allow_unknown = True  # plan through floor not yet seen; replanning corrects it
    out = {"task": task, "target_label": want}
    log("command", f"task: {task.replace('_', ' ')}")

    def look():
        frame = base.capture()
        tb.memory.update(frame)
        # standing reach only (0.55-1.15 m); low handles need a crouch the policy is not
        # commanded to do yet
        return [f for f in locate(env, frame, labels, want) if 0.55 <= f[1][2] <= 1.15]

    # 1. find: turn only until the target class is visible
    found, turned = look(), 0.0
    while not found and turned < 2 * math.pi:
        base.walk(0.0, 0.0, 0.5, 0.6)
        turned += 0.3
        found = look()
    if not found:
        log("result", f"no {want} in view after a full turn")
        return {**out, "success": False, "reason": "not found"}
    prim, p_odom, _ = found[0]
    log("see", f"{want} at {np.round(p_odom, 2).tolist()} ({prim.split('/')[-1]})")
    out["target_prim"] = prim
    # 2. go: stand facing it at arm's length
    pose = base.odometry()
    d = p_odom[:2] - pose[:2]
    u = d / max(np.linalg.norm(d), 1e-6)
    stand = stand_point(tb, p_odom[:2], pose[:2], STAND_OFF[want])
    log("go", f"walk to {np.round(stand, 2).tolist()} facing the {want}")
    r = tb.call("go_to", {"x": float(stand[0]), "y": float(stand[1]),
                          "heading": float(math.atan2(u[1], u[0]))})  # fmt: skip
    log("arrived", f"go_to {r.status} {r.message}")
    face(base, p_odom[:2])
    tuck_arms(base)  # hands down by the hips, so the body (not the hands) meets the counter
    log("tuck", "arms down by the sides for the last steps")
    # final approach: a slow straight step until it is at arm's length (people walk right
    # up to a counter; the planner keeps a robot-radius margin from it)
    for _ in range(15):
        refined = [f for f in look() if f[0] == prim]
        if refined:
            p_odom = refined[0][1]
        gap_xy = float(np.linalg.norm(p_odom[:2] - base.odometry()[:2]))
        if gap_xy <= STAND_OFF[want] + 0.02:
            break
        touching = env.contact_steps
        base.walk(min(0.15, gap_xy - STAND_OFF[want] + 0.05), 0.0, 0.0, 0.4)
        if env.contact_steps > touching:  # touched the furniture: back off 3 cm and stop
            base.walk(-0.08, 0.0, 0.0, 0.4)
            log("approach", "touched the furniture; stopped at its edge")
            break
    base.walk(0.0, 0.0, 0.0, 0.5)
    face(base, p_odom[:2])
    x, y, yaw = base.odometry()
    left = -math.sin(yaw) * (p_odom[0] - x) + math.cos(yaw) * (p_odom[1] - y)
    shift = left + 0.12  # put it in front of the right shoulder (12 cm right of centre)
    if abs(shift) > 0.03:
        base.walk(0.0, math.copysign(0.1, shift), 0.0, abs(shift) / 0.1)
        base.walk(0.0, 0.0, 0.0, 0.4)
        log("align", f"side-stepped {shift * 100:.0f} cm so it is in front of the right hand")
    pose = base.odometry()
    d = p_odom[:2] - pose[:2]
    u = d / max(np.linalg.norm(d), 1e-6)
    log("face", f"{want} {np.linalg.norm(d):.2f} m ahead, bearing "
        f"{math.degrees(math.atan2(d[1], d[0]) - pose[2]):.0f} deg")  # fmt: skip
    target = odom_to_world(start, p_odom)
    # 3. reach
    ik = ArmIK(env, base.controller)
    ik.mirror()
    mj_tip = ik.data.xpos[ik.arm.hand].copy()
    log("ik_check", f"MuJoCo mirror vs Isaac fingertip: "
        f"{np.linalg.norm(mj_tip - env.link_pose(HAND_LINK)[0]) * 100:.1f} cm")  # fmt: skip
    root, _ = env._root()
    yaw = env.true_pose()[2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    side = np.array([*root[:2], target[2]]) + fwd * 0.15 + right * 0.22  # beside the body
    tw = np.array([*u, 0.0])
    pre = target - tw * 0.08  # 8 cm in front of it
    for goal, label in ((side, "lower beside body"), (pre, "pre-grasp"), (target, "contact")):
        q_goal = ik.solve(goal)
        q0 = ik.arm.current_targets()
        for k in range(1, 26):  # 1 s: small steps the balance policy can absorb
            ik.arm.set_targets(q0 + (q_goal - q0) * k / 25)
            base.walk(0, 0, 0, 0.04)
        base.walk(0, 0, 0, 0.4)
        hand = env.link_pose(HAND_LINK)[0]
        log("reach", f"{label}: fingertip {np.linalg.norm(hand - goal) * 100:.1f} cm from goal")
    gap = float(np.linalg.norm(env.link_pose(HAND_LINK)[0] - target))
    out["reach_error_m"] = gap
    body = rigid_ancestor(stage, prim)
    out["target_body"] = body
    if body is None or gap > 0.06:
        log("result", f"cannot grasp: rigid body={body}, fingertip {gap * 100:.1f} cm away")
        return {**out, "success": False, "reason": "grasp not possible"}
    out["grasp"] = make_grasp(stage, env, body)
    base.walk(0, 0, 0, 0.3)
    log("grasp", f"assisted grasp of {body.split('/')[-1]}")
    from pxr import Usd, UsdGeom

    def body_pos():
        m = UsdGeom.Xformable(stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())  # fmt: skip
        return np.array(m.ExtractTranslation())

    before = body_pos()
    # 4. act
    if task == "open_cabinet":
        for _ in range(12):  # step back ~0.3 m; the hinge decides how the door swings
            base.walk(-0.12, 0.0, 0.0, 0.25)
        release(stage)
        base.walk(0, 0, 0, 0.3)
        ik.arm.set_targets(np.zeros(7))
        upright(base)
        base.walk(-0.1, 0, 0, 1.0)
        moved = float(np.linalg.norm(body_pos() - before))
        out.update(success=moved > 0.08, door_moved_m=moved)
        log("result", f"door moved {moved * 100:.0f} cm")
    else:
        lift = target + np.array([0, 0, 0.15])
        q_goal, q0 = ik.solve(lift), ik.arm.current_targets()
        for k in range(1, 26):
            ik.arm.set_targets(q0 + (q_goal - q0) * k / 25)
            base.walk(0, 0, 0, 0.04)
        lifted = float(body_pos()[2] - before[2])
        log("lift", f"banana lifted {lifted * 100:.0f} cm")
        upright(base)
        log("go", "carry it back to where I started")
        r = tb.call("go_to", {"x": 0.0, "y": 0.0, "heading": None})
        log("arrived", f"go_to {r.status} {r.message}")
        carried = float(np.linalg.norm(body_pos()[:2] - before[:2]))
        release(stage)
        base.walk(0, 0, 0, 0.5)
        out.update(success=lifted > 0.05 and carried > 0.5, lifted_m=lifted, carried_m=carried)
        log("result", f"lifted {lifted * 100:.0f} cm, carried {carried:.2f} m, released")
    out["fell"] = base.stop_reason is not None
    return out
