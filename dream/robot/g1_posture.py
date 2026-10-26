"""Whole-body posture planner for the free-standing G1: where to stand and how to hold the body
so a hand reaches a perceived point, found by optimisation over the full robot configuration.

Robot self-knowledge only: the planning model is NVIDIA's robot MJCF (g1_gear_wbc.xml, the one
paired with the Decoupled WBC policies) plus planning-only sites and temporary box geoms that
model the PERCEIVED obstacles handed in by the caller. Nothing here reads a simulator or a
scene, and nothing here knows a controller: limits come only from a ``ControllerCapabilities``
(dream/robot/wholebody.py) and results are ``WholeBodyTarget``s. Crouching, leaning and the
choice of arm are never coded as rules: every candidate (stance x arm) runs the same
differential IK with the same tasks and limits.

IK PROPOSES, PHYSICS DISPOSES (PLAN C5). Mink IK and the static gravity-torque proxy only RANK
candidates; passing them proves nothing about feasibility. The feasibility gate is a contact-
aware rollout of the real controller on a planning copy (dream/robot/g1_rollout.py), followed by
the monitored real execution, which is the only thing that counts as success.

IK (Mink 1.3.0, Apache-2.0, example humanoid_g1.py as the idiom), per candidate:
- feet FrameTasks pinned flat at the stance (position and orientation, heavy);
- hand FrameTask on the grip point (position; plus an AxisAlignTask only if an approach axis
  is given);
- ComTask holding the CoM xy at the centre of the support polygon;
- PostureTask toward the nominal standing pose (light, lighter on the reaching arm);
- ConfigurationLimit, pelvis-height and torso-rpy limits from the capabilities, and
  CollisionAvoidanceLimits for selected self-collision pairs, floor, and perceived boxes.
The IK runs with dt = 1 so one QP step is a (damped) Newton step in configuration space.

Each converged candidate gets a quasi-static prefilter (hand residual, feet on stance, clearance,
CoM inside the support polygon, joint limits, limits of the capabilities) and ranking terms (CoM
margin, joint-limit margin, static gravity-torque ratio, deviation from nominal, walking
distance).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mink
import mujoco
import numpy as np

from dream.robot.g1_wbc import DEFAULT, N_ACTION, N_BODY
from dream.robot.wholebody import (
    ControllerCapabilities,
    HandGoal,
    WholeBodyTarget,
    torso_rpy_from_matrices,
    yaw_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "third_party/gr00t_wbc_g1/g1_gear_wbc.xml"
MESHES = ROOT / "third_party/unitree_g1/meshes"  # same meshes dream.sim.g1_mobile uses

# Footprint radius used for navigation (dream/tools/mobile.py COLLISION_RADIUS), duplicated
# because the robot package must not import dream.tools; tests assert the two are equal.
FOOTPRINT_RADIUS = 0.27  # m
# An obstacle blocks a stance when it overlaps this height band: above a step the legs can
# clear (assumed 0.05 m) and below the head (assumed 1.5 m). DREAM planning assumption.
BLOCK_BAND = (0.05, 1.5)

HAND_TOLERANCE = 0.02  # m, a plan whose IK hand residual exceeds this is rejected
FOOT_TOLERANCE = 0.005  # m, feet must stay on their stance targets
MIN_CLEARANCE = 0.01  # m kept from perceived boxes and the floor by the IK
PERCEPTION_MARGIN = 0.02  # m, boxes are inflated by this for IK and stance tests (assumed)
HAND_COVARIANCE_SIGMAS = 2.0  # hand clearance grows by this many target std devs
COM_COMFORT = 0.03  # m, CoM margin below which the score starts to penalise
JOINT_COMFORT = 0.10  # fraction of joint range kept from limits before the score penalises

# Score weights (lower total cost is better). Design choices of the planner, not calibrated
# quantities; they rank feasible candidates and never gate feasibility.
W_WALK = 0.5  # per metre (plus 0.3 m per rad of turn) from the current pose
W_DEVIATION = 2.0  # per rad rms of leg+waist joints from nominal
W_COM = 1.0
W_JOINT = 1.0
W_TORQUE = 1.0  # per unit of static gravity-torque ratio
W_RESIDUAL = 0.5  # per hand-tolerance of residual
TURN_TO_METRES = 0.3

# IK task costs (mink convention: residual weight), documented in the module docstring.
COST_FOOT = 100.0
COST_HAND = 5.0
COST_AXIS = 1.0
COST_COM = 2.0
COST_POSTURE = 0.005
COST_POSTURE_REACHING_ARM = 0.001
COST_POSTURE_IDLE_ARM = 0.2  # the arm not reaching stays near its rest pose
COST_HOLD_ARM = 0.05
COST_HOLD_BODY = 10.0  # closed-loop refinement: legs stay at their measured values
COST_POSTURE_WAIST = 0.05  # comfort: twisting the torso is costlier than bending knees
COST_PELVIS_ORIENTATION = 5.0  # the controller commands no pelvis tilt: plan it upright
IK_GAIN = 0.8
IK_DAMPING = 1e-3
IK_ITERATIONS = 100
IK_STEP_TOL = 2e-4
N_OBSTACLE_SLOTS = 16
FAR = 1.0e3  # parking position of unused obstacle slots
ARMS = ("left", "right")
# Reach point = the tip of the index finger: the point of the hand that touches. (It was the mean
# of the index and middle distal-link ORIGINS; the finger geometry extends about 5 cm beyond
# those origins, so no surface could ever be touched with that point within 3 cm.)
GRIP_BODIES = ("index_1",)
TIP_VERTICES = 20  # the reach point averages the farthest this many mesh vertices


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class PerceivedBox:
    """An obstacle as perceived (RGB-D): world/odometry-frame centre, half extents, yaw.

    ``hand_may_touch`` marks the surface that carries the target: the hand geoms are exempt
    from clearance against it (the body still keeps clear)."""

    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw: float = 0.0
    hand_may_touch: bool = False


@dataclass(frozen=True)
class PlannerConfig:
    radii: tuple[float, ...] = (0.45, 0.6, 0.75)  # m, stance distance from the target (xy)
    n_angles: int = 8
    yaw_offsets: tuple[float, ...] = (0.0,)  # added to the heading that faces the target
    max_stances: int = 24
    arms: tuple[str, ...] = ARMS


@dataclass(frozen=True)
class PosturePlan:
    stance: tuple[float, float, float]  # pelvis x, y, yaw the feet are placed for
    arm: str
    target: WholeBodyTarget  # q, hand goal, derived pelvis height / torso rpy / 14 arm joints
    score: float  # ranking cost (lower is better), never a feasibility proof
    metrics: dict[str, float]

    @property
    def pelvis_height(self) -> float:
        return self.target.pelvis_height

    @property
    def torso_rpy(self) -> tuple[float, float, float]:
        return self.target.torso_rpy

    @property
    def arm_targets(self) -> np.ndarray:
        return self.target.arm_joints


@dataclass(frozen=True)
class PlanFailure:
    stance: tuple[float, float, float]
    arm: str
    reasons: tuple[str, ...]
    metrics: dict[str, float]


@dataclass
class PlanResult:
    plans: list[PosturePlan] = field(default_factory=list)  # best first
    failures: list[PlanFailure] = field(default_factory=list)
    n_stances_blocked: int = 0
    executable: bool = False  # the capabilities were measured

    @property
    def feasible(self) -> bool:
        return bool(self.plans)

    def reasons(self) -> dict[str, int]:
        """Histogram of failure reasons, for declaring a target unreachable with evidence."""
        out: dict[str, int] = {}
        for f in self.failures:
            for r in f.reasons:
                out[r] = out.get(r, 0) + 1
        if self.n_stances_blocked:
            out["stance_blocked"] = self.n_stances_blocked
        return out


# ---------------------------------------------------------------------------------------------
# Planning model


class _Limit(mink.Limit):
    """Shared base: nothing extra, kept to type the two custom limits below."""


class _PelvisHeightLimit(_Limit):
    def __init__(self, model: mujoco.MjModel, lo: float, hi: float, relax: bool = False):
        # relax: a measured start already outside the bound is not forced back (refinement)
        self.model, self.lo, self.hi, self.relax = model, lo, hi, relax
        self.pelvis = model.body("pelvis").id
        self._jac = np.zeros((3, model.nv))

    def compute_qp_inequalities(self, configuration, dt):
        d = configuration.data
        mujoco.mj_jacBody(self.model, d, self._jac, None, self.pelvis)
        z, row = d.xpos[self.pelvis][2], self._jac[2]
        return mink.Constraint(
            np.vstack([row, -row]),
            np.array([self.hi - z, z - self.lo]) * IK_GAIN
            if not self.relax
            else np.array([max(self.hi - z, 0.0), max(z - self.lo, 0.0)]) * IK_GAIN,
        )


class _TorsoRpyLimit(_Limit):
    """Bounds torso rpy (see torso_rpy_from_matrices); Jacobian by finite differences."""

    def __init__(self, model: mujoco.MjModel, lo, hi, relax: bool = False):
        self.model, self.lo, self.hi = model, np.asarray(lo), np.asarray(hi)
        self.relax = relax
        self.pelvis, self.torso = model.body("pelvis").id, model.body("torso_link").id
        self._scratch = mujoco.MjData(model)

    def rpy(self, data) -> np.ndarray:
        return torso_rpy_from_matrices(
            data.xmat[self.pelvis].reshape(3, 3), data.xmat[self.torso].reshape(3, 3)
        )

    def compute_qp_inequalities(self, configuration, dt):
        m, s, eps = self.model, self._scratch, 1e-6
        q0, rpy0 = configuration.data.qpos.copy(), self.rpy(configuration.data)
        jac = np.zeros((3, m.nv))
        for i in range(m.nv):
            dq = np.zeros(m.nv)
            dq[i] = eps
            s.qpos[:] = q0
            mujoco.mj_integratePos(m, s.qpos, dq, 1.0)
            mujoco.mj_kinematics(m, s)
            jac[:, i] = (self.rpy(s) - rpy0) / eps
        return mink.Constraint(
            np.vstack([jac, -jac]),
            np.concatenate([self.hi - rpy0, rpy0 - self.lo]) * IK_GAIN
            if not self.relax
            else np.concatenate([np.maximum(self.hi - rpy0, 0.0), np.maximum(rpy0 - self.lo, 0.0)])
            * IK_GAIN,
        )


def _load_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.meshdir = str(MESHES)
    return spec


@dataclass(frozen=True)
class _Anatomy:
    """Robot-intrinsic offsets read from the MJCF by forward kinematics at the nominal pose."""

    grip: dict[str, np.ndarray]  # side -> grip point in wrist_yaw_link frame
    sole: np.ndarray  # sole centre in ankle_roll_link frame
    sole_corners: np.ndarray  # (4, 3) contact-sphere centres in ankle_roll_link frame


def _nominal_qpos(model: mujoco.MjModel, height: float, x=0.0, y=0.0, yaw=0.0) -> np.ndarray:
    q = np.zeros(model.nq)
    q[0:3] = [x, y, height]
    q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    q[7 : 7 + N_ACTION] = DEFAULT  # legs and waist; the arms rest at joint zero
    return q


def _fingertip(model: mujoco.MjModel, data: mujoco.MjData, side: str, name: str) -> np.ndarray:
    """World point at the tip of a distal finger link at the current pose: the mean of the mesh
    vertices of its collision geoms farthest from the proximal link, along the finger axis."""
    body = model.body(f"{side}_hand_{name}_link").id
    proximal = model.body(f"{side}_hand_{name.replace('_1', '_0')}_link").id
    axis = data.xpos[body] - data.xpos[proximal]
    axis = axis / np.linalg.norm(axis)
    pts = []
    for g in np.flatnonzero(model.geom_bodyid == body):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not model.geom_contype[g]:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[
            model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
        ]
        pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    pts = np.concatenate(pts)
    far = np.argsort((pts - data.xpos[body]) @ axis)[-TIP_VERTICES:]
    return pts[far].mean(axis=0)


def _read_anatomy() -> _Anatomy:
    model = _load_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:] = _nominal_qpos(model, 0.8)
    mujoco.mj_forward(model, data)
    grip = {}
    for side in ARMS:
        wrist = model.body(f"{side}_wrist_yaw_link").id
        tips = [_fingertip(model, data, side, n) for n in GRIP_BODIES]
        grip[side] = data.xmat[wrist].reshape(3, 3).T @ (np.mean(tips, axis=0) - data.xpos[wrist])
    foot = model.body("left_ankle_roll_link").id
    spheres = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == foot and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    corners = np.array([model.geom_pos[g] for g in spheres])
    radius = float(model.geom_size[spheres[0], 0])
    sole = corners.mean(axis=0) - np.array([0.0, 0.0, radius])
    return _Anatomy(grip, sole, corners)


class PlanningModel:
    """Robot-only model, hand/foot sites, and parked box slots for perceived obstacles."""

    def __init__(self) -> None:
        self.anatomy = _read_anatomy()
        spec = _load_spec()
        for side in ARMS:
            spec.body(f"{side}_wrist_yaw_link").add_site(
                name=f"{side}_grip", pos=list(self.anatomy.grip[side]), size=[0.01] * 3
            )
            spec.body(f"{side}_ankle_roll_link").add_site(
                name=f"{side}_sole", pos=list(self.anatomy.sole), size=[0.01] * 3
            )
        box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, contype=1, conaffinity=1, group=4)
        spec.worldbody.add_geom(
            name="floor_box", pos=[0, 0, -0.5], size=[6.0, 6.0, 0.5], **box
        )  # top face at z = 0: clearance from the floor for the upper body
        for i in range(N_OBSTACLE_SLOTS):
            spec.worldbody.add_geom(
                name=f"obstacle_{i}", pos=[FAR, FAR, FAR], size=[0.1] * 3, **box
            )
        self.model = spec.compile()
        m = self.model
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso_link").id
        joints = [m.actuator(i).trnid[0] for i in range(N_BODY)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([m.jnt_dofadr[j] for j in joints])
        self.torque_limit = m.jnt_actfrcrange[joints, 1].copy()
        self.joint_range = m.jnt_range[joints].copy()
        data = mujoco.MjData(m)
        data.qpos[:] = _nominal_qpos(m, 0.8)
        mujoco.mj_forward(m, data)
        sole_z = min(data.site_xpos[m.site(f"{s}_sole").id][2] for s in ARMS)
        self.nominal_height = 0.8 - sole_z  # pelvis height with both soles on the floor
        self.q_nominal = _nominal_qpos(m, self.nominal_height)
        d = self.data_at(self.q_nominal)
        pelvis_r = d.xmat[self.pelvis].reshape(3, 3)
        self.foot_offset = {  # sole centre relative to the pelvis, in the pelvis (yaw) frame
            s: pelvis_r.T @ (d.site_xpos[m.site(f"{s}_sole").id] - d.xpos[self.pelvis])
            for s in ARMS
        }
        self.foot_rot = {
            s: pelvis_r.T @ d.site_xmat[m.site(f"{s}_sole").id].reshape(3, 3) for s in ARMS
        }
        self._init_geoms()

    def data_at(self, q: np.ndarray) -> mujoco.MjData:
        d = mujoco.MjData(self.model)
        d.qpos[:] = q
        mujoco.mj_forward(self.model, d)
        return d

    # -- collision geometry -------------------------------------------------------------
    def _geoms(self, *bodies: str, subtree: bool = False) -> list[int]:
        m, out = self.model, []
        for b in bodies:
            bid = m.body(b).id
            ids = (
                mink.get_subtree_geom_ids(m, bid)
                if subtree
                else list(np.flatnonzero(m.geom_bodyid == bid))
            )
            out += [g for g in ids if m.geom_contype[g] and m.geom_group[g] != 4]
        return out

    def _init_geoms(self) -> None:
        both = lambda fmt: [fmt.format(s=s) for s in ARMS]  # noqa: E731
        self.hand_geoms = {s: self._geoms(f"{s}_wrist_yaw_link", subtree=True) for s in ARMS}
        self.arm_geoms = {
            s: self._geoms(f"{s}_elbow_link", subtree=True)
            for s in ARMS  # elbow and distal
        }
        shoulder = [
            g
            for s in ARMS
            for b in (
                f"{s}_shoulder_pitch_link",
                f"{s}_shoulder_roll_link",
                f"{s}_shoulder_yaw_link",
            )
            for g in self._geoms(b)
        ]
        self.upper_body = self._geoms("pelvis", "torso_link") + shoulder
        self.thighs = self._geoms(*both("{s}_hip_yaw_link"), *both("{s}_knee_link"))
        self.no_foot = self._geoms(
            "pelvis", "torso_link", *both("{s}_hip_pitch_link"), *both("{s}_hip_roll_link"),
            *both("{s}_hip_yaw_link"), *both("{s}_knee_link"), *both("{s}_shoulder_pitch_link"),
            *both("{s}_shoulder_roll_link"), *both("{s}_shoulder_yaw_link"),
        ) + [g for s in ARMS for g in self.arm_geoms[s]]  # fmt: skip
        self.floor_geom = self.model.geom("floor_box").id
        self.slot_geoms = [self.model.geom(f"obstacle_{i}").id for i in range(N_OBSTACLE_SLOTS)]

    def self_pairs(self) -> list[tuple[list[int], list[int]]]:
        """Self-collision pairs that matter for reaching; pairs already closer than the
        clearance at the nominal pose are dropped (they are adjacent by construction)."""
        m = self.model
        torso, pelvis = self._geoms("torso_link"), self._geoms("pelvis")
        cand = []
        for s, o in (("left", "right"), ("right", "left")):
            cand.append((self.arm_geoms[s], torso + pelvis + self.arm_geoms[o] + self.thighs))
        cand.append((torso, self.thighs))
        d = self.data_at(self.q_nominal)
        out, fromto = [], np.zeros(6)
        for a_set, b_set in cand:
            a_ok = []
            for a in a_set:
                far = all(
                    mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto) > MIN_CLEARANCE or a == b
                    for b in b_set
                )
                if far:
                    a_ok.append(a)
            b_ok = [
                b
                for b in b_set
                if all(
                    mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto) > MIN_CLEARANCE for a in a_ok
                )
            ]
            if a_ok and b_ok:
                out.append((a_ok, b_ok))
        return out

    def kinematic_state(self, quat, q_body, feet_contact=(True, True)) -> dict:
        """Robot-frame state from the robot's own sensors (IMU quaternion, 29 joint encoders):
        pelvis height above the lowest sole, torso rpy, hand grip points and CoM relative to the
        pelvis xy in the pelvis-yaw frame, and the CoM margin to the support polygon of the feet
        in contact (both, if no contact information is given). Assumes flat feet."""
        m = self.model
        q = self.q_nominal.copy()
        q[0:3] = [0.0, 0.0, 1.0]
        q[3:7] = quat
        q[self.qadr] = q_body
        d = self.data_at(q)
        soles = {s: d.site_xpos[m.site(f"{s}_sole").id] for s in ARMS}
        floor = min(z[2] for z in soles.values())
        yaw = math.atan2(d.xmat[self.pelvis][3], d.xmat[self.pelvis][0])
        back = yaw_matrix(yaw).T
        origin = np.array([d.xpos[self.pelvis][0], d.xpos[self.pelvis][1], floor])
        rel = lambda p: back @ (np.asarray(p) - origin)  # noqa: E731
        pts = []
        for s, on in zip(ARMS, feet_contact, strict=True):
            if on:
                site = m.site(f"{s}_sole").id
                r = d.site_xmat[site].reshape(3, 3)
                pts += [
                    d.site_xpos[site] + r @ (c - self.anatomy.sole)
                    for c in self.anatomy.sole_corners
                ]
        com = d.subtree_com[1]
        margin = polygon_margin(com[:2], _hull(np.array(pts)[:, :2])) if len(pts) >= 3 else -1.0
        return {
            "height": float(d.xpos[self.pelvis][2] - floor),
            "rpy": torso_rpy_from_matrices(
                d.xmat[self.pelvis].reshape(3, 3), d.xmat[self.torso].reshape(3, 3)
            ),
            "hands": {s: rel(d.site_xpos[m.site(f"{s}_grip").id]) for s in ARMS},
            "com": rel(com),
            "com_margin": float(margin),
        }

    def load_obstacles(self, boxes: list[PerceivedBox]) -> list[int]:
        """Place perceived boxes (inflated by PERCEPTION_MARGIN) in the slots; return used ids."""
        if len(boxes) > N_OBSTACLE_SLOTS:
            raise ValueError(f"at most {N_OBSTACLE_SLOTS} perceived boxes are supported")
        m = self.model
        m.geom_pos[self.slot_geoms] = FAR
        used = []
        for gid, b in zip(self.slot_geoms, boxes, strict=False):
            m.geom_pos[gid] = b.center
            m.geom_size[gid] = np.asarray(b.half_extents) + PERCEPTION_MARGIN
            m.geom_quat[gid] = [math.cos(b.yaw / 2), 0, 0, math.sin(b.yaw / 2)]
            used.append(gid)
        return used


# ---------------------------------------------------------------------------------------------
# Geometry helpers


def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross2(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def polygon_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance of ``point`` inside a counter-clockwise convex polygon (+ inside)."""
    best = math.inf
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = b - a
        n = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        best = min(best, float(n @ (point - a)))
    return best


def _box_circle_distance(center, half, yaw, p) -> float:
    """Distance from planar point p to a yawed box footprint (0 inside)."""
    d = yaw_matrix(yaw).T[:2, :2] @ (np.asarray(p) - np.asarray(center[:2]))
    out = np.maximum(np.abs(d) - np.asarray(half[:2]), 0.0)
    return float(np.linalg.norm(out))


def stance_blocked(xy, boxes: list[PerceivedBox]) -> bool:
    for b in boxes:
        zlo, zhi = b.center[2] - b.half_extents[2], b.center[2] + b.half_extents[2]
        if zhi < BLOCK_BAND[0] or zlo > BLOCK_BAND[1]:
            continue
        if (
            _box_circle_distance(b.center, b.half_extents, b.yaw, xy)
            < FOOTPRINT_RADIUS + PERCEPTION_MARGIN
        ):
            return True
    return False


def static_torque_ratio(pm: PlanningModel, q: np.ndarray) -> float:
    """Max |joint torque| / actuator limit holding ``q`` statically on two flat feet.

    Gravity generalised force from MuJoCo's recursive Newton-Euler (qvel = 0), then the foot
    wrenches (6 per sole) that balance the floating base are chosen to minimise the
    limit-normalised joint torques (equality-constrained least squares)."""
    m = pm.model
    d = pm.data_at(q)
    bias = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 0, bias)
    jac = []
    for s in ARMS:
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, m.site(f"{s}_sole").id)
        jac.append(np.vstack([jp, jr]))
    j = np.vstack(jac)  # 12 x nv, generalised force of wrench w is j.T @ w
    a, b = j[:, :6].T, bias[:6]  # base rows: bias_b = a @ w
    w0 = np.linalg.lstsq(a, b, rcond=None)[0]
    null = np.linalg.svd(a)[2][np.linalg.matrix_rank(a) :].T
    jj = j[:, pm.vadr].T / pm.torque_limit[:, None]  # normalised joint torque map: bias - jj w
    g = bias[pm.vadr] / pm.torque_limit
    z = np.linalg.lstsq(jj @ null, g - jj @ w0, rcond=None)[0] if null.size else np.zeros(0)
    w = w0 + null @ z
    return float(np.max(np.abs(g - jj @ w)))


# ---------------------------------------------------------------------------------------------
# Planner


def _support_points(pm: PlanningModel, d: mujoco.MjData) -> np.ndarray:
    """World xy of the eight sole contact-sphere centres (the support polygon vertices)."""
    pts = []
    for s in ARMS:
        site = pm.model.site(f"{s}_sole").id
        r = d.site_xmat[site].reshape(3, 3)
        pts += [d.site_xpos[site] + r @ (c - pm.anatomy.sole) for c in pm.anatomy.sole_corners]
    return np.array(pts)[:, :2]


class PosturePlanner:
    def __init__(self, capabilities: ControllerCapabilities, config: PlannerConfig | None = None):
        self.caps = capabilities
        self.config = config or PlannerConfig()
        self.pm = PlanningModel()
        self._self_pairs = self.pm.self_pairs()
        self.last_timings: list[float] = []  # seconds per evaluated (stance, arm) candidate

    # -- candidate generation -----------------------------------------------------------
    def stances(self, target, robot_pose, boxes) -> tuple[list[tuple[float, float, float]], int]:
        """Stances on rings around the target that face it and clear every perceived box,
        nearest (walking cost) first, truncated to ``config.max_stances``."""
        c, tx, ty = self.config, float(target[0]), float(target[1])
        out, blocked = [], 0
        for r in c.radii:
            for k in range(c.n_angles):
                a = 2 * math.pi * k / c.n_angles
                x, y = tx + r * math.cos(a), ty + r * math.sin(a)
                if stance_blocked((x, y), boxes):
                    blocked += 1
                    continue
                face = math.atan2(ty - y, tx - x)
                out += [(x, y, wrap(face + o)) for o in c.yaw_offsets]
        out.sort(key=lambda s: self._walk(robot_pose, s))
        return out[: c.max_stances], blocked

    @staticmethod
    def _walk(pose, stance) -> float:
        return math.hypot(stance[0] - pose[0], stance[1] - pose[1]) + TURN_TO_METRES * abs(
            wrap(stance[2] - pose[2])
        )

    def _feet_targets(self, stance) -> dict[str, mink.SE3]:
        x, y, yaw = stance
        r = yaw_matrix(yaw)
        out = {}
        for s in ARMS:
            p = np.array([x, y, 0.0]) + r @ self.pm.foot_offset[s]
            p[2] = 0.0  # sole on the floor
            rot = mink.SO3.from_matrix(r @ self.pm.foot_rot[s])
            out[s] = mink.SE3.from_rotation_and_translation(rot, p)
        return out

    # -- IK -----------------------------------------------------------------------------
    def _limits(self, boxes_regular, boxes_touch, sigma, relax=False):
        pm, m, caps = self.pm, self.pm.model, self.caps
        hands = {g for s in ARMS for g in pm.hand_geoms[s]}
        body = [g for g in pm.no_foot if g not in hands]
        near = dict(
            minimum_distance_from_collisions=MIN_CLEARANCE, collision_detection_distance=0.1
        )
        limits = [
            mink.ConfigurationLimit(m),
            _PelvisHeightLimit(m, *caps.height_range, relax),
            _TorsoRpyLimit(m, caps.torso_rpy_min, caps.torso_rpy_max, relax),
        ]
        if self._self_pairs:
            limits.append(mink.CollisionAvoidanceLimit(m, self._self_pairs, **near))
        limits.append(mink.CollisionAvoidanceLimit(m, [(pm.no_foot, [pm.floor_geom])], **near))
        if boxes_regular or boxes_touch:
            pairs = [(body, boxes_regular + boxes_touch)]
            limits.append(mink.CollisionAvoidanceLimit(m, pairs, **near))
        if boxes_regular:
            wide = MIN_CLEARANCE + HAND_COVARIANCE_SIGMAS * sigma
            hand_near = dict(near, minimum_distance_from_collisions=wide)
            pairs = [(sorted(hands), boxes_regular)]
            limits.append(mink.CollisionAvoidanceLimit(m, pairs, **hand_near))
        return limits

    def _ik(
        self,
        q0,
        stance,
        arm,
        target,
        approach,
        limits,
        iterations: int = IK_ITERATIONS,
        feet=None,
        hold_body=False,
        hold_waist=False,
    ) -> np.ndarray | None:
        pm, m = self.pm, self.pm.model
        feet = feet or self._feet_targets(stance)
        conf = mink.Configuration(m, q0.copy())
        tasks = []
        for s in ARMS:
            t = mink.FrameTask(f"{s}_sole", "site", COST_FOOT, COST_FOOT, gain=IK_GAIN)
            t.set_target(feet[s])
            tasks.append(t)
        hand = mink.FrameTask(f"{arm}_grip", "site", COST_HAND, 0.0, gain=IK_GAIN)
        hand.set_target(mink.SE3.from_translation(np.asarray(target, float)))
        tasks.append(hand)
        if approach is not None:  # finger direction (grip frame +x) along the approach axis
            axis = mink.AxisAlignTask(f"{arm}_grip", "site", (1.0, 0.0, 0.0), COST_AXIS, IK_GAIN)
            axis.set_target(np.asarray(approach, float) / np.linalg.norm(approach))
            tasks.append(axis)
        pelvis = mink.FrameTask(
            "pelvis", "body", 0.0, 0.0 if hold_body else COST_PELVIS_ORIENTATION, gain=IK_GAIN
        )
        rot = mink.SO3.from_matrix(yaw_matrix(stance[2]))
        pelvis.set_target(mink.SE3.from_rotation_and_translation(rot, np.zeros(3)))
        tasks.append(pelvis)
        centre = np.mean([f.translation()[:2] for f in feet.values()], axis=0)
        com = mink.ComTask(
            np.array([COST_COM, COST_COM, 0.0]) * (0.0 if hold_body else 1.0), gain=IK_GAIN
        )
        com.set_target(np.array([*centre, 0.0]))
        tasks.append(com)
        cost = np.full(m.nv, COST_POSTURE)
        cost[pm.vadr[N_ACTION : N_ACTION + 14]] = COST_POSTURE_IDLE_ARM
        i0 = N_ACTION + (0 if arm == "left" else 7)
        cost[pm.vadr[i0 : i0 + 7]] = COST_POSTURE_REACHING_ARM
        cost[pm.vadr[12:15]] = COST_POSTURE_WAIST
        if (
            hold_body
        ):  # closed-loop update: legs stay measured; the waist (torso rpy command) and arm may move
            cost[pm.vadr[:12]] = COST_HOLD_BODY  # legs: the stance and height stay as measured
            cost[pm.vadr[i0 : i0 + 7]] = COST_HOLD_ARM  # min-norm step: no redundancy drift
            if hold_waist:  # the torso rpy command stays as it is: only the arm moves
                cost[pm.vadr[12:15]] = COST_HOLD_BODY
        posture = mink.PostureTask(m, cost)
        posture.set_target(q0 if hold_body else pm.q_nominal)
        tasks.append(posture)
        for _ in range(iterations):
            try:
                v = mink.solve_ik(conf, tasks, 1.0, "daqp", damping=IK_DAMPING, limits=limits)
            except mink.NoSolutionFound:
                return None
            conf.integrate_inplace(v, 1.0)
            if np.abs(v).max() < IK_STEP_TOL:
                break
        return conf.q.copy()

    # -- evaluation ---------------------------------------------------------------------
    def _min_clearance(self, d, boxes_regular, boxes_touch) -> float:
        pm, m = self.pm, self.pm.model
        hands = {g for s in ARMS for g in pm.hand_geoms[s]}
        fromto, best = np.zeros(6), 1.0

        def dist(a, b):
            return mujoco.mj_geomDistance(m, d, a, b, 0.1, fromto)

        for a_set, b_set in self._self_pairs:
            best = min(best, min(dist(a, b) for a in a_set for b in b_set))
        for g in pm.no_foot:
            best = min(best, dist(g, pm.floor_geom))
            for b in boxes_regular:
                best = min(best, dist(g, b))
            if g not in hands:
                for b in boxes_touch:
                    best = min(best, dist(g, b))
        return best

    def evaluate(self, q, stance, arm, target, boxes_regular=(), boxes_touch=()):
        """Metrics, torso rpy and failure reasons of a configuration (verifies IK output)."""
        pm, m, caps = self.pm, self.pm.model, self.caps
        d = pm.data_at(q)
        hand = d.site_xpos[m.site(f"{arm}_grip").id]
        feet = self._feet_targets(stance)
        foot_err = max(
            float(np.linalg.norm(d.site_xpos[m.site(f"{s}_sole").id] - feet[s].translation()))
            for s in ARMS
        )
        height = float(d.xpos[pm.pelvis][2])
        rpy = torso_rpy_from_matrices(
            d.xmat[pm.pelvis].reshape(3, 3), d.xmat[pm.torso].reshape(3, 3)
        )
        hull = _hull(_support_points(pm, d))
        com_margin = polygon_margin(d.subtree_com[1][:2], hull)
        qj = q[pm.qadr]
        lo, hi = pm.joint_range[:, 0], pm.joint_range[:, 1]
        joint_margin = float(np.min(np.minimum(qj - lo, hi - qj) / (hi - lo)))
        ratio = static_torque_ratio(pm, q)
        clearance = self._min_clearance(d, list(boxes_regular), list(boxes_touch))
        nominal = pm.q_nominal[pm.qadr][:N_ACTION]
        deviation = float(np.sqrt(np.mean((qj[:N_ACTION] - nominal) ** 2)))
        z_pelvis = d.xmat[pm.pelvis].reshape(3, 3)[2, 2]
        metrics = {
            "hand_residual": float(np.linalg.norm(hand - target)),
            "foot_error": foot_err,
            "com_margin": float(com_margin),
            "joint_margin": joint_margin,
            "torque_ratio": ratio,
            "clearance": float(clearance),
            "deviation": deviation,
            "pelvis_height": height,
            "pelvis_tilt": float(math.acos(np.clip(z_pelvis, -1, 1))),
        }
        reasons = []
        if metrics["hand_residual"] > HAND_TOLERANCE:
            reasons.append("hand_residual")
        if metrics["pelvis_tilt"] > caps.max_pelvis_tilt:
            reasons.append("pelvis_tilt")
        if foot_err > FOOT_TOLERANCE:
            reasons.append("feet_off_stance")
        if clearance < -1e-3:
            reasons.append("collision")
        if com_margin < 0:
            reasons.append("com_outside_support")
        if joint_margin < -1e-3:
            reasons.append("joint_limit")
        if caps.violations(height, rpy):
            reasons.append("outside_capabilities")
        return metrics, rpy, reasons

    @staticmethod
    def _score(m: dict[str, float], walk: float) -> float:
        return (
            W_WALK * walk
            + W_DEVIATION * m["deviation"]
            + W_COM * max(0.0, COM_COMFORT - m["com_margin"]) / COM_COMFORT
            + W_JOINT * max(0.0, JOINT_COMFORT - m["joint_margin"]) / JOINT_COMFORT
            + W_TORQUE * m["torque_ratio"]
            + W_RESIDUAL * m["hand_residual"] / HAND_TOLERANCE
        )

    def _q_start(self, stance) -> np.ndarray:
        q = self.pm.q_nominal.copy()
        q[0:2] = stance[:2]
        q[3:7] = [math.cos(stance[2] / 2), 0, 0, math.sin(stance[2] / 2)]
        return q

    def refine(
        self,
        q_measured,
        stance,
        arm,
        point,
        obstacles=(),
        approach_axis=None,
        iterations=30,
        hold_waist=False,
    ) -> np.ndarray | None:
        """Small closed-loop IK update from a MEASURED configuration (``q_measured``: full qpos
        built from IMU, encoders and odometry). Stance stays fixed (feet pinned there), limits
        are the capabilities' and the perceived obstacles'. Returns the new qpos or None."""
        boxes = list(obstacles)
        ids = self.pm.load_obstacles(boxes)
        regular = [g for g, b in zip(ids, boxes, strict=True) if not b.hand_may_touch]
        touch = [g for g, b in zip(ids, boxes, strict=True) if b.hand_may_touch]
        limits = self._limits(regular, touch, 0.0, relax=True)
        pt = np.asarray(point, float)
        d = self.pm.data_at(q_measured)  # feet stay where they are: the stance is fixed
        feet = {}
        for side in ARMS:
            sid = self.pm.model.site(f"{side}_sole").id
            rot = mink.SO3.from_matrix(d.site_xmat[sid].reshape(3, 3))
            feet[side] = mink.SE3.from_rotation_and_translation(rot, d.site_xpos[sid].copy())
        return self._ik(
            q_measured,
            stance,
            arm,
            pt,
            approach_axis,
            limits,
            iterations,
            feet,
            hold_body=True,
            hold_waist=hold_waist,
        )

    def retarget(
        self, plan: PosturePlan, point, obstacles=(), iterations=60, approach_axis=None
    ) -> PosturePlan | None:
        """The same stance, arm, legs, pelvis height and torso rpy as ``plan``, with only the arm
        moved so the hand is at ``point``. Used for a pre-contact pose a few centimetres in
        front of the plan's target: one body posture for both stages, so the approach is a small
        arm motion. None if the result fails the same checks a plan passes."""
        boxes = list(obstacles)
        ids = self.pm.load_obstacles(boxes)
        regular = [g for g, b in zip(ids, boxes, strict=True) if not b.hand_may_touch]
        touch = [g for g, b in zip(ids, boxes, strict=True) if b.hand_may_touch]
        pt = np.asarray(point, float)
        q = self.refine(
            plan.target.q,
            plan.stance,
            plan.arm,
            pt,
            boxes,
            approach_axis,
            iterations=iterations,
            hold_waist=True,
        )
        if q is None:
            return None
        metrics, rpy, reasons = self.evaluate(q, plan.stance, plan.arm, pt, regular, touch)
        if reasons:
            return None
        metrics["walk"] = plan.metrics.get("walk", 0.0)
        metrics["target_sigma"] = plan.metrics.get("target_sigma", 0.0)
        arms14 = q[self.pm.qadr[N_ACTION : N_ACTION + 14]].copy()
        goal = HandGoal(plan.arm, tuple(float(v) for v in pt))
        whole = WholeBodyTarget(
            q, plan.stance, (goal,), metrics["pelvis_height"], tuple(float(v) for v in rpy), arms14
        )
        return PosturePlan(
            plan.stance, plan.arm, whole, self._score(metrics, metrics["walk"]), metrics
        )

    def measured_qpos(self, prop, pose, stance_height) -> np.ndarray:
        """Full qpos from the robot's own sensors: IMU quaternion, joint encoders, odometry
        xy and the FK pelvis height above the lowest sole (flat feet on the floor)."""
        q = self.pm.q_nominal.copy()
        q[3:7] = prop.quat
        q[self.pm.qadr] = prop.q
        q[0:2] = pose[:2]
        q[2] = stance_height
        return q

    # -- public -------------------------------------------------------------------------
    def plan(
        self,
        target,
        covariance=None,
        robot_pose=(0.0, 0.0, 0.0),
        obstacles=(),
        approach_axis=None,
        stances=None,
        arms=None,
    ) -> PlanResult:
        """Rank feasible (stance, arm) postures reaching ``target`` (odometry/world frame).

        ``obstacles`` are PERCEIVED boxes only. The result lists failures with reasons.
        ``stances``: explicit (x, y, yaw) candidates (for example the robot's measured pose,
        where it already stands); default: sample rings around the target. ``arms``: restrict the
        candidate arms (default: ``config.arms``)."""
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target must be a finite 3-vector")
        cov = np.zeros((3, 3)) if covariance is None else np.asarray(covariance, float)
        sigma = float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        boxes = list(obstacles)
        ids = self.pm.load_obstacles(boxes)
        regular = [g for g, b in zip(ids, boxes, strict=True) if not b.hand_may_touch]
        touch = [g for g, b in zip(ids, boxes, strict=True) if b.hand_may_touch]
        if stances is None:
            stances, blocked = self.stances(target, robot_pose, boxes)
        else:
            stances, blocked = [tuple(float(v) for v in st) for st in stances], 0
        result = PlanResult(n_stances_blocked=blocked, executable=self.caps.executable)
        self.last_timings = []
        limits = self._limits(regular, touch, sigma)
        for stance in stances:
            for arm in arms or self.config.arms:
                t0 = time.perf_counter()
                q = self._ik(self._q_start(stance), stance, arm, target, approach_axis, limits)
                if q is None:
                    result.failures.append(PlanFailure(stance, arm, ("solver",), {}))
                    self.last_timings.append(time.perf_counter() - t0)
                    continue
                metrics, rpy, reasons = self.evaluate(q, stance, arm, target, regular, touch)
                walk = self._walk(robot_pose, stance)
                metrics["walk"] = walk
                metrics["target_sigma"] = sigma
                self.last_timings.append(time.perf_counter() - t0)
                if reasons:
                    result.failures.append(PlanFailure(stance, arm, tuple(reasons), metrics))
                    continue
                arms14 = q[self.pm.qadr[N_ACTION : N_ACTION + 14]].copy()
                goal = HandGoal(
                    arm, tuple(float(v) for v in target),
                    None if approach_axis is None else tuple(float(v) for v in approach_axis),
                )  # fmt: skip
                whole = WholeBodyTarget(
                    q, stance, (goal,), metrics["pelvis_height"],
                    tuple(float(v) for v in rpy), arms14,
                )  # fmt: skip
                result.plans.append(
                    PosturePlan(stance, arm, whole, self._score(metrics, walk), metrics)
                )
        result.plans.sort(key=lambda p: p.score)
        return result
