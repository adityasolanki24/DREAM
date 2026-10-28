"""A compartment with a hidden mechanism, built into the Panda tabletop scene.

Every variant looks the same: a closed box with a handle at its front-top edge. What moves is
hidden:
    drawer  the container slides out of a fixed housing whose roof covers the cavity
    lid     the top plate is hinged at the back edge and swings up
    fixed   no joint: the box cannot be opened (negative control)
While closed, the roof or lid physically covers the cavity, so nothing can be placed inside
until the compartment is opened. Housing and moving part are excluded from colliding with
each other (joint limits act as the closed and open stops). Everything else, including the
robot's fingers and the cube, collides with the full geometry.

``CompartmentSpec`` is ground truth. Like ``MechanismSpec``, it is only for building the scene and
scoring, never for decision-making code. Local frame: origin at the box's base centre, +x points
out of the front (toward the robot), +z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

KINDS = ("drawer", "lid", "fixed")
INNER_HALF = 0.09  # m, cavity half width (x and y)
WALL = 0.008  # m, wall and plate thickness
# m, floor top to roof underside. Rims must stay below the Panda hand housing, which is only
# 37 mm above the TCP (a cube released 1 cm above the floor puts it 67 mm up); 7 cm hit it.
CAVITY_HEIGHT = 0.05
FLOOR = 0.006  # m, floor plate thickness (the drawer floor rides this high above the table)
# A spherical knob on a central post, pinched with the fingers closing parallel to the front
# face. Flat pads on a sphere touch at one point each, so the knob pivots freely about the
# closing axis: a hand pointing down can follow a lid swinging up. A bar handle pinched across
# its thickness twisted in the fingers instead, and the Panda stalled at ~0.8 rad.
HANDLE_RADIUS = 0.0125  # m
HANDLE_GAP = 0.032  # m between the front face and the knob (clear of the fingers' width)
# The original bar handle (challenge variant ``handle="bar"``): a 25 x 60 x 25 mm block on two
# end posts, pinched across its 25 mm thickness.
BAR_HALF = (0.0125, 0.03, 0.0125)  # m
HANDLES = ("knob", "bar")
DRAWER_LIFT = 0.002  # m: the drawer rides on its slide, clear of the table
BODY_RGBA = (0.62, 0.5, 0.36, 1.0)


@dataclass(frozen=True)
class CompartmentSpec:
    """Ground truth. Privileged: never give this to decision-making code."""

    kind: str  # "drawer" | "lid" | "fixed"
    looks_like: str  # geometry of the fixed variant ("drawer" or "lid"); == kind otherwise
    xy: tuple[float, float]  # m, base centre on the table
    yaw: float  # rad, direction of local +x (the front) in the world
    upper: float  # m (drawer) or rad (lid), the open limit; the closed limit is 0
    friction: float  # N, Coulomb resistance referred to the handle
    damping: float  # N s/m or N m s/rad
    # Variants that keep the original, harder conditions reproducible (README audit table):
    handle: str = "knob"  # "bar": the original handle, which twisted in the grip on lids
    cavity_height: float = CAVITY_HEIGHT  # m; 0.07 was the original depth
    counterbalanced: bool = True  # lid only; False: gravity closes a released lid
    return_stiffness: float = 0.0  # N/m or N m/rad toward closed: out of the model class

    @property
    def front_dir(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    @property
    def outer_half(self) -> np.ndarray:
        return np.array([INNER_HALF + WALL, INNER_HALF + WALL, self.height / 2])

    @property
    def height(self) -> float:
        return FLOOR + self.cavity_height + WALL

    @property
    def handle_half(self) -> np.ndarray:
        """Visible half extent of the handle in its local frame (x out of the face)."""
        return np.array(BAR_HALF if self.handle == "bar" else [HANDLE_RADIUS] * 3)

    def to_world(self, p_local: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x, y, z = p_local
        return np.array([self.xy[0] + c * x - s * y, self.xy[1] + s * x + c * y, z])

    @property
    def handle_local(self) -> np.ndarray:
        """Handle centre in the closed state (local frame)."""
        hx, _, hz = self.handle_half
        return np.array([INNER_HALF + WALL + HANDLE_GAP + hx, 0.0, self.height - hz])

    @property
    def hinge_local(self) -> np.ndarray:
        """A point on the lid's hinge line (back top edge)."""
        return np.array([-(INNER_HALF + WALL), 0.0, self.height])

    def joint_axis_world(self) -> np.ndarray:
        """+q opens. Drawer: out of the front. Lid: about local -y, so the front edge rises."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        if self.kind == "drawer":
            return np.array([c, s, 0.0])
        if self.kind == "lid":
            return -np.array([-s, c, 0.0])
        return np.zeros(3)


def sample_compartment(
    kind: str,
    rng: np.random.Generator,
    looks_like: str | None = None,
    friction_range: tuple[float, float] = (0.5, 3.0),
    yaw_jitter_deg: float = 15.0,
    **variant,
) -> CompartmentSpec:
    """A compartment in reach of the Panda, front facing the robot base +- ``yaw_jitter_deg``.

    ``variant`` sets the challenge fields of ``CompartmentSpec`` (handle, cavity_height,
    counterbalanced, return_stiffness); the random draws do not depend on them.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if variant.get("handle", "knob") not in HANDLES:
        raise ValueError(f"unknown handle {variant['handle']!r}")
    dist = rng.uniform(0.6, 0.68)  # m from the robot base to the box centre
    bearing = rng.uniform(-0.35, 0.35)  # rad
    xy = (float(dist * np.cos(bearing)), float(dist * np.sin(bearing)))
    yaw = float(
        np.arctan2(-xy[1], -xy[0]) + np.deg2rad(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
    )
    if looks_like is None:
        looks_like = kind if kind != "fixed" else str(rng.choice(["drawer", "lid"]))
    upper = float(rng.uniform(0.16, 0.2)) if kind == "drawer" else float(rng.uniform(1.4, 1.8))
    return CompartmentSpec(
        kind=kind,
        looks_like=looks_like,
        xy=xy,
        yaw=yaw,
        upper=upper if kind != "fixed" else 0.0,
        friction=float(rng.uniform(*friction_range)),
        damping=2.0 if kind == "drawer" else 0.05,
        **variant,
    )


def _box(body, name: str, half, pos, rgba=BODY_RGBA, **kw):
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        pos=list(pos),
        rgba=list(rgba),
        **kw,
    )


def _add_handle(body, local_origin: np.ndarray, spec: CompartmentSpec) -> None:
    """Spherical knob on one thin central post (the fingers close beside the post), or the
    original bar on two thin end posts (the rear finger fits between bar and face)."""
    h = spec.handle_local - local_origin
    if spec.handle == "bar":
        _box(
            body,
            "handle",
            BAR_HALF,
            h,
            rgba=(0.2, 0.2, 0.22, 1),
            friction=[1.5, 0.005, 0.0001],
            condim=4,
        )
        post_half = (HANDLE_GAP / 2, 0.004, 0.006)
        for side in (-1, 1):
            post = h + [-(BAR_HALF[0] + HANDLE_GAP / 2), side * (BAR_HALF[1] - 0.004), 0]
            _box(body, f"handle_post{side + 1}", post_half, post, rgba=(0.2, 0.2, 0.22, 1))
        body.add_site(name="handle", pos=list(h), size=[0.006] * 3, rgba=[1, 0.5, 0, 0.6])
        return
    body.add_geom(
        name="handle",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[HANDLE_RADIUS, 0, 0],
        pos=list(h),
        rgba=[0.2, 0.2, 0.22, 1],
        friction=[1.5, 0.005, 0.0001],
        condim=3,  # no torsional friction: the knob pivots about the closing axis
    )
    post = h - [HANDLE_RADIUS / 2 + HANDLE_GAP / 2, 0, 0]
    _box(
        body,
        "handle_post",
        (HANDLE_GAP / 2 + HANDLE_RADIUS / 2, 0.004, 0.004),
        post,
        rgba=(0.2, 0.2, 0.22, 1),
    )
    body.add_site(name="handle", pos=list(h), size=[0.006] * 3, rgba=[1, 0.5, 0, 0.6])


def add_compartment(s: mujoco.MjSpec, spec: CompartmentSpec) -> None:
    """Add the housing and the moving part to an ``MjSpec`` (world body)."""
    quat = [float(np.cos(spec.yaw / 2)), 0.0, 0.0, float(np.sin(spec.yaw / 2))]
    housing = s.worldbody.add_body(name="housing", pos=[*spec.xy, 0.0], quat=quat)
    a, w, top = INNER_HALF, WALL, spec.height
    side_half = (a + w, w / 2, (top - w) / 2)
    back_half = (w / 2, a + w, (top - w) / 2)
    looks = spec.looks_like

    if looks == "drawer":
        # Fixed housing: sides, back, roof. The drawer (floor, walls, front face) slides out.
        for side in (-1, 1):
            _box(
                housing,
                f"housing_side{side + 1}",
                side_half,
                (0, side * (a + w / 2), (top - w) / 2),
            )
        _box(housing, "housing_back", back_half, (-(a + w / 2), 0, (top - w) / 2))
        _box(housing, "roof", (a + w, a + w, w / 2), (0, 0, top - w / 2))
        moving = housing.add_body(name="compartment_part", pos=[0, 0, DRAWER_LIFT])
        _box(
            moving,
            "cavity_floor",
            (a, a - 0.001, FLOOR / 2),
            (0, 0, FLOOR / 2),
            rgba=(0.5, 0.42, 0.3, 1),
            mass=0.3,
        )
        inner_wall_h = (spec.cavity_height - 0.01) / 2  # drawer walls stay 1 cm below the roof
        for side in (-1, 1):
            _box(
                moving,
                f"drawer_side{side + 1}",
                (a, 0.003, inner_wall_h),
                (0, side * (a - 0.004), FLOOR + inner_wall_h),
                mass=0.05,
            )
        _box(
            moving,
            "drawer_back",
            (0.003, a, inner_wall_h),
            (-(a - 0.004), 0, FLOOR + inner_wall_h),
            mass=0.05,
        )
        _box(moving, "front_face", (w / 2, a + w, top / 2), (a + w / 2, 0, top / 2), mass=0.2)
        _add_handle(moving, np.array([0.0, 0.0, DRAWER_LIFT]), spec)
    else:
        # Fixed open box: floor, four walls. The lid (top plate + front lip) closes it.
        _box(
            housing, "cavity_floor", (a, a, FLOOR / 2), (0, 0, FLOOR / 2), rgba=(0.5, 0.42, 0.3, 1)
        )
        for side in (-1, 1):
            _box(
                housing,
                f"housing_side{side + 1}",
                side_half,
                (0, side * (a + w / 2), (top - w) / 2),
            )
        _box(housing, "housing_back", back_half, (-(a + w / 2), 0, (top - w) / 2))
        _box(housing, "front_face", (w / 2, a + w, (top - w) / 2), (a + w / 2, 0, (top - w) / 2))
        hinge = spec.hinge_local
        moving = housing.add_body(name="compartment_part", pos=list(hinge))
        _box(moving, "roof", (a + w, a + w, w / 2), (a + w, 0, -w / 2), mass=0.2)
        _box(
            moving,
            "lid_lip",
            (w / 2, a + w, 0.006),
            (2 * (a + w) - w / 2, 0, -w - 0.006),
            mass=0.02,
        )
        _add_handle(moving, hinge, spec)

    if spec.kind == "drawer":
        moving.add_joint(
            name="compartment_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[0, spec.upper],
            frictionloss=spec.friction,
            damping=spec.damping,
            stiffness=spec.return_stiffness,
            solref_friction=[0.004, 1],
        )
    elif spec.kind == "lid":
        lever = np.linalg.norm((spec.handle_local - spec.hinge_local)[[0, 2]])
        moving.add_joint(
            name="compartment_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, -1, 0],
            range=[0, spec.upper],
            frictionloss=spec.friction * lever,
            damping=spec.damping,
            stiffness=spec.return_stiffness * lever**2,
            solref_friction=[0.004, 1],
        )
        if spec.counterbalanced:  # e.g. a gas strut: the lid stays where it is put
            moving.gravcomp = 1.0
    # The housing and moving part never collide; joint limits are the stops.
    s.add_exclude(bodyname1="housing", bodyname2="compartment_part")
