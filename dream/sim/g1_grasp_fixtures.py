"""Procedural rigid objects for the R1 tabletop contact benchmark."""

import mujoco
import numpy as np


def add_grasp_object(
    spec: mujoco.MjSpec,
    shape: str,
    center: tuple[float, float, float],
    half_extents: np.ndarray,
    mass: float,
    sliding_friction: float | None = None,
) -> None:
    """Add a free object, with a compound collision surface for a bottle or mug."""
    radius, width, height = half_extents
    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    if shape != "box" and (radius != width or (shape == "sphere" and radius != height)):
        raise ValueError("round fixtures require equal horizontal half extents")
    zero = (0.0, 0.0, 0.0)
    identity = (1.0, 0.0, 0.0, 0.0)
    if shape == "box":
        parts = [(box, half_extents, zero, identity)]
    elif shape == "cylinder":
        parts = [(cylinder, (radius, height, 0.0), zero, identity)]
    elif shape == "sphere":
        parts = [(sphere, (radius, 0.0, 0.0), zero, identity)]
    elif shape == "bottle":
        # The upper quarter is a narrower neck rather than a cylindrical proxy.
        parts = [
            (cylinder, (radius, 0.75 * height, 0.0), (0.0, 0.0, -0.25 * height), identity),
            (cylinder, (0.55 * radius, 0.25 * height, 0.0), (0.0, 0.0, 0.75 * height), identity),
        ]
    elif shape == "mug":
        # Twelve solid wall segments leave the cavity open. A three-piece
        # handle leaves a genuine gap for a finger rather than a solid block.
        bottom = 0.1 * height
        wall = 0.06 * radius
        ring = radius - wall
        parts = [(cylinder, (radius, bottom, 0.0), (0.0, 0.0, bottom - height), identity)]
        for angle in np.arange(12) * (np.pi / 6):
            parts.append((
                box, (wall, radius * np.tan(np.pi / 12), height - bottom),
                (ring * np.cos(angle), ring * np.sin(angle), bottom),
                (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)),
            ))  # fmt: skip
        stem = 0.14 * radius
        gap = 0.58 * radius
        for z in (-0.55 * height, 0.55 * height):
            parts.append((box, (stem, gap / 2, stem), (0.0, radius + gap / 2, z), identity))
        parts.append((
            box, (stem, stem, 0.55 * height), (0.0, radius + gap, 0.0), identity,
        ))  # fmt: skip
    else:
        raise ValueError(f"unsupported grasp fixture shape {shape!r}")

    body = spec.worldbody.add_body(name="box", pos=center)
    body.add_freejoint()
    if len(parts) == 1:
        weights = [1.0]
    else:
        # A shared density over the components fixes the total configured mass.
        volumes = np.array([
            8 * np.prod(size) if geom == box else 2 * np.pi * size[0] ** 2 * size[1]
            for geom, size, _, _ in parts
        ])  # fmt: skip
        weights = volumes / volumes.sum()
    material = {}
    if sliding_friction is not None:
        if not np.isfinite(sliding_friction) or sliding_friction <= 0:
            raise ValueError("sliding friction must be finite and positive")
        material["friction"] = (sliding_friction, 0.005, 0.0001)
        # Explicit surface friction defines the contact pair. Equal priorities
        # would let the larger coefficient of the table or hand dominate.
        material["priority"] = 1
    for index, ((geom, size, pos, quat), weight) in enumerate(zip(parts, weights, strict=True)):
        body.add_geom(
            name="box" if index == 0 else f"object_part_{index}",
            type=geom,
            size=size,
            pos=pos,
            quat=quat,
            mass=mass * weight,
            **material,
        )
