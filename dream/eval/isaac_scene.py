"""Ground truth of an externally authored USD scene, for scene setup and scoring only.

Reads the scene's own semantic labels, collision geometry and physics joints with pxr. The
agent never receives any of it (invariant 1): it is used to pick a free start pose by a
rule fixed in the config, to audit whether the scene is genuinely interactive, and to score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

CELL = 0.05  # m, start-pose occupancy grid


@dataclass(frozen=True)
class LabelledObject:
    path: str
    label: str
    lo: np.ndarray  # world AABB min (3,)
    hi: np.ndarray  # world AABB max (3,)

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


def _bbox(cache, prim) -> tuple[np.ndarray, np.ndarray]:
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(box.GetMin(), float), np.array(box.GetMax(), float)


def labelled_objects(stage, exclude_prefix: str = "/World/G1") -> list[LabelledObject]:
    """Every prim carrying a class semantic label (the scene author's ground truth)."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(exclude_prefix):
            continue
        attr = prim.GetAttribute("primvars:semantics:class:semanticData")
        if attr and attr.HasAuthoredValue():
            lo, hi = _bbox(cache, prim)
            if np.isfinite(lo).all() and np.isfinite(hi).all():
                out.append(LabelledObject(path, str(attr.Get()), lo, hi))
    return out


def physics_audit(stage, exclude_prefix: str = "/World/G1") -> dict:
    """Counts that distinguish an interactive physics scene from a visual one."""
    from pxr import UsdPhysics

    c = Counter()
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(exclude_prefix):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            c["colliders"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            c["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            c["articulation_roots"] += 1
        for kind in ("RevoluteJoint", "PrismaticJoint", "FixedJoint", "SphericalJoint"):
            if prim.IsA(getattr(UsdPhysics, kind)):
                c[kind] += 1
    return dict(c)


def obstacle_boxes(stage, z_range=(0.02, 1.9), exclude_prefix="/World/G1", floors=()) -> list:
    """World AABBs of colliders that a walking robot could hit (conservative footprints)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
    boxes = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(exclude_prefix) or any(path.startswith(f) for f in floors):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        lo, hi = _bbox(cache, prim)
        if np.isfinite(lo).all() and hi[2] > z_range[0] and lo[2] < z_range[1]:
            boxes.append((lo, hi))
    return boxes


def free_starts(boxes, floor_lo, floor_hi, clearance: float) -> np.ndarray:
    """Cell centres on the floor at least ``clearance`` from every obstacle footprint."""
    xs = np.arange(floor_lo[0] + CELL / 2, floor_hi[0], CELL)
    ys = np.arange(floor_lo[1] + CELL / 2, floor_hi[1], CELL)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d = np.minimum(
        np.minimum(gx - floor_lo[0], floor_hi[0] - gx),
        np.minimum(gy - floor_lo[1], floor_hi[1] - gy),
    )
    for lo, hi in boxes:
        dx = np.maximum(np.maximum(lo[0] - gx, gx - hi[0]), 0)
        dy = np.maximum(np.maximum(lo[1] - gy, gy - hi[1]), 0)
        d = np.minimum(d, np.hypot(dx, dy))
    ok = d >= clearance
    return np.column_stack([gx[ok], gy[ok]])
