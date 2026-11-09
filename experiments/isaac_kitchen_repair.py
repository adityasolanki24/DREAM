"""Repair of broken asset references in NVIDIA's replicator_kitchen scenes, at load time.

Evaluation-environment repair of a BROKEN ASSET REFERENCE (Rule 3: not a simplification): the
kitchens reference ``SimReady/Residential/Kitchen/Appliances/Refrigerator_A01/
sm_rc_appliance_refrigerator_a01_01.usd`` (and ``Refrigerator_A01_2/``), which return 404. The
asset server holds the same refrigerator as ``sm_refrigerator_a01_01.usd`` in ``Refrigerator_A01``
(listing of 9 Nov 2026; validated in results/isaac_env_validation). The oven reference
(``Oven_A01/sm_rc_appliance_oven_a01_01.usd``) has a different cause: no oven asset of any name
exists under SimReady/Residential/Kitchen/Appliances, so it cannot be repaired by remapping and is
reported as unrepaired (the stove of those kitchens stays absent).

The scene files on the server are not modified: ``apply_repairs`` edits the in-memory layers of
the loaded stage only. Not applying it keeps the unrepaired behaviour reproducible (the default
of every runner; opt in with ``--kitchen-repair``).
"""

from __future__ import annotations

import posixpath

# basename of the broken reference -> basename of the existing asset
FILENAME_MAP = {"sm_rc_appliance_refrigerator_a01_01.usd": "sm_refrigerator_a01_01.usd"}
# directory name of the broken reference -> existing directory
DIRECTORY_MAP = {"Refrigerator_A01_2": "Refrigerator_A01"}
# broken references with no existing asset of any name (cause differs: nothing to remap to)
UNREPAIRABLE = {
    "sm_rc_appliance_oven_a01_01.usd": "no oven asset exists under SimReady/Residential/"
    "Kitchen/Appliances (listing of 9 Nov 2026)",
}  # fmt: skip


def repair_reference(asset_path: str) -> str | None:
    """The repaired asset path for a broken reference, or None when it is not a known one."""
    head, base = posixpath.split(asset_path)
    if base not in FILENAME_MAP:
        return None
    parent, directory = posixpath.split(head)
    return posixpath.join(parent, DIRECTORY_MAP.get(directory, directory), FILENAME_MAP[base])


def unrepairable_reason(asset_path: str) -> str | None:
    return UNREPAIRABLE.get(posixpath.basename(asset_path))


def disable_nonfinite_physics(stage) -> list[dict]:  # pragma: no cover - needs pxr
    """Switch off the collider / rigid body of every prim whose world bound is empty or
    non-finite. A placeholder whose asset reference cannot be repaired (the oven) leaves such a
    prim; PhysX then reports "Illegal BroadPhaseUpdateData - non-finite bounds" every step and
    the l_island kitchen crashed in PhysX at reset (cloud job of 9 Nov 2026, again after the
    refrigerator was repaired). Evaluation-environment repair of a broken asset reference: the
    unresolvable object stays absent, it just stops poisoning the physics scene. Returns what
    was disabled."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
         UsdGeom.Tokens.guide],
    )  # fmt: skip
    out = []
    for prim in stage.Traverse():
        collides = prim.HasAPI(UsdPhysics.CollisionAPI)
        body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if not (collides or body):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = list(box.GetMin()), list(box.GetMax())
        bad = box.IsEmpty() or max(abs(v) for v in lo + hi) > 1e8
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            bad = bad or points is None or len(points) == 0
        if not bad:
            continue
        if collides:
            prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        if body:
            prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        out.append({"prim": str(prim.GetPath()), "type": prim.GetTypeName(),
                    "collider": collides, "rigid_body": body})  # fmt: skip
    return out


def apply_repairs(stage) -> dict:  # pragma: no cover - needs pxr
    """Remap the broken references on a loaded kitchen stage (in memory). Returns what was
    repaired and what could not be."""
    from pxr import Sdf

    repaired, unrepaired = [], []
    for prim in stage.Traverse():
        for spec in prim.GetPrimStack():
            lst = spec.referenceList
            for name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
                items = list(getattr(lst, name))
                new_items, changed = [], False
                for ref in items:
                    fixed = repair_reference(ref.assetPath)
                    if fixed:
                        new_items.append(Sdf.Reference(fixed, ref.primPath, ref.layerOffset))
                        repaired.append({"prim": str(prim.GetPath()), "from": ref.assetPath,
                                         "to": fixed})  # fmt: skip
                        changed = True
                    else:
                        new_items.append(ref)
                        why = unrepairable_reason(ref.assetPath)
                        if why:
                            unrepaired.append({"prim": str(prim.GetPath()),
                                               "reference": ref.assetPath, "why": why})  # fmt: skip
                if changed:
                    setattr(lst, name, new_items)
    return {"repaired": repaired, "unrepaired": unrepaired}
