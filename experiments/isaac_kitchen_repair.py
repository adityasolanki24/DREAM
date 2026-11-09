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
