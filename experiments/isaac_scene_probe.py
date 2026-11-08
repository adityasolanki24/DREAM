"""Why do the refrigerator and oven prims of the NVIDIA kitchens report no bounds?

Opens a kitchen (no robot, no model) and lists, for every labelled prim whose own world bound
is empty, its type, instancing, payload state, children and ancestors' bounds, plus every prim
whose name suggests a refrigerator or oven. Evaluation-side ground-truth diagnostics only.

    python experiments/isaac_scene_probe.py --out <dir> [--kitchen l_shape]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HUGE = 1e8


def describe(cache, prim, depth: int = 1) -> dict:
    from pxr import Usd, UsdGeom

    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = list(box.GetMin()), list(box.GetMax())
    empty = bool(box.IsEmpty()) or max(abs(v) for v in lo + hi) > HUGE
    row = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "active": prim.IsActive(),
        "instanceable": prim.IsInstanceable(),
        "is_instance_proxy": prim.IsInstanceProxy(),
        "has_payload": prim.HasAuthoredPayloads(),
        "loaded": prim.IsLoaded(),
        "boundable": prim.IsA(UsdGeom.Boundable),
        "bound_empty": empty,
        "bound": None if empty else [lo, hi],
        "n_children": len(list(prim.GetChildren())),
    }
    if depth > 0:
        row["children"] = [
            describe(cache, c, depth - 1)
            for c in list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))[:12]
        ]
    return row


def main() -> None:  # pragma: no cover - needs Isaac Sim
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--kitchen", default="l_shape")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    failure = None
    try:
        from isaacsim.core.utils.stage import get_current_stage, open_stage
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Usd, UsdGeom

        from experiments.isaac_g1_kitchen import kitchen_url

        url = kitchen_url(get_assets_root_path(), args.kitchen)
        if not open_stage(url):
            raise RuntimeError(f"could not open {url}")
        app.update()
        stage = get_current_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
        labelled, named = [], []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            attr = prim.GetAttribute("primvars:semantics:class:semanticData")
            label = str(attr.Get()) if attr and attr.HasAuthoredValue() else None
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = list(box.GetMin()), list(box.GetMax())
            empty = bool(box.IsEmpty()) or max(abs(v) for v in lo + hi) > HUGE
            if label in ("refrigerator", "oven", "sink") and empty:
                row = describe(cache, prim, 2)
                row["label"] = label
                row["ancestors"] = [
                    describe(cache, a, 0) for a in [prim.GetParent(), prim.GetParent().GetParent()]
                ]
                labelled.append(row)
            low = path.lower()
            if any(k in low for k in ("refrig", "fridge", "oven")) and len(named) < 60:
                named.append(describe(cache, prim, 0))
        result = {
            "kitchen": args.kitchen,
            "labelled_empty": labelled,
            "named_like_fridge_or_oven": named,
        }
        (args.out / "probe.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({"labelled_empty": len(labelled), "named": len(named)}), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.out / "error.txt").write_text(failure)
    finally:
        if failure is not None:
            sys.stdout.flush()
            os._exit(1)
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
