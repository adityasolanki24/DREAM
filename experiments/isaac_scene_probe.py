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
    parser.add_argument("--kitchens", default="l_shape,g_shape,l_island,peninsula,u_shape")
    parser.add_argument("--updates", type=int, default=150, help="app updates to let assets load")
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

        results = {}
        for kitchen in args.kitchens.split(","):
            url = kitchen_url(get_assets_root_path(), kitchen)
            if not open_stage(url):
                raise RuntimeError(f"could not open {url}")
            for _ in range(args.updates):
                app.update()
            stage = get_current_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
            guide = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render", "guide", "proxy"]
            )
            rows = []
            for prim in stage.Traverse():
                attr = prim.GetAttribute("primvars:semantics:class:semanticData")
                label = str(attr.Get()) if attr and attr.HasAuthoredValue() else None
                if label not in ("refrigerator", "oven", "sink", "countertop"):
                    continue
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                empty = (
                    bool(box.IsEmpty())
                    or max(abs(v) for v in [*box.GetMin(), *box.GetMax()]) > HUGE
                )
                row = describe(cache, prim, 1)
                gbox = guide.ComputeWorldBound(prim).ComputeAlignedRange()
                gempty = (
                    bool(gbox.IsEmpty())
                    or max(abs(v) for v in [*gbox.GetMin(), *gbox.GetMax()]) > HUGE
                )
                row.update(
                    label=label,
                    render_bound_empty=empty,
                    all_purposes_bound=None
                    if gempty
                    else [list(gbox.GetMin()), list(gbox.GetMax())],
                )
                refs = []
                for child in prim.GetChildren():
                    for spec in child.GetPrimStack():
                        refs += [
                            str(r.assetPath) for r in spec.referenceList.GetAddedOrExplicitItems()
                        ]
                row["child_reference_assets"] = refs
                rows.append(row)
            results[kitchen] = rows
            print(kitchen, len(rows), flush=True)
        (args.out / "probe.json").write_text(json.dumps(results, indent=2, default=str))
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
