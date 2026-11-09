"""Validate candidate USD assets for the composed Isaac homes, and inventory ready-made scenes.

Evaluation/simulator side. For every candidate in experiments/configs/isaac_asset_candidates.json
(class -> USD paths under <assets_root>/Isaac/) this job records, as it goes (one JSON per
candidate, written before the next starts):

* resolves: the layer opens (a 404 is a failure, as the NVIDIA kitchens' refrigerator was) and
  every dependency of it resolves (``UsdUtils.ComputeAllDependencies``; missing textures or MDLs
  are a warning, any other missing layer a failure);
* composes through a plain ``Usd`` reference on a fresh stage (the Isaac 6.2 quirk: no deprecated
  helpers) and yields geometry (meshes with points);
* the world bound after normalising up-axis and metres-per-unit is finite, non-empty and a
  plausible size for its class (the l_island failure mode was a non-finite bound);
* physics: collider / rigid body / joint / articulation counts, and a short timeline play with
  the asset present, after which its bound must still be finite;
* four rendered views (from +x, +y, -x, -y of the asset's own frame) so the front can be read.

``--build-manifest`` runs locally (no Isaac): selects, per class, the first candidate (candidate
order is the preference order, fixed before the run) that passes every check and writes
experiments/configs/isaac_assets.json from the records and a hand-read front-axis table.
Classes with no passing candidate are listed as failures with the reason of each candidate.

    python experiments/isaac_asset_validation.py --out <dir> [--classes a,b] [--budget-s 2400]
    python experiments/isaac_asset_validation.py --build-manifest <records-dir> --fronts <json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANDIDATES = ROOT / "experiments/configs/isaac_asset_candidates.json"
MANIFEST = ROOT / "experiments/configs/isaac_assets.json"
HUGE = 1e8
# Plausible-size window: each sorted dimension of the asset within [lo, hi] times the generator's
# nominal DIMS for the class (written by the benchmark author from typical product sizes), so a
# wrong-unit or mis-scaled asset fails. Extra classes outside the generator carry explicit
# full sizes (m, largest dimension range).
SIZE_WINDOW = (0.35, 2.8)
EXTRA_LARGEST_M = {"dishwasher": (0.5, 1.1), "kettle": (0.15, 0.4), "tv": (0.5, 2.2),
                   "plant": (0.2, 2.2)}  # fmt: skip
TEXTURE_EXT = (".png", ".jpg", ".jpeg", ".exr", ".dds", ".tga", ".tif", ".tiff", ".hdr", ".mdl")
VIEWS = {"px": (1, 0), "py": (0, 1), "nx": (-1, 0), "ny": (0, -1)}  # camera side -> unit (x, y)
SHOT_PX = 384


# -- pure helpers (tested on CPU) ------------------------------------------------------------
def normalise_matrix(up_axis: str, meters_per_unit: float) -> list[list[float]]:
    """4x4, column-vector: asset native -> metre, z-up frame. Y-up assets turn +90 deg about x
    (y -> z, z -> -y)."""
    s = float(meters_per_unit)
    if up_axis.upper() == "Y":
        return [[s, 0, 0, 0], [0, 0, -s, 0], [0, s, 0, 0], [0, 0, 0, 1]]
    return [[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]]


def finite_box(lo, hi) -> bool:
    vals = [*lo, *hi]
    return (
        all(math.isfinite(v) for v in vals)
        and max(abs(v) for v in vals) < HUGE
        and all(
            h > l
            for l, h in zip(lo, hi, strict=True)  # noqa: E741
        )
    )


def size_problem(cls: str, size_xyz) -> str | None:
    """None when the sorted dimensions are plausible for the class."""
    from dream.sim.home import DIMS

    dims = sorted(size_xyz)
    if cls in DIMS:
        want = sorted(DIMS[cls])
        for got, ref in zip(dims, want, strict=True):
            if not SIZE_WINDOW[0] * ref <= got <= SIZE_WINDOW[1] * ref:
                return f"size {tuple(round(v, 2) for v in size_xyz)} m vs nominal {want}"
        return None
    lo, hi = EXTRA_LARGEST_M.get(cls, (0.0, 1e9))
    return None if lo <= dims[-1] <= hi else f"largest dimension {dims[-1]:.2f} m outside {lo}-{hi}"


def failures(cls: str, rec: dict) -> list[str]:
    """Every reason a measured candidate record fails (empty list = validated)."""
    out = []
    if not rec.get("resolves"):
        return ["reference does not resolve: " + str(rec.get("error", "layer did not open"))]
    bad = [u for u in rec.get("unresolved", []) if not u.lower().endswith(TEXTURE_EXT)]
    if bad:
        out.append(f"{len(bad)} unresolved dependencies, e.g. {bad[0]}")
    if not rec.get("composed_children"):
        out.append("reference composed no prims")
    if rec.get("meshes_with_points", 0) < 1:
        out.append("no mesh with points")
    box = rec.get("bound_normalised")
    if not box or not finite_box(*box):
        out.append(f"world bound non-finite or empty: {box}")
    else:
        problem = size_problem(cls, [h - l for l, h in zip(*box, strict=True)])  # noqa: E741
        if problem:
            out.append(problem)
    if rec.get("bound_after_play") is not None and not finite_box(*rec["bound_after_play"]):
        out.append(f"bound after physics play non-finite: {rec['bound_after_play']}")
    after, before = rec.get("bound_after_play"), rec.get("bound_normalised")
    if after and before and finite_box(*after):
        moved = max(
            abs(a - b)
            for u, v in zip(after, before, strict=True)
            for a, b in zip(u, v, strict=True)
        )
        if moved > 0.05:
            out.append(f"moved {moved:.2f} m under physics although made static")
    if rec.get("play_error"):
        out.append("physics play failed: " + str(rec["play_error"]))
    if rec.get("error"):
        out.append("error: " + str(rec["error"]))
    return out


def front_to_yaw_offset(front: str) -> float:
    """Degrees to rotate the asset about z so the face toward ``front`` (+x, +y, -x, -y of the
    normalised asset frame) faces the generator's +x."""
    return {"+x": 0.0, "+y": -90.0, "-x": 180.0, "-y": 90.0}[front]


def manifest_entry(cls: str, url: str, rec: dict, front: str, note: dict) -> dict:
    lo, hi = rec["bound_normalised"]
    size = [h - l for l, h in zip(lo, hi, strict=True)]  # noqa: E741
    dx, dy, dz = size
    depth, width = (dx, dy) if front in ("+x", "-x") else (dy, dx)
    return {
        "status": "ok",
        "url": url,
        "candidate_path": rec["path"],
        "front_axis": front,
        "yaw_offset_deg": front_to_yaw_offset(front),
        "size_m": [depth, width, dz],
        "base_centre_native": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]],
        "normalise": normalise_matrix(rec["up_axis"], rec["meters_per_unit"]),
        "up_axis": rec["up_axis"],
        "meters_per_unit": rec["meters_per_unit"],
        "collision": "authored" if rec.get("colliders", 0) else "bounding_box",
        "authored_colliders": rec.get("colliders", 0),
        "rigid_body_paths": rec.get("rigid_body_paths", []),
        "joint_paths": rec.get("joint_paths", []),
        "collider_paths": rec.get("collider_paths", []),
        "evidence": f"results/isaac_env_validation/assets/{cls}/{rec['tag']}.json",
        "warnings": rec.get("unresolved", []),
        **note,
    }


def build_manifest(records_dir: pathlib.Path, fronts: dict, validated_utc: str) -> dict:
    """Select per class: the first candidate (file order = preference) with no failure. Classes
    without one are written as ``status: failed`` with every candidate's reasons."""
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))["classes"]
    classes = {}
    for cls, paths in cands.items():
        reasons, chosen = [], None
        for i, path in enumerate(paths):
            f = records_dir / cls / f"c{i:02d}.json"
            if not f.is_file():
                reasons.append({"path": path, "failures": ["not run"]})
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            why = failures(cls, rec)
            if not why and chosen is None:
                chosen = (i, path, rec)
            reasons.append({"path": path, "failures": why})
        if chosen is None:
            classes[cls] = {"status": "failed", "failure": "; ".join(
                f"{r['path'].split('/')[-1]}: {r['failures'][0]}" for r in reasons
            ) or "no candidate asset exists on the server", "candidates": reasons}  # fmt: skip
            continue
        i, path, rec = chosen
        front = fronts.get(cls)
        if front is None:
            classes[cls] = {"status": "failed", "failure": "validated but front axis not read",
                            "candidates": reasons}  # fmt: skip
            continue
        note = fronts.get("_notes", {}).get(cls, {})
        classes[cls] = {**manifest_entry(cls, rec["url"], rec, front, note),
                        "candidates": reasons}  # fmt: skip
    return {"validated_utc": validated_utc, "isaac_sim": "6.2.0.0", "classes": classes}


# -- Isaac side -------------------------------------------------------------------------------
def _look_at(Gf, eye, target):
    import numpy as np

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    m = Gf.Matrix4d(1.0)
    for i, row in enumerate((right, up, -f)):
        m.SetRow(i, Gf.Vec4d(*row, 0.0))
    m.SetRow(3, Gf.Vec4d(*eye, 1.0))
    return m


def main() -> None:  # pragma: no cover - needs Isaac Sim
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--classes", default="")
    ap.add_argument("--budget-s", type=float, default=2400.0)
    ap.add_argument("--max-per-class", type=int, default=99)
    ap.add_argument("--no-snapshots", action="store_true", help="skip the four renders")
    ap.add_argument("--indexes", default="", help="cls:i,cls:i - run only these candidates")
    ap.add_argument(
        "--stop-after-pass",
        action="store_true",
        help="skip a class's remaining candidates once one validates",
    )
    ap.add_argument("--environments", action="store_true", help="also inventory ready-made scenes")
    ap.add_argument("--build-manifest", type=pathlib.Path)
    ap.add_argument("--fronts", type=pathlib.Path)
    args = ap.parse_args()
    if args.build_manifest:
        fronts = json.loads(args.fronts.read_text(encoding="utf-8"))
        out = build_manifest(args.build_manifest, fronts, time.strftime("%Y-%m-%d"))
        MANIFEST.write_text(json.dumps(out, indent=1), encoding="utf-8")
        ok = [c for c, e in out["classes"].items() if e["status"] == "ok"]
        print("validated:", ok)
        print("failed:", [c for c in out["classes"] if c not in ok])
        return
    run(args)


def run(args) -> None:  # pragma: no cover - needs Isaac Sim
    from isaacsim import SimulationApp

    args.out.mkdir(parents=True, exist_ok=True)
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    t0 = time.monotonic()
    failure = None
    try:
        from experiments.isaac_g1_kitchen import Evidence

        ev = Evidence(args.out)
        _validate_all(app, args, ev, t0)
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.out / "error.txt").write_text(failure, encoding="utf-8")
    finally:
        if failure is not None:
            sys.stdout.flush()
            os._exit(1)
        app.close(wait_for_replicator=False, skip_cleanup=True)


def _validate_all(app, args, ev, t0) -> None:  # pragma: no cover - needs Isaac Sim
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils

    root = get_assets_root_path()
    ev.stage("assets_root", root=root)
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))["classes"]
    want = [c for c in args.classes.split(",") if c] or list(cands)
    only: dict[str, set[int]] = {}
    for pair in [p for p in args.indexes.split(",") if p]:
        c, i = pair.split(":")
        only.setdefault(c, set()).add(int(i))
    if only and not args.classes:
        want = [c for c in want if c in only]
    ctx = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    ctx.new_stage()
    app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 40))
    cam = UsdGeom.Camera.Define(stage, "/World/ValCam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    cam_op = xf.AddTransformOp()
    product = rep.create.render_product("/World/ValCam", (SHOT_PX, SHOT_PX))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for _ in range(10):
        app.update()
    cache_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]

    def bound(prim, xform=None):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), cache_purposes)
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = list(box.GetMin()), list(box.GetMax())
        if box.IsEmpty() or max(abs(v) for v in lo + hi) > HUGE:
            return None
        return [lo, hi]

    def shoot(centre, radius, tag, folder) -> list:
        shots = []
        for name, (ux, uy) in VIEWS.items():
            dist = 1.7 * radius
            eye = [centre[0] + ux * dist, centre[1] + uy * dist, centre[2] + 0.35 * radius]
            cam_op.Set(_look_at(Gf, eye, centre))
            for _ in range(3):  # job 2: three of four views of some assets were the first frame
                app.update()
            img = np.asarray(rgb.get_data())
            for _ in range(6):
                rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
                img = np.asarray(rgb.get_data())
                if img.size and img.reshape(-1, img.shape[-1])[:, :3].std() > 1.0:
                    break
            if img.size:
                path = folder / f"{tag}_{name}.png"
                Image.fromarray(img[..., :3].astype(np.uint8)).save(path)
                shots.append(path.name)
        return shots

    def one(cls: str, index: int, rel: str) -> dict:
        tag = f"c{index:02d}"
        url = f"{root}/Isaac/{rel}"
        rec = {"class": cls, "index": index, "tag": tag, "path": rel, "url": url,
               "started_s": time.monotonic() - t0}  # fmt: skip
        print(f"[candidate] {cls} {tag} {rel}", flush=True)
        prim_path = "/World/Cand"
        try:
            layer = Sdf.Layer.FindOrOpen(url)
            rec["resolves"] = layer is not None
            if layer is None:
                rec["error"] = "Sdf.Layer.FindOrOpen returned None (404 or unreadable)"
                return rec
            try:
                _l, _a, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(url))
                rec["unresolved"] = sorted(str(u) for u in unresolved)
            except Exception as error:  # recorded, not hidden
                rec["unresolved"] = []
                rec["dependency_error"] = repr(error)
            probe = Usd.Stage.Open(layer)
            rec["up_axis"] = str(UsdGeom.GetStageUpAxis(probe))
            rec["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(probe))
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(url)
            for _ in range(30):
                app.update()
            loading = getattr(ctx, "get_stage_loading_status", None)
            waited = 0
            while loading and waited < 400:
                status = loading()
                if not status or status[2] == 0:
                    break
                app.update()
                waited += 1
            rec["composed_children"] = [c.GetName() for c in prim.GetChildren()]
            from experiments.isaac_home_compose import uninstance

            rec["uninstanced_prims"] = uninstance(stage, prim_path)
            norm = np.array(normalise_matrix(rec["up_axis"], rec["meters_per_unit"]))
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            m = Gf.Matrix4d()
            for i in range(4):
                m.SetRow(i, Gf.Vec4d(*norm[:, i]))
            xform.AddTransformOp().Set(m)
            meshes = 0
            colliders = bodies = 0
            rigid, joints, collider_paths = [], [], []
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if p.IsA(UsdGeom.Mesh):
                    pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                    meshes += 1 if pts is not None and len(pts) > 0 else 0
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    colliders += 1
                    collider_paths.append(str(p.GetPath())[len(prim_path) :])
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    bodies += 1
                    rigid.append(str(p.GetPath())[len(prim_path) :])
                if p.IsA(UsdPhysics.Joint):
                    joints.append(str(p.GetPath())[len(prim_path) :])
            rec.update(meshes_with_points=meshes, colliders=colliders, rigid_bodies=bodies,
                       rigid_body_paths=rigid, joint_paths=joints,
                       collider_paths=collider_paths)  # fmt: skip
            box = bound(prim)
            rec["bound_normalised"] = box
            if box and finite_box(*box):
                centre = [(a + b) / 2 for a, b in zip(*box, strict=True)]
                radius = 0.5 * math.dist(*box)
                folder = args.out / "snapshots" / cls
                folder.mkdir(parents=True, exist_ok=True)
                rec["snapshots"] = [] if args.no_snapshots else shoot(centre, radius, tag, folder)
                try:  # physics play with the asset treated as the composer treats it: authored
                    # bodies, joints and colliders disabled (the 664 s first candidate was PhysX
                    # cooking 20 SDF colliders, and its dynamic bodies fell through the void), a
                    # bounding-box collider in their place
                    for rel in rigid:
                        stage.OverridePrim(prim_path + rel).CreateAttribute(
                            "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool
                        ).Set(False)
                    for rel in joints:
                        stage.OverridePrim(prim_path + rel).CreateAttribute(
                            "physics:jointEnabled", Sdf.ValueTypeNames.Bool
                        ).Set(False)
                    for rel in collider_paths:
                        stage.OverridePrim(prim_path + rel).CreateAttribute(
                            "physics:collisionEnabled", Sdf.ValueTypeNames.Bool
                        ).Set(False)
                    half = [(b - a) / 2 for a, b in zip(*box, strict=True)]
                    cube = UsdGeom.Cube.Define(stage, "/World/CandCollider")
                    cube.CreateSizeAttr(2.0)
                    cxf = UsdGeom.Xformable(cube.GetPrim())
                    cxf.AddTranslateOp().Set(Gf.Vec3d(*centre))
                    cxf.AddScaleOp().Set(Gf.Vec3f(*half))
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                    UsdGeom.Imageable(cube.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
                    timeline.play()
                    for _ in range(20):
                        app.update()
                    rec["bound_after_play"] = bound(prim)
                except Exception as error:
                    rec["play_error"] = repr(error)
                finally:
                    timeline.stop()
                    app.update()
            rec["failures"] = failures(cls, rec)
        except BaseException as error:  # a bad asset is a record, never a crash
            rec["error"] = repr(error)
            rec["traceback"] = traceback.format_exc()
            rec["failures"] = failures(cls, rec)
        finally:
            try:
                stage.RemovePrim(prim_path)
                stage.RemovePrim("/World/CandCollider")
                app.update()
            except Exception:
                pass
        rec["wall_s"] = time.monotonic() - t0 - rec["started_s"]
        return rec

    for cls in want:
        for i, rel in enumerate(cands[cls][: args.max_per_class]):
            if only and cls in only and i not in only[cls]:
                continue
            if time.monotonic() - t0 > args.budget_s:
                ev.stage("budget_reached", next=f"{cls}/{i}")
                return
            rec = one(cls, i, rel)
            folder = args.out / "assets" / cls
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{rec['tag']}.json").write_text(json.dumps(rec, indent=1, default=str))
            ev.line(
                "candidates.jsonl",
                {k: rec.get(k) for k in ("class", "tag", "path", "failures", "wall_s")},
            )
            print(f"[result] {cls} {rec['tag']} failures={rec.get('failures')}", flush=True)
            if args.stop_after_pass and not rec.get("failures"):
                break
        if not cands[cls]:
            (args.out / "assets").mkdir(exist_ok=True)
            ev.line("candidates.jsonl", {"class": cls, "failures": ["no candidate exists"]})
    if args.environments:
        _inventory_environments(app, args, ev, root, t0)


ENVIRONMENTS = ("Office/office.usd", "Hospital/hospital.usd", "Simple_Warehouse/full_warehouse.usd",
                "Simple_Room/simple_room.usd")  # fmt: skip


def _inventory_environments(app, args, ev, root, t0) -> None:  # pragma: no cover
    """Open each ready-made scene: loads, unresolved dependencies, size, semantic labels,
    physics counts. Inventory only: nothing is evaluated in them."""
    from collections import Counter

    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    ctx = omni.usd.get_context()
    out = {}
    for rel in ENVIRONMENTS:
        if time.monotonic() - t0 > args.budget_s + 600:
            break
        url = f"{root}/Isaac/Environments/{rel}"
        rec = {"url": url}
        t1 = time.monotonic()
        try:
            rec["resolves"] = Sdf.Layer.FindOrOpen(url) is not None
            if rec["resolves"]:
                _l, _a, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(url))
                rec["unresolved_n"] = len(unresolved)
                rec["unresolved_sample"] = sorted(str(u) for u in unresolved)[:8]
                ok = ctx.open_stage(url)
                for _ in range(200):
                    app.update()
                stage = ctx.get_stage()
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                )
                box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
                types, labels, classes = Counter(), Counter(), Counter()
                colliders = 0
                for p in stage.Traverse():
                    types[p.GetTypeName()] += 1
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        colliders += 1
                    a = p.GetAttribute("primvars:semantics:class:semanticData")
                    if a and a.HasAuthoredValue():
                        labels["primvar"] += 1
                        classes[str(a.Get())] += 1
                    for name in p.GetAppliedSchemas():
                        if "SemanticsAPI" in str(name):
                            labels["SemanticsAPI"] += 1
                rec.update(opened=bool(ok), bound=[list(box.GetMin()), list(box.GetMax())],
                           prims=sum(types.values()), meshes=types.get("Mesh", 0),
                           colliders=colliders, label_prims=dict(labels),
                           label_classes=dict(classes.most_common(40)))  # fmt: skip
        except BaseException as error:
            rec["error"] = repr(error)
        rec["wall_s"] = time.monotonic() - t1
        out[rel] = rec
        (args.out / "environments.json").write_text(json.dumps(out, indent=1, default=str))
        print(
            f"[environment] {rel} resolves={rec.get('resolves')} opened={rec.get('opened')} "
            f"error={rec.get('error')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
