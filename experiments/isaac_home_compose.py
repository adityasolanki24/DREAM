"""Compose a multi-room Isaac Sim home from DREAM's procedural home generator.

Simulator side and evaluation side only: nothing under dream/runtime, agent, perception,
semantics or navigation may import this module (tests/test_isolation.py). It turns a
``HomeSpec`` (dream/sim/home.py: rooms, wall segments, furniture with class, pose and size,
sampled per generator variant, seed, scenario and target) into

* a USD stage: floor slab, wall boxes (plain coloured geometry), and one validated photoreal
  asset per furniture instance placed at the generator's pose, each with a bounding-box
  collider and a class semantic label (``primvars:semantics:class:semanticData``, as the
  NVIDIA kitchens carry it, read back by dream/eval/isaac_scene.labelled_objects);
* an evaluation-side truth manifest (``scene_truth.json``: instances, classes, rooms, which
  instances are the target class, the asset used and the scale applied);
* an agent-facing view (``agent_view``): the stage path, the start pose and the request text.
  That is all the agent receives. Neither asset choice nor any object class other than the
  one the request itself names reaches it.

Refusal rule (owner, 9 Nov 2026): every class the home contains must have a validated asset in
the manifest (experiments/configs/isaac_assets.json). If one does not, ``plan_placements``
raises ``MissingAssetError`` naming every missing class. It never drops an object, swaps in a
different class, or turns a target-present scenario into an absent one.

The pure part (placement maths, refusal, truth and agent view) runs without Isaac; ``build_stage``
needs ``pxr``.

    python experiments/isaac_home_compose.py --variant nominal --seed 100 --scenario B \
        --target refrigerator --out <dir>          # inside Isaac's Python
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.sim.home import (  # noqa: E402
    TARGETS,
    WALL_HEIGHT,
    HomeSpec,
    ObjectSpec,
    Rect,
    RoomBox,
    _overlap,
    footprint,
    sample_home,
)

MANIFEST = ROOT / "experiments/configs/isaac_assets.json"
# Scale rule (coordinator, 9 Nov 2026): no visibly squashed furniture, which would make perception
# results meaningless. A uniform scale in [SCALE_MIN, SCALE_MAX] with each axis within AXIS_DEV of
# it. An asset that does not fit the generator footprint under that rule is placed at native size
# against the same wall (back face fixed) when that is collision-free, else the home is refused.
SCALE_MIN, SCALE_MAX = 0.75, 1.3
AXIS_DEV = 0.15
SHIFTS_M = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5)  # lateral shifts tried, native mode
TOUCH_TOL_M = 0.02  # flush contact is not a collision
FLOOR_THICKNESS = 0.1  # m, slab below z = 0
ROOT_PRIM = "/World/Home"
# Isaac's World() defines its own scene at /physicsScene unless one exists there; a scene at any
# other path left two scenes in the first composed-home job (the G1 was thrown 40 m at spawn).
PHYSICS_SCENE = "/physicsScene"
GROUND = f"{ROOT_PRIM}/Ground"  # the one continuous ground collider, top at z = 0 like the kitchens
GROUND_MARGIN_M = 1.0
VISUAL_FLOOR_M = 0.001  # room floors are visual only, a millimetre above the ground
# Plain colours per room type for the floor (rooms stay visually distinguishable to the camera
# without a texture library). Appearance only; the agent is never told the room type.
FLOOR_RGB = {
    "kitchen": (0.75, 0.72, 0.65), "bedroom": (0.62, 0.55, 0.45), "bathroom": (0.7, 0.74, 0.78),
    "living_room": (0.58, 0.5, 0.42), "office": (0.55, 0.57, 0.6), "garage": (0.45, 0.45, 0.46),
    "laundry": (0.68, 0.7, 0.72), "storage": (0.5, 0.48, 0.44), "studio": (0.6, 0.55, 0.5),
}  # fmt: skip
WALL_RGB = (0.85, 0.85, 0.82)


class MissingAssetError(RuntimeError):
    """A class the home contains has no validated asset (or no usable fit)."""

    def __init__(self, classes: dict[str, str]) -> None:
        self.classes = classes
        super().__init__(
            "no validated asset for: "
            + "; ".join(f"{c} ({why})" for c, why in sorted(classes.items()))
        )


class CaseRefused(RuntimeError):
    """The case cannot be composed without changing what it means (see ``check_case``)."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def excluded_classes(manifest: dict) -> frozenset[str]:
    """Classes the Isaac environment omits (owner decision of 9 Nov 2026: no loadable or no
    faithful asset). Versioned in the manifest. Rule 3 class: environment simplification; the
    full-class generator stays the MuJoCo challenge case."""
    return frozenset(manifest.get("excluded", {}).get("classes", ()))


def omissions(home: HomeSpec, manifest: dict) -> list[dict]:
    """Every instance the composer leaves out, with its class, pose and room."""
    gone = excluded_classes(manifest)
    version = manifest.get("excluded", {}).get("version")
    return [
        {"instance_id": o.instance_id, "class": o.cls, "xy": list(o.xy), "room_id": o.room_id,
         "reason": f"class excluded from the Isaac environment (list version {version})"}
        for o in home.objects if o.cls in gone
    ]  # fmt: skip


def check_case(home: HomeSpec, manifest: dict) -> None:
    """Refuse, rather than silently change, a case whose meaning depends on an excluded class:
    an excluded target; a look-alike scenario (F) with no look-alike left; a conflicting-evidence
    scenario (K) that placed an excluded mismatch object."""
    gone = excluded_classes(manifest)
    target = home.target_class
    why = []
    if target in gone:
        why.append(f"target class {target} is excluded")
    info = TARGETS.get(target) if target else None
    if info and home.scenario[:1] == "F":
        if not any(o.cls in info.confusers and o.cls not in gone for o in home.objects):
            why.append("no look-alike of the target remains after omissions")
    if info and home.scenario[:1] == "K":
        hit = sorted({o.cls for o in home.objects if o.cls in info.mismatch and o.cls in gone})
        if hit:
            why.append(f"mismatch object(s) {hit} are excluded")
    if why:
        raise CaseRefused(why)


@dataclass(frozen=True)
class Placement:
    instance_id: str
    cls: str
    prim_path: str  # neutral name: no class word
    url: str
    xy: tuple[float, float]
    z0: float
    yaw: float  # world yaw of the object's front (+x of the generator frame), rad
    scale: tuple[float, float, float]  # (depth, width, height) scale
    matrix: tuple[tuple[float, ...], ...]  # 4x4, column-vector convention: world = M @ native
    size_m: tuple[float, float, float]  # placed (depth, width, height) in the object frame
    rigid_body_paths: tuple[str, ...] = ()  # asset-relative prims made static (furniture is fixed)
    joint_paths: tuple[str, ...] = ()  # asset-relative joints disabled for the same reason
    collider_paths: tuple[str, ...] = ()  # authored colliders disabled: the bounding box stands in
    mode: str = "fit"  # "fit" (scaled into the generator footprint) or "native_anchored"
    shift_m: tuple[float, float] = (0.0, 0.0)  # centre shift from the generator pose (x, y)


def load_manifest(path: pathlib.Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_problem(entry: dict | None) -> str | None:
    if entry is None:
        return "not in the manifest"
    if entry.get("status") != "ok":
        return str(entry.get("failure") or "validation failed")
    for key in ("url", "size_m", "base_centre_native", "yaw_offset_deg"):
        if key not in entry:
            return f"manifest entry lacks {key}"
    if min(entry["size_m"]) <= 0:
        return "non-positive validated size"
    return None


def placement_matrix(
    base_centre_native,
    yaw_offset_deg: float,
    scale,
    xy,
    z0: float,
    yaw: float,
    normalise=None,
) -> np.ndarray:
    """world = T(xy, z0) Rz(yaw + offset) S(scale) T(-base_centre_native) Norm, column vectors;
    ``scale`` is a scalar or a triple along the native axes.

    ``normalise`` maps the asset's native frame to metres, z up (up-axis and metres-per-unit,
    measured by experiments/isaac_asset_validation.py); ``base_centre_native`` is the centre of
    the asset's footprint at its lowest point in that normalised frame. After the offset
    rotation the asset's front faces the generator's +x, so the object's base centre lands
    on (xy, z0) with its front facing yaw."""
    a = yaw + radians(yaw_offset_deg)
    c, s = cos(a), sin(a)
    rot = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    scl = np.diag([*np.broadcast_to(np.asarray(scale, float), 3), 1.0])
    to_base = np.eye(4)
    to_base[:3, 3] = -np.asarray(base_centre_native, float)
    move = np.eye(4)
    move[:3, 3] = [xy[0], xy[1], z0]
    norm = np.eye(4) if normalise is None else np.asarray(normalise, float)
    return move @ rot @ scl @ to_base @ norm


def fit_scale(entry: dict, obj: ObjectSpec) -> tuple[float, float, float] | None:
    """(depth, width, height) scale fitting the asset's footprint inside the generator's under the
    scale rule, or None when the rule cannot fit it."""
    depth, width, _ = entry["size_m"]
    rd, rw = 2 * obj.half[0] / depth, 2 * obj.half[1] / width
    m = min(rd, rw)
    if m >= SCALE_MIN:  # uniform: no distortion at all
        s = min(m, SCALE_MAX)
        return s, s, s
    s = SCALE_MIN  # smallest uniform scale; axes may shrink to (1 - AXIS_DEV) of it
    lo, hi = s * (1 - AXIS_DEV), s * (1 + AXIS_DEV)
    if min(rd, rw) < lo:
        return None
    return min(max(rd, lo), hi), min(max(rw, lo), hi), s


def native_scale(entry: dict, fit: tuple[float, float, float]) -> tuple[float, float, float]:
    """The (depth, width, height) scale as a triple along the asset's native x, y, z."""
    sd, sw, sh = fit
    return (sd, sw, sh) if entry.get("front_axis", "+x") in ("+x", "-x") else (sw, sd, sh)


def _rect(xy, depth: float, width: float, yaw: float) -> Rect:
    return Rect(xy[0], xy[1], depth / 2, width / 2, yaw)


def anchored_native(entry: dict, obj: ObjectSpec, others: list[Rect], walls: list[Rect]):
    """Native-size placement anchored where the generator put the object: the back face stays on
    the generator's back line and the footprint is centred laterally. Tries small lateral shifts
    along the wall; returns (xy, shift) or raises ValueError naming the collision."""
    depth, width, _ = entry["size_m"]
    fx, fy = cos(obj.yaw), sin(obj.yaw)
    grow = (depth - 2 * obj.half[0]) / 2  # centre moves forward by this to keep the back face
    base = (obj.xy[0] + fx * grow, obj.xy[1] + fy * grow)
    why = "no collision-free anchored position"
    for shift in SHIFTS_M:
        xy = (base[0] - fy * shift, base[1] + fx * shift)  # lateral = (-fy, fx)
        r = _rect(xy, depth - TOUCH_TOL_M, width - TOUCH_TOL_M, obj.yaw)
        hit = next((("wall", w) for w in walls if _overlap(r, w)), None) or next(
            (("object", o) for o in others if _overlap(r, o)), None
        )
        if hit is None:
            return xy, (xy[0] - obj.xy[0], xy[1] - obj.xy[1])
        why = f"native footprint {depth:.2f} x {width:.2f} m collides with a {hit[0]}"
    raise ValueError(why)


def plan_placements(home: HomeSpec, manifest: dict) -> list[Placement]:
    """One placement per furniture instance; raises ``MissingAssetError`` for all gaps."""
    classes = manifest.get("classes", {})
    gone = excluded_classes(manifest)
    kept = [o for o in home.objects if o.cls not in gone]
    missing: dict[str, str] = {}
    for cls in sorted({o.cls for o in kept}):
        problem = _entry_problem(classes.get(cls))
        if problem:
            missing[cls] = problem
    if missing:
        raise MissingAssetError(missing)
    walls = [Rect(w.centre[0], w.centre[1], w.half[0], w.half[1]) for w in home.walls]
    rects = {o.instance_id: footprint(o) for o in kept}  # replaced by the placed footprint
    plans: dict[str, tuple] = {}
    for o in kept:
        entry = classes[o.cls]
        fit = fit_scale(entry, o)
        if fit is not None:
            plans[o.instance_id] = ("fit", fit, o.xy, (0.0, 0.0))
            rects[o.instance_id] = _rect(
                o.xy, entry["size_m"][0] * fit[0], entry["size_m"][1] * fit[1], o.yaw
            )
            continue
        others = [r for i, r in rects.items() if i != o.instance_id]
        try:
            xy, shift = anchored_native(entry, o, others, walls)
        except ValueError as error:
            missing[o.cls] = f"{o.instance_id}: {error}"
            continue
        plans[o.instance_id] = ("native_anchored", (1.0, 1.0, 1.0), xy, shift)
        rects[o.instance_id] = _rect(xy, entry["size_m"][0], entry["size_m"][1], o.yaw)
    if missing:
        raise MissingAssetError(missing)
    out = []
    for n, o in enumerate(kept):
        entry = classes[o.cls]
        mode, s, xy, shift = plans[o.instance_id]
        m = placement_matrix(
            entry["base_centre_native"],
            entry["yaw_offset_deg"],
            native_scale(entry, s),
            xy,
            o.z0,
            o.yaw,
            entry.get("normalise"),
        )
        out.append(
            Placement(
                o.instance_id, o.cls, f"{ROOT_PRIM}/o{n:03d}", entry["url"], xy, o.z0, o.yaw, s,
                tuple(tuple(float(v) for v in row) for row in m),
                tuple(float(v * k) for v, k in zip(entry["size_m"], s, strict=True)),
                tuple(entry.get("rigid_body_paths", ())), tuple(entry.get("joint_paths", ())),
                tuple(entry.get("collider_paths", ())), mode, (float(shift[0]), float(shift[1])),
            )
        )  # fmt: skip
    return out


def truth_manifest(home: HomeSpec, placements: list[Placement], manifest: dict) -> dict:
    """EVALUATION-SIDE truth. Never given to the agent."""
    target = home.target_class
    return {
        "generator": home.generator,
        "seed": home.seed,
        "scenario": home.scenario,
        "target_class": target,
        "request": home.request,
        "target_present": bool(target and any(o.cls == target for o in home.objects)),
        "target_instances": [o.instance_id for o in home.objects if o.cls == target],
        "extent_m": [home.width, home.depth],
        "start": list(home.start),
        "rooms": [
            {"id": r.id, "type": r.type, "lo": list(r.lo), "hi": list(r.hi)} for r in home.rooms
        ],  # fmt: skip
        "doorways": [
            {"a": d.a, "b": d.b, "centre": list(d.centre), "width": d.width, "axis": d.axis}
            for d in home.doorways
        ],  # fmt: skip
        "instances": [
            {
                "instance_id": p.instance_id,
                "class": p.cls,
                "prim_path": p.prim_path,
                "asset_url": p.url,
                "xy": list(p.xy),
                "z0": p.z0,
                "yaw": p.yaw,
                "scale": p.scale,
                "mode": p.mode,
                "shift_m": list(p.shift_m),
                "size_m": list(p.size_m),
                "room_id": next(o.room_id for o in home.objects if o.instance_id == p.instance_id),
            }
            for p in placements
        ],  # fmt: skip
        "omitted": omissions(home, manifest),
        "exclusion_list": manifest.get("excluded"),
        "rule3_class": "environment simplification (excluded classes omitted)",
        "asset_manifest_validated_utc": manifest.get("validated_utc"),
    }


def agent_view(home: HomeSpec, stage_path: str) -> dict:
    """The ONLY description of the scene the agent side receives: where to load the stage, the
    start pose (setup, as in the kitchens) and the request text."""
    return {"stage": stage_path, "start": list(home.start), "request": home.request}


def compose_plan(
    variant: str, seed: int, scenario: str, target: str, manifest: dict | None = None
) -> tuple[HomeSpec, list[Placement], dict]:
    home = sample_home(np.random.default_rng(seed), scenario, target, variant, seed)
    manifest = load_manifest() if manifest is None else manifest
    check_case(home, manifest)
    placements = plan_placements(home, manifest)
    return home, placements, truth_manifest(home, placements, manifest)


# -- USD (needs pxr) -----------------------------------------------------------------------------
def uninstance(stage, root_path: str) -> int:
    """Make every instanceable prim under ``root_path`` an ordinary prim, so that opinions can be
    authored on its descendants (SimReady Residential, Hospital and Industrial assets are
    instanced: the first validation job's ``OverridePrim`` calls failed with "_ValidateEditPrim"
    inside them). Returns the number of prims changed. Needs pxr."""
    from pxr import Usd

    changed = 0
    for _ in range(8):  # nested instancing
        todo = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)) if p.IsInstanceable()]
        if not todo:
            break
        for p in todo:
            p.SetInstanceable(False)
        changed += len(todo)
    return changed


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _box(stage, path, centre, half, material=None, collide=True, guide=False, yaw_deg=0.0):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)  # half extents become the scale
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddRotateZOp().Set(float(yaw_deg))  # T R S: scale acts in the box's own axes
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if guide:
        UsdGeom.Imageable(cube.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
    return cube


def apply_static_overrides(stage, placements) -> int:
    """Make every placed asset static: un-instance it, then switch off its authored rigid bodies,
    joints and colliders (the bounding-box collider stands in). Idempotent. Called when the stage
    is written and again on the stage Isaac opened, because the written overrides only bind if
    the remote asset had composed (and un-instanced) at writing time. Returns prims touched."""
    from pxr import Sdf

    n = 0
    for p in placements:
        uninstance(stage, p.prim_path)
        for rels, attr in ((p.rigid_body_paths, "physics:rigidBodyEnabled"),
                           (p.collider_paths, "physics:collisionEnabled"),
                           (p.joint_paths, "physics:jointEnabled")):  # fmt: skip
            for rel in rels:
                stage.OverridePrim(p.prim_path + rel).CreateAttribute(
                    attr, Sdf.ValueTypeNames.Bool
                ).Set(False)
                n += 1
    return n


def build_stage(home: HomeSpec, placements: list[Placement], path: pathlib.Path) -> None:
    """Write the home as a USD layer. Furniture is a plain reference to the asset URL (a
    file:// URI for local files; Isaac 6.2 does not compose bare paths under an https layer)."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    stage = __import__("pxr").Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, ROOT_PRIM)
    UsdPhysics.Scene.Define(stage, PHYSICS_SCENE).CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    stage.GetPrimAtPath(PHYSICS_SCENE).CreateAttribute(
        "physics:gravityMagnitude", Sdf.ValueTypeNames.Float
    ).Set(9.81)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60, 0, 30))

    half_w, half_d = home.width / 2 + GROUND_MARGIN_M, home.depth / 2 + GROUND_MARGIN_M
    _box(stage, GROUND, (0.0, 0.0, -FLOOR_THICKNESS / 2), (half_w, half_d, FLOOR_THICKNESS / 2),
         guide=True)  # fmt: skip
    wall_mat = _material(stage, f"{ROOT_PRIM}/Looks/wall", WALL_RGB)
    for r in home.rooms:  # one floor slab per room (floor colour hints at nothing semantic)
        c = ((r.lo[0] + r.hi[0]) / 2, (r.lo[1] + r.hi[1]) / 2)
        h = ((r.hi[0] - r.lo[0]) / 2, (r.hi[1] - r.lo[1]) / 2)
        mat = _material(stage, f"{ROOT_PRIM}/Looks/floor{r.id}", FLOOR_RGB.get(r.type, (0.6,) * 3))
        _box(stage, f"{ROOT_PRIM}/Floor{r.id}", (*c, VISUAL_FLOOR_M / 2),
             (h[0], h[1], VISUAL_FLOOR_M / 2), material=mat, collide=False)  # fmt: skip
    for w in home.walls:
        _box(stage, f"{ROOT_PRIM}/Wall_{w.name.replace(':', '_')}", (*w.centre, WALL_HEIGHT / 2),
             (w.half[0], w.half[1], WALL_HEIGHT / 2), material=wall_mat)  # fmt: skip
    for p in placements:
        prim = UsdGeom.Xform.Define(stage, p.prim_path).GetPrim()
        prim.GetReferences().AddReference(p.url)
        apply_static_overrides(stage, [p])
        xf = UsdGeom.Xformable(prim)
        m = Gf.Matrix4d()
        mat = np.asarray(p.matrix, float)
        for i in range(4):  # USD is row-vector: transpose the column-vector matrix
            m.SetRow(i, Gf.Vec4d(*mat[:, i]))
        xf.AddTransformOp().Set(m)
        attr = prim.CreateAttribute(
            "primvars:semantics:class:semanticData", Sdf.ValueTypeNames.String
        )
        attr.Set(p.cls)
        prim.CreateAttribute("dream:instance_id", Sdf.ValueTypeNames.String).Set(p.instance_id)
        # Bounding-box collider of the placed asset (purpose guide: never rendered).
        d, wd, ht = p.size_m
        _box(stage, p.prim_path + "_collider", (*p.xy, p.z0 + ht / 2), (d / 2, wd / 2, ht / 2),
             guide=True, yaw_deg=float(np.degrees(p.yaw)))  # fmt: skip
    stage.GetRootLayer().Save()


def write_composition(home, placements, truth, out: pathlib.Path) -> dict:
    """Write stage.usda, scene_truth.json (evaluation only) and agent_view.json."""
    out.mkdir(parents=True, exist_ok=True)
    build_stage(home, placements, out / "stage.usda")
    (out / "scene_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    view = agent_view(home, str(out / "stage.usda"))
    (out / "agent_view.json").write_text(json.dumps(view, indent=2), encoding="utf-8")
    return view


def room_of_point(home: HomeSpec, xy) -> RoomBox | None:
    from dream.sim.home import room_at

    return room_at(home, xy)


def main() -> None:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="nominal")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--target", default="refrigerator")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    home, placements, truth = compose_plan(args.variant, args.seed, args.scenario, args.target)
    print(json.dumps(write_composition(home, placements, truth, args.out), indent=2))


if __name__ == "__main__":
    main()
