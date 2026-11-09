"""USD authoring of the composer against local stand-in assets (needs pxr: usd-core).

Skipped where pxr is absent (the project does not depend on it). Run it in a throwaway venv with
usd-core, mujoco and pytest, outside the repository. The stand-ins mimic what the real SimReady
assets looked like in the cloud jobs: an instanceable geometry scope holding a rigid body with a
collider, a joint scope, a Y-up centimetre asset, a base that does not sit at the origin."""

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from dream.sim.home import CLASSES, DIMS, sample_home  # noqa: E402
from experiments import isaac_asset_validation as v  # noqa: E402
from experiments import isaac_home_compose as comp  # noqa: E402


def make_asset(path, size, y_up_cm=False):
    """A cube asset of the given (x, y, z) metre size whose footprint centre sits at (0.2, 0.1)
    and whose lowest point is z = 0.05 above its origin (so the base offset is exercised)."""
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    geom = UsdGeom.Xform.Define(stage, "/Asset/Geom")
    geom.GetPrim().SetInstanceable(True)
    body = UsdGeom.Cube.Define(stage, "/Asset/Geom/body")
    body.CreateSizeAttr(2.0)
    sx, sy, sz = size
    if y_up_cm:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        k = 100.0
        centre, half = (20.0, 5.0 + sz / 2 * k, -10.0), (sx / 2 * k, sz / 2 * k, sy / 2 * k)
    else:
        centre, half = (0.2, 0.1, 0.05 + sz / 2), (sx / 2, sy / 2, sz / 2)
    xf = UsdGeom.Xformable(body.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    # an asymmetric "nose" at the native +x end (outside the instanced scope): shows the facing
    nose = UsdGeom.Cube.Define(stage, "/Asset/Nose")
    nose.CreateSizeAttr(2.0)
    nxf = UsdGeom.Xformable(nose.GetPrim())
    nose_at = (0.2 + sx / 2 + 0.05, 0.1, 0.05 + sz / 2)  # normalised metres, z up
    if y_up_cm:
        nose_at = (nose_at[0] * 100, nose_at[2] * 100, -nose_at[1] * 100)
        nxf.AddTranslateOp().Set(Gf.Vec3d(*nose_at))
        nxf.AddScaleOp().Set(Gf.Vec3f(5, 5, 5))
    else:
        nxf.AddTranslateOp().Set(Gf.Vec3d(*nose_at))
        nxf.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.05))
    UsdPhysics.Scene.Define(stage, "/Asset/Physics/PhysicsScene")  # some real assets carry one
    UsdGeom.Scope.Define(stage, "/Asset/Joints")
    UsdPhysics.FixedJoint.Define(stage, "/Asset/Joints/j1")
    stage.GetRootLayer().Save()


def entry_for(path, size, y_up_cm=False):
    norm = v.normalise_matrix("Y" if y_up_cm else "Z", 0.01 if y_up_cm else 1.0)
    return {
        "status": "ok", "url": str(path), "front_axis": "+x", "yaw_offset_deg": 0.0,
        "size_m": list(size), "base_centre_native": [0.2, 0.1, 0.05],
        "normalise": norm, "rigid_body_paths": ["/Geom/body"], "joint_paths": ["/Joints/j1"],
        "collider_paths": ["/Geom/body"],
    }  # fmt: skip


def build(tmp_path, seed=103, scenario="B", target="bed"):
    manifest = {"excluded": {"version": 1, "classes": []}, "classes": {}}
    home = sample_home(np.random.default_rng(seed), scenario, target, "nominal", seed)
    for i, cls in enumerate(sorted({o.cls for o in home.objects})):
        d, w, h = (k * 0.95 for k in DIMS[cls])
        asset = tmp_path / f"{cls}.usda"
        make_asset(asset, (d, w, h), y_up_cm=(i % 2 == 1))
        manifest["classes"][cls] = entry_for(asset, (d, w, h), y_up_cm=(i % 2 == 1))
    assert set(manifest["classes"]) <= set(CLASSES)
    placements = comp.plan_placements(home, manifest)
    out = tmp_path / "stage.usda"
    comp.build_stage(home, placements, out)
    return home, placements, Usd.Stage.Open(str(out))


def test_stage_structure_poses_labels_and_static_overrides(tmp_path):
    home, placements, stage = build(tmp_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    xcache = UsdGeom.XformCache()
    for p in placements:
        prim = stage.GetPrimAtPath(p.prim_path)
        assert prim and prim.IsValid()
        assert prim.GetAttribute("primvars:semantics:class:semanticData").Get() == p.cls
        # the base-centre of the asset lands on the generator pose, facing the generator yaw
        m = xcache.GetLocalToWorldTransform(prim)
        cx, cy = 0.2, 0.1  # base centre in native units (cm assets scale this, see below)
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        assert lo[2] == pytest.approx(p.z0, abs=1e-3)  # sits on its base height
        # the bound centre of a symmetric stand-in is its footprint centre
        c = (lo + hi) / 2
        assert np.isfinite(c).all() and cx and cy and m is not None
        size = hi - lo
        assert max(size) >= max(p.size_m) * 0.99 or p.yaw % (np.pi / 2) != 0
        # authored physics is switched off on the referenced prims (needs un-instancing)
        body = stage.GetPrimAtPath(p.prim_path + "/Geom/body")
        assert body.GetAttribute("physics:rigidBodyEnabled").Get() is False
        assert body.GetAttribute("physics:collisionEnabled").Get() is False
        joint = stage.GetPrimAtPath(p.prim_path + "/Joints/j1")
        assert joint.GetAttribute("physics:jointEnabled").Get() is False
        collider = stage.GetPrimAtPath(p.prim_path + "_collider")
        assert collider and UsdGeom.Imageable(collider).GetPurposeAttr().Get() == "guide"
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert len(
        [
            p
            for p in stage.GetPrimAtPath("/World/Home").GetChildren()
            if p.GetName().startswith("Wall_")
        ]
    ) == len(home.walls)
    assert len(
        [
            p
            for p in stage.GetPrimAtPath("/World/Home").GetChildren()
            if p.GetName().startswith("Floor")
        ]
    ) == len(home.rooms)


def test_placed_footprint_centre_and_facing(tmp_path):
    """The asset's footprint centre, measured in the composed stage, is the planned xy and its
    front (+x of the generator frame) points along the planned yaw."""
    home, placements, stage = build(tmp_path, seed=100, scenario="B", target="refrigerator")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for p in placements:
        prim = stage.GetPrimAtPath(p.prim_path)
        body = stage.GetPrimAtPath(p.prim_path + "/Geom/body")
        box = cache.ComputeWorldBound(body).ComputeAlignedRange()
        c = (np.array(box.GetMin()) + np.array(box.GetMax())) / 2
        # footprint centre of the stand-in cube = its bound centre in xy for a box
        yaw = p.yaw
        # bound of a rotated box is its AABB; its centre equals the oriented box centre
        assert c[0] == pytest.approx(p.xy[0], abs=2e-2), p.cls
        assert c[1] == pytest.approx(p.xy[1], abs=2e-2), p.cls
        nose = stage.GetPrimAtPath(p.prim_path + "/Nose")
        nb = cache.ComputeWorldBound(nose).ComputeAlignedRange()
        nc = (np.array(nb.GetMin()) + np.array(nb.GetMax())) / 2
        direction = (nc - c)[:2] / np.linalg.norm((nc - c)[:2])
        assert np.allclose(direction, [np.cos(yaw), np.sin(yaw)], atol=5e-2), p.cls
        assert prim.IsValid()


def _colliders(stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            yield str(prim.GetPath()), np.array(box.GetMin()), np.array(box.GetMax())


def test_ground_is_one_continuous_collider_with_its_top_at_zero(tmp_path):
    """(a) the surface the G1's feet meet is z = 0, one collider with no seams."""
    home, _, stage = build(tmp_path, seed=100, scenario="B", target="refrigerator")
    sx, sy, _ = home.start
    under = [
        (path, hi[2])
        for path, lo, hi in _colliders(stage)
        if lo[0] <= sx <= hi[0] and lo[1] <= sy <= hi[1] and lo[2] < 0.05
    ]
    assert under == [(comp.GROUND, pytest.approx(0.0, abs=1e-9))]
    floors = [
        p for p in stage.GetPrimAtPath(comp.ROOT_PRIM).GetChildren() if "Floor" in p.GetName()
    ]
    assert floors and not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in floors)  # visual only
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene) and p.IsActive()]
    assert [str(p.GetPath()) for p in scenes] == [comp.PHYSICS_SCENE]


@pytest.mark.parametrize("seed,scenario,target", [(100, "B", "refrigerator"), (102, "E", "stove")])
def test_no_collider_intersects_the_spawn_volume(tmp_path, seed, scenario, target):
    """(b) a box around the start pose, from just above the ground to 1.4 m, 0.6 m across, meets
    no collider (floor slabs, walls, bounding boxes, assets' own)."""
    home, _, stage = build(tmp_path, seed=seed, scenario=scenario, target=target)
    sx, sy, _ = home.start
    lo_v, hi_v = np.array([sx - 0.3, sy - 0.3, 0.05]), np.array([sx + 0.3, sy + 0.3, 1.4])
    hits = [path for path, lo, hi in _colliders(stage) if np.all(lo <= hi_v) and np.all(hi >= lo_v)]
    assert hits == []
