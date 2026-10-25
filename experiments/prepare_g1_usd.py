"""Convert DREAM's walking-G1 MJCF to USD for Isaac Sim, unmodified apart from two scene edits.

    python experiments/prepare_g1_usd.py --out <dir outside Git>   # in a converter environment

Runs NVIDIA's open-source mujoco-usd-converter (Apache-2.0, pinned in the session setup) on
third_party/gr00t_wbc_g1/g1_gear_wbc.xml, the exact model the validated MuJoCo walking
results use. Two edits, both scene rather than robot changes: the mesh directory is
redirected (as dream/sim/g1_mobile.py does), and the MJCF's own floor plane is removed so
the external scene's floor is the only ground. The converter writes MuJoCo/Newton joint
attributes (armature, friction loss) that PhysX does not read; dream/sim/isaac_g1.py applies
the PhysX equivalents and records which properties have no exact equivalent.

The receipt records input/output hashes and the converted mass and joint counts, so the
Isaac run can check that it loaded the same embodiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MJCF = ROOT / "third_party/gr00t_wbc_g1/g1_gear_wbc.xml"
MESHES = ROOT / "third_party/unitree_g1/meshes"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    import mujoco
    import mujoco_usd_converter
    from pxr import Usd, UsdPhysics

    out = args.out.resolve()
    if out.is_relative_to(ROOT.resolve()):
        raise ValueError("generated assets stay outside Git")
    out.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.meshdir = MESHES.resolve().as_posix()
    floors = [g for g in spec.geoms if g.name == "floor"]
    if len(floors) != 1:
        raise ValueError("expected exactly one MJCF floor geom")
    spec.delete(floors[0])
    model = spec.compile()
    xml = out / "g1_gear_wbc_scene_free.xml"
    xml.write_text(spec.to_xml(), encoding="utf-8")
    converter = mujoco_usd_converter.Converter(layer_structure=False, scene=False)
    asset = pathlib.Path(converter.convert(str(xml), str(out / "usd")).path)
    stage = Usd.Stage.Open(str(asset))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    mass = sum(
        UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if len(joints) != model.nu or len(roots) != 1 or abs(mass - model.body_subtreemass[1]) > 1e-3:
        raise ValueError("converted asset does not match the MJCF embodiment")
    receipt = {
        "mjcf": MJCF.relative_to(ROOT).as_posix(),
        "mjcf_sha256": sha256(MJCF),
        "converter": "mujoco-usd-converter " + mujoco_usd_converter.__version__,
        "mujoco": mujoco.__version__,
        "python": sys.version.split()[0],
        "edits": ["meshdir redirected", "MJCF floor geom removed (external scene supplies it)"],
        "usd": asset.name,
        "usd_sha256": sha256(asset),
        "articulation_root": str(roots[0].GetPath()),
        "revolute_joints": len(joints),
        "mjcf_actuators": model.nu,
        "mass_kg": mass,
        "mjcf_mass_kg": float(model.body_subtreemass[1]),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
