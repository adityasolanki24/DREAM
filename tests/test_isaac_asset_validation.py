"""CPU-side contracts of the asset validator: failure rules and manifest selection."""

import json

import numpy as np

from experiments import isaac_asset_validation as v
from experiments import isaac_home_compose as comp


def good(cls="refrigerator", size=(0.7, 0.7, 1.8)):
    return {
        "resolves": True, "unresolved": [], "composed_children": ["Geom"],
        "meshes_with_points": 3, "colliders": 0, "up_axis": "Z", "meters_per_unit": 1.0,
        "bound_normalised": [
            [-size[0] / 2, -size[1] / 2, 0.0],
            [size[0] / 2, size[1] / 2, size[2]],
        ],
        "bound_after_play": [[-0.3, -0.3, 0.0], [0.3, 0.3, 1.0]],
        "path": f"SimReady/{cls}.usd", "tag": "c00", "url": f"https://x/{cls}.usd",
    }  # fmt: skip


def test_good_record_passes():
    assert v.failures("refrigerator", good()) == []


def test_404_is_a_failure():
    assert "does not resolve" in v.failures("refrigerator", {"resolves": False})[0]


def test_nonfinite_or_empty_bound_fails():
    rec = good()
    rec["bound_normalised"] = [[-3.4e38] * 3, [3.4e38] * 3]
    assert any("non-finite" in f for f in v.failures("refrigerator", rec))
    rec["bound_normalised"] = None
    assert any("non-finite" in f for f in v.failures("refrigerator", rec))


def test_wrong_unit_size_fails_but_texture_warning_does_not():
    rec = good(size=(70, 70, 180))  # centimetres read as metres
    assert any("size" in f for f in v.failures("refrigerator", rec))
    rec = good()
    rec["unresolved"] = ["./textures/a.png"]
    assert v.failures("refrigerator", rec) == []
    rec["unresolved"] = ["./sub/layer.usd"]
    assert v.failures("refrigerator", rec)


def test_y_up_centimetre_matrix_maps_to_z_up_metres():
    m = np.array(v.normalise_matrix("Y", 0.01))
    p = m @ [0.0, 180.0, 50.0, 1.0]  # 1.8 m up in y, 0.5 m toward +z
    assert np.allclose(p[:3], [0.0, -0.5, 1.8])


def test_front_axis_offsets_face_plus_x():
    for front, yaw in (("+x", 0), ("+y", -90), ("-x", 180), ("-y", 90)):
        assert v.front_to_yaw_offset(front) == yaw


def test_manifest_selects_first_passing_candidate_and_lists_failures(tmp_path):
    cands = json.loads(v.CANDIDATES.read_text())["classes"]
    for cls, paths in cands.items():
        for i in range(len(paths)):
            rec = good(cls) if cls == "toilet" and i == 1 else {"resolves": False}
            if cls == "toilet" and i == 1:
                rec["bound_normalised"] = [[-0.2, -0.35, 0.0], [0.2, 0.35, 0.75]]
            (tmp_path / cls).mkdir(exist_ok=True)
            (tmp_path / cls / f"c{i:02d}.json").write_text(json.dumps(rec))
    out = v.build_manifest(tmp_path, {"toilet": "-y"}, "2026-11-09")
    toilet = out["classes"]["toilet"]
    assert toilet["status"] == "ok" and toilet["candidate_path"].endswith("toilet.usd")
    assert toilet["yaw_offset_deg"] == 90.0 and toilet["size_m"][:2] == [
        0.7,
        0.4,
    ]  # front -y: depth is native y
    assert out["classes"]["bathtub"]["status"] == "failed"  # no candidate exists
    assert out["classes"]["refrigerator"]["status"] == "failed"
    assert (
        "404" in out["classes"]["refrigerator"]["failure"]
        or "resolve" in (out["classes"]["refrigerator"]["failure"])
    )
    # a manifest entry composes with the composer
    m = comp.placement_matrix(toilet["base_centre_native"], toilet["yaw_offset_deg"], 1.0,
                              (0, 0), 0.0, 0.0, toilet["normalise"])  # fmt: skip
    assert np.isfinite(m).all()
