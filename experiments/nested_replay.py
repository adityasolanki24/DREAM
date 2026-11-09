"""Replay captured Isaac frames through SemanticMap with the nested-object option off and on.

    python experiments/nested_replay.py --raw <capture item dir> --source span --out <json>
    python experiments/nested_replay.py --raw <item dir> --source file \
        --detections <detector_compare/detections/NAME.json> --out <json>

``--raw`` is a capture item folder (``frames.jsonl``, ``frames/NNNNN.npz`` with ``depth``, and
``true_camera`` / ``true_pose`` / ``odometry_pose`` per row). Poses: ``true`` uses the true
head-camera pose in the world frame; ``odometry`` re-expresses the same camera in the robot's
odometry frame (the camera's pose relative to the base, composed with the odometry pose), which
is the frame the live system maps in.

Detection sources:
- ``file``: the validated detections of a detector_compare run (the real evidence).
- ``tight``: the instance boxes of the segmentation truth, labelled by their true class (a
  control: nothing is wrong with the boxes).
- ``span``: as ``tight``, but the countertop box is the union of countertop and sink boxes (the
  box encloses what is mounted in it) and so is the cabinet box with the countertop; this
  imitates the reported Qwen failure and tests the mechanism. SURROGATE boxes built from truth:
  evidence about the mechanism, not about any detector.
- ``qwen_recorded``: the Qwen answers stored in the capture (two frames only).

Reports track purity of the sink and its neighbours as the verifier sees it (label shares),
track counts per class (fragmentation), and the verifier's verdict, for option off and on.
Development data only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = {
    "sink": {"raw": ["sink", "sink_basin", "sink_rim"], "merge": True},
    "countertop": {"raw": ["countertop"]},
    "kitchen_cabinet": {"raw": ["kitchen_cabinet"]},
    "refrigerator": {"raw": ["refrigerator"]},
}


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_in_odometry(camera: dict, true_pose, odometry_pose) -> dict:
    """The world-frame camera (position/forward/up) seen from the odometry frame: express it in
    the base frame with the TRUE base pose, then place it with the ODOMETRY pose."""
    tx, ty, tyaw = (float(v) for v in true_pose)
    ox, oy, oyaw = (float(v) for v in odometry_pose)
    to_base, to_odom = rot_z(-tyaw), rot_z(oyaw)
    pos = np.asarray(camera["position"], float) - np.array([tx, ty, 0.0])
    return {
        "position": (to_odom @ (to_base @ pos) + np.array([ox, oy, 0.0])).tolist(),
        "forward": (to_odom @ to_base @ np.asarray(camera["forward"], float)).tolist(),
        "up": (to_odom @ to_base @ np.asarray(camera["up"], float)).tolist(),
        "fovy_deg": camera["fovy_deg"],
    }


def union_box(boxes: list[list[float]]) -> list[float]:
    a = np.asarray(boxes, float)
    return [a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()]


def surrogate_boxes(truth: list[dict], kind: str) -> list[tuple[str, list[float]]]:
    """(label, box) detections from the truth instances of one frame (see module docstring)."""
    from experiments.detector_compare import MIN_GT_PIXELS

    by = {}
    for t in truth:
        if t["pixels"] >= MIN_GT_PIXELS:
            by.setdefault(t["cls"], []).append(t["box"])
    out = []
    for cls, boxes in by.items():
        for box in boxes:
            out.append((cls, box))
    if kind == "span":
        out = [(c, b) for c, b in out if c not in ("countertop", "kitchen_cabinet")]
        if by.get("countertop"):
            out.append(("countertop", union_box(by["countertop"] + by.get("sink", []))))
        if by.get("kitchen_cabinet"):
            out.append(
                ("kitchen_cabinet", union_box(by["kitchen_cabinet"] + by.get("countertop", [])))
            )
    return out


def box_mask(box, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    m = np.zeros((h, w), bool)
    m[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
    return m


def load_rows(raw: pathlib.Path) -> list[dict]:
    return [
        json.loads(x) for x in (raw / "frames.jsonl").read_text("utf-8").splitlines() if x.strip()
    ]


def detections_for(row: dict, source: str, file_dets: dict | None, min_score: float):
    from experiments.detector_compare import truth_instances

    if source in ("tight", "span"):
        return surrogate_boxes(truth_instances(row.get("gt", {}), CLASSES), source)
    if source == "qwen_recorded":
        q = row.get("qwen")
        return None if q is None else [(d["label"], d["box"]) for d in q["detections"]]
    f = file_dets.get(row["i"])
    return None if f is None else [(d["label"], d["box"]) for d in f if d["score"] >= min_score]


def build_frame(raw: pathlib.Path, row: dict, pose_mode: str, dets, size=(320, 240)):
    from dream.perception.geometry import calibration_from_look
    from dream.types import RGBDFrame, SemanticDetection

    depth = np.load(raw / "frames" / f"{row['i']:05d}.npz")["depth"].astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    h, w = depth.shape
    cam = row["true_camera"]
    if pose_mode == "odometry":
        cam = camera_in_odometry(cam, row["true_pose"], row["odometry_pose"])
    cal = calibration_from_look(cam["position"], cam["forward"], cam["up"], cam["fovy_deg"], w, h)
    detections = (
        None
        if dets is None
        else tuple(
            SemanticDetection(label, 0.9, box_mask(box, h, w), "replay") for label, box in dets
        )
    )
    return RGBDFrame(
        float(row["sim_time_s"]),
        np.zeros((h, w, 3), np.uint8),
        depth,
        np.zeros((h, w), bool),
        cal,
        detections,
    )


def track_row(t) -> dict:
    return {
        "id": t.id,
        "labels": dict(sorted(t.labels.items(), key=lambda kv: -kv[1])),
        "views": t.views,
        "position": np.round(t.position, 2).tolist(),
        "extent": np.round(t.hi - t.lo, 2).tolist(),
    }


def evaluate_map(sm, classes: list[str]) -> dict:
    """What the verifier sees: for each class, the track with the most support for it."""
    from dream.semantics.verify import VerifierThresholds, check_track

    out = {"n_tracks": len(sm.tracks), "tracks": [track_row(t) for t in sm.tracks]}
    per = {}
    for c in classes:
        cands = sm.candidates(c)
        if not cands:
            per[c] = None
            continue
        best = cands[0]
        report = check_track(best, c, VerifierThresholds())
        per[c] = {
            "track": best.id,
            "support": best.support(c),
            "purity": round(best.purity(c), 3),
            "distinct_labels": len(best.labels),
            "verified": report["verified"],
            "tracks_with_label": len(cands),
        }
    out["per_class"] = per
    out["mixed_tracks"] = sum(
        1 for t in sm.tracks if sum(1 for n in t.labels.values() if n >= 2) >= 2
    )
    return out


def replay(
    raw: pathlib.Path, source: str, pose_mode: str, nested, file_dets=None, min_score: float = 0.0
) -> dict:
    from dream.perception.occupancy import OccupancyGrid
    from dream.perception.semantic import SemanticMap

    sm = SemanticMap(OccupancyGrid(), nested)
    used = 0
    for row in load_rows(raw):
        dets = detections_for(row, source, file_dets, min_score)
        if dets is None:
            continue
        sm.update(build_frame(raw, row, pose_mode, dets))
        used += 1
    return {"frames_with_detections": used, **evaluate_map(sm, list(CLASSES))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=pathlib.Path, required=True)
    ap.add_argument("--source", choices=("tight", "span", "qwen_recorded", "file"), required=True)
    ap.add_argument("--detections", type=pathlib.Path)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from experiments.code_version import code_dirty, code_sha

    file_dets = None
    if args.source == "file":
        saved = json.loads(args.detections.read_text("utf-8"))
        file_dets = {f["i"]: f["dets"] for f in saved["frames"]}
    variants = variant_options()
    result = {
        "raw": str(args.raw),
        "source": args.source,
        "detections_file": str(args.detections),
        "min_score": args.min_score,
        "variants": {k: v.__dict__ if v else None for k, v in variants.items()},
        "code_sha": code_sha(ROOT),
        "code_dirty": code_dirty(ROOT),
        "runs": {},
    }
    lines = []
    for pose_mode in ("true", "odometry"):
        for name, opt in variants.items():
            r = replay(args.raw, args.source, pose_mode, opt, file_dets, args.min_score)
            result["runs"][f"{pose_mode}/{name}"] = r
            lines.append(summary_line(args.source, pose_mode, name, r))
            print(lines[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    args.out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def variant_options() -> dict:
    """``off`` is the frozen behaviour; ``nested`` is the option with its default values; the
    rest are ablations and one alternative exclusion rule (a smaller detection owns pixels it
    shares with a larger one, however little they nest). Disclosure: after the first replay
    (tight boxes) the gate's size measure was changed from footprint area to longest side
    because area fragmented flat faces seen head-on (the refrigerator split into 8 tracks); the
    alternative exclusion rule was added after seeing that sink purity stayed low. Everything
    here is therefore development evidence, tuned on the l_shape frames it is reported on."""
    from dream.perception.semantic import NestedOptions as N

    return {
        "off": None,
        "nested": N(),
        "exclude_only": N(gate_extent=False),
        "gate_only": N(exclude_contained=False),
        "smaller_owns_overlap+gate": N(contained_fraction=0.1, max_area_ratio=1.0),
    }


def summary_line(source: str, pose_mode: str, name: str, r: dict) -> str:
    def cell(c):
        v = r["per_class"][c]
        return (
            "none"
            if v is None
            else f"{v['purity']:.2f}/{v['support']}/{'V' if v['verified'] else '-'}"
        )

    return (
        f"{source:13s} {pose_mode:8s} {name:27s} tracks {r['n_tracks']:3d} "
        f"mixed {r['mixed_tracks']:2d}"
        f" | purity/support/verified: sink {cell('sink'):11s} countertop {cell('countertop'):11s}"
        f" cabinet {cell('kitchen_cabinet'):11s} refrigerator {cell('refrigerator')}"
    )


if __name__ == "__main__":
    main()
