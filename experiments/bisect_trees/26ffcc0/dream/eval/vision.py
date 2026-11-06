"""Scoring camera perception against ground truth. Evaluation only.

Detections are matched to true objects by nearest centre within ``MATCH_RADIUS``; colours are
checked against the scene's colour names (a correct name is one the goal could refer to).
"""

from __future__ import annotations

import numpy as np

from dream.perception.detect import Detection
from dream.sim.tabletop_env import TabletopEnv

MATCH_RADIUS = 0.03  # m


def _yaw_error(a: float, b: float) -> float:
    """Smallest difference between two square-box yaws (mod 90 deg), in degrees."""
    d = np.degrees(a - b) % 90.0
    return float(min(d, 90.0 - d))


def score_detections(env: TabletopEnv, dets: list[Detection]) -> dict:
    spec = env.privileged_spec()
    objs = [d for d in dets if d.kind == "object"]
    truth = {
        o.name: (env.privileged_object_pos(o.name), 2 * o.half_size, env.data.body(o.name).xquat)
        for o in spec.objects
    }
    matched, used = [], set()
    for name, (pos, size, quat) in truth.items():
        cands = [
            (np.linalg.norm(d.position - pos), i)
            for i, d in enumerate(objs)
            if i not in used and np.linalg.norm(d.position - pos) < MATCH_RADIUS
        ]
        if not cands:
            continue
        err, i = min(cands)
        used.add(i)
        d = objs[i]
        w, x, y, z = quat
        true_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        matched.append(
            {
                "object": name,
                "position_error_m": float(err),
                "xy_error_m": float(np.linalg.norm(d.position[:2] - pos[:2])),
                "size_error_m": float(d.size - size),
                "yaw_error_deg": _yaw_error(d.yaw, true_yaw),
                "color_correct": d.color == name,
                "position_std_m": float(d.position_std),
                "z_score": float(err / max(d.position_std, 1e-6)),
                "occluded": d.occluded,
            }
        )
    out = {
        "n_true": len(truth),
        "n_detected_objects": len(objs),
        "recall": len(matched) / max(len(truth), 1),
        "precision": len(used) / max(len(objs), 1),
        "matches": matched,
    }
    if spec.tray is not None:
        trays = [d for d in dets if d.kind == "container"]
        if trays:
            t = min(trays, key=lambda d: np.linalg.norm(d.position[:2] - np.array(spec.tray.xy)))
            out["tray_centre_error_m"] = float(np.linalg.norm(t.position[:2] - spec.tray.xy))
            out["tray_half_error_m"] = float(t.inner_half - spec.tray.inner_half)
        out["tray_detected"] = bool(trays)
        out["containers_detected"] = len(trays)
    return out
