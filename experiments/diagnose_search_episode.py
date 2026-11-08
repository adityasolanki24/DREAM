"""Diagnose one object-search episode: why did the verifier accept the track it accepted?

    python experiments/diagnose_search_episode.py CONFIG --split H2 --scenario E \
        --target bathtub --seed 6001 --arm proposed [--code-root DIR] [--focus-tracks 88,124] \
        --out diagnosis.json

Replays one job from a config exactly as experiments/semantic_search_eval.py runs it, with
read-only in-process instrumentation (nothing in the repository is edited, and the patches
only record): every detection-to-track association (time, label, range, camera position), the
target's candidate tracks before each planner decision, and every verifier verdict. The JSON
holds the result, the focus tracks' label timelines, the verifier's check dict when the final
claim was accepted, and the footprints and labels of the tracks around the claimed one.

``--code-root`` runs another checkout's code (for example the worktree pinned at a frozen
commit) so the diagnosis documents what that code did; the config is read from the path given.
The focus tracks default to the verified track at the end of the episode.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--split", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", default="proposed")
    ap.add_argument("--base", default=None, help="kinematic or g1 (default: the split's)")
    ap.add_argument("--code-root", default=str(HERE), help="checkout whose code is run")
    ap.add_argument("--focus-tracks", help="comma-separated track ids (default: the claimed)")
    ap.add_argument("--neighbour-m", type=float, default=1.0,
                    help="tracks whose footprint is within this of a focus track's")  # fmt: skip
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.code_root).resolve()
    sys.path.insert(0, str(root))
    import experiments.semantic_search_eval as ev
    from dream.perception import semantic as sem
    from dream.runtime import semantic_search as ss
    from dream.semantics import world as world_module

    cfg = json.loads(pathlib.Path(args.config).read_text())
    base = args.base or cfg["splits"][args.split].get("base", "kinematic")
    jobs = ev.jobs_for(cfg, args.split, [args.arm], base)
    job = next(
        j for j in jobs
        if j["scenario"] == args.scenario and j["target"] == args.target and j["seed"] == args.seed
    )  # fmt: skip

    log: dict = {"associations": [], "decisions": [], "verdicts": []}
    state: dict = {"planner": None}

    def step() -> int:
        planner = state["planner"]
        return len(planner.trace) if planner is not None else 0

    original_associate = sem.SemanticMap._associate

    def associate(self, det, pts, cam, t):
        tid = original_associate(self, det, pts, cam, t)
        centre = np.median(pts, axis=0)
        log["associations"].append({
            "t": round(float(t), 2), "track": tid, "label": det.label,
            "range_m": round(float(np.linalg.norm(centre - cam)), 2),
            "camera_xy": np.round(cam[:2], 2).tolist(), "step": step(),
        })  # fmt: skip
        return tid

    sem.SemanticMap._associate = associate

    original_init = ss.SemanticSearchPlanner.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        state["planner"] = self

    ss.SemanticSearchPlanner.__init__ = init

    def snapshot(tr, target):
        return {
            "track": tr.id, "footprint": [np.round(tr.lo[:2], 2).tolist(),
                                          np.round(tr.hi[:2], 2).tolist()],
            "labels": dict(tr.labels), "purity": round(tr.purity(target), 3),
            "support": tr.support(target), "misses": tr.misses,
        }  # fmt: skip

    original_candidate = ss.SemanticSearchPlanner._candidate

    def candidate(self, world, target, passable, dist, history):
        log["decisions"].append({"step": step(), "tracks": [
            snapshot(t, target) for t in world.map.candidates(target)]})  # fmt: skip
        return original_candidate(self, world, target, passable, dist, history)

    ss.SemanticSearchPlanner._candidate = candidate

    original_verify = world_module.verify_find

    def verify(semantic, target, outcome, pose, now, th=None):
        report = original_verify(semantic, target, outcome, pose, now, th)
        log["verdicts"].append({
            "step": step(), "now": round(float(now), 2), "status": report["status"],
            "verified": [r["track"] for r in report["tracks"] if r["verified"]],
            "tracks": report["tracks"],
        })  # fmt: skip
        return report

    world_module.verify_find = verify

    result = ev.episode(job)

    claimed = (result.get("verdict") or {}).get("track")
    focus = (
        [int(x) for x in args.focus_tracks.split(",")] if args.focus_tracks
        else ([claimed] if claimed is not None else [])
    )  # fmt: skip
    last = log["decisions"][-1]["tracks"] if log["decisions"] else []
    boxes = {r["track"]: np.array(r["footprint"]) for r in last}

    def gap(a, b):
        d = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.0)
        return float(np.linalg.norm(d))

    near = sorted(
        {
            r["track"]
            for r in last
            for f in focus
            if f in boxes
            and r["track"] not in focus
            and gap(boxes[f], boxes[r["track"]]) <= args.neighbour_m
        }
    )
    first_verified = next((v for v in log["verdicts"] if v["verified"]), None)
    accepted = None
    if first_verified is not None and claimed is not None:
        accepted = next(r for r in first_verified["tracks"] if r["track"] == claimed)
    labelled = [r["track"] for r in last]
    out = {
        "job": {k: args.__dict__[k] for k in ("split", "scenario", "target", "seed", "arm")},
        "code_root": str(root),
        "code_sha": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "result": {
            k: result.get(k)
            for k in (
                "result",
                "claimed",
                "completion",
                "sim_s",
                "verdict",
                "claim_nearest",
                "home_seed",
                "crash",
            )
        },  # fmt: skip
        "verifier_thresholds": cfg.get("verifier"),
        "accepted_at": None
        if first_verified is None
        else {"step": first_verified["step"], "sim_s": first_verified["now"]},  # fmt: skip
        "check_track_at_acceptance": accepted,
        "focus_tracks": focus,
        "timelines": {
            str(tid): [a for a in log["associations"] if a["track"] == tid] for tid in focus
        },
        "neighbours": [r for r in last if r["track"] in near],
        "focus_footprints": [r for r in last if r["track"] in focus],
        "candidate_tracks_at_end": labelled,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["result"], default=float))


if __name__ == "__main__":
    main()
