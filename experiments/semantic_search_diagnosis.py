"""Diagnosis of why room knowledge does not speed up object search (task 2026-11-09-001).

EVALUATION ONLY. Two parts:

``Diagnosis`` is an event hook for ``TaskAgent.run(on_event=...)``. It reads the robot's regions
and belief (what the robot knows) next to the true layout (what only the evaluator knows) after
every planner step, and writes only to itself, so it cannot influence the agent. Per step it
records the robot time, the purpose of the action, whether the room that typically holds the
target exists in the robot's map and is labelled correctly, whether a candidate track of the
target exists near a true instance, and the largest semantic prior factor of any region.

``main`` reads an ``episodes.jsonl`` written with ``--diagnose`` (which also stores the planner's
per-decision log, see ``SemanticSearchPlanner._record_decision``) and prints the aggregate:

    python experiments/semantic_search_diagnosis.py RESULT_DIR [--out summary.json]

Development seeds only. It must not be pointed at a held-out directory to tune anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PULL_FACTOR = 1.5  # a region whose expected likelihood ratio for the target is at least this
SEEN_RADIUS = 1.0  # m: a target-labelled track this close to a true instance is that instance


class Diagnosis:
    """See the module docstring. ``instances`` are ``TrueInstance`` rows, ``room_at`` maps a
    world point to a true room type (or None), ``typical`` is the generator's typical room
    type of the target (an evaluator's reference, never given to the agent)."""

    def __init__(self, world, planner, base, true_pose, instances, room_at, target, typical):
        from dream.eval.semantic_scoring import ROOM_MATCH

        self.world, self.planner, self.base = world, planner, base
        self.true_pose, self.room_at = true_pose, room_at
        self.target, self.typical = target, typical
        self.match = ROOM_MATCH
        self.targets = [i for i in instances if i.cls == target]
        self.timeline: list[dict] = []
        self.steps: list[dict] = []
        self._open: dict | None = None

    # ----- event hook ------------------------------------------------------------------------
    def on_event(self, kind: str, data: dict) -> None:
        if kind == "verification":
            self._snapshot()
        elif kind == "action":
            self._open = {
                "step": len(self.steps), "t0": round(float(self.base.time), 2),
                "call": data["tool"], "purpose": self.planner._purpose.get("purpose"),
            }  # fmt: skip
        elif kind == "result" and self._open is not None:
            self._open["t1"] = round(float(self.base.time), 2)
            self.steps.append(self._open)
            self._open = None

    def _snapshot(self) -> None:
        from dream.eval.semantic_scoring import odometry_to_world

        world = self.world
        to_world = odometry_to_world(self.true_pose(), self.base.odometry())
        factor = world.target.region_factor if world.target is not None else {}
        in_typical = labelled_typical = labelled_target_room = n_labelled = 0
        target_rooms = {t.room_type for t in self.targets}
        for r in world.regions.values():
            truth = self.room_at(to_world(r.centroid))
            top = r.leading()[0] if r.evidence else None
            labelled = top is not None and top != "unknown"
            n_labelled += labelled
            ok = labelled and truth is not None and top in self.match.get(truth, {truth})
            if truth == self.typical:
                in_typical += 1
                labelled_typical += ok
            if truth in target_rooms:
                labelled_target_room += ok
        seen = False
        for tr in world.map.candidates(self.target):
            w = to_world(tr.position)
            if any(t.distance(w) <= SEEN_RADIUS for t in self.targets):
                seen = True
                break
        self.timeline.append({
            "step": len(self.steps), "t": round(float(self.base.time), 2),
            "regions": len(world.regions), "labelled": int(n_labelled),
            "typical_region_exists": bool(in_typical),
            "typical_labelled_correct": bool(labelled_typical),
            "target_room_labelled_correct": bool(labelled_target_room),
            "max_factor": round(max(factor.values(), default=1.0), 3),
            "target_seen": seen,
        })  # fmt: skip

    def result(self) -> dict:
        return {
            "typical_room": self.typical,
            "target_room_types": sorted({t.room_type for t in self.targets}),
            "timeline": self.timeline, "steps": self.steps,
        }


# ----- aggregation -------------------------------------------------------------------------------
def first(timeline: list[dict], pred) -> dict | None:
    return next((s for s in timeline if pred(s)), None)


def episode_stats(row: dict) -> dict | None:
    d, decisions = row.get("diagnosis"), row.get("decisions")
    if d is None or not row.get("target_exists"):
        return None
    tl = d["timeline"]
    seen = first(tl, lambda s: s["target_seen"])
    pull = first(tl, lambda s: s["max_factor"] >= PULL_FACTOR)
    typical_exists = first(tl, lambda s: s["typical_region_exists"])
    typical_label = first(tl, lambda s: s["typical_labelled_correct"])
    room_label = first(tl, lambda s: s["target_room_labelled_correct"])
    changed = [x for x in decisions if x["changed_uniform"]]
    first_change = min((x["step"] for x in changed), default=None)
    steps = d["steps"]
    by_purpose: dict[str, float] = defaultdict(float)
    for s in steps:
        by_purpose[s["purpose"] or "other"] += s["t1"] - s["t0"]
    total = sum(by_purpose.values()) or 1.0
    t_seen = None if seen is None else seen["t"]
    verifying = sum(v for k, v in by_purpose.items() if k in ("verify", "verify_look", "reach",
                                                              "reach_look"))  # fmt: skip
    return {
        "arm": row["arm_name"], "scenario": row["scenario"], "target": row["target"],
        "seed": row["seed"], "result": row["result"], "sim_s": row["sim_s"],
        "decisions": len(decisions),
        "changed_uniform": len(changed),
        "changed_no_detection": sum(x["changed_no_detection"] for x in decisions),
        "first_change_step": first_change,
        "seen_t": t_seen, "seen_step": None if seen is None else seen["step"],
        "seen_at_initial_look": seen is not None and seen["step"] <= 1,
        "pull_t": None if pull is None else pull["t"],
        "typical_exists_t": None if typical_exists is None else typical_exists["t"],
        "typical_label_t": None if typical_label is None else typical_label["t"],
        "target_room_label_t": None if room_label is None else room_label["t"],
        "seen_before_pull": seen is not None and (pull is None or seen["t"] <= pull["t"]),
        "seen_before_typical_label": seen is not None
        and (typical_label is None or seen["t"] <= typical_label["t"]),
        "seen_before_first_change": seen is not None and (
            first_change is None or seen["step"] <= first_change),
        "seen_before_typical_exists": seen is not None and (
            typical_exists is None or seen["t"] <= typical_exists["t"]),
        "explore_s": by_purpose.get("explore", 0.0) + by_purpose.get("explore_look", 0.0)
        + by_purpose.get("initial_look", 0.0),
        "verify_s": verifying,
        "time_total_s": total,
        "time_to_seen_s": t_seen,
        "time_seen_to_end_s": None if t_seen is None else row["sim_s"] - t_seen,
        "frac_decisions_label_labelled": float(np.mean(
            [x["chosen"]["label"] != "none" for x in decisions])) if decisions else None,
        "frac_candidates_in_region": float(np.mean(
            [x["n_in_region"] / max(x["n_views"], 1) for x in decisions])) if decisions else None,
        "frac_candidates_factor_off_1": float(np.mean(
            [x["n_factor_off_1"] / max(x["n_views"], 1) for x in decisions]
        )) if decisions else None,
        "chosen_frontier_share": float(np.mean(
            [x["chosen"]["frontier"] for x in decisions])) if decisions else None,
        "median_sem_term_chosen": float(np.median(
            [x["chosen"]["sem_term"] for x in decisions])) if decisions else None,
        "median_gap": float(np.median(
            [x["gap_to_second"] for x in decisions if x["gap_to_second"] is not None]))
        if any(x["gap_to_second"] is not None for x in decisions) else None,
        "median_max_sem_shift": float(np.median(
            [x["max_abs_sem_shift"] for x in decisions])) if decisions else None,
    }  # fmt: skip


def med(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else round(float(np.median(xs)), 2)


def aggregate(rows: list[dict]) -> dict:
    stats = [s for s in (episode_stats(r) for r in rows) if s]
    out: dict = {"episodes": len(stats), "by_arm": {}}
    for arm in sorted({s["arm"] for s in stats}):
        ss = [s for s in stats if s["arm"] == arm]
        n_dec = sum(s["decisions"] for s in ss)
        ch_u = sum(s["changed_uniform"] for s in ss)
        ch_n = sum(s["changed_no_detection"] for s in ss)
        eps_with_change = sum(s["changed_uniform"] > 0 for s in ss)
        seen = [s for s in ss if s["seen_t"] is not None]
        out["by_arm"][arm] = {
            "episodes": len(ss), "found": sum(s["result"] == "success" for s in ss),
            "decisions": n_dec,
            "frac_decisions_changed_by_knowledge": round(ch_u / n_dec, 4) if n_dec else None,
            "frac_decisions_changed_by_dropping_detection": round(ch_n / n_dec, 4)
            if n_dec else None,
            "episodes_with_any_knowledge_change": eps_with_change,
            "episodes_target_seen": len(seen),
            "seen_at_initial_look": sum(s["seen_at_initial_look"] for s in ss),
            "median_time_to_seen_s": med(s["time_to_seen_s"] for s in ss),
            "median_time_seen_to_end_s": med(s["time_seen_to_end_s"] for s in ss),
            "seen_before_any_pull": sum(s["seen_before_pull"] for s in seen),
            "seen_before_typical_room_labelled": sum(s["seen_before_typical_label"] for s in seen),
            "seen_before_typical_room_in_map": sum(s["seen_before_typical_exists"] for s in seen),
            "seen_before_first_knowledge_change": sum(s["seen_before_first_change"] for s in seen),
            "typical_room_ever_in_map": sum(s["typical_exists_t"] is not None for s in ss),
            "typical_room_ever_labelled_correct": sum(s["typical_label_t"] is not None for s in ss),
            "median_t_typical_in_map_s": med(s["typical_exists_t"] for s in ss),
            "median_t_typical_labelled_s": med(s["typical_label_t"] for s in ss),
            "median_t_first_pull_s": med(s["pull_t"] for s in ss),
            "median_explore_share": med(s["explore_s"] / s["time_total_s"] for s in ss),
            "median_verify_share": med(s["verify_s"] / s["time_total_s"] for s in ss),
            "mean_candidates_in_a_labelled_region": round(float(np.mean(
                [s["frac_candidates_in_region"] for s in ss
                 if s["frac_candidates_in_region"] is not None])), 3),
            "mean_candidates_with_factor_off_1": round(float(np.mean(
                [s["frac_candidates_factor_off_1"] for s in ss
                 if s["frac_candidates_factor_off_1"] is not None])), 3),
            "median_gap_to_second_m2": med(s["median_gap"] for s in ss),
            "median_max_sem_shift_m2": med(s["median_max_sem_shift"] for s in ss),
            "by_scenario": {},
        }
        for sc in sorted({s["scenario"] for s in ss}):
            sub = [s for s in ss if s["scenario"] == sc]
            d = sum(s["decisions"] for s in sub)
            out["by_arm"][arm]["by_scenario"][sc] = {
                "episodes": len(sub),
                "frac_changed": round(sum(s["changed_uniform"] for s in sub) / d, 4) if d else None,
                "median_t_typical_labelled_s": med(s["typical_label_t"] for s in sub),
                "median_time_to_seen_s": med(s["time_to_seen_s"] for s in sub),
            }
    out["episode_stats"] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    path = pathlib.Path(args.result_dir) / "episodes.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    agg = aggregate(rows)
    out = pathlib.Path(args.out or pathlib.Path(args.result_dir) / "diagnosis.json")
    out.write_text(json.dumps(agg, indent=1, default=float))
    print(json.dumps(agg["by_arm"], indent=1))


if __name__ == "__main__":
    main()
