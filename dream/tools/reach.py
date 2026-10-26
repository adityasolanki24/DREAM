"""Goal-conditioned whole-body reach: "touch the <colour> handle", with refusal.

A reusable skill, not a task script. The decision flow below contains no height rules and no
task-family branches: the posture (stance, arm, pelvis height, torso pitch) is whatever the
posture planner returns for the perceived target, and whether a target is reachable at all is
decided by the planner, a contact-aware rollout on a planning copy that contains only PERCEIVED
obstacles, and finally the measured real execution.

Honest labels (AGENTS.md section 6, invariant 5):
- ``parse_reach_instruction`` is STRUCTURED GOAL PARSING of a fixed phrase, not language
  understanding and not VLM reasoning (PLAN C8).
- Perception is the CLASSICAL colour-and-depth baseline (dream/perception/reach_target.py).
- Odometry is ORACLE-ASSISTED in the simulator (dream/sim/g1_mobile.py).

Isolation: this module imports no simulator and no robot code. Everything physical sits behind
the ``ReachBody`` port (implemented by dream/robot/g1_reach_ports.py), so the decision flow is
testable with fakes. Plans are objects with ``.stance``, ``.arm``, ``.target`` (a whole-body
target), ``.score`` and ``.metrics``; plan results have ``.plans``, ``.failures`` and
``.reasons()``.

Flow: parse -> look (a gaze sweep, then turning in place) -> estimate the handle in the
odometry frame with covariance -> plan from the current pose -> walk to the planned stance ->
re-observe -> re-plan from the measured pose at that stance -> rollout check on the planning
copy -> execute under the supervisor -> closed-loop correction against fresh observations ->
verify by the robot's own forward kinematics against the latest perceived target. If the plan
or the correction is limited, ONE bounded re-stance is tried (return to nominal, plan with a new
stance allowed, walk, repeat). Verdicts: reached (with measured error), refused (no feasible
plan within the measured envelope, with the planner's failure reasons) or failed (why).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dream.perception.detect import COLOR_NAMES
from dream.perception.reach_target import HandleEstimator, TargetEstimate

COLORS = tuple(dict.fromkeys(name for name, _, _ in COLOR_NAMES))
VERBS = ("touch", "reach", "tap")
TOUCH_TOLERANCE = (
    0.03  # m, the task's definition of "touched": reach point within this of the handle
)
CLAIM_SIGMAS = (
    1.645  # one-sided 95%: the claim threshold is the tolerance less this many target std devs
)
TURN_STEP = math.pi / 4  # look-around step
MAX_TURNS = 8
MAX_APPROACHES = 2  # moves toward an unresolved box-like object before turning on
MIN_FURNITURE_SIDE = 0.4  # m, footprint side of a box worth walking up to
APPROACH_RANGE = 1.3  # m short of it, where a small handle is resolvable
FUSE_VIEWS = 3  # frames fused while standing still
MAX_STANCE_TRIES = 2  # distinct planned stances tried when the walk fails
PRE_CONTACT_OFFSET = 0.06  # m in front of the handle along the face normal (assumption)
PRE_CONTACT_TOLERANCE = 0.015  # m, free-air tolerance of the pre-contact pose
MAX_PLAN_CHECKS = 4  # ranked plans whose rollouts are checked at a stance before giving up on it
RESTANCE_SHIFT = 0.12  # m: a re-stance must lie at least this far from every stance already tried
RESTANCE_TURN = 0.3  # rad: ... or turned this far
MAX_RESTANCES = 1  # bounded re-stance (PLAN section 2: correction limited -> one new stance)


@dataclass(frozen=True)
class ReachGoal:
    verb: str
    color: str
    noun: str = "handle"
    tolerance: float = TOUCH_TOLERANCE


_PATTERN = re.compile(
    rf"^\s*(?P<verb>{'|'.join(VERBS)})\s+the\s+(?P<color>[a-z]+)\s+(?P<noun>handle)\s*\.?\s*$", re.I
)


def parse_reach_instruction(text: str) -> ReachGoal:
    """STRUCTURED GOAL PARSING of "<verb> the <colour> handle". Anything else is rejected;
    this is neither language understanding nor VLM reasoning."""
    m = _PATTERN.match(text or "")
    if not m:
        raise ValueError(f"unsupported instruction {text!r}: expected 'touch the <colour> handle'")
    color = m["color"].lower()
    if color not in COLORS:
        raise ValueError(f"unknown colour {color!r}; the detector names {', '.join(COLORS)}")
    return ReachGoal(m["verb"].lower(), color, m["noun"].lower())


def approach_axis(target: TargetEstimate):
    """Finger direction for the touch: into the face the handle protrudes from (horizontal, the
    opposite of the face's outward normal), or None when no face was seen. A soft preference of
    the planner, so the fingertip meets the handle squarely instead of edge-on."""
    n = target.face_normal
    return None if n is None else (-float(n[0]), -float(n[1]), 0.0)


def claim_threshold(goal: ReachGoal, estimate: TargetEstimate) -> float:
    """Largest measured error (reach point to the perceived handle) at which "reached" may be
    claimed: the task tolerance less ``CLAIM_SIGMAS`` standard deviations of the perception."""
    return goal.tolerance - CLAIM_SIGMAS * estimate.sigma


@dataclass
class Sight:
    """One head-camera look: detections, camera position (odometry frame) and robot time."""

    detections: list
    camera: np.ndarray
    time: float
    gaze: float | None = None


class ReachBody(Protocol):
    """What the skill needs from a robot (see dream/robot/g1_reach_ports.py)."""

    def pose(self) -> np.ndarray: ...
    def gaze_range(self) -> tuple[float, float]: ...
    def look(self, gaze: float | None) -> Sight: ...
    def turn(self, angle: float) -> bool: ...
    def go_to(self, x: float, y: float, heading: float) -> tuple[bool, str]: ...
    def plan(self, target, covariance, obstacles, stance=None, arm=None, approach=None): ...
    def pre_contact(self, plan, point, obstacles, approach=None): ...
    def verify(self, plans, obstacles): ...
    def execute(self, plan): ...
    def correct(
        self, plan, target_provider, obstacles, executed=None, tolerance=None, approach=None
    ): ...
    def hand_position(self, arm: str) -> np.ndarray: ...
    def recover(self) -> bool: ...
    def stopped(self) -> str | None: ...
    def now(self) -> float: ...


@dataclass
class ReachReport:
    verdict: str  # "reached" | "refused" | "failed"
    reason: str
    goal: ReachGoal
    measured_error: float | None = None  # |FK hand - latest perceived target| (m)
    claim_threshold: float | None = None
    target: dict | None = None
    posture: dict | None = None
    plans: list[dict] = field(default_factory=list)
    rollouts: list[dict] = field(default_factory=list)
    execution: dict = field(default_factory=dict)
    restances: int = 0
    events: list[dict] = field(default_factory=list)


def _describe(est: TargetEstimate) -> dict:
    return {
        "position": [round(float(v), 4) for v in est.position],
        "covariance": np.round(est.covariance, 7).tolist(),
        "sigma": round(est.sigma, 4),
        "color": est.color, "views": est.n_views, "same_colour_candidates": est.candidates,
        "n_obstacles": len(est.obstacles),
        "obstacles": [
            {
                "kind": o.kind, "color": o.color, "hand_may_touch": bool(o.hand_may_touch),
                "center": [round(float(v), 3) for v in o.center],
                "half_extents": [round(float(v), 3) for v in o.half_extents],
                "yaw": round(float(o.yaw), 3),
            }
            for o in est.obstacles
        ],
        "fixture": None if est.fixture is None else {
            "center": [round(v, 3) for v in est.fixture.center],
            "half_extents": [round(v, 3) for v in est.fixture.half_extents],
            "yaw": round(est.fixture.yaw, 3),
        },
    }  # fmt: skip


def _plan_record(stage: str, result) -> dict:
    """Every candidate with its score, and every failure with its reasons."""
    plans = result.plans
    return {
        "stage": stage, "feasible": len(plans), "failed": len(result.failures),
        "stances_blocked": result.n_stances_blocked, "failure_reasons": result.reasons(),
        "candidates": [
            {"stance": [round(float(v), 3) for v in p.stance], "arm": p.arm,
             "pelvis_height": round(float(p.pelvis_height), 3),
             "torso_rpy": [round(float(v), 3) for v in p.torso_rpy],
             "score": round(float(p.score), 3)}
            for p in plans
        ],
        "failures": [
            {"stance": [round(float(v), 3) for v in f.stance], "arm": f.arm,
             "reasons": list(f.reasons)}
            for f in result.failures
        ],
    }  # fmt: skip


class ReachSkill:
    def __init__(self, body: ReachBody, estimator_factory=HandleEstimator, on_event=None):
        self.body, self.estimator_factory = body, estimator_factory
        self.on_event = on_event or (lambda *a, **k: None)

    # -- helpers -------------------------------------------------------------------------
    def _log(self, rep: ReachReport, kind: str, text: str, **data) -> None:
        rep.events.append(
            {"t": round(float(self.body.now()), 2), "kind": kind, "text": text, **data}
        )
        self.on_event(kind, text)

    def _observe(self, est: HandleEstimator, rep: ReachReport, search: bool) -> bool:
        """Look until the named handle is seen, then fuse FUSE_VIEWS frames standing still.

        A gaze sweep first (the head camera rides on the torso; the sweep covers the pitch
        range measured for looking). If the handle is not resolved but a box-like object is in
        view, move closer to it (a small handle is only resolvable at short range); otherwise,
        if ``search``, turn in place and repeat. The nominal gaze is restored afterwards."""
        lo, _ = self.body.gaze_range()
        gazes = [0.0] + ([lo] if lo < -1e-6 else [])
        turns = MAX_TURNS if search else 1
        good: float | None = None
        approaches = 0
        step = 0
        while step < turns:
            last = None
            for g in gazes:
                last = self.body.look(g)
                if est.ingest(last.detections, last.camera, last.time):
                    good = g
                    break
            if good is not None:
                break
            near = self._unresolved_box(last, search and approaches < MAX_APPROACHES)
            if near is not None:
                approaches += 1
                self._log(rep, "look", f"handle not resolved; moving closer to {near}")
                self.body.go_to(*near)
                if self.body.stopped():
                    return False
                continue
            step += 1
            if step < turns:
                self._log(rep, "look", f"not in view; turning {math.degrees(TURN_STEP):.0f} deg")
                if not self.body.turn(TURN_STEP):
                    return False
        if good is None:
            self.body.look(0.0)
            return False
        for _ in range(FUSE_VIEWS - 1):
            s = self.body.look(good)
            est.ingest(s.detections, s.camera, s.time)
        if good != 0.0:
            self.body.look(0.0)
        return True

    def _unresolved_box(self, sight, allowed: bool):
        """A point ``APPROACH_RANGE`` short of the nearest box-like detection in view, with the
        heading that faces it; None if there is no such box or it is already that close."""
        if not allowed or sight is None:
            return None
        boxes = [
            d
            for d in sight.detections
            if d.kind == "box" and max(d.extent[:2]) >= MIN_FURNITURE_SIDE
        ]
        if not boxes:
            return None
        pose = self.body.pose()
        d = min(boxes, key=lambda b: float(np.linalg.norm(b.position[:2] - pose[:2])))
        v = d.position[:2] - pose[:2]
        dist = float(np.linalg.norm(v))
        if dist <= APPROACH_RANGE + 0.3:
            return None
        u = v / dist
        p = d.position[:2] - u * APPROACH_RANGE
        return float(p[0]), float(p[1]), float(math.atan2(u[1], u[0]))

    def _select(self, rep, plans, target, label: str):
        """The first ranked plan whose contact posture passes the rollout gate AND, when the face
        the handle protrudes from is known, whose pre-contact posture (same body, arm only) is
        reachable and passes it too. Returns (plan, pre-contact plan or None, [reasons])."""
        normal, why = target.face_normal, []
        for plan in plans[:MAX_PLAN_CHECKS]:
            picked, tried = self.body.verify([plan], target.obstacles)
            rep.rollouts += [{"stage": label, **_rollout_record(p, r)} for p, r in tried]
            if picked is None:
                why.append(list(tried[0][1].reasons))
                continue
            if normal is None:
                return plan, None, why
            pre_point = target.position + PRE_CONTACT_OFFSET * np.array([normal[0], normal[1], 0.0])
            pre = self.body.pre_contact(
                plan, pre_point, target.obstacles, approach=approach_axis(target)
            )
            if pre is None:
                why.append(["pre_contact_unreachable"])
                continue
            ok, tried_pre = self.body.verify([pre], target.obstacles)
            rep.rollouts += [
                {"stage": label + "_pre", **_rollout_record(p, r)} for p, r in tried_pre
            ]
            if ok is None:
                why.append(["pre_contact:" + ",".join(tried_pre[0][1].reasons)])
                continue
            return plan, pre, why
        return None, None, why

    def _verified_first(self, rep, plans, target, label: str):
        """Walk to a stance whose postures already pass the physical gate: the best ranked plan
        that does goes first (the rest keep their order, as fall-backs). The gate is repeated at
        the stance actually reached."""
        plan, _, _ = self._select(rep, plans, target, label)
        return plans if plan is None else [plan] + [p for p in plans if p is not plan]

    def _new_stances(self, plans):
        """Plans at stances not already tried (a re-stance must be a different stance)."""
        out = []
        for p in plans:
            near = any(
                math.hypot(p.stance[0] - t[0], p.stance[1] - t[1]) < RESTANCE_SHIFT
                and abs((p.stance[2] - t[2] + math.pi) % (2 * math.pi) - math.pi) < RESTANCE_TURN
                for t in self._tried
            )
            if not near:
                out.append(p)
        return out

    def _reobserve(self, est: HandleEstimator, rep: ReachReport) -> TargetEstimate | None:
        est.reset()
        return est.estimate() if self._observe(est, rep, search=True) else None

    # -- main ----------------------------------------------------------------------------
    def run(self, instruction: str) -> ReachReport:
        try:
            goal = parse_reach_instruction(instruction)
        except ValueError as e:
            return ReachReport("refused", f"unsupported instruction: {e}", ReachGoal("", "", ""))
        rep = ReachReport("failed", "", goal)
        self._tried: list = []  # measured poses at which a posture was judged limited
        est = self.estimator_factory(goal.color)
        self._log(
            rep, "goal", f"{goal.verb} the {goal.color} {goal.noun} (structured goal parsing)"
        )
        if not self._observe(est, rep, search=True):
            rep.reason = f"the {goal.color} handle was not found"
            self._log(rep, "result", rep.reason)
            return rep
        target = est.estimate()
        rep.target = _describe(target)
        self._log(rep, "perceived", f"handle at {rep.target['position']} sigma {target.sigma:.3f}")
        if target.candidates > 1:
            self._log(
                rep, "warn", f"{target.candidates} handles of that colour in view; nearest used"
            )

        result = self.body.plan(
            target.position, target.covariance, target.obstacles, approach=approach_axis(target)
        )
        rep.plans.append(_plan_record("initial", result))
        if not result.feasible:
            rep.verdict = "refused"
            rep.reason = f"no feasible plan within the measured envelope: {result.reasons()}"
            self._log(rep, "result", rep.reason)
            return rep
        plans = self._verified_first(rep, result.plans, target, "ring")
        for attempt in range(MAX_RESTANCES + 1):
            if attempt:
                rep.restances += 1
                self._log(rep, "restance", "limited at this stance: planning a new stance")
                if self.body.stopped():
                    rep.reason = "robot stopped: " + str(self.body.stopped())
                    return rep
                if not self.body.recover():
                    rep.reason = "could not return to the nominal posture before re-stancing"
                    return rep
                target = self._reobserve(est, rep)
                if target is None:
                    rep.reason = "handle lost before re-stancing"
                    return rep
                result = self.body.plan(
                    target.position,
                    target.covariance,
                    target.obstacles,
                    approach=approach_axis(target),
                )
                rep.plans.append(_plan_record("restance", result))
                if not result.feasible:
                    rep.verdict = "refused"
                    rep.reason = (
                        f"no feasible plan within the measured envelope: {result.reasons()}"
                    )
                    self._log(rep, "result", rep.reason)
                    return rep
                plans = self._new_stances(result.plans)
                if not plans:
                    rep.verdict = "refused"
                    rep.reason = "no feasible plan at a stance not already tried"
                    self._log(rep, "result", rep.reason)
                    return rep
                plans = self._verified_first(rep, plans, target, "ring")
            done = self._walk_then_reach(rep, est, goal, plans, attempt)
            if done in ("reached", "refused", "failed_final"):
                break
        return rep

    def _walk_then_reach(self, rep, est, goal, plans, attempt) -> str:
        """One stance attempt. Returns "reached", "refused", "failed_final" or "limited"."""
        arrived = None
        for plan in plans[:MAX_STANCE_TRIES]:
            x, y, yaw = plan.stance
            ok, msg = self.body.go_to(x, y, yaw)
            self._log(rep, "go_to", f"stance {x:.2f} {y:.2f} {yaw:.2f}: {'ok' if ok else msg}")
            if self.body.stopped():
                rep.reason = f"robot stopped while walking: {self.body.stopped()}"
                rep.verdict = "failed"
                return "failed_final"
            if ok:
                arrived = plan
                break
        if arrived is None:
            rep.verdict, rep.reason = "failed", "could not walk to any planned stance"
            return "failed_final"
        target = self._reobserve(est, rep)
        if target is None:
            rep.verdict, rep.reason = "failed", "handle lost after walking to the stance"
            return "failed_final"
        rep.target = _describe(target)
        # Re-plan from the MEASURED pose: the stance is where the feet actually are.
        pose = self.body.pose()
        here = self.body.plan(
            target.position,
            target.covariance,
            target.obstacles,
            stance=pose,
            approach=approach_axis(target),
        )
        rep.plans.append(_plan_record("at_stance", here))
        chosen = stage1 = None
        why: list = []
        if here.feasible:
            chosen, pre, why = self._select(rep, here.plans, target, "at_stance")
            stage1 = pre or chosen
        if chosen is None:
            self._tried.append(tuple(float(v) for v in pose))
            if attempt < MAX_RESTANCES:
                return "limited"
            rep.verdict = "refused"
            reason = here.reasons() if not here.feasible else {"rollout": why}
            rep.reason = f"no feasible plan within the measured envelope: {reason}"
            self._log(rep, "result", rep.reason)
            return "refused"
        rep.posture = _posture(chosen)
        rep.posture["pre_contact"] = None if pre is None else _posture(pre)
        self._log(rep, "posture", f"chosen {rep.posture}")
        # Stage 1 is a pre-contact pose a few cm in front of the handle along the face normal,
        # corrected in free air; stage 2 advances onto the handle. Contact is only made with a
        # closed-loop approach, never by the open-loop execution (the pretrained controller
        # yaws and sags under an extended arm, by several cm).
        normal = target.face_normal
        staged = stage1 is not chosen
        run = self.body.execute(stage1)
        rep.execution = {
            "completed": run.completed, "refused": run.refused, "staged": staged,
            "stops": [(s.t, s.reason) for s in run.stops],
        }  # fmt: skip
        if run.refused or run.stops or not run.completed:
            why = run.refused or (run.stops[0].reason if run.stops else "incomplete")
            self.body.recover()
            rep.verdict, rep.reason = "failed", f"execution stopped: {why}"
            self._log(rep, "result", rep.reason)
            return "failed_final"
        latest = {"target": target, "fresh": 0, "stale": 0, "normal": normal}

        def provider():
            est.reset()
            s = self.body.look(None)
            if est.ingest(s.detections, s.camera, s.time):
                fresh = est.estimate()
                latest["target"], latest["fresh"] = fresh, latest["fresh"] + 1
                latest["normal"] = fresh.face_normal or latest["normal"]
            else:
                latest["stale"] += 1
            return latest["target"].position

        def pre_provider():
            p = provider()
            n = latest["normal"]
            return p + PRE_CONTACT_OFFSET * np.array([n[0], n[1], 0.0])

        corrections = []
        executed = None
        if staged:
            corr = self.body.correct(
                stage1,
                pre_provider,
                target.obstacles,
                tolerance=PRE_CONTACT_TOLERANCE,
                approach=approach_axis(target),
            )
            corrections.append(("pre_contact", corr))
            executed = corr.final_target or stage1.target
        if not (corrections and (corrections[-1][1].stops or corrections[-1][1].refused)):
            corr = self.body.correct(
                chosen,
                provider,
                target.obstacles,
                executed=executed,
                approach=approach_axis(latest["target"]),
            )
            corrections.append(("contact" if staged else "direct", corr))
        rep.execution["correction"] = [
            {"stage": name, **{k: v for k, v in it.items() if k != "hand"}}
            for name, c in corrections for it in c.iterations
        ]  # fmt: skip
        rep.execution["observations"] = {"fresh": latest["fresh"], "stale": latest["stale"]}
        bad = next((c for _, c in corrections if c.stops or c.refused), None)
        if bad is not None:
            why = bad.refused or bad.stops[0].reason
            self.body.recover()
            rep.verdict, rep.reason = "failed", f"correction stopped: {why}"
            self._log(rep, "result", rep.reason)
            return "failed_final"
        corr = corrections[-1][1]
        # Verify by the robot's OWN measurement: FK hand against the latest perceived target.
        final = latest["target"]
        err = float(np.linalg.norm(self.body.hand_position(chosen.arm) - final.position))
        thr = claim_threshold(goal, final)
        rep.measured_error, rep.claim_threshold = err, thr
        rep.target = _describe(final)
        if err <= thr:
            rep.verdict = "reached"
            rep.reason = f"reach point {err * 1000:.1f} mm from the perceived handle"
            self._log(rep, "result", rep.reason)
            return "reached"
        last = corr.iterations[-1]["action"] if corr.iterations else None
        rep.reason = (
            f"hand {err * 1000:.1f} mm from the perceived handle, above the claim threshold "
            f"{thr * 1000:.1f} mm (correction ended: {last})"
        )
        self._log(rep, "limited", rep.reason)
        if err <= goal.tolerance:
            # Probably touching (inside the task tolerance) but not certain enough to claim: a
            # new stance cannot make the perception better, so this is reported, not retried.
            rep.verdict = "failed"
            rep.reason += "; within the tolerance but not certain enough to claim"
            self._log(rep, "result", rep.reason)
            return "failed_final"
        self._tried.append(tuple(float(v) for v in pose))
        if attempt < MAX_RESTANCES:
            return "limited"
        rep.verdict = "failed"
        self._log(rep, "result", rep.reason)
        return "failed_final"


def _posture(plan) -> dict:
    return {
        "stance": [round(float(v), 3) for v in plan.stance], "arm": plan.arm,
        "pelvis_height": round(float(plan.pelvis_height), 3),
        "torso_rpy": [round(float(v), 3) for v in plan.torso_rpy],
        "score": round(float(plan.score), 3),
        "metrics": {k: round(float(v), 4) for k, v in plan.metrics.items()},
    }  # fmt: skip


def _rollout_record(plan, r) -> dict:
    return {
        "stance": [round(float(v), 3) for v in plan.stance], "arm": plan.arm,
        "passed": bool(r.passed), "reasons": list(r.reasons),
        "metrics": {k: round(float(v), 4) for k, v in r.metrics.items()},
        "seconds": round(float(r.seconds), 1), "simulated": round(float(r.simulated), 1),
    }  # fmt: skip
