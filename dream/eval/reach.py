"""Independent evaluation of the unfamiliar-height reach test. GROUND TRUTH IS READ ONLY HERE.

Nothing in this module feeds the agent: it samples the simulator after every policy tick
(``ReachProbe`` wraps ``env.step``) and judges the end state against the scene's truth.

Measured per episode:
- true distance between the hand's reach point (the index fingertip, from the robot model) and
  the handle centre, for both hands, at the end;
- collapse by the C3 criteria (PLAN 2b), never by pelvis height: tilt latch of the controller,
  undesired ground contact (any non-foot geom of the robot on the floor), loss of foot support,
  joint-limit contact; joint-limit contact is also reported on its own;
- protective stops (from the supervisor) and whether one is still unresolved (the backend is
  holding) at the end;
- contacts between the robot and the scene: the intended one (a hand on the handle, with its peak
  force) and the unintended ones (the fixture body, distractors, any non-hand part), with the
  peak force;
- false success claims: the agent says "reached" and the truth says otherwise;
- reference feasibility: the same planner and rollout, given the TRUE handle and the TRUE fixture
  boxes and an exhaustive stance sampling, under the same envelope. A refusal is correct exactly
  when the target is reference-infeasible; a refusal of a reference-feasible target is a missed
  opportunity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

SUCCESS_RADIUS = 0.03  # m, pre-registered
TICK = 0.02  # s, one policy step
# Collapse gates are the ones the W1 envelope experiment froze (experiments/configs/
# g1_wbc_envelope_v2.json); the joint margin is its joint_limit_margin_rad.
UNDESIRED_CONTACT_MAX_S = 0.1
FOOT_LOSS_MAX_S = 0.3
JOINT_LIMIT_MAX_S = 0.2
JOINT_LIMIT_MARGIN = 0.02
FOOT_FORCE_MIN = 1.0  # N
HAND_LINKS = ("wrist_yaw", "hand_thumb", "hand_middle", "hand_index", "hand_palm")


def longest(flags: list[bool]) -> float:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * TICK


@dataclass
class ReachProbe:
    """Samples contacts, support and joint limits every policy tick. Attach before the episode."""

    env: object
    controller: object
    grips: dict  # arm -> reach point in the wrist_yaw_link frame (robot anatomy)
    scene_geoms: dict  # geom name -> category: "handle" | "fixture" | "distractor"
    rows: list = field(default_factory=list)
    force: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = self.env.model
        self.floor = m.geom("floor").id
        self.feet = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
        self.foot_ids = (m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id)
        self.scene = {m.geom(n).id: c for n, c in self.scene_geoms.items()}
        root = m.body("pelvis").id
        self.robot_body = np.array([_is_under(m, b, root) for b in range(m.nbody)])
        self.hand_body = np.array(
            [
                self.robot_body[b]
                and any(f"_{k}" in m.body(b).name or m.body(b).name.endswith(k) for k in HAND_LINKS)
                for b in range(m.nbody)
            ]
        )
        self.wrist = {a: m.body(f"{a}_wrist_yaw_link").id for a in ("left", "right")}
        joints = m.actuator_trnid[self.env.body_actuators, 0]
        self.jlo, self.jhi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        self.peak = {k: 0.0 for k in ("handle_hand", "fixture_hand", "fixture_other", "distractor")}
        self.touched = {k: 0 for k in self.peak}
        self.min_dist = {"left": math.inf, "right": math.inf}
        self._handle = None

    def attach(self, handle_center) -> None:
        """Wrap ``env.step`` so every policy tick is sampled; nothing is changed in the physics."""
        self._handle = np.asarray(handle_center, float)
        step = self.env.step

        def stepped(controller):
            step(controller)
            self.sample()

        self.env.step = stepped

    def hand_points(self) -> dict:
        d = self.env.data
        return {
            a: d.xpos[w] + d.xmat[w].reshape(3, 3) @ self.grips[a] for a, w in self.wrist.items()
        }

    def sample(self) -> None:
        env, m, d = self.env, self.env.model, self.env.data
        foot, bad = [False, False], False
        f6 = np.zeros(6)
        step_peak = dict.fromkeys(self.peak, 0.0)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                body = int(m.geom_bodyid[a])
                if not self.robot_body[body]:
                    continue
                if b == self.floor:
                    if body in self.foot_ids:
                        mujoco.mj_contactForce(m, d, i, f6)
                        if f6[0] > FOOT_FORCE_MIN:
                            foot[self.foot_ids.index(body)] = True
                    else:
                        bad = True
                elif b in self.scene:
                    mujoco.mj_contactForce(m, d, i, f6)
                    cat = self.scene[b]
                    hand = bool(self.hand_body[body])
                    key = (
                        "handle_hand" if cat == "handle" and hand
                        else "fixture_hand" if cat == "fixture" and hand
                        else "fixture_other" if cat in ("fixture", "handle")
                        else "distractor"
                    )  # fmt: skip
                    step_peak[key] = max(step_peak[key], float(f6[0]))
                    self.touched[key] += 1
        for k, v in step_peak.items():
            self.peak[k] = max(self.peak[k], v)
        q = d.qpos[env.qadr]
        near = bool(
            np.any((q - self.jlo < JOINT_LIMIT_MARGIN) | (self.jhi - q < JOINT_LIMIT_MARGIN))
        )
        pts = self.hand_points()
        if self._handle is not None:
            for a, p in pts.items():
                self.min_dist[a] = min(self.min_dist[a], float(np.linalg.norm(p - self._handle)))
        cmd = np.asarray(getattr(self.controller, "command", np.zeros(3)))
        self.rows.append(
            {
                "t": float(env.time), "foot": foot, "bad": bad, "limit": near,
                "walking": bool(np.linalg.norm(cmd) >= 0.05),
                "stopped": bool(self.controller.stopped),
            }
        )  # fmt: skip

    # -- verdict -------------------------------------------------------------------------
    def collapse(self) -> dict:
        rows = self.rows
        contact = longest([r["bad"] for r in rows])
        # Support: standing (balance policy) needs both feet; walking needs at least one.
        lost = [
            (not (r["foot"][0] and r["foot"][1])) if not r["walking"] else not any(r["foot"])
            for r in rows
        ]
        support = longest(lost)
        limit_total = sum(r["limit"] for r in rows) * TICK
        tilt = any(r["stopped"] for r in rows)
        flags = {
            "tilt_latch": tilt,
            "undesired_ground_contact": contact > UNDESIRED_CONTACT_MAX_S,
            "foot_support_loss": support > FOOT_LOSS_MAX_S,
            "joint_limit_contact": limit_total > JOINT_LIMIT_MAX_S,
        }
        return {
            **flags, "collapsed": any(flags.values()),
            "undesired_contact_max_s": contact, "foot_loss_max_s": support,
            "joint_limit_total_s": limit_total,
        }  # fmt: skip


def _is_under(m, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def scene_geom_categories(scene) -> dict:
    out = {scene.fixture.name: "fixture", scene.handle.name: "handle"}
    out.update({s.name: "distractor" for s in scene.distractors})
    return out


def judge(probe: ReachProbe, scene, verdict: str, held, stops: list) -> dict:
    """The evaluator's verdict for one finished episode (see the module docstring)."""
    target = np.asarray(scene.handle_center, float)
    pts = probe.hand_points()
    dist = {a: float(np.linalg.norm(p - target)) for a, p in pts.items()}
    true_distance = min(dist.values())
    collapse = probe.collapse()
    unresolved = held is not None
    reached = true_distance <= SUCCESS_RADIUS
    ok = reached and not collapse["collapsed"] and not unresolved
    unintended = probe.touched["fixture_other"] + probe.touched["distractor"] > 0
    return {
        "true_distance_m": true_distance,
        "true_distance_by_arm": dist,
        "min_distance_during_m": dict(probe.min_dist),
        "physical_success": bool(ok),
        "claimed": verdict == "reached",
        "claimed_success": bool(ok and verdict == "reached"),
        "false_success_claim": bool(verdict == "reached" and not reached),
        "collapse": collapse,
        "joint_limit_contact_s": collapse["joint_limit_total_s"],
        "protective_stops": [{"t": t, "reason": r} for t, r in stops],
        "unresolved_protective_stop": unresolved,
        "contacts": {
            "peak_force_n": dict(probe.peak),
            "ticks": dict(probe.touched),
            "unintended_contact": bool(unintended),
        },  # fmt: skip
    }


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson 95% score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


# -- reference feasibility -----------------------------------------------------------------------
def true_boxes(scene) -> list:
    """The scene's solid bodies as planner boxes (TRUE geometry, evaluator only). The fixture is
    marked hand-touchable (it carries the target); the handle is the target, not an obstacle."""
    from dream.robot.g1_posture import PerceivedBox

    out = [
        PerceivedBox(
            tuple(scene.fixture.center), tuple(scene.fixture.half), scene.fixture.yaw, True
        )
    ]
    out += [PerceivedBox(tuple(s.center), tuple(s.half), s.yaw, False) for s in scene.distractors]
    return out


def reference_feasibility(scene, planner, backend_factory, k: int = 60) -> dict:
    """Feasible with the TRUE target and obstacles, exhaustive stance sampling, same envelope.

    The planner here is a denser sampler (more rings and angles) than the agent's; plans are
    rolled out on a planning copy in rank order, up to ``k``, until one passes."""
    from dataclasses import replace

    from dream.robot.g1_rollout import verify_top_k

    boxes = true_boxes(scene)
    dense = replace(
        planner.config, radii=(0.35, 0.4, 0.45, 0.5, 0.6, 0.7), n_angles=16, max_stances=10_000
    )
    planner = type(planner)(planner.caps, dense)
    # The robot's start is irrelevant to feasibility; stances are ranked by distance from the
    # origin.
    result = planner.plan(np.asarray(scene.handle_center), None, (0.0, 0.0, 0.0), boxes)
    chosen, tried = verify_top_k(
        result.plans, backend_factory, boxes, k=k, planning_model=planner.pm
    )
    best = chosen
    return {
        "feasible": chosen is not None,
        "n_plans": len(result.plans),
        "n_failures": len(result.failures),
        "failure_reasons": result.reasons(),
        "rollouts": [
            {
                "stance": [round(float(v), 3) for v in p.stance],
                "arm": p.arm,
                "passed": bool(r.passed),
                "reasons": list(r.reasons),
                "hand_error": round(float(r.metrics.get("hand_error", math.nan)), 4),
            }
            for p, r in tried
        ],  # fmt: skip
        "posture": None
        if best is None
        else {
            "stance": [round(float(v), 3) for v in best.stance],
            "arm": best.arm,
            "pelvis_height": round(float(best.pelvis_height), 3),
            "torso_rpy": [round(float(v), 3) for v in best.torso_rpy],
        },  # fmt: skip
    }
