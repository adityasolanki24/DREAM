"""Operate the mechanism using the agent's current best explanation of it.

Task: "open it", meaning drive the joint to at least ``TASK_GOAL`` (0.10 m for a slide, 0.5 rad
for a hinge). If the most probable explanation is that the part is fixed, the agent declines
to act and reports that instead.

The plan is a sequence of gripper targets along the believed constraint manifold. Each target
leads the believed handle position slightly along the tangent, so the compliant spring always
pulls in the direction of motion and overcomes friction.
"""

from __future__ import annotations

import numpy as np

from dream.agent.belief import Belief
from dream.agent.model import FIXED, PRISMATIC, REVOLUTE, Hypotheses, handle_at, project, tangent

TASK_GOAL = {PRISMATIC: 0.10, REVOLUTE: 0.5}  # m, rad
GOAL_MARGIN = 1.25  # aim past the goal; joint limits and the force cap stop us safely


def best_explanation(belief: Belief) -> Hypotheses | None:
    """Posterior-mean parameters of the most probable type, or ``None`` if that type is fixed."""
    kind = belief.map_type()
    if kind == FIXED:
        return None
    est = belief.estimate(kind)
    return Hypotheses(
        kind=np.array([kind]),
        axis=est["axis"][None],
        origin=est.get("origin", np.zeros(3))[None],
        upper=np.array([np.inf]),  # the limit estimate is rarely informed before we hit it
        friction=np.array([float(est["friction"])]),
    )


def plan_opening(
    belief: Belief, handle_now: np.ndarray, n_waypoints: int = 30, lead: float = 0.015
) -> list[np.ndarray] | None:
    """Gripper targets (world frame) that open the mechanism, or ``None`` to declare it fixed."""
    hyp = best_explanation(belief)
    if hyp is None:
        return None
    kind = int(hyp.kind[0])
    q_now = float(project(hyp, handle_now, belief.h0)[0])
    q_goal = max(q_now, GOAL_MARGIN * TASK_GOAL[kind])
    return _targets(belief, hyp, np.linspace(q_now, q_goal, n_waypoints)[1:], lead)


def plan_segment(
    belief: Belief, handle_now: np.ndarray, span: float, n_waypoints: int = 8, lead: float = 0.015
) -> tuple[list[np.ndarray], float, float] | None:
    """Targets that advance the believed joint by ``span`` (m or rad) from the handle's position.

    For closed-loop opening in short segments: after each one the caller re-observes, checks
    the handle stayed on the believed constraint, and stops when progress stalls at a limit.
    Returns ``(targets, q_now, q_goal)``, or ``None`` if the best explanation is "fixed".
    """
    hyp = best_explanation(belief)
    if hyp is None:
        return None
    q_now = float(project(hyp, handle_now, belief.h0)[0])
    qs = np.linspace(q_now, q_now + span, n_waypoints + 1)[1:]
    return _targets(belief, hyp, qs, lead), q_now, q_now + span


def constraint_deviation(belief: Belief, handle: np.ndarray) -> tuple[float, float]:
    """(joint coordinate, distance in m) of an observed handle from the believed constraint."""
    hyp = best_explanation(belief)
    if hyp is None:
        return 0.0, float(np.linalg.norm(handle - belief.h0))
    q = project(hyp, handle, belief.h0)
    return float(q[0]), float(np.linalg.norm(handle_at(hyp, q, belief.h0)[0] - handle))


def _targets(belief: Belief, hyp: Hypotheses, qs: np.ndarray, lead: float) -> list[np.ndarray]:
    """Each target leads the believed handle position along the tangent by enough to overcome
    the believed friction through the controller's spring."""
    targets = []
    for q in qs:
        x = handle_at(hyp, np.array([q]), belief.h0)
        t = tangent(hyp, x)[0]
        t /= max(np.linalg.norm(t), 1e-9)
        push = lead + float(hyp.friction[0]) / belief.controller.stiffness
        targets.append(x[0] + t * push)
    return targets
