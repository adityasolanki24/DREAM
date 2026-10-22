"""One closed-loop episode: observe -> hypothesise -> predict -> act -> measure -> update.

The runner connects the environment and the agent, and is the only place where both appear.
Agent decisions use only observations. Ground truth is used afterwards, on forked copies of the
simulator, to score identification, held-out prediction and task success at every step.
"""

from __future__ import annotations

import time

import numpy as np

from dream.agent.belief import Belief, Prior, make_record
from dream.agent.exploration import predict_for_probes
from dream.agent.model import TYPE_NAMES, relative_rotvec
from dream.agent.operate import TASK_GOAL, plan_opening
from dream.eval.metrics import score_belief
from dream.sim.env import MechanismEnv
from dream.sim.mechanisms import MechanismSpec
from dream.types import Probe


def _obs_dict(obs) -> dict:
    return {
        "time": obs.time,
        "handle_pos": obs.handle_pos.round(6).tolist(),
        "handle_quat": obs.handle_quat.round(6).tolist(),
        "ee_pos": obs.ee_pos.round(6).tolist(),
        "ee_force": obs.ee_force.round(4).tolist(),
    }


def heldout_probes(rng: np.random.Generator, n: int = 6) -> list[Probe]:
    """Random test probes, never chosen by the agent, used to measure prediction accuracy."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return [Probe(direction=v, distance=0.06, max_force=12.0) for v in d]


def _heldout_error(env, belief, probes, rng, n_samples=128) -> dict:
    """Posterior-predictive error on held-out probes, each run on a forked simulator."""
    obs = env.observe()
    hyps = belief.sample(n_samples, rng)
    y, _ = predict_for_probes(hyps, belief.h0, obs.handle_pos, obs.ee_pos, probes)  # (K, M, 6)
    pos_err = []
    for k, probe in enumerate(probes):
        twin = env.privileged_fork(rng)
        actual = twin.execute_probe(probe).after.handle_pos
        pos_err.append(float(np.linalg.norm(y[k, :, :3].mean(axis=0) - actual)))
    return {"heldout_pos_err_m": float(np.mean(pos_err))}


def _try_operation(env, belief, rng) -> dict:
    """Attempt the opening task on a forked simulator using the current belief."""
    spec = env.privileged_spec()
    twin = env.privileged_fork(rng)
    plan = plan_opening(belief, twin.observe().handle_pos)
    if spec.joint_type == "fixed":
        return {"task_success": plan is None, "declared_fixed": plan is None}
    if plan is None:
        return {"task_success": False, "declared_fixed": True}
    twin.follow_path(plan, max_force=15.0)
    kind = TYPE_NAMES.index("revolute" if spec.joint_type == "screw" else spec.joint_type)
    goal = TASK_GOAL[kind]
    q = twin.privileged_joint_position()
    return {"task_success": bool(q >= goal), "declared_fixed": False, "q_after_task": q}


def run_episode(
    spec: MechanismSpec,
    selector,
    budget: int,
    seed: int,
    n_particles: int = 600,
    evaluate: bool = True,
    on_step=None,
    env_hook=None,
    prior_kwargs: dict | None = None,
) -> dict:
    """Run ``budget`` probes chosen by ``selector`` and return a JSON-serialisable log.

    ``on_step(step_log, belief, info)`` is called after every belief update (e.g. for the demo),
    and ``env_hook(model, data)`` after every physics step of the real (non-forked) environment.
    """
    env_rng = np.random.default_rng([seed, 0])
    agent_rng = np.random.default_rng([seed, 1])
    eval_rng = np.random.default_rng([seed, 2])
    env = MechanismEnv(spec, env_rng)
    env.step_hook = env_hook
    geometry = env.geometry()
    env.move_ee_to(geometry.handle_pos)
    if not env.grasp():
        raise RuntimeError("could not grasp the handle")
    q_ref = env.observe().handle_quat
    prior = Prior(geometry, **(prior_kwargs or {}))
    belief = Belief(geometry, agent_rng, n_particles=n_particles, prior=prior)
    tests = heldout_probes(np.random.default_rng([seed, 3]))

    def evaluate_now() -> dict:
        if not evaluate:
            return {}
        out = score_belief(belief, spec)
        out.update(_heldout_error(env, belief, tests, eval_rng))
        out.update(_try_operation(env, belief, eval_rng))
        return out

    steps = [{"step": 0, "type_probs": belief.type_probs().tolist(), **evaluate_now()}]
    for t in range(1, budget + 1):
        now = env.observe()
        t0 = time.perf_counter()
        probe, info = selector.select(belief, now.handle_pos, now.ee_pos, agent_rng)
        select_s = time.perf_counter() - t0
        # Prediction made *before* acting, compared with what then happens.
        sampled = belief.sample(128, agent_rng)
        pred = predict_for_probes(sampled, belief.h0, now.handle_pos, now.ee_pos, [probe])[0][0]
        result = env.execute_probe(probe)
        rotvec = relative_rotvec(q_ref, result.after.handle_quat)
        record = make_record(
            result.before.handle_pos,
            result.before.ee_pos,
            probe.direction,
            probe.distance,
            probe.max_force,
            result.after.handle_pos,
            rotvec,
        )
        t0 = time.perf_counter()
        belief.update(record)
        update_s = time.perf_counter() - t0
        steps.append(
            {
                "step": t,
                "probe": {
                    "direction": probe.direction.round(4).tolist(),
                    "distance": probe.distance,
                    "max_force": probe.max_force,
                },
                "obs_before": _obs_dict(result.before),
                "obs_after": _obs_dict(result.after),
                "observed_displacement_m": float(
                    np.linalg.norm(result.after.handle_pos - result.before.handle_pos)
                ),
                "observed_rotation_rad": float(np.linalg.norm(rotvec)),
                "pre_action_pred_err_m": float(
                    np.linalg.norm(pred[:, :3].mean(axis=0) - result.after.handle_pos)
                ),
                "chosen_eig": info.get("eig"),
                "type_probs": belief.type_probs().tolist(),
                "belief": belief.summary(),
                "select_s": select_s,
                "update_s": update_s,
                **evaluate_now(),
            }
        )
        if on_step is not None:
            on_step(steps[-1], belief, {**info, "pre_action_prediction": pred})
    return {
        "family": spec.family,
        "joint_type": spec.joint_type,
        "strategy": selector.name,
        "seed": seed,
        "budget": budget,
        "truth": {
            "axis": spec.axis.tolist(),
            "origin": spec.origin.tolist(),
            "upper": spec.upper_limit,
            "friction": spec.friction,
        },
        "mjcf": env.mjcf,
        "steps": steps,
    }
