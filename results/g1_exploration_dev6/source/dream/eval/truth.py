"""Ground-truth conversions used for scoring and model validation. Never import from dream.agent."""

from __future__ import annotations

import numpy as np

from dream.agent.model import FIXED, PRISMATIC, REVOLUTE, Hypotheses
from dream.sim.mechanisms import MechanismSpec

_KIND = {"fixed": FIXED, "prismatic": PRISMATIC, "revolute": REVOLUTE}


def spec_to_hypothesis(spec: MechanismSpec) -> Hypotheses:
    """The true mechanism expressed as a single hypothesis, for validating the agent's model."""
    return Hypotheses(
        kind=np.array([_KIND[spec.joint_type]]),
        axis=spec.axis[None].astype(float) if spec.joint_type != "fixed" else np.zeros((1, 3)),
        origin=spec.origin[None].astype(float),
        upper=np.array([spec.upper_limit]),
        friction=np.array([spec.friction]),
    )


def compartment_to_hypothesis(spec) -> Hypotheses:
    """The true compartment mechanism as one hypothesis, for validating the Panda probe model."""
    kind = {"drawer": PRISMATIC, "lid": REVOLUTE, "fixed": FIXED}[spec.kind]
    return Hypotheses(
        kind=np.array([kind]),
        axis=spec.joint_axis_world()[None].astype(float),
        origin=spec.to_world(spec.hinge_local)[None],
        upper=np.array([spec.upper]),
        friction=np.array([spec.friction]),
    )
