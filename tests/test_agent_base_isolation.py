"""The base object handed to the agent exposes the MobileBase protocol and nothing else."""

import numpy as np
import pytest

from experiments.semantic_search_eval import AGENT_BASE_PROTOCOL, AgentBase, PathMeter

PROTOCOL = {"capture", "odometry", "stop_reason", "time", "walk"}
PRIVILEGED = ("true_pose", "home", "model", "data", "inner", "base", "sim", "env", "_inner",
              "_true_pose", "contact_log", "__wrapped__")  # fmt: skip


class FakeSimBase:
    """Stands in for the simulator base, with the privileged members a real one has."""

    home, model, data = object(), object(), object()
    time, stop_reason = 1.5, None

    def __init__(self):
        self.x = 0.0

    def true_pose(self):
        return np.array([self.x, 0.0, 0.0])

    def contact_log(self):
        return []

    def odometry(self):
        return np.array([self.x, 0.0, 0.0])

    def capture(self):
        return "frame"

    def walk(self, forward, lateral, turn, seconds):
        self.x += forward * seconds
        return True


def handed_to_agent():
    sim = FakeSimBase()
    meter = PathMeter(sim, sim.true_pose)
    return meter, meter.agent_base


def public(obj):
    return {n for n in dir(obj) if not n.startswith("_")}


def test_agent_object_exposes_exactly_the_protocol():
    _, agent = handed_to_agent()
    assert isinstance(agent, AgentBase)
    assert public(agent) == PROTOCOL == set(AGENT_BASE_PROTOCOL)
    assert not hasattr(type(agent), "__getattr__")
    assert not hasattr(agent, "__dict__")


@pytest.mark.parametrize("name", PRIVILEGED)
def test_privileged_names_raise_attribute_error(name):
    _, agent = handed_to_agent()
    with pytest.raises(AttributeError):
        getattr(agent, name)
    with pytest.raises(AttributeError):
        setattr(agent, name, 1)


def test_protocol_members_work_and_the_meter_still_measures():
    meter, agent = handed_to_agent()
    assert agent.time == 1.5 and agent.stop_reason is None
    agent.capture()
    assert agent.walk(1.0, 0, 0, 2.0)
    assert agent.odometry()[0] == 2.0
    assert meter.frames == 1 and meter.path_m == pytest.approx(2.0)
