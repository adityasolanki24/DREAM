"""Typed tools for the single-target tabletop humanoid grasp."""

from __future__ import annotations

import threading
from collections.abc import Callable

from dream.tools.toolbox import ToolResult, _validate
from dream.types import Humanoid, RGBDFrame

HUMANOID_SPECS = [
    {
        "name": "get_robot_state",
        "description": "Observe humanoid proprioception.",
        "parameters": {},
    },
    {
        "name": "grasp_tabletop",
        "description": "Approach and grasp the tabletop object and check that it stays lifted.",
        "parameters": {},
    },
]


class HumanoidToolbox:
    """Dispatch a bounded grasp on a live humanoid and report its sensed outcome."""

    specs = HUMANOID_SPECS

    def __init__(
        self,
        robot: Humanoid,
        grasp_tabletop: Callable[[], bool],
        cancel: threading.Event,
    ) -> None:
        self.robot, self.grasp_tabletop, self.cancel = robot, grasp_tabletop, cancel
        self._attempted = False

    def capture(self) -> RGBDFrame:
        return self.robot.capture()

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        spec = next((spec for spec in self.specs if spec["name"] == name), None)
        if spec is None:
            return ToolResult(name, "rejected", "unsupported humanoid tool")
        problem = _validate(spec["parameters"], dict(args or {}))
        if problem:
            return ToolResult(name, "rejected", problem)
        before = self.robot.state()
        if name == "get_robot_state":
            return ToolResult(
                name,
                "succeeded",
                data={
                    "joint_names": before.joint_names,
                    "joint_pos_rad": before.joint_pos.tolist(),
                    "joint_vel_rad_s": before.joint_vel.tolist(),
                    "root_position_m": before.root_position.tolist(),
                    "root_quaternion_wxyz": before.root_quaternion.tolist(),
                    "finger_forces_n": {
                        side: self.robot.finger_forces(side) for side in ("left", "right")
                    },
                    "protective_stop": before.protective_stop,
                    "stop_reason": before.stop_reason,
                },
            )
        if before.protective_stop or self.cancel.is_set():
            self.robot.hold("cancelled")
            return ToolResult(name, "stopped", before.stop_reason or "cancelled")
        if self._attempted:
            return ToolResult(name, "rejected", "tabletop grasp already attempted in this episode")
        self._attempted = True
        observed = self.grasp_tabletop()
        after = self.robot.state()
        elapsed = after.time - before.time
        if after.protective_stop or self.cancel.is_set():
            self.robot.hold("cancelled")
            return ToolResult(name, "stopped", after.stop_reason or "cancelled", sim_time=elapsed)
        return ToolResult(
            name,
            "succeeded" if observed else "failed",
            "sensed sustained lift" if observed else "sustained lift was not observed",
            {"grasp_observed": observed},
            elapsed,
        )
