"""Scoped hand tools using DREAM's validation/result contract, with bounded cancellation.

No object perception or grasp verification is implemented here. Closing the hand reports
measured articulation, never object acquisition. Cartesian reaching and learned motor
policies remain unsupported until their embodiment contracts have been validated.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from dream.tools.toolbox import ToolResult, _validate
from dream.types import DexterousEmbodiment

SIDE = {"side": {"type": "string", "required": True}}
DEXTEROUS_SPECS = [
    {"name": name, "description": description, "parameters": SIDE}
    for name, description in (
        ("open_hand", "Open the named hand; this does not verify object release."),
        ("close_hand", "Close a fixed finger synergy; this does not verify a grasp."),
    )
] + [
    {
        "name": "get_robot_state",
        "description": "Named joint and root proprioception.",
        "parameters": {},
    }
]


class DexterousToolbox:
    """Same call/result boundary as Toolbox; explicit unsupported perception and goals."""

    specs = DEXTEROUS_SPECS
    system_prompt = (
        "You control a supported Unitree G1 with seven joints per hand in MuJoCo. "
        "Call one named tool per turn in <tool_call> JSON format. Tools report measured "
        "joint motion only. Object perception, grasp verification, walking and goal "
        "completion are unsupported. Never claim a grasp from finger positions. "
        "Use finish(success=false) to explain these limits."
    )

    def __init__(
        self,
        robot: DexterousEmbodiment,
        step: Callable[[], None],
        cancel: threading.Event | None = None,
    ) -> None:
        self.robot, self.step = robot, step
        self.cancel = cancel if cancel is not None else threading.Event()

    def held_object(self) -> None:
        return None  # unknown; verify_goal is always rejected

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        spec = next((s for s in self.specs if s["name"] == name), None)
        if spec is None:
            return ToolResult(name, "rejected", "unsupported dexterous capability")
        problem = _validate(spec["parameters"], args)
        if problem:
            return ToolResult(name, "rejected", problem)
        state = self.robot.state()
        if name == "get_robot_state":
            return ToolResult(
                name,
                "succeeded",
                data={
                    "joint_names": state.joint_names,
                    "joint_pos_rad": state.joint_pos.tolist(),
                    "joint_vel_rad_s": state.joint_vel.tolist(),
                    "root_position_m": state.root_position.tolist(),
                    "root_quaternion_wxyz": state.root_quaternion.tolist(),
                    "protective_stop": state.protective_stop,
                    "stop_reason": state.stop_reason,
                    "support": "fixed pelvis",
                    "object_grasp_verified": False,
                },
            )
        if self.cancel.is_set() or state.protective_stop:
            self.robot.hold()
            return ToolResult(name, "stopped", state.stop_reason or "cancelled")
        before = state.joint_pos.copy()
        try:
            self.robot.set_hand(args["side"], float(name == "close_hand"))
        except ValueError as error:
            return ToolResult(name, "rejected", str(error))
        # Finite 2 s commissioning budget, checked every physics tick.
        while self.robot.state().time - state.time < 2.0:
            if self.cancel.is_set() or self.robot.state().protective_stop:
                self.robot.hold()
                return ToolResult(
                    name,
                    "stopped",
                    "cancelled or protective stop",
                    sim_time=self.robot.state().time - state.time,
                )
            self.step()
        after = self.robot.state()
        if after.protective_stop:
            return ToolResult(name, "stopped", after.stop_reason, sim_time=after.time - state.time)
        ids = [i for i, n in enumerate(after.joint_names) if n.startswith(args["side"] + "_hand_")]
        moved = float(np.max(np.abs(after.joint_pos[ids] - before[ids])))
        return ToolResult(
            name,
            "succeeded" if moved > 0.01 else "failed",
            "measured articulation only; grasp unverified",
            {"max_motion_rad": moved, "object_grasp_verified": False},
            after.time - state.time,
        )
