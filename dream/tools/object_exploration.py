"""Sensor-only supported-hand exploration tools over DREAM's existing tool boundary.

Scoped assumptions: one red sphere, a blue horizontal target, a supported G1 left hand.
No simulator/evaluation imports, object ground truth or generated metric coordinates.
Movability means response under the available bounded lift, not universal immobility.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from dream.agent.movability import MovabilityBelief
from dream.perception.sphere import SphereEstimate, SpherePerception
from dream.tools.dexterous import DEXTEROUS_SPECS, DexterousToolbox
from dream.tools.goals import InRegion, parse_goal
from dream.tools.toolbox import ToolResult, _validate
from dream.types import DexterousEmbodiment

OBJECT = {"object": {"type": "string", "required": True}}
EXPLORATION_SPECS = [
    *[s for s in DEXTEROUS_SPECS if s["name"] == "get_robot_state"],
    *[
        {"name": name, "description": description, "parameters": parameters}
        for name, description, parameters in (
            ("observe_scene", "Fresh RGB-D sphere/target estimates and current belief.", {}),
            (
                "inspect_object",
                "Re-observe uncertainty; preserve an existing grip during reacquisition.",
                OBJECT,
            ),
            ("reach_object", "Reach the sphere measured by RGB-D; no coordinate argument.", OBJECT),
            (
                "probe_object",
                "Select an information-gathering bounded lift and update movability belief.",
                OBJECT,
            ),
            (
                "grasp_object",
                "Attempt a tactile-gated grip and verify a further lift from vision.",
                OBJECT,
            ),
            (
                "move_object",
                "Carry the visually verified held object above the observed blue target.",
                {**OBJECT, "target": {"type": "string", "required": True}},
            ),
            ("release_object", "Lower to the observed target if holding, release and retract.", {}),
            (
                "verify_goal",
                "Verify a caller-bound region goal from fresh vision after release.",
                {"goal": {"type": "goal", "required": True}},
            ),
        )
    ],
]


class ActionStopped(RuntimeError):
    pass


class ExplorationToolbox(DexterousToolbox):
    specs = EXPLORATION_SPECS
    system_prompt = (
        "You control a fixed-pelvis G1 through named exploration tools. The red sphere's "
        "position and blue target come only from RGB-D. It may resist movement. Inspect "
        "uncertainty, reach, probe, then use observed displacement and belief to decide "
        "whether to grasp and transfer. Re-observe after failures; retries are bounded. "
        "Never infer grasp from finger closure. Never invent coordinates or claim walking. "
        "Call exactly one tool per turn in <tool_call> JSON format. Success requires "
        "verify_goal and a caller-bound in red target goal after release."
    )

    def __init__(
        self,
        robot: DexterousEmbodiment,
        step: Callable[[], None],
        perception: SpherePerception,
        strategy: str = "eig",
        cancel: threading.Event | None = None,
    ) -> None:
        super().__init__(robot, step, cancel)
        if strategy not in ("eig", "fixed_short"):
            raise ValueError("unsupported exploration strategy")
        self.perception, self.strategy = perception, strategy
        self.belief = MovabilityBelief()
        self.orientation = robot.grasp_pose("left")[1]
        self.closure = 0.0
        self.grasp_offset = np.zeros(3)  # acquired from vision/contact in this session
        self.holding: str | None = None
        self.released, self.at_target = False, False
        self.probes = 0
        self.events: list[dict] = []
        self.action = "initialization"

    def held_object(self) -> str | None:
        return self.holding

    def hold_for_reasoning(self) -> None:
        self.robot.hold_for_reasoning()

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        own = {s["name"]: s for s in self.specs if s not in DEXTEROUS_SPECS}
        if name not in own:
            result = super().call(name, args)
            if name == "open_hand" and result.ok:
                self.closure, self.holding = 0, None
            return result
        problem = _validate(own[name]["parameters"], args)
        if problem:
            return ToolResult(name, "rejected", problem)
        if args.get("object", "red") != "red" or args.get("target", "target") != "target":
            return ToolResult(
                name, "rejected", "only the observed red sphere and blue target are supported"
            )
        if name not in ("observe_scene", "verify_goal") and (
            self.cancel.is_set() or self.robot.state().protective_stop
        ):
            return ToolResult(name, "stopped", "cancelled or protective stop")
        self.action = name
        start = self.robot.state().time
        holding_before = self.holding
        try:
            result = getattr(self, "_" + name)(**args)
        except (ValueError, ActionStopped) as error:
            self.robot.hold()
            result = ToolResult(
                name, "stopped" if isinstance(error, ActionStopped) else "failed", str(error)
            )
        result.sim_time = self.robot.state().time - start
        self.events.append(
            {
                "tool": name,
                "arguments": repr(args),
                "status": result.status,
                "message": result.message,
                "data": result.data,
                "time_s": self.robot.state().time,
                "start_time_s": start,
                "holding_before": holding_before,
                "holding_after": self.holding,
            }
        )
        return result

    def _advance(self, duration: float) -> None:
        end = self.robot.state().time + duration
        while self.robot.state().time < end - 1e-9:
            if self.cancel.is_set() or self.robot.state().protective_stop:
                raise ActionStopped("cancelled or protective stop")
            # Additional fixed commissioning task budget; motor/joint guards underneath
            # remain unchanged. It is checked at every 1 ms step, never model-overridable.
            if sum(self.robot.finger_forces("left").values()) > 12.0:
                raise ActionStopped("12 N tactile load budget exceeded")
            self.step()
            if sum(self.robot.finger_forces("left").values()) > 12.0:
                raise ActionStopped("12 N tactile load budget exceeded")

    def _object(self) -> SphereEstimate:
        observed = self.perception.observe()
        if observed is None:
            raise ValueError("red sphere not localized; inspect before another attempt")
        if np.sqrt(np.linalg.eigvalsh(observed.covariance).max()) > 0.006:
            raise ValueError("position uncertainty exceeds supported reach precision")
        return observed

    def _move(self, position: np.ndarray, timeout: float = 6.0) -> float:
        self.robot.reach("left", np.asarray(position), self.orientation)
        end = self.robot.state().time + timeout
        while self.robot.state().time < end:
            error = float(np.linalg.norm(self.robot.grasp_pose("left")[0] - position))
            if error < 0.006:  # 3x systematic sensor floor; object success verified separately
                self._advance(0.1)
                return error
            self._advance(0.01)
        raise ValueError("bounded Cartesian reach timed out")

    def _open(self) -> None:
        self.robot.set_hand("left", 0)
        self.closure = 0
        self._advance(1.5)

    def _close(self) -> bool:
        # Stop increasing closure once two tactile finger chains support a light load;
        # avoid crushing to a fixed closure setpoint. This is a classical heuristic.
        for _ in range(45):
            forces = self.robot.finger_forces("left")
            if sum(f > 0.1 for f in forces.values()) >= 2 and sum(forces.values()) >= 1.5:
                self._advance(0.15)
                return True
            self.closure = min(1, self.closure + 0.025)
            self.robot.set_hand("left", self.closure)
            self._advance(0.05)
        return False

    def _observe_scene(self) -> ToolResult:
        obj = self.perception.observe()
        target = self.perception.target
        grasp_observation = "unobserved" if obj is None else "no verified grasp"
        if self.holding is not None and obj is not None:
            tracks = (
                np.linalg.norm(obj.position - self.robot.grasp_pose("left")[0] - self.grasp_offset)
                < 0.018
            )  # same measured tracking gate as grasp/move verification
            grasp_observation = "tracks hand" if tracks else "object no longer tracks hand"
            if not tracks:
                self.holding, self.at_target = None, False
        reachable = (
            obj is not None
            and np.linalg.norm(obj.position - self.robot.grasp_pose("left")[0]) < 0.008
        )
        return ToolResult(
            "observe_scene",
            "succeeded",
            data={
                "objects": [] if obj is None else [{"name": "red", **obj.to_dict()}],
                "fixtures": {} if target is None else {"target": (target[0].tolist(), target[1])},
                "holding": self.holding,
                "grasp_observation": grasp_observation,
                "at_target": self.at_target,
                "released": self.released,
                "hand_closed": self.closure > 0.01,
                "reached": bool(reachable),
                "in_contact": sum(f > 0.1 for f in self.robot.finger_forces("left").values()) >= 2,
                "belief": {
                    "p_movable": self.belief.p_movable,
                    "probes": self.probes,
                    "model_inadequate": self.belief.inadequate,
                },
                "unobserved_properties": ["mass", "attachment_type"],
            },
        )

    def _inspect_object(self, object: str) -> ToolResult:
        observed = self.perception.observe()
        if observed is not None:
            return ToolResult(
                "inspect_object", "succeeded", "visible without moving", observed.to_dict()
            )
        contact = sum(f > 0.1 for f in self.robot.finger_forces("left").values()) >= 2
        if self.holding is None and not contact:
            self._open()
        # When contact/holding persists, move the existing grasp together, never
        # open a potentially held object just because the camera lost sight of it.
        self._move(self.robot.grasp_pose("left")[0] + [0, 0, 0.035])
        obj = self._object()
        return ToolResult("inspect_object", "succeeded", data=obj.to_dict())

    def _reach_object(self, object: str) -> ToolResult:
        obj = self._object()
        initial = float(np.linalg.norm(obj.position - self.robot.grasp_pose("left")[0]))
        error = self._move(obj.position)
        after = self._object()
        return ToolResult(
            "reach_object",
            "succeeded",
            data={
                "sensor_target_m": obj.position.tolist(),
                "initial_error_m": initial,
                "final_command_tracking_error_m": error,
                "after": after.to_dict(),
            },
        )

    def _probe_object(self, object: str) -> ToolResult:
        if self.probes >= 2:
            return ToolResult("probe_object", "rejected", "two-probe budget exhausted")
        obj = self._object()
        if (
            np.linalg.norm(obj.position - self.robot.grasp_pose("left")[0] - self.grasp_offset)
            > 0.01
        ):
            return ToolResult("probe_object", "rejected", "reach the observed object first")
        valid = self._close()
        before = self._object()
        hand_before = self.robot.grasp_pose("left")[0].copy()
        # 4 mm contact/model spread is a declared development assumption; sensor
        # covariance is added rather than replacing it with a gantry likelihood.
        sigma = float(np.sqrt(2 * before.covariance[2, 2] + 0.004**2))
        selection = self.belief.select(sigma, self.strategy)
        self.probes += 1
        goal = hand_before + [0, 0, selection["distance_m"]]
        self.robot.reach("left", goal, self.orientation)
        # A probe observes the response over a bounded interval. Failure to track is
        # expected for a constrained object; demanding convergence discards evidence.
        self._advance(1.5)
        error = float(np.linalg.norm(self.robot.grasp_pose("left")[0] - goal))
        after = self._object()
        hand_after = self.robot.grasp_pose("left")[0]
        forces = self.robot.finger_forces("left")
        valid &= sum(f > 0.1 for f in forces.values()) >= 2
        sigma = float(np.sqrt(before.covariance[2, 2] + after.covariance[2, 2] + 0.004**2))
        measured = float(after.position[2] - before.position[2])
        expected = selection["distance_m"]
        update = self.belief.update(measured, expected, sigma, valid)
        if valid and measured > 2 * sigma:
            self.grasp_offset = after.position - self.robot.grasp_pose("left")[0]
        return ToolResult(
            "probe_object",
            "succeeded",
            "bounded-lift response only",
            {
                "selection": selection,
                "evidence": update,
                "tracking_error_m": error,
                "finger_forces_n": forces,
                "measured_hand_displacement_m": float(hand_after[2] - hand_before[2]),
                "before": before.to_dict(),
                "after": after.to_dict(),
            },
        )

    def _grasp_object(self, object: str) -> ToolResult:
        if self.belief.p_movable < 0.95 or self.belief.inadequate:
            return ToolResult(
                "grasp_object", "rejected", "insufficient bounded-movability evidence"
            )
        before = self._object()
        contact = self._close()
        point = self.robot.grasp_pose("left")[0] + [0, 0, 0.035]
        self._move(point)
        self._advance(0.3)
        after = self._object()
        lift = float(after.position[2] - before.position[2])
        matches = (
            np.linalg.norm(after.position - self.robot.grasp_pose("left")[0] - self.grasp_offset)
            < 0.018
        )
        ok = contact and lift > 0.02 and matches
        self.holding = "red" if ok else None
        if ok:
            self.at_target, self.released = False, False
            self.grasp_offset = after.position - self.robot.grasp_pose("left")[0]
        return ToolResult(
            "grasp_object",
            "succeeded" if ok else "failed",
            "vision lift and tactile evidence" if ok else "grasp lift not verified",
            {"measured_lift_m": lift, "contact": contact, "tracks_hand": bool(matches)},
        )

    def _move_object(self, object: str, target: str) -> ToolResult:
        obj = self._object()
        if self.holding != "red" or self.perception.target is None:
            return ToolResult(
                "move_object", "rejected", "verified grasp and visible target required"
            )
        centre, _ = self.perception.target
        goal = centre + [0, 0, obj.radius + 0.035] - self.grasp_offset
        error = self._move(goal)
        after = self._object()
        self.at_target = bool(
            np.linalg.norm(after.position - self.robot.grasp_pose("left")[0] - self.grasp_offset)
            < 0.018
        )
        if not self.at_target:
            self.holding = None
        return ToolResult(
            "move_object",
            "succeeded" if self.at_target else "failed",
            data={"target_from_rgbd_m": goal.tolist(), "tracking_error_m": error},
        )

    def _release_object(self) -> ToolResult:
        if self.holding is not None:
            obj = self._object()
            if self.perception.target is None:
                raise ValueError("target no longer visible; refusing unchecked placement")
            # Upstream finger bases are 28.5 mm off centre; mesh thickness plus
            # 6 mm sensor allowance gives ~50 mm swept opening clearance.
            height = max(obj.radius + 0.01, 0.05)
            self._move(self.perception.target[0] + [0, 0, height] - self.grasp_offset)
            # Opening fully beside a sphere sweeps it sideways. Unload tactile
            # contact first, then withdraw the side grasp before the full sweep.
            for _ in range(80):
                self.closure = max(0, self.closure - 0.01)
                self.robot.set_hand("left", self.closure)
                self._advance(0.03)
                if sum(self.robot.finger_forces("left").values()) < 0.2:
                    break
            else:
                raise ValueError("contact did not unload during bounded release")
            self._advance(0.15)
            # Lift clear before lateral withdrawal: the horizontal DLS path can
            # initially dip under load and drag a supported object along the pad.
            self._move(self.robot.grasp_pose("left")[0] + [0, 0, 0.065])
        self._open()
        self.holding, self.released = None, True
        self._move(self.robot.grasp_pose("left")[0] + [0, 0, 0.06])
        self._advance(0.5)
        return ToolResult(
            "release_object", "succeeded", "release commanded; goal still needs vision verification"
        )

    def _verify_goal(self, goal) -> ToolResult:
        if not isinstance(goal, InRegion):
            goal = parse_goal(goal)
        if goal != InRegion("red", "target"):
            return ToolResult("verify_goal", "rejected", "unsupported caller-bound goal")
        obj = self.perception.observe()
        target = self.perception.target
        satisfied = False
        margin = None
        if obj is not None and target is not None:
            centre, half = target
            allowance = 3 * np.sqrt(np.linalg.eigvalsh(obj.covariance).max() + 0.002**2)
            xy = half - obj.radius - np.max(np.abs(obj.position[:2] - centre[:2])) - allowance
            vertical = 0.012 - abs(obj.position[2] - obj.radius - centre[2])
            margin = float(min(xy, vertical))
            satisfied = margin > 0 and self.released and self.holding is None
        return ToolResult(
            "verify_goal",
            "succeeded",
            "RGB-D placement verification",
            {
                "satisfied": bool(satisfied),
                "margin_m": margin,
                "uncertain": obj is None or target is None,
            },
        )
