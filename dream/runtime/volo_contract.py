"""Fail-closed, non-actuating boundary for evaluating NVIDIA VoLoAgent (NVlabs/VoLoAgent).

VoLo is a proxy between a robolab eval client and a VLA policy server. Its manipulation
"skills" (``grasp``, ``place``) are executed by its own state machines, which emit 8-D
joint-position chunks for a single Franka arm with a 1-D binary gripper. This module only
*describes and validates* VoLo's messages against DREAM types and wraps each VoLo skill as a
DREAM-style tool that reports ``not executable: <reason>`` or returns a dry-run plan.

There is deliberately no actuator path here: nothing in this file imports ``dream.robot``,
``dream.sim`` or VoLo, opens a socket, or produces a joint, gripper or velocity command.
Upstream facts and file/line citations are in ``docs/phase1/VOLOAGENT.md``; the message
examples used by the tests are recorded from the real upstream code in
``tests/fixtures/volo_upstream_messages.json`` (``experiments/record_volo_messages.py``).
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field

import numpy as np

from dream.tools.toolbox import ToolResult

UPSTREAM_COMMIT = "b4e6230"  # NVlabs/VoLoAgent "Release"; Apache-2.0

# --- what VoLo's released stack assumes (cited in docs/phase1/VOLOAGENT.md) -------------------
VOLO_ARM_DOF = 7  # single Franka Panda arm (grasp/ik.py)
VOLO_ACTION_DIM = 8  # 7 joint-position targets (rad) + 1 gripper in [0, 1], 0 open / 1 closed
VOLO_ACTION_HORIZON = 8  # rows per chunk (grasp/tool.py ACTION_HORIZON)
VOLO_CONTROL_HZ = 20.0  # protocols/stub.py DEFAULT_METADATA["control_frequency_hz"]
VOLO_IMAGE_HW = (224, 224)  # robolab wire images, RGB uint8
GR00T_IMAGE_HW = (180, 320)  # protocols/gr00t_zmq.py GR00T_RESOLUTION

# The embodiment DREAM targets (Unitree G1 with Dex3 hands): two 7-DoF arms, 7-DoF hands.
# Used only to *report* the mismatch; nothing here is sent to a robot.
G1_DEX3_ARM_DOF = 7
G1_DEX3_ARMS = 2
G1_DEX3_HAND_DOF = 7

# Phases of the upstream skill state machines (grasp/tool.py GraspPhase, place/tool.py
# PlacePhase). Used to build dry-run plans; a test cross-checks them against upstream when it
# is importable.
GRASP_PHASES = (
    "releasing",
    "settling",
    "lifting",
    "perceiving",
    "planning",
    "approaching",
    "final_approach",
    "closing",
    "retreating",
    "settle_hold",
    "integral_settle",
    "measuring",
    "done",
)
PLACE_PHASES = (
    "ensure_closed",
    "lifting",
    "perceiving",
    "planning",
    "approaching",
    "final_approach",
    "measuring",
    "releasing",
    "settling",
    "retreating",
    "done",
)
VOLO_FAILURE_REASONS = ("perception_no_target", "ik_unreachable", "motion_blocked")

# External services a VoLo skill needs. None of them is part of the released repository.
GRASP_SERVICES = (
    "grasp server :8003 (GraspGen + SAM3/SAM2/GroundingDINO; GPU)",
    "cuRobo motion planning on the grasp server (default planner)",
    "orchestrator VLM endpoint (OpenAI-compatible)",
    "eval client providing RGB-D, camera K/extrinsic, EE quaternion (robolab/Isaac Sim)",
)
PLACE_SERVICES = GRASP_SERVICES + ("Molmo2 pointing sidecar :8122 (default place-seg-mode)",)

SUBGOAL_ACTIONS = ("advance", "continue", "replan", "abort")
VOLO_TOOLS = ("grasp", "place", "noop")


@dataclass(frozen=True)
class VoLoCompatibility:
    supported: bool
    reason: str
    accepted_fields: tuple[str, ...] = ()


def inspect_volo_contract(payload: object) -> VoLoCompatibility:
    """Report compatibility of a passive report; any executable action is rejected."""
    if not isinstance(payload, dict):
        return VoLoCompatibility(False, "expected a JSON object")
    forbidden = {"action", "actions", "joint_positions", "joint_targets", "robot_action"}
    if forbidden.intersection(payload):
        return VoLoCompatibility(False, "actuation-bearing payloads are unsupported")
    allowed = {"status", "completed", "total", "info", "score", "all_status_codes"}
    unknown = set(payload) - allowed
    if unknown:
        return VoLoCompatibility(False, "unknown fields: " + ", ".join(sorted(unknown)))
    return VoLoCompatibility(True, "passive status only", tuple(sorted(payload)))


def load_upstream():
    """Return the upstream ``vlm_orchestrator`` package, or ``None`` if it is not installed.

    Optional and lazy: DREAM's CPU tests and runtime never require it.
    """
    try:
        return importlib.import_module("vlm_orchestrator")
    except ImportError:
        return None


# --- tool_chain decision messages (VLM -> orchestrator) ---------------------------------------
class VoLoContractError(ValueError):
    """A message does not satisfy VoLo's documented contract."""


@dataclass(frozen=True)
class VoLoDecision:
    subgoal_action: str
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def stack(self) -> bool | None:
        value = self.args.get("stack")
        return value if isinstance(value, bool) else None


def parse_volo_decision(message: str | dict) -> VoLoDecision:
    """Validate a tool_chain decision with the same rules as upstream.

    Mirrors ``strategies/tool_chain.py:ToolChainDecision`` and
    ``parse_tool_chain_response`` (required keys, allowed values, per-tool required args and
    the rule that ``advance``/``replan``/``abort`` pair with ``noop`` while ``continue``
    pairs with a tool). Two rules are stricter than upstream, which only logs or ignores them:
    ``continue`` with ``noop`` and a non-boolean ``stack``. Raises :class:`VoLoContractError`.
    """
    if isinstance(message, str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            raise VoLoContractError(f"not a JSON object: {exc.msg}") from exc
    else:
        data = message
    if not isinstance(data, dict):
        raise VoLoContractError("decision must be a JSON object")
    action, tool = data.get("subgoal_action"), data.get("tool")
    if action is None or tool is None:
        raise VoLoContractError("missing subgoal_action or tool")
    if action not in SUBGOAL_ACTIONS:
        raise VoLoContractError(f"invalid subgoal_action {action!r}")
    if tool not in VOLO_TOOLS:
        raise VoLoContractError(f"invalid tool {tool!r}; expected one of {VOLO_TOOLS}")
    args = data.get("args") or {}
    if not isinstance(args, dict):
        raise VoLoContractError("args must be an object")
    required = {"grasp": "target", "place": "destination"}.get(tool)
    if required is not None:
        value = args.get(required)
        if not isinstance(value, str) or not value.strip():
            raise VoLoContractError(f"{tool} requires args.{required} (non-empty string)")
    if "stack" in args and not isinstance(args["stack"], bool):
        raise VoLoContractError("args.stack must be a boolean")
    if action != "continue" and tool != "noop":
        raise VoLoContractError(f"subgoal_action={action!r} must be paired with tool='noop'")
    if action == "continue" and tool == "noop":
        raise VoLoContractError("subgoal_action='continue' needs tool grasp or place")
    return VoLoDecision(action, tool, dict(args), str(data.get("reason", "")).strip())


# --- mapping to DREAM tools --------------------------------------------------------------------
@dataclass(frozen=True)
class DreamToolCall:
    """A candidate call of an existing DREAM Toolbox tool (see ``dream/tools/toolbox.py``)."""

    name: str
    args: dict
    unresolved: tuple[str, ...] = ()  # VoLo arguments DREAM cannot ground without perception


_DESTINATION_FILLER = {"in", "on", "the", "a", "an", "inside", "onto", "empty", "space", "of"}
_DREAM_PLACE_TARGETS = ("table", "tray")


def to_dream_call(decision: VoLoDecision) -> DreamToolCall | None:
    """Map a VoLo decision to a candidate DREAM tool call; ``None`` for pointer moves."""
    if decision.tool == "noop":
        return None
    if decision.tool == "grasp":
        return DreamToolCall("grasp", {"object": decision.args["target"]})
    destination = decision.args["destination"]
    words = [
        w for w in destination.lower().replace(",", " ").split() if w not in _DESTINATION_FILLER
    ]
    target = words[0] if words in ([t] for t in _DREAM_PLACE_TARGETS) else None
    hint = decision.args.get("held_object_hint")
    unresolved = []
    if target is None:
        unresolved.append("destination")
    if not hint:
        unresolved.append("held_object_hint")
    args = {"object": hint or "", "target": target or destination}
    return DreamToolCall("place", args, tuple(unresolved))


# --- observation / action wire schema ---------------------------------------------------------
def validate_wire_observation(obs: dict, *, need_grasp_keys: bool = False) -> list[str]:
    """Return problems with a robolab-style observation frame (empty list = valid).

    Schema: docs/instructions/eval-client-contract.md section 3 in the upstream clone.
    """
    problems: list[str] = []

    def arr(key, shape, dtype_kind):
        value = obs.get(key)
        if not isinstance(value, np.ndarray):
            problems.append(f"{key}: missing")
        elif tuple(value.shape) != shape:
            problems.append(f"{key}: shape {tuple(value.shape)} != {shape}")
        elif value.dtype.kind != dtype_kind:
            problems.append(f"{key}: dtype {value.dtype} not kind {dtype_kind!r}")

    if not isinstance(obs.get("prompt"), str):
        problems.append("prompt: missing")
    for cam in ("exterior_image_1_left", "wrist_image_left"):
        arr(f"observation/{cam}", (*VOLO_IMAGE_HW, 3), "u")
    arr("observation/joint_position", (VOLO_ARM_DOF,), "f")
    arr("observation/gripper_position", (1,), "f")
    if need_grasp_keys:
        depth = obs.get("observation/depth_exterior_image_1_left")
        if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype != np.float32:
            problems.append("observation/depth_exterior_image_1_left: need float32 (H, W) metres")
        arr("observation/camera_K", (9,), "f")
        arr("observation/camera_extrinsic", (16,), "f")
        arr("observation/ee_quat", (4,), "f")
    return problems


def validate_wire_actions(actions: object) -> list[str]:
    """Return problems with a VoLo ``actions`` chunk (``(horizon, 8)``: 7 joints + gripper)."""
    if not isinstance(actions, np.ndarray) or actions.ndim != 2:
        return ["actions: expected a 2-D array"]
    problems = []
    if actions.shape[1] != VOLO_ACTION_DIM:
        problems.append(f"actions: width {actions.shape[1]} != {VOLO_ACTION_DIM}")
    elif actions[:, 7].min() < 0.0 or actions[:, 7].max() > 1.0:
        problems.append("actions: gripper column outside [0, 1]")
    if not np.isfinite(actions).all():
        problems.append("actions: non-finite values")
    return problems


def embodiment_mismatches(action_dim: int = VOLO_ACTION_DIM) -> list[str]:
    """Why a VoLo action chunk cannot drive a Unitree G1 with Dex3 hands (reporting only)."""
    reasons = []
    g1_dim = G1_DEX3_ARMS * (G1_DEX3_ARM_DOF + G1_DEX3_HAND_DOF)
    if action_dim != g1_dim:
        reasons.append(
            f"VoLo emits {action_dim} dims (1 arm + binary gripper); G1+Dex3 needs {g1_dim} "
            f"for two arms and hands, plus a whole-body controller for legs and torso"
        )
    reasons.append(
        "joint IK/FK is Franka-only (grasp/ik.py); grasp poses assume a Panda/Robotiq gripper"
    )
    reasons.append("no base, locomotion or torso interface exists in the VoLo action message")
    return reasons


# --- skill wrapper (non-actuating) --------------------------------------------------------------
@dataclass(frozen=True)
class VoLoSkill:
    name: str
    description: str
    parameters: dict
    preconditions: tuple[str, ...]
    outputs: tuple[str, ...]
    failure_modes: tuple[str, ...]
    phases: tuple[str, ...]
    services: tuple[str, ...]
    dream_tool: str


VOLO_SKILLS: dict[str, VoLoSkill] = {
    "grasp": VoLoSkill(
        name="grasp",
        description="Pick up a named object using GraspGen + SAM3 perception + cuRobo planning.",
        parameters={
            "target": {"type": "string", "required": True},
            "stack": {"type": "boolean", "required": False},
        },
        preconditions=(
            "RGB-D frame with camera intrinsics, camera-to-world extrinsic and EE quaternion",
            "target is a noun phrase a detector can localise",
            "arm is a Franka Panda; gripper is binary",
        ),
        outputs=("8-D joint-position action chunks", "grasp debug images and grasp_log.json"),
        failure_modes=(
            "no depth in observation",
            "target mask empty or no point-cloud overlap",
            "no grasp candidate (HTTP 422)",
            "IK unreachable",
            "planner blocked (cuRobo 503/none)",
            "grasp success is not verified by the tool; the VLM inspects the next image",
        ),
        phases=GRASP_PHASES,
        services=GRASP_SERVICES,
        dream_tool="grasp",
    ),
    "place": VoLoSkill(
        name="place",
        description="Place the held object at a free-form destination phrase.",
        parameters={
            "destination": {"type": "string", "required": True},
            "held_object_hint": {"type": "string", "required": False},
            "stack": {"type": "boolean", "required": False},
        },
        preconditions=(
            "an object is already held (the grasp tool closed on it)",
            "RGB-D frame with intrinsics, extrinsic, EE quaternion",
            "destination phrase follows the in / empty space / on rules",
        ),
        outputs=("8-D joint-position action chunks", "place debug images and place_log.json"),
        failure_modes=VOLO_FAILURE_REASONS + ("held object slips (not detected by the tool)",),
        phases=PLACE_PHASES,
        services=PLACE_SERVICES,
        dream_tool="place",
    ),
}


def dry_run_plan(decision: VoLoDecision) -> dict:
    """Describe what VoLo would do for ``decision``. Pure data; executes nothing."""
    skill = VOLO_SKILLS.get(decision.tool)
    call = to_dream_call(decision)
    return {
        "executable": False,
        "subgoal_action": decision.subgoal_action,
        "volo_tool": decision.tool,
        "volo_args": dict(decision.args),
        "volo_phases": list(skill.phases) if skill else [],
        "requires_services": list(skill.services) if skill else [],
        "dream_candidate": None
        if call is None
        else {
            "tool": call.name,
            "args": call.args,
            "unresolved": list(call.unresolved),
        },
        "embodiment_mismatches": embodiment_mismatches() if skill else [],
    }


class VoLoSkillTool:
    """DREAM-style tool that validates a VoLo skill call and refuses to execute it.

    ``call`` returns a :class:`ToolResult` with ``status='rejected'`` and the message
    ``not executable: <reason>``; ``data['dry_run']`` carries the plan when the arguments are
    valid. Invalid arguments are also ``rejected``, with the validation error as the reason.
    """

    def __init__(
        self,
        reason: str = (
            "VoLo skills need an external grasp server, cuRobo, a Franka arm and a "
            "joint-position interface; DREAM drives a G1 under a whole-body controller"
        ),
    ):
        self.reason = reason

    def call(self, name: str, **args) -> ToolResult:
        if name not in VOLO_SKILLS:
            return ToolResult(name, "rejected", f"not executable: unknown VoLo skill {name!r}")
        try:
            decision = parse_volo_decision(
                {"subgoal_action": "continue", "tool": name, "args": args, "reason": "dry run"}
            )
        except VoLoContractError as exc:
            return ToolResult(name, "rejected", f"not executable: invalid arguments ({exc})")
        return ToolResult(
            name,
            "rejected",
            f"not executable: {self.reason}",
            data={"dry_run": dry_run_plan(decision), "executable": False},
        )

    def specs(self) -> list[dict]:
        """JSON-schema style specs shaped like ``dream.tools.toolbox.TOOL_SPECS`` entries."""
        return [
            {"name": f"volo_{s.name}", "description": s.description, "parameters": s.parameters}
            for s in VOLO_SKILLS.values()
        ]
