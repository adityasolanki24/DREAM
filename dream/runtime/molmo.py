"""Auditable MolmoAct2-DROID observation and decoding. No direct physical dispatch.

Source contracts and unresolved Panda/DROID differences are in the README. A prediction
is not permission to execute: camera and hand transfer require physical evaluation.
Optional inference dependencies are imported only when a real checkpoint is requested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dream.types import RobotState

MODEL_ID = "allenai/MolmoAct2-DROID"
REVISION = "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"
HORIZON = 15  # checkpoint norm_stats.json, franka_droid.action_horizon
SAMPLE_PERIOD_S = 1 / 15  # pinned DROID port_droid.py, DROID_FPS=15; sim_eval's 30 Hz differs


@dataclass(frozen=True)
class DecodedChunk:
    joint_rad: np.ndarray
    width_m: np.ndarray
    saturated: np.ndarray  # retain every physical bound projection in the execution trace


def decode_actions(actions: np.ndarray, joint_limits: np.ndarray) -> DecodedChunk:
    """Training-data semantics: absolute joint radians and normalized gripper closure.

    DROID robot.py explicitly projects closure to [0,1] before width=max_width*(1-closure).
    This is physical actuator saturation, not a fitted tolerance for the saved predictions.
    Panda uses its own 80 mm aperture; equal normalized aperture does not establish grasp
    geometry transfer from Robotiq. Raw predictions must be logged before this projection.
    """
    report = inspect_actions(actions, joint_limits)
    if not report["arm_within_panda_joint_limits"]:
        raise ValueError("absolute joint target outside physical limits")
    a = np.asarray(actions, dtype=float)
    closure = a[:, 7]
    return DecodedChunk(
        a[:, :7].copy(),
        0.08 * (1 - np.clip(closure, 0, 1)),
        (closure < 0) | (closure > 1),
    )


def verify_checkpoint(path: Path, hashes: dict[str, str]) -> None:
    """Reject changed custom code and normalisation metadata before importing it."""
    if path.name != REVISION or not hashes:
        raise ValueError("checkpoint must be the reviewed pinned snapshot")
    for name, digest in hashes.items():
        if (
            Path(name).name != name
            or hashlib.sha256((path / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"checkpoint source checksum mismatch: {name}")


def observation(
    external_rgb: np.ndarray, wrist_rgb: np.ndarray, state: RobotState, instruction: str
) -> dict:
    """Raw RGB and proprioception only; duplicate exterior slot as the model card permits.

    The eighth state value follows DROID robot.py's normalized measured aperture. Molmo's
    separate ManiSkill wrapper instead copies a Robotiq knuckle angle; that is not the
    training dataset's state contract. Equal closure does not reproduce hand geometry.
    """
    for rgb in (external_rgb, wrist_rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("camera input must be nonempty HxWx3 uint8 RGB")
    q = np.asarray(state.joint_pos)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("seven finite joint angles in radians are required")
    if not np.isfinite(state.gripper_width) or not 0 <= state.gripper_width <= 0.081:
        raise ValueError("gripper width outside the Panda's measured range")
    if not instruction.strip() or state.protective_stop:
        raise ValueError("instruction missing or robot is stopped")
    # DROID uses its hardware metadata max_width. Panda's measured maximum is 80 mm.
    closure = 1 - np.clip(state.gripper_width / 0.08, 0, 1)
    return {
        "images": [external_rgb.copy(), external_rgb.copy(), wrist_rgb.copy()],
        "state": np.asarray([*q, closure], dtype=np.float32),
        "task": instruction,
    }


def inspect_actions(actions: np.ndarray, joint_limits: np.ndarray) -> dict:
    """Audit the complete raw chunk without clipping, renormalising or executing it."""
    a = np.asarray(actions)
    limits = np.asarray(joint_limits)
    if a.shape != (HORIZON, 8) or not np.isfinite(a).all():
        raise ValueError("expected a finite 15x8 action chunk in robot scale")
    if (
        limits.shape != (7, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("expected seven ordered joint limit pairs")
    return {
        "shape": list(a.shape),
        "finite": True,
        "arm_within_panda_joint_limits": bool(
            np.all((a[:, :7] >= limits[:, 0]) & (a[:, :7] <= limits[:, 1]))
        ),
        "gripper_within_unit_interval": bool(np.all((a[:, 7] >= 0) & (a[:, 7] <= 1))),
        "execution_permitted": False,
        "reason": "inference-only: gripper mapping, camera shift and control frequency unvalidated",
    }


class MolmoBackend:
    """Real pinned checkpoint inference, with no fake outputs or controller fallback.

    The checkpoint contains custom Python: review its pinned source before use. BF16 uses
    the model card's autocast recipe; no upstream cache files are patched in place.
    """

    def __init__(self, model_path: Path, seed: int = 0) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this pilot")
        self.torch = torch
        self.generator = torch.Generator(device="cuda:0").manual_seed(seed)
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True, extra_special_tokens={}
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to("cuda:0")
            .eval()
        )

    def predict(self, request: dict) -> np.ndarray:
        from PIL import Image

        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            result = self.model.predict_action(
                processor=self.processor,
                images=[Image.fromarray(x) for x in request["images"]],
                task=request["task"],
                state=request["state"],
                norm_tag="franka_droid",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                enable_cuda_graph=False,
                num_steps=10,
                normalize_language=True,
                generator=self.generator,
            )
        # predict_action already unnormalises with pinned norm_stats.json. Do not do it twice.
        actions = result.actions.detach().float().cpu().numpy()
        if actions.shape == (1, HORIZON, 8):
            return actions[0]
        return actions
