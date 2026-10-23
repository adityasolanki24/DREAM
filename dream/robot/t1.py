"""CPU-testable Isaac Lab T1 policy contract, isolated from navigation and scene state.

Reference: APM f74e2dec, src/apm_isaac/robots.py and lab/t1.py. No APM code or weights
are vendored. The 81-value Isaac Lab PPO contract is not Booster's different 47-value
published leg-policy contract. Config and policy hashes are required at the boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class T1Policy:
    """Map named proprioception to the exported PPO actor and bounded joint targets."""

    def __init__(self, cfg: dict, policy, joint_names: list[str]) -> None:
        expected = cfg["joint_names"]
        if (
            len(expected) != 23
            or len(set(expected)) != 23
            or len(set(joint_names)) != len(joint_names)
            or set(expected) != set(joint_names)
            or cfg["sim_dt"] != 0.005
            or cfg["decimation"] != 4
            or cfg["action_scale"] != 0.5
        ):
            raise ValueError("unsupported T1 policy/robot contract")
        self.indices = np.array([joint_names.index(n) for n in expected])
        self.default = np.asarray(cfg["default_joint_pos"], float)
        if self.default.shape != (23,) or not np.isfinite(self.default).all():
            raise ValueError("invalid default joint positions")
        self.policy = policy
        self.last = np.zeros(23, dtype=np.float32)

    @classmethod
    def load(cls, folder: Path, hashes: dict, joint_names: list[str]):
        """Load only a reviewed external policy; no download, training or stand-in."""
        for name in ("env_cfg.json", "policy.onnx"):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != hashes[name]:
                raise ValueError("T1 policy/config checksum mismatch")
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        actor = onnxruntime.InferenceSession(
            str(folder / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = actor.get_inputs(), actor.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != 81 or outputs[0].shape[-1] != 23:
            raise ValueError("wrong T1 actor input/output dimensions")

        def predict(obs):
            return actor.run([outputs[0].name], {inputs[0].name: obs[None]})[0][0]

        return cls(json.loads((folder / "env_cfg.json").read_text()), predict, joint_names)

    def target(self, q, dq, linear_body, gyro_body, gravity_body, command, limits) -> np.ndarray:
        """Proprioception only; reject unsafe actions rather than silently clipping them."""
        q, dq = np.asarray(q), np.asarray(dq)
        limits = np.asarray(limits)
        vectors = [np.asarray(v) for v in (linear_body, gyro_body, gravity_body, command)]
        if (
            q.shape != (23,)
            or dq.shape != (23,)
            or limits.shape != (23, 2)
            or any(v.shape != (3,) for v in vectors)
            or not all(np.isfinite(v).all() for v in [q, dq, limits, *vectors])
            or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(q < limits[:, 0])
            or np.any(q > limits[:, 1])
            or np.any(np.abs(command) > [0.2, 0.0, 0.4])
        ):
            raise ValueError("invalid T1 proprioception/command")
        obs = np.concatenate(
            [
                *vectors,
                q[self.indices] - self.default,
                dq[self.indices],
                self.last,
            ]
        ).astype(np.float32)
        assert obs.shape == (81,)
        action = np.asarray(self.policy(obs), dtype=np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("invalid locomotion policy action")
        target = np.empty(23)
        target[self.indices] = self.default + 0.5 * action
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise ValueError("locomotion policy target exceeds physical joint limits")
        self.last = action.copy()
        return target
