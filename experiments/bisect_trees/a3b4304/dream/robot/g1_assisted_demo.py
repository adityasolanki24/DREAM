"""Upper-body reaching for the walking G1 while the pretrained WBC policy balances the legs.

The WBC lower-body policy (dream/robot/g1_wbc.py) holds the arms with PD targets
(``controller.arm_target``); this module chooses those targets by damped least-squares
inverse kinematics of the right arm on a scratch copy of the robot state, and moves them in
small steps so the policy sees a slowly changing upper body. Physical: the arm reaches under
PD torques in the same simulation the legs balance in.

Grasping is ASSISTED (as in BEHAVIOR/OmniGibson's assisted grasping): this hand has no
actuated fingers, so when the fingertips are within ATTACH_DISTANCE of the target a MuJoCo
``connect`` constraint (a ball joint, inactive until then) pins the hand to the handle's
body. Everything the handle body then does (slide, swing, stay put) is the mechanism's own
joint physics responding to the robot's pull.
"""

from __future__ import annotations

import mujoco
import numpy as np

HAND_BODY = "right_hand_index_0_link"  # fingertip point that touches the handle
ARM_JOINTS = tuple(f"right_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                   "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))  # fmt: skip
RIGHT_ARM_SLICE = slice(7, 14)  # in WBCLocomotion.arm_target: left arm 0-6, right arm 7-13
ATTACH_DISTANCE = 0.04  # m
GRASP_EQ = "assisted_grasp"


def add_grasp_constraint(spec: mujoco.MjSpec, target_body: str) -> None:
    """Scene construction: an inactive hand-to-``target_body`` connect constraint."""
    eq = spec.add_equality()
    eq.name = GRASP_EQ
    eq.type = mujoco.mjtEq.mjEQ_CONNECT
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = HAND_BODY, target_body
    eq.active = False


class RightArm:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, controller) -> None:
        self.m, self.d, self.ctrl = model, data, controller
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS])
        self.dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS], 1]
        self.hand = model.body(HAND_BODY).id
        self.scratch = mujoco.MjData(model)
        self.eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_EQ)

    def hand_position(self) -> np.ndarray:
        return self.d.xpos[self.hand].copy()

    def solve(self, target: np.ndarray, iters: int = 60) -> np.ndarray:
        """Arm joint angles putting the fingertip at ``target`` (world), from the current
        state; joint limits respected; the rest of the body is held where it is."""
        s = self.scratch
        s.qpos[:] = self.d.qpos
        q = s.qpos[self.qadr].copy()
        jac = np.zeros((3, self.m.nv))
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            err = target - s.xpos[self.hand]
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jacBody(self.m, s, jac, None, self.hand)
            j = jac[:, self.dadr]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q

    def set_targets(self, q: np.ndarray) -> None:
        self.ctrl.arm_target[RIGHT_ARM_SLICE] = q

    def current_targets(self) -> np.ndarray:
        return self.ctrl.arm_target[RIGHT_ARM_SLICE].copy()

    def attach(self) -> bool:
        """Pin the fingertip to the target body at the current contact point."""
        if self.eq < 0:
            return False
        other = self.m.eq_obj2id[self.eq]
        p = self.d.xpos[self.hand]
        r_hand = self.d.xmat[self.hand].reshape(3, 3)
        r_other = self.d.xmat[other].reshape(3, 3)
        self.m.eq_data[self.eq, 0:3] = 0.0  # fingertip body origin
        self.m.eq_data[self.eq, 3:6] = r_other.T @ (p - self.d.xpos[other])
        del r_hand
        self.d.eq_active[self.eq] = 1
        return True

    def release(self) -> None:
        if self.eq >= 0:
            self.d.eq_active[self.eq] = 0
