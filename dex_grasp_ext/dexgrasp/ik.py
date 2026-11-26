"""Damped-least-squares IK for the extension's arm.

Written for this extension so the dexterous-grasp code stays self-contained: it
only needs the MuJoCo model and the name of the body that carries the hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    joint_position: np.ndarray
    converged: bool
    position_error: float
    orientation_error: float


class ArmIK:
    def __init__(self, model: mujoco.MjModel, tool_body: str = "palm", arm_dof: int = 7):
        self.model = model
        self.data = mujoco.MjData(model)
        self.tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_body)
        if self.tool_body < 0:
            raise ValueError(f"unknown tool body {tool_body!r}")
        self.arm_dof = arm_dof
        self.jnt_low = model.jnt_range[:arm_dof, 0].copy()
        self.jnt_high = model.jnt_range[:arm_dof, 1].copy()

    def forward(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = 0.0
        self.data.qpos[: self.arm_dof] = qpos
        mujoco.mj_forward(self.model, self.data)
        return (
            self.data.xpos[self.tool_body].copy(),
            self.data.xmat[self.tool_body].reshape(3, 3).copy(),
        )

    def solve(
        self,
        initial_qpos: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        iterations: int = 300,
        damping: float = 0.05,
        step_limit: float = 0.2,
        position_tolerance: float = 1.5e-3,
        orientation_tolerance: float = 3e-2,
    ) -> IKResult:
        qpos = np.asarray(initial_qpos, dtype=np.float64).copy()
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        best: tuple[float, np.ndarray] | None = None

        for _ in range(iterations):
            position, rotation = self.forward(qpos)
            error_position = target_position - position
            error_rotation = _rotation_vector(target_rotation @ rotation.T)
            norm = float(np.linalg.norm(error_position) + np.linalg.norm(error_rotation))
            if best is None or norm < best[0]:
                best = (norm, qpos.copy())
            if (
                np.linalg.norm(error_position) < position_tolerance
                and np.linalg.norm(error_rotation) < orientation_tolerance
            ):
                return IKResult(
                    qpos,
                    True,
                    float(np.linalg.norm(error_position)),
                    float(np.linalg.norm(error_rotation)),
                )
            error = np.concatenate([error_position, error_rotation])
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.tool_body)
            jacobian = np.vstack([jacp[:, : self.arm_dof], jacr[:, : self.arm_dof]])
            gain = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            norm_gain = float(np.linalg.norm(gain))
            if norm_gain > step_limit:
                gain *= step_limit / norm_gain
            qpos = np.clip(qpos + gain, self.jnt_low, self.jnt_high)

        position, rotation = self.forward(best[1])
        return IKResult(
            best[1],
            False,
            float(np.linalg.norm(target_position - position)),
            float(np.linalg.norm(_rotation_vector(target_rotation @ rotation.T))),
        )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).reshape(9))
    if quat[0] < 0:
        quat = -quat
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
    axis = quat[1:]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(3)
    return axis / norm * angle
