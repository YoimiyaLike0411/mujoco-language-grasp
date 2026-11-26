"""Thin wrapper around the generated MuJoCo scene."""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

from .config import ARM_DOF, HAND_OPEN, HOME_Q_DEX
from .hand import HandSynergy, body_id, tcp_position

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/scene_dex.xml"
# Objects are parked far to the side rather than below the floor: the scene has
# an infinite ground plane, so pushing bodies underneath it makes the contact
# solver explode.
PARK_XY = (6.0, 0.0)


class DexEnv:
    """Loads the arm+hand scene and provides scripting-level helpers."""

    def __init__(self, scene: pathlib.Path | str = SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.arm_dof = ARM_DOF
        self.object_bodies = [
            "red_cube",
            "green_cylinder",
            "blue_box",
            "banana",
            "apple",
            "orange",
            "mustard_bottle",
            "potted_meat_can",
            "scissors",
            "marker",
        ]

    # ---------------------------------------------------------------- helpers
    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def object_id(self, name: str) -> int:
        return body_id(self.model, name)

    def object_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.object_id(name)].copy()

    def tcp(self) -> np.ndarray:
        return tcp_position(self.model, self.data)

    def set_arm(self, qpos: np.ndarray) -> None:
        self.data.qpos[: self.arm_dof] = qpos
        self.data.ctrl[: self.arm_dof] = qpos

    def set_hand(self, synergy: HandSynergy) -> None:
        self.data.ctrl[self.arm_dof :] = synergy.ctrl(self.model, self.arm_dof)

    def reset(self, *, settle_steps: int = 400) -> None:
        """Park the arm at the raised home pose with an open hand."""
        self.data.qpos[: self.arm_dof] = HOME_Q_DEX
        self.data.ctrl[: self.arm_dof] = HOME_Q_DEX
        self.set_hand(HandSynergy(HAND_OPEN))
        self.data.qvel[:] = 0.0
        self.forward()
        self.settle(settle_steps)

    # ------------------------------------------------------------- scene setup
    def park_objects(self, keep: str | None = None) -> None:
        """Move every object far off the table except `keep`.

        The stock scene positions objects for the course reset routine; for
        targeted hand experiments we start from a single object on the table.
        Parked bodies are spread out so they cannot interpenetrate each other.
        """
        for index, name in enumerate(self.object_bodies):
            if name == keep:
                continue
            adr = self._free_joint_qpos_adr(name)
            self.data.qpos[adr] = PARK_XY[0] + 0.6 * index
            self.data.qpos[adr + 1] = PARK_XY[1]
            self.data.qpos[adr + 2] = 0.5
        self.data.qvel[:] = 0.0

    def place_object(self, name: str, position, yaw: float = 0.0) -> None:
        adr = self._free_joint_qpos_adr(name)
        self.data.qpos[adr : adr + 3] = position
        half = yaw / 2.0
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        self.data.qvel[:] = 0.0

    def _free_joint_qpos_adr(self, name: str) -> int:
        body = self.object_id(name)
        joint = self.model.body_jntadr[body]
        if joint < 0:
            raise ValueError(f"body {name!r} has no free joint")
        return int(self.model.jnt_qposadr[joint])

    def settle(self, steps: int = 400) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
