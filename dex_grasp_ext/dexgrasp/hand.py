"""Kinematic helpers for the mounted dexterous hand."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Grasp cage centre expressed in the palm body frame.  Measured with the hand
# closed to GRASP_CLOSURE, as the midpoint between the thumb tip and the mean of
# the three finger tips (see scripts/03_scene_probe.py).
GRASP_CENTRE_LOCAL = np.array([0.0701, 0.0198, -0.0219])
GRASP_CLOSURE = 0.8
OPEN_CLOSURE = 0.15
FINGERS = ("ff_tip", "mf_tip", "rf_tip", "th_tip")


@dataclass(frozen=True)
class HandSynergy:
    """A uniform finger-closure command over the hand's position actuators."""

    fraction: float

    def ctrl(self, model: mujoco.MjModel, arm_dof: int = 7) -> np.ndarray:
        low = model.actuator_ctrlrange[arm_dof:, 0]
        high = model.actuator_ctrlrange[arm_dof:, 1]
        return low + self.fraction * (high - low)


def body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def palm_frame(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    palm = body_id(model, "palm")
    return data.xpos[palm].copy(), data.xmat[palm].reshape(3, 3).copy()


def tcp_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the hand's grasp centre."""
    position, rotation = palm_frame(model, data)
    return position + rotation @ GRASP_CENTRE_LOCAL


def fingertips(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World positions of the four fingertips, shape (4, 3)."""
    return np.array([data.xpos[body_id(model, name)].copy() for name in FINGERS])


def grip_radius(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Mean fingertip distance to their centroid: the enclosed object radius."""
    tips = fingertips(model, data)
    return float(np.linalg.norm(tips - tips.mean(axis=0), axis=1).mean())


def opposed_cage(tips: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the thumb-vs-fingers opposition cage.

    The three fingers of the Allegro hand close onto nearly the same point, so
    a general "equidistant from four points" fit is ill-conditioned.  The grasp
    that actually forms is the thumb opposing the three-finger pad, so the cage
    centre is the midpoint between the thumb tip and the mean finger tip, and
    the cage radius is half the thumb-to-fingers gap.
    """
    tips = np.asarray(tips, dtype=np.float64)
    thumb = tips[3]
    fingers = tips[:3].mean(axis=0)
    gap = float(np.linalg.norm(tips[:3] - thumb, axis=1).mean())
    return 0.5 * (thumb + fingers), 0.5 * gap


def cage_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    fractions: np.ndarray | None = None,
    settle_steps: int = 900,
    arm_dof: int = 7,
) -> list[dict]:
    """Measure, for each closure, the enclosed radius and its centre.

    The centre is expressed in the palm body frame and is the point that is
    (as closely as possible) equidistant from the four finger tips.  Grasp
    planning then picks the closure whose radius matches the target object.
    """
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 21)
    palm = body_id(model, "palm")
    rows = []
    for fraction in fractions:
        data.ctrl[arm_dof:] = HandSynergy(float(fraction)).ctrl(model, arm_dof)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        rotation = data.xmat[palm].reshape(3, 3)
        origin = data.xpos[palm].copy()
        local_tips = np.array(
            [rotation.T @ (tip - origin) for tip in fingertips(model, data)]
        )
        centre, radius = opposed_cage(local_tips)
        rows.append(
            {
                "fraction": float(fraction),
                "radius_m": radius,
                "centre_local": centre.tolist(),
                "tip_spread_m": float(
                    np.linalg.norm(local_tips - local_tips.mean(axis=0), axis=1).mean()
                ),
            }
        )
    return rows


def plan_closure(calibration: list[dict], object_radius: float) -> dict:
    """Pick the closure whose enclosed radius best matches `object_radius`."""
    return min(
        calibration,
        key=lambda row: abs(row["radius_m"] - object_radius),
    )


def load_calibration(path) -> list[dict]:
    import json
    import pathlib

    payload = json.loads(pathlib.Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"empty cage calibration at {path}")
    return payload
