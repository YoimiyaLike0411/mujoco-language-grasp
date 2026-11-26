"""Reusable pieces for the extension's grasp experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import ARM_DOF, HAND_OPEN, TABLE_Z
from .env import DexEnv
from .hand import GRASP_CENTRE_LOCAL, HandSynergy, fingertips, tcp_position
from .ik import ArmIK


def set_hand_gains(model: mujoco.MjModel, kp: float, kv: float, arm_dof: int = ARM_DOF) -> None:
    """Stiffen the hand's position servos (stock Menagerie uses kp=1)."""
    for actuator in range(arm_dof, model.nu):
        model.actuator_gainprm[actuator, 0] = kp
        model.actuator_biasprm[actuator, 1] = -kp
        model.actuator_biasprm[actuator, 2] = -kv
        model.actuator_forcerange[actuator] = [-3.0, 3.0]


def servo_to(
    env: DexEnv,
    target_qpos: np.ndarray,
    synergy: HandSynergy,
    *,
    steps: int = 700,
    tolerance: float = 0.02,
    recorder=None,
    camera: str = "diagonal",
) -> None:
    """Ramp the arm to `target_qpos` while holding the given hand posture."""
    start = env.data.qpos[: env.arm_dof].copy()
    ramp = max(1, steps // 3)
    for step in range(steps):
        alpha = min(1.0, (step + 1) / ramp)
        env.data.ctrl[: env.arm_dof] = start + alpha * (target_qpos - start)
        env.set_hand(synergy)
        mujoco.mj_step(env.model, env.data)
        if recorder is not None:
            recorder.capture(env.data, camera)
        if alpha >= 1.0 and np.linalg.norm(env.data.qpos[: env.arm_dof] - target_qpos) < tolerance:
            break


def cage_solution(
    ik: ArmIK,
    env: DexEnv,
    target_xyz: np.ndarray,
    palm_rotation: np.ndarray,
    centre_local: np.ndarray = GRASP_CENTRE_LOCAL,
):
    """IK for placing the hand's grasp cage centre at `target_xyz`."""
    palm_target = np.asarray(target_xyz, dtype=np.float64) - palm_rotation @ centre_local
    return ik.solve(env.data.qpos[: env.arm_dof].copy(), palm_target, palm_rotation)


def move_tcp(
    ik: ArmIK,
    env: DexEnv,
    target_xyz: np.ndarray,
    synergy: HandSynergy,
    *,
    steps: int = 500,
    recorder=None,
    centre_local: np.ndarray = GRASP_CENTRE_LOCAL,
) -> bool:
    """Servo the grasp cage centre to a world point; returns convergence."""
    _, palm_rotation = ik.forward(env.data.qpos[: env.arm_dof])
    solution = cage_solution(
        ik,
        env,
        np.asarray(target_xyz, dtype=np.float64),
        palm_rotation,
        centre_local=centre_local,
    )
    if not solution.converged:
        return False
    servo_to(env, solution.joint_position, synergy, steps=steps, recorder=recorder)
    return True


def move_tcp_with_fallback(
    ik: ArmIK,
    env: DexEnv,
    target_xyz: np.ndarray,
    synergy: HandSynergy,
    *,
    heights: tuple[float, ...] = (0.0, -0.04, -0.08, -0.12),
    steps: int = 600,
    recorder=None,
    centre_local: np.ndarray = GRASP_CENTRE_LOCAL,
) -> bool:
    """Move above a target, lowering the waypoint until IK succeeds.

    Reaching a fixed transit height is not always possible for every spawn pose
    on this arm; stepping the waypoint down keeps the planner productive without
    ever giving up on the approach itself.
    """
    target = np.asarray(target_xyz, dtype=np.float64)
    for offset in heights:
        candidate = target.copy()
        candidate[2] += offset
        if move_tcp(
            ik,
            env,
            candidate,
            synergy,
            steps=steps,
            recorder=recorder,
            centre_local=centre_local,
        ):
            return True
    return False


def release_and_retract(
    ik: ArmIK,
    env: DexEnv,
    retract_xyz: np.ndarray,
    *,
    start_closure: float,
    steps: int = 500,
    recorder=None,
    centre_local: np.ndarray = GRASP_CENTRE_LOCAL,
) -> bool:
    """Open the fingers while easing upward, so the object is not dragged.

    Releasing at a fixed pose lets the still-closed fingers flick the object
    sideways; ramping the closure down while the wrist lifts leaves the object
    on the tray floor.
    """
    _, palm_rotation = ik.forward(env.data.qpos[: env.arm_dof])
    solution = cage_solution(
        ik, env, np.asarray(retract_xyz, dtype=np.float64), palm_rotation,
        centre_local=centre_local,
    )
    if not solution.converged:
        return False
    start_joints = env.data.qpos[: env.arm_dof].copy()
    for step in range(steps):
        alpha = min(1.0, (step + 1) / (steps * 0.6))
        env.data.ctrl[: env.arm_dof] = start_joints + alpha * (
            solution.joint_position - start_joints
        )
        closure = start_closure + alpha * (HAND_OPEN - start_closure)
        env.set_hand(HandSynergy(closure))
        mujoco.mj_step(env.model, env.data)
        if recorder is not None:
            recorder.capture(env.data)
    return True


@dataclass
class GraspOutcome:
    object: str
    closure: float
    success: bool = False
    lift_m: float = 0.0
    slip_m: float = 0.0
    object_drift_m: float = 0.0
    finger_contacts: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "object": self.object,
            "closure": self.closure,
            "success": self.success,
            "lift_m": self.lift_m,
            "slip_m": self.slip_m,
            "object_drift_m": self.object_drift_m,
            "finger_contacts": self.finger_contacts,
            "notes": self.notes,
        }


def attempt_grasp(
    env: DexEnv,
    ik: ArmIK,
    object_name: str,
    *,
    x: float = 0.44,
    y: float = 0.02,
    closure: float = 0.8,
    approach: float = 0.10,
    lift: float = 0.10,
    recorder=None,
    settle_steps: int = 300,
) -> GraspOutcome:
    outcome = GraspOutcome(object=object_name, closure=closure)

    env.park_objects(keep=object_name)
    env.place_object(object_name, [x, y, TABLE_Z + 0.05])
    env.reset(settle_steps=settle_steps)
    start_position = env.object_position(object_name).copy()

    _, palm_rotation = ik.forward(env.data.qpos[: env.arm_dof])
    above = np.array([start_position[0], start_position[1], start_position[2] + approach])
    grasp = np.array([start_position[0], start_position[1], start_position[2]])

    approach_solution = cage_solution(ik, env, above, palm_rotation)
    if not approach_solution.converged:
        outcome.notes.append("approach IK failed")
        return outcome
    servo_to(env, approach_solution.joint_position, HandSynergy(HAND_OPEN), recorder=recorder)

    grasp_solution = cage_solution(ik, env, grasp, palm_rotation)
    if not grasp_solution.converged:
        outcome.notes.append("grasp IK failed")
        return outcome
    servo_to(env, grasp_solution.joint_position, HandSynergy(HAND_OPEN), steps=500, recorder=recorder)

    servo_to(
        env,
        grasp_solution.joint_position,
        HandSynergy(closure),
        steps=900,
        recorder=recorder,
    )
    outcome.finger_contacts = count_object_finger_contacts(env, object_name)
    outcome.object_drift_m = float(np.linalg.norm(env.object_position(object_name) - start_position))

    lifted = np.array([grasp[0], grasp[1], grasp[2] + lift])
    lift_solution = cage_solution(ik, env, lifted, palm_rotation)
    if not lift_solution.converged:
        outcome.notes.append("lift IK failed")
        return outcome
    servo_to(env, lift_solution.joint_position, HandSynergy(closure), steps=900, recorder=recorder)

    final_position = env.object_position(object_name).copy()
    outcome.lift_m = float(final_position[2] - start_position[2])
    outcome.slip_m = float(np.linalg.norm(final_position - tcp_position(env.model, env.data)))
    outcome.success = outcome.lift_m > 0.04
    return outcome


def count_object_finger_contacts(env: DexEnv, object_name: str) -> int:
    object_body = env.object_id(object_name)
    contacts = 0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        bodies = {
            env.model.geom_bodyid[contact.geom1],
            env.model.geom_bodyid[contact.geom2],
        }
        if object_body in bodies:
            other = next(b for b in bodies if b != object_body)
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if name.startswith(("ff_", "mf_", "rf_", "th_")):
                contacts += 1
    return contacts


def tip_positions_local(env: DexEnv) -> np.ndarray:
    palm = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    rotation = env.data.xmat[palm].reshape(3, 3)
    origin = env.data.xpos[palm]
    return np.array([rotation.T @ (tip - origin) for tip in fingertips(env.model, env.data)])


class VideoRecorder:
    """Collects frames for an mp4 without keeping a renderer per experiment."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480, every: int = 2):
        self.renderer = mujoco.Renderer(model, height, width)
        self.frames: list = []
        self.every = every
        self._counter = 0

    def capture(self, data: mujoco.MjData, camera: str = "diagonal") -> None:
        self._counter += 1
        if self._counter % self.every:
            return
        self.renderer.update_scene(data, camera=camera)
        self.frames.append(self.renderer.render().copy())

    def save(self, path) -> None:
        if not self.frames:
            return
        import imageio

        imageio.mimsave(path, self.frames, fps=30)
