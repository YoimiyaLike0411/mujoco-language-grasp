"""Robot-only motion primitives. Never access simulator object state."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from graspbench.config import CONTAINER_SPEC_BY_NAME, OBJECT_SPEC_BY_NAME, TABLE_TOP_Z
from graspbench.ik import DampedLeastSquaresIK, _rotation_vector
from graspbench.types import JointPositionCommand


def matrix(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quat))
    return result.reshape(3, 3)


def quaternion(rotation):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).ravel())
    return result


def grasp_rotation(closing_axis, tilt_direction, tilt):
    radial = np.asarray(tilt_direction, dtype=float)
    radial /= max(np.linalg.norm(radial), 1e-9)
    y = np.r_[np.asarray(closing_axis), 0.0]
    y /= np.linalg.norm(y)
    direction = np.r_[radial, 0.0]
    direction -= y * np.dot(direction, y)
    if np.linalg.norm(direction) < 1e-5:
        direction = np.cross(y, [0.0, 0.0, 1.0])
    direction /= np.linalg.norm(direction)
    z = direction * np.sin(tilt) + np.array([0.0, 0.0, -np.cos(tilt)])
    return np.column_stack((np.cross(y, z), y, z))


class Motion:
    def __init__(self, model):
        self.model = model
        # IK needs kinematics only. Avoid solving unrelated scene contacts in
        # every iteration; retain the original physics and collision checker.
        ik_model = copy.copy(model)
        ik_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.ik = DampedLeastSquaresIK(ik_model, orientation_weight=0.5)
        self.data = mujoco.MjData(model)
        self.last_error = {}
        # A separate planning world is sanitized before use: runtime model body
        # poses of objects/containers must never become a perception shortcut.
        self.collision_model = copy.copy(model)
        for name in CONTAINER_SPEC_BY_NAME:
            self.collision_model.body_pos[self.collision_model.body(name).id] = [
                5.0,
                5.0,
                TABLE_TOP_Z,
            ]
        self.collision_data = mujoco.MjData(self.collision_model)
        for name in OBJECT_SPEC_BY_NAME:
            joint = self.collision_model.joint(name + "_free").id
            adr = self.collision_model.jnt_qposadr[joint]
            self.collision_data.qpos[adr : adr + 3] = [5.0, 5.0, -2.0]
        self.robot_names = {"link" + str(i) for i in range(8)} | {
            "hand",
            "left_finger",
            "right_finger",
        }

    def set_scene(self, estimates):
        for name in CONTAINER_SPEC_BY_NAME:
            if name in estimates:
                self.collision_model.body_pos[self.collision_model.body(name).id] = np.r_[
                    estimates[name].center[:2], TABLE_TOP_Z
                ]

    def safe_pose(self, q):
        self.collision_data.qpos[self.ik.qpos_ids] = q
        self.collision_data.qpos[7:9] = 0.04
        mujoco.mj_forward(self.collision_model, self.collision_data)
        for c in self.collision_data.contact:
            a = self.collision_model.body(int(self.collision_model.geom_bodyid[c.geom1])).name
            b = self.collision_model.body(int(self.collision_model.geom_bodyid[c.geom2])).name
            if (a in self.robot_names or b in self.robot_names) and c.dist < -0.00005:
                return False
        return True

    def safe_grasp_pose(self, q, estimate):
        """Check the open hand against a target placed from visual estimates."""
        joint = self.collision_model.joint(estimate.name + "_free").id
        adr = self.collision_model.jnt_qposadr[joint]
        saved = self.collision_data.qpos[adr : adr + 7].copy()
        try:
            self.collision_data.qpos[adr : adr + 3] = estimate.center
            self.collision_data.qpos[adr + 3 : adr + 7] = quaternion(estimate.rotation)
            return self.safe_pose(q)
        finally:
            self.collision_data.qpos[adr : adr + 7] = saved

    def command(self, q, observation, opening):
        # Actuator torque = kp*(ctrl-q)-kd*qvel. Static feed-forward therefore
        # belongs in ctrl as gravity/kp. This does not change the model/physics.
        self.data.qpos[self.ik.qpos_ids] = observation.joint_position
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        compensation = self.data.qfrc_bias[self.ik.dof_ids] / self.model.actuator_gainprm[:7, 0]
        return JointPositionCommand(np.asarray(q) + compensation, opening)

    def hold(self, observation, opening):
        return self.command(observation.joint_position, observation, opening)

    def joint_move(self, observation, target, opening, step=0.065):
        difference = target - observation.joint_position
        self.last_error["joint_error"] = float(np.max(np.abs(difference)))
        q = observation.joint_position + difference * min(
            1.0, step / max(np.max(np.abs(difference)), 1e-9)
        )
        return self.command(q, observation, opening), np.max(np.abs(difference)) < 0.012

    def pose_move(self, observation, position, quat, opening, *, speed=0.12, joint_step=0.045):
        # Stateless with respect to worker calls: equal observations produce
        # equal commands; a stale dropped command cannot advance a trajectory.
        xyz_error = np.asarray(position) - observation.ee_position
        ori_error = _rotation_vector(matrix(quat) @ matrix(observation.ee_quaternion).T)
        self.last_error = {
            "position_error": float(np.linalg.norm(xyz_error)),
            "orientation_error": float(np.linalg.norm(ori_error)),
        }
        reached = np.linalg.norm(xyz_error) < 0.003 and np.linalg.norm(ori_error) < 0.045
        delta = xyz_error * min(1.0, speed * 0.16 / max(np.linalg.norm(xyz_error), 1e-9))
        solution = self.ik.solve(
            observation.joint_position,
            observation.ee_position + delta,
            quat,
            max_iterations=35,
            position_tolerance=0.0005,
            orientation_tolerance=0.01,
        )
        if solution.position_error > 0.006 or solution.orientation_error > 0.10:
            self.last_error["ik_rejected"] = True
            return self.hold(observation, opening), False
        command, _ = self.joint_move(observation, solution.joint_position, opening, step=joint_step)
        return command, bool(reached)

    def tracked_descent(self, observation, position, quat, opening, *, speed=0.08):
        """Follow the live fruit laterally without scaling it by descent distance."""
        target = np.array(position, copy=True)
        target[2] = max(target[2], observation.ee_position[2] - speed * 0.16)
        command, reached = self.pose_move(
            observation, target, quat, opening, speed=max(speed, 0.12)
        )
        # The intermediate waypoint must never trigger an early close.
        return command, bool(
            reached and np.linalg.norm(np.asarray(position) - observation.ee_position) < 0.003
        )
