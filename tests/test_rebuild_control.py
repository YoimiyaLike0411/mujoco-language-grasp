"""Control invariants that the starter's API-only test does not exercise."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from graspbench.config import HOME_Q, SCENE_PATH
from graspbench.types import Observation
from policies.grounding import parse_actions
from policies.motion import Motion, grasp_rotation, quaternion


def observation(model, data, motion):
    mujoco.mj_forward(model, data)
    return Observation(
        float(data.time),
        "Place the red cube into the square tray.",
        data.qpos[motion.ik.qpos_ids].copy(),
        data.qvel[motion.ik.dof_ids].copy(),
        data.site_xpos[motion.ik.site_id].copy(),
        quaternion(data.site_xmat[motion.ik.site_id].reshape(3, 3)),
        1.0,
        (),
    )


def test_tilt_does_not_mix_height_into_jaw_width():
    for angle in np.linspace(0, 2 * np.pi, 17):
        for tilt in (-0.6, -0.25, 0, 0.25, 0.6):
            rot = grasp_rotation(
                np.array([np.cos(angle), np.sin(angle)]), np.array([0.7, 0.2]), tilt
            )
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
            assert abs(rot[2, 1]) < 1e-12
            assert np.linalg.det(rot) > 0.999999


def test_repeated_snapshot_does_not_advance_motion():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    motion = Motion(model)
    data = mujoco.MjData(model)
    data.qpos[motion.ik.qpos_ids] = HOME_Q
    obs = observation(model, data, motion)
    goal = obs.ee_position + np.array([0.01, 0.0, 0.01])
    first, _ = motion.pose_move(obs, goal, obs.ee_quaternion, 1.0)
    for _ in range(5):
        again, _ = motion.pose_move(obs, goal, obs.ee_quaternion, 1.0)
        np.testing.assert_allclose(again.joint_position, first.joint_position, atol=1e-12)


def test_gravity_feedforward_preserves_the_model_and_cancels_static_arm_load():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    motion = Motion(model)
    data = mujoco.MjData(model)
    data.qpos[motion.ik.qpos_ids] = HOME_Q
    obs = observation(model, data, motion)
    gravcomp = model.body_gravcomp.copy()
    gains = model.actuator_gainprm.copy()
    command = motion.hold(obs, 1.0)
    static_torque = model.actuator_gainprm[:7, 0] * (command.joint_position - obs.joint_position)
    np.testing.assert_allclose(static_torque, data.qfrc_bias[motion.ik.dof_ids], atol=1e-10)
    np.testing.assert_array_equal(model.body_gravcomp, gravcomp)
    np.testing.assert_array_equal(model.actuator_gainprm, gains)


def test_destinations_come_from_clauses_not_fixed_category_mapping():
    actions = parse_actions("把所有水果放进方盘，把所有包装食品放进圆盘。")
    assert set(actions) == {
        ("apple", "square_tray"),
        ("orange", "square_tray"),
        ("mustard_bottle", "round_tray"),
        ("potted_meat_can", "round_tray"),
    }


def test_official_physics_and_evaluator_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    import json

    for name, digest in json.loads((root / "original_files.sha256.json").read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_ik_contact_optimization_preserves_scene_and_solution():
    from graspbench.ik import DampedLeastSquaresIK

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    flags = int(model.opt.disableflags)
    motion = Motion(model)
    reference = DampedLeastSquaresIK(model, orientation_weight=0.5)
    data = mujoco.MjData(model)
    data.qpos[motion.ik.qpos_ids] = HOME_Q
    obs = observation(model, data, motion)
    target = obs.ee_position + np.array([0.025, -0.015, 0.02])
    expected = reference.solve(HOME_Q, target, obs.ee_quaternion)
    actual = motion.ik.solve(HOME_Q, target, obs.ee_quaternion)
    assert expected.converged and actual.converged
    np.testing.assert_allclose(actual.joint_position, expected.joint_position, atol=1e-12)
    assert int(model.opt.disableflags) == flags
    assert int(motion.collision_model.opt.disableflags) == flags
    assert motion.ik.model is not model
