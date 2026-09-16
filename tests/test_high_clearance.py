"""Regression for high travel and vertical departure, including actual IK."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np

from graspbench.config import HOME_Q, SCENE_PATH
from graspbench.types import JointPositionCommand, Observation
from policies.grasp_geometry import Estimate, plan_grasp
from policies.motion import Motion, quaternion
from policies.student_policy import StudentPolicy


def policy_at(stage, opening=0.6):
    p = StudentPolicy()
    p.stage, p.entered, p.reason = stage, 0.0, stage
    p.index = p.retry = 0
    p.travel_height = 0.64
    p.actions = [("potted_meat_can", "square_tray")]
    p.validation, p.estimates = {}, {}
    p.verification_q = None
    p.grasp = SimpleNamespace(
        position=np.array([0.65, -0.15, 0.45]),
        pregrasp=np.array([0.62, -0.15, 0.51]),
        quaternion=np.array([0., 1., 0., 0.]),
    )
    o = Observation(2.0, "", np.zeros(7), np.zeros(7),
                    np.array([0.65, -0.15, 0.455]), p.grasp.quaternion, opening, ())
    command = JointPositionCommand(np.zeros(7), opening)
    p.motion = SimpleNamespace(last_error={}, hold=lambda *a: command,
                               command=lambda *a: command,
                               pose_move=lambda *a, **kw: (command, False))
    return p, o


def test_close_departure_is_vertical_before_any_retraction():
    p, o = policy_at("close")
    p.close_q = o.joint_position
    d = p.act(o)
    assert d.stage == "lift"
    np.testing.assert_array_equal(p.lift_target[:2], o.ee_position[:2])
    assert p.lift_target[2] == 0.64


def test_lift_cannot_start_horizontal_retraction_until_reached():
    p, o = policy_at("lift")
    p.lift_target = np.array([0.65, -0.15, 0.64])
    assert p.act(o).stage == "lift"
    assert not hasattr(p, "retract_target")


def test_adaptive_lift_retracts_once_selected_height_is_stably_reached():
    p, o = policy_at("lift")
    p.grasp.clearance_height = 0.531
    p.lift_target = np.array([0.65, -0.15, 0.531])
    p.motion.last_error = {"position_error": 0.011, "orientation_error": 0.01}
    o = replace(
        o,
        time=2.0,
        ee_position=np.array([0.645, -0.15, 0.538]),
        joint_velocity=np.zeros(7),
    )
    assert p.act(o).stage == "retract"
    assert p.retract_target[2] == o.ee_position[2]


def test_recovery_first_raises_at_current_xy():
    p, o = policy_at("descend")
    p.fail(o, "retry")
    np.testing.assert_array_equal(p.raise_target[:2], o.ee_position[:2])
    assert p.raise_target[2] == 0.64
    assert p.act(replace(o, time=2.1)).stage == "recover"


def test_home_can_raise_without_changing_attitude():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    motion = Motion(model)
    data = mujoco.MjData(model)
    data.qpos[motion.ik.qpos_ids] = HOME_Q
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[motion.ik.site_id].copy()
    quat = quaternion(data.site_xmat[motion.ik.site_id].reshape(3, 3))
    for z in np.linspace(pos[2], 0.64, 12):
        solution = motion.ik.solve(HOME_Q, np.r_[pos[:2], z], quat)
        assert solution.converged
        assert motion.safe_pose(solution.joint_position)


def test_seed_977989_grasps_have_reachable_vertical_exit():
    motion = Motion(mujoco.MjModel.from_xml_path(str(SCENE_PATH)))
    for name, center, size in (
        ("mustard_bottle", [0.6051, 0.0363, 0.467], [0.068, 0.048, 0.134]),
        ("apple", [0.6982, 0.1571, 0.4363], [0.076, 0.076, 0.072]),
    ):
        item = Estimate(name, np.array(center), np.array(size), np.eye(3), {})
        grasp = plan_grasp(item, motion, HOME_Q)
        assert grasp.clearance_height == 0.64
        q = grasp.pregrasp_q
        for z in np.linspace(grasp.position[2], 0.64, 12):
            solution = motion.ik.solve(q, np.r_[grasp.position[:2], z], grasp.quaternion)
            assert solution.converged, (name, z)
            q = solution.joint_position


def test_edge_can_falls_back_to_local_half_meter_departure():
    motion = Motion(mujoco.MjModel.from_xml_path(str(SCENE_PATH)))
    current_q = np.array(
        [-0.2080, -0.6909, -0.2996, -1.9483, -0.0336, 1.4575, 0.1403]
    )
    item = Estimate(
        "potted_meat_can",
        np.array([0.6721, 0.1758, 0.4420]),
        np.array([0.1, 0.06, 0.084]),
        np.array(
            [
                [-0.01077, -0.99994, 0.0],
                [0.99994, -0.01077, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        {},
    )
    grasp = plan_grasp(item, motion, current_q)
    assert 0.53 <= grasp.clearance_height < 0.64
