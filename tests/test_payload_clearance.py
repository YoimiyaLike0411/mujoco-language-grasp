"""Measured collision-world isolation and raised-retraction invariants."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from graspbench.config import HOME_Q, SCENE_PATH
from graspbench.types import JointPositionCommand, Observation
from policies.grasp_geometry import Estimate
from policies.motion import Motion as ClearanceMotion
from policies.student_policy import StudentPolicy as ClearancePolicy


def test_target_collision_probe_restores_estimated_object_state():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    motion = ClearanceMotion(model)
    assert motion.safe_pose(HOME_Q)
    data = mujoco.MjData(model)
    data.qpos[motion.ik.qpos_ids] = HOME_Q
    mujoco.mj_forward(model, data)
    center = data.site_xpos[motion.ik.site_id].copy() + np.array([0, 0, 0.04])
    estimate = Estimate("banana", center, np.array([0.109, 0.178, 0.037]), np.eye(3), {})
    adr = motion.collision_model.jnt_qposadr[motion.collision_model.joint("banana_free").id]
    saved = motion.collision_data.qpos[adr : adr + 7].copy()
    model_pose = model.body_pos.copy()
    flags = int(model.opt.disableflags)
    assert not motion.safe_grasp_pose(HOME_Q, estimate)
    np.testing.assert_array_equal(motion.collision_data.qpos[adr : adr + 7], saved)
    np.testing.assert_array_equal(model.body_pos, model_pose)
    assert int(model.opt.disableflags) == flags


def test_collision_probe_restores_target_even_when_probe_errors(monkeypatch):
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    motion = ClearanceMotion(model)
    estimate = Estimate(
        "banana", np.array([0.55, 0, 0.42]), np.array([0.109, 0.178, 0.037]), np.eye(3), {}
    )
    adr = motion.collision_model.jnt_qposadr[motion.collision_model.joint("banana_free").id]
    saved = motion.collision_data.qpos[adr : adr + 7].copy()

    def fail(q):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(motion, "safe_pose", fail)
    with pytest.raises(RuntimeError, match="probe failure"):
        motion.safe_grasp_pose(HOME_Q, estimate)
    np.testing.assert_array_equal(motion.collision_data.qpos[adr : adr + 7], saved)


@pytest.mark.parametrize(
    "name,height", [("mustard_bottle", 0.134), ("potted_meat_can", 0.084)]
)
def test_all_payloads_retract_at_clearance_height_after_vertical_lift(name, height):
    p = ClearancePolicy()
    q = np.zeros(7)
    o = Observation(
        2.0,
        "Place object in tray",
        q,
        q,
        np.array([0.689, 0.072, 0.64]),
        np.array([1.0, 0, 0, 0]),
        0.6,
        (),
    )
    command = JointPositionCommand(q, 0.0)
    p.motion = SimpleNamespace(
        last_error={}, hold=lambda *args: command, pose_move=lambda *args, **kwargs: (command, True)
    )
    p.travel_height = 0.64
    p.stage = "lift"
    p.entered = 1.0
    p.reason = "Lift"
    p.index = 0
    p.retry = 0
    p.validation = {}
    p.verification_q = None
    p.actions = [(name, "square_tray")]
    p.grasp = SimpleNamespace(
        position=np.array([0.698, 0.059, 0.503]), quaternion=np.array([1.0, 0, 0, 0])
    )
    p.tray = np.array([0.3, -0.2, 0.41])
    p.lift_target = o.ee_position.copy()
    p.estimates = {
        name: Estimate(
            name, np.array([0.698, 0.059, 0.467]), np.array([0.068, 0.048, height]), np.eye(3), {}
        )
    }
    d = p.act(o)
    assert d.stage == "transfer"
    assert p.carry_target[2] == pytest.approx(0.64)
    assert p.carry_height == pytest.approx(0.64)
