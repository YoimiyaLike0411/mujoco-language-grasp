"""Behavior regression for the isolated direct-support candidate."""

from types import SimpleNamespace

import numpy as np
import pytest

from graspbench.types import JointPositionCommand, Observation
from policies import student_policy as module
from policies.grasp_geometry import Estimate

DirectCornerPolicy = module.StudentPolicy
namespace = module.__dict__


class Motion:
    def __init__(self):
        self.last_error = {}

    def command(self, q, o, opening):
        return JointPositionCommand(np.array(q, copy=True), opening)

    def hold(self, o, opening):
        return self.command(o.joint_position, o, opening)

    def set_scene(self, estimates):
        pass

    def pose_move(self, o, *args, **kwargs):
        return self.hold(o, 0), False


def setup_policy():
    p = DirectCornerPolicy()
    p.motion = Motion()
    p.travel_height = 0.64
    p.index = 0
    p.retry = 0
    p.entered = 0
    p.validation = {}
    p.verification_q = None
    p.actions = [("apple", "square_tray")]
    p.grasp = SimpleNamespace(
        position=np.array([0.65, -0.1, 0.434]), quaternion=np.array([1.0, 0, 0, 0])
    )
    p.estimates = {
        "square_tray": Estimate(
            "square_tray", np.array([0.3, -0.2, 0.410]), np.array([0.22, 0.22, 0.01]), np.eye(3), {}
        )
    }
    o = Observation(
        0.2,
        "Place the apple into the square tray.",
        np.zeros(7),
        np.zeros(7),
        np.array([0.31, -0.2, 0.55]),
        np.array([1.0, 0, 0, 0]),
        1.0,
        (),
    )
    return p, o


def test_drop_above_destination_clears_hand_before_verifying():
    from dataclasses import replace

    p, o = setup_policy()
    p.stage = "transfer"
    p.reason = "transfer"
    p.carry_target = np.array([0.3, -0.2, 0.55])
    p.carry_quat = np.array([1.0, 0, 0, 0])
    d = p.act(replace(o, gripper_opening=0.02))
    assert d.stage == "release" and not d.done and not d.request_retry
    assert p.retry == 0


@pytest.mark.parametrize("z,expected", [(0.452, "verify"), (0.55, "approach_raise")])
def test_only_supported_clear_fruit_skips_regrasp(monkeypatch, z, expected):
    from dataclasses import replace

    p, o = setup_policy()
    p.stage = "perceive"
    p.reason = "perceive"
    apple = Estimate(
        "apple", np.array([0.3, -0.2, z]), np.array([0.076, 0.076, 0.072]), np.eye(3), {}
    )
    p.perception = SimpleNamespace(estimate=lambda *args: {**p.estimates, "apple": apple})
    monkeypatch.setitem(namespace, "plan_grasp", lambda *args: p.grasp)
    d = p.act(replace(o, ee_position=np.array([0.426, 0, 0.50])))
    assert d.stage == expected and not d.done
    assert d.command.gripper_opening == 1.0


def test_middle_fruit_jitter_does_not_block_remaining_objects():
    from dataclasses import replace

    p, o = setup_policy()
    p.actions.append(("orange", "square_tray"))
    positions = iter(
        [
            np.array([0.295, -0.2, 0.452]),
            np.array([0.305, -0.2, 0.452]),
            np.array([0.3, -0.205, 0.452]),
        ]
    )
    p.perception = SimpleNamespace(held_center=lambda *args, **kwargs: next(positions))
    p.enter("verify", o, "Hand is clear")
    for i in range(3):
        d = p.act(replace(o, time=1.0 + i))
        assert d.stage not in ("verify", "verify_place") and not d.done
    assert p.index == 1 and p.stage == "perceive"


def test_final_verification_waits_for_camera_clear_motion():
    from dataclasses import replace

    p, o = setup_policy()
    p.stage = "retreat"
    p.reason = "retreat"
    p.retreat_target = np.array([0.31, -0.2, 0.64])
    p.carry_quat = np.array([1.0, 0, 0, 0])
    p.motion.pose_move = lambda obs, *args, **kwargs: (p.motion.hold(obs, 1.0), True)
    d = p.act(o)
    assert p.stage == "observe_clear" and d.stage == "observe_clear"
    p.motion.pose_move = lambda obs, *args, **kwargs: (p.motion.hold(obs, 1.0), False)
    d = p.act(replace(o, time=0.4))
    assert d.stage == "observe_clear" and not d.done
    p.motion.pose_move = lambda obs, *args, **kwargs: (p.motion.hold(obs, 1.0), True)
    d = p.act(replace(o, time=0.6, ee_position=p.observation_target.copy()))
    assert d.stage == "verify" and not d.done
