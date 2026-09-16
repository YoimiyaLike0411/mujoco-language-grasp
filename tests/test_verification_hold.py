from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from graspbench.types import JointPositionCommand, Observation
from policies.grasp_geometry import Estimate
from policies.student_policy import StudentPolicy


class MotionStub:
    def __init__(self):
        self.last_error = {}

    def hold(self, observation, opening):
        return self.command(observation.joint_position, observation, opening)

    def command(self, target, observation, opening):
        return JointPositionCommand(np.array(target, copy=True), opening)


def test_verification_restores_clear_pose_and_leaves_physical_success_to_runner():
    observation = Observation(
        1.0,
        "Place apple into square tray",
        np.arange(7, dtype=float),
        np.zeros(7),
        np.array([0.5, 0, 0.55]),
        np.array([1.0, 0, 0, 0]),
        1.0,
        (),
    )
    policy = StudentPolicy()
    policy.motion = MotionStub()
    policy.perception = SimpleNamespace(
        held_center=lambda *args, **kwargs: np.array([0.5, 0.0, 0.452])
    )
    # No estimate method: fruit verification must reuse the measured fixed tray.
    policy.actions = [("apple", "square_tray")]
    policy.index = 0
    policy.retry = 0
    policy.grasp = None
    policy.validation = {}
    policy.estimates = {
        "square_tray": Estimate(
            "square_tray", np.array([0.5, 0, 0.410]), np.array([0.22, 0.22, 0.01]), np.eye(3), {}
        )
    }
    policy.enter("verify", observation, "Hand has cleared the released fruit")
    cleared_q = observation.joint_position.copy()
    for i in range(3):
        sagged = replace(observation, time=1.6 + i, joint_position=cleared_q - 0.1 * (i + 1))
        decision = policy.act(sagged)
        np.testing.assert_array_equal(decision.command.joint_position, cleared_q)
        assert not decision.done
    assert policy.stage == "settle"
    assert decision.stage == "verify_place"
    assert policy.validation["cached_static_tray"]
    decision = policy.act(replace(observation, time=50.0, joint_position=cleared_q - 0.5))
    np.testing.assert_array_equal(decision.command.joint_position, cleared_q)
    assert not decision.done and not decision.request_retry
    assert decision.stage == "verify_place"


def test_placed_can_verification_does_not_require_upright_cad(monkeypatch):
    import policies.student_policy as module

    o = Observation(
        1.0,
        "Place can into square tray",
        np.arange(7, dtype=float),
        np.zeros(7),
        np.array([0.5, 0, 0.6]),
        np.array([1.0, 0, 0, 0]),
        1.0,
        (),
    )
    p = StudentPolicy()
    p.motion = MotionStub()

    def no_pose(*args, **kwargs):
        raise AssertionError("Placement verification attempted grasp pose registration")

    p.perception = SimpleNamespace(estimate=no_pose)
    p.actions = [("potted_meat_can", "square_tray")]
    p.index = p.retry = 0
    p.grasp = None
    p.validation = {}
    p.estimates = {
        "square_tray": Estimate(
            "square_tray", np.array([0.3, -0.2, 0.41]), np.array([0.22, 0.22, 0.01]), np.eye(3), {}
        ),
        "potted_meat_can": Estimate(
            "potted_meat_can", np.array([0.6, 0, 0.44]), np.array([0.1, 0.05, 0.07]), np.eye(3), {}
        ),
    }
    monkeypatch.setattr(
        module,
        "placed_center",
        lambda *args: (np.array([0.35, -0.15, 0.46]), {"service": "test_surface"}),
    )
    p.enter("verify", o, "Hand clear")
    d = p.act(replace(o, time=1.6))
    assert p.stage == "settle" and p.retry == 0 and not d.done
    assert d.stage == "verify_place" and p.validation["cached_static_tray"]
