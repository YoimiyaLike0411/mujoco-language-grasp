"""Focused checks for image tracking without a model service or simulation truth."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from graspbench.perception import ModelServiceError
from policies.fruit_tracking import TrackingPerception


def setup():
    n = 1200
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    phi = i * np.pi * (3 - np.sqrt(5))
    cloud = (
        np.column_stack((np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z))
        * 0.037
    )
    p = TrackingPerception.__new__(TrackingPerception)
    p.cad = SimpleNamespace(models={"apple": (cloud, cKDTree(cloud), None)})
    p.last_tracking_evidence = {}
    center = np.array([0.65, -0.05, 0.436])
    p.tracks = {
        "apple": {
            "center": center.copy(),
            "rotation": np.eye(3),
            "lo": np.array([0.7, 0, 0]),
            "hi": np.array([1, 0.2, 0.2]),
            "history": [],
            "last_time": None,
            "last_output": None,
        }
    }
    p.sam = SimpleNamespace(_unproject_many=lambda c, x, y: c.points[x])

    def observation(offset=(0, 0, 0), time=1, visible=True):
        points = cloud + center + offset
        # A nearby blue distractor must not enter the fruit registration.
        points = np.concatenate((points, cloud + center + np.array([0.008, 0.008, 0])))
        rgb = np.tile(np.array([200, 20, 20]), (2 * n, 1))
        rgb[n:] = [20, 20, 200]
        if not visible:
            rgb[:] = [20, 20, 200]
        c = SimpleNamespace(rgb=rgb[None], points=points)
        return SimpleNamespace(time=time, cameras=(c,))

    return p, observation, center


def test_submillimetre_translation_despite_color_distractor():
    p, obs, c = setup()
    result = p.track_center(obs((0.002, -0.002, 0)), "apple")
    assert np.linalg.norm(result[:2] - c[:2] - [0.002, -0.002]) < 0.0007
    assert result[2] == c[2]


def test_duplicate_snapshot_is_idempotent_and_returns_copy():
    p, obs, c = setup()
    first = p.track_center(obs(), "apple")
    second = p.track_center(obs(visible=False), "apple")
    np.testing.assert_array_equal(first, second)
    second[:] = 0
    np.testing.assert_array_equal(first, p.track_center(obs(), "apple"))
    assert len(p.tracks["apple"]["history"]) == 1


def test_occlusion_rejects_without_advancing_reference():
    p, obs, c = setup()
    with pytest.raises(ModelServiceError, match="visible"):
        p.track_center(obs(visible=False), "apple")
    np.testing.assert_array_equal(p.tracks["apple"]["center"], c)
    assert p.tracks["apple"]["last_time"] is None


def test_missing_reference_never_guesses_identity():
    p, obs, c = setup()
    with pytest.raises(ModelServiceError, match="reference"):
        p.track_center(obs(), "orange")


def test_prediction_is_bounded_and_observed_height_is_preserved():
    p, obs, c = setup()
    p.track_center(obs(), "apple")
    out = p.track_center(obs((0.004, 0, 0), time=1.04), "apple")
    state = p.tracks["apple"]
    assert np.linalg.norm(out[:2] - state["center"][:2]) <= 0.00240001
    assert out[2] == c[2]


def test_lateral_tracking_keeps_full_xy_correction_while_descending(monkeypatch):
    from policies.motion import Motion

    p = Motion.__new__(Motion)
    observation = SimpleNamespace(ee_position=np.array([0.65, -0.05, 0.50]))
    target = np.array([0.652, -0.053, 0.434])
    seen = {}

    def probe(self, o, position, quat, opening, *, speed):
        seen["position"] = position.copy()
        seen["speed"] = speed
        return "command", True

    monkeypatch.setattr(Motion, "pose_move", probe)
    cmd, reached = p.tracked_descent(observation, target, np.array([1.0, 0, 0, 0]), 1.0, speed=0.08)
    np.testing.assert_array_equal(seen["position"][:2], target[:2])
    assert abs(seen["position"][2] - (0.50 - 0.08 * 0.16)) < 1e-10
    assert not reached  # Reaching the intermediate descent waypoint is not a grasp.
    np.testing.assert_array_equal(target, [0.652, -0.053, 0.434])


def test_lateral_tracking_closes_only_near_original_goal(monkeypatch):
    from policies.motion import Motion

    p = Motion.__new__(Motion)
    monkeypatch.setattr(Motion, "pose_move", lambda *a, **kw: ("command", True))
    o = SimpleNamespace(ee_position=np.array([0.65, -0.05, 0.435]))
    assert p.tracked_descent(o, np.array([0.65, -0.05, 0.434]), np.array([1.0, 0, 0, 0]), 1.0)[1]
