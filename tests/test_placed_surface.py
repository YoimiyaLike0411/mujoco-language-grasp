"""Placement verification must tolerate orientation changes without fitting a grasp."""

from types import SimpleNamespace
import numpy as np
import pytest
from graspbench.perception import ModelServiceError
from policies.placed_surface import placed_center


def setup():
    x, y = np.meshgrid(np.linspace(-0.01, 0.01, 10), np.linspace(-0.01, 0.01, 10))
    cloud = np.column_stack((x.ravel(), y.ravel(), np.zeros(100)))

    def camera(center):
        return SimpleNamespace(
            name="view", rgb=np.ones((10, 10, 3)), depth=np.ones((10, 10)), points=cloud + center
        )

    sam = SimpleNamespace(
        SAM_PROMPTS={"potted_meat_can": "can"},
        _call_sam3=lambda *a: (
            {"results": [{"scores": [0.9], "masks": {"count": 1, "packed": "mask"}}]},
            0,
        ),
        _decode_masks=lambda *a: np.ones((1, 10, 10), dtype=bool),
        _unproject_many=lambda c, x, y: c.points[y * 10 + x],
    )
    p = SimpleNamespace(sam=sam)  # Deliberately no CAD registration available.
    return p, camera


def test_two_partial_views_verify_position_without_fitting_pose():
    p, camera = setup()
    o = SimpleNamespace(cameras=(camera([0.35, -0.1, 0.47]), camera([0.37, -0.1, 0.47])))
    center, e = placed_center(p, o, "potted_meat_can", [0.33, -0.15, 0.41])
    np.testing.assert_allclose(center, [0.36, -0.1, 0.47], atol=1e-12)
    assert len(e["views"]) == 2


def test_table_invalid_depth_and_distant_objects_are_excluded():
    p, camera = setup()
    c = camera([0.35, -0.1, 0.47])
    c.points[:10] = [0.35, -0.1, 0.4]
    c.points[10:20] = np.nan
    c.points[20:30] = [0.8, 0.3, 0.5]
    center, e = placed_center(
        p, SimpleNamespace(cameras=(c,)), "potted_meat_can", [0.33, -0.15, 0.41]
    )
    assert e["views"][0]["points"] == 70
    assert 0.34 < center[0] < 0.36 and -0.11 < center[1] < -0.09 and center[2] == 0.47


def test_missing_visible_surface_rejects_instead_of_guessing_tray_center():
    p, camera = setup()
    c = camera([0.35, -0.1, 0.4])
    with pytest.raises(ModelServiceError, match="visible"):
        placed_center(p, SimpleNamespace(cameras=(c,)), "potted_meat_can", [0.33, -0.15, 0.41])
