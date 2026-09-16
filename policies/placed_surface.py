"""Verify object location from live segmentation surfaces, without upright pose fitting."""

import numpy as np
from graspbench.perception import ModelServiceError


def placed_center(perception, observation, name, near):
    clouds = []
    views = []
    for camera in observation.cameras:
        response, _ = perception.sam._call_sam3(camera.rgb, [perception.sam.SAM_PROMPTS[name]])
        result = response["results"][0]
        scores = np.asarray(result.get("scores", []))
        payload = result.get("masks", {})
        count = int(payload.get("count", 0))
        if not count or scores.shape != (count,):
            continue
        mask = perception.sam._decode_masks(payload["packed"], count, camera.depth.shape)[
            int(np.argmax(scores))
        ]
        yy, xx = np.nonzero(mask)
        points = perception.sam._unproject_many(camera, xx, yy)
        valid = np.isfinite(points).all(1) & (points[:, 2] > 0.417) & (points[:, 2] < 0.8)
        valid &= np.linalg.norm(points[:, :2] - np.asarray(near)[:2], axis=1) < 0.22
        points = points[valid]
        if len(points) < 12:
            continue
        clouds.append(points)
        views.append({"camera": camera.name, "points": len(points), "score": float(scores.max())})
    points = np.concatenate(clouds) if clouds else np.empty((0, 3))
    if len(points) < 30:
        raise ModelServiceError("Insufficient visible placed-object surface")
    lower, upper = np.percentile(points, [2, 98], axis=0)
    center = (lower + upper) / 2
    evidence = {
        "service": "sam3_two_view_placed_surface",
        "views": views,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "center": center.tolist(),
    }
    return center, evidence
