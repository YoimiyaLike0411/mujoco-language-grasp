"""Language and real SAM3 observations, separate from robot control."""

from __future__ import annotations

import re

import numpy as np

from graspbench.config import CONTAINER_SPEC_BY_NAME, OBJECT_SPEC_BY_NAME
from graspbench.perception import FoundationModelPerception, ModelServiceError

from .cad_geometry import CadGeometry, tray_center
from .grasp_geometry import Estimate

TERMS = {
    "red_cube": ("red cube", "red block", "红色方块", "红方块", "红色立方体"),
    "green_cylinder": ("green cylinder", "绿色圆柱", "绿圆柱"),
    "blue_box": ("blue box", "blue rectangular", "blue brick", "蓝色长方体", "蓝色方块"),
    "banana": ("banana", "香蕉"),
    "apple": ("apple", "苹果"),
    "orange": ("orange", "橙子"),
    "mustard_bottle": ("mustard", "芥末"),
    "potted_meat_can": ("meat can", "packaged meat", "午餐肉", "罐头"),
    "square_tray": ("square tray", "square container", "yellow tray", "方盘", "方形盘", "黄盘"),
    "round_tray": ("round tray", "round container", "purple tray", "圆盘", "圆形盘", "紫盘"),
}


def parse_actions(instruction):
    """Explicit names/category clauses; never task IDs, labels or seeds."""
    text = instruction.casefold()
    actions = []
    for clause in re.split(r"[;；。.!！\n]|\bwhile\b", text):
        places = [
            key for key in CONTAINER_SPEC_BY_NAME if any(term in clause for term in TERMS[key])
        ]
        picks = [
            key
            for key in TERMS
            if key in OBJECT_SPEC_BY_NAME and any(term in clause for term in TERMS[key])
        ]
        if "水果" in clause or "fruit" in clause:
            picks = list(dict.fromkeys(picks + ["apple", "orange"]))
        if "包装食品" in clause or "packaged food" in clause:
            picks = list(dict.fromkeys(picks + ["mustard_bottle", "potted_meat_can"]))
        if picks and len(places) == 1:
            actions.extend((pick, places[0]) for pick in picks)
        elif picks and len(places) > 1:
            parts = re.split(
                r",|，|\band\b(?=\s+(?:packaged|fruit|the packaged|the fruit))", clause
            )
            if len(parts) <= 1:
                raise ValueError("Ambiguous relationships; structured VLM grounding needed")
            for part in parts:
                actions.extend(parse_actions(part))
    actions = list(dict.fromkeys(actions))
    if not actions:
        raise ValueError("No unambiguous pick/place instruction found")
    if len({p for p, d in actions}) != len(actions):
        raise ValueError("Conflicting destinations for one object")
    return actions


class Perception:
    def __init__(self, model=None):
        self.cad = CadGeometry(model) if model is not None else None
        self.sam = FoundationModelPerception()

    def held_center(self, observation, name, near=None):
        """Estimate a held fruit's center from both live RGB-D views."""
        from scipy.optimize import least_squares

        clouds = []
        for camera in observation.cameras:
            response, _ = self.sam._call_sam3(camera.rgb, [self.sam.SAM_PROMPTS[name]])
            result = response["results"][0]
            scores = np.asarray(result.get("scores", []))
            payload = result.get("masks", {})
            count = int(payload.get("count", 0))
            if not count or scores.shape != (count,):
                continue
            mask = self.sam._decode_masks(payload["packed"], count, camera.depth.shape)[
                int(np.argmax(scores))
            ]
            rows, cols = np.nonzero(mask)
            points = self.sam._unproject_many(camera, cols, rows)
            valid = np.isfinite(points).all(1) & (points[:, 2] > 0.403) & (points[:, 2] < 0.8)
            reference = observation.ee_position if near is None else np.asarray(near)
            valid &= np.linalg.norm(points - reference, axis=1) < (0.12 if near is None else 0.20)
            clouds.append(points[valid])
        points = np.concatenate(clouds) if clouds else np.empty((0, 3))
        if len(points) < 30:
            raise ModelServiceError("Insufficient visible held-fruit surface")
        radius = 0.037 if name == "apple" else 0.0365
        fit = least_squares(
            lambda c: np.linalg.norm(points - c, axis=1) - radius,
            points.mean(0),
            loss="soft_l1",
            f_scale=0.002,
            max_nfev=100,
        )
        residual = float(np.median(abs(np.linalg.norm(points - fit.x, axis=1) - radius)))
        if residual > 0.006:
            raise ModelServiceError("Inconsistent held-fruit geometry")
        return fit.x

    def language_plan(self, observation):
        """Use the starter's validated VLM adapter for unresolved instructions."""
        from graspbench.vlm import OpenAICompatibleVLM

        camera = next(c for c in observation.cameras if c.name == "overhead")
        vlm = OpenAICompatibleVLM()  # fails explicitly if service credentials are absent
        detections, evidence = self.sam.detect_scene(camera)
        plan = vlm.plan_task(observation.instruction, camera, detections, evidence)
        if any(a.place_id is None for a in plan.actions):
            raise ModelServiceError("This controller requires a placement destination")
        return [(a.pick_id, a.place_id) for a in plan.actions]

    def estimate(self, observation, names):
        camera = next(c for c in observation.cameras if c.name == "overhead")
        prompts = [self.sam.SAM_PROMPTS[name] for name in names]
        response, latency = self.sam._call_sam3(camera.rgb, prompts)
        results = response.get("results", [])
        if len(results) != len(names):
            raise ModelServiceError("SAM3 prompt/result mismatch")
        estimates = {}
        for name, prompt, result in zip(names, prompts, results):
            scores = np.asarray(result.get("scores", []))
            payload = result.get("masks", {})
            count = int(payload.get("count", 0))
            if not count or scores.shape != (count,):
                raise ModelServiceError(f"SAM3 missing {name}")
            masks = self.sam._decode_masks(payload["packed"], count, camera.depth.shape)
            mask = masks[int(np.argmax(scores))]
            rows, cols = np.nonzero(mask)
            points = self.sam._unproject_many(camera, cols, rows)
            valid = np.isfinite(points).all(1) & (points[:, 2] > 0.403) & (points[:, 2] < 0.8)
            valid &= (points[:, 0] > 0.17) & (points[:, 0] < 0.84) & (abs(points[:, 1]) < 0.33)
            points = points[valid]
            if len(points) < 12:
                raise ModelServiceError(f"Insufficient depth points for {name}")
            evidence = {
                "service": "sam3",
                "prompt": prompt,
                "score": float(scores.max()),
                "mask_pixels": int(mask.sum()),
                "geometry_pixels": len(points),
                "latency_s": latency,
            }
            if name in CONTAINER_SPEC_BY_NAME:
                xy = tray_center(name, points)
                estimates[name] = Estimate(
                    name, np.r_[xy, 0.410], np.array([0.22, 0.22, 0.01]), np.eye(3), evidence
                )
                continue
            top = float(np.percentile(points[:, 2], 98))
            top_points = points[points[:, 2] > top - 0.006]
            if len(top_points) < 10:
                top_points = points
            xy = (
                np.percentile(top_points[:, :2], 3, axis=0)
                + np.percentile(top_points[:, :2], 97, axis=0)
            ) / 2
            shape = OBJECT_SPEC_BY_NAME[name]
            center = np.r_[xy, top - shape.half_height]
            best = None
            for angle in np.linspace(0, np.pi / 2, 181):
                axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                coords = top_points[:, :2] @ axes
                sizes = np.percentile(coords, 98, axis=0) - np.percentile(coords, 2, axis=0)
                if best is None or np.prod(sizes) < best[0]:
                    best = (np.prod(sizes), axes, sizes)
            rot = np.eye(3)
            rot[:2, :2] = best[1]
            sizes = np.r_[best[2], 2 * shape.half_height]
            if name in ("apple", "orange"):
                sizes[:2] = 0.076 if name == "apple" else 0.074
            if self.cad is not None and name in (
                "mustard_bottle",
                "potted_meat_can",
                "banana",
                "apple",
                "orange",
            ):
                if name in ("banana", "apple", "orange"):
                    # A side view constrains roll/pitch that overhead top surfaces
                    # alone leave ambiguous for a curved, resting object.
                    front = next((c for c in observation.cameras if c.name == "front"), None)
                    if front is not None:
                        response, side_latency = self.sam._call_sam3(front.rgb, [prompt])
                        result = response["results"][0]
                        scores = np.asarray(result.get("scores", []))
                        payload = result.get("masks", {})
                        count = int(payload.get("count", 0))
                        if count and scores.shape == (count,):
                            masks = self.sam._decode_masks(
                                payload["packed"], count, front.depth.shape
                            )
                            rows, cols = np.nonzero(masks[int(np.argmax(scores))])
                            side = self.sam._unproject_many(front, cols, rows)
                            valid = (
                                np.isfinite(side).all(1)
                                & (side[:, 2] > 0.403)
                                & (side[:, 2] < 0.8)
                                & (side[:, 0] > 0.17)
                                & (side[:, 0] < 0.84)
                                & (abs(side[:, 1]) < 0.33)
                            )
                            side = side[valid]
                            if len(side) > 20:
                                points = np.concatenate((points, side))
                                evidence["front_points"] = len(side)
                                evidence["front_latency_s"] = side_latency
                registration = self.cad.register(name, points, center)
                if registration is not None:
                    center, rot, sizes, score = registration
                    evidence["cad_residual_m"] = score
                    if score > 0.008:
                        raise ModelServiceError(
                            f"Unreliable CAD registration for {name}: {score:.4f} m"
                        )
            estimates[name] = Estimate(name, center, sizes, rot, evidence)
            if name == "banana" and self.cad is not None:
                item = estimates[name]
                item.grasp_center, item.grasp_rotation, item.grasp_width = self.cad.banana_grasp(
                    center, rot
                )
                # Refine the contact section directly in the observed cloud.
                # This avoids amplifying a small CAD tilt error at the curved edge.
                major = item.grasp_rotation[:, 0]
                minor = item.grasp_rotation[:, 1]
                band = points[abs((points - item.grasp_center) @ major) < 0.012]
                if len(band) >= 20:
                    transverse = (band - item.grasp_center) @ minor
                    correction = (np.percentile(transverse, 2) + np.percentile(transverse, 98)) / 2
                    item.grasp_center += minor * correction
                    lower, upper = np.percentile(band[:, 2], [5, 95])
                    item.grasp_center[2] = lower + 0.375 * (upper - lower)
                    evidence["contact_section_points"] = len(band)
                # The end section clears the palm and permits a low supporting
                # pinch; the middle section slipped under asynchronous control.
                item.grasp_center += item.grasp_rotation[:, 0] * 0.020
                item.grasp_center[2] = 0.414
        return estimates
