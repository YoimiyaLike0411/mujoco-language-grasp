"""Fast camera-only fruit tracking, initialized by real SAM3/CAD grounding."""

import numpy as np
from graspbench.perception import ModelServiceError
from .grounding import Perception


class TrackingPerception(Perception):
    def __init__(self, model):
        super().__init__(model)
        self.tracks = {}
        self.last_tracking_evidence = {}

    def estimate(self, observation, names):
        result = super().estimate(observation, names)
        for name, e in result.items():
            if name not in ("apple", "orange"):
                continue
            colors = []
            for camera in observation.cameras:
                yy, xx = np.indices(camera.depth.shape)
                points = self.sam._unproject_many(camera, xx.ravel(), yy.ravel())
                rgb = camera.rgb.reshape(-1, 3).astype(float)
                total = rgb.sum(1)
                keep = (
                    np.isfinite(points).all(1)
                    & (np.linalg.norm(points - e.center, axis=1) < 0.045)
                    & (points[:, 2] > 0.403)
                    & (total > 20)
                )
                colors.append(rgb[keep] / total[keep, None])
            colors = np.concatenate(colors)
            if len(colors) < 30:
                raise ModelServiceError("Insufficient fruit appearance for tracking")
            lo = np.clip(np.percentile(colors, 5, axis=0) - 0.03, 0, 1)
            hi = np.clip(np.percentile(colors, 95, axis=0) + 0.03, 0, 1)
            self.tracks[name] = {
                "center": e.center.copy(),
                "rotation": e.rotation.copy(),
                "lo": lo,
                "hi": hi,
                "history": [],
                "last_time": None,
                "last_output": None,
            }
        return result

    def track_center(self, observation, name):
        if name not in self.tracks:
            raise ModelServiceError("Fruit tracker has no model-grounded reference")
        state = self.tracks[name]
        if state["last_time"] == observation.time:
            return state["last_output"].copy()
        reference = state["center"]
        clouds = []
        for camera in observation.cameras:
            rgb = camera.rgb.astype(float)
            total = rgb.sum(2)
            chroma = rgb / np.maximum(total[:, :, None], 1.0)
            mask = (total > 20) & (chroma >= state["lo"]).all(2) & (chroma <= state["hi"]).all(2)
            yy, xx = np.nonzero(mask)
            points = self.sam._unproject_many(camera, xx, yy)
            keep = np.isfinite(points).all(1) & (
                np.linalg.norm(points[:, :2] - reference[:2], axis=1) < 0.05
            )
            keep &= (points[:, 2] > 0.403) & (points[:, 2] < reference[2] + 0.045)
            clouds.append(points[keep])
        points = np.concatenate(clouds) if clouds else np.empty((0, 3))
        if len(points) < 30:
            raise ModelServiceError("Insufficient visible fruit surface for tracking")
        points = points[np.linspace(0, len(points) - 1, min(400, len(points))).astype(int)]
        cloud, tree, _ = self.cad.models[name]
        rotation = state["rotation"]
        center = reference.copy()
        for _ in range(10):
            distances, indices = tree.query((points - center) @ rotation)
            keep = (distances <= np.percentile(distances, 85)) & (distances < 0.008)
            if keep.sum() < 25:
                raise ModelServiceError("Fruit tracking surface mismatch")
            center[:2] = (points[keep] - cloud[indices[keep]] @ rotation.T)[:, :2].mean(0)
        distances, _ = tree.query((points - center) @ rotation)
        residual = float(np.median(distances))
        if residual > 0.004 or np.linalg.norm(center[:2] - reference[:2]) > 0.01:
            raise ModelServiceError("Unreliable fruit tracking update")
        history = [(t, c) for t, c in state["history"] if observation.time - t < 0.4]
        history.append((observation.time, center[:2].copy()))
        history = history[-6:]
        velocity = np.zeros(2)
        if len(history) >= 2:
            times = np.array([h[0] for h in history])
            times -= times.mean()
            coords = np.array([h[1] for h in history])
            denom = float(np.dot(times, times))
            if denom > 1e-6:
                velocity = (times[:, None] * (coords - coords.mean(0))).sum(0) / denom
            velocity *= min(1.0, 0.03 / max(np.linalg.norm(velocity), 1e-9))
        predicted = center.copy()
        predicted[:2] += velocity * 0.08
        state.update(
            center=center, history=history, last_time=observation.time, last_output=predicted
        )
        self.last_tracking_evidence = {
            "observation_time": float(observation.time),
            "target": name,
            "points": len(points),
            "surface_residual_m": residual,
            "center": center.tolist(),
            "predicted_center": predicted.tolist(),
            "velocity_xy": velocity.tolist(),
        }
        return predicted.copy()
