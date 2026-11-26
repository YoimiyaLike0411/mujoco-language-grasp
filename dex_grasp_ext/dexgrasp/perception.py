"""Real-model perception for the extension: MuJoCo RGB-D + the SAM3 service.

The extension talks to the same SAM3 HTTP service as the course baseline, but
implements its own client, camera model and unprojection so that nothing in the
course project has to be imported or modified.

The service contract (verified against the course service) is:

    POST {"image_jpeg_b64": str, "prompts": [str, ...]}
    ->   {"results": [{"scores": [...], "boxes": [[x1,y1,x2,y2], ...],
                       "masks": {"count": n, "packed": "<base64 bits>"}}, ...]}

Masks are packed with ``np.packbits(..., bitorder="little")`` over a
``(count, height, width)`` boolean array.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image

DEFAULT_ENDPOINT = "http://127.0.0.1:8766/infer"

# Catalogue copied from the course object list (public task definition): the
# extension needs the same prompt strings / half heights for its own pipeline.
CATALOGUE: dict[str, dict] = {
    "red_cube": {"prompt": "small red cube", "half_height": 0.026, "grasp_radius": 0.034},
    "green_cylinder": {"prompt": "green round object", "half_height": 0.031, "grasp_radius": 0.028},
    "blue_box": {"prompt": "blue brick", "half_height": 0.026, "grasp_radius": 0.030},
    "banana": {"prompt": "yellow banana", "half_height": 0.020, "grasp_radius": 0.026},
    "apple": {"prompt": "red apple", "half_height": 0.038, "grasp_radius": 0.038},
    "orange": {"prompt": "small orange round object", "half_height": 0.038, "grasp_radius": 0.038},
    "mustard_bottle": {
        "prompt": "small green object",
        "half_height": 0.067,
        "surface_percentile": 98.0,
        "grasp_radius": 0.042,
    },
    "potted_meat_can": {
        "prompt": "small blue object",
        "half_height": 0.042,
        "grasp_radius": 0.046,
    },
    "scissors": {"prompt": "scissors", "half_height": 0.009, "grasp_radius": 0.050},
    "marker": {"prompt": "black marker pen", "half_height": 0.011, "grasp_radius": 0.016},
    "square_tray": {"prompt": "yellow square tray", "half_height": 0.0, "surface_percentile": 70.0},
    "round_tray": {"prompt": "purple round tray", "half_height": 0.0, "surface_percentile": 70.0},
}


def grasp_radius(name: str) -> float:
    """Effective cage radius the planner should aim for on `name`."""
    spec = CATALOGUE.get(name)
    if spec is None:
        raise PerceptionError(f"unknown object {name!r}")
    return float(spec.get("grasp_radius", spec["half_height"]))


class PerceptionError(RuntimeError):
    """Raised when perception cannot produce a trustworthy estimate."""


@dataclass
class Detection:
    name: str
    position: np.ndarray
    score: float
    mask_pixels: int
    geometry_pixels: int
    latency_s: float
    box_xyxy: tuple[float, float, float, float]
    surface_height_m: float


class Sam3Client:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout_s: float = 30.0):
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.calls = 0

    def detect(self, rgb: np.ndarray, prompts: list[str]) -> tuple[list[dict], float]:
        prepared = np.asarray(rgb, dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(prepared).save(buffer, format="JPEG", quality=92)
        payload = {
            "image_jpeg_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "prompts": prompts,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PerceptionError(f"SAM3 request failed: {exc}") from exc
        latency = time.perf_counter() - started
        self.calls += 1
        results = parsed.get("results")
        if not isinstance(results, list) or len(results) != len(prompts):
            raise PerceptionError("SAM3 response does not match the prompt list")
        return results, latency


def decode_masks(packed_b64: str, count: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    packed = np.frombuffer(base64.b64decode(packed_b64), dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")
    required = count * height * width
    if bits.size < required:
        raise PerceptionError("SAM3 mask payload is shorter than expected")
    return bits[:required].reshape(count, height, width).astype(bool)


@dataclass
class Camera:
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    fovy_degrees: float

    @property
    def focal(self) -> float:
        height = self.depth.shape[0]
        return 0.5 * height / np.tan(np.deg2rad(self.fovy_degrees) * 0.5)

    def unproject(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        depth = self.depth[rows, cols].astype(np.float64)
        height, width = self.depth.shape
        focal = self.focal
        points_camera = np.column_stack(
            [
                (cols - (width - 1) * 0.5) * depth / focal,
                ((height - 1) * 0.5 - rows) * depth / focal,
                -depth,
            ]
        )
        return self.position + points_camera @ self.rotation.T


def render_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    *,
    height: int = 256,
    width: int = 256,
) -> Camera:
    """Render RGB + depth for a scene camera and return its pinhole model."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise PerceptionError(f"unknown camera {camera_name!r}")
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.close()
    return Camera(
        name=camera_name,
        rgb=rgb,
        depth=depth,
        position=data.cam_xpos[camera_id].copy(),
        rotation=data.cam_xmat[camera_id].reshape(3, 3).copy(),
        fovy_degrees=float(model.cam_fovy[camera_id]),
    )


def detect_object(
    client: Sam3Client,
    camera: Camera,
    name: str,
    *,
    workspace: tuple[float, float, float, float] = (0.20, 0.78, -0.30, 0.30),
    min_geometry_pixels: int = 10,
    surface_percentile: float | None = None,
) -> Detection:
    """Segment `name` in `camera` and return its world-frame centre."""
    spec = CATALOGUE.get(name)
    if spec is None:
        raise PerceptionError(f"unknown object {name!r}")
    if surface_percentile is None:
        surface_percentile = float(spec.get("surface_percentile", 97.0))
    results, latency = client.detect(camera.rgb, [spec["prompt"]])
    result = results[0]
    scores = np.asarray(result.get("scores", []), dtype=np.float64)
    boxes = np.asarray(result.get("boxes", []), dtype=np.float64)
    masks_payload = result.get("masks") or {}
    count = int(masks_payload.get("count", 0))
    if count <= 0 or scores.shape != (count,) or boxes.shape != (count, 4):
        raise PerceptionError(f"SAM3 found no valid instance of {name}")
    masks = decode_masks(str(masks_payload.get("packed", "")), count, camera.rgb.shape[:2])
    best = int(np.argmax(scores))
    rows, cols = np.nonzero(masks[best])
    if len(rows) < min_geometry_pixels:
        raise PerceptionError(f"SAM3 mask for {name} is too small")

    points = camera.unproject(cols, rows)
    x_low, x_high, y_low, y_high = workspace
    keep = (
        (points[:, 0] >= x_low)
        & (points[:, 0] <= x_high)
        & (points[:, 1] >= y_low)
        & (points[:, 1] <= y_high)
        & (points[:, 2] >= 0.405)
        & (points[:, 2] <= 0.80)
    )
    points = points[keep]
    if len(points) < min_geometry_pixels:
        raise PerceptionError(f"SAM3 mask for {name} has no valid RGB-D geometry")

    xy = np.median(points[:, :2], axis=0)
    surface_height = float(np.percentile(points[:, 2], surface_percentile))
    centre = np.array([xy[0], xy[1], surface_height - spec["half_height"]])
    return Detection(
        name=name,
        position=centre,
        score=float(scores[best]),
        mask_pixels=int(len(rows)),
        geometry_pixels=int(len(points)),
        latency_s=latency,
        box_xyxy=tuple(float(value) for value in boxes[best]),
        surface_height_m=surface_height,
    )
