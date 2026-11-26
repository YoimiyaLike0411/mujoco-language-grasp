"""Check the extension's perception chain against ground truth.

Places an object at a known pose, renders the scene camera, asks the SAM3
service for a mask, unprojects it and reports the world-frame error.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import DexEnv  # noqa: E402
from dexgrasp import ArmIK  # noqa: E402
from dexgrasp.config import HAND_OPEN, PERCEPTION_PARK_XYZ, TABLE_Z  # noqa: E402
from dexgrasp.experiment import move_tcp, set_hand_gains  # noqa: E402
from dexgrasp.hand import HandSynergy  # noqa: E402
from dexgrasp.perception import (  # noqa: E402
    CATALOGUE,
    PerceptionError,
    Sam3Client,
    detect_object,
    render_camera,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*", default=["apple", "red_cube", "mustard_bottle"])
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766/infer")
    parser.add_argument("--out", default=str(ROOT / "runs/perception_check.json"))
    args = parser.parse_args()

    env = DexEnv()
    set_hand_gains(env.model, 6.0, 0.15)
    ik = ArmIK(env.model)
    client = Sam3Client(args.endpoint)
    rows = []
    for name in args.objects:
        env.park_objects(keep=name)
        env.place_object(name, [0.45, 0.02, TABLE_Z + 0.06])
        env.reset(settle_steps=400)
        truth = env.object_position(name).copy()
        # Move the hand out of the overhead view before taking the shot.
        move_tcp(ik, env, PERCEPTION_PARK_XYZ, HandSynergy(HAND_OPEN), steps=400)
        camera = render_camera(env.model, env.data, args.camera)
        try:
            detection = detect_object(client, camera, name)
            error = float(np.linalg.norm(detection.position - truth))
            print(
                f"{name:<16} truth={np.round(truth, 4).tolist()} "
                f"sam3={np.round(detection.position, 4).tolist()} "
                f"err={error * 1000:.1f} mm score={detection.score:.3f} "
                f"latency={detection.latency_s * 1000:.0f} ms "
                f"pixels={detection.geometry_pixels}"
            )
            rows.append(
                {
                    "object": name,
                    "truth": truth.tolist(),
                    "sam3": detection.position.tolist(),
                    "error_m": error,
                    "score": detection.score,
                    "latency_s": detection.latency_s,
                    "geometry_pixels": detection.geometry_pixels,
                }
            )
        except PerceptionError as exc:
            print(f"{name:<16} FAILED: {exc}")
            rows.append({"object": name, "error": str(exc)})

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    print("catalogue size:", len(CATALOGUE), "| wrote", out_path)


if __name__ == "__main__":
    main()
