"""Run the full closed loop with real SAM3 perception and record a video."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import ArmIK, DexEnv  # noqa: E402
from dexgrasp.config import TABLE_Z  # noqa: E402
from dexgrasp.experiment import VideoRecorder, set_hand_gains  # noqa: E402
from dexgrasp.perception import Sam3Client  # noqa: E402
from dexgrasp.hand import load_calibration  # noqa: E402
from dexgrasp.planning import plan_grasp  # noqa: E402
from dexgrasp.pick_place import run_episode  # noqa: E402
import json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="apple")
    parser.add_argument("--destination", default="square_tray")
    parser.add_argument("--closure", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=20.0)
    parser.add_argument("--x", type=float, default=0.45)
    parser.add_argument("--y", type=float, default=0.02)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766/infer")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out or (ROOT / f"runs/pick_place_{args.object}"))
    env = DexEnv()
    set_hand_gains(env.model, 6.0, 0.15)
    ik = ArmIK(env.model)
    client = Sam3Client(args.endpoint)

    calibration = load_calibration(ROOT / "runs/cage_calibration.json")
    # Prefer the closure/yaw measured by the synthesis sweep for this object.
    plan = plan_grasp(args.object, calibration, yaw_deg=args.yaw, closure_override=args.closure)
    table_path = ROOT / "runs/grasp_synthesis/grasp_table.json"
    if args.closure is None and table_path.exists():
        best = json.loads(table_path.read_text()).get(args.object, {}).get("best")
        if best:
            plan = plan_grasp(
                args.object,
                calibration,
                yaw_deg=best["yaw_deg"],
                closure_override=best["closure"],
            )
    print("grasp plan:", plan.as_dict())

    env.park_objects(keep=args.object)
    env.place_object(args.object, [args.x, args.y, TABLE_Z + 0.06])
    env.reset(settle_steps=500)
    truth = env.object_position(args.object).copy()
    print("ground truth object position:", [round(v, 4) for v in truth.tolist()])

    recorder = VideoRecorder(env.model, width=800, height=600) if args.video else None
    result = run_episode(
        env,
        ik,
        client,
        args.object,
        args.destination,
        plan=plan,
        recorder=recorder,
        out_dir=out_dir,
    )

    def mm(value):
        return "n/a" if value is None else f"{value * 1000:.1f} mm"

    print("stages:", " -> ".join(result.stages))
    print("perception error", mm(result.perception_error_m))
    print(f"lift {result.lift_m * 1000:.1f} mm, finger contacts {result.finger_contacts}")
    print("object drift after close", mm(result.object_drift_after_close_m))
    print("placement offset", mm(result.placement_offset_m))
    print("SAM3 calls:", client.calls, "latencies:", [round(s, 3) for s in result.sam3_latency_s])
    print("SUCCESS" if result.success else f"FAILED: {result.failure}")

    if recorder is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        recorder.save(out_dir / "episode.mp4")
        print("wrote", out_dir / "episode.mp4", len(recorder.frames), "frames")


if __name__ == "__main__":
    main()
