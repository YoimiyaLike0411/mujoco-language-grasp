"""Record a dexterous grasp: approach, enclose, lift, and report success."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import ArmIK, DexEnv  # noqa: E402
from dexgrasp.experiment import VideoRecorder, attempt_grasp, set_hand_gains  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="apple")
    parser.add_argument("--closure", type=float, default=0.6)
    parser.add_argument("--x", type=float, default=0.44)
    parser.add_argument("--y", type=float, default=0.02)
    parser.add_argument("--hand-kp", type=float, default=6.0)
    parser.add_argument("--camera", default="diagonal")
    parser.add_argument("--out", default=str(ROOT / "runs/demo_grasp"))
    args = parser.parse_args()

    env = DexEnv()
    set_hand_gains(env.model, args.hand_kp, 0.15)
    ik = ArmIK(env.model)
    recorder = VideoRecorder(env.model)

    outcome = attempt_grasp(
        env,
        ik,
        args.object,
        x=args.x,
        y=args.y,
        closure=args.closure,
        recorder=recorder,
    )

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder.save(out_dir / f"{args.object}_close{args.closure:.2f}.mp4")
    (out_dir / "result.json").write_text(json.dumps(outcome.as_dict(), indent=2) + "\n")

    print(
        f"{args.object} closure={args.closure}: success={outcome.success} "
        f"lift={outcome.lift_m:+.3f} m drift={outcome.object_drift_m:.3f} m "
        f"finger contacts={outcome.finger_contacts}"
    )
    print("frames:", len(recorder.frames))


if __name__ == "__main__":
    main()
