"""Simulation-based grasp synthesis for the dexterous hand.

For every object the extension sweeps the finger closure and the wrist yaw of
the approach, runs the full perceive -> grasp -> lift pipeline for each
candidate, and keeps the candidate with the best lift (tie-broken by the
smallest object drift).  The resulting table is what the closed-loop episodes
consume, so the parameters are measured rather than hand-tuned.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import ArmIK, DexEnv  # noqa: E402
from dexgrasp.config import TABLE_Z  # noqa: E402
from dexgrasp.experiment import set_hand_gains  # noqa: E402
from dexgrasp.perception import Sam3Client  # noqa: E402
from dexgrasp.hand import load_calibration  # noqa: E402
from dexgrasp.perception import grasp_radius  # noqa: E402
from dexgrasp.planning import plan_grasp  # noqa: E402
from dexgrasp.pick_place import pick_and_lift, prepare_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*", default=["apple", "red_cube", "orange"])
    parser.add_argument("--closures", nargs="*", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--yaws", nargs="*", type=float, default=[-30.0, 0.0, 30.0])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766/infer")
    parser.add_argument("--out", default=str(ROOT / "runs/grasp_synthesis"))
    args = parser.parse_args()

    env = DexEnv()
    set_hand_gains(env.model, 6.0, 0.15)
    ik = ArmIK(env.model)
    client = Sam3Client(args.endpoint)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(ROOT / "runs/cage_calibration.json")
    table: dict[str, dict] = {}
    for name in args.objects:
        env.park_objects(keep=name)
        env.place_object(name, [0.45, 0.02, TABLE_Z + 0.06])
        env.reset(settle_steps=500)

        rows = []
        for closure in args.closures:
            for yaw in args.yaws:
                prepare_scene(env, name)
                plan = plan_grasp(name, calibration, yaw_deg=yaw, closure_override=closure)
                result = pick_and_lift(env, ik, client, name, plan=plan)
                rows.append(result.as_dict())
                print(
                    f"{name:<16} closure={closure:<4} yaw={yaw:>6} "
                    f"lift={result.lift_m * 1000:7.1f} mm "
                    f"drift={(result.object_drift_after_close_m or 0.0) * 1000:6.1f} mm "
                    f"contacts={result.finger_contacts:<3} "
                    f"{'OK' if result.lifted else (result.failure or 'no lift')}"
                )
        best = max(rows, key=lambda row: (row["lift_m"], -(row["object_drift_after_close_m"] or 0.0)))
        successful = [row for row in rows if row["lift_m"] > 0.04]
        table[name] = {
            "best": {
                "closure": best["closure"],
                "yaw_deg": best["yaw_deg"],
                "lift_m": best["lift_m"],
                "drift_m": best["object_drift_after_close_m"],
            },
            "success_rate": len(successful) / len(rows),
            "trials": rows,
        }
        print(
            f"--> {name}: best closure={best['closure']} yaw={best['yaw_deg']} "
            f"lift={best['lift_m'] * 1000:.1f} mm, {len(successful)}/{len(rows)} candidates lifted\n"
        )

    (out_dir / "grasp_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print("wrote", out_dir / "grasp_table.json")


if __name__ == "__main__":
    main()
