"""Batch evaluation of the dexterous pick-and-place closed loop."""

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
from dexgrasp.hand import load_calibration  # noqa: E402
from dexgrasp.perception import Sam3Client  # noqa: E402
from dexgrasp.pick_place import prepare_scene, run_episode  # noqa: E402
from dexgrasp.planning import plan_grasp, rank_plans  # noqa: E402

DEFAULT_CASES = [
    ("apple", "square_tray"),
    ("orange", "round_tray"),
    ("red_cube", "square_tray"),
    ("green_cylinder", "round_tray"),
    ("banana", "round_tray"),
    ("potted_meat_can", "square_tray"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None, help="object:destination pairs")
    parser.add_argument("--repeats", type=int, default=1, help="episodes per case")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766/infer")
    parser.add_argument("--out", default=str(ROOT / "runs/episode_eval"))
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.cases:
        cases = [tuple(item.split(":")) for item in args.cases]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DexEnv()
    set_hand_gains(env.model, 6.0, 0.15)
    ik = ArmIK(env.model)
    client = Sam3Client(args.endpoint)
    calibration = load_calibration(ROOT / "runs/cage_calibration.json")
    table_path = ROOT / "runs/grasp_synthesis/grasp_table.json"
    grasp_table = json.loads(table_path.read_text()) if table_path.exists() else {}

    # A few spawn poses, cycled so a single lucky layout cannot carry the run.
    spawns = [(0.45, 0.02), (0.47, 0.06), (0.43, -0.02)]

    rows = []
    for index, (object_name, destination) in enumerate(cases):
        for repeat in range(args.repeats):
            trial = index * args.repeats + repeat
            x, y = spawns[trial % len(spawns)]
            prepare_scene(env, object_name, x=x, y=y, ik=ik)
            best = grasp_table.get(object_name, {}).get("best") or {}
            plan = plan_grasp(
                object_name,
                calibration,
                yaw_deg=float(best.get("yaw_deg", 20.0)),
                closure_override=best.get("closure"),
            )
            result = run_episode(
                env,
                ik,
                client,
                object_name,
                destination,
                plan=plan,
                retry_plans=rank_plans(object_name, calibration, grasp_table, limit=3),
                out_dir=out_dir / f"{trial:02d}_{object_name}",
            )
            record = result.as_dict()
            record["spawn"] = [x, y, TABLE_Z + 0.06]
            record["repeat"] = repeat
            rows.append(record)
            print(
                f"{object_name:<16} -> {destination:<12} #{repeat} "
                f"{'SUCCESS' if result.success else 'FAIL':<7} "
                f"lift={(result.lift_m or 0) * 1000:6.1f} mm "
                f"offset={(result.placement_offset_m or 0) * 1000:6.1f} mm "
                f"perr={(result.perception_error_m or 0) * 1000:4.1f} mm "
                f"closure={result.closure} yaw={result.yaw_deg} "
                f"{'' if result.success else (result.failure or '')}"
            )

    successes = sum(row["success"] for row in rows)
    (out_dir / "eval.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n{successes}/{len(rows)} episodes succeeded")
    print("wrote", out_dir / "eval.json")


if __name__ == "__main__":
    main()
