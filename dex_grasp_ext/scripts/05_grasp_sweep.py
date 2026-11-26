"""Sweep objects x finger closures and report which combinations hold."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import ArmIK, DexEnv  # noqa: E402
from dexgrasp.experiment import attempt_grasp, set_hand_gains  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*", default=["apple", "red_cube", "orange"])
    parser.add_argument("--closures", nargs="*", type=float, default=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--hand-kp", type=float, default=6.0)
    parser.add_argument("--out", default=str(ROOT / "runs/grasp_sweep.json"))
    args = parser.parse_args()

    env = DexEnv()
    set_hand_gains(env.model, args.hand_kp, 0.15)
    ik = ArmIK(env.model)

    rows = []
    for name in args.objects:
        for closure in args.closures:
            outcome = attempt_grasp(env, ik, name, closure=closure)
            print(
                f"{name:<16} close={closure:<4} success={int(outcome.success)} "
                f"lift={outcome.lift_m:+.3f} drift={outcome.object_drift_m:.3f} "
                f"contacts={outcome.finger_contacts} {' '.join(outcome.notes)}"
            )
            rows.append(outcome.as_dict())

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    passed = sum(row["success"] for row in rows)
    print(f"\n{passed}/{len(rows)} configurations lifted the object")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
