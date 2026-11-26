"""Calibrate the hand's grasp cage: closure -> (enclosed radius, centre)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexgrasp import DexEnv  # noqa: E402
from dexgrasp.hand import cage_calibration  # noqa: E402
from dexgrasp.experiment import set_hand_gains  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs/cage_calibration.json"))
    args = parser.parse_args()

    env = DexEnv()
    set_hand_gains(env.model, 6.0, 0.15)
    env.park_objects()
    env.reset(settle_steps=200)

    rows = cage_calibration(env.model, env.data)
    for row in rows:
        print(
            f"closure={row['fraction']:.2f}  enclosed_radius={row['radius_m'] * 1000:6.1f} mm  "
            f"centre_local={[round(v, 4) for v in row['centre_local']]}"
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    # Regress radius on closure so the planner can invert it analytically.
    fractions = np.array([row["fraction"] for row in rows])
    radii = np.array([row["radius_m"] for row in rows])
    slope, intercept = np.polyfit(fractions, radii, 1)
    print(f"\nradius ≈ {intercept * 1000:.1f} mm + {slope * 1000:.1f} mm * closure")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
