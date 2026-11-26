"""Inspect the mounted hand: flange frame, grasp axis, reach and renders."""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOME_Q = np.array([0.0, -0.45, 0.0, -2.15, 0.0, 1.75, 0.785], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(ROOT / "assets/scene_dex.xml"))
    parser.add_argument("--open-fraction", type=float, default=0.15)
    parser.add_argument("--out", default=str(ROOT / "runs/scene_probe"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    qpos = np.zeros(model.nq)
    qpos[:7] = HOME_Q
    data.qpos[:] = qpos
    # hand actuators sit after the 7 arm actuators
    low = model.actuator_ctrlrange[7:, 0]
    high = model.actuator_ctrlrange[7:, 1]
    data.ctrl[:7] = HOME_Q
    data.ctrl[7:] = low + args.open_fraction * (high - low)
    for _ in range(400):
        mujoco.mj_step(model, data)

    bid = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    palm = bid("palm")
    print("flange/attachment world pos", np.round(data.xpos[bid("attachment")], 4).tolist())
    print("palm world pos           ", np.round(data.xpos[palm], 4).tolist())
    palm_mat = data.xmat[palm].reshape(3, 3)
    for axis, vec in zip("xyz", palm_mat.T):
        print(f"  palm local {axis} -> world {np.round(vec, 4).tolist()}")
    for tip in ("ff_tip", "mf_tip", "rf_tip", "th_tip"):
        index = bid(tip)
        if index >= 0:
            print(f"  {tip:<8} {np.round(data.xpos[index], 4).tolist()}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, 480, 640)
    for camera in ("front", "overhead", "diagonal"):
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(out_dir / f"mount_{camera}.png")
    print("wrote renders to", out_dir)


if __name__ == "__main__":
    main()
