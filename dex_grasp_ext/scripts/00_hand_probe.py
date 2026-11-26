"""Probe the vendored dexterous-hand models: morphology, frames, rendering.

This is the first feasibility check of the extension: it loads the hand models
directly from the vendored MuJoCo Menagerie copy, applies a scripted finger
synergy, and renders the result so the palm/finger frames can be inspected.
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
HANDS = {
    "allegro": ROOT / "assets/menagerie/wonik_allegro/right_hand.xml",
    "leap": ROOT / "assets/menagerie/leap_hand/right_hand.xml",
}


def finger_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="allegro", choices=sorted(HANDS))
    parser.add_argument("--close", type=float, default=0.6, help="synergy fraction [0,1]")
    parser.add_argument("--out", default=str(ROOT / "runs/hand_probe"))
    args = parser.parse_args()

    path = HANDS[args.hand]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"model={path.name} nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name in {"palm", "hand_base", "palm_lower"}:
            print(
                "root body",
                name,
                "pos",
                np.round(model.body_pos[body_id], 4).tolist(),
                "quat",
                np.round(model.body_quat[body_id], 4).tolist(),
            )

    # Scripted synergy: drive every joint towards its upper limit by `close`.
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    target = low + args.close * (high - low)
    data.ctrl[:] = target
    for _ in range(1500):
        mujoco.mj_step(model, data)

    print("\nactuator".ljust(10), "ctrl", "qpos", "fingertip xyz (world)")
    tip_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("ff_tip", "mf_tip", "rf_tip", "th_tip")
    ]
    for actuator_id, name in enumerate(finger_joint_names(model)):
        ctrl = data.ctrl[actuator_id]
        joint_id = model.actuator_trnid[actuator_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        print(f"{name:<10} {ctrl:6.3f} {data.qpos[qpos_adr]:6.3f}")
    for body_id, name in zip(tip_ids, ("ff_tip", "mf_tip", "rf_tip", "th_tip")):
        if body_id >= 0:
            print(f"{name:<8} tip={np.round(data.xpos[body_id], 4).tolist()}")

    # Palm frame: where do the fingers point relative to the root body?
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if root < 0:
        root = 1
    root_mat = data.xmat[root].reshape(3, 3)
    print("\nroot body world pos", np.round(data.xpos[root], 4).tolist())
    print("root body axes (world):")
    for axis, vec in zip("xyz", root_mat.T):
        print(" ", axis, np.round(vec, 4).tolist())

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model.ncam:
        renderer = mujoco.Renderer(model, 480, 640)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        if cam_id < 0:
            cam_id = 0
        renderer.update_scene(data, camera=cam_id)
        image = renderer.render()
        from PIL import Image

        Image.fromarray(image).save(out_dir / f"{args.hand}_closed.png")
        print("\nwrote", out_dir / f"{args.hand}_closed.png")
    else:
        print("\nmodel has no camera; skipping render")


if __name__ == "__main__":
    main()
