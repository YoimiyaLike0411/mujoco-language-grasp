"""Build the arm+dexterous-hand MuJoCo assets for the extension.

The course ships a Franka Panda (MuJoCo Menagerie) with its parallel gripper, and
MuJoCo Menagerie ships standalone dexterous hands.  Nothing in either project is
modified by this extension.  Instead this script generates, inside the extension
only:

* ``assets/robot/<hand>_top.xml``   - the hand's asset/default/actuator blocks,
  included at the top level of the scene,
* ``assets/robot/<hand>_body.xml``  - the hand's body subtree, included inside
  the Franka flange body (MuJoCo cannot nest a full model inside a body), and
* ``assets/robot/panda_<hand>.xml`` - a copy of the course arm with the hand
  mounted on the flange.

Because MuJoCo resolves relative asset paths against the file that declares them
and the generated files live in a different directory, mesh paths are written as
absolute paths.  Re-run this script after moving the checkout.
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Blocks that must be declared before the arm (defaults/assets) and blocks that
# must come after it (actuators/contact), so that actuator indices stay
# "7 arm joints first, then the hand".
PRE_BLOCKS = ("asset", "default")
POST_BLOCKS = ("actuator", "contact", "equality", "tendon")
HANDS = {
    # hand name -> stock Menagerie model
    "allegro": ROOT / "assets/menagerie/wonik_allegro/right_hand.xml",
    "leap": ROOT / "assets/menagerie/leap_hand/right_hand.xml",
}

# Flange -> hand mount.  The hand must be rotated so that its finger axis
# (palm local +Z) coincides with the tool axis of the course gripper (flange
# local -Z), and shifted forward so that the palm body clears the flange.
# Solved numerically by work/mount_probe/solve_mount.py.
MOUNTS = {
    "allegro": {
        # Desired palm axes expressed in the flange frame.  +Z of the flange
        # frame is the course gripper's tool axis (world "down" at the home
        # pose), so the palm faces down and the fingers point horizontally,
        # which is the classic table-top power-grasp configuration.
        "palm_x_target": (0.0, 0.0, 1.0),
        "palm_z_target": (-1.0, 0.0, 0.0),
        "standoff": 0.042,
    },
    "leap": {
        "palm_x_target": (0.0, 0.0, 1.0),
        "palm_z_target": (-1.0, 0.0, 0.0),
        "standoff": 0.042,
    },
}


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, unit, angle)
    return quat


def mat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    quat = quat_from_axis_angle(axis, angle)
    matrix = np.zeros((3, 3))
    mujoco.mju_quat2Mat(matrix.reshape(9), quat)
    return matrix


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).reshape(9))
    if quat[0] < 0:
        quat = -quat
    return quat


def measure_unmounted_palm(
    arm_xml: pathlib.Path, body_path: pathlib.Path, pre_path: pathlib.Path
) -> np.ndarray:
    """Palm frame relative to the flange, with an identity mount."""
    root = ET.parse(arm_xml).getroot()
    root.insert(0, ET.Element("include", {"file": str(pre_path.resolve())}))
    for compiler in root.findall("compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir:
            compiler.set("meshdir", str((arm_xml.parent / meshdir).resolve()))
    for body in root.iter("body"):
        if body.get("name") == "attachment":
            include = ET.SubElement(body, "include")
            include.set("file", str(body_path.resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    attach = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "attachment")
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    return data.xmat[attach].reshape(3, 3).T @ data.xmat[palm].reshape(3, 3)


def mount_transform(hand: str, palm_pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (quaternion, standoff offset) for the flange -> hand mount."""
    spec = MOUNTS[hand]
    palm_x = np.asarray(spec["palm_x_target"], dtype=np.float64)
    palm_z = np.asarray(spec["palm_z_target"], dtype=np.float64)
    palm_x = palm_x / np.linalg.norm(palm_x)
    palm_z = palm_z / np.linalg.norm(palm_z)
    target = np.column_stack([palm_x, np.cross(palm_z, palm_x), palm_z])
    rotation = target @ palm_pre.T
    return quat_from_matrix(rotation), spec["standoff"] * palm_x


def write_fragment(children: list[ET.Element], path: pathlib.Path, comment: str) -> None:
    root = ET.Element("mujoco", {"model": path.stem})
    root.append(ET.Comment(comment))
    for child in children:
        root.append(child)
    ET.indent(root, space="  ")
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def absolutise_meshes(root: ET.Element, mesh_dir: pathlib.Path) -> None:
    for asset in root.findall("asset"):
        for tag in ("mesh", "texture"):
            for node in asset.findall(tag):
                file_attr = node.get("file")
                if file_attr and not file_attr.startswith("/"):
                    node.set("file", str((mesh_dir / file_attr).resolve()))


def namespace_assets(root: ET.Element, prefix: str) -> dict[str, str]:
    """Rename hand assets so they cannot collide with the arm's assets.

    The Franka model and the Menagerie hands both declare generic asset names
    such as ``white``; MuJoCo rejects duplicate names across an include.
    """
    renames: dict[str, str] = {}
    for asset in root.findall("asset"):
        for tag in ("mesh", "material", "texture"):
            for node in asset.findall(tag):
                name = node.get("name")
                if not name:
                    continue
                new_name = f"{prefix}{name}"
                renames[name] = new_name
                node.set("name", new_name)
    for node in root.iter():
        for attr in ("mesh", "material", "texture"):
            value = node.get(attr)
            if value in renames:
                node.set(attr, renames[value])
    return renames


def build_hand(hand: str, hand_xml: pathlib.Path, out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    hand_dir = hand_xml.parent
    root = ET.parse(hand_xml).getroot()

    for compiler in root.findall("compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir:
            compiler.attrib.pop("meshdir", None)
    absolutise_meshes(root, hand_dir / "assets")
    namespace_assets(root, f"{hand}_")

    worldbody = root.find("worldbody")
    if worldbody is None or len(worldbody) == 0:
        raise SystemExit(f"{hand_xml} has no worldbody content")
    body_children = list(worldbody)

    pre_children = [block for name in PRE_BLOCKS if (block := root.find(name)) is not None]
    post_children = [block for name in POST_BLOCKS if (block := root.find(name)) is not None]

    pre_path = out_dir / f"{hand}_pre.xml"
    post_path = out_dir / f"{hand}_post.xml"
    body_path = out_dir / f"{hand}_body.xml"
    comment = f"generated from {hand_xml.name} by scripts/01_build_assets.py"
    write_fragment(pre_children, pre_path, comment)
    write_fragment(post_children, post_path, comment)
    write_fragment(body_children, body_path, comment)
    return pre_path, post_path, body_path


def build_arm(
    hand: str,
    arm_xml: pathlib.Path,
    pre_path: pathlib.Path,
    body_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    root = ET.parse(arm_xml).getroot()
    for compiler in root.findall("compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir:
            compiler.set("meshdir", str((arm_xml.parent / meshdir).resolve()))

    flange = None
    for body in root.iter("body"):
        if body.get("name") == "attachment":
            flange = body
            break
    if flange is None:
        raise SystemExit(f"{arm_xml} has no 'attachment' body to mount the hand on")

    palm_pre = measure_unmounted_palm(arm_xml, body_path, pre_path)
    mount_quat, standoff = mount_transform(hand, palm_pre)
    wrapper = ET.SubElement(
        flange,
        "body",
        {
            "name": "dex_mount",
            "pos": " ".join(f"{value:.6f}" for value in standoff),
            "quat": " ".join(f"{value:.6f}" for value in mount_quat),
        },
    )
    include = ET.SubElement(wrapper, "include")
    include.set("file", str(body_path.resolve()))

    out_path = out_dir / f"panda_{hand}.xml"
    ET.indent(root, space="  ")
    out_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="allegro", choices=sorted(HANDS))
    parser.add_argument(
        "--arm-xml",
        default=str(ROOT / "assets/vendor/franka_emika_panda/panda_nohand.xml"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "assets/robot"))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_path, post_path, body_path = build_hand(args.hand, HANDS[args.hand], out_dir)
    arm_path = build_arm(args.hand, pathlib.Path(args.arm_xml), pre_path, body_path, out_dir)
    for path in (pre_path, post_path, body_path, arm_path):
        print("wrote", path)


if __name__ == "__main__":
    main()
