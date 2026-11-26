"""Generate the extension's scene from the course scene.

Starting from the course ``scene.xml`` guarantees that the extension keeps the
same table, trays, objects, cameras and physics defaults as the graded task; the
only change is the robot: the course Panda + parallel gripper is replaced by the
same Panda arm carrying a Menagerie dexterous hand.

Relative asset paths are rewritten to absolute paths, because the generated file
lives in a different directory than the course scene.
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
COURSE_ASSETS = pathlib.Path(
    "/Users/yoimiyalike/Desktop/2026/短学期/mujoco-grasp-assignment/src/graspbench/assets"
)


def resolve_asset(value: str, bases: list[pathlib.Path]) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if candidate.is_absolute():
        return candidate
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise SystemExit(f"cannot resolve asset {value!r} against {[str(b) for b in bases]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-assets", default=str(COURSE_ASSETS))
    parser.add_argument("--hand", default="allegro")
    parser.add_argument("--out", default=str(ROOT / "assets/scene_dex.xml"))
    args = parser.parse_args()

    course_dir = pathlib.Path(args.course_assets)
    panda_assets = course_dir / "vendor/franka_emika_panda/assets"
    bases = [course_dir, panda_assets]

    root = ET.parse(course_dir / "scene.xml").getroot()

    # 1. swap the robot include for our arm + hand fragments
    robot_include = None
    for include in root.findall("include"):
        if "panda.xml" in (include.get("file") or ""):
            robot_include = include
    if robot_include is None:
        raise SystemExit("course scene does not include the panda robot")
    robot_include.set("file", str((ROOT / f"assets/robot/panda_{args.hand}.xml").resolve()))

    # Defaults/assets must be declared before the arm; actuators and contact
    # blocks after it, so actuator indices remain "7 arm joints, then hand".
    hand_pre = ET.Element("include")
    hand_pre.set("file", str((ROOT / f"assets/robot/{args.hand}_pre.xml").resolve()))
    hand_post = ET.Element("include")
    hand_post.set("file", str((ROOT / f"assets/robot/{args.hand}_post.xml").resolve()))
    index = list(root).index(robot_include)
    root.insert(index, hand_pre)
    root.insert(index + 2, hand_post)

    # 2. rewrite asset paths so they resolve from the generated file's location
    for asset in root.findall("asset"):
        for tag in ("mesh", "texture"):
            for node in asset.findall(tag):
                file_attr = node.get("file")
                if not file_attr:
                    continue
                node.set("file", str(resolve_asset(file_attr, bases)))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    out_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
