"""Isolated dexterous-grasping extension for the MuJoCo course assignment.

Nothing in this package imports from, or writes to, the course project; the
extension reuses the course *scene* (table, trays, objects, cameras) through a
generated XML file and adds the dexterous hand on the same Franka arm.
"""

from .env import DexEnv
from .hand import HandSynergy, fingertips, palm_frame, tcp_position
from .ik import ArmIK

__all__ = [
    "DexEnv",
    "ArmIK",
    "HandSynergy",
    "fingertips",
    "palm_frame",
    "tcp_position",
]
