"""Constants for the extension.

``HOME_Q_DEX`` is the course home configuration *raised* so that the longer
dexterous fingers clear the table: solved with the extension's IK from the
course home pose, moving the grasp centre to 0.56 m (see scripts/05_home_pose.py).
The course file is never modified.
"""

from __future__ import annotations

import numpy as np

ARM_DOF = 7
TABLE_Z = 0.4

# course home pose (docs: graspbench.config.HOME_Q)
COURSE_HOME_Q = np.array([0.0, -0.45, 0.0, -2.15, 0.0, 1.75, 0.785], dtype=np.float64)

# raised home pose for the arm + four-finger hand
HOME_Q_DEX = np.array([0.0, -0.328, 0.0, -1.648, 0.0, 1.37, 0.785], dtype=np.float64)

# hand posture fractions
HAND_OPEN = 0.15
HAND_CLOSED = 0.9

# Arm pose used while the cameras take a perception shot: the hand is moved off
# the table centre so the overhead view of the objects is not occluded.  The
# value was picked by rendering the overhead camera for several candidates
# (scripts/03_scene_probe.py style check); this one clears the whole workspace.
PERCEPTION_PARK_XYZ = np.array([0.30, -0.34, 0.66], dtype=np.float64)
