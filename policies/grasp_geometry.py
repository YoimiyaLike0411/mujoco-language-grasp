from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graspbench.config import HOME_Q

from .motion import grasp_rotation, quaternion


@dataclass
class Estimate:
    name: str
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    evidence: dict
    grasp_center: np.ndarray | None = None
    grasp_rotation: np.ndarray | None = None
    grasp_width: float | None = None


@dataclass
class Grasp:
    position: np.ndarray
    quaternion: np.ndarray
    pregrasp: np.ndarray
    pregrasp_q: np.ndarray
    width: float
    clearance_height: float = 0.64


def plan_grasp(estimate, motion, current_q, clearance_height=0.64):
    """Search narrow-axis grasps and modest radial tilts; check approach too."""
    radial = estimate.center[:2].copy()
    candidates = []
    dims = estimate.size
    axes = estimate.grasp_rotation if estimate.grasp_rotation is not None else estimate.rotation
    # Box principal axes supply antipodal candidates; spheres are yaw invariant.
    angles = (
        (0, np.pi / 2, np.pi, 3 * np.pi / 2)
        if estimate.name not in ("apple", "orange", "green_cylinder")
        else tuple(np.arange(0, 2 * np.pi, np.pi / 6))
    )
    for yaw in angles:
        axis = axes[:2, :2] @ np.array([np.cos(yaw), np.sin(yaw)])
        if estimate.name in ("apple", "orange", "green_cylinder"):
            axis = np.array([np.cos(yaw), np.sin(yaw)])
        width = abs(np.cos(yaw)) * dims[0] + abs(np.sin(yaw)) * dims[1]
        if estimate.grasp_width is not None:
            if abs(np.cos(yaw)) > 0.01:
                continue
            width = estimate.grasp_width
        if estimate.name in ("apple", "orange", "green_cylinder"):
            width = max(dims[:2])
        if width > 0.077:
            continue
        for tilt in (
            (0, 15, -15, 25, -25, 35, -35)
            if estimate.name == "banana"
            else (0, 15, 25, 35)
        ):
            rot = grasp_rotation(axis, radial, np.deg2rad(tilt))
            pos = estimate.center.copy()
            if estimate.grasp_center is not None:
                pos = estimate.grasp_center.copy()
            # Keep the palm above tall objects and finger mesh above the table.
            pos[2] = max(pos[2], estimate.center[2] + dims[2] * 0.5 - 0.032, 0.436)
            if estimate.name in ("apple", "orange"):
                pos[2] = max(0.434, estimate.center[2] - 0.002)
            if estimate.name == "banana":
                pos[2] = max(
                    0.414,
                    (
                        estimate.grasp_center
                        if estimate.grasp_center is not None
                        else estimate.center
                    )[2],
                )
            quat = quaternion(rot)
            pre = pos - rot[:, 2] * 0.065
            best = None
            for seed in (current_q, HOME_Q):
                grasp_sol = motion.ik.solve(seed, pos, quat, max_iterations=180)
                if not grasp_sol.converged:
                    continue
                approach_sol = motion.ik.solve(
                    grasp_sol.joint_position, pre, quat, max_iterations=120
                )
                if not approach_sol.converged:
                    continue
                # Prefer the full travel height. At the workspace edge, search
                # for the highest reachable departure above 0.50 m, then raise
                # again after retracting inward. If even the lower probe fails,
                # retain the otherwise valid grasp so execution still tries it.
                full_height = max(clearance_height, pre[2])
                lower_height = max(0.50, pre[2])
                departure_height = full_height
                departure_verified = False
                high = pos.copy()
                high[2] = full_height
                full_sol = motion.ik.solve(
                    grasp_sol.joint_position, high, quat, max_iterations=180
                )
                if full_sol.converged and motion.safe_pose(full_sol.joint_position):
                    departure_verified = True
                else:
                    high[2] = lower_height
                    lower_sol = motion.ik.solve(
                        grasp_sol.joint_position, high, quat, max_iterations=180
                    )
                    if lower_sol.converged and motion.safe_pose(lower_sol.joint_position):
                        departure_height = lower_height
                        departure_verified = True
                        lower_q = lower_sol.joint_position
                        upper_height = full_height
                        # Four bisection probes select the highest verified
                        # height to within roughly one centimetre.
                        for _ in range(4):
                            probe_height = 0.5 * (departure_height + upper_height)
                            high[2] = probe_height
                            probe_sol = motion.ik.solve(
                                lower_q, high, quat, max_iterations=180
                            )
                            if probe_sol.converged and motion.safe_pose(
                                probe_sol.joint_position
                            ):
                                departure_height = probe_height
                                lower_q = probe_sol.joint_position
                            else:
                                upper_height = probe_height
                    else:
                        departure_height = lower_height
                margin = np.min(
                    np.minimum(
                        approach_sol.joint_position - motion.ik.lower,
                        motion.ik.upper - approach_sol.joint_position,
                    )
                )
                if margin < 0.04:
                    continue
                if not motion.safe_pose(approach_sol.joint_position):
                    continue
                if estimate.name == "banana" and not motion.safe_grasp_pose(
                    grasp_sol.joint_position, estimate
                ):
                    continue
                cost = (
                    width
                    + abs(tilt) * 0.001
                    + np.linalg.norm(approach_sol.joint_position - current_q) * 0.003
                )
                if estimate.name in ("apple", "orange"):
                    cost += 0.10 * (1 - abs(axis[1]))
                if departure_height < clearance_height - 1e-3:
                    cost += 0.10 if departure_verified else 0.20
                candidate = (
                    cost,
                    Grasp(
                        pos,
                        quat,
                        pre,
                        approach_sol.joint_position,
                        width,
                        departure_height,
                    ),
                )
                if best is None or cost < best[0]:
                    best = candidate
                break  # one feasible IK branch per grasp; avoid duplicate solves
            if best:
                candidates.append(best)
                # Preserve the original full-height preference. A lower
                # departure remains a fallback while other tilts are searched.
                if best[1].clearance_height >= clearance_height - 1e-3:
                    break
    if not candidates:
        raise ValueError("No feasible grasp and approach pose")
    return min(candidates, key=lambda value: value[0])[1]
