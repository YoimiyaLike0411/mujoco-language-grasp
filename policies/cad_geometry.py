"""Register observed SAM3 depth points to public object geometry.

Only body-local mesh vertices and dimensions are read from the model. No body
world position, free-joint state, task label or random seed enters this module.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from graspbench.config import OBJECT_SPEC_BY_NAME

from .motion import matrix


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CadGeometry:
    def __init__(self, model):
        self.models = {}
        rng = np.random.default_rng(1701)  # deterministic mesh sampling, not an episode seed
        for name in OBJECT_SPEC_BY_NAME:
            body = model.body(name).id
            geoms = range(
                model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]
            )
            cloud = []
            physical = None
            for gid in geoms:
                if model.geom_contype[gid] and model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    physical = model.geom_size[gid].copy() * 2
                    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        physical = np.array([physical[0], physical[0], physical[1]])
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[gid]:
                    continue
                mesh = model.geom_dataid[gid]
                start = model.mesh_vertadr[mesh]
                count = model.mesh_vertnum[mesh]
                verts = model.mesh_vert[start : start + count]
                face_start = model.mesh_faceadr[mesh]
                face_count = model.mesh_facenum[mesh]
                faces = model.mesh_face[face_start : face_start + face_count]
                triangles = verts[faces]
                area = np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=1,
                )
                chosen = rng.choice(len(faces), 6000, p=area / area.sum())
                tri = triangles[chosen]
                uv = rng.random((len(tri), 2))
                reflect = uv.sum(1) > 1
                uv[reflect] = 1 - uv[reflect]
                points = (
                    tri[:, 0]
                    + uv[:, 0, None] * (tri[:, 1] - tri[:, 0])
                    + uv[:, 1, None] * (tri[:, 2] - tri[:, 0])
                )
                points = points @ matrix(model.geom_quat[gid]).T + model.geom_pos[gid]
                cloud.append(points)
            if cloud:
                cloud = np.concatenate(cloud)
                self.models[name] = (cloud, cKDTree(cloud), physical)

    def register(self, name, observed, initial):
        if name not in self.models:
            return None
        cloud, tree, physical = self.models[name]
        # Uniform subsampling keeps the entire visible surface, not only the cap.
        points = observed[np.linspace(0, len(observed) - 1, min(600, len(observed))).astype(int)]
        initial = np.asarray(initial).copy()
        if name == "banana":
            initial[:2] = (
                np.percentile(points[:, :2], 1, axis=0) + np.percentile(points[:, :2], 99, axis=0)
            ) / 2
        initial[2] = np.percentile(points[:, 2], 99) - np.percentile(cloud[:, 2], 99.5)
        candidates = []
        for yaw in np.linspace(-np.pi, np.pi, 60, endpoint=False):
            local = (points - initial) @ rotation_z(yaw)
            distances, _ = tree.query(local)
            score = np.mean(np.sort(distances)[: max(10, int(0.85 * len(distances)))])
            candidates.append((score, yaw))
        best = None
        for _, yaw in sorted(candidates)[: (12 if name == "banana" else 4)]:
            translation = initial.copy()
            rot = rotation_z(yaw)
            for _ in range(40 if name == "banana" else 18):
                distance, index = tree.query((points - translation) @ rot)
                keep = distance <= np.percentile(distance, 85)
                source = cloud[index[keep]]
                target = points[keep]
                sc = source.mean(0)
                tc = target.mean(0)
                if name == "banana":
                    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
                    correction = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
                    rot = vt.T @ correction @ u.T
                else:
                    cross = (source - sc)[:, :2].T @ (target - tc)[:, :2]
                    yaw = np.arctan2(cross[0, 1] - cross[1, 0], cross[0, 0] + cross[1, 1])
                    rot = rotation_z(yaw)
                translation = tc - sc @ rot.T
            distance, _ = tree.query((points - translation) @ rot)
            score = float(np.mean(np.sort(distance)[: int(0.85 * len(distance))]))
            if best is None or score < best[0]:
                best = (score, translation, rot)
        score, center, rot = best
        dimensions = physical if physical is not None else np.ptp(cloud, axis=0)
        return center, rot, dimensions, score

    def banana_grasp(self, center, rotation):
        """A cross-section on the curved body, expressed in the sensed pose."""
        cloud = self.models["banana"][0]
        _, axes = np.linalg.eigh(np.cov(cloud[:, :2].T))
        major = axes[:, -1]
        if major[1] < 0:
            major = -major
        minor = np.array([-major[1], major[0]])
        axes = np.column_stack((major, minor))
        coords = cloud[:, :2] @ axes
        longitudinal = (coords[:, 0].min() + coords[:, 0].max()) / 2 + 0.025
        band = coords[abs(coords[:, 0] - longitudinal) < 0.009]
        lateral = (band[:, 1].min() + band[:, 1].max()) / 2
        local = np.r_[axes @ np.array([longitudinal, lateral]), -0.004]
        point = center + rotation @ local
        direction = rotation @ np.r_[major, 0.0]
        direction[2] = 0
        direction /= np.linalg.norm(direction)
        grasp_axes = np.column_stack(
            (direction, np.cross([0.0, 0.0, 1.0], direction), [0.0, 0.0, 1.0])
        )
        return point, grasp_axes, float(np.ptp(band[:, 1]))


def tray_center(name, points):
    rim = points[(points[:, 2] > 0.455) & (points[:, 2] < 0.47), :2]
    if len(rim) < 20:
        raise ValueError("Insufficient tray rim geometry")
    if name == "round_tray":
        solution = np.linalg.lstsq(
            np.column_stack((2 * rim, np.ones(len(rim)))), (rim**2).sum(1), rcond=None
        )[0]
        initial = solution[:2]
        residual = lambda c: np.linalg.norm(rim - c, axis=1) - 0.122
    else:
        initial = (rim.min(0) + rim.max(0)) / 2
        residual = lambda c: np.min(abs(abs(rim - c) - 0.130), axis=1)
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=0.003, max_nfev=40)
    return fit.x
