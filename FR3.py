from math import pi
from pathlib import Path

import numpy as np
import scipy

if not hasattr(scipy, 'randn'):
    scipy.randn = np.random.randn

from roboticstoolbox import DHRobot, RevoluteDH
from spatialgeometry import Mesh
from spatialmath import SE3


class FR3(DHRobot):
    MESH_DIR = Path(__file__).resolve().parent / 'assets' / 'FR3' / 'meshes'

    _CAD_TO_BASE = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def __init__(self, base=None):
        self.world_pose = SE3() if base is None else SE3(base)
        limits = [
            (-175, 175),
            (-265, 85),
            (-150, 150),
            (-265, 85),
            (-175, 175),
            (-175, 175),
        ]
        specifications = [
            (0.0, 0.140, pi / 2),
            (-0.280, 0.0, 0.0),
            (-0.240, 0.0, 0.0),
            (0.0, 0.102, pi / 2),
            (0.0, 0.102, -pi / 2),
            (0.0, 0.100, 0.0),
        ]
        links = [
            RevoluteDH(a=a, d=d, alpha=alpha, qlim=np.deg2rad(limit))
            for (a, d, alpha), limit in zip(specifications, limits)
        ]
        super().__init__(links, name='FAIRINO FR3', base=self.world_pose)
        self.q = np.zeros(6)
        self.meshes = []
        self._zero_frames = self._joint_frames(np.zeros(6))
        self._zero_inverse = [np.linalg.inv(frame) for frame in self._zero_frames]

    @staticmethod
    def _joint_frames(q):
        q = np.asarray(q, dtype=float).reshape(6)
        specifications = [
            (0.0, 0.140, pi / 2),
            (-0.280, 0.0, 0.0),
            (-0.240, 0.0, 0.0),
            (0.0, 0.102, pi / 2),
            (0.0, 0.102, -pi / 2),
            (0.0, 0.100, 0.0),
        ]
        pose = SE3()
        frames = [pose.A.copy()]
        for angle, (a, d, alpha) in zip(q, specifications):
            pose = pose * SE3.Rz(float(angle)) * SE3(0, 0, d) * SE3(a, 0, 0) * SE3.Rx(alpha)
            frames.append(pose.A.copy())
        return frames

    def add_to_env(self, env):
        if self.meshes:
            raise RuntimeError('FR3 meshes are already loaded')
        paths = [self.MESH_DIR / f'FR3_link{i}.stl' for i in range(7)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing FR3 meshes: ' + ', '.join(missing))
        for i, path in enumerate(paths):
            color = [0.27, 0.29, 0.33, 1.0] if i in (0, 6) else [0.90, 0.92, 0.94, 1.0]
            mesh = Mesh(filename=str(path), color=color)
            env.add(mesh)
            self.meshes.append(mesh)
        self.update_meshes()

    def update_meshes(self):
        if not self.meshes:
            return
        frames = self._joint_frames(self.q)
        for i, mesh in enumerate(self.meshes):
            mesh.T = self.world_pose.A @ frames[i] @ self._zero_inverse[i] @ self._CAD_TO_BASE

    def set_joint_angles(self, q):
        q = np.asarray(q, dtype=float).reshape(6)
        if not np.all(np.isfinite(q)):
            raise ValueError('Joint angles must be finite')
        if np.any(q < self.qlim[0]) or np.any(q > self.qlim[1]):
            raise ValueError('FR3 joint angle outside limits')
        self.q = q
        self.update_meshes()
