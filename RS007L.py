"""Kawasaki RS007L: assumed standard DH model and Kawasaki ROS STL visuals.

Required meshes, unchanged filenames:
    assets/RS007L/meshes/RS007L_J0.stl ... RS007L_J6.stl

No DH/reference FK verification is performed.
"""

from pathlib import Path
from math import pi

import numpy as np
import scipy

# Compatibility workaround for older roboticstoolbox-python 1.1.0.
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from spatialgeometry import Mesh


class RS007L(DHRobot):
    MESH_DIR = Path(__file__).resolve().parent / "assets" / "RS007L" / "meshes"

    def __init__(self, base=None):
        self.world_pose = SE3() if base is None else SE3(base)
        links = [
            RevoluteDH(d=0.360, a=0, alpha=pi / 2,
                       flip=True, qlim=np.deg2rad([-180, 180])),
            RevoluteDH(d=0, a=0.455, alpha=0,
                       offset=pi / 2, qlim=np.deg2rad([-135, 135])),
            RevoluteDH(d=0, a=0, alpha=pi / 2,
                       flip=True, offset=pi / 2, qlim=np.deg2rad([-157, 157])),
            RevoluteDH(d=0.475, a=0, alpha=pi / 2,
                       qlim=np.deg2rad([-200, 200])),
            RevoluteDH(d=0, a=0, alpha=-pi / 2,
                       qlim=np.deg2rad([-125, 125])),
            RevoluteDH(d=0.078, a=0, alpha=0,
                       qlim=np.deg2rad([-360, 360])),
        ]
        super().__init__(
            links,
            name="Kawasaki RS007L",
            base=self.world_pose * SE3.Rz(-pi / 2),
            tool=SE3.Rz(-pi / 2),
        )
        self.q = np.zeros(6)
        self.meshes = []

        # Kawasaki ROS visual mesh origins, rather than STEP assembly offsets.
        self._visual_origins = [
            SE3().A,
            SE3().A,
            SE3.Ry(pi / 2).A,
            SE3.Ry(pi / 2).A,
            SE3(0, 0, 0.3825).A,
            SE3.Ry(pi / 2).A,
            SE3().A,
        ]

    @staticmethod
    def _joint_frames(q):
        """Poses for the seven Kawasaki ROS mesh links, base through wrist."""
        q = np.asarray(q, dtype=float).reshape(6)
        T = SE3()
        frames = [T.A.copy()]
        T = T * SE3(0, 0, 0.360) * SE3.Rz(-q[0]); frames.append(T.A.copy())
        T = T * SE3.Ry(-pi / 2) * SE3.Rz(q[1]); frames.append(T.A.copy())
        T = T * SE3(0.455, 0, 0) * SE3.Rz(-q[2]); frames.append(T.A.copy())
        T = T * SE3(0.0925, 0, 0) * SE3.Ry(pi / 2) * SE3.Rz(q[3]); frames.append(T.A.copy())
        T = T * SE3(0, 0, 0.3825) * SE3.Ry(-pi / 2) * SE3.Rz(-q[4]); frames.append(T.A.copy())
        T = T * SE3(0.078, 0, 0) * SE3.Ry(pi / 2) * SE3.Rz(q[5]); frames.append(T.A.copy())
        return frames

    def add_to_env(self, env):
        """Add only the official Kawasaki RS007L_J0..J6 STL files."""
        paths = [self.MESH_DIR / f"RS007L_J{i}.stl" for i in range(7)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing Kawasaki meshes:\n" + "\n".join(missing))
        if self.meshes:
            raise RuntimeError("Kawasaki meshes already added to Swift")

        for i, path in enumerate(paths):
            color = [0.26, 0.28, 0.30, 1.0] if i == 6 else [0.84, 0.89, 0.87, 1.0]
            mesh = Mesh(filename=str(path), color=color)
            env.add(mesh)
            self.meshes.append(mesh)
        self.update_meshes()
        print("Loaded Kawasaki RS007L meshes")

    def update_meshes(self):
        if not self.meshes:
            return
        frames = self._joint_frames(self.q)
        for i, mesh in enumerate(self.meshes):
            mesh.T = self.world_pose.A @ frames[i] @ self._visual_origins[i]

    def set_joint_angles(self, q):
        q = np.asarray(q, dtype=float).reshape(6)
        if np.any(q < self.qlim[0]) or np.any(q > self.qlim[1]):
            raise ValueError("Joint angle outside RS007L limits")
        self.q = q
        self.update_meshes()
