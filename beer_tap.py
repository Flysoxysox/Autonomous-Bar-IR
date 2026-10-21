"""One-joint beer tap using the guide's UTSMeshRobot/link-local workflow."""

import json
from math import isfinite, radians
from pathlib import Path

import roboticstoolbox as rtb
from spatialmath import SE3
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


ASSET_DIR = Path(__file__).resolve().parent / 'beer-tap' / 'swift_assets'
HANDLE_LIMITS = (0.0, radians(35))  # demonstration range rather than measured mechanical stops


class BeerTap(UTSMeshRobot):
    """Fixed tap body plus a handle revolving about its actual hinge pin.

    ``pose`` places the foot of the stand on the counter (metres). Stand +Z
    is up; the nozzle faces +Y. Native FBX size is retained: about 0.514 m
    tall when closed. The source FBX glass is excluded.

    As in the supplied guide, the two DAE meshes are local to their DH link
    frames. A single zero-length revolute link models the hinge, with local
    +Z aligned to stand +X. Positive q pulls the handle toward stand +Y.
    q is in radians. Use set_handle_angle() for range checking, or assign
    q=[angle] in an existing controller. Mesh transforms update explicitly.

    Inherited ``base`` is the DH HINGE frame. Use set_pose() to place the
    STAND frame without needing to account for that internal offset.
    """

    def __init__(self, pose=None, angle_limits=HANDLE_LIMITS):
        metadata = json.loads((ASSET_DIR / 'frames.json').read_text())
        # this transform locates the hinge frame relative to the foot of the tap stand
        self.stand_to_hinge = SE3(metadata['hinge_in_stand'], check=False)
        # the grip point is the centre of the plastic mesh bounds in the moving handle frame
        # it is about six centimetres from the hinge along handle negative x
        self.tr_handle_grip = SE3(-0.064417429285, 0.00000042934, -0.000000056345)
        self.angle_limits = tuple(float(v) for v in angle_limits)
        if (len(self.angle_limits) != 2
                or not all(isfinite(v) for v in self.angle_limits)
                or not self.angle_limits[0] <= 0 <= self.angle_limits[1]
                or self.angle_limits[0] >= self.angle_limits[1]):
            raise ValueError('angle_limits must be increasing radians containing zero.')
        stand_pose = SE3() if pose is None else self._single_pose(pose)
        # the fixed body is the base mesh and the moving handle is the joint mesh
        # the joint rotates about hinge local z which points along stand local x
        # flip makes a positive angle pull the handle towards stand positive y
        super().__init__(
            links=[rtb.RevoluteDH(a=0, d=0, alpha=0, flip=True,
                                  qlim=list(self.angle_limits))],
            mesh_stem='BeerTap',
            mesh_dir=str(ASSET_DIR),
            name='BeerTap',
            home_q=[0.0],
            # combine the world stand pose with the stand to hinge offset
            base=stand_pose * self.stand_to_hinge,
        )

    @staticmethod
    def _single_pose(T):
        if not isinstance(T, SE3) or len(T) != 1:
            raise TypeError('pose must be a single SE3 transform.')
        return T

    @property
    def pose(self):
        """World-from-stand transform; origin is at the centre of its foot."""
        # remove the hinge offset from the robot base to recover the world stand pose
        return self.base * self.stand_to_hinge.inv()

    @property
    def hinge_pose(self):
        """World-from-hinge transform; local +Z is the rotation axis."""
        # the hinge stays fixed in the world while the handle frame rotates around it
        return SE3(self.base.A.copy(), check=False)

    def set_pose(self, T):
        # move the tap stand while keeping the hinge at its correct offset
        self.base = self._single_pose(T) * self.stand_to_hinge

    def set_handle_angle(self, angle, *, unit='rad'):
        """Set the opening angle; accepts radians or unit='deg'."""
        if unit not in ('rad', 'deg'):
            raise ValueError("unit must be 'rad' or 'deg'.")
        angle = radians(float(angle)) if unit == 'deg' else float(angle)
        # check the requested handle angle before assigning the simulated joint
        if not isfinite(angle) or not self.angle_limits[0] <= angle <= self.angle_limits[1]:
            raise ValueError(f'Handle angle must be within {self.angle_limits} radians.')
        # assigning the joint also updates the handle mesh through the parent robot class
        self.q = [angle]

    def add_to_env(self, env):
        super().add_to_env(env)
        return self
