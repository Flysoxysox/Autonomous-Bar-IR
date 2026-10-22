"""Reusable A2 scene for an already launched Swift environment."""

from importlib import import_module

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cylinder
from IR_objects import create_bar_counter
from ir_support.robots import UR3e
from beer_tap import BeerTap
from ir_support_extra_parts.parts import part_mesh

# Import the existing model from its folder, whose name contains a space.
FairinoFR3 = import_module("Fairano FR3.fr3").FairinoFR3


# Main worktop height in metres; mounted objects follow this height.
TABLE_HEIGHT = 1.0

# Relative to the worktop frame: customer edge, on the raised counter.
# Move/rotate this frame to reposition both IR light curtain units together.
LIGHT_CURTAIN_POSE = SE3.Trans(0, 0.33, 0.30)
LIGHT_CURTAIN_HALF_SPAN = 1.04

# Relative to the main worktop frame; base rests on the raised tabletop.
ACTIVATION_LIGHT_POSE = SE3.Trans(-0.75, 0.25, 0.30)


class A2Scene:
    """Add all scene objects to an existing Swift environment.

    The caller handles launching, stepping and closing the environment.
    Access the robots as scene.ur3 and scene.fr3, and the beer tap as scene.tap.
    Robot controls and the teach pendant are created separately by the caller.
    """

    def __init__(self, env, table_height=TABLE_HEIGHT):
        self.env = env
        self.table_height = table_height
        self.tray_parts = []
        self.glasses = []

        # ---------------- DIRTY GLASS COLLECTION ----------------
        self.bar_table_pos = SE3.Trans(0, 1.0, 0)
        self.bar_table = create_bar_counter(
            self.env,
            pose=self.bar_table_pos,
            lower_height=self.table_height,
            upper_height=self.table_height + 0.3,
        )
        self.bar_table_offset = self.bar_table_pos.A @ SE3.Trans(0, 0, self.table_height).A

        self.light_curtain_pose = (
            SE3(self.bar_table_offset) * LIGHT_CURTAIN_POSE
        )
        self.light_curtain = part_mesh("SafetyLightCurtain",
            pose=self.light_curtain_pose * SE3.Trans(-LIGHT_CURTAIN_HALF_SPAN, 0, 0) * SE3.Rz(np.pi/2),
        )
        self.light_curtain_receiver = part_mesh(
            "SafetyLightCurtain",
            pose=(self.light_curtain_pose
                  * SE3.Trans(LIGHT_CURTAIN_HALF_SPAN, 0, 0) * SE3.Rz(-np.pi/2)),
        )
        self.env.add(self.light_curtain)
        self.env.add(self.light_curtain_receiver)

        self.activation_light_pose = (
            SE3(self.bar_table_offset) * ACTIVATION_LIGHT_POSE
        )
        self.activation_light = part_mesh(
            "LightTower", pose=self.activation_light_pose
        )
        self.env.add(self.activation_light)
        # dark sleeves cover the mesh lenses and fixed-colour lamps sit over them
        # this swift version supports pose updates but cannot update colours after adding
        self.safety_lamps = {}
        self.safety_lamp_poses = {}
        self.safety_light_colour = None
        colours = {"red": [1, 0.05, 0.02, 1], "orange": [1, 0.45, 0, 1],
                   "green": [0.05, 1, 0.05, 1]}
        for colour, height in (("green", 0.269), ("orange", 0.370), ("red", 0.466)):
            pose = (self.activation_light_pose * SE3.Tz(height)).A
            sleeve = Cylinder(radius=0.053, length=0.085, color=[0.08, 0.08, 0.08, 1])
            sleeve.T = pose
            self.env.add(sleeve)
            lamp = Cylinder(radius=0.054, length=0.083, color=colours[colour])
            lamp.T = SE3(0, 0, -5).A
            self.env.add(lamp)
            self.safety_lamps[colour] = lamp
            self.safety_lamp_poses[colour] = pose
        self.set_safety_light("green")

        self.tap = BeerTap()
        self.tap.set_pose(SE3(self.bar_table_offset) * SE3.Trans(0.90, 0.1, 0) * SE3.Rz(np.pi))
        self.tap.q = 0
        self.tap.add_to_env(self.env)


        self.person = part_mesh("personMaleCasual")
        self.person.T = SE3.Trans(0, 2.2, 0).A @ SE3.Rz(-np.pi/2).A
        self.env.add(self.person)


        self.ur3_q_start = np.deg2rad([180, 0, 0, -90, 90, 0])
        self.ur3 = UR3e()
        self.ur3.base = self.bar_table_offset @ SE3.Trans(0.5, -0.26, 0).A
        self.ur3.q = self.ur3_q_start
        self.ur3.add_to_env(self.env)

        # FR3 on the left side of the lower worktop, beside the UR3.
        self.fr3_q_start = np.deg2rad([0, -180, 0, 0, 0, 0])
        self.fr3 = FairinoFR3(
            base=SE3(self.bar_table_offset) * SE3.Trans(-0.4, -0.10, 0) * SE3.Rz(np.pi)
        )
        self.fr3.q = self.fr3_q_start.copy()
        self.fr3.add_to_env(self.env)

        self.tray_pose = SE3(self.bar_table_offset) * SE3(0.9, -.6, 0) * SE3.Rz(np.pi / 2)

    def set_safety_light(self, colour):
        if colour == self.safety_light_colour:
            return
        # inactive lamps are below the floor so only the selected lamp is visible
        for name, lamp in self.safety_lamps.items():
            lamp.T = self.safety_lamp_poses[name] if name == colour else SE3(0, 0, -5).A
        self.safety_light_colour = colour
