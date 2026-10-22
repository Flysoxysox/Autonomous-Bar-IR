"""Reusable A2 scene for an already launched Swift environment."""

import numpy as np
from spatialmath import SE3
from IR_objects import create_bar_counter
from ir_support.robots import UR3e
from beer_tap import BeerTap
from ir_support_extra_parts.parts import part_mesh



# Main worktop height in metres; mounted objects follow this height.
TABLE_HEIGHT = 1.0


class A2Scene:
    """Add all scene objects to an existing Swift environment.

    The caller handles launching, stepping and closing the environment.
    Access the robot as scene.ur3 and the beer tap as scene.tap.
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

        self.tap = BeerTap()
        self.tap.set_pose(SE3(self.bar_table_offset) * SE3.Trans(0.90, 0.1, 0) * SE3.Rz(np.pi))
        self.tap.q = 0
        self.tap.add_to_env(self.env)


        self.person = part_mesh("personMaleCasual")
        self.person.T = SE3.Trans(0, 2.0, 0).A @ SE3.Rz(-np.pi/2).A
        self.env.add(self.person)


        self.ur3_q_start = np.deg2rad([180, 0, 0, -90, 90, 0])
        self.ur3 = UR3e()
        self.ur3.base = self.bar_table_offset @ SE3.Trans(0.50, -0.26, 0).A
        self.ur3.q = self.ur3_q_start
        self.ur3.add_to_env(self.env)

        # ---------------- DISHWASHING STATION ----------------
        # Retain the former tray reference pose for callers.
        self.tray_pose = SE3(self.bar_table_offset) * SE3(0.9, -.6, 0) * SE3.Rz(np.pi / 2)

        # ---------------- CLEAN GLASS STATION ----------------

        # ---------------- BEER SERVING STATION ----------------

        # ---------------- ADDITIONAL BAR OBJECTS ----------------
