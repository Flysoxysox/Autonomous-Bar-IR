"""Reusable A2 scene for an already launched Swift environment."""

import swift
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
    Set teach_pendant=False to use your own robot controller.
    """

    def __init__(self, env, table_height=TABLE_HEIGHT, teach_pendant=True):
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
        self.ur3.base = self.bar_table_offset @ SE3.Trans(0.50, -0.28, 0).A
        self.ur3.q = self.ur3_q_start
        self.ur3.add_to_env(self.env)

        # ---------------- DISHWASHING STATION ----------------
        # Retain the former tray reference pose for callers.
        self.tray_pose = SE3(self.bar_table_offset) * SE3(0.9, -.6, 0) * SE3.Rz(np.pi / 2)

        # ---------------- CLEAN GLASS STATION ----------------

        # ---------------- BEER SERVING STATION ----------------

        # ---------------- ADDITIONAL BAR OBJECTS ----------------

        if teach_pendant:
            self._add_teach_pendant()

    def sync_teach_pendant(self):
        """Read the robot's latest joints into the pendant without moving it."""
        if not hasattr(self, "joint_slider"):
            return
        self.state["q"] = np.array(self.ur3.q, dtype=float, copy=True)
        index = self.state["joint_index"]
        self.joint_label.desc = f"Selected joint: q{index + 1}"
        self.q_label.desc = "Joint values (rad): " + "  ".join(
            f"q{i + 1}={value:+.2f}" for i, value in enumerate(self.state["q"])
        )
        self.joint_slider.min = float(self.joint_limits[0, index])
        self.joint_slider.max = float(self.joint_limits[1, index])
        self.joint_slider.value = float(self.state["q"][index])

    def _add_teach_pendant(self):
        """Add the Swift controls and callbacks for this scene's UR3."""
        self.state = {
            "q": np.array(self.ur3.q, dtype=float),
            "joint_index": 0,
            "nudge_step": 0.10,
        }

        self.status_label = swift.Label("Use the controls below to move the UR3.")
        self.joint_label = swift.Label("Selected joint: q1")
        self.q_label = swift.Label("")

        # Use the UR3 joint limits, with a full-turn fallback if unavailable.
        self.joint_limits = np.array(self.ur3.qlim, dtype=float)
        if self.joint_limits.shape != (2, self.ur3.n):
            self.joint_limits = np.tile([[-2 * np.pi], [2 * np.pi]], (1, self.ur3.n))
        self.joint_limits[0] = np.where(np.isfinite(self.joint_limits[0]), self.joint_limits[0], -2 * np.pi)
        self.joint_limits[1] = np.where(np.isfinite(self.joint_limits[1]), self.joint_limits[1], 2 * np.pi)


        def update_display():
            """Update the UR3 and the joint readouts."""
            self.ur3.q = self.state["q"].copy()
            self.sync_teach_pendant()


        def set_joint_from_slider(value):
            """Move the selected joint within its limits."""
            self.state["q"] = np.array(self.ur3.q, dtype=float, copy=True)
            index = self.state["joint_index"]
            self.state["q"][index] = float(np.clip(float(value), *self.joint_limits[:, index]))
            self.status_label.desc = f"Moved q{index + 1} with the slider."
            update_display()


        def select_joint(index):
            """Choose the joint controlled by the slider and nudge buttons."""
            self.state["joint_index"] = int(index)
            index = self.state["joint_index"]
            self.status_label.desc = f"Selected q{index + 1}."
            self.sync_teach_pendant()


        def nudge_selected_joint(amount):
            index = self.state["joint_index"]
            set_joint_from_slider(self.ur3.q[index] + amount)
            self.status_label.desc = f"Nudged q{index + 1} by {amount:+.2f} rad."


        def zero_robot(_=None):
            self.state["q"][:] = np.clip(np.zeros(self.ur3.n), self.joint_limits[0], self.joint_limits[1])
            self.status_label.desc = "Reset joints to zero within their limits."
            update_display()


        def move_to_home_pose(_=None):
            self.state["q"][:] = self.ur3_q_start
            self.status_label.desc = "Returned to the starting pose."
            update_display()


        self.env.add(swift.Label("UR3 teach pendant"))
        self.env.add(self.status_label)
        self.env.add(swift.Select(
            select_joint,
            desc="Joint",
            options=[f"q{i + 1}" for i in range(self.ur3.n)],
            value=0,
        ))

        self.joint_slider = swift.Slider(
            set_joint_from_slider,
            min=float(self.joint_limits[0, 0]),
            max=float(self.joint_limits[1, 0]),
            step=0.01,
            value=float(self.state["q"][0]),
            desc="Selected joint",
            unit="rad",
        )
        self.env.add(self.joint_slider)
        self.env.add(swift.Button(lambda _: nudge_selected_joint(-self.state["nudge_step"]), desc="Nudge -0.10 rad"))
        self.env.add(swift.Button(lambda _: nudge_selected_joint(self.state["nudge_step"]), desc="Nudge +0.10 rad"))
        self.env.add(swift.Button(zero_robot, desc="Zero joints"))
        self.env.add(swift.Button(move_to_home_pose, desc="Home pose"))
        self.env.add(self.joint_label)
        self.env.add(self.q_label)
        update_display()
