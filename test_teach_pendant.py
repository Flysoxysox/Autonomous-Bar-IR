"""Headless checks: python -m unittest test_teach_pendant -v."""

import unittest
from types import SimpleNamespace

import numpy as np
import scipy

if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

from ir_support.robots import UR3e
from scipy.spatial.transform import Rotation
from spatialmath import SE3
from teach_pendant import TeachPendant


class TeachPendantTest(unittest.TestCase):
    def setUp(self):
        self.robot = UR3e(base=SE3(0.5, 0.72, 1.0) * SE3.Rz(0.4))
        self.start_q = np.deg2rad([90, -60, 70, -90, 90, 0])
        self.robot.q = self.start_q.copy()
        self.elements = []
        self.pendant = TeachPendant(SimpleNamespace(add=self.elements.append), self.robot)

    def test_speed_slider_echo_does_not_start_motion(self):
        self.pendant.speed_slider.cb(self.pendant.speed_slider.value)
        self.pendant.sync()
        self.pendant.update(0.02)
        self.assertIsNone(self.pendant.command)
        np.testing.assert_array_equal(self.robot.q, self.start_q)

    def test_slider_changes_joint_velocity_and_takes_effect_while_jogging(self):
        self.pendant.jog_joint(1)
        self.pendant.speed_slider.value = 25
        self.pendant.update(0.02)
        slow_step = self.robot.q[0] - self.start_q[0]
        before = self.robot.q.copy()
        self.pendant.speed_slider.value = 100
        self.pendant.update(0.02)
        fast_step = self.robot.q[0] - before[0]
        self.assertAlmostEqual(fast_step, 4 * slow_step)
        self.assertAlmostEqual(fast_step, self.pendant.joint_speed * 0.02)
        np.testing.assert_array_equal(self.robot.q[1:], self.start_q[1:])

    def test_translation_buttons_move_along_world_axes(self):
        for axis in range(3):
            for direction in (1, -1):
                with self.subTest(axis=axis, direction=direction):
                    self.robot.q = self.start_q.copy()
                    start = self.robot.fkine(self.robot.q)
                    self.pendant.jog(axis, direction)
                    for _ in range(20):
                        self.pendant.update(0.01)
                    actual = self.robot.fkine(self.robot.q)
                    movement = actual.t - start.t
                    self.assertGreater(direction * movement[axis], 0)
                    expected = np.zeros(3)
                    expected[axis] = direction * self.pendant.linear_speed * 0.5 * 0.2
                    np.testing.assert_allclose(movement, expected, atol=2e-4)
                    np.testing.assert_allclose(actual.R, start.R, atol=1e-3)
                    self.pendant.release_jog()

    def test_rotation_buttons_turn_about_world_axes_and_hold_position(self):
        for axis in range(3):
            for direction in (1, -1):
                with self.subTest(axis=axis, direction=direction):
                    self.robot.q = self.start_q.copy()
                    start = self.robot.fkine(self.robot.q)
                    self.pendant.jog(axis, direction, rotation=True)
                    for _ in range(20):
                        self.pendant.update(0.01)
                    actual = self.robot.fkine(self.robot.q)
                    rotation = Rotation.from_matrix(actual.R @ start.R.T).as_rotvec()
                    self.assertGreater(direction * rotation[axis], 1e-4)
                    np.testing.assert_allclose(np.delete(rotation, axis), 0, atol=0.005)
                    np.testing.assert_allclose(actual.t, start.t, atol=1e-3)
                    self.pendant.release_jog()

    def test_new_direction_replaces_old_command_and_release_leaves_no_work(self):
        self.pendant.jog(0, 1)
        self.pendant.jog(1, -1)
        self.assertEqual(self.pendant.command, ("cartesian", 1, -1))
        self.pendant.update(0.02)
        self.pendant.release_jog()
        released_q = self.robot.q.copy()
        for _ in range(100):
            self.pendant.update(0.02)
        self.assertIsNone(self.pendant.command)
        np.testing.assert_array_equal(self.robot.q, released_q)

    def test_joint_translation_and_rotation_buttons_are_present(self):
        import swift
        buttons = [element for element in self.elements if isinstance(element, swift.Button)]
        self.assertEqual([button.desc for button in buttons[:8]], [
            "Request beer", "Deposit dirty glass", "Run wash cycle", "Return tray",
            "E-STOP", "RESET", "MANUAL", "PAUSE",
        ])
        self.assertEqual([button.desc for button in buttons[8:]], [
            "+q", "-q", "+X", "-X", "+Y", "-Y", "+Z", "-Z",
            "+RX", "-RX", "+RY", "-RY", "+RZ", "-RZ", "Release jog",
        ])
        buttons[8].cb(None)
        self.assertEqual(self.pendant.command, ("joint", 0, 1))
        buttons[16].cb(None)
        self.assertEqual(self.pendant.command, ("cartesian", 3, 1))
        buttons[-1].cb(None)
        self.assertIsNone(self.pendant.command)

    def test_robot_selector_inspects_all_arms_and_jogging_requires_manual(self):
        import swift
        from importlib import import_module
        from unittest.mock import Mock
        FR3 = import_module("Fairano FR3.fr3").FairinoFR3
        from RS007L import RS007L
        from rs007l_controller import RS007LController
        from program_control import ProgramControl
        from ur3_controller import UR3Controller

        kawasaki = RS007L(base=SE3(0, 0, 0.35))
        fr3 = FR3(base=SE3(-0.79, 1.05, 1.0))
        scene = SimpleNamespace(tray_pose=SE3(0, 0.92, 1), door_open=False,
                                update_fork=Mock())
        tray = RS007LController(kawasaki, scene, None)
        tray.check_collision = Mock(return_value=None)
        control = ProgramControl(UR3Controller(self.robot), tray, [self.robot, kawasaki, fr3])
        elements = []
        pendant = TeachPendant(SimpleNamespace(add=elements.append), self.robot, control)
        selector = next(element for element in elements
                        if isinstance(element, swift.Select) and element.desc == "Robot")
        self.assertEqual(selector.options, ["UR3", "Kawasaki", "FR3"])
        original = [arm.q.copy() for arm in control.robots]
        for index, arm in enumerate(control.robots):
            selector.cb(index)
            self.assertIs(pendant.robot, arm)
            np.testing.assert_array_equal(pendant.joint_limits, arm.qlim)
            pendant.jog(0, 1)
            self.assertIsNone(pendant.command)
        for arm, q in zip(control.robots, original):
            np.testing.assert_array_equal(arm.q, q)
        control.toggle_manual()
        pendant.jog(2, 1, rotation=True)
        self.assertEqual(pendant.command, ("cartesian", 5, 1))
        pendant.select_robot(0)
        self.assertIsNone(pendant.command)


if __name__ == "__main__":
    unittest.main()
