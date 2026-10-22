"""Headless safety checks: python -m unittest test_program_control -v."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import scipy

if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

from ir_support.robots import UR3e
from spatialmath import SE3
from RS007L import RS007L
from rs007l_controller import RS007LController
from ur3_controller import UR3Controller
from fr3_controller import FR3Controller
from program_control import ProgramControl


class ProgramControlTest(unittest.TestCase):
    def setUp(self):
        self.robots = [UR3e(), RS007L(base=SE3.Tz(0.35)), UR3e()]
        self.scene = SimpleNamespace(
            tray_pose=SE3(0, 0.92, 1), door_open=False, glasses={},
            update_fork=Mock(), set_tray_pose=Mock(), free_slot=Mock(return_value=(0, 0))
        )
        self.ur3 = UR3Controller(self.robots[0])
        self.tray = RS007LController(self.robots[1], self.scene, None, dt=0.02)
        # isolate state transitions from workcell geometry in these unit checks
        self.tray.check_collision = Mock(return_value=None)
        self.tray.check_trajectory = Mock()
        self.fr3 = FR3Controller(self.robots[2])
        self.scanner = Mock()
        self.scanner.detect.return_value = []
        self.control = ProgramControl(self.ur3, self.tray, self.robots, self.fr3,
                                      scanner=self.scanner)

    def test_idle_allows_collection_and_blocks_new_task(self):
        self.scanner.detect.return_value = ["Person", "Glass"]
        self.control.request_wash()
        self.tray.dishwasher_cycle = Mock()
        self.control.update(0.02)
        self.assertEqual(self.control.state, "IDLE")
        self.assertEqual(self.control.light_colour, "green")
        self.tray.dishwasher_cycle.assert_not_called()
        self.assertEqual(self.control.tray_actions, ["WASH"])

    def test_every_object_faults_in_every_keep_out_state(self):
        for state in ("RUNNING", "HOMING", "MANUAL", "PAUSED"):
            names = ("Person", "Glass") if state == "HOMING" else ("Person", "Glass", "UR3", "Kawasaki", "FR3")
            for name in names:
                with self.subTest(state=state, name=name):
                    self.control.state = state
                    self.scanner.detect.return_value = [name]
                    self.assertFalse(self.control.check_safety())
                    self.assertEqual(self.control.state, "FAULT")
                    self.assertIn(name, self.control.fault_reason)
                    self.assertEqual(self.control.light_colour, "orange")
                    for controller in self.control._controllers():
                        self.assertEqual(controller.state, "STOPPED")
                    for robot in self.robots:
                        np.testing.assert_array_equal(robot.qd, np.zeros(6))

    def test_fault_discards_work_but_keeps_held_glass(self):
        glass = SimpleNamespace(T=np.eye(4))
        self.ur3.held_glass = glass
        self.control.request_beer()
        self.control.deposit_dirty_glass()
        self.control.request_wash()
        self.control.fault("Planning failed")
        self.assertIs(self.ur3.held_glass, glass)
        self.assertEqual(self.ur3.pending_beers, 0)
        self.assertEqual(self.control.tray_actions, [])
        self.assertEqual(self.control.dirty_glasses_pending, 0)
        self.control.request_beer()
        self.assertEqual(self.ur3.pending_beers, 0)

    def test_estop_release_never_restarts_motion(self):
        self.control.stop()
        self.control.reset()
        self.assertEqual(self.control.state, "FAULT")
        self.control.toggle_estop()
        self.assertEqual(self.control.state, "FAULT")
        self.assertFalse(self.control.manual_enabled)
        self.control.reset()
        self.assertEqual(self.control.state, "HOMING")

    def test_reset_is_blocked_by_curtain_or_collision(self):
        self.control.fault("Test fault")
        self.scanner.detect.return_value = ["Glass"]
        self.control.reset()
        self.assertEqual(self.control.state, "FAULT")
        self.scanner.detect.return_value = []
        self.tray.check_collision.return_value = "blocked path"
        self.control.reset()
        self.assertEqual(self.control.state, "FAULT")

    def test_controller_error_and_collision_stop_every_robot(self):
        self.ur3.state = "ERROR"
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        self.assertIn("UR3", self.control.fault_reason)
        self.control.state = "RUNNING"
        self.tray.check_collision.return_value = "fork near table"
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        self.assertIn("collision", self.control.fault_reason)

    def test_detector_failure_is_a_fault(self):
        self.scanner.detect.side_effect = RuntimeError("detector unavailable")
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        self.assertIn("detector unavailable", self.control.fault_reason)

    def test_task_failure_is_a_fault(self):
        self.control.state = "RUNNING"
        self.control._update_tasks = Mock(side_effect=RuntimeError("No IK solution"))
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        self.assertIn("No IK solution", self.control.fault_reason)

    def test_manual_is_explicit_and_unavailable_in_fault(self):
        target = self.robots[0].q + 0.01
        self.control.set_jog_joints(target)
        self.assertFalse(self.control.manual_enabled)
        self.control.toggle_manual()
        self.assertTrue(self.control.manual_enabled)
        self.assertEqual(self.control.light_colour, "red")
        self.control.set_jog_joints(target)
        np.testing.assert_allclose(self.robots[0].q, target)
        self.control.fault("Test fault")
        self.control.toggle_manual()
        self.assertFalse(self.control.manual_enabled)

    def test_manual_crossing_is_rejected_at_last_clear_pose(self):
        before = self.robots[0].q.copy()
        self.control.toggle_manual()
        self.scanner.detect.side_effect = lambda robots: (
            ["UR3"] if self.robots[0].q[0] > before[0] + 0.005 else []
        )
        self.control.set_jog_joints(before + 0.01)
        self.assertEqual(self.control.state, "FAULT")
        np.testing.assert_array_equal(self.robots[0].q, before)

    def test_automatic_crossing_is_rejected_at_last_clear_pose(self):
        before = self.robots[0].q.copy()
        self.control.state = "RUNNING"
        self.control._update_tasks = lambda dt, world: setattr(self.robots[0], "q", before + 0.01)
        self.scanner.detect.side_effect = lambda robots: (
            ["UR3"] if self.robots[0].q[0] > before[0] + 0.005 else []
        )
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        np.testing.assert_array_equal(self.robots[0].q, before)

    def test_homing_can_pause_and_resume_without_restarting_the_path(self):
        self.robots[0].q = self.robots[0].q + 0.02
        self.control.fault("Test fault")
        self.control.reset()
        self.control.update(0.02)
        path = self.control.home_path
        step = self.control.home_step
        self.control.toggle_pause()
        self.control.update(0.02)
        self.assertEqual(self.control.state, "PAUSED")
        self.assertIs(self.control.home_path, path)
        self.assertEqual(self.control.home_step, step)
        self.control.toggle_pause()
        self.assertEqual(self.control.state, "HOMING")

    def test_pause_preserves_task_until_resume(self):
        path = np.ones((10, 6))
        self.ur3.trajectory = path
        self.ur3.state = "LIFT_BEER"
        self.ur3.step = 3
        self.control.state = "RUNNING"
        self.control.toggle_pause()
        self.control.update(0.02)
        self.assertEqual(self.control.state, "PAUSED")
        self.assertEqual(self.control.light_colour, "red")
        self.assertIs(self.ur3.trajectory, path)
        self.assertEqual(self.ur3.step, 3)
        self.control.toggle_pause()
        self.assertEqual(self.control.state, "RUNNING")

    def test_homing_is_sequential_and_limited_to_thirty_degrees_per_second(self):
        for robot in self.robots:
            robot.q = robot.q + 0.02
        self.control.fault("Test fault")
        self.control.reset()
        for _ in range(200):
            before = [r.q.copy() for r in self.robots]
            self.control.update(0.02)
            moved = [np.max(np.abs(r.q - q)) for r, q in zip(self.robots, before)]
            self.assertLessEqual(sum(d > 1e-10 for d in moved), 1)
            self.assertLessEqual(max(moved), self.control.home_speed * 0.02 + 1e-8)
            if self.control.state == "IDLE":
                break
        self.assertEqual(self.control.state, "IDLE")
        self.assertEqual(self.control.light_colour, "green")
        for robot, home in zip(self.robots, self.control.home_positions):
            np.testing.assert_allclose(robot.q, home, atol=1e-10)
        self.assertEqual(self.ur3.pending_beers, 0)

    def test_robot_intrusion_allows_reset_and_does_not_retrigger_while_homing(self):
        # the detector only sees the robot when it is actually asked to check robots
        self.scanner.detect.side_effect = lambda robots: ["UR3"] if robots else []
        self.control.state = "RUNNING"
        self.assertFalse(self.control.check_safety())
        self.assertEqual(self.control.state, "FAULT")
        self.control.reset()
        self.assertEqual(self.control.state, "HOMING")
        self.scanner.detect.assert_called_with([])
        self.control.update(0.02)
        self.assertEqual(self.control.state, "HOMING")
        self.control.toggle_pause()
        self.assertTrue(self.control.check_safety())
        self.scanner.detect.assert_called_with([])
        self.control.toggle_pause()
        self.assertEqual(self.control.state, "HOMING")

    def test_homing_obstacle_keeps_system_in_fault(self):
        self.robots[1].q = self.robots[1].q + 0.02
        before = self.robots[1].q.copy()
        self.control.fault("Test fault")
        self.control.reset()
        self.tray.check_trajectory.side_effect = RuntimeError("home path blocked")
        self.control.update(0.02)
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")
        np.testing.assert_array_equal(self.robots[1].q, before)

    def test_intrusion_during_homing_stops_recovery(self):
        self.control.fault("Test fault")
        self.control.reset()
        self.scanner.detect.return_value = ["Person"]
        self.control.update(0.02)
        self.assertEqual(self.control.state, "FAULT")

    def test_finished_task_returns_to_green_idle(self):
        self.control.state = "RUNNING"
        self.control._update_tasks = Mock()
        self.control.update(0.02)
        self.assertEqual(self.control.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
