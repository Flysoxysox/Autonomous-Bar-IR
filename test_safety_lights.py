"""Check light changes through Swift's real step and GUI event processing."""

import json
import time
import unittest
import numpy as np
import scipy

if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import swift
from A2_scene import A2Scene


class SafetyLightTest(unittest.TestCase):
    def setUp(self):
        self.env = swift.Swift()
        self.env.launch(headless=True)
        self.scene = A2Scene(self.env)

    def test_light_changes_do_not_send_unsupported_shape_updates(self):
        # the installed browser has no shape_update handler or acknowledgement
        def unexpected_message(code, *args, **kwargs):
            self.fail(f"Headless light change sent a socket message: {code}")

        self.env._send_socket = unexpected_message
        for colour in ("green", "red", "orange", "green"):
            self.scene.set_safety_light(colour)
            for name, lamp in self.scene.safety_lamps.items():
                self.assertFalse(lamp._changed)
                if name == colour:
                    np.testing.assert_allclose(lamp.T, self.scene.safety_lamp_poses[name])
                else:
                    self.assertLess(lamp.T[2, 3], 0)
            before = self.env.sim_time
            self.env.step(0.005)
            self.assertGreater(self.env.sim_time, before)

    def test_clock_and_button_events_continue_after_light_changes(self):
        events = {}
        presses = []
        messages = []

        def browser_reply(code, data=None, expected=True):
            messages.append(code)
            self.assertNotEqual(code, "shape_update")
            if code == "shape_poses":
                reply = json.dumps(events)
                events.clear()
                return reply
            return "0"

        self.env._send_socket = browser_reply
        button = swift.Button(lambda value: presses.append(value), desc="Test button")
        self.env.add(button)
        # exercise the normal rendering path using the installed browser's replies
        self.env.headless = False
        self.env.realtime = True
        self.env.last_time = time.time()
        for colour in ("green", "red", "orange", "green"):
            self.scene.set_safety_light(colour)
            events[str(button._id)] = 1
            self.env.step(0.005)
        self.assertEqual(presses, [1, 1, 1, 1])
        self.assertAlmostEqual(self.env.sim_time, 0.020)
        self.assertEqual(messages.count("sim_time"), 4)


if __name__ == "__main__":
    unittest.main()
