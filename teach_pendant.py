"""Live joint and Cartesian velocity jogging for the Swift pendant."""

import numpy as np
import swift


class TeachPendant:
    def __init__(self, env, robot, control=None):
        self.control = control
        self.robots = control.robots if control is not None else [robot]
        self.robot_names = control.robot_names if control is not None else ("UR3",)
        self.robot_index = 0
        self.robot = self.robots[0]
        self.joint_index = 0
        self.command = None
        self.linear_speed = 0.2  # metres per second at full speed
        self.rotation_speed = np.deg2rad(30)  # radians per second at full speed
        self.joint_speed = 2  # radians per second at full speed
        self.damping = 0.01
        self._read_joint_limits()

        self.state_labels = {}
        for name in ("Program", "Safety", "UR3", "Kawasaki", "FR3", "Dirty glasses", "Beers requested"):
            self.state_labels[name] = swift.Label("")
            env.add(self.state_labels[name])
        env.add(swift.Button(self.request_beer, desc="Request beer"))
        env.add(swift.Button(self.deposit_dirty_glass, desc="Deposit dirty glass"))
        env.add(swift.Button(lambda _: self.control.request_wash() if self.control else None, desc="Run wash cycle"))
        env.add(swift.Button(lambda _: self.control.request_return() if self.control else None, desc="Return tray"))
        self.estop_button = swift.Button(self.stop_all, desc="E-STOP")
        env.add(self.estop_button)
        env.add(swift.Button(self.reset, desc="RESET"))
        self.manual_button = swift.Button(self.toggle_manual, desc="MANUAL")
        env.add(self.manual_button)
        self.pause_button = swift.Button(self.toggle_pause, desc="PAUSE")
        env.add(self.pause_button)
        self.status_label = swift.Label("Hold a jog button to move")
        env.add(self.status_label)
        env.add(swift.Select(
            self.select_robot, desc="Robot", options=list(self.robot_names), value=0,
        ))
        self.q_label = swift.Label("")
        self.xyz_label = swift.Label("")
        env.add(self.q_label)
        env.add(self.xyz_label)
        self.joint_select = swift.Select(
            self.select_joint, desc="Joint",
            options=[f"q{i + 1}" for i in range(self.robot.n)], value=0,
        )
        env.add(self.joint_select)
        self.speed_slider = swift.Slider(
            lambda _: None, min=1, max=100, step=1, value=50,
            desc="Speed", unit="%",
        )
        env.add(self.speed_slider)

        # the page sends button press and release events for these jog controls
        for direction in (1, -1):
            env.add(swift.Button(
                lambda _, d=direction: self.jog_joint(d),
                desc=f"{'+' if direction > 0 else '-'}q",
            ))
        for rotation in (False, True):
            for axis, name in enumerate("XYZ"):
                for direction in (1, -1):
                    env.add(swift.Button(
                        lambda _, a=axis, d=direction, r=rotation: self.jog(a, d, r),
                        desc=f"{'+' if direction > 0 else '-'}{'R' if rotation else ''}{name}",
                    ))
        # keep release last so a quick press and release is processed in order
        env.add(swift.Button(self.release_jog, desc="Release jog"))
        self.sync()

    @property
    def manual_enabled(self):
        return self.control is None or self.control.manual_enabled

    def sync(self):
        states = self.control.states() if self.control is not None else {
            "Program": "IDLE", "UR3": "IDLE", "Kawasaki": "IDLE", "Dirty glasses": "0",
            "Beers requested": "0 | queued: 0",
        }
        if self.manual_enabled and self.command is not None:
            states["Program"] = "MANUAL"
            states[self.robot_names[self.robot_index]] = "JOGGING"
        for name, value in states.items():
            # keep the system state separate from each robot's task or recovery state
            if name == "Program":
                title = "System state"
            elif name in self.robot_names:
                title = f"{name} state"
            else:
                title = name
            text = f"{title}: {value}"
            if self.state_labels[name].desc != text:
                self.state_labels[name].desc = text
        if self.control is not None:
            self.estop_button.desc = "RELEASE E-STOP" if self.control.estop_active else "E-STOP"
            self.manual_button.desc = "AUTO" if self.control.state == "MANUAL" else "MANUAL"
            text = "RESUME" if self.control.paused else "PAUSE"
            if text != self.pause_button.desc:
                self.pause_button.desc = text
            if not self.manual_enabled:
                self.command = None
                if self.control.state == "FAULT":
                    self.status_label.desc = "Release E-stop, clear fault, then RESET"
                elif self.control.state == "HOMING":
                    self.status_label.desc = "Returning home slowly"
                elif self.control.paused:
                    self.status_label.desc = "Paused"
                else:
                    self.status_label.desc = "Select MANUAL from idle to jog"
        robot_name = self.robot_names[self.robot_index]
        text = f"{robot_name} joints (deg): " + " ".join(f"{value:+.1f}" for value in np.rad2deg(self.robot.q))
        if text != self.q_label.desc:
            self.q_label.desc = text
        xyz = self._pose(self.robot.q).t
        text = f"{robot_name} EE (m): " + "  ".join(f"{axis} {value:.3f}" for axis, value in zip("XYZ", xyz))
        if text != self.xyz_label.desc:
            self.xyz_label.desc = text

    def request_beer(self, _=None):
        if self.control is not None:
            self.control.request_beer()
        self.stop()

    def deposit_dirty_glass(self, _=None):
        if self.control is not None:
            self.control.deposit_dirty_glass()
        self.sync()

    def stop_all(self, _=None):
        if self.control is not None:
            self.control.toggle_estop()
        self.stop()

    def reset(self, _=None):
        self.command = None
        if self.control is not None:
            self.control.reset()
        self.sync()

    def toggle_manual(self, _=None):
        self.command = None
        if self.control is not None:
            self.control.toggle_manual()
        self.sync()

    def toggle_pause(self, _=None):
        if self.control is not None:
            self.control.toggle_pause()
        self.stop()

    def select_robot(self, index):
        index = int(index)
        if index == self.robot_index:
            return
        # changing robots cancels the old command before selecting the new arm
        self.command = None
        self.robot_index = index
        self.robot = self.robots[index]
        if self.control is not None:
            self.control.select_robot(index)
        self.joint_index = 0
        self.joint_select.value = 0
        self._read_joint_limits()
        self.status_label.desc = "Hold a jog button to move"
        self.sync()

    def select_joint(self, index):
        if int(index) != self.joint_index:
            self.command = None
            self.joint_index = int(index)

    def _read_joint_limits(self):
        self.joint_limits = np.array(self.robot.qlim, dtype=float, copy=True)
        if self.joint_limits.shape != (2, self.robot.n):
            self.joint_limits = np.tile([[-2 * np.pi], [2 * np.pi]], (1, self.robot.n))
        self.joint_limits[0] = np.where(
            np.isfinite(self.joint_limits[0]), self.joint_limits[0], -2 * np.pi,
        )
        self.joint_limits[1] = np.where(
            np.isfinite(self.joint_limits[1]), self.joint_limits[1], 2 * np.pi,
        )

    def _pose(self, q):
        return self.control.jog_pose(q) if self.control is not None else self.robot.fkine(q)

    def _set_joints(self, q):
        if self.control is not None:
            self.control.set_jog_joints(q)
        elif hasattr(self.robot, "set_joint_angles"):
            self.robot.set_joint_angles(q)
        else:
            self.robot.q = q

    def jog_joint(self, direction):
        if self.manual_enabled:
            self.command = ("joint", self.joint_index, direction)
            self.status_label.desc = f"Jog q{self.joint_index + 1}"

    def jog(self, axis, direction, rotation=False):
        if self.manual_enabled:
            # replace the current velocity command; there is no target queue
            self.command = ("cartesian", axis + (3 if rotation else 0), direction)
            self.status_label.desc = "Rotate (world axes)" if rotation else "Jog (world axes)"

    def release_jog(self, _=None):
        self.command = None
        self.status_label.desc = "Hold a jog button to move"
        self.sync()

    def stop(self, _=None):
        self.release_jog()

    def update(self, dt):
        if not self.manual_enabled:
            self.command = None
            return
        if self.command is None or dt <= 0:
            return
        dt = min(float(dt), 0.05)
        scale = float(self.speed_slider.value) / 100.0
        kind, axis, direction = self.command
        q = np.array(self.robot.q, dtype=float, copy=True)
        if kind == "joint":
            qdot = np.zeros(self.robot.n)
            qdot[axis] = direction * self.joint_speed * scale
        else:
            # rmrc maps the requested world velocity to joint velocity
            velocity = np.zeros(6)
            speed = self.linear_speed if axis < 3 else self.rotation_speed
            velocity[axis] = direction * speed * scale
            jacobian = (self.control.jog_jacobian(q) if self.control is not None
                        else self.robot.jacob0(q))
            qdot = np.linalg.solve(
                jacobian.T @ jacobian + self.damping**2 * np.eye(self.robot.n),
                jacobian.T @ velocity,
            )
            qdot /= max(1.0, np.max(np.abs(qdot)) / (self.joint_speed * scale))
        next_q = q + dt * qdot
        if (not np.all(np.isfinite(next_q))
                or np.any(next_q < self.joint_limits[0])
                or np.any(next_q > self.joint_limits[1])):
            self.command = None
            self.status_label.desc = "Joint limit"
            if self.control is not None:
                self.control.fault("Manual motion reached a joint limit")
            return
        self._set_joints(next_q)
