import numpy as np
from roboticstoolbox import jtraj
from spatialmath import SE3

class ProgramControl:
    def __init__(self, ur3_controller, tray_controller, robots, fr3_controller=None, handover=None,
                 scanner=None):
        self.ur3 = ur3_controller
        self.kawasaki = tray_controller
        self.robots = robots
        self.fr3 = fr3_controller
        self.handover = handover
        self.robot_names = ("UR3", "Kawasaki", "FR3") if len(robots) == 3 else ("UR3", "Kawasaki")
        self.selected_robot_index = 0
        # one state decides which movements are allowed and which light is on
        self.state = "IDLE"
        self.next_state = "IDLE"
        self.scanner = scanner
        if scanner is not None:
            from scene import glass_bounds
            self.glass_box_bounds = glass_bounds()
        self.curtain_objects = []
        self.estop_active = False
        self.fault_reason = ""
        # the startup joint positions are the agreed home positions for recovery
        self.home_positions = [np.array(robot.q, copy=True) for robot in robots]
        # recovery is slower than normal jogging without taking excessively long
        self.home_speed = np.deg2rad(30)
        self.home_robot = 0
        self.home_path = None
        self.home_step = 0
        self.dirty_glasses_pending = 0
        self.tray_actions = []
        self.tray_location = "PICKUP"
        self.active_tray_action = None

    @property
    def paused(self):
        return self.state == "PAUSED"

    @property
    def stopped(self):
        return self.state == "FAULT"

    @property
    def light_colour(self):
        if self.state == "IDLE":
            return "green"
        if self.state in ("FAULT", "HOMING"):
            return "orange"
        return "red"

    @property
    def manual_enabled(self):
        
        return self.state == "MANUAL" and not self.estop_active and not self.curtain_objects

    @property
    def selected_robot(self):
        return self.robots[self.selected_robot_index]

    def select_robot(self, index):
        self.selected_robot_index = int(index)

    def jog_pose(self, q=None):
        robot = self.selected_robot
        q = robot.q if q is None else q
        
        if self.selected_robot_index == 1:
            return SE3(self.kawasaki.tool_pose(q))
        return robot.fkine(q)

    def jog_jacobian(self, q):
        if self.selected_robot_index == 1:
            return self.kawasaki._calculate_jacobian(q)
        return self.selected_robot.jacob0(q)

    def set_jog_joints(self, q):
        if not self.manual_enabled:
            return
        before = self.selected_robot.q.copy()
        try:
            self._set_robot_joints(self.selected_robot_index, q)
            if not self.check_safety():
                # keep the last clear pose when a proposed movement crosses the curtain
                self._set_robot_joints(self.selected_robot_index, before)
        except Exception as error:
            self._set_robot_joints(self.selected_robot_index, before)
            self.fault(str(error))

    def _set_robot_joints(self, index, q):
        robot = self.robots[index]
        
        if hasattr(robot, "set_joint_angles"):
            robot.set_joint_angles(q)
        else:
            robot.q = np.array(q, dtype=float, copy=True)
        if index == 1:
            self.kawasaki.scene.update_fork()
            if self.kawasaki.carrying_tray:
                pose = self.kawasaki.wrist_pose(robot.q) @ self.kawasaki.tray_offset_from_wrist
                self.kawasaki.scene.set_tray_pose(SE3(pose))
        elif index == 0 and self.ur3.held_glass is not None:
            pose = robot.fkine(robot.q) * self.ur3.tr_ee_held_glass
            self.ur3.held_glass.T = pose.A
            if self.ur3.held_glass_axes is not None:
                self.ur3.held_glass_axes.T = pose.A
        elif index == 2 and self.fr3 is not None:
            if self.fr3.held_glass is not None:
                pose = robot.fkine(robot.q) * self.fr3.tr_ee_held_glass
                self.fr3.held_glass.T = pose.A
                if self.fr3.held_glass_axes is not None:
                    self.fr3.held_glass_axes.T = pose.A

    def toggle_pause(self):
        if self.state in ("RUNNING", "MANUAL", "HOMING"):
            self.next_state = self.state
            self.state = "PAUSED"
            self._zero_velocities()
        elif self.state == "PAUSED" and self.check_safety():
            self.state = self.next_state

    def toggle_manual(self):
        if self.state == "MANUAL":
            self.state = "IDLE"
            self._zero_velocities()
        elif self.state == "IDLE" and self.check_safety() and not self.curtain_objects:
            self.state = "MANUAL"

    def toggle_estop(self):
        if self.estop_active:
            # releasing the button does not clear the latched fault or start motion
            self.estop_active = False
        else:
            self.stop()

    def request_beer(self):
        if self.state in ("IDLE", "RUNNING", "PAUSED"):
            self.ur3.request_beer()

    def deposit_dirty_glass(self):
        
        if self.state in ("IDLE", "RUNNING", "PAUSED"):
            self.dirty_glasses_pending += 1

    def request_wash(self):
        if self.state in ("IDLE", "RUNNING", "PAUSED"):
            self.tray_actions.append("WASH")

    def request_return(self):
        if self.state in ("IDLE", "RUNNING", "PAUSED"):
            self.tray_actions.append("RETURN")

    def update(self, dt, world=None):
        if not self.check_safety() or self.state in ("PAUSED", "MANUAL"):
            return
        if dt <= 0:
            return
        before = [robot.q.copy() for robot in self.robots]
        try:
            if self.state == "IDLE":
                # a queued order cannot start while someone is collecting a glass
                if self.curtain_objects or not self._has_work():
                    return
                self.state = "RUNNING"
            if self.state == "HOMING":
                self._update_homing(dt)
            elif self.state == "RUNNING":
                self._update_tasks(dt, world)
            if not self.check_safety():
                for index, q in enumerate(before):
                    self._set_robot_joints(index, q)
            elif self.state == "RUNNING" and not self._has_work():
                self.state = "IDLE"
        except Exception as error:
            self.fault(str(error))
            for index, q in enumerate(before):
                self._set_robot_joints(index, q)

    def _has_work(self):
        if any(c.state != "IDLE" for c in self._controllers()):
            return True
        if self.active_tray_action is not None:
            return True
        if self.tray_actions:
            action = self.tray_actions[0]
            if (action == "WASH" and self.tray_location == "PICKUP"
                    or action == "RETURN" and self.tray_location == "HANDOVER"):
                return True
        if (self.fr3 is not None and self.dirty_glasses_pending
                and self.tray_location == "PICKUP" and self.kawasaki.scene.free_slot() is not None):
            return True
        return (self.ur3.pending_beers > 0 and self.ur3.tray_ready
                and self.ur3.held_glass is None and self.ur3._next_available_glass() is not None)

    def _update_tasks(self, dt, world):
        # one automatic owner holds the shared tray and glass workspace at a time
        if self.fr3 is not None and self.fr3.state != "IDLE":
            self.fr3.update(dt, world)
            return
        if self.active_tray_action is not None:
            self.kawasaki.update(dt)
            if self.kawasaki.state == "IDLE":
                if self.active_tray_action == "WASH":
                    self.active_tray_action = "HANDOVER"
                    self.kawasaki.start_sequence(self.handover.sequence())
                elif self.active_tray_action == "HANDOVER":
                    self.tray_location = "HANDOVER"
                    self.ur3.glasses = dict(self.kawasaki.scene.glasses)
                    self.active_tray_action = None
                else:
                    if np.linalg.norm(self.kawasaki.scene.tray_pose.t - [0, 0.92, 1.00]) > 0.004:
                        raise RuntimeError("Tray return position was not confirmed")
                    self.tray_location = "PICKUP"
                    self.active_tray_action = None
            return
        if self.kawasaki.state != "IDLE":
            return
        if self.ur3.state != "IDLE":
            self.ur3.update(dt, world)
            return
        if self.tray_actions:
            action = self.tray_actions[0]
            if action == "WASH" and self.tray_location == "PICKUP":
                self.ur3.tray_ready = False
                self.active_tray_action = action
                self.tray_location = "AWAY"
                self.tray_actions.pop(0)
                self.kawasaki.dishwasher_cycle()
                return
            if action == "RETURN" and self.tray_location == "HANDOVER":
                self.ur3.tray_ready = False
                self.active_tray_action = action
                self.tray_location = "AWAY"
                self.tray_actions.pop(0)
                self.kawasaki.start_sequence(self.handover.return_sequence())
                return
        if self.fr3 is not None and self.dirty_glasses_pending and self.tray_location == "PICKUP":
            scene = self.kawasaki.scene
            if np.linalg.norm(scene.tray_pose.A - SE3(0, 0.92, 1.00).A) < 0.004 and scene.free_slot() is not None:
                self.dirty_glasses_pending -= 1
                positions = np.zeros((3, 3), dtype=bool)
                for cell in scene.occupied_slots():
                    positions[cell] = True
                self.fr3.receive_tray(scene.tray_pose, positions)
                self.fr3.request_glass()
                self.fr3.update(dt, world)
                return
        self.ur3.update(dt, world)

    def stop(self):
        self.estop_active = True
        self.fault("E-stop pressed")

    def _controllers(self):
        return [self.ur3, self.kawasaki] + ([self.fr3] if self.fr3 is not None else [])

    def _zero_velocities(self):
        for robot in self.robots:
            robot.qd = np.zeros(robot.n)
            robot.qdd = np.zeros(robot.n)

    def fault(self, reason):
        # latch the first reason so later stop messages do not hide the cause
        if self.state != "FAULT":
            self.fault_reason = str(reason)
            print(f"Safety fault: {self.fault_reason}")
        self.state = self.next_state = "FAULT"
        for controller in self._controllers():
            controller.stop()
        self._zero_velocities()
        # discard interrupted work instead of unexpectedly restarting it after reset
        self.ur3.pending_beers = 0
        if self.fr3 is not None:
            self.fr3.pending_glasses = 0
        self.dirty_glasses_pending = 0
        self.tray_actions.clear()
        self.active_tray_action = None
        self.home_path = None

    def _read_curtain(self):
        if self.scanner is None:
            self.curtain_objects = []
            return
        glasses = list(self.kawasaki.scene.glasses.values()) + list(self.ur3.glasses.values())
        glasses += [self.ur3.held_glass, self.ur3.placed_glass]
        if self.fr3 is not None:
            glasses += [self.fr3.selected_glass, self.fr3.held_glass, self.fr3.placed_glass]
        for glass in glasses:
            if glass is not None:
                self.scanner.add_box("Glass", glass, self.glass_box_bounds)
        # returning robots must be allowed to leave the curtain without triggering it again
        returning_home = self.state == "HOMING" or (self.paused and self.next_state == "HOMING")
        robots = [] if returning_home else list(zip(self.robot_names, self.robots))
        self.curtain_objects = self.scanner.detect(robots)

    def _motion_error(self):
        for name, robot in zip(self.robot_names, self.robots):
            q = np.asarray(robot.q)
            if (not np.isfinite(q).all() or np.any(q < robot.qlim[0] - 1e-8)
                    or np.any(q > robot.qlim[1] + 1e-8)):
                return f"{name}: invalid joints or joint limit"
        # reuse the kawasaki controller's existing obstacle and carried-tray checks
        collision = self.kawasaki.check_collision(
            self.robots[1].q, carrying=self.kawasaki.carrying_tray
        )
        if collision:
            return f"Kawasaki collision: {collision}"
        for name, controller in zip(self.robot_names, self._controllers()):
            if controller.state == "ERROR":
                return f"{name} controller failed"
        return ""

    def check_safety(self):
        try:
            self._read_curtain()
            if self.estop_active:
                self.fault("E-stop pressed")
            elif self.state != "FAULT":
                reason = self._motion_error()
                if reason:
                    self.fault(reason)
                elif self.curtain_objects and self.state != "IDLE":
                    self.fault("Light curtain: " + ", ".join(dict.fromkeys(self.curtain_objects)))
        except Exception as error:
            # losing a detector must stop the program instead of assuming it is clear
            self.fault(f"Safety check failed: {error}")
        return self.state != "FAULT"

    def reset(self):
        if self.state != "FAULT" or self.estop_active:
            return
        try:
            # use the homing detection rules when checking whether recovery can start
            self.state = self.next_state = "HOMING"
            self._read_curtain()
            if self.curtain_objects:
                self.fault("Light curtain: " + ", ".join(dict.fromkeys(self.curtain_objects)))
                return
            reason = self._motion_error()
            if reason:
                self.fault(reason)
                return
            # reset starts recovery only; the cancelled task is never resumed
            self.home_robot = 0
            self.home_path = None
            self.home_step = 0
            self.fault_reason = ""
        except Exception as error:
            self.fault(f"Reset failed: {error}")

    def _update_homing(self, dt):
        if self.home_robot == len(self.robots):
            for controller in self._controllers():
                controller.state = controller.next_state = "IDLE"
                controller.elapsed = 0.0
                controller.step = 0
            self.kawasaki.stopped = False
            # a reset cannot claim a tray handover that was interrupted
            position = self.kawasaki.scene.tray_pose.t
            if self.kawasaki.carrying_tray:
                self.tray_location = "CARRIED"
            elif np.linalg.norm(position - [0, 0.92, 1.00]) < 0.004:
                self.tray_location = "PICKUP"
            else:
                self.tray_location = "AWAY"
            self.ur3.tray_ready = False
            if self.fr3 is not None:
                self.fr3.tray_ready = self.tray_location == "PICKUP"
            self.state = self.next_state = "IDLE"
            return
        robot = self.robots[self.home_robot]
        if self.home_path is None:
            target = self.home_positions[self.home_robot]
            distance = np.max(np.abs(target - robot.q))
            if distance < 1e-8:
                self.home_robot += 1
                return
            # allow extra time for jtraj's smooth acceleration and deceleration
            # twice the average travel time keeps the peak joint speed below the limit
            steps = max(2, int(np.ceil(2 * distance / (self.home_speed * dt))) + 1)
            self.home_path = jtraj(robot.q, target, steps).q
            lower, upper = robot.qlim
            if np.any(self.home_path < lower - 1e-8) or np.any(self.home_path > upper + 1e-8):
                raise RuntimeError("Home trajectory exceeds joint limits")
            if self.home_robot == 1:
                # check the path at about one-degree intervals before moving the tray robot
                stride = max(1, int(np.deg2rad(1) / (self.home_speed * dt)))
                self.kawasaki.check_trajectory(
                    self.home_path[::stride], carrying=self.kawasaki.carrying_tray,
                    label="slow return home"
                )
            self.home_step = 0
        self._set_robot_joints(self.home_robot, self.home_path[self.home_step])
        self.home_step += 1
        if self.home_step == len(self.home_path):
            self.home_path = None
            self.home_robot += 1

    def states(self):
        states = {
            "Program": self.state,
            "Safety": self.fault_reason or ("Curtain blocked" if self.curtain_objects else "Clear"),
            "UR3": self.ur3.state,
            "Kawasaki": self.kawasaki.state,
            "FR3": self.fr3.state if self.fr3 is not None else (f"QUEUED ({self.dirty_glasses_pending})" if self.dirty_glasses_pending else "IDLE"),
            "Dirty glasses": str(self.dirty_glasses_pending),
            "Beers requested": f"{self.ur3.total_beer_requests} | queued: {self.ur3.pending_beers}",
        }
        # show the actual recovery or safety status instead of an old task state
        for index, name in enumerate(self.robot_names):
            if self.state == "HOMING" or (self.paused and self.next_state == "HOMING"):
                if index < self.home_robot:
                    states[name] = "HOME"
                elif index == self.home_robot:
                    states[name] = "RETURNING HOME"
                else:
                    states[name] = "WAITING TO HOME"
            if self.paused:
                states[name] = f"PAUSED ({states[name]})"
            elif self.state == "FAULT":
                states[name] = "STOPPED (FAULT)"
            elif self.state == "MANUAL" and index == self.selected_robot_index:
                states[name] = "MANUAL READY"
        return states
