import numpy as np
from spatialmath import SE3

class ProgramControl:
    def __init__(self, ur3_controller, tray_controller, robots):
        self.ur3 = ur3_controller
        self.kawasaki = tray_controller
        self.robots = robots
        self.robot_names = ("UR3", "Kawasaki")
        self.selected_robot_index = 0
        self.phase = "STARTING"
        self.stopped = False
        self.paused = False
        self.dirty_glasses_pending = 0

    @property
    def manual_enabled(self):
        
        return self.stopped and self.ur3.state == "STOPPED" and self.kawasaki.stopped

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
        robot = self.selected_robot
        
        if hasattr(robot, "set_joint_angles"):
            robot.set_joint_angles(q)
        else:
            robot.q = np.array(q, dtype=float, copy=True)
        if self.selected_robot_index == 1:
            self.kawasaki.scene.update_fork()
            if self.kawasaki.carrying_tray:
                pose = self.kawasaki.wrist_pose(robot.q) @ self.kawasaki.tray_offset_from_wrist
                self.kawasaki.scene.set_tray_pose(SE3(pose))
        elif self.selected_robot_index == 0 and self.ur3.held_glass is not None:
            pose = robot.fkine(robot.q) * self.ur3.tr_ee_held_glass
            self.ur3.held_glass.T = pose.A
            if self.ur3.held_glass_axes is not None:
                self.ur3.held_glass_axes.T = pose.A

    def toggle_pause(self):
        
        if not self.stopped:
            self.paused = not self.paused

    def request_beer(self):
        if not self.stopped:
            self.ur3.request_beer()

    def deposit_dirty_glass(self):
        
        if not self.stopped:
            self.dirty_glasses_pending += 1

    def stop(self):
        
        self.stopped = True
        self.paused = False
        self.ur3.stop()
        self.kawasaki.stop()
        for robot in self.robots:
            robot.qd = np.zeros(robot.n)
            robot.qdd = np.zeros(robot.n)

    def states(self):
        program = self.phase
        if self.stopped:
            program = "STOPPED"
        elif self.paused:
            program = "PAUSED"
        elif self.phase == "RUNNING":
            if self.ur3.state == "ERROR":
                program = "ERROR"
            elif self.ur3.state != "IDLE":
                program = "SERVING"
            elif self.ur3.pending_beers:
                program = "WAITING FOR GLASS" if self.ur3.tray_ready else "WAITING FOR TRAY"
            else:
                program = "IDLE"
        states = {
            "Program": program,
            "UR3": self.ur3.state,
            "Kawasaki": self.kawasaki.state,
            "Dirty glasses": str(self.dirty_glasses_pending),
            "Beers requested": f"{self.ur3.total_beer_requests} | queued: {self.ur3.pending_beers}",
        }
        if self.paused:
            for name in self.robot_names:
                states[name] = f"PAUSED / {states[name]}"
        return states
