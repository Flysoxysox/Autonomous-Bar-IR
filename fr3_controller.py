import numpy as np
import roboticstoolbox as rtb
from scipy import linalg
from scipy.spatial.transform import Rotation
from spatialmath import SE3
from ir_support_extra_parts.parts import part_mesh
from scene import glass_bounds


# a transform stores the position and orientation of one frame in another frame
# world is the shared reference frame for the robots the cup and the tray
# ee means end effector which is the frame at the end of the robot arm
# multiplying transforms joins the frames in order from left to right
# an inverse reverses the direction of a transform
# putting a translation before a world pose moves along the world axes
# putting a translation after a pose moves along the local axes of that pose


class FR3Controller:
    """Collect dirty glasses and place them in the shared tray."""

    def __init__(self, fr3, dt=0.02):
        self.fr3 = fr3
        # each trajectory sample represents this many seconds of robot motion
        self.dt = dt
        self.min_manip_measure = 0.01
        # save the current flat joint configuration as the exact home pose
        self.q_home = np.deg2rad([0, -180, 0, 0, 0, 0])
        self.joint_limits = np.array(fr3.qlim, dtype=float, copy=True)
        self.state = "IDLE"
        self.next_state = "IDLE"
        self.elapsed = 0.0
        self.trajectory = None
        self.step = 0
        self.wait_time = 0.0

        self.glasses = {}
        self.glass_positions = np.zeros((3, 3), dtype=bool)
        self.selected_glass = None
        self.selected_slot = None
        self.tray_ready = False
        self.pending_glasses = 0
        self.total_glass_requests = 0
        self.workcell = None
        self.held_glass = None
        self.held_glass_axes = None
        self.tr_ee_held_glass = None
        self.placed_glass = None
        self.placed_glass_axes = None
        self.tr_ee_placed_glass = None
        self.q_placed_glass = None
        # this point lies on the upper bar closer to the fr3 than the beer target
        self.collection_position = np.array([-0.20, 1.20, 1.30])

    def configure_task_poses(self, tr_world_tray, glasses, workcell=None):
        """Receive scene references without starting motion."""
        self.tr_tray = SE3(tr_world_tray)
        self.glasses = dict(glasses)
        self.workcell = workcell

    def select_glass(self, glass, q0=None):
        """Read the actual cup mesh and calculate the pickup transforms."""
        # an occupied slot towards negative x would obstruct the placement approach
        occupied = self.workcell.occupied_slots()
        if self.selected_slot is None or self.selected_slot in occupied:
            self.state = "ERROR"
            raise RuntimeError("The selected tray slot is not free")
        if any(column < self.selected_slot[1] for _, column in occupied):
            self.state = "ERROR"
            raise RuntimeError("A tray glass blocks the placement approach from negative x")
        self.selected_glass = glass
        self.tr_world_glass = SE3(glass.T)
        lower, upper = glass_bounds()
        # grip the middle of the glass from five centimetres behind its centre
        grip_centre = self.tr_world_glass.t + [0, -0.05, (upper[2] - lower[2]) / 2]
        # end effector x points up and end effector z points along world positive y
        ee_rotation = np.column_stack(([0, 0, 1], [1, 0, 0], [0, 1, 0]))
        self.tr_glass_grasp = SE3.Rt(ee_rotation, grip_centre)
        # right multiplication retreats five centimetres along end effector negative z
        self.tr_above_glass = self.tr_glass_grasp * SE3.Tz(-0.05)

    def change_state(self, new_state):
        """Choose the trajectory and next state explicitly in each branch."""
        # each state plans its own motion and chooses the state that follows it
        self.state = new_state
        self.elapsed = 0.0
        self.step = 0
        self.trajectory = None
        self.wait_time = 0.0

        if new_state == "IDLE":
            self.next_state = "IDLE"

        elif new_state == "APPROACH_GLASS":
            # this joint guess is in radians and helps choose the pickup posture
            q0 = np.deg2rad([-3.6, -80.3, 127.9, -49.0, 99.8, 0.0])
            self.select_glass(self.selected_glass, q0=q0)
            solution = self.fr3.ikine_LM(self.tr_above_glass, q0=q0, joint_limits=True)
            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Approach IK failed: {solution.reason}")
            self.trajectory = rtb.jtraj(self.fr3.q, solution.q, 200).q
            self.next_state = "COLLECT_GLASS"

        elif new_state == "COLLECT_GLASS":
            # rmrc moves forwards five centimetres along end effector positive z
            self.trajectory = self.plan_rmrc(self.tr_glass_grasp, steps=120)
            self.next_state = "RAISE_GLASS"

        elif new_state == "RAISE_GLASS":
            # collection has finished so save the actual grasp before lifting
            self.held_glass = self.selected_glass
            ee = self.fr3.fkine(self.fr3.q)
            self.tr_ee_held_glass = ee.inv() * SE3(self.held_glass.T)
            # left multiplication lifts five centimetres along world positive z
            ee_goal = SE3.Tz(0.05) * self.fr3.fkine(self.fr3.q)
            self.trajectory = self.plan_rmrc(ee_goal, steps=120)
            self.next_state = "APPROACH_TRAY"

        elif new_state == "APPROACH_TRAY":
            self.tr_glass_target = self.workcell.slot_pose(self.selected_slot) * SE3.Rz(-np.pi / 2)
            # account for the held glass offset at ten centimetres above the slot
            ee_goal = SE3.Tz(0.10) * self.tr_glass_target * self.tr_ee_held_glass.inv()
            q0 = np.deg2rad([-3.6, -80.3, 127.9, -49.0, 99.8, 0.0])
            solution = self.fr3.ikine_LM(ee_goal, q0=q0, joint_limits=True)
            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Tray approach IK failed: {solution.reason}")
            self.trajectory = rtb.jtraj(self.fr3.q, solution.q, 240).q
            self.next_state = "LOWER_TRAY"

        elif new_state == "LOWER_TRAY":
            # lower the held glass ten centimetres along world negative z
            ee_goal = SE3.Tz(-0.10) * self.fr3.fkine(self.fr3.q)
            self.trajectory = self.plan_rmrc(ee_goal, steps=120)
            self.next_state = "RAISE_TRAY"

        elif new_state == "RAISE_TRAY":
            # lowering has finished so release before moving the empty gripper
            self._release_glass()
            ee_goal = SE3.Tz(0.10) * self.fr3.fkine(self.fr3.q)
            self.trajectory = self.plan_rmrc(ee_goal, steps=120)
            self.next_state = "RETURN_HOME"

        elif new_state == "RETURN_HOME":
            self.trajectory = rtb.jtraj(self.fr3.q, self.q_home, 240).q
            self.trajectory[-1] = self.q_home.copy()
            self.next_state = "IDLE"

        elif new_state == "STOPPED":
            self.next_state = "STOPPED"

        elif new_state == "ERROR":
            self.next_state = "ERROR"

        else:
            self.state = "ERROR"
            raise ValueError(f"Unknown or unimplemented state: {new_state}")

        if self.trajectory is not None:
            # reject the whole planned motion if any joint sample exceeds its limits
            lower, upper = self.joint_limits
            if np.any(self.trajectory < lower) or np.any(self.trajectory > upper):
                self.trajectory = None
                self.state = "ERROR"
                raise RuntimeError(f"{new_state} trajectory exceeds joint limits")

    def _release_glass(self):
        """Save the placed pose and stop the glass following the wrist."""
        self.workcell.register_glass(self.selected_slot, self.held_glass)
        self.placed_glass = self.held_glass
        self.placed_glass_axes = self.held_glass_axes
        self.tr_ee_placed_glass = SE3(self.tr_ee_held_glass)
        self.q_placed_glass = self.fr3.q.copy()
        self.held_glass = None
        self.held_glass_axes = None
        self.tr_ee_held_glass = None

    def plan_rmrc(self, target_pose, steps=100, q_start=None, tool_offset=None):
        """Use RMRC and Damped Least Squares to follow a Cartesian trajectory."""
        if steps < 2:
            raise ValueError("RMRC needs at least two samples")
        delta_t = self.dt
        min_manip_measure = self.min_manip_measure
        m = np.zeros([1, steps - 1])
        error = np.empty([6, steps - 1])

        if q_start is None:
            q_start = self.fr3.q

        # create the cartesian trajectory for the end effector or held glass
        if tool_offset is None:
            poses = rtb.ctraj(self.fr3.fkine(q_start), target_pose, steps)
        else:
            cup_start = self.fr3.fkine(q_start) * tool_offset
            cup_poses = rtb.ctraj(cup_start, target_pose, steps)
            offset_inverse = tool_offset.inv()
            poses = [pose * offset_inverse for pose in cup_poses]

        q_matrix = [np.array(q_start, dtype=float, copy=True)]

        # use rmrc to solve joint velocities at each time step
        for i in range(steps - 1):
            actual = self.fr3.fkine(q_matrix[i])
            desired = poses[i + 1]
            xdot = np.r_[desired.t - actual.t, Rotation.from_matrix(desired.R @ actual.R.T).as_rotvec()] / delta_t
            J = self.fr3.jacob0(q_matrix[i])
            m[:, i] = np.sqrt(max(0.0, linalg.det(J @ J.T)))

            # apply damped least squares for configurations with low manipulability
            if m[0, i] < min_manip_measure:
                qdot = linalg.inv(J.T @ J + 0.01**2 * np.eye(self.fr3.n)) @ J.T @ xdot
            else:
                qdot = linalg.inv(J) @ xdot

            error[:, i] = xdot - J @ qdot
            q_matrix.append(q_matrix[i] + delta_t * qdot)

        self.rmrc_manipulability = m
        self.rmrc_velocity_error = error
        return np.asarray(q_matrix)

    def request_glass(self):
        """Remember one order for update to start when the tray is available."""
        if self.state == "STOPPED":
            return
        self.pending_glasses += 1
        self.total_glass_requests += 1

    def stop(self):
        # cancel playback and keep the controller stopped until restart
        self.state = self.next_state = "STOPPED"
        self.trajectory = None

    def _next_available_glass(self):
        """Choose the next free slot for the requested dirty glass."""
        return self.workcell.free_slot()

    def pick_glass(self):
        """Start pickup with the approach branch defining its own joint guess."""
        if self.state != "IDLE":
            raise RuntimeError("Controller must be IDLE before starting pickup")
        if self.held_glass is not None:
            raise RuntimeError("Release the held glass before picking another one")
        self.selected_glass = part_mesh("BeerGlass")
        lower, _ = glass_bounds()
        position = self.collection_position.copy()
        position[2] -= lower[2]
        self.selected_glass.T = SE3.Trans(position).A
        self.workcell.simulator.add(self.selected_glass)
        self.change_state("APPROACH_GLASS")

    def update(self, dt, world=None):
        """Execute one sample when due with the caller stepping Swift separately."""
        if self.state == "STOPPED":
            return
        if self.state == "IDLE" and self.tray_ready and self.pending_glasses > 0 and self.held_glass is None:
            cell = self._next_available_glass()
            if cell is not None:
                self.selected_slot = cell
                self.pick_glass()
                self.pending_glasses -= 1

        if self.state not in ("IDLE", "ERROR"):
            if self.trajectory is None:
                self.elapsed += dt
                if self.elapsed + 1e-12 >= self.wait_time:
                    self.change_state(self.next_state)
            else:
                self.elapsed += dt
                if self.elapsed >= self.dt:
                    self.elapsed -= self.dt
                    if self.step < len(self.trajectory):
                        self.fr3.q = self.trajectory[self.step]
                        self.step += 1
                    else:
                        self.change_state(self.next_state)

        # the held cup follows the saved offset during automatic and manual movement
        if self.held_glass is not None:
            ee = self.fr3.fkine(self.fr3.q)
            self.held_glass.T = (ee * self.tr_ee_held_glass).A
            if self.held_glass_axes is not None:
                self.held_glass_axes.T = self.held_glass.T

    def receive_tray(self, tr_tray, glass_positions):
        """Confirm the tray pose and store the occupied slot mask."""
        positions = np.array(glass_positions, dtype=bool, copy=True)
        if positions.shape != (3, 3):
            raise ValueError("glass_positions must be a 3 x 3 availability mask")
        self.tr_tray = SE3(tr_tray)
        self.glass_positions = positions
        self.tray_ready = True

    def recieve_tray(self, tr_tray, glass_positions):
        """Keep the original spelling working for existing callers."""
        self.receive_tray(tr_tray, glass_positions)
