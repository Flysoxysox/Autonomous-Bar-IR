import numpy as np
import roboticstoolbox as rtb
from scipy import linalg
from scipy.spatial.transform import Rotation
from spatialmath import SE3


# a transform stores the position and orientation of one frame in another frame
# world is the shared reference frame for the robots the cup and the tap
# ee means end effector which is the frame at the end of the robot arm
# multiplying transforms joins the frames in order from left to right
# an inverse reverses the direction of a transform
# putting a translation before a world pose moves along the world axes
# putting a translation after a pose moves along the local axes of that pose


class UR3Controller:
    """Queue beer orders and serve them after the tray handoff."""

    def __init__(self, ur3, dt=0.02):
        self.ur3 = ur3
        # each trajectory sample represents this many seconds of robot motion
        self.dt = dt
        # a small manipulability value means some directions are difficult to move in
        self.min_manip_measure = 0.01
        self.q_idle = np.array([0, -np.pi / 2, 0, 0, 0, 0])
        self.joint_limits = np.array(ur3.qlim, dtype=float, copy=True)
        self.state = "IDLE"
        self.next_state = "IDLE"
        self.elapsed = 0.0
        self.trajectory = None
        self.step = 0
        # this state timer holds the arm still without blocking the simulation loop
        self.wait_time = 0.0
        # change this value to adjust how many seconds the beer is given to fill
        self.beer_fill_time = 2.0
        # lift the filled cup five centimetres along world z after grasping it
        self.beer_lift_height = 0.05
        # this is the final cup origin position in world coordinates in metres
        self.beer_target_position = np.array([0.45, 1.2, 1.3])
        # increase this time in seconds to make the straight beer movement slower
        self.beer_move_time = 2.5

        self.glasses = {}
        self.glass_positions = np.ones((3, 3), dtype=bool)
        self.selected_glass = (2, 0)
        self.tray_ready = False
        self.pending_beers = 0
        self.total_beer_requests = 0

        # the grip point is five centimetres from the cup origin along cup local z
        self.tr_cup_grip = SE3.Tz(0.05)
        # the imaginary gripper centre is five centimetres along end effector local z
        self.tr_ee_gripper = SE3.Tz(0.05)
        # leave fifteen centimetres between the cup grip and the pickup approach pose
        self.glass_clearance = 0.15
        self.workcell = None
        self.held_glass = None
        self.held_glass_axes = None
        # this stores the fixed cup pose relative to the end effector while held
        self.tr_ee_held_glass = None
        # save the released cup and its grasp so the arm can collect the same cup later
        self.placed_glass = None
        self.placed_glass_axes = None
        self.tr_ee_placed_glass = None
        self.q_placed_glass = None
        self.tap = None
        # these handle angles match the arm samples during opening and closing
        self.tap_angles = None
        # this stores the end effector pose relative to the moving handle frame
        self.tr_handle_ee = None

    def configure_task_poses(self, tr_world_tray, tr_world_tap, tr_world_hinge,
                             glasses, workcell=None, tap=None):
        """Receive scene references without starting motion."""
        # these are the tray and tap stand poses expressed in the world frame
        self.tr_tray = SE3(tr_world_tray)
        self.tr_tap = SE3(tr_world_tap)
        # place the cup target eighteen centimetres along tap local y
        # then turn the target frame half a turn about its local z axis
        self.tr_below_tap = self.tr_tap * SE3.Ty(0.18) * SE3.Rz(np.pi)
        # the hinge frame is at the handle pivot with local z along the rotation axis
        self.tr_hinge = SE3(tr_world_hinge)
        self.glasses = dict(glasses)
        self.workcell = workcell
        # keep the actual tap object so its joint angle can change during playback
        self.tap = tap
        self.select_glass(*self.selected_glass)

    def select_glass(self, row, column, q0=None):
        """Read the actual cup mesh and calculate the pickup transforms."""
        cell = (row, column)
        if cell not in self.glasses:
            raise ValueError(f"No scene glass at cell {cell}; configure the workcell glasses first")
        if not self.glass_positions[cell]:
            raise ValueError(f"No available glass at cell {cell}")
        self.selected_glass = cell
        # read the current cup mesh pose so any tray movement is included
        self.tr_world_glass = SE3(self.glasses[cell].T)
        # transform the cup grip point into world coordinates
        grip_centre = (self.tr_world_glass * self.tr_cup_grip).t
        # the third rotation column gives the cup local z axis in world coordinates
        cup_z = self.tr_world_glass.R[:, 2]

        # align end effector x with the length of the cup
        # use the guessed arm pose to choose which side of the cup to approach
        guess_rotation = self.ur3.fkine(self.ur3.q if q0 is None else q0).R
        # remove the part parallel to the cup axis to get a sideways approach
        ee_z = guess_rotation[:, 2] - cup_z * np.dot(guess_rotation[:, 2], cup_z)
        # use another axis if the first choice is almost parallel to the cup
        if np.linalg.norm(ee_z) < 1e-8:
            ee_z = guess_rotation[:, 1] - cup_z * np.dot(guess_rotation[:, 1], cup_z)
        # make the approach axis unit length then find the remaining perpendicular axis
        ee_z /= np.linalg.norm(ee_z)
        ee_y = np.cross(ee_z, cup_z)
        # build the world grip pose using the three axes and the cup grip position
        self.tr_world_grip = SE3.Rt(np.column_stack((cup_z, ee_y, ee_z)), grip_centre)
        # the cups start upside down so negative cup z points above the rack
        self.tr_world_grip_above = (
            SE3.Trans(-self.glass_clearance * cup_z) * self.tr_world_grip
        )
        # remove the gripper offset to turn each grip target into an end effector target
        self.tr_glass_grasp = self.tr_world_grip * self.tr_ee_gripper.inv()
        self.tr_above_glass = self.tr_world_grip_above * self.tr_ee_gripper.inv()

    def change_state(self, new_state):
        """Choose the trajectory and next state explicitly in each branch."""
        # each state plans its own motion and chooses the state that follows it
        self.state = new_state
        self.elapsed = 0.0
        self.step = 0
        self.trajectory = None
        self.wait_time = 0.0
        # opening and closing supply a matching tap angle for every arm sample
        self.tap_angles = None

        if new_state == "IDLE":
            self.next_state = "IDLE"

        elif new_state == "APPROACH_GLASS":
            # this joint guess is in radians and helps choose the pickup posture
            q0 = np.array([1.58, -1.13, 2.29, -1.10, 1.57, 0.00])
            self.select_glass(*self.selected_glass, q0=q0)
            # inverse kinematics finds joint angles that reach the end effector target
            solution = self.ur3.ikine_LM(
                self.tr_above_glass, q0=q0, joint_limits=True,
            )
            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Approach IK failed: {solution.reason}")
            # joint interpolation gives a smooth arm motion between the two joint poses
            self.trajectory = rtb.jtraj(self.ur3.q, solution.q, 100).q
            self.next_state = "LOWER_TO_GLASS"

        elif new_state == "LOWER_TO_GLASS":
            # rmrc follows a straight cartesian approach to the cup grip
            self.trajectory = self.plan_rmrc(self.tr_glass_grasp, steps=100)
            self.next_state = "GRASP_GLASS"

        elif new_state == "GRASP_GLASS":
            if self.held_glass is not None:
                self.state = "ERROR"
                raise RuntimeError("The UR3 is already holding a glass")
            self.held_glass = self.glasses[self.selected_glass]
            # forward kinematics gives the current end effector pose in the world
            ee = self.ur3.fkine(self.ur3.q)
            # express the cup in the end effector frame to keep their current spacing
            self.tr_ee_held_glass = ee.inv() * SE3(self.held_glass.T)
            if self.workcell is not None:
                self.held_glass_axes = self.workcell.glass_axes[self.selected_glass]
                # stop the tray from moving this cup and its displayed axes
                self.workcell.tray_objects[:] = [
                    (shape, pose) for shape, pose in self.workcell.tray_objects
                    if shape is not self.held_glass and shape is not self.held_glass_axes
                ]
            self.glass_positions[self.selected_glass] = False
            self.next_state = "LIFT_FROM_RACK"

        elif new_state == "LIFT_FROM_RACK":
            # retrace the pickup approach while the cup follows the arm
            self.trajectory = self.plan_rmrc(self.tr_above_glass, steps=100)
            self.next_state = "APPROACH_TAP"

        elif new_state == "APPROACH_TAP":
            # move the cup target two centimetres upwards along world z
            # remove the held cup offset to find the required end effector pose
            ee_goal = SE3.Tz(0.02) * self.tr_below_tap * self.tr_ee_held_glass.inv()
            print(self.ur3.q)
            q0 = np.array([3.12440996, -0.47280468, 1.00078927, -0.52798016, 0.02781334, 1.57079189])  # radians
            solution = self.ur3.ikine_LM(ee_goal, q0=q0, joint_limits=True, mask=[1, 1, 1, 1, 1, 1])
            print("IK solution:", solution.q)

            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Approach tap IK failed: {solution.reason}")
            q_target = solution.q.copy()
            # the solver can return negative angles for poses just beyond positive pi
            # adding whole turns keeps the same pose and chooses the angle nearest the guess
            q_target[0] += 2 * np.pi * np.round(
                (q0[0] - q_target[0]) / (2 * np.pi)
            )
            if not self.joint_limits[0, 0] <= q_target[0] <= self.joint_limits[1, 0]:
                self.state = "ERROR"
                raise RuntimeError("Approach tap q1 exceeds joint limits")
            self.trajectory = rtb.jtraj(self.ur3.q, q_target, 200).q
            self.next_state = "LOWER_TO_TAP"

        elif new_state == "LOWER_TO_TAP":
            # put the cup origin at the target while accounting for the actual grasp offset
            ee_goal = self.tr_below_tap * self.tr_ee_held_glass.inv()
            self.trajectory = self.plan_rmrc(ee_goal, steps=100)
            self.next_state = "RELEASE_GLASS"

        elif new_state == "RELEASE_GLASS":
            self._release_glass()
            self.next_state = "RETREAT_FROM_CUP"

        elif new_state == "RETREAT_FROM_CUP":
            # multiplication on the right moves eight centimetres along end effector negative z
            ee_goal = self.ur3.fkine(self.ur3.q) * SE3.Tz(-0.08)
            self.trajectory = self.plan_rmrc(ee_goal, steps=100)
            self.next_state = "APPROACH_HANDLE" if self.tap is not None else "IDLE"

        elif new_state == "APPROACH_HANDLE":
            # point end effector x along the grip and approach along the hinge axis
            # reversing two axes keeps a valid rotation with perpendicular unit axes
            grip_rotation = np.diag([-1, 1, -1])
            # start at the grip centre in the moving handle frame
            # apply the grasp orientation then remove the imaginary gripper offset
            self.tr_handle_ee = (
                self.tap.tr_handle_grip * SE3.Rt(grip_rotation, np.zeros(3))
                * self.tr_ee_gripper.inv()
            )
            # tap forward kinematics puts the handle frame at its current angle in the world
            self.tr_handle_grasp = self.tap.fkine(self.tap.q) * self.tr_handle_ee
            # stop five centimetres before contact along the gripper approach direction
            ee_goal = self.tr_handle_grasp * SE3.Tz(-0.05)
            q0 = np.deg2rad([56.1, -179.5, 83.1, -83.6, 33.9, -61.5])
            solution = self.ur3.ikine_LM(
                ee_goal, q0=q0, joint_limits=True,
            )
            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Handle approach IK failed: {solution.reason}")
            # choose equivalent joint angles near the current posture to avoid extra turns
            q_near = solution.q + 2 * np.pi * np.round(
                (self.ur3.q - solution.q) / (2 * np.pi)
            )
            lower, upper = self.joint_limits
            # keep the original solution wherever the equivalent angle would exceed a limit
            q_target = np.where(
                (q_near >= lower) & (q_near <= upper), q_near, solution.q,
            )
            self.trajectory = rtb.jtraj(self.ur3.q, q_target, 200).q
            self.next_state = "GRASP_HANDLE"

        elif new_state == "GRASP_HANDLE":
            # approach the grip in a straight line while smoothly matching its orientation
            poses = rtb.ctraj(self.ur3.fkine(self.ur3.q), self.tr_handle_grasp, 100)
            self.trajectory = self.plan_handle_rmrc(poses)
            self.next_state = "OPEN_TAP"

        elif new_state == "OPEN_TAP":
            # open the handle to the chosen angle using the same circular grasp path
            open_angle = np.deg2rad(35)
            # use the tap maximum when available and thirty five degrees otherwise
            limits = getattr(self.tap, "qlim", None)
            if limits is not None:
                maximum = float(np.asarray(limits)[1, 0])
                if np.isfinite(maximum):
                    open_angle = maximum
            self.trajectory, self.tap_angles = self.plan_tap_rotation(open_angle)
            self.next_state = "WAIT_FOR_BEER"

        elif new_state == "WAIT_FOR_BEER":
            # keep holding the open handle while the cup fills for the chosen time
            self.wait_time = self.beer_fill_time
            self.next_state = "CLOSE_TAP"

        elif new_state == "CLOSE_TAP":
            # follow the handle back to zero before collecting the beer
            self.trajectory, self.tap_angles = self.plan_tap_rotation(0.0)
            self.next_state = "RETREAT_FROM_HANDLE"

        elif new_state == "RETREAT_FROM_HANDLE":
            # release the simulated handle and clear it along end effector negative z
            ee_goal = self.ur3.fkine(self.ur3.q) * SE3.Tz(-0.10)
            self.trajectory = self.plan_rmrc(ee_goal, steps=100)
            self.next_state = "APPROACH_BEER"

        elif new_state == "APPROACH_BEER":
            # use the actual cup pose and the original grasp offset to line up again
            self.tr_beer_grasp = SE3(self.placed_glass.T) * self.tr_ee_placed_glass.inv()
            # stop ten centimetres behind the grasp along end effector negative z
            ee_goal = self.tr_beer_grasp * SE3.Tz(-0.10)
            solution = self.ur3.ikine_LM(
                ee_goal, q0=self.q_placed_glass, joint_limits=True,
            )
            if not solution.success:
                self.state = "ERROR"
                raise RuntimeError(f"Beer approach IK failed: {solution.reason}")
            # keep the same joint turns as the saved cup placement where limits allow
            q_near = solution.q + 2 * np.pi * np.round(
                (self.q_placed_glass - solution.q) / (2 * np.pi)
            )
            lower, upper = self.joint_limits
            q_target = np.where(
                (q_near >= lower) & (q_near <= upper), q_near, solution.q,
            )
            self.trajectory = rtb.jtraj(self.ur3.q, q_target, 200).q
            self.next_state = "MOVE_TO_BEER"

        elif new_state == "MOVE_TO_BEER":
            # move straight forwards from the approach pose to the cup using rmrc
            self.trajectory = self.plan_rmrc(self.tr_beer_grasp, steps=100)
            self.next_state = "GRASP_BEER"

        elif new_state == "GRASP_BEER":
            # attach the same cup again using the actual end effector pose at contact
            self.held_glass = self.placed_glass
            self.held_glass_axes = self.placed_glass_axes
            ee = self.ur3.fkine(self.ur3.q)
            self.tr_ee_held_glass = ee.inv() * SE3(self.held_glass.T)
            self.next_state = "LIFT_BEER"

        elif new_state == "LIFT_BEER":
            # multiplication on the left lifts vertically in the world frame
            ee_goal = SE3.Tz(self.beer_lift_height) * self.ur3.fkine(self.ur3.q)
            self.trajectory = self.plan_rmrc(ee_goal, steps=100)
            self.next_state = "MOVE_BEER_TO_TARGET"

        elif new_state == "MOVE_BEER_TO_TARGET":
            # Final EE axes: X points up, Y along world X, Z along world +Y.
            ee_rotation = (SE3.Rz(-np.pi / 2) * SE3.Ry(-np.pi / 2)).R
            # Keep the actual grasp offset when converting EE rotation to cup rotation.
            cup_rotation = ee_rotation @ self.tr_ee_held_glass.R
            self.tr_beer_target = SE3.Rt(cup_rotation, self.beer_target_position)
            # Derive the approach from the final cup pose using world -Y and +Z.
            self.tr_beer_approach = SE3(0, -0.13, 0.02) * self.tr_beer_target
            if not np.isfinite(self.beer_move_time) or self.beer_move_time <= 0:
                self.state = "ERROR"
                raise ValueError("Beer movement time must be positive and finite")
            # use the controller sample time so the duration is independent of the frame rate
            steps = int(np.ceil(self.beer_move_time / self.dt)) + 1
            # build the straight ctraj path for the cup rather than the wrist
            self.trajectory = self.plan_rmrc(
                self.tr_beer_approach, steps=steps, tool_offset=self.tr_ee_held_glass,
            )
            self.next_state = "PLACE_BEER"

        elif new_state == "PLACE_BEER":
            # Move the cup forwards 10 cm and down 2 cm without changing its rotation.
            self.trajectory = self.plan_rmrc(
                self.tr_beer_target, steps=200, tool_offset=self.tr_ee_held_glass,
            )
            self.next_state = "RELEASE_BEER"

        elif new_state == "RELEASE_BEER":
            self._release_glass()
            self.next_state = "RETREAT_FROM_BEER"

        elif new_state == "RETREAT_FROM_BEER":
            # Right multiplication retreats along the EE's own negative Z axis.
            ee_goal = self.ur3.fkine(self.ur3.q) * SE3.Tz(-0.05)
            self.trajectory = self.plan_rmrc(ee_goal, steps=100)
            self.next_state = "IDLE"


        elif new_state == "RETURN_HOME":
            self.trajectory = rtb.jtraj(self.ur3.q, self.q_idle, 100).q
            self.next_state = "IDLE"

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
        self.placed_glass = self.held_glass
        self.placed_glass_axes = self.held_glass_axes
        self.tr_ee_placed_glass = SE3(self.tr_ee_held_glass)
        self.q_placed_glass = self.ur3.q.copy()
        self.held_glass = None
        self.held_glass_axes = None
        self.tr_ee_held_glass = None

    def plan_rmrc(self, target_pose, steps=100, q_start=None, tool_offset=None):
        """Use RMRC and Damped Least Squares to follow a Cartesian trajectory."""
        # 1: Assign parameters for RMRC
        if steps < 2:
            raise ValueError("RMRC needs at least two samples")
        delta_t = self.dt  # Discrete time step
        min_manip_measure = self.min_manip_measure  # Threshold for MoM
        m = np.zeros([1, steps - 1])  # Assign memory to store MoM
        error = np.empty([6, steps - 1])  # Assign memory to record velocity error

        if q_start is None:
            q_start = self.ur3.q

        # 2: Create a Cartesian trajectory for the robot end-effector to follow
        if tool_offset is None:
            poses = rtb.ctraj(self.ur3.fkine(q_start), target_pose, steps)
        else:
            # Make the attached cup follow the straight line even while it turns
            cup_start = self.ur3.fkine(q_start) * tool_offset
            cup_poses = rtb.ctraj(cup_start, target_pose, steps)
            # Convert each cup pose into the matching end-effector pose
            offset_inverse = tool_offset.inv()
            poses = [pose * offset_inverse for pose in cup_poses]

        # 3: Start from the current joint angles (no initial IK move is needed)
        q_matrix = [np.array(q_start, dtype=float, copy=True)]

        # 4: Use RMRC to solve joint velocities at each time step
        for i in range(steps - 1):
            actual = self.ur3.fkine(q_matrix[i])
            desired = poses[i + 1]
            # Calculate velocity at discrete time step, correcting any tracking error
            # Include world-frame rotation velocity for the six-axis robot
            xdot = np.r_[
                desired.t - actual.t,
                Rotation.from_matrix(desired.R @ actual.R.T).as_rotvec(),
            ] / delta_t
            J = self.ur3.jacob0(q_matrix[i])  # Get the Jacobian at the current state
            # Clamp small negative determinants caused by floating-point rounding
            m[:, i] = np.sqrt(max(0.0, linalg.det(J @ J.T)))  # MoM

            # Apply Damped Least Squares for configurations with low manipulability
            if m[0, i] < min_manip_measure:
                qdot = linalg.inv(J.T @ J + 0.01**2 * np.eye(self.ur3.n)) @ J.T @ xdot
            else:
                qdot = linalg.inv(J) @ xdot  # Solve velocities via RMRC

            error[:, i] = xdot - J @ qdot  # Calculate and store velocity error
            q_matrix.append(q_matrix[i] + delta_t * qdot)  # Update next joint state

        # 5: Store manipulability and velocity error for inspection or plotting
        self.rmrc_manipulability = m
        self.rmrc_velocity_error = error
        return np.asarray(q_matrix)

    def plan_tap_rotation(self, target_angle, steps=200):
        """Plan matching arm and handle samples for opening or closing."""
        start_angle = float(self.tap.q[0])
        # smooth joint interpolation makes the handle start and finish gently
        angles = rtb.jtraj([start_angle], [target_angle], steps).q[:, 0]
        # set the endpoints exactly to avoid rounding outside the handle limits
        angles[0], angles[-1] = start_angle, target_angle
        # save the actual end effector pose relative to the handle at contact
        self.tr_handle_ee = (
            self.tap.fkine([start_angle]).inv() * self.ur3.fkine(self.ur3.q)
        )
        # rotating the handle with a fixed grasp offset creates the circular path
        # this also turns the gripper with the handle so the grasp stays aligned
        poses = [self.tap.fkine([angle]) * self.tr_handle_ee for angle in angles]
        trajectory = self.plan_handle_rmrc(poses)
        # planning leaves the scene still and the angles are applied during playback
        return trajectory, angles

    def plan_handle_rmrc(self, poses):
        """Track the handle poses using damped RMRC, without moving the tap."""
        # this separate planner follows the supplied handle poses instead of making a straight path
        if len(poses) < 2:
            raise ValueError("Handle RMRC needs at least two poses")
        q_matrix = np.zeros((len(poses), self.ur3.n))
        q_matrix[0] = self.ur3.q
        lower, upper = self.joint_limits

        for i in range(len(poses) - 1):
            # compare the predicted arm pose with the next desired grasp pose
            actual = self.ur3.fkine(q_matrix[i])
            desired = poses[i + 1]
            # request the world position and orientation change over one sample
            xdot = np.r_[
                desired.t - actual.t,
                Rotation.from_matrix(desired.R @ actual.R.T).as_rotvec(),
            ] / self.dt
            J = self.ur3.jacob0(q_matrix[i])
            # damped least squares converts end effector velocity into joint velocity
            # the small diagonal damping keeps the solve stable near singularities
            qdot = linalg.solve(
                J.T @ J + 0.001**2 * np.eye(self.ur3.n), J.T @ xdot,
            )
            q_next = q_matrix[i] + self.dt * qdot
            # reject invalid numbers or joint limit violations before executing the motion
            if (not np.isfinite(q_next).all()
                    or np.any(q_next < lower) or np.any(q_next > upper)):
                self.state = "ERROR"
                raise RuntimeError("Handle RMRC exceeds joint limits")
            reached = self.ur3.fkine(q_next)
            # check that the predicted arm pose still follows the handle closely
            position_error = np.linalg.norm(desired.t - reached.t)
            rotation_error = Rotation.from_matrix(desired.R @ reached.R.T).magnitude()
            if position_error > 0.001 or rotation_error > np.deg2rad(1):
                # stop if tracking error exceeds one millimetre or one degree
                self.state = "ERROR"
                raise RuntimeError(
                    f"Handle RMRC cannot track pose {i + 1}: "
                    f"position error {position_error:.4f} m, "
                    f"rotation error {np.rad2deg(rotation_error):.2f} deg"
                )
            q_matrix[i + 1] = q_next

        return q_matrix

    def request_beer(self):
        """Remember one order; update() starts it after the tray arrives."""
        if self.state == "STOPPED":
            return
        self.pending_beers += 1
        self.total_beer_requests += 1

    def stop(self):
        # cancel playback and keep the controller stopped until restart
        self.state = self.next_state = "STOPPED"
        self.trajectory = None
        self.tap_angles = None

    def _next_available_glass(self):
        """Nearest row first; columns run from right to left."""
        for row in (2, 1, 0):
            for column in range(3):
                cell = (row, column)
                if cell in self.glasses and self.glass_positions[cell]:
                    return cell
        return None

    def pick_glass(self):
        """Start pickup; the approach branch defines its own q0 guess."""
        if self.state != "IDLE":
            raise RuntimeError("Controller must be IDLE before starting pickup")
        if self.held_glass is not None:
            raise RuntimeError("Release the held glass before picking another one")
        self.change_state("APPROACH_GLASS")

    def update(self, dt, world=None):
        """Execute one sample when due; the caller steps Swift separately."""
        if self.state == "STOPPED":
            return
        if (self.state == "IDLE" and self.tray_ready
                and self.pending_beers > 0 and self.held_glass is None):
            cell = self._next_available_glass()
            if cell is not None:
                self.selected_glass = cell
                self.pick_glass()
                self.pending_beers -= 1

        if self.state not in ("IDLE", "ERROR"):
            if self.trajectory is None:
                # waiting uses simulation time so the display remains responsive
                self.elapsed += dt
                if self.elapsed + 1e-12 >= self.wait_time:
                    # states without a wait advance on the next update as usual
                    self.change_state(self.next_state)
            else:
                self.elapsed += dt
                if self.elapsed >= self.dt:
                    # apply one trajectory sample whenever its time interval has passed
                    self.elapsed -= self.dt
                    if self.step < len(self.trajectory):
                        self.ur3.q = self.trajectory[self.step]
                        if self.tap_angles is not None:
                            # use the same sample index so the arm and handle move together
                            self.tap.set_handle_angle(float(self.tap_angles[self.step]))
                        self.step += 1
                    else:
                        self.change_state(self.next_state)

        # combine the world end effector pose with the saved cup offset to move the held cup
        # this also runs while idle so manual arm movement carries the cup
        if self.held_glass is not None:
            ee = self.ur3.fkine(self.ur3.q)
            self.held_glass.T = (ee * self.tr_ee_held_glass).A
            if self.held_glass_axes is not None:
                self.held_glass_axes.T = self.held_glass.T

    def receive_tray(self, tr_tray, glass_positions):
        """Confirm handoff and store the tray pose and 3 x 3 availability mask."""
        positions = np.array(glass_positions, dtype=bool, copy=True)
        if positions.shape != (3, 3):
            raise ValueError("glass_positions must be a 3 x 3 availability mask")
        self.tr_tray = SE3(tr_tray)
        self.glass_positions = positions
        self.tray_ready = True

    def recieve_tray(self, tr_tray, glass_positions):
        """Keep the original spelling working for existing callers."""
        self.receive_tray(tr_tray, glass_positions)
