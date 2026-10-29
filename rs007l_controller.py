from math import pi

import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from scene import (
    BAR_Y, BAR_HEIGHT, CONVEYOR_Y, CONVEYOR_HEIGHT,
    CONVEYOR_TRAVEL_DISTANCE, CONVEYOR_END_X,
    CONVEYOR_LENGTH, CONVEYOR_WIDTH, CONVEYOR_CENTRE_X, CONVEYOR_START_X,
    BUTTON_X, BUTTON_Y, BUTTON_TOP, BUTTON_PRESS_DEPTH,
    BUTTON_HOVER_HEIGHT,
    BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP,
    DISHWASHER_FRONT_X, DISHWASHER_BACK_X, DISHWASHER_WIDTH,
    DISHWASHER_TOP, DOOR_CLOSED_CENTRE, DOOR_RISE,
    TRAY_WIDTH, TRAY_LENGTH, RUNNER_HEIGHT, DECK_THICKNESS,
    RUNNER_WIDTH, RUNNER_X_POSITIONS, RIM_HEIGHT, RIM_THICKNESS,
    FORK_MOUNT, FORK_X_POSITIONS, FORK_WIDTH, FORK_LENGTH, FORK_THICKNESS,
    DOOR_HEIGHT,
)

STEP_TIME = 0.005
# movement durations in seconds; increase these values to slow the mechanisms down
DOOR_MOVE_TIME = 2.0
CONVEYOR_MOVE_TIME = 4.0
PICKUP_ANGLES_DEGREES = [0, 22, -79, 0, 10, 0]
PICKUP_HEIGHT_CORRECTION = 0.012
PICKUP_BACKOFF = 0.015
APPROACH_DISTANCE = 0.370
CONTACT_RISE = 0.006
TRAY_LIFT = 0.3
DISPLAY_EVERY = 1

class MotionPlanningError(Exception):
    pass

class MotionStopped(Exception):
    pass

class RS007LController:

    def __init__(self, robot, scene, simulator, dt=STEP_TIME):
        self.robot, self.scene, self.simulator = robot, scene, simulator
        self.stopped = False
        self.on_step = None
        self.pause_requested = None
        self.carrying_tray = False
        self.tray_placed = False
        self.conveyor_finished = False
        self.washed = False
        self.wash_complete = False
        self.rear_pickup_paths = None
        self.tray_offset_from_wrist = None
        self.pickup_paths = None
        self.button_hover_angles = {}
        self.lower = np.asarray(robot.qlim[0], dtype=float).copy()
        self.upper = np.asarray(robot.qlim[1], dtype=float).copy()
        self.obstacles = self._build_obstacles()
        self.dt = float(dt)
        self.state = "IDLE"
        self.next_state = "IDLE"
        self.elapsed = 0.0
        self.trajectory = None
        self.step = 0
        self.wait_time = 0.0
        self._sequence = None
        self._carrying_segment = False

    def wrist_pose(self, joint_angles):

        return self.robot.world_pose.A @ self.robot._joint_frames(joint_angles)[-1]

    def tool_pose(self, joint_angles):

        return self.wrist_pose(joint_angles) @ FORK_MOUNT.A @ SE3(0, 0.24, -0.025).A

    def _build_obstacles(self, tray_position=None):
        objects = []

        def add(name, centre, size, tray=False):
            centre = np.asarray(centre, dtype=float)
            half_size = np.asarray(size, dtype=float) / 2
            objects.append((name, centre - half_size, centre + half_size, tray))

        add('conveyor surface', (CONVEYOR_CENTRE_X, CONVEYOR_Y, CONVEYOR_HEIGHT-.055),
            (CONVEYOR_LENGTH, CONVEYOR_WIDTH, .11))
        for x_position in (CONVEYOR_START_X + .12, CONVEYOR_END_X - .12):
            for y_position in (-.255, .255):
                add('conveyor leg',
                    (x_position, CONVEYOR_Y + y_position, (CONVEYOR_HEIGHT-.12)/2),
                    (.075, .075, CONVEYOR_HEIGHT-.12))

        add('button pedestal', (-.445, -.085, self.robot.world_pose.t[2] / 2),
            (.57, .43, self.robot.world_pose.t[2]))
        add('green switch housing', (BUTTON_X, BUTTON_Y, BUTTON_TOP-.0525),
            (.12, .12, .065))
        add('blue switch housing', (BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP-.0525),
            (.12, .12, .065))

        if tray_position is None:
            tray_position = np.asarray(self.scene.tray_pose.t, dtype=float)
        tray_x, tray_y, tray_z = np.asarray(tray_position, dtype=float)
        for x_position in RUNNER_X_POSITIONS:
            add('tray underside runner', (tray_x+x_position, tray_y, tray_z + RUNNER_HEIGHT/2),
                (RUNNER_WIDTH, TRAY_LENGTH, RUNNER_HEIGHT), True)
        add('tray deck', (tray_x, tray_y, tray_z + RUNNER_HEIGHT + DECK_THICKNESS/2),
            (TRAY_WIDTH, TRAY_LENGTH, DECK_THICKNESS), True)
        rim_z = tray_z + RUNNER_HEIGHT + DECK_THICKNESS + RIM_HEIGHT/2
        for x_position in (-TRAY_WIDTH/2 + RIM_THICKNESS/2,
                   TRAY_WIDTH/2 - RIM_THICKNESS/2):
            add('tray side rim', (tray_x+x_position, tray_y, rim_z),
                (RIM_THICKNESS, TRAY_LENGTH, RIM_HEIGHT), True)
        for y_position in (-TRAY_LENGTH/2 + RIM_THICKNESS/2,
                   TRAY_LENGTH/2 - RIM_THICKNESS/2):
            add('tray front/back rim', (tray_x, tray_y + y_position, rim_z),
                (TRAY_WIDTH - 2*RIM_THICKNESS, RIM_THICKNESS, RIM_HEIGHT), True)

        for x_position in (-.118, 0.0, .118):
            for y_position in (-.118, 0.0, .118):
                add('beer glass (proxy)',
                    (tray_x+x_position, tray_y + y_position, tray_z + RUNNER_HEIGHT + DECK_THICKNESS + .085),
                    (.078, .078, .170), True)
        length = DISHWASHER_BACK_X - DISHWASHER_FRONT_X
        centre_x = (DISHWASHER_FRONT_X + DISHWASHER_BACK_X) / 2
        side_y = DISHWASHER_WIDTH / 2
        wall_height = DISHWASHER_TOP + .045
        wall_z = wall_height / 2
        for direction in (-1, 1):
            add('dishwasher side', (centre_x, CONVEYOR_Y + direction*side_y, wall_z),
                (length, .035, wall_height))
        add('dishwasher roof', (centre_x, CONVEYOR_Y, DISHWASHER_TOP + .0225),
            (length, DISHWASHER_WIDTH+.035, .045))
        add('dishwasher back', (DISHWASHER_BACK_X - .02, CONVEYOR_Y, wall_z),
            (.04, DISHWASHER_WIDTH+.035, wall_height))
        rail_height = CONVEYOR_HEIGHT + DOOR_HEIGHT + DOOR_RISE
        for direction in (-1, 1):
            add('dishwasher door rail',
                (DISHWASHER_FRONT_X, CONVEYOR_Y + direction * (DISHWASHER_WIDTH/2 + .0175), rail_height/2),
                (.06, .06, rail_height))
        if not self.scene.door_open:
            add('dishwasher door', (DISHWASHER_FRONT_X, CONVEYOR_Y, DOOR_CLOSED_CENTRE),
                (.034, DISHWASHER_WIDTH-.02, DOOR_HEIGHT))
        return objects

    @staticmethod
    def _sphere_intersects_box(point, radius, min_corner, max_corner):
        nearest = np.maximum(min_corner, np.minimum(point, max_corner))
        return np.linalg.norm(point - nearest) < radius

    @staticmethod
    def _boxes_intersect(centre, axes, sizes, min_corner, max_corner):

        fixed_half_size = (max_corner - min_corner) / 2
        moving_half_size = np.asarray(sizes, dtype=float) / 2
        t = centre - (min_corner + max_corner) / 2
        box_rotation = axes
        absolute_rotation = np.abs(box_rotation) + 1e-10
        for index in range(3):
            if abs(t[index]) > fixed_half_size[index] + np.dot(moving_half_size, absolute_rotation[index, :]):
                return False
        for axis_index in range(3):
            if abs(np.dot(t, box_rotation[:, axis_index])) > np.dot(fixed_half_size, absolute_rotation[:, axis_index]) + moving_half_size[axis_index]:
                return False
        for index in range(3):
            axis_a, axis_b = (index + 1) % 3, (index + 2) % 3
            for axis_index in range(3):
                left = abs(t[axis_b] * box_rotation[axis_a, axis_index] - t[axis_a] * box_rotation[axis_b, axis_index])
                right = fixed_half_size[axis_a] * absolute_rotation[axis_b, axis_index] + fixed_half_size[axis_b] * absolute_rotation[axis_a, axis_index]
                right += moving_half_size[(axis_index + 1) % 3] * absolute_rotation[index, (axis_index + 2) % 3]
                right += moving_half_size[(axis_index + 2) % 3] * absolute_rotation[index, (axis_index + 1) % 3]
                if left > right:
                    return False
        return True

    def get_link_poses(self, joint_angles):
        return [self.robot.world_pose.A @ frame
                for frame in self.robot._joint_frames(joint_angles)]

    def check_collision(self, joint_angles, carrying=False, obstacles=None):

        joint_angles = np.asarray(joint_angles, dtype=float)
        if not np.isfinite(joint_angles).all() or np.any(joint_angles < self.lower) or np.any(joint_angles > self.upper):
            return 'joint limit or invalid joint value'
        frames = self.get_link_poses(joint_angles)
        if obstacles is None:
            obstacles = self.obstacles
        obstacles = [o for o in obstacles if not carrying or not o[3]]

        for index in range(2, 6):
            link_start, link_end = frames[index][:3, 3], frames[index+1][:3, 3]
            link_radius = .037 if index in (2, 3, 4) else .028
            sample_count = max(3, int(np.linalg.norm(link_end-link_start)/.025) + 1)
            # a link cannot touch an obstacle outside its bounds, including its radius
            link_min = np.minimum(link_start, link_end) - link_radius
            link_max = np.maximum(link_start, link_end) + link_radius
            nearby_obstacles = [o for o in obstacles
                                if not (np.any(link_max < o[1]) or np.any(link_min > o[2]))]
            if not nearby_obstacles:
                continue
            for t in np.linspace(0, 1, sample_count):
                point = link_start * (1-t) + link_end * t
                for name, min_corner, max_corner, _ in nearby_obstacles:
                    if self._sphere_intersects_box(point, link_radius, min_corner, max_corner):
                        return f'robot link {index} near {name}'

        transform = self.wrist_pose(joint_angles) @ FORK_MOUNT.A
        boxes = [([0, .0525, -.010], [.23, .035, .018], 'fork mounting box')]
        for x_position in FORK_X_POSITIONS:
            boxes.append(([x_position, .065 + FORK_LENGTH/2, -.025],
                          [FORK_WIDTH, FORK_LENGTH, FORK_THICKNESS], 'fork prong'))
        for local_center, dimensions, part in boxes:
            centre = (transform @ np.r_[local_center, 1.0])[:3]
            for name, min_corner, max_corner, _ in obstacles:
                if self._boxes_intersect(centre, transform[:3, :3], dimensions, min_corner, max_corner):
                    return f'{part} near {name}'

        if carrying and self.tray_offset_from_wrist is not None:
            tray_transform = self.wrist_pose(joint_angles) @ self.tray_offset_from_wrist
            if tray_transform[2, 2] < np.cos(np.deg2rad(1.0)):
                return 'carried tray is not level'
            load_height = RUNNER_HEIGHT + DECK_THICKNESS + RIM_HEIGHT + .17
            local_center = np.array([0, 0, load_height/2, 1.0])
            centre = (tray_transform @ local_center)[:3]
            for name, min_corner, max_corner, is_tray in obstacles:
                if is_tray:
                    continue

                surface_limit = max_corner.copy()
                if name in ('bar countertop', 'conveyor surface'):
                    surface_limit[2] -= .002
                if self._boxes_intersect(centre, tray_transform[:3,:3],
                                 [TRAY_WIDTH, TRAY_LENGTH, load_height],
                                 min_corner, surface_limit):
                    return f'carried tray / glasses near {name}'
        return None

    def check_trajectory(self, trajectory, carrying=False, label='segment', obstacles=None):
        previous = None
        max_step = np.deg2rad(5)
        for index, joint_angles in enumerate(trajectory):
            joint_angles = np.asarray(joint_angles, dtype=float)
            samples = [joint_angles]
            if (previous is not None and np.isfinite(previous).all()
                    and np.isfinite(joint_angles).all()):
                angle_change = np.max(np.abs(joint_angles - previous))
                if angle_change > max_step:
                    steps = int(np.ceil(angle_change / max_step)) + 1
                    path = rtb.jtraj(previous, joint_angles, steps).q
                    while np.any(np.abs(np.diff(path, axis=0)) > max_step):
                        steps += 1
                        path = rtb.jtraj(previous, joint_angles, steps).q
                    samples = list(path[1:-1]) + samples
            for q in samples:
                collision_message = self.check_collision(q, carrying=carrying, obstacles=obstacles)
                if collision_message is not None:
                    raise MotionPlanningError(f'{label}: collision at step {index}: {collision_message}')
            previous = joint_angles

    def _tool_pose_error(self, joint_angles, target):
        current_tool_pose = self.tool_pose(joint_angles)
        position_difference = target[:3, 3] - current_tool_pose[:3, 3]
        rotation_difference = Rotation.from_matrix(target[:3, :3] @ current_tool_pose[:3, :3].T).as_rotvec()
        return np.r_[position_difference, rotation_difference]

    def _calculate_jacobian(self, joint_angles):

        small_step = 1e-5
        current_tool_pose = self.tool_pose(joint_angles)
        jacobian = np.zeros((6, 6))
        for axis_index in range(6):
            shifted = np.asarray(joint_angles, dtype=float).copy()
            shifted[axis_index] += small_step
            next_pose = self.tool_pose(shifted)
            jacobian[:3, axis_index] = (next_pose[:3, 3] - current_tool_pose[:3, 3]) / small_step
            jacobian[3:, axis_index] = Rotation.from_matrix(
                next_pose[:3, :3] @ current_tool_pose[:3, :3].T).as_rotvec() / small_step
        return jacobian

    def _solve_inverse_kinematics(self, target, starting_guess, label):

        safe_minimum = self.lower + 1e-5
        safe_maximum = self.upper - 1e-5
        guess = np.clip(starting_guess, safe_minimum, safe_maximum)
        # Convert the fork tool target into a wrist target for the DH model.
        fork_offset = FORK_MOUNT * SE3(0, 0.24, -0.025)
        wrist_target = SE3(target) * fork_offset.inv()
        solution = self.robot.ikine_LM(
            wrist_target, q0=guess, joint_limits=True,
            ilimit=400, slimit=1,
        )
        if not solution.success:
            raise MotionPlanningError(f'{label}: IK failed: {solution.reason}')
        pose_error = self._tool_pose_error(solution.q, target)
        if np.linalg.norm(pose_error[:3]) > .002 or np.linalg.norm(pose_error[3:]) > np.deg2rad(1):
            raise MotionPlanningError(f'{label}: IK could not reach desired tool pose')
        return solution.q

    def _fork_tip_position(self, joint_angles, prong=0):

        local_x = FORK_X_POSITIONS[prong]
        point = np.array([local_x, 0.065 + FORK_LENGTH, -0.025, 1.0])
        frame = self.wrist_pose(joint_angles) @ FORK_MOUNT.A
        return (frame @ point)[:3]

    def _fork_direction(self, joint_angles):

        return (self.wrist_pose(joint_angles) @ FORK_MOUNT.A)[:3, 1]

    def _find_button_approach(self, x, y, height, name):
        goal = np.array([x, y, height + BUTTON_HOVER_HEIGHT])
        downward_direction = np.array([0., 0., -1.])
        guesses = [
            np.deg2rad([-84, 22, -84, 0, -73, -23]),
            np.deg2rad([-100, 22, -84, 0, -73, -23]),
            np.deg2rad([-70, 22, -84, 0, -73, -23]),
            np.deg2rad([-90, 25, -80, 0, -75, 0]),
            np.deg2rad([-90, 10, -100, 0, -70, 40]),
            np.zeros(6),
        ]
        lower = self.lower + 1e-5
        upper = self.upper - 1e-5
        home = np.zeros(6)
        for guess in guesses:
            guess = np.clip(guess, lower, upper)
            # Keep the fork's orientation close to the guess while pointing down.
            fork_frame = self.wrist_pose(guess) @ FORK_MOUNT.A
            alignment, _ = Rotation.align_vectors(
                [downward_direction], [fork_frame[:3, 1]]
            )
            target_rotation = alignment.as_matrix() @ fork_frame[:3, :3]
            fork_tip_offset = SE3(
                FORK_X_POSITIONS[0], 0.065 + FORK_LENGTH, -0.025
            )
            target_fork_frame = SE3.Rt(target_rotation, goal) * fork_tip_offset.inv()
            wrist_target = target_fork_frame * FORK_MOUNT.inv()
            solution = self.robot.ikine_LM(
                wrist_target, q0=guess, joint_limits=True,
                ilimit=400, slimit=1,
            )
            if not solution.success:
                continue
            q = solution.q
            if np.linalg.norm(self._fork_tip_position(q) - goal) > .0025:
                continue
            if self._fork_direction(q)[2] > -np.cos(np.deg2rad(3)):
                continue
            if self.check_collision(q):
                continue
            path = rtb.jtraj(home, q, 105).q
            try:
                self.check_trajectory(path, label=f'home to {name} button')
            except MotionPlanningError:
                continue
            return q, path
        raise MotionPlanningError(f'No collision-free approach for {name} button')

    def _plan_button_paths(self, name):
        if name == 'blue':
            x, y, height = BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP
        else:
            x, y, height = BUTTON_X, BUTTON_Y, BUTTON_TOP

        current = np.asarray(self.robot.q, dtype=float).copy()
        if name in self.button_hover_angles:
            hover_angles = self.button_hover_angles[name]
            home_approach = None
        else:
            hover_angles, home_approach = self._find_button_approach(x, y, height, name)
            self.button_hover_angles[name] = hover_angles.copy()

        if np.allclose(current, hover_angles, atol=1e-6):
            approach = []
        else:
            direct_path = rtb.jtraj(current, hover_angles, 95).q
            try:
                self.check_trajectory(direct_path, label=f'approach {name} button')
                approach = [direct_path]
            except MotionPlanningError:
                to_home = rtb.jtraj(current, np.zeros(6), 110).q
                self.check_trajectory(to_home, label=f'safe approach to {name} button')
                if home_approach is None:
                    home_approach = rtb.jtraj(np.zeros(6), hover_angles, 105).q
                self.check_trajectory(home_approach, label=f'home to {name} button')
                approach = [to_home, home_approach]

        press_goal = self.tool_pose(hover_angles)
        press_goal[2, 3] -= BUTTON_HOVER_HEIGHT + BUTTON_PRESS_DEPTH
        press = self._plan_cartesian_move(hover_angles, press_goal, 65, f'press {name} button')
        contact = np.array([x, y, height - BUTTON_PRESS_DEPTH])
        if np.linalg.norm(self._fork_tip_position(press[-1]) - contact) > .0035:
            raise MotionPlanningError(f'{name} button not reached by fork tip')
        release = press[::-1].copy()
        self.check_trajectory(release, label=f'release {name} button')
        return approach, press, release

    def wait_if_paused(self):
        
        while self.pause_requested is not None and self.pause_requested():
            if self.stopped:
                raise MotionStopped()
            if self.on_step is not None:
                self.on_step()
            self.simulator.step(STEP_TIME)
        if self.stopped:
            raise MotionStopped()

    def step_simulation(self, dt=STEP_TIME):
        
        self.wait_if_paused()
        if self.on_step is not None:
            self.on_step()
        self.simulator.step(dt)
        self.wait_if_paused()

    def _animate_path(self, joint_path, carrying=False):
        if len(joint_path) == 0:
            return
        frame_indices = list(range(0, len(joint_path), DISPLAY_EVERY))
        if frame_indices[-1] != len(joint_path) - 1:
            frame_indices.append(len(joint_path) - 1)
        for index in frame_indices:
            self.wait_if_paused()
            joint_angles = joint_path[index]
            self.robot.set_joint_angles(joint_angles)
            self.scene.update_fork()
            if carrying:
                self.scene.set_tray_pose(SE3(self.wrist_pose(joint_angles) @ self.tray_offset_from_wrist))
            self.step_simulation()

    def plan_rmrc(self, target_pose, steps=100, q_start=None,
                  carrying=False, label="RMRC", obstacles=None):
        if steps < 2:
            raise ValueError("RMRC needs at least two trajectory samples")
        q_start = np.asarray(
            self.robot.q if q_start is None else q_start, dtype=float
        ).reshape(self.robot.n).copy()
        start_pose = SE3(self.tool_pose(q_start))
        poses = rtb.ctraj(start_pose, SE3(target_pose), steps)
        q_matrix = np.empty((steps, self.robot.n))
        q_matrix[0] = q_start

        for i in range(steps - 1):
            q = q_matrix[i].copy()
            target = poses[i + 1].A

            for _ in range(15):
                error = self._tool_pose_error(q, target)
                if (np.linalg.norm(error[:3]) < 0.00045
                        and np.linalg.norm(error[3:]) < 0.003):
                    break

                xdot = error / self.dt
                J = self._calculate_jacobian(q)
                damping = 0.006
                qdot = np.linalg.solve(
                    J.T @ J + damping**2 * np.eye(self.robot.n),
                    J.T @ xdot,
                )
                dq = self.dt * qdot
                max_step = np.max(np.abs(dq))
                if max_step > np.deg2rad(2):
                    dq *= np.deg2rad(2) / max_step
                q += dq
                if (not np.isfinite(q).all()
                        or np.any(q < self.lower)
                        or np.any(q > self.upper)):
                    raise MotionPlanningError(f"{label}: joint limit at step {i + 1}")

            error = self._tool_pose_error(q, target)
            if (np.linalg.norm(error[:3]) > 0.002
                    or np.linalg.norm(error[3:]) > np.deg2rad(1)):
                raise MotionPlanningError(f"{label}: RMRC tracking error at step {i + 1}")
            collision = self.check_collision(
                q, carrying=carrying, obstacles=obstacles,
            )
            if collision is not None:
                raise MotionPlanningError(f"{label}: collision at step {i + 1}: {collision}")
            q_matrix[i + 1] = q

        return q_matrix

    def _plan_cartesian_move(self, starting_joint_angles, destination, step_count,
                             label, carrying=False):

        return self.plan_rmrc(
            SE3(destination), steps=step_count, q_start=starting_joint_angles,
            carrying=carrying, label=label,
        )

    def change_state(self, new_state):

        if self.stopped:
            return
        if self._sequence is not None:
            raise RuntimeError(f"Kawasaki is already executing {self.state}")

        actions = {
            "PICKUP": self._pickup_sequence,
            "PLACE": self._place_sequence,
            "OPEN_DOOR": lambda: self._toggle_door_sequence(True),
            "CLOSE_DOOR": lambda: self._toggle_door_sequence(False),
            "CONVEY_IN": lambda: self._conveyor_sequence(True),
            "CONVEY_OUT": lambda: self._conveyor_sequence(False),
            "WASH": self._wash_sequence,
            "COLLECT_CLEAN_TRAY": self._collect_clean_tray_sequence,
            "DISHWASHER_CYCLE": self._dishwasher_sequence,
        }
        if new_state not in actions:
            raise ValueError(f"Unknown Kawasaki action: {new_state}")
        self._sequence = actions[new_state]()
        self.next_state = new_state
        self._advance_state()

    def _advance_state(self):

        if self._sequence is None or self.stopped:
            return
        try:
            name, path, carrying, wait_time = next(self._sequence)
        except StopIteration:
            self._sequence = None
            self.state = self.next_state = "IDLE"
            self.trajectory = None
            self.wait_time = 0.0
            self.elapsed = 0.0
            return
        except Exception:
            self.state = "ERROR"
            self._sequence = None
            self.trajectory = None
            raise

        self.state = name
        self.step = 0
        self.elapsed = 0.0
        self.wait_time = float(wait_time)
        self.trajectory = None if path is None else np.asarray(path, dtype=float)
        self._carrying_segment = bool(carrying)

    @staticmethod
    def _segment(name, path=None, carrying=False, wait_time=0.0):
        return name, path, carrying, wait_time

    def update(self, dt, world=None):

        if self.stopped or self.state in ("IDLE", "ERROR", "STOPPED"):
            return
        if self.pause_requested is not None and self.pause_requested():
            return
        if dt <= 0:
            return

        self.elapsed += float(dt)
        if self.trajectory is None:
            if self.elapsed + 1e-12 >= self.wait_time:
                self._advance_state()
            return

        if self.elapsed + 1e-12 >= self.dt:
            self.elapsed -= self.dt
            if self.step < len(self.trajectory):
                q = self.trajectory[self.step]
                self.robot.set_joint_angles(q)
                self.scene.update_fork()
                if self._carrying_segment:
                    if self.tray_offset_from_wrist is None:
                        raise MotionPlanningError("Tray attachment missing during motion")
                    tr_world_tray = self.wrist_pose(q) @ self.tray_offset_from_wrist
                    self.scene.set_tray_pose(SE3(tr_world_tray))
                self.step += 1
            else:
                self._advance_state()

    def stop(self):

        self.stopped = True
        self.state = self.next_state = "STOPPED"
        self._sequence = None
        self.trajectory = None
        self.step = 0
        self.robot.qd = np.zeros(self.robot.n)
        self.robot.qdd = np.zeros(self.robot.n)

    def _pickup_sequence(self):
        if self.tray_placed or self.carrying_tray:
            raise MotionPlanningError("Tray is already placed or being carried")

        seed = np.deg2rad(PICKUP_ANGLES_DEGREES)
        target = self.tool_pose(seed)
        target[:3, 3] += [0, -PICKUP_BACKOFF + (BAR_Y - 1.02), PICKUP_HEIGHT_CORRECTION]
        insertion_q = self._solve_inverse_kinematics(target, seed, "aligned insertion pose")
        staging = target.copy()
        staging[1, 3] -= APPROACH_DISTANCE
        staging_q = self._solve_inverse_kinematics(staging, insertion_q, "approach pose")

        approach = rtb.jtraj(self.robot.q, staging_q, 120).q
        self.check_trajectory(approach, label="staged approach")
        insertion = self.plan_rmrc(target, 85, q_start=staging_q,
                                   label="horizontal fork insertion")
        contact_target = self.tool_pose(insertion[-1])
        contact_target[2, 3] += CONTACT_RISE
        contact = self.plan_rmrc(contact_target, 14, q_start=insertion[-1],
                                 label="raise forks")
        lift_target = self.tool_pose(contact[-1])
        lift_target[2, 3] += TRAY_LIFT

        future_offset = np.linalg.inv(self.wrist_pose(contact[-1])) @ self.scene.tray_pose.A
        original_offset = self.tray_offset_from_wrist
        self.tray_offset_from_wrist = future_offset
        try:
            lift = self.plan_rmrc(lift_target, 45, q_start=contact[-1],
                                  carrying=True, label="lift loaded tray")
        finally:
            self.tray_offset_from_wrist = original_offset

        self.pickup_paths = {
            "insertion": insertion.copy(),
            "contact": contact.copy(),
            "lift": lift.copy(),
        }
        self.original_pickup_paths = {name: path.copy() for name, path in self.pickup_paths.items()}
        yield self._segment("APPROACH_TRAY", approach)
        yield self._segment("INSERT_FORKS", insertion)
        yield self._segment("CONTACT_TRAY", contact)

        self.tray_offset_from_wrist = np.linalg.inv(self.wrist_pose(self.robot.q)) @ self.scene.tray_pose.A
        self.carrying_tray = True
        yield self._segment("LIFT_TRAY", lift, carrying=True)

    def _place_sequence(self):
        if not self.carrying_tray or self.pickup_paths is None:
            raise MotionPlanningError("No picked-up tray available for placement")
        if not np.isclose(CONVEYOR_HEIGHT, BAR_HEIGHT, atol=1e-6):
            raise MotionPlanningError("Placement requires matching bar and conveyor heights")
        if abs(-BAR_Y - CONVEYOR_Y) + TRAY_LENGTH / 2 >= (0.72 - 0.07) / 2:
            raise MotionPlanningError("Mirrored tray position overhangs conveyor")

        start = np.asarray(self.robot.q, dtype=float).copy()
        turn = (-pi + 1e-5) if start[0] > -0.01 else (pi - 1e-5)
        finish = start.copy()
        finish[0] += turn
        if not self.lower[0] <= finish[0] <= self.upper[0]:
            raise MotionPlanningError("JT1 cannot complete its 180 degree turn")

        def mirror(path):
            mirrored = np.asarray(path, dtype=float).copy()
            mirrored[:, 0] += turn
            return mirrored

        rotation = rtb.jtraj(start, finish, 160).q
        lowering = mirror(self.pickup_paths["lift"][::-1])
        disengage = mirror(self.pickup_paths["contact"][::-1])
        withdrawal = mirror(self.pickup_paths["insertion"][::-1])

        self.check_trajectory(rotation, carrying=True, label="180 degree swivel")
        self.check_trajectory(lowering, carrying=True, label="lowering")
        tr_final = self.wrist_pose(lowering[-1]) @ self.tray_offset_from_wrist
        target = np.array([0.0, -BAR_Y, CONVEYOR_HEIGHT])
        if np.linalg.norm(tr_final[:3, 3] - target) > 0.008:
            raise MotionPlanningError("Predicted tray pose does not reach conveyor")
        for path in (rotation, lowering):
            for q in path:
                tray_frame = self.wrist_pose(q) @ self.tray_offset_from_wrist
                if tray_frame[2, 2] < np.cos(np.deg2rad(1)):
                    raise MotionPlanningError("Placement would tilt tray")
        placed_obstacles = self._build_obstacles(tray_position=target)
        self.check_trajectory(disengage, label="fork disengagement",
                              obstacles=placed_obstacles)
        self.check_trajectory(withdrawal, label="fork withdrawal",
                              obstacles=placed_obstacles)

        yield self._segment("SWIVEL_TO_CONVEYOR", rotation, carrying=True)
        yield self._segment("LOWER_TRAY", lowering, carrying=True)
        self.scene.set_tray_pose(SE3(tr_final))
        self.carrying_tray = False
        self.tray_offset_from_wrist = None
        self.obstacles = placed_obstacles
        yield self._segment("DISENGAGE_FORKS", disengage)
        yield self._segment("WITHDRAW_FORKS", withdrawal)

        self.rear_pickup_paths = {
            "insertion": withdrawal[::-1].copy(),
            "contact": disengage[::-1].copy(),
            "lift": lowering[::-1].copy(),
        }
        self.pickup_paths = None
        self.tray_placed = True

    def _button_press_sequence(self, name):
        self.obstacles = self._build_obstacles()
        approach, press, release = self._plan_button_paths(name)
        for index, path in enumerate(approach):
            yield self._segment(f"APPROACH_{name.upper()}_BUTTON_{index + 1}", path)
        yield self._segment(f"PRESS_{name.upper()}_BUTTON", press)
        if name == "blue":
            self.scene.set_blue_button_pressed(True)
        else:
            self.scene.set_button_pressed(True)
        return release

    def _button_release_sequence(self, name, release):
        yield self._segment(f"RELEASE_{name.upper()}_BUTTON", release)
        if name == "blue":
            self.scene.set_blue_button_pressed(False)
        else:
            self.scene.set_button_pressed(False)
        self.obstacles = self._build_obstacles()

    def _toggle_door_sequence(self, open_door):
        if self.carrying_tray:
            raise MotionPlanningError("Cannot move dishwasher door while carrying tray")
        if self.scene.door_open == open_door:
            return
        release = yield from self._button_press_sequence("blue")
        start = float(self.scene.door_fraction)
        target = 1.0 if open_door else 0.0
        steps = max(1, int(np.ceil(DOOR_MOVE_TIME / self.dt)))
        for fraction in np.linspace(start, target, steps + 1)[1:]:
            self.scene.door_fraction = float(fraction)
            self.scene.door.T = SE3(
                DISHWASHER_FRONT_X, -BAR_Y,
                DOOR_CLOSED_CENTRE + DOOR_RISE * fraction,
            ).A
            yield self._segment("OPEN_DOOR" if open_door else "CLOSE_DOOR", wait_time=self.dt)
        self.scene.door_open = bool(open_door)
        yield from self._button_release_sequence("blue", release)

    def _conveyor_sequence(self, forward):
        if not self.tray_placed or self.carrying_tray or not self.scene.door_open:
            raise MotionPlanningError("Tray must be on conveyor and dishwasher door open")
        start = self.scene.tray_pose.A.copy()
        target_x = CONVEYOR_TRAVEL_DISTANCE if forward else 0.0
        if not forward and not self.washed:
            raise MotionPlanningError("Tray cannot return before wash")
        if forward and start[0, 3] > 0.01:
            raise MotionPlanningError("Tray already inside dishwasher")
        if not forward and abs(start[0, 3] - CONVEYOR_TRAVEL_DISTANCE) > 0.01:
            raise MotionPlanningError("Tray not inside dishwasher")
        if target_x + TRAY_WIDTH / 2 > CONVEYOR_END_X - 0.02:
            raise MotionPlanningError("Tray would overrun conveyor")
        if forward and target_x - TRAY_WIDTH / 2 <= DISHWASHER_FRONT_X + 0.04:
            raise MotionPlanningError("Tray would not fully enter dishwasher")
        if abs(self.scene.tray_pose.t[1] - CONVEYOR_Y) + TRAY_LENGTH / 2 >= DISHWASHER_WIDTH / 2 - 0.035:
            raise MotionPlanningError("Tray would hit dishwasher walls")
        if CONVEYOR_HEIGHT + RUNNER_HEIGHT + DECK_THICKNESS + 0.17 >= DISHWASHER_TOP:
            raise MotionPlanningError("Glasses would hit dishwasher roof")
        if target_x + TRAY_WIDTH / 2 >= DISHWASHER_BACK_X - 0.03:
            raise MotionPlanningError("Tray would hit dishwasher back")

        release = yield from self._button_press_sequence("green")
        steps = max(1, int(np.ceil(CONVEYOR_MOVE_TIME / self.dt)))
        for progress in np.linspace(0, 1, steps + 1)[1:]:
            smooth = progress**2 * (3 - 2 * progress)
            pose = start.copy()
            pose[0, 3] = start[0, 3] * (1 - smooth) + target_x * smooth
            self.scene.set_tray_pose(SE3(pose))
            yield self._segment("CONVEY_IN" if forward else "CONVEY_OUT", wait_time=self.dt)
        self.obstacles = self._build_obstacles()
        yield from self._button_release_sequence("green", release)
        self.conveyor_finished = forward

    def _wash_sequence(self):
        if not self.scene.door_open or abs(self.scene.tray_pose.t[0] - CONVEYOR_TRAVEL_DISTANCE) > 0.01:
            raise MotionPlanningError("Open door and position tray inside before washing")
        yield from self._toggle_door_sequence(False)
        yield self._segment("WASHING", wait_time=2.0)
        self.washed = True
        yield from self._toggle_door_sequence(True)

    def _collect_clean_tray_sequence(self):
        if not self.washed or not self.scene.door_open:
            raise MotionPlanningError("Dishwasher must finish and door be open")
        if np.linalg.norm(self.scene.tray_pose.t - [0, -BAR_Y, CONVEYOR_HEIGHT]) > 0.008:
            raise MotionPlanningError("Tray not in the loading position")
        if self.rear_pickup_paths is None:
            raise MotionPlanningError("Missing return pickup trajectory")
        insert = self.rear_pickup_paths["insertion"]
        contact = self.rear_pickup_paths["contact"]
        lift = self.rear_pickup_paths["lift"]
        current = np.asarray(self.robot.q, dtype=float).copy()
        arrival = rtb.jtraj(current, insert[0], 115).q
        self.obstacles = self._build_obstacles()
        try:
            self.check_trajectory(arrival, label="approach clean tray")
            arrival_paths = [arrival]
        except MotionPlanningError:
            to_home = rtb.jtraj(current, np.zeros(6), 110).q
            from_home = rtb.jtraj(np.zeros(6), insert[0], 115).q
            self.check_trajectory(to_home, label="leave button area")
            self.check_trajectory(from_home, label="approach clean tray from home")
            arrival_paths = [to_home, from_home]
        self.check_trajectory(insert, label="insert forks under clean tray")
        self.check_trajectory(contact, label="contact clean tray")

        future_offset = np.linalg.inv(self.wrist_pose(contact[-1])) @ self.scene.tray_pose.A
        original_offset = self.tray_offset_from_wrist
        self.tray_offset_from_wrist = future_offset
        try:
            self.check_trajectory(lift, carrying=True, label="lift clean tray")
        finally:
            self.tray_offset_from_wrist = original_offset

        for index, path in enumerate(arrival_paths):
            yield self._segment(f"RETURN_TO_TRAY_{index + 1}", path)
        yield self._segment("INSERT_CLEAN_TRAY", insert)
        yield self._segment("CONTACT_CLEAN_TRAY", contact)
        self.tray_offset_from_wrist = np.linalg.inv(self.wrist_pose(self.robot.q)) @ self.scene.tray_pose.A
        self.carrying_tray = True
        yield self._segment("LIFT_CLEAN_TRAY", lift, carrying=True)
        self.tray_placed = False
        self.wash_complete = True
        self.rear_pickup_paths = None

    def _dishwasher_sequence(self):
        if not self.tray_placed and not self.carrying_tray:
            yield from self._pickup_sequence()
        if self.carrying_tray and not self.wash_complete:
            yield from self._place_sequence()
        if not self.scene.door_open:
            yield from self._toggle_door_sequence(True)
        if not self.conveyor_finished:
            yield from self._conveyor_sequence(True)
        if not self.washed:
            yield from self._wash_sequence()
        if self.conveyor_finished:
            yield from self._conveyor_sequence(False)
        if not self.carrying_tray:
            yield from self._collect_clean_tray_sequence()

    def pickup(self):
        self.change_state("PICKUP")

    def place(self):
        self.change_state("PLACE")

    def toggle_door(self, open_door):
        self.change_state("OPEN_DOOR" if open_door else "CLOSE_DOOR")

    def wash(self):
        self.change_state("WASH")

    def collect_clean_tray(self):
        self.change_state("COLLECT_CLEAN_TRAY")

    def button_and_convey(self):
        self.change_state("CONVEY_IN")

    def dishwasher_cycle(self):
        if self.state != "IDLE" or self.carrying_tray:
            raise MotionPlanningError("Kawasaki must be idle at the original pickup")
        self.tray_placed = False
        self.conveyor_finished = self.washed = self.wash_complete = False
        self.obstacles = self._build_obstacles()
        self.change_state("DISHWASHER_CYCLE")

    def start_sequence(self, sequence):
        if self.stopped or self.state != "IDLE" or self._sequence is not None:
            raise MotionPlanningError("Kawasaki cannot accept another action")
        self._sequence = sequence
        self._advance_state()
