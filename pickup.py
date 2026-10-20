from math import pi

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from roboticstoolbox import jtraj
from spatialmath import SE3

from scene import (
    BAR_Y, BAR_HEIGHT, CONVEYOR_Y, CONVEYOR_HEIGHT,
    CONVEYOR_LENGTH, CONVEYOR_WIDTH, CONVEYOR_CENTRE_X, CONVEYOR_TRAVEL_DISTANCE,
    CONVEYOR_START_X, CONVEYOR_END_X,
    BUTTON_X, BUTTON_Y, BUTTON_TOP, BUTTON_PRESS_DEPTH, BUTTON_RADIUS,
    BUTTON_HOVER_HEIGHT, BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP,
    DISHWASHER_FRONT_X, DISHWASHER_BACK_X, DISHWASHER_WIDTH, DISHWASHER_TOP,
    DOOR_HEIGHT, DOOR_CLOSED_CENTRE, DOOR_RISE,
    TRAY_WIDTH, TRAY_LENGTH, DECK_THICKNESS, RUNNER_HEIGHT,
    RUNNER_WIDTH, RUNNER_X_POSITIONS, RIM_HEIGHT, RIM_THICKNESS,
    FORK_MOUNT, FORK_X_POSITIONS, FORK_WIDTH, FORK_LENGTH, FORK_THICKNESS,
)

PICKUP_ANGLES_DEGREES = [0, 22, -79, 0, 10, 0]
STEP_TIME = 0.1
DISPLAY_EVERY = 4

PICKUP_HEIGHT_CORRECTION = 0.012
PICKUP_BACKOFF = 0.015
APPROACH_DISTANCE = 0.370
CONTACT_RISE = 0.006
TRAY_LIFT = 0.055

class MotionPlanningError(Exception):
    pass

class TrayController:
    def __init__(self, robot, scene, simulator):
        self.robot, self.scene, self.simulator = robot, scene, simulator
        self.carrying_tray = False
        self.tray_placed = False
        self.conveyor_finished = False
        self.washed = False
        self.wash_complete = False
        self.rear_pickup_paths = None
        self.tray_offset_from_wrist = None
        self.pickup_paths = None
        self.button_hover_angles = {}
        self.obstacles = self._build_obstacles()
        self.lower = np.asarray(robot.qlim[0], dtype=float).copy()
        self.upper = np.asarray(robot.qlim[1], dtype=float).copy()

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

        add('bar body', (0, BAR_Y, (BAR_HEIGHT - 0.06)/2),
            (1.25, 0.72, BAR_HEIGHT - 0.06))
        add('bar countertop', (0, BAR_Y, BAR_HEIGHT - .03),
            (1.32, .77, .06))

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

    def check_collision(self, joint_angles, carrying=False, obstacles=None):

        joint_angles = np.asarray(joint_angles, dtype=float)
        if not np.isfinite(joint_angles).all() or np.any(joint_angles < self.lower) or np.any(joint_angles > self.upper):
            return 'joint limit or invalid joint value'
        frames = [self.robot.world_pose.A @ transform for transform in self.robot._joint_frames(joint_angles)]
        if obstacles is None:
            obstacles = self.obstacles
        obstacles = [o for o in obstacles if not carrying or not o[3]]

        for index in range(2, 6):
            link_start, link_end = frames[index][:3, 3], frames[index+1][:3, 3]
            link_radius = .037 if index in (2, 3, 4) else .028
            sample_count = max(3, int(np.linalg.norm(link_end-link_start)/.025) + 1)
            for t in np.linspace(0, 1, sample_count):
                point = link_start * (1-t) + link_end * t
                for name, min_corner, max_corner, _ in obstacles:
                    if self._sphere_intersects_box(point, link_radius, min_corner, max_corner):
                        return f'robot link {index} near {name}'

        transform = self.wrist_pose(joint_angles) @ FORK_MOUNT.A
        boxes = [([0, .035, 0], [.23, .070, .065], 'fork mounting box')]
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
        for index, joint_angles in enumerate(trajectory):
            collision_message = self.check_collision(joint_angles, carrying=carrying, obstacles=obstacles)
            if collision_message is not None:
                raise MotionPlanningError(f'{label}: collision at step {index}: {collision_message}')

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
        def residual(joint_angles):
            pose_error = self._tool_pose_error(joint_angles, target)
            return np.r_[3*pose_error[:3], pose_error[3:]]
        solution = least_squares(residual, np.clip(starting_guess, safe_minimum, safe_maximum),
                               bounds=(safe_minimum, safe_maximum), max_nfev=400)
        pose_error = self._tool_pose_error(solution.x, target)
        if np.linalg.norm(pose_error[:3]) > .002 or np.linalg.norm(pose_error[3:]) > np.deg2rad(1):
            raise MotionPlanningError(f'{label}: IK could not reach desired tool pose')
        return solution.x

    def _plan_cartesian_move(self, starting_joint_angles, destination, step_count, label, carrying=False):

        start = self.tool_pose(starting_joint_angles)

        if np.linalg.norm(Rotation.from_matrix(
                destination[:3,:3] @ start[:3,:3].T).as_rotvec()) > 1e-3:
            raise MotionPlanningError(f'{label}: changing tool rotation not supported')
        positions = np.linspace(start[:3, 3], destination[:3, 3], step_count)
        joint_angles = starting_joint_angles.copy()
        trajectory = [joint_angles.copy()]
        for point in positions[1:]:
            target = start.copy()
            target[:3,3] = point

            for _ in range(15):
                pose_error = self._tool_pose_error(joint_angles, target)
                if np.linalg.norm(pose_error[:3]) < .00045 and np.linalg.norm(pose_error[3:]) < .003:
                    break
                jacobian = self._calculate_jacobian(joint_angles)
                damping = .006
                joint_step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2*np.eye(6), pose_error)
                largest_joint_step = np.max(np.abs(joint_step))
                if largest_joint_step > np.deg2rad(2):
                    joint_step *= np.deg2rad(2) / largest_joint_step
                joint_angles = joint_angles + joint_step
                if np.any(joint_angles < self.lower) or np.any(joint_angles > self.upper):
                    raise MotionPlanningError(f'{label}: joint limit encountered')
            pose_error = self._tool_pose_error(joint_angles, target)
            if np.linalg.norm(pose_error[:3]) > .002 or np.linalg.norm(pose_error[3:]) > np.deg2rad(1):
                raise MotionPlanningError(f'{label}: RMRC lost the desired tool pose')
            collision_message = self.check_collision(joint_angles, carrying=carrying)
            if collision_message:
                raise MotionPlanningError(f'{label}: collision: {collision_message}')
            trajectory.append(joint_angles.copy())
        return np.asarray(trajectory)

    def _animate_path(self, joint_path, carrying=False):
        if len(joint_path) == 0:
            return
        frame_indices = list(range(0, len(joint_path), DISPLAY_EVERY))
        if frame_indices[-1] != len(joint_path) - 1:
            frame_indices.append(len(joint_path) - 1)
        for index in frame_indices:
            joint_angles = joint_path[index]
            self.robot.set_joint_angles(joint_angles)
            self.scene.update_fork()
            if carrying:
                self.scene.set_tray_pose(SE3(self.wrist_pose(joint_angles) @ self.tray_offset_from_wrist))
            self.simulator.step(STEP_TIME)

    def pickup(self):
        if self.tray_placed:
            print('The tray has already been placed on the conveyor. Restart to reset the scene.')
            return
        if self.carrying_tray:
            print('The tray is already being carried.')
            return

        print('Planning collision-checked approach and RMRC insertion...')
        pickup_joint_angles = np.deg2rad(PICKUP_ANGLES_DEGREES)
        target = self.tool_pose(pickup_joint_angles)
        target[:3,3] += [0, -PICKUP_BACKOFF, PICKUP_HEIGHT_CORRECTION]
        insertion_joint_angles = self._solve_inverse_kinematics(target, pickup_joint_angles, 'aligned insertion pose')

        staging = target.copy()
        staging[1,3] -= APPROACH_DISTANCE
        staging_joint_angles = self._solve_inverse_kinematics(staging, insertion_joint_angles, 'approach pose')

        current_joint_angles = np.asarray(self.robot.q, dtype=float).copy()
        approach_path = jtraj(current_joint_angles, staging_joint_angles, 120).q
        self.check_trajectory(approach_path, label='staged approach')

        insertion_path = self._plan_cartesian_move(staging_joint_angles, target, 85, 'horizontal fork insertion')

        contact_pose = self.tool_pose(insertion_path[-1])
        contact_pose[2,3] += CONTACT_RISE
        contact_path = self._plan_cartesian_move(insertion_path[-1], contact_pose, 14,
                                  'raise prongs to tray underside')

        lifting_pose = self.tool_pose(contact_path[-1])
        lifting_pose[2,3] += TRAY_LIFT
        lift_path = self._plan_cartesian_move(contact_path[-1], lifting_pose, 45,
                               'lift loaded tray', carrying=True)

        print('Plan accepted: approach -> horizontal insertion -> contact -> lift.')
        self._animate_path(approach_path)
        self._animate_path(insertion_path)
        self._animate_path(contact_path)

        self.tray_offset_from_wrist = np.linalg.inv(self.wrist_pose(self.robot.q)) @ self.scene.tray_pose.A
        self.carrying_tray = True
        self._animate_path(lift_path, carrying=True)
        self.pickup_paths = {'insertion': insertion_path.copy(),
                                'contact': contact_path.copy(),
                                'lift': lift_path.copy()}
        print('Tray picked up and lifted approximately 55 mm. Holding position.')
        print('Final joints:', np.round(np.rad2deg(self.robot.q), 1))

    def place(self):

        if not self.carrying_tray or self.pickup_paths is None:
            print('No tray held. Type "pick" first.')
            return
        if not np.isclose(CONVEYOR_HEIGHT, BAR_HEIGHT, atol=1e-6):
            raise MotionPlanningError('Placement assumes matching bar and conveyor heights.')
        if not np.isclose(CONVEYOR_Y, -BAR_Y, atol=1e-6):
            raise MotionPlanningError('Placement assumes conveyor loading position behind the bar.')

        lifted_joint_angles = np.asarray(self.robot.q, dtype=float).copy()

        turn_angle = (-pi + 1e-5) if lifted_joint_angles[0] > -0.01 else (pi - 1e-5)
        rear_joint_angles = lifted_joint_angles.copy()
        rear_joint_angles[0] += turn_angle
        if rear_joint_angles[0] < self.lower[0] or rear_joint_angles[0] > self.upper[0]:
            raise MotionPlanningError('JT1 cannot complete a 180 degree turn.')

        def mirror(path):
            mirrored = np.asarray(path, dtype=float).copy()
            mirrored[:, 0] += turn_angle
            return mirrored

        print('Planning collision-checked half-turn, lowering and withdrawal...')

        rotation_path = jtraj(lifted_joint_angles, rear_joint_angles, 160).q
        lowering_path = mirror(self.pickup_paths['lift'][::-1])
        disengage_path = mirror(self.pickup_paths['contact'][::-1])
        withdrawal_path = mirror(self.pickup_paths['insertion'][::-1])

        self.check_trajectory(rotation_path, carrying=True, label='180-degree swivel')
        self.check_trajectory(lowering_path, carrying=True, label='level lowering')

        final_tray_pose = self.wrist_pose(lowering_path[-1]) @ self.tray_offset_from_wrist
        target_position = np.array([0.0, CONVEYOR_Y, CONVEYOR_HEIGHT])
        if np.linalg.norm(final_tray_pose[:3, 3] - target_position) > .008:
            raise MotionPlanningError(
                f'Predicted tray position {np.round(final_tray_pose[:3,3], 3)} '
                f'is not centred over the conveyor loading position at {target_position}.')
        for label, trajectory in [('swivel', rotation_path), ('lowering', lowering_path)]:
            for joint_angles in trajectory:
                transform = self.wrist_pose(joint_angles) @ self.tray_offset_from_wrist
                if np.dot(transform[:3, 2], [0, 0, 1]) < np.cos(np.deg2rad(1)):
                    raise MotionPlanningError(f'{label}: tray would tilt')

        placed_obstacles = self._build_obstacles(tray_position=target_position)
        self.check_trajectory(disengage_path, label='fork disengagement',
                              obstacles=placed_obstacles)
        self.check_trajectory(withdrawal_path, label='fork withdrawal',
                              obstacles=placed_obstacles)

        print('Placement accepted: swivel -> lower -> release -> withdraw.')
        self._animate_path(rotation_path, carrying=True)
        self._animate_path(lowering_path, carrying=True)

        self.scene.set_tray_pose(SE3(final_tray_pose))
        self.carrying_tray = False
        self.tray_offset_from_wrist = None
        self.obstacles = placed_obstacles
        self._animate_path(disengage_path)
        self._animate_path(withdrawal_path)
        self.rear_pickup_paths = {
            "insertion": withdrawal_path[::-1].copy(),
            "contact": disengage_path[::-1].copy(),
            "lift": lowering_path[::-1].copy(),
        }
        self.pickup_paths = None
        self.tray_placed = True
        print('Tray placed on the conveyor. Forks withdrawn.')
        print('Final joints:', np.round(np.rad2deg(self.robot.q), 1))

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

            def error(q):
                return np.r_[4*(self._fork_tip_position(q) - goal),
                             .9*(self._fork_direction(q) - downward_direction),
                             .003*(q-guess)]

            result = least_squares(error, guess, bounds=(lower, upper), max_nfev=250)
            q = result.x
            if np.linalg.norm(self._fork_tip_position(q) - goal) > .0025:
                continue
            if self._fork_direction(q)[2] > -np.cos(np.deg2rad(3)):
                continue
            if self.check_collision(q):
                continue
            path = jtraj(home, q, 105).q
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
            direct_path = jtraj(current, hover_angles, 95).q
            try:
                self.check_trajectory(direct_path, label=f'approach {name} button')
                approach = [direct_path]
            except MotionPlanningError:
                to_home = jtraj(current, np.zeros(6), 110).q
                self.check_trajectory(to_home, label=f'safe approach to {name} button')
                if home_approach is None:
                    home_approach = jtraj(np.zeros(6), hover_angles, 105).q
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

    def _move_to_button(self, name):
        self.obstacles = self._build_obstacles()
        approach, press, release = self._plan_button_paths(name)
        for path in approach:
            self._animate_path(path)
        self._animate_path(press)
        if name == 'blue':
            self.scene.set_blue_button_pressed(True)
        else:
            self.scene.set_button_pressed(True)
        self.simulator.step(STEP_TIME)
        return release

    def _release_button(self, name, release):
        self._animate_path(release)
        if name == 'blue':
            self.scene.set_blue_button_pressed(False)
        else:
            self.scene.set_button_pressed(False)
        self.simulator.step(STEP_TIME)
        self.obstacles = self._build_obstacles()

    def toggle_door(self, open_door):
        if self.carrying_tray:
            raise MotionPlanningError('Cannot operate the dishwasher while carrying the tray')
        if self.scene.door_open == open_door:
            return
        print('Opening dishwasher door...' if open_door else 'Closing dishwasher door...')
        release = self._move_to_button('blue')
        self.scene.set_door_open(open_door, step_time=STEP_TIME)
        self._release_button('blue', release)
        self.obstacles = self._build_obstacles()

    def _move_conveyor(self, forward):
        if not self.tray_placed or self.carrying_tray:
            raise MotionPlanningError('Tray must be on the conveyor')
        if not self.scene.door_open:
            raise MotionPlanningError('Open dishwasher door before moving the conveyor')
        start = self.scene.tray_pose.A.copy()
        target_x = CONVEYOR_TRAVEL_DISTANCE if forward else 0.0
        if not forward and not self.washed:
            raise MotionPlanningError('Dishwasher cycle has not finished')
        if forward and start[0, 3] > .01:
            raise MotionPlanningError('Tray is already in the dishwasher')
        if not forward and abs(start[0, 3] - CONVEYOR_TRAVEL_DISTANCE) > .01:
            raise MotionPlanningError('Tray is not in the dishwasher')
        if target_x + TRAY_WIDTH/2 > CONVEYOR_END_X-.02:
            raise MotionPlanningError('Tray would overrun conveyor')
        if forward and target_x - TRAY_WIDTH/2 <= DISHWASHER_FRONT_X + .04:
            raise MotionPlanningError('Tray would not fully enter dishwasher')
        if abs(self.scene.tray_pose.t[1] - CONVEYOR_Y) + TRAY_LENGTH/2 >= DISHWASHER_WIDTH/2 - .035:
            raise MotionPlanningError('Tray would hit dishwasher side walls')
        if CONVEYOR_HEIGHT + RUNNER_HEIGHT + DECK_THICKNESS + .17 >= DISHWASHER_TOP:
            raise MotionPlanningError('Glasses would hit dishwasher roof')
        if target_x + TRAY_WIDTH/2 >= DISHWASHER_BACK_X - .03:
            raise MotionPlanningError('Tray would hit dishwasher back wall')
        self.obstacles = self._build_obstacles()
        release = self._move_to_button('green')
        print('Green button held. Moving tray into dishwasher...' if forward else
              'Green button held. Returning tray along conveyor...')
        for progress in np.linspace(0, 1, 30):
            smooth = progress**2*(3-2*progress)
            pose = start.copy()
            pose[0,3] = start[0,3]*(1-smooth) + target_x*smooth
            self.scene.set_tray_pose(SE3(pose))
            self.simulator.step(STEP_TIME)
        self.obstacles = self._build_obstacles()
        self._release_button('green', release)
        self.conveyor_finished = forward
        print('Conveyor stopped. Green button released.')

    def wash(self):
        import time
        if not self.scene.door_open or abs(self.scene.tray_pose.t[0] - CONVEYOR_TRAVEL_DISTANCE) > .01:
            raise MotionPlanningError('Tray must be inside dishwasher with door open before washing')
        self.toggle_door(False)
        print('Dishwasher closed. Washing for 2 seconds...')
        time.sleep(2)
        self.washed = True
        self.toggle_door(True)
        print('Dishwasher open. Ready to return tray.')

    def collect_clean_tray(self):
        if not self.washed or self.scene.door_open is False:
            raise MotionPlanningError('Dishwasher must finish and door must be open')
        if np.linalg.norm(self.scene.tray_pose.t - [0, CONVEYOR_Y, CONVEYOR_HEIGHT]) > .008:
            raise MotionPlanningError('Tray is not at the loading position')
        if self.rear_pickup_paths is None:
            raise MotionPlanningError('Missing return pickup trajectory')
        insert = self.rear_pickup_paths['insertion']
        contact = self.rear_pickup_paths['contact']
        lift = self.rear_pickup_paths['lift']
        current_angles = np.asarray(self.robot.q, dtype=float).copy()
        start = jtraj(current_angles, insert[0], 115).q
        self.obstacles = self._build_obstacles()
        try:
            self.check_trajectory(start, label='approach clean tray')
            arrival_paths = [start]
        except MotionPlanningError:
            to_home = jtraj(current_angles, np.zeros(6), 110).q
            from_home = jtraj(np.zeros(6), insert[0], 115).q
            self.check_trajectory(to_home, label='leave button area before collecting tray')
            self.check_trajectory(from_home, label='approach clean tray from home')
            arrival_paths = [to_home, from_home]
        self.check_trajectory(insert, label='insert forks under clean tray')
        self.check_trajectory(contact, label='raise forks under clean tray')
        self.check_trajectory(lift, carrying=True, label='lift clean tray')
        for path in arrival_paths:
            self._animate_path(path)
        self._animate_path(insert)
        self._animate_path(contact)
        self.tray_offset_from_wrist = np.linalg.inv(self.wrist_pose(self.robot.q)) @ self.scene.tray_pose.A
        self.carrying_tray = True
        self._animate_path(lift, carrying=True)
        self.tray_placed = False
        self.wash_complete = True
        self.rear_pickup_paths = None
        print('Clean tray lifted off conveyor. Kawasaki holding tray.')

    def dishwasher_cycle(self):
        if not self.tray_placed and not self.carrying_tray:
            self.pickup()
        if self.carrying_tray and not self.wash_complete:
            self.place()
        if not self.scene.door_open:
            self.toggle_door(True)
        if not self.conveyor_finished:
            self._move_conveyor(True)
        if not self.washed:
            self.wash()
        if self.conveyor_finished:
            self._move_conveyor(False)
        if not self.carrying_tray:
            self.collect_clean_tray()

    def button_and_convey(self):
        self._move_conveyor(True)
