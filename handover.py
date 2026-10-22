import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation
from roboticstoolbox import jtraj
from spatialmath import SE3

from pickup import MotionPlanningError
from scene import (
    HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT,
    HANDOVER_SIZE_X, HANDOVER_SIZE_Y,
    FORK_MOUNT,
)


class TrayHandover:
    def __init__(self, controller, ur3, receiving_controller=None):
        self.controller = controller
        self.ur3 = ur3
        self.receiving_controller = receiving_controller

    def send_tray(self):
        if self.receiving_controller is None:
            return
        scene = self.controller.scene
        available = np.zeros((3, 3), dtype=bool)
        for cell, glass in scene.glasses.items():
            available[cell] = any(shape is glass for shape, _ in scene.tray_objects)
        tray_pose = SE3(scene.tray_pose.A.copy())
        self.receiving_controller.receive_tray(tray_pose, available)

    def tray_transform(self, angles):
        motion = self.controller
        return motion.wrist_pose(angles) @ motion.tray_offset_from_wrist

    def tray_target_to_tool(self, tray_transform):
        motion = self.controller
        wrist = tray_transform @ np.linalg.inv(motion.tray_offset_from_wrist)
        return wrist @ FORK_MOUNT.A @ SE3(0, 0.24, -0.025).A

    def add_obstacle(self, name, centre, size):
        centre = np.asarray(centre, dtype=float)
        half = np.asarray(size, dtype=float) / 2
        return (name, centre - half, centre + half, False)

    def _find_level_wrist(self, angles):
        test_angles = np.zeros(6)

        def tilt_error(wrist_angle):
            test_angles[4] = wrist_angle
            upward = self.tray_transform(test_angles)[:3, 2]
            return np.sum((upward - [0, 0, 1]) ** 2)

        bounds = (float(self.controller.lower[4]), float(self.controller.upper[4]))
        result = minimize_scalar(tilt_error, bounds=bounds, method='bounded',
                                 options={'xatol': 1e-10})
        test_angles[4] = result.x
        if tilt_error(result.x) > np.sin(np.deg2rad(0.3)) ** 2:
            raise MotionPlanningError('Could not find a level near-zero holding pose')
        return test_angles

    def _check_level(self, path, label):
        motion = self.controller
        for index, angles in enumerate(path):
            if self.tray_transform(angles)[2, 2] < np.cos(np.deg2rad(1.0)):
                raise MotionPlanningError(f'{label} would tilt the tray at step {index}')

    def _make_waypoint_path(self, angles, position, steps, label):
        motion = self.controller
        target = self.tray_transform(angles).copy()
        target[:3, 3] = position
        target_tool = self.tray_target_to_tool(target)
        try:
            path = motion._plan_cartesian_move(
                angles, target_tool, steps, label, carrying=True
            )
            self._check_level(path, label)
            return path
        except MotionPlanningError as error:
            original_error = error

        for guess in (angles, np.asarray(motion.robot.q, dtype=float)):
            try:
                destination = motion._solve_inverse_kinematics(target_tool, guess, label)
                count = max(steps, int(np.max(np.abs(destination - angles)) / np.deg2rad(1.5)) + 1)
                path = jtraj(angles, destination, count).q
                motion.check_trajectory(path, carrying=True, label=label)
                self._check_level(path, label)
                return path
            except MotionPlanningError:
                continue
        raise MotionPlanningError(f'{label}: no safe Cartesian or joint-space route: {original_error}')

    def _orientation_preserving_path(self, start_angles, end_angles, obstacles, label, steps=150):
        motion = self.controller
        target_rotation = self.tray_transform(start_angles)[:3, :3]
        path = []
        previous_angles = start_angles.copy()
        for progress in np.linspace(0, 1, steps):
            smooth = progress * progress * (3 - 2 * progress)
            angles = previous_angles.copy()
            angles[:3] = (1 - smooth) * start_angles[:3] + smooth * end_angles[:3]

            def orientation_error(wrist_angles):
                angles[3:] = wrist_angles
                actual_rotation = self.tray_transform(angles)[:3, :3]
                return Rotation.from_matrix(target_rotation @ actual_rotation.T).as_rotvec()

            solution = least_squares(
                orientation_error, previous_angles[3:],
                bounds=(motion.lower[3:] + 1e-6, motion.upper[3:] - 1e-6),
                max_nfev=90
            )
            angles[3:] = solution.x
            if np.linalg.norm(orientation_error(angles[3:])) > np.deg2rad(0.5):
                raise MotionPlanningError(f'{label}: could not keep the tray square and level')
            if np.max(np.abs(angles - previous_angles)) > np.deg2rad(10):
                raise MotionPlanningError(f'{label}: discontinuity in joint movement')
            collision = motion.check_collision(angles, carrying=True, obstacles=obstacles)
            if collision is not None:
                raise MotionPlanningError(f'{label}: {collision}')
            path.append(angles.copy())
            previous_angles = angles.copy()
        return np.asarray(path)

    def _plan_handover_approach(self, front_angles, old_obstacles, extra_obstacles):
        motion = self.controller
        start_tray = self.tray_transform(front_angles)
        failures = []
        for clearance_height in (1.42,):
            target = start_tray.copy()
            target[:3, 3] = (HANDOVER_X, HANDOVER_Y, clearance_height)
            tool_target = self.tray_target_to_tool(target)
            for guess in (
                front_angles,
                np.deg2rad([87, 1, -79, 89, -88, -81]),
            ):
                try:
                    destination_angles = motion._solve_inverse_kinematics(
                        tool_target, guess, 'handover clearance pose'
                    )
                    over_box = self._orientation_preserving_path(
                        front_angles, destination_angles, old_obstacles + extra_obstacles,
                        'swing-back to handover box'
                    )
                    motion.obstacles = old_obstacles + extra_obstacles[1:]
                    final_target = self.tray_transform(over_box[-1]).copy()
                    final_target[:3, 3] = (HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT)
                    descent = motion._plan_cartesian_move(
                        over_box[-1], self.tray_target_to_tool(final_target),
                        85, 'lower tray onto handover box', carrying=True
                    )
                    self._check_level(descent, 'handover lowering')
                    return [over_box, descent]
                except MotionPlanningError as error:
                    failures.append(str(error))
                finally:
                    motion.obstacles = old_obstacles + extra_obstacles
        raise MotionPlanningError('No safe swing-back handover route: ' + '; '.join(failures))

    def run(self):
        motion = self.controller
        if not motion.carrying_tray or not motion.wash_complete:
            raise MotionPlanningError('The Kawasaki must first collect the washed tray')
        if motion.tray_offset_from_wrist is None:
            raise MotionPlanningError('No tray-to-wrist transform available')

        old_obstacles = list(motion.obstacles)
        extra_obstacles = [
            self.add_obstacle('handover platform',
                              (HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT / 2),
                              (HANDOVER_SIZE_X, HANDOVER_SIZE_Y, HANDOVER_HEIGHT)),
            self.add_obstacle('UR3e lower worktop',
                              (0, 1.0, 0.975), (2.20, 0.75, 0.05)),
            self.add_obstacle('UR3e upper bar',
                              (0, 1.25, 1.275), (2.20, 0.25, 0.05)),
            self.add_obstacle('UR3e parked pedestal',
                              np.asarray(self.ur3.base.A)[:3, 3] + [0, 0, 0.14],
                              (0.17, 0.17, 0.28)),
            self.add_obstacle('UR3e parked arm',
                              np.asarray(self.ur3.base.A)[:3, 3] + [0.28, 0, 0.16],
                              (0.70, 0.18, 0.24)),
        ]
        motion.obstacles = old_obstacles + extra_obstacles

        try:
            current_angles = np.asarray(motion.robot.q, dtype=float).copy()
            start_angles = current_angles.copy()
            start_angles[0] = float(np.clip(
                start_angles[0], self.controller.lower[0] + np.deg2rad(1),
                self.controller.upper[0] - np.deg2rad(1)
            ))
            preparation_path = jtraj(current_angles, start_angles, 15).q
            motion.check_trajectory(preparation_path, carrying=True,
                                    label='move clear of base joint limit')
            raised_tray_height = 1.325
            present_height = self.tray_transform(start_angles)[2, 3]
            if present_height >= raised_tray_height:
                raise_path = np.asarray([start_angles])
            else:
                goal = motion.tool_pose(start_angles)
                goal[:3, 3] += [0, 0, raised_tray_height - present_height]
                raise_path = motion._plan_cartesian_move(
                    start_angles, goal, 72, 'raise clean tray over bar', carrying=True
                )

            raised_angles = raise_path[-1].copy()
            front_angles = raised_angles.copy()
            front_angles[0] = 0.0
            swivel_path = jtraj(raised_angles, front_angles, 135).q
            motion.check_trajectory(swivel_path, carrying=True,
                                    label='move around to UR3e station')
            self._check_level(swivel_path, 'swivel to UR3e')

            handover_paths = self._plan_handover_approach(
                front_angles, old_obstacles, extra_obstacles
            )
            lowering_path = handover_paths[-1]

            self._check_level(preparation_path, 'initial clearance')
            self._check_level(raise_path, 'raising')
            for index, path in enumerate(handover_paths):
                self._check_level(path, f'handover segment {index + 1}')

            final_pose = self.tray_transform(lowering_path[-1])
            if np.linalg.norm(final_pose[:3, 3] -
                              [HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT]) > 0.004:
                raise MotionPlanningError('Tray cannot be positioned on the handover box')
            tray_heading = np.arctan2(final_pose[1, 0], final_pose[0, 0])
            square_error = abs((tray_heading + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            if square_error > np.deg2rad(1):
                raise MotionPlanningError('Tray is not square to the handover box')

            for path in (preparation_path, raise_path, swivel_path,
                         *handover_paths):
                motion._animate_path(path, carrying=True)

            motion.scene.set_tray_pose(SE3(final_pose))
            motion.carrying_tray = False
            motion.tray_placed = True
            motion.tray_offset_from_wrist = None
            motion.obstacles = motion._build_obstacles(
                tray_position=final_pose[:3, 3]
            ) + extra_obstacles[1:]
            # leave the forks in the placed tray while the ur3 serves
            self.send_tray()
            return final_pose
        except Exception:
            motion.obstacles = old_obstacles
            raise
