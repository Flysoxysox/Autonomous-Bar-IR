from math import pi

import numpy as np
from spatialgeometry import Axes, Cuboid, Cylinder
from spatialmath import SE3
from ir_support_extra_parts.parts import part_mesh

BAR_HEIGHT = 1.00
CONVEYOR_HEIGHT = 1.00
PEDESTAL_HEIGHT = 0.35

HANDOVER_X = 0.48
HANDOVER_Y = 0.405
HANDOVER_SIZE_X = 0.44
HANDOVER_SIZE_Y = 0.43
HANDOVER_HEIGHT = 1.00

BAR_Y = 0.92
CONVEYOR_Y = -1.02
CONVEYOR_VISUAL_Y = -BAR_Y

CONVEYOR_LENGTH = 1.50
CONVEYOR_START_X = -0.30
CONVEYOR_END_X = CONVEYOR_START_X + CONVEYOR_LENGTH
CONVEYOR_CENTRE_X = (CONVEYOR_START_X + CONVEYOR_END_X) / 2
CONVEYOR_WIDTH = 0.72
CONVEYOR_TRAVEL_DISTANCE = 0.95

BUTTON_X = -0.62
BUTTON_Y = 0.0
BUTTON_TOP = 0.438
BUTTON_PRESS_DEPTH = 0.009
BUTTON_RADIUS = 0.044
BUTTON_HOVER_HEIGHT = 0.090

BLUE_BUTTON_X = -0.62
BLUE_BUTTON_Y = -0.20
BLUE_BUTTON_TOP = 0.438

DISHWASHER_FRONT_X = 0.65
DISHWASHER_BACK_X = 1.34
DISHWASHER_WIDTH = 0.90
DISHWASHER_TOP = 1.53
DOOR_HEIGHT = 0.44
DOOR_RISE = 0.53
DOOR_CLOSED_CENTRE = CONVEYOR_HEIGHT + DOOR_HEIGHT / 2

TRAY_WIDTH = 0.38
TRAY_LENGTH = 0.38
DECK_THICKNESS = 0.028
RUNNER_HEIGHT = 0.042
RUNNER_WIDTH = 0.042
RUNNER_X_POSITIONS = (-0.15, 0.0, 0.15)
CHANNEL_X_POSITIONS = (-0.075, 0.075)
CHANNEL_WIDTH = 0.108
RIM_THICKNESS = 0.014
RIM_HEIGHT = 0.022

FORK_WIDTH = 0.060
FORK_LENGTH = 0.35
FORK_THICKNESS = 0.018
FORK_X_POSITIONS = CHANNEL_X_POSITIONS

FORK_MOUNT = SE3.Rx(pi / 2)

GLASS_ORIGIN_ABOVE_DECK = 0.150

BAR_COLOR = [0.32, 0.23, 0.17, 1]
BAR_TOP_COLOR = [0.51, 0.36, 0.24, 1]
CONVEYOR_FRAME_COLOR = [0.20, 0.24, 0.27, 1]
CONVEYOR_BODY_COLOR = [0.24, 0.28, 0.30, 1]
BELT_COLOR = [0.13, 0.14, 0.15, 1]
BUTTON_COLOR = [0.16, 0.78, 0.26, 1]
BUTTON_HOUSING_COLOR = [0.16, 0.20, 0.22, 1]
PEDESTAL_COLOR = [0.35, 0.39, 0.43, 1]
TRAY_COLOR = [0.18, 0.26, 0.32, 1]
TRAY_RIM_COLOR = [0.22, 0.31, 0.37, 1]
FORK_COLOR = [0.60, 0.65, 0.69, 1]

class GlassTrayWorkcell:

    def __init__(self, simulator, robot):
        self.simulator = simulator
        self.robot = robot
        self.tray_pose = SE3(0, BAR_Y, BAR_HEIGHT)
        self.tray_objects = []
        self.glasses = {}  # (row, column) -> actual mesh; rows follow tray y
        self.fork_objects = []
        self._make_furniture()
        self._make_handover_box()
        self._make_tray()
        self._make_fork()
        self._make_button()
        self._make_dishwasher()
        self.update_tray()
        self.update_fork()

        print(f"Workcell ready: bar {BAR_HEIGHT:.2f} m, conveyor {CONVEYOR_HEIGHT:.2f} m, pedestal {PEDESTAL_HEIGHT:.2f} m")
        print("Tray: 9 inverted BeerGlass meshes, two underside channels")
        print("Forks follow the wrist; tray starts on the bar until the pick command")

    def _add_box(self, size, pose, color):

        box = Cuboid(scale=list(size), color=color)
        box.T = SE3(pose).A
        self.simulator.add(box)
        return box

    def _make_furniture(self):

        self._add_box([0.94, 0.51, PEDESTAL_HEIGHT],
                  SE3(-0.26, -0.045, PEDESTAL_HEIGHT / 2), PEDESTAL_COLOR)

        self._add_box([CONVEYOR_LENGTH, CONVEYOR_WIDTH, 0.105],
                  SE3(CONVEYOR_CENTRE_X, CONVEYOR_VISUAL_Y, CONVEYOR_HEIGHT - 0.058),
                  CONVEYOR_BODY_COLOR)
        self._add_box([CONVEYOR_LENGTH - 0.04, CONVEYOR_WIDTH - 0.07, 0.012],
                  SE3(CONVEYOR_CENTRE_X, CONVEYOR_VISUAL_Y, CONVEYOR_HEIGHT - 0.006),
                  BELT_COLOR)
        for side_offset in (-CONVEYOR_WIDTH/2 + 0.018, CONVEYOR_WIDTH/2 - 0.018):
            self._add_box([CONVEYOR_LENGTH, 0.025, 0.065],
                      SE3(CONVEYOR_CENTRE_X, CONVEYOR_VISUAL_Y + side_offset,
                          CONVEYOR_HEIGHT - 0.033), CONVEYOR_FRAME_COLOR)
        for x_position in (CONVEYOR_START_X + 0.12, CONVEYOR_END_X - 0.12):
            for side_offset in (-0.255, 0.255):
                self._add_box([0.075, 0.075, CONVEYOR_HEIGHT - 0.12],
                          SE3(x_position, CONVEYOR_VISUAL_Y + side_offset,
                              (CONVEYOR_HEIGHT - 0.12)/2), CONVEYOR_FRAME_COLOR)


    def _make_handover_box(self):
        self.handover_box = self._add_box(
            [HANDOVER_SIZE_X, HANDOVER_SIZE_Y, HANDOVER_HEIGHT],
            SE3(HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT / 2),
            [0.38, 0.43, 0.47, 1.0]
        )

    def _add_tray_box(self, size, local_pose, color):
        shape = self._add_box(size, self.tray_pose * local_pose, color)
        self.tray_objects.append((shape, SE3(local_pose)))

    def _make_tray(self):
        self.glass_axes = {}  # Keep each cup's displayed frame with its mesh.
        for x_position in RUNNER_X_POSITIONS:
            self._add_tray_box([RUNNER_WIDTH, TRAY_LENGTH, RUNNER_HEIGHT],
                               SE3(x_position, 0, RUNNER_HEIGHT / 2), TRAY_COLOR)

        deck_height = RUNNER_HEIGHT + DECK_THICKNESS / 2
        self._add_tray_box([TRAY_WIDTH, TRAY_LENGTH, DECK_THICKNESS],
                           SE3(0, 0, deck_height), TRAY_COLOR)

        deck_surface = RUNNER_HEIGHT + DECK_THICKNESS
        rim_center_height = deck_surface + RIM_HEIGHT / 2
        for x_position in (-TRAY_WIDTH / 2 + RIM_THICKNESS / 2,
                  TRAY_WIDTH / 2 - RIM_THICKNESS / 2):
            self._add_tray_box([RIM_THICKNESS, TRAY_LENGTH, RIM_HEIGHT],
                               SE3(x_position, 0, rim_center_height), TRAY_RIM_COLOR)
        for y_position in (-TRAY_LENGTH / 2 + RIM_THICKNESS / 2,
                  TRAY_LENGTH / 2 - RIM_THICKNESS / 2):
            self._add_tray_box([TRAY_WIDTH - 2 * RIM_THICKNESS,
                                RIM_THICKNESS, RIM_HEIGHT],
                               SE3(0, y_position, rim_center_height), TRAY_RIM_COLOR)

        for column, x_position in enumerate((-0.118, 0.0, 0.118)):
            for row, y_position in enumerate((-0.118, 0.0, 0.118)):
                glass = part_mesh("BeerGlass")
                local_pose = (SE3(x_position, y_position, deck_surface + GLASS_ORIGIN_ABOVE_DECK)
                              * SE3.Rx(pi))
                glass.T = (self.tray_pose * local_pose).A
                self.simulator.add(glass)
                self.tray_objects.append((glass, local_pose))
                self.glasses[row, column] = glass
                # Register cup frames; main displays only the selected cup's.
                axes = Axes(length=0.12)
                axes.T = glass.T
                self.tray_objects.append((axes, local_pose))
                self.glass_axes[row, column] = axes

    def _add_fork_box(self, size, local_pose, color):
        mounted_pose = FORK_MOUNT * SE3(local_pose)
        shape = self._add_box(size, mounted_pose, color)
        self.fork_objects.append((shape, mounted_pose))

    def _make_fork(self):

        self._add_fork_box([0.028, 0.100, 0.012],
                           SE3(0, 0.050, -0.025), PEDESTAL_COLOR)
        self._add_fork_box([0.190, 0.018, 0.007],
                           SE3(0, 0.101, -0.025), PEDESTAL_COLOR)
        for x_position in FORK_X_POSITIONS:
            self._add_fork_box([FORK_WIDTH, FORK_LENGTH, FORK_THICKNESS],
                               SE3(x_position, 0.065 + FORK_LENGTH / 2, -0.025),
                               FORK_COLOR)

    def update_fork(self):

        wrist_frame = self.robot._joint_frames(self.robot.q)[-1]
        world_wrist = self.robot.world_pose.A @ wrist_frame
        for shape, local_pose in self.fork_objects:
            shape.T = world_wrist @ local_pose.A

    def update_tray(self):
        for shape, local_pose in self.tray_objects:
            shape.T = (self.tray_pose * local_pose).A

    def set_tray_pose(self, pose):

        self.tray_pose = SE3(pose)
        self.update_tray()

    def _make_button(self):

        self._add_box([0.12, 0.12, 0.065],
                  SE3(BUTTON_X, BUTTON_Y, BUTTON_TOP - 0.0525), BUTTON_HOUSING_COLOR)
        self.button = Cylinder(radius=BUTTON_RADIUS, length=0.022,
                                   color=BUTTON_COLOR)
        self.button.T = SE3(BUTTON_X, BUTTON_Y, BUTTON_TOP - 0.011).A
        self.simulator.add(self.button)
        self.button_pressed = False

        self.blue_button = Cylinder(radius=BUTTON_RADIUS, length=0.022, color=[0.14, 0.43, 0.95, 1])
        self._add_box([0.12, 0.12, 0.065],
                      SE3(BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP - 0.0525),
                      BUTTON_HOUSING_COLOR)
        self.blue_button.T = SE3(BLUE_BUTTON_X, BLUE_BUTTON_Y, BLUE_BUTTON_TOP - 0.011).A
        self.simulator.add(self.blue_button)
        self.blue_button_pressed = False

    def set_button_pressed(self, pressed):

        self.button_pressed = bool(pressed)
        height = BUTTON_TOP - 0.011 - (BUTTON_PRESS_DEPTH if pressed else 0.0)
        self.button.T = SE3(BUTTON_X, BUTTON_Y, height).A

    def set_blue_button_pressed(self, pressed):
        self.blue_button_pressed = bool(pressed)
        height = BLUE_BUTTON_TOP - 0.011 - (BUTTON_PRESS_DEPTH if pressed else 0.0)
        self.blue_button.T = SE3(BLUE_BUTTON_X, BLUE_BUTTON_Y, height).A

    def _make_dishwasher(self):
        metal = [0.54, 0.60, 0.65, 1]
        dark = [0.19, 0.24, 0.28, 1]
        length = DISHWASHER_BACK_X - DISHWASHER_FRONT_X
        centre_x = (DISHWASHER_FRONT_X + DISHWASHER_BACK_X) / 2
        side_y = DISHWASHER_WIDTH / 2
        wall_height = DISHWASHER_TOP + 0.045
        wall_centre = wall_height / 2

        for direction in (-1, 1):
            self._add_box([length, 0.035, wall_height],
                          SE3(centre_x, CONVEYOR_VISUAL_Y + direction * side_y, wall_centre), metal)

        self._add_box([length, DISHWASHER_WIDTH + 0.035, 0.045],
                      SE3(centre_x, CONVEYOR_VISUAL_Y, DISHWASHER_TOP + 0.0225), metal)
        self._add_box([0.04, DISHWASHER_WIDTH + 0.035, wall_height],
                      SE3(DISHWASHER_BACK_X - 0.02, CONVEYOR_VISUAL_Y, wall_centre), metal)

        rail_height = CONVEYOR_HEIGHT + DOOR_HEIGHT + DOOR_RISE
        for direction in (-1, 1):
            self._add_box([0.06, 0.06, rail_height],
                          SE3(DISHWASHER_FRONT_X, CONVEYOR_VISUAL_Y + direction * (side_y + 0.0175),
                              rail_height / 2), dark)

        self.door = self._add_box([0.034, DISHWASHER_WIDTH - 0.02, DOOR_HEIGHT],
                                  SE3(DISHWASHER_FRONT_X, CONVEYOR_VISUAL_Y, DOOR_CLOSED_CENTRE),
                                  [0.24, 0.48, 0.67, 1])
        self.door_open = False
        self.door_fraction = 0.0

    def set_door_open(self, opened, frames=16, step_time=0.005):
        start = self.door_fraction
        target = 1.0 if opened else 0.0
        for fraction in np.linspace(start, target, frames):
            self.door_fraction = float(fraction)
            self.door.T = SE3(DISHWASHER_FRONT_X, CONVEYOR_VISUAL_Y,
                              DOOR_CLOSED_CENTRE + DOOR_RISE * self.door_fraction).A
            self.simulator.step(step_time)
        self.door_open = bool(opened)
