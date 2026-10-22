import sys

# set true to bypass the kawasaki dishwashing sequence for ur3 testing
UR3_TEST_ONLY = False
# set the inspection flag to true to inspect poses using the teach pendant
UR3_INSPECT_TARGET = False


def start_simulation():
    import numpy as np
    import scipy

    # provide the older random function expected by this robotics toolbox version
    if not hasattr(scipy, 'randn'):
        scipy.randn = np.random.randn

    import swift
    from spatialmath import SE3
    from RS007L import RS007L
    from scene import (
        GlassTrayWorkcell, PEDESTAL_HEIGHT,
        HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT,
    )
    from pickup import TrayController, MotionStopped, STEP_TIME
    from A2_scene import A2Scene
    from handover import TrayHandover
    from ur3_controller import UR3Controller
    from teach_pendant import TeachPendant
    from program_control import ProgramControl
    from swift_panel import launch_swift

    simulator = swift.Swift()
    try:
        launch_swift(simulator)
        # the station owns the tap and ur3 while the controller plans and plays the arm motion
        ur3_station = A2Scene(simulator)
        ur3_controller = UR3Controller(ur3_station.ur3, dt=STEP_TIME)
        kawasaki = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        kawasaki.add_to_env(simulator)
        kawasaki.set_joint_angles(np.zeros(6))

        workcell = GlassTrayWorkcell(simulator, kawasaki)
        controller = TrayController(kawasaki, workcell, simulator)
        control = ProgramControl(ur3_controller, controller, [ur3_station.ur3, kawasaki])
        pendant = TeachPendant(simulator, ur3_station.ur3, control)
        # keep status readouts and stop available during the kawasaki sequence
        controller.on_step = pendant.sync
        controller.pause_requested = lambda: control.paused
        workcell.step_callback = controller.step_simulation
        simulator.set_camera_pose([2.4, 2.5, 2.2], [0.3, 0.1, 0.9])
        simulator.step(STEP_TIME)

        # give the controller the scene poses and the actual cup and tap objects
        ur3_controller.configure_task_poses(
            tr_world_tray=workcell.tray_pose,
            tr_world_tap=ur3_station.tap.pose,
            tr_world_hinge=ur3_station.tap.hinge_pose,
            glasses=workcell.glasses,
            workcell=workcell,
            tap=ur3_station.tap,
        )
        if not UR3_INSPECT_TARGET:
            ur3_controller.request_beer()

        handover = TrayHandover(controller, ur3_station.ur3, ur3_controller)
        try:
            if control.stopped:
                raise MotionStopped()
            if UR3_TEST_ONLY:
                # use the same arrival signal when bypassing the dishwasher
                workcell.set_tray_pose(SE3(HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT))
                handover.send_tray()
            else:
                control.phase = controller.state = "WASHING"
                controller.dishwasher_cycle()
                control.phase = controller.state = "HANDOVER"
                handover.run()
            controller.state = "IDLE"
            control.phase = "RUNNING"
        except MotionStopped:
            # keep the window open with all robots holding their stopped poses
            pass
        while True:
            # apply the next arm and handle sample before updating the scene display
            pendant.update(STEP_TIME)
            if not control.paused:
                ur3_controller.update(STEP_TIME, simulator)
            pendant.sync()
            simulator.step(STEP_TIME)
    finally:
        simulator.close()


if __name__ == '__main__':
    try:
        input('Press Enter to begin')
    except EOFError:
        pass
    except KeyboardInterrupt:
        sys.exit(0)

    try:
        start_simulation()
    except KeyboardInterrupt:
        pass
