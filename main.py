import sys

UR3_TEST_ONLY = False

UR3_INSPECT_TARGET = False

def start_simulation():
    import numpy as np
    import scipy

    if not hasattr(scipy, 'randn'):
        scipy.randn = np.random.randn

    import swift
    from spatialmath import SE3
    from RS007L import RS007L
    from scene import (
        GlassTrayWorkcell, PEDESTAL_HEIGHT,
        HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT,
    )
    from rs007l_controller import RS007LController, MotionStopped, STEP_TIME
    from A2_scene import A2Scene
    from handover import TrayHandover
    from ur3_controller import UR3Controller
    from teach_pendant import TeachPendant
    from program_control import ProgramControl
    from swift_panel import launch_swift

    simulator = swift.Swift()
    try:
        launch_swift(simulator)
        
        ur3_station = A2Scene(simulator)
        ur3_controller = UR3Controller(ur3_station.ur3, dt=STEP_TIME)
        kawasaki = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        kawasaki.add_to_env(simulator)
        kawasaki.set_joint_angles(np.zeros(6))

        workcell = GlassTrayWorkcell(simulator, kawasaki)
        controller = RS007LController(kawasaki, workcell, simulator, dt=STEP_TIME)
        control = ProgramControl(ur3_controller, controller, [ur3_station.ur3, kawasaki])
        pendant = TeachPendant(simulator, ur3_station.ur3, control)
        
        controller.on_step = pendant.sync
        controller.pause_requested = lambda: control.paused
        workcell.step_callback = controller.step_simulation
        simulator.set_camera_pose([2.4, 2.5, 2.2], [0.3, 0.1, 0.9])
        simulator.step(STEP_TIME)

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
        handover_complete = False
        if UR3_TEST_ONLY:
            workcell.set_tray_pose(SE3(HANDOVER_X, HANDOVER_Y, HANDOVER_HEIGHT))
            handover.send_tray()
            handover_complete = True
            control.phase = "RUNNING"
        elif not control.stopped:
            
            control.phase = "WASHING"
            controller.dishwasher_cycle()

        while True:
            pendant.update(STEP_TIME)
            if not control.paused and not control.stopped:
                controller.update(STEP_TIME)

                if (not handover_complete and controller.state == "IDLE"
                        and controller.wash_complete):
                    control.phase = controller.state = "HANDOVER"
                    try:
                        handover.run()
                    except MotionStopped:
                        pass
                    if not control.stopped:
                        controller.state = "IDLE"
                        control.phase = "RUNNING"
                        handover_complete = True

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
