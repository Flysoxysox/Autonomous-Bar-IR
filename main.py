import sys


def start_simulation():
    import numpy as np
    import scipy
    if not hasattr(scipy, "randn"):
        scipy.randn = np.random.randn
    import swift
    from spatialmath import SE3
    from RS007L import RS007L
    from scene import GlassTrayWorkcell, PEDESTAL_HEIGHT
    from rs007l_controller import RS007LController, STEP_TIME
    from A2_scene import A2Scene
    from handover import TrayHandover
    from ur3_controller import UR3Controller
    from fr3_controller import FR3Controller
    from teach_pendant import TeachPendant
    from program_control import ProgramControl
    from swift_panel import launch_swift
    from light_scanner import LightScanner
    from A2_scene import LIGHT_CURTAIN_HALF_SPAN

    simulator = swift.Swift()
    try:
        launch_swift(simulator)
        station = A2Scene(simulator)
        ur3 = UR3Controller(station.ur3, dt=STEP_TIME)
        kawasaki = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        kawasaki.add_to_env(simulator)
        kawasaki.set_joint_angles(np.zeros(6))
        workcell = GlassTrayWorkcell(simulator, kawasaki)
        tray = RS007LController(kawasaki, workcell, simulator, dt=STEP_TIME)
        fr3 = FR3Controller(station.fr3, dt=STEP_TIME)
        fr3.configure_task_poses(workcell.tray_pose, workcell.glasses, workcell=workcell)
        ur3.selected_glass = (2, 2)
        ur3.configure_task_poses(workcell.tray_pose, station.tap.pose, station.tap.hinge_pose, workcell.glasses, workcell=workcell, tap=station.tap)
        handover = TrayHandover(tray, station.ur3, ur3)
        scanner = LightScanner(station.light_curtain_pose, LIGHT_CURTAIN_HALF_SPAN, 0.816)
        # local bounds of the packaged person mesh, rounded outwards in metres
        scanner.add_box("Person", station.person,
                        ([-0.133, -0.301, -0.002], [0.656, 0.284, 1.868]))
        control = ProgramControl(ur3, tray, [station.ur3, kawasaki, station.fr3],
                                 fr3, handover, scanner=scanner)
        pendant = TeachPendant(simulator, station.ur3, control)
        tray.pause_requested = lambda: control.paused
        simulator.set_camera_pose([2.4, 2.5, 2.2], [0.3, 0.1, 0.9])
        display_time = 0.0
        display_interval = 0.02
        while True:
            # manual motion needs a check before jogging; update checks automatic motion
            if control.state == "MANUAL":
                control.check_safety()
                try:
                    pendant.update(STEP_TIME)
                except Exception as error:
                    control.fault(f"Manual control failed: {error}")
            else:
                control.update(STEP_TIME, simulator)
            station.set_safety_light(control.light_colour)
            # keep the small control steps but send the scene and labels at 50 fps
            display_time += STEP_TIME
            if display_time + 1e-12 >= display_interval:
                pendant.sync()
                simulator.step(display_time)
                display_time = 0.0
    finally:
        simulator.close()


if __name__ == "__main__":
    try:
        input("Press Enter to begin")
    except EOFError:
        pass
    except KeyboardInterrupt:
        sys.exit(0)
    try:
        start_simulation()
    except KeyboardInterrupt:
        pass
