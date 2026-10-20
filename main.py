import sys
import time
import traceback
from pathlib import Path


def start_simulation():
    import numpy as np
    import scipy

    if not hasattr(scipy, 'randn'):
        scipy.randn = np.random.randn

    import swift
    from spatialmath import SE3
    from RS007L import RS007L
    from scene import GlassTrayWorkcell, PEDESTAL_HEIGHT
    from pickup import TrayController, STEP_TIME
    from A2_scene import A2Scene

    simulator = swift.Swift()
    try:
        simulator.launch(realtime=True)

        a2_scene = A2Scene(simulator)

        robot = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        robot.add_to_env(simulator)
        robot.set_joint_angles(np.zeros(6))

        scene = GlassTrayWorkcell(simulator, robot)
        controller = TrayController(robot, scene, simulator)
        simulator.set_camera_pose([2.3, 2.5, 2.1], [0.35, -0.35, 0.90])
        simulator.step(STEP_TIME)

        controller.dishwasher_cycle()

        while True:
            simulator.step(STEP_TIME)
            time.sleep(STEP_TIME)
    finally:
        simulator.close()


if __name__ == '__main__':
    try:
        input('Press Enter to begin')
    except EOFError:
        pass  # Allow launchers without an interactive stdin to start the scene.
    except KeyboardInterrupt:
        sys.exit(0)

    try:
        start_simulation()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        error_log = Path(__file__).with_name('simulation_error.log')
        with error_log.open('w', encoding='utf-8') as file:
            traceback.print_exc(file=file)
        print(f'Simulation failed. Error details saved to {error_log}', file=sys.stderr)
        sys.exit(1)
