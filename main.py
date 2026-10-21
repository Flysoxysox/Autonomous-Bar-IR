import contextlib
import os
import sys
import time
import warnings


def start_simulation():
    import numpy as np
    import scipy

    if not hasattr(scipy, 'randn'):
        scipy.randn = np.random.randn

    import swift
    from spatialmath import SE3
    from RS007L import RS007L
    from FR3 import FR3
    from scene import GlassTrayWorkcell, PEDESTAL_HEIGHT
    from pickup import TrayController, STEP_TIME
    from A2_scene import A2Scene
    from handover import TrayHandover

    simulator = swift.Swift()
    try:
        simulator.launch(realtime=True)
        ur3_station = A2Scene(simulator, teach_pendant=False)
        fairino = FR3(base=SE3(-0.79, 1.05, 1.0) * SE3.Rz(np.pi))
        fairino.add_to_env(simulator)
        fairino.set_joint_angles(np.deg2rad([0, -70, 110, 0, 0, 0]))
        kawasaki = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        kawasaki.add_to_env(simulator)
        kawasaki.set_joint_angles(np.zeros(6))

        workcell = GlassTrayWorkcell(simulator, kawasaki)
        controller = TrayController(kawasaki, workcell, simulator)
        simulator.set_camera_pose([2.4, 2.5, 2.2], [0.3, 0.1, 0.9])
        simulator.step(STEP_TIME)

        controller.dishwasher_cycle()
        TrayHandover(controller, ur3_station.ur3).run()

        while True:
            simulator.step(STEP_TIME)
            time.sleep(STEP_TIME)
    finally:
        simulator.close()


if __name__ == '__main__':
    try:
        input('Press Enter to begin')
    except EOFError:
        pass
    except KeyboardInterrupt:
        sys.exit(0)

    with open(os.devnull, 'w') as nowhere:
        with contextlib.redirect_stdout(nowhere), contextlib.redirect_stderr(nowhere):
            warnings.filterwarnings('ignore')
            try:
                start_simulation()
            except (KeyboardInterrupt, Exception):
                pass
