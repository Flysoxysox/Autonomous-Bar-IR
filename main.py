import contextlib
import os
import sys
import time
import traceback
from pathlib import Path


@contextlib.contextmanager
def quiet_console():
    with open(os.devnull, 'w') as nowhere:
        originals = []
        for stream in (sys.stdout, sys.stderr):
            try:
                descriptor = stream.fileno()
                backup = os.dup(descriptor)
                os.dup2(nowhere.fileno(), descriptor)
                originals.append((descriptor, backup))
            except (OSError, ValueError):
                continue
        try:
            with contextlib.redirect_stdout(nowhere), contextlib.redirect_stderr(nowhere):
                yield
        finally:
            for descriptor, backup in originals:
                os.dup2(backup, descriptor)
                os.close(backup)


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

    simulator = swift.Swift()
    try:
        simulator.launch(realtime=True)

        robot = RS007L(base=SE3(0, 0, PEDESTAL_HEIGHT))
        robot.add_to_env(simulator)
        robot.set_joint_angles(np.zeros(6))

        scene = GlassTrayWorkcell(simulator, robot)
        controller = TrayController(robot, scene, simulator)
        simulator.set_camera_pose([2.3, 2.5, 2.1], [0.35, -0.35, 0.90])
        simulator.step(STEP_TIME)

        controller.dishwasher_cycle()

        while True:
            time.sleep(0.25)
    finally:
        simulator.close()


if __name__ == '__main__':
    try:
        input('Press Enter to begin')
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    with quiet_console():
        try:
            start_simulation()
        except KeyboardInterrupt:
            pass
        except Exception:
            error_log = Path(__file__).with_name('simulation_error.log')
            with error_log.open('w', encoding='utf-8') as file:
                traceback.print_exc(file=file)
