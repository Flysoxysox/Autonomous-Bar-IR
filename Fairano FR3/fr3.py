import os
import argparse
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot


class FairinoFR3(UTSMeshRobot):
    def __init__(self, base=None):
        links = self._create_DH()
        mesh_dir = os.path.dirname(os.path.abspath(__file__))
        link_names = {f"link{i}": f"FR3_link{i}" for i in range(7)}

        # These STLs share an assembly frame at q = 0 (coordinates in metres).
        # CAD +Y becomes base +Z; CAD +X becomes base -X.
        # This is the existing workcell model's alignment, pending visual checks.
        cad_to_base = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        self.link_colors = [(1.0, 1.0, 1.0, 1.0)] * 7

        super().__init__(
            links=links,
            mesh_stem="FairinoFR3",
            mesh_dir=mesh_dir,
            name="FairinoFR3",
            home_q=[0, 0, 0, 0, 0, 0],
            base=base,
            link3d_names=link_names,
            # UTSMeshRobot computes each mesh's relation to its DH link frame.
            qtest_transforms=[cad_to_base.copy() for _ in range(7)],
        )

    @staticmethod
    def _create_DH():

        dh_params = [
            # d,       a,        alpha (deg)
            [0.1400,   0.0000,    90],
            [0.0000,  -0.2800,     0],
            [0.0000,  -0.2400,     0],
            [0.1020,   0.0000,    90],
            [0.1020,   0.0000,   -90],
            [0.1000,   0.0000,     0],
        ]

        dh_links = []

        for d, a, alpha_deg in dh_params:
            alpha_rad = np.deg2rad(alpha_deg)

            dh_links.append(
                rtb.RevoluteDH(
                    a=a,
                    d=d,
                    alpha=alpha_rad
                )
            )

        return dh_links


# ---------------- MAIN PROGRAM ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone FR3 preview")
    parser.add_argument("--dh", action="store_true", help="Show the original Matplotlib DH test")
    args = parser.parse_args()

    if args.dh:
        robot = rtb.DHRobot(FairinoFR3._create_DH(), name="FairinoFR3")
        # Convenient test configuration, not a confirmed physical home pose.
        q_start = np.deg2rad([0, -90, 90, 0, 90, 0])
        print(robot)
        robot.teach(q=q_start, backend="pyplot", block=True)
        print("Joint angles (deg):", np.rad2deg(robot.q))
        print("End-effector transform:")
        print(robot.fkine(robot.q))
    else:
        import swift

        robot = FairinoFR3()
        env = swift.Swift()
        env.launch(realtime=True)
        robot.add_to_env(env)

        def move_joint(value, joint):
            q = robot.q.copy()
            q[joint] = np.deg2rad(float(value))
            # Assign the whole array so UTSMeshRobot updates the meshes.
            robot.q = q

        for joint in range(robot.n):
            # Small inspection range around CAD zero, not physical joint limits.
            env.add(swift.Slider(
                lambda value, j=joint: move_joint(value, j),
                min=-30, max=30, step=1, value=0,
                desc=f"J{joint + 1}", unit="deg",
            ))

        print("Mesh preview: move one joint at a time, then return it to zero.")
        print("Slider range is for alignment inspection only (-30 to +30 deg).")
        # Stepping sends mesh poses to Swift and processes slider callbacks.
        try:
            while True:
                env.step(0.05)
        except KeyboardInterrupt:
            env.close()
