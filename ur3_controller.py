import numpy as np
from spatialmath import SE3
from ir_support.robots import UR3


class UR3Controller:
    def __init__(self, ur3):
        self.ur3 = ur3
        self.q_start = [0, -np.pi/2, 0, 0, 0, 0]
        self.ur3.q = self.q_start
        

    def update(dt, world):

        