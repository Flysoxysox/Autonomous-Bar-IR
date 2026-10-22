"""Simple Week 5 line-plane and triangle checks for the light curtain."""

from itertools import product
import numpy as np
from ir_support import line_plane_intersection


def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    # same barycentric-coordinate calculation as week 5, questions 2 and 3
    u = triangle_verts[1] - triangle_verts[0]
    v = triangle_verts[2] - triangle_verts[0]
    w = intersect_p - triangle_verts[0]
    uu, uv, vv = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    wu, wv = np.dot(w, u), np.dot(w, v)
    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    t = (uv * wu - uu * wv) / D
    return s >= -1e-8 and t >= -1e-8 and s + t <= 1 + 1e-8


class LightScanner:
    def __init__(self, pose, half_width, height):
        # rectangle in the scanner frame: x is width, z is height, y is normal
        corners = np.array([[-half_width, 0, 0], [half_width, 0, 0],
                            [half_width, 0, height], [-half_width, 0, height]])
        corners = corners @ pose.R.T + pose.t
        self.triangles = corners[[[0, 1, 2], [0, 2, 3]]]
        self.normal = pose.R[:, 1]
        self.boxes = {}

    def check_segment(self, start, end):
        point, check = line_plane_intersection(
            self.normal, self.triangles[0, 0], start, end
        )
        if check == 1:
            return any(is_intersection_point_inside_triangle(point, triangle)
                       for triangle in self.triangles)
        if check == 2:
            # the lab helper returns no intersection point for coplanar lines
            return any(is_intersection_point_inside_triangle(p, triangle)
                       for p in (start, end) for triangle in self.triangles)
        return False

    def add_box(self, name, shape, bounds):
        # keep the actual swift object so that detection follows its live pose
        # re-registering a glass updates its entry without making duplicates
        self.boxes[id(shape)] = (name, shape, bounds)

    def check_box(self, shape, bounds):
        minimum, maximum = bounds
        vertices = np.array(list(product(*zip(minimum, maximum))))
        vertices = vertices * shape.scale
        vertices = vertices @ shape.T[:3, :3].T + shape.T[:3, 3]
        # skip the edge tests when the whole box is clearly on one side of the plane
        distances = (vertices - self.triangles[0, 0]) @ self.normal
        if np.all(distances > 1e-7) or np.all(distances < -1e-7):
            return False
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for i, j in edges:
            if self.check_segment(vertices[i], vertices[j]):
                return True
        return False

    def check_robot(self, robot):
        # week 5: model each link as a line between successive joint origins
        if hasattr(robot, "_joint_frames"):
            # the kawasaki visuals use these frames rather than its assumed dh model
            frames = [robot.world_pose.A @ frame for frame in robot._joint_frames(robot.q)]
        else:
            frames = [robot.base.A.copy()]
            for link, q in zip(robot.links, robot.q):
                frames.append(frames[-1] @ link.A(q).A)
        for start, end in zip(frames[:-1], frames[1:]):
            if self.check_segment(start[:3, 3], end[:3, 3]):
                return True
        return False

    def detect(self, robots):
        """Return the names of the objects currently intersecting the curtain."""
        detected = [name for name, robot in robots if self.check_robot(robot)]
        for name, shape, bounds in self.boxes.values():
            if self.check_box(shape, bounds):
                detected.append(name)
        return detected
