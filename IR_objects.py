"""Reusable Cuboid scenery for the shared Swift simulation (dimensions in m)."""

from math import isfinite

from spatialgeometry import Cuboid
from spatialmath import SE3


def create_bar_counter(
    env,
    pose=None,
    *,
    length=2.20,
    depth=0.75,
    lower_height=1.2,
    upper_height=1.5,
    upper_depth=0.25,
    top_thickness=0.05,
    post_width=0.06,
    frame_height=0.08,
    support_inset=0.04,
    wood_color=(0.48, 0.27, 0.12, 1.0),
    frame_color=(0.075, 0.085, 0.095, 1.0),
):
    """Build and add a two-tier bar; return a name -> Cuboid dictionary.

    The local origin is on the floor at the footprint centre. X runs along
    the bar, +Y points toward customers/rear, and +Z points up. ``pose`` is
    an optional SE3 placement of that local frame in the shared scene.
    Heights specify TOP surfaces, not slab centres. The lower slab spans
    the full footprint; its unobstructed front depth is depth - upper_depth
    (0.75 m by default). The upper slab occupies the rear 0.25 m.

    Only two corner posts rise above the lower top, both inside the rear
    section. Four legs and a shallow under-top frame leave the space below
    open. Flat brown RGBA simulates wood without meshes or textures.
    Robot reach and collision clearance must be checked after placement.
    """
    dimensions = (
        length, depth, lower_height, upper_height, upper_depth,
        top_thickness, post_width, frame_height,
    )
    if any(not isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("All dimensions must be finite and positive.")
    if not isfinite(support_inset) or support_inset < 0:
        raise ValueError("support_inset must be finite and nonnegative.")
    if upper_depth >= depth:
        raise ValueError("upper_depth must leave a clear front work area.")
    if upper_height - top_thickness <= lower_height:
        raise ValueError("The upper slab underside must be above the lower top.")
    if lower_height <= top_thickness + frame_height:
        raise ValueError("The lower slab and frame must leave room for legs.")
    if 2 * (support_inset + post_width) >= min(length, depth):
        raise ValueError("Supports must leave space between opposite legs.")
    if support_inset + post_width > upper_depth:
        raise ValueError("Rear supports must fit beneath the raised section.")

    base = SE3() if pose is None else pose
    if not isinstance(base, SE3) or len(base) != 1:
        raise TypeError("pose must be a single SE3 transform.")

    # Each entry is (name, xyz size, xyz centre) in the bar's local frame.
    # Cuboid poses are at their centres, so subtract half the slab thickness.
    specs = [
        ("lower_worktop", (length, depth, top_thickness),
         (0, 0, lower_height - top_thickness / 2)),
        ("upper_worktop", (length, upper_depth, top_thickness),
         (0, (depth - upper_depth) / 2, upper_height - top_thickness / 2)),
    ]

    x_post = length / 2 - support_inset - post_width / 2
    y_post = depth / 2 - support_inset - post_width / 2
    underside = lower_height - top_thickness
    frame_z = underside - frame_height / 2
    riser_height = upper_height - top_thickness - lower_height

    for side, x in (("left", -x_post), ("right", x_post)):
        for end, y in (("front", -y_post), ("rear", y_post)):
            specs.append((f"leg_{side}_{end}",
                          (post_width, post_width, underside),
                          (x, y, underside / 2)))
        specs.append((f"frame_{side}",
                      (post_width, 2 * y_post - post_width, frame_height),
                      (x, 0, frame_z)))
        specs.append((f"upper_support_{side}",
                      (post_width, post_width, riser_height),
                      (x, y_post, lower_height + riser_height / 2)))

    for end, y in (("front", -y_post), ("rear", y_post)):
        specs.append((f"frame_{end}",
                      (2 * x_post - post_width, post_width, frame_height),
                      (0, y, frame_z)))

    parts = {}
    for name, size, centre in specs:
        part = Cuboid(
            scale=list(size),
            pose=base * SE3.Trans(*centre),
            color=wood_color if name.endswith("worktop") else frame_color,
        )
        env.add(part)
        parts[name] = part
    return parts
