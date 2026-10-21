"""Preview every IR Support *part* in a numbered Swift grid.

Choose a part from the Swift sidebar to highlight it and move the camera.
Part sizes are normalised for browsing, NOT shown at true scale.
"""

import math
from pathlib import Path

import numpy as np
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Mesh
from swift import Select, Label, Button
from ir_support_extra_parts.parts import part_names, part_mesh


# ---------------- SETTINGS ----------------
COLUMNS = 8
SPACING = 2.2                 # metres between display cells
MAX_MODEL_SIZE = 1.4          # normalised maximum mesh dimension (metres)
MODEL_BOTTOM_Z = 0.08        # lift above grid surface


def mesh_bounds(model):
    """Return local mesh bounds as a (3, 2) min/max array.

    Newer SpatialGeometry versions provide ``Mesh.bounds()``. Older versions
    do not, so load the source file with trimesh as a compatibility fallback.
    """
    bounds = getattr(model, 'bounds', None)
    if callable(bounds):
        bounds = bounds()
    elif bounds is None:
        filename = getattr(model, 'filename', None)
        if filename is None:
            raise AttributeError('Mesh has neither bounds() nor filename')
        try:
            import trimesh
        except ImportError as error:
            raise RuntimeError(
                'Mesh bounds are unavailable in this SpatialGeometry version; '
                'install trimesh to enable model size normalisation.'
            ) from error
        bounds = trimesh.load(str(filename), force='mesh').bounds

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape == (2, 3):
        bounds = bounds.T
    if bounds.shape != (3, 2):
        raise ValueError(f'Unexpected mesh bounds shape: {bounds.shape}')
    return bounds


# ---------------- NUMBERED FLOOR LABELS ----------------
def create_number_labels(numbered_positions, filename):
    """Create an OBJ mesh and matching material for the floor labels."""
    obj_path = Path(filename)
    mtl_path = obj_path.with_suffix('.mtl')

    # 7-segment digit strokes: a(top), b(upper right), ..., g(middle)
    segments = {
        'a': (0.03, 0.27, 0.12, 0.03),
        'b': (0.15, 0.15, 0.03, 0.12),
        'c': (0.15, 0.03, 0.03, 0.12),
        'd': (0.03, 0.00, 0.12, 0.03),
        'e': (0.00, 0.03, 0.03, 0.12),
        'f': (0.00, 0.15, 0.03, 0.12),
        'g': (0.03, 0.135, 0.12, 0.03),
    }
    digits = ['abcdef', 'bc', 'abged', 'abgcd', 'fgbc',
              'afgcd', 'afgecd', 'abc', 'abcdefg', 'abcdfg']

    with open(obj_path, 'w', encoding='ascii') as f:
        f.write('# Numbered IR Support grid labels\n')
        f.write(f'mtllib {mtl_path.name}\n')
        f.write('usemtl label\n')
        vertex_count = 0
        for number, x, y in numbered_positions:
            for digit_index, digit in enumerate(f'{number:03d}'):
                left = x - 0.31 + digit_index * 0.22
                bottom = y - 0.78
                for segment in digits[int(digit)]:
                    dx, dy, w, h = segments[segment]
                    x0, y0 = left + dx, bottom + dy
                    z = 0.025
                    for vx, vy in [(x0, y0), (x0+w, y0),
                                   (x0+w, y0+h), (x0, y0+h)]:
                        f.write(f'v {vx:.5f} {vy:.5f} {z:.5f}\n')
                    a = vertex_count + 1
                    f.write(f'f {a} {a+1} {a+2}\n')
                    f.write(f'f {a} {a+2} {a+3}\n')
                    vertex_count += 4

    # Swift's OBJ loader also requests the companion MTL file, even when a
    # flat color is supplied to Mesh, so always create it beside the OBJ.
    with open(mtl_path, 'w', encoding='ascii') as f:
        f.write('newmtl label\n')
        f.write('Ka 0.05 0.05 0.05\n')
        f.write('Kd 0.05 0.05 0.05\n')
        f.write('Ks 0.0 0.0 0.0\n')
        f.write('d 1.0\n')
        f.write('illum 1\n')


# ---------------- CREATE ENVIRONMENT ----------------
names = sorted(part_names(), key=str.lower)
rows = math.ceil(len(names) / COLUMNS)

env = swift.Swift()
env.launch(realtime=True, ground_pattern='@grid')

loaded = []                     # (number, name, x, y)
numbered_positions = []

for i, name in enumerate(names):
    number = i + 1
    col, row = i % COLUMNS, i // COLUMNS
    x, y = col * SPACING, -row * SPACING

    try:
        model = part_mesh(name)

        # Make all meshes similarly sized so even tiny glasses are easy to see.
        # Also centre meshes whose built-in origins are far from their geometry.
        try:
            bounds = mesh_bounds(model)
            extents = bounds[:, 1] - bounds[:, 0]
            largest = float(np.max(extents))
            factor = MAX_MODEL_SIZE / largest if largest > 0 else 1.0
            centre_xy = np.mean(bounds[:2], axis=1)
            bottom_z = bounds[2, 0]

            model.scale = (np.asarray(model.scale, dtype=float) * factor).tolist()
            model.T = SE3.Trans(
                x - factor * centre_xy[0],
                y - factor * centre_xy[1],
                MODEL_BOTTOM_Z - factor * bottom_z,
            ).A
        except Exception as error:
            # A model may still render even if its bounding box cannot be read.
            print(f'  Size normalisation unavailable for {name}: {error}')
            model.T = SE3.Trans(x, y, MODEL_BOTTOM_Z).A

        # Square display tile, with the mesh sitting just above it.
        tile = Cuboid([1.9, 1.9, 0.025], color=[0.73, 0.78, 0.82, 1.0])
        tile.T = SE3.Trans(x, y, -0.03).A
        env.add(tile)
        env.add(model)

        loaded.append((number, name, x, y))
        numbered_positions.append((number, x, y))
        print(f'{number:03d}  {name}')

    except Exception as error:
        print(f'{number:03d}  FAILED: {name} - {error}')

# Add number labels as ONE OBJ mesh (much faster than individual text shapes).
label_file = Path(__file__).resolve().with_name('swift_parts_grid_labels.obj')
create_number_labels(numbered_positions, label_file)
labels = Mesh(filename=str(label_file), color=[0.05, 0.05, 0.05, 1.0])
env.add(labels)

if not loaded:
    raise RuntimeError('No models loaded. Check that ir-support-full is installed.')


# ---------------- SELECT / HIGHLIGHT AN OBJECT ----------------
highlight = Cuboid([1.96, 1.96, 0.035], color=[1.0, 0.8, 0.1, 1.0])
env.add(highlight)

selection_label = Label('Selected: none')
env.add(selection_label)


def focus_on(index):
    number, name, x, y = loaded[int(index)]
    highlight.T = SE3.Trans(x, y, -0.017).A
    selection_label.label = f'{number:03d}: {name}'
    env.set_camera_pose(
        position=[x + 2.0, y - 2.4, 2.0],
        look_at=[x, y, 0.6],
    )


def overview(_=None):
    cx = (COLUMNS - 1) * SPACING / 2
    cy = -(rows - 1) * SPACING / 2
    env.set_camera_pose(
        position=[cx + 18, cy - 23, 34],
        look_at=[cx, cy, 0],
    )


options = [f'{number:03d} - {name}' for number, name, _, _ in loaded]
env.add(Select(focus_on, label='Inspect model', options=options, value=0))
env.add(Button(overview, label='Show entire grid'))

overview()
focus_on(0)  # Start by highlighting the first object

overview()  # But show the entire catalogue initially
print(f'\nLoaded {len(loaded)} of {len(names)} part models.')
print('Select a name in the Swift sidebar to zoom to its yellow tile.')
print('Press Ctrl+C in the terminal to close.')

# Keep Swift active so dropdown and button callbacks are processed.
try:
    while True:
        env.hold(0.05)
except KeyboardInterrupt:
    pass
