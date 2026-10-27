"""Parametric 3-D model of the SignAid glasses frame.

Built with Blender's Python API rather than a CAD kernel because CadQuery's
OpenCascade binary would not fit on this disk. Everything is solid geometry
made from primitives and booleans, so it exports as a clean printable mesh.

    blender -b --python hardware/glasses_frame.py

Outputs, next to this file:
    signaid_glasses.stl   for the slicer, and for importing into Onshape
    signaid_glasses.blend the editable source

DESIGN NOTES

Camera is centred above the bridge, not on a temple. Centre-mounting keeps the
view aligned with the wearer's own, avoids the parallax of an off-axis camera,
and balances the weight so the frame does not rotate on the nose.

It is tilted DOWN by CAMERA_TILT. Signing happens below eye level, and a
camera pointed straight ahead loses the hands at exactly the moment they matter.
This is the same problem that makes the phone worse than the glasses: the hand
has to be large and centred in frame for MediaPipe to resolve finger detail.
15-20 degrees puts natural signing space in the middle of the image.

The Pi is NOT on the frame. A Pi 4 is 85x56mm and about 46g before its power
supply; hung off a face-worn frame it is both uncomfortable and front-heavy.
The cable leaves the camera pod, runs along a channel in the left temple, and
exits behind the ear to a Pi in a pocket.

MEASURE THESE BEFORE PRINTING
Every dimension below is a named constant. The ones that must match your actual
hardware are in the CAMERA block — a pocket that is 2mm too small means the
board does not go in, and 2mm too large means it rattles. Measure the stripped
PCB with calipers and edit those three numbers.
"""
import math
import os
import sys

import bpy
import bmesh

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- face fit
# Proportions follow the chunky acetate wayfarer shape that Ray-Ban Meta uses,
# rather than the thin wire-frame look of the first version. Thicker rims are
# not only closer to the reference, they give somewhere to hide the camera and
# the cable, and they print far better than 4mm walls.
FRAME_WIDTH = 142.0        # hinge to hinge, a common adult size
LENS_WIDTH = 48.0          # opening width, per eye
LENS_HEIGHT = 40.0         # opening height
BRIDGE_WIDTH = 18.0        # gap between the openings, over the nose
RIM = 7.0                  # chunky acetate rim, not a wire frame
FRAME_DEPTH = 9.0          # front-to-back thickness of the frame front
BROW_HEIGHT = 6.0          # extra material along the top edge
LENS_RAKE = 6.0            # bottom edge pulled in, giving the wayfarer taper
BEVEL = 1.6                # edge rounding; acetate frames have no sharp edges

# --------------------------------------------------------------- temples
TEMPLE_LENGTH = 132.0
TEMPLE_THICK = 6.5         # chunky, and thick enough to bury the cable
TEMPLE_HEIGHT = 12.0
TEMPLE_TAPER = 0.6         # tip thickness as a fraction of the root
CABLE_CHANNEL = 3.2        # square channel for the USB cable, left temple

# ----------------------------------------------------------------- CAMERA
# *** MEASURE YOUR STRIPPED WEBCAM BOARD AND EDIT THESE THREE ***
# Defaults are a typical bare USB webcam PCB. A pocket 0.4mm larger than the
# board on each side is the usual allowance for FDM printing tolerance.
CAM_BOARD_W = 32.0
CAM_BOARD_H = 32.0
CAM_BOARD_T = 8.0          # including the tallest component on the back
CAM_LENS_DIA = 11.0        # lens barrel diameter
CAM_FIT = 0.4              # clearance per side
CAM_WALL = 2.0             # pod wall thickness
CAMERA_TILT = 18.0         # degrees downward, toward the signing space


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.objects):
        for item in list(block):
            block.remove(item)


def box(name, size, location=(0, 0, 0), rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    ob = bpy.context.object
    ob.name = name
    # primitive_cube_add(size=1) already spans one unit, so the scale IS the
    # dimension. Halving it here made every part half-size while the positions
    # still assumed full size, and the booleans cut in the wrong places.
    ob.scale = (size[0], size[1], size[2])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return ob


def cyl(name, radius, depth, location=(0, 0, 0), rotation=(0, 0, 0), verts=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, vertices=verts,
                                        location=location, rotation=rotation)
    ob = bpy.context.object
    ob.name = name
    return ob


def round_edges(ob, width, segments=3):
    """Bevel every edge. Acetate frames have no sharp corners, and neither do
    printed parts that survive being sat on."""
    mod = ob.modifiers.new(name='bevel', type='BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return ob


def boolean(target, tool, op='DIFFERENCE'):
    mod = target.modifiers.new(name='bool', type='BOOLEAN')
    mod.operation = op
    mod.object = tool
    mod.solver = 'EXACT'
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool, do_unlink=True)
    return target


def join(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.object


def build_front():
    """The frame front: a brow bar with two eye openings cut out of it."""
    total_h = LENS_HEIGHT + 2 * RIM + BROW_HEIGHT
    front = box('front', (FRAME_WIDTH, FRAME_DEPTH, total_h),
                location=(0, 0, 0))

    # Two openings, each raked so the bottom edge tucks inward — the taper that
    # makes a wayfarer read as a wayfarer instead of as a rectangle.
    offset_x = (LENS_WIDTH + BRIDGE_WIDTH) / 2
    z = -BROW_HEIGHT / 2
    for side in (-1, 1):
        hole = box(f'hole{side}', (LENS_WIDTH, FRAME_DEPTH * 3, LENS_HEIGHT),
                   location=(side * offset_x, 0, z))
        round_edges(hole, 3.0, segments=4)
        front = boolean(front, hole)
        # The wayfarer's raked lower-outer corner is deliberately NOT cut here.
        # A wedge big enough to read as a rake also reached through the bottom
        # rim and left the openings open at the base. The bevelled corners
        # already give the softened acetate look; the taper is not worth an
        # unclosed rim.

    # Nose notch: a semicircular relief so the bridge sits on the nose rather
    # than pinching it.
    notch = cyl('notch', radius=9.0, depth=FRAME_DEPTH * 3,
                location=(0, 0, z - LENS_HEIGHT / 2),
                rotation=(math.radians(90), 0, 0))
    front = boolean(front, notch)
    round_edges(front, BEVEL, segments=3)
    return front


def build_camera_pod():
    """Camera housing in the front-left corner, flush with the frame face.

    Moved off the bridge. A pod perched above the nose is the thing that makes a
    build look like a science project, and Ray-Ban Meta puts its cameras in the
    outer corners of the front for exactly that reason. The cost is a little
    parallax versus the wearer's own view, which does not matter for tracking a
    hand a metre away.
    """
    pw = CAM_BOARD_W + 2 * (CAM_FIT + CAM_WALL)
    ph = CAM_BOARD_H + 2 * (CAM_FIT + CAM_WALL)
    pt = CAM_BOARD_T + CAM_FIT + CAM_WALL

    total_h = LENS_HEIGHT + 2 * RIM + BROW_HEIGHT
    # Sit it in the outer corner, overlapping the rim so it reads as part of
    # the frame rather than something bolted on.
    x = -(FRAME_WIDTH / 2 - pw / 2)
    z = total_h / 2 - ph / 2
    tilt = math.radians(-CAMERA_TILT)

    pod = box('pod', (pw, pt, ph), location=(x, -pt / 2 + FRAME_DEPTH / 2, z),
              rotation=(tilt, 0, 0))
    round_edges(pod, BEVEL, segments=3)

    pocket = box('pocket',
                 (CAM_BOARD_W + 2 * CAM_FIT, CAM_BOARD_T + CAM_FIT,
                  CAM_BOARD_H + 2 * CAM_FIT),
                 location=(x, -pt / 2 + FRAME_DEPTH / 2 - CAM_WALL / 2, z),
                 rotation=(tilt, 0, 0))
    pod = boolean(pod, pocket)

    aperture = cyl('aperture', radius=CAM_LENS_DIA / 2 + CAM_FIT,
                   depth=pt * 4, location=(x, -pt / 2 + FRAME_DEPTH / 2, z),
                   rotation=(math.radians(90) + tilt, 0, 0))
    pod = boolean(pod, aperture)

    exit_hole = box('cable_exit', (CABLE_CHANNEL, pt * 3, CABLE_CHANNEL),
                    location=(x - pw / 2 + CABLE_CHANNEL,
                              -pt / 2 + FRAME_DEPTH / 2, z - ph / 2 + CABLE_CHANNEL))
    pod = boolean(pod, exit_hole)
    return pod


def build_temple(side):
    """One arm: thick at the hinge, tapering toward the ear, hooked at the tip."""
    x = side * (FRAME_WIDTH / 2 - TEMPLE_THICK / 2)
    arm = box(f'temple{side}', (TEMPLE_THICK, TEMPLE_LENGTH, TEMPLE_HEIGHT),
              location=(x, -TEMPLE_LENGTH / 2, 0))

    # Taper: shave a wedge off the outer face so the tip is thinner than the
    # root, which is what every real frame does and what stops the arms looking
    # like two rectangular sticks.
    shave = TEMPLE_THICK * (1 - TEMPLE_TAPER)
    wedge = box(f'taper{side}', (shave * 3, TEMPLE_LENGTH * 1.2, TEMPLE_HEIGHT * 3),
                location=(x + side * (TEMPLE_THICK / 2 + shave * 1.5 - shave * 0.2),
                          -TEMPLE_LENGTH / 2, 0),
                rotation=(0, math.radians(side * 1.4), 0))
    arm = boolean(arm, wedge)

    hook_len = 32.0
    ang = math.radians(38)
    hook = box(f'hook{side}', (TEMPLE_THICK * TEMPLE_TAPER, hook_len,
                               TEMPLE_HEIGHT * 0.85),
               location=(x, -TEMPLE_LENGTH + math.cos(ang) * hook_len / 2,
                         -math.sin(ang) * hook_len / 2),
               rotation=(-ang, 0, 0))
    arm = boolean(arm, hook, op='UNION')
    round_edges(arm, BEVEL, segments=3)

    if side < 0:
        chan = box('chan', (CABLE_CHANNEL, TEMPLE_LENGTH * 1.2, CABLE_CHANNEL),
                   location=(x - TEMPLE_THICK / 2 + CABLE_CHANNEL / 2,
                             -TEMPLE_LENGTH / 2, 0))
        arm = boolean(arm, chan)
    return arm


def main():
    clear()
    parts = [build_front(), build_camera_pod(),
             build_temple(-1), build_temple(1)]
    frame = join(parts)
    frame.name = 'SignAid_Glasses'

    # Merge coincident vertices so the export is manifold and the slicer is
    # not left guessing at the boolean seams.
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(frame.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.update_edit_mesh(frame.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    stl = os.path.join(HERE, 'signaid_glasses.stl')
    bpy.ops.wm.stl_export(filepath=stl, export_selected_objects=False)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'signaid_glasses.blend'))

    dims = frame.dimensions
    print(f'\nBUILT  {dims.x:.1f} x {dims.y:.1f} x {dims.z:.1f} mm '
          f'({len(frame.data.polygons)} faces)')
    print(f'STL -> {stl}')


if __name__ == '__main__':
    main()
