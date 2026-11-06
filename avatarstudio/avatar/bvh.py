"""BVH motion export (Blender, MotionBuilder, most 3D tools) of the avatar rig.

BVH is right-handed, Y up, in centimetres; the rig is Unity's left-handed space in metres, so Z is flipped
(the character faces -Z, Blender's default BVH forward axis)."""
import math

import numpy as np

from avatarstudio.avatar.rig import IDENTITY, REST, mat_from_q

FLIP = np.diag([1.0, 1.0, -1.0])


def euler_zxy(m):
    """Angles (z, x, y) in degrees with m = Rz @ Rx @ Ry, BVH channel order 'Zrotation Xrotation Yrotation'."""
    x = math.asin(float(np.clip(m[2, 1], -1, 1)))
    if abs(m[2, 1]) < 0.9999:
        y = math.atan2(-m[2, 0], m[2, 2])
        z = math.atan2(-m[0, 1], m[1, 1])
    else:  # gimbal lock
        y, z = 0.0, math.atan2(m[1, 0], m[0, 0])
    return [math.degrees(v) for v in (z, x, y)]


def write_bvh(path, frames, fps):
    """frames: list of {"bones": {name: quaternion}, "root": xyz} from AvatarRig."""
    children = {name: [] for name in REST}
    for name, (parent, _) in REST.items():
        if parent:
            children[parent].append(name)
    lines = ["HIERARCHY"]

    def joint(name, depth):
        pad = "  " * depth
        off = np.array(REST[name][1]) * 100 * np.array([1, 1, -1])
        lines.append(f"{pad}{'ROOT' if depth == 0 else 'JOINT'} {name}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
        if depth == 0:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for c in children[name]:
            joint(c, depth + 1)
        if not children[name]:
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET 0.0000 2.0000 0.0000")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    joint("Hips", 0)
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1.0 / fps:.6f}"]
    order = []  # channel order in MOTION lines = depth-first order of the hierarchy

    def walk(n):
        order.append(n)
        for c in children[n]:
            walk(c)

    walk("Hips")
    for fr in frames:
        vals = []
        root = (np.array(REST["Hips"][1]) + np.asarray(fr["root"])) * 100 * np.array([1, 1, -1])
        vals += list(root)
        for name in order:
            m = FLIP @ mat_from_q(fr["bones"].get(name, IDENTITY)) @ FLIP
            vals += euler_zxy(m)
        lines.append(" ".join(f"{v:.4f}" for v in vals))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return order
