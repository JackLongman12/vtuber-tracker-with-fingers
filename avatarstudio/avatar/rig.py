"""Turn tracker output into avatar data for a VRM / Unity humanoid:
local bone rotations, root position and blendshape (face shape) values.

Coordinate system: Unity's (left-handed, metres, y up). The avatar stands at the origin in a T-pose
facing +Z, toward the camera, so its right hand is +X. Every bone rotation is local (relative to its parent)
and identity = T-pose, which is what VRM's normalized humanoid bones and the VMC protocol expect.
MediaPipe world coordinates (x right in the image, y down, z away from the camera) map to this space as
(x, y, z) -> (-x, -y, -z).
"""
import json
import math
import os

import numpy as np

from avatarstudio.audio.mic import fuse
from avatarstudio.tracking.filters import FilterBank
from avatarstudio.core.paths import user_file

# ---------------------------------------------------------------- quaternion helpers (x, y, z, w)
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def q_from_mat(m):
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = np.asarray(m, float).tolist()  # plain floats: faster
    t = m00 + m11 + m22
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return np.array(q) / n


def mat_from_q(q):
    x, y, z, w = q.tolist() if isinstance(q, np.ndarray) else q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def q_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.append(axis * math.sin(angle / 2), math.cos(angle / 2))


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def twist_about(q, axis):
    """The part of rotation q that turns around `axis` (swing-twist decomposition)."""
    v = np.dot(q[:3], axis) * np.asarray(axis, float)
    tw = np.append(v, q[3])
    n = np.linalg.norm(tw)
    return tw / n if n > 1e-9 else IDENTITY.copy()


def q_slerp(a, b, t):
    """Spherical interpolation from quaternion a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    d = min(d, 1.0)
    if d > 0.9995:
        q = a + t * (b - a)
        return q / math.sqrt(float(q @ q))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def q_slerp_identity(q, t):
    """Rotation `t` of the way from identity to q."""
    q = q if q[3] >= 0 else -q
    angle = 2 * math.acos(min(q[3], 1.0))
    if angle < 1e-6:
        return IDENTITY.copy()
    return q_axis_angle(q[:3], angle * t)


def rot_x(a): return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
def rot_y(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
def rot_z(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _cross(a, b):
    """Cross product of two 3-vectors. np.cross is general (any shape) and ~10x slower on single vectors, which
    added up to most of the rig's time (it runs thousands of times a frame in the joint solvers)."""
    a0, a1, a2 = a
    b0, b1, b2 = b
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _norm(v):
    n = math.sqrt(float(v @ v))
    return v / n if n > 1e-9 else v


def _clamp(x, lo, hi):
    """float(np.clip(x, lo, hi)) for a single number, without numpy's per-call overhead."""
    return float(lo if x < lo else hi if x > hi else x)


def basis(primary, secondary):
    """Orthonormal frame (as matrix columns) whose first axis is `primary`, second as close to `secondary`."""
    a = _norm(np.asarray(primary, dtype=float))
    b = _norm(secondary - a * np.dot(secondary, a))
    return np.column_stack([a, b, _cross(a, b)])


def to_unity(p):
    return -np.asarray(p, dtype=float)


# ---------------------------------------------------------------- rest skeleton (T-pose, metres)
# bone -> (parent, offset from parent). Used by the 3D viewer, the BVH exporter and the FK self-test.
REST = {"Hips": (None, (0, 0.95, 0)), "Spine": ("Hips", (0, 0.1, 0)), "Chest": ("Spine", (0, 0.15, 0)),
        "Neck": ("Chest", (0, 0.2, 0)), "Head": ("Neck", (0, 0.1, 0))}
for S, sx in (("Left", -1), ("Right", 1)):
    REST.update({
        f"{S}Shoulder": ("Chest", (sx * 0.04, 0.17, 0)), f"{S}UpperArm": (f"{S}Shoulder", (sx * 0.14, 0, 0)),
        f"{S}LowerArm": (f"{S}UpperArm", (sx * 0.27, 0, 0)),
        f"{S}Hand": (f"{S}LowerArm", (sx * 0.25, 0, 0)),
        f"{S}UpperLeg": ("Hips", (sx * 0.09, -0.05, 0)), f"{S}LowerLeg": (f"{S}UpperLeg", (0, -0.42, 0)),
        f"{S}Foot": (f"{S}LowerLeg", (0, -0.42, 0)),
        f"{S}Eye": ("Head", (sx * 0.03, 0.06, 0.08)),
    })
    for finger, base, seg in (("Thumb", (sx * 0.025, -0.01, 0.03), (sx * 0.025, 0, 0.02)),
                              ("Index", (sx * 0.09, 0, 0.03), (sx * 0.035, 0, 0)),
                              ("Middle", (sx * 0.095, 0, 0.01), (sx * 0.04, 0, 0)),
                              ("Ring", (sx * 0.09, 0, -0.01), (sx * 0.035, 0, 0)),
                              ("Little", (sx * 0.08, 0, -0.03), (sx * 0.028, 0, 0))):
        REST[f"{S}{finger}Proximal"] = (f"{S}Hand", base)
        REST[f"{S}{finger}Intermediate"] = (f"{S}{finger}Proximal", seg)
        REST[f"{S}{finger}Distal"] = (f"{S}{finger}Intermediate", seg)

# rest directions / bend-plane normals in avatar space (see module docstring)
ARM_DIR = {"Left": np.array([-1.0, 0, 0]), "Right": np.array([1.0, 0, 0])}
ARM_BEND = {"Left": np.array([0, 1.0, 0]), "Right": np.array([0, -1.0, 0])}  # elbow bends forward (+Z)
LEG_DIR, LEG_BEND = np.array([0, -1.0, 0]), np.array([1.0, 0, 0])           # knee bends backward (-Z)
PALM_DOWN = np.array([0, -1.0, 0])
POSE_IDS = {"Left": dict(s=11, e=13, w=15, h=23, k=25, a=27), "Right": dict(s=12, e=14, w=16, h=24, k=26, a=28)}
FINGER_IDS = {"Thumb": [1, 2, 3, 4], "Index": [5, 6, 7, 8], "Middle": [9, 10, 11, 12],
              "Ring": [13, 14, 15, 16], "Little": [17, 18, 19, 20]}
ARMS_DOWN = math.radians(70)  # resting arm angle below horizontal when arms aren't visible


SMOOTHING_PATH = user_file("smoothing.json")  # from tune_smoothing.py
# One Euro (min_cutoff Hz, beta) per bone group, unless tuned from a recording
# (fingers 1.5 Hz with little speed boost: with the stretched finger range (MAX_GAIN), measured on a 38 s take,
# as few single-frame finger glitches as the old solver and less shake)
SMOOTH_DEFAULTS = {"body": (1.5, 1.0), "hands": (1.5, 1.0), "fingers": (1.5, 0.3), "eyes": (3.0, 1.0)}


def smooth_group(bone):
    if "Eye" in bone:
        return "eyes"
    if any(f in bone for f in FINGER_IDS):
        return "fingers"
    return "hands" if bone.endswith("Hand") else "body"


def load_smoothing(path=SMOOTHING_PATH):
    params = dict(SMOOTH_DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path) as f:
                for g, p in json.load(f)["groups"].items():
                    params[g] = (float(p["min_cutoff"]), float(p["beta"]))
        except (OSError, ValueError, KeyError):
            pass
    return params


HAND_HOLD_S = 0.5  # keep a lost hand's last pose this long
OCCLUDED_HOLD_S = 10.0  # ...but this long when it's hidden near the head / the face is hidden behind a hand
TORSO_LEN = 0.5  # hips to shoulders (m): how much a forward lean lowers the shoulders
HAND_EASE_S = 0.2  # a hand that (re)appears eases in over this long
HINGE_FINGERS = True  # fingers solved as hinges (see _hinge_finger); False = free 3D joints (the old way)
LOSS_HOLD_S, LOSS_FADE_S = 0.3, 1.0  # lost head/torso/arm: hold this long, then ease to rest over this long
HEAD_SWITCH_S = 0.25  # head handed between the face mesh and the body model: crossfade this long
# the body model's head frame (ears + nose) vs the face mesh's: a fixed ~11 deg apart (measured on take.mp4), used
# until the face has been seen; after that the offset of the last face frame is used (see AvatarRig._head)
BODY_HEAD_OFFSET = (-0.0484, 0.063, 0.0503, 0.9956)
# finger joint limits in degrees: (min flexion, max flexion, max sideways spread); flexion > 0 curls toward the palm
FINGER_LIMITS = {"Proximal": (-25, 95, 25), "Intermediate": (-5, 110, 8), "Distal": (-10, 90, 8)}


def _rot(k, c, s):
    """Rotation matrix about unit axis k with cos / sin of the angle (Rodrigues: c I + s [k]x + (1 - c) k k^T)."""
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    t = 1.0 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _arc(a, b):
    """Shortest rotation (matrix) taking unit vector a onto unit vector b."""
    axis = _cross(a, b)
    s, c = math.sqrt(float(axis @ axis)), min(max(float(a @ b), -1.0), 1.0)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # opposite directions: half a turn about any axis at right angles to a
        p = _cross(a, np.array([1.0, 0, 0]) if abs(float(a[0])) < 0.9 else np.array([0, 1.0, 0]))
        return _rot(p / math.sqrt(float(p @ p)), -1.0, 0.0)
    return _rot(axis / s, c, s)


def limit_finger_joint(m, rest_dir, seg):
    """Clamp one finger joint's local rotation to human limits (no bending backwards or sideways too far),
    which keeps fingers believable when the tracker guesses badly, e.g. when fingers overlap or interlock.
    Works in the parent bone's frame: rest direction = rest_dir (along the finger), palm = -Y, side = Z."""
    lo, hi, spread_max = FINGER_LIMITS[seg]
    d = m @ rest_dir
    along = float(np.dot(d, rest_dir))
    flex = math.degrees(math.atan2(-d[1], along))
    spread = math.degrees(math.atan2(d[2], math.hypot(along, d[1])))  # angle out of the curl plane
    if lo <= flex <= hi and abs(spread) <= spread_max:
        return m
    f, s = math.radians(np.clip(flex, lo, hi)), math.radians(np.clip(spread, -spread_max, spread_max))
    target = rest_dir * math.cos(f) * math.cos(s) + np.array([0, -1.0, 0]) * math.sin(f) * math.cos(s) \
        + np.array([0, 0, 1.0]) * math.sin(s)
    return _arc(d, _norm(target)) @ m


# wrist range of motion (degrees): bend toward the palm (+) / back (-), sideways toward the thumb (+) / little
# finger (-), twist relative to the forearm. On a recorded take the solved wrist went past these in 30-40% of
# hand frames (sideways up to 138 deg), mostly where the body and hand trackers disagree about the forearm.
WRIST_LIMITS = {"bend": (-70.0, 80.0), "side": (-35.0, 25.0), "twist": 90.0}


def limit_wrist(q, S):
    """Keep a hand's local rotation (relative to the forearm) inside WRIST_LIMITS: the pointing direction is
    pulled back onto the edge of the allowed (elliptical) range, the twist is clamped."""
    along, side_ax = ARM_DIR[S], np.array([0, 0, 1.0])
    tw = twist_about(q, along)
    swing = q_mul(q, np.array([-tw[0], -tw[1], -tw[2], tw[3]]))
    d = mat_from_q(swing) @ along
    a, p, s = float(d @ along), float(d @ PALM_DOWN), float(d @ side_ax)
    bend, side = math.degrees(math.atan2(p, a)), math.degrees(math.atan2(s, a))
    (b_lo, b_hi), (s_lo, s_hi) = WRIST_LIMITS["bend"], WRIST_LIMITS["side"]
    e = (bend / (b_hi if bend > 0 else -b_lo)) ** 2 + (side / (s_hi if side > 0 else -s_lo)) ** 2
    if e > 1:
        k = 1 / math.sqrt(e)
        bend, side = bend * k, side * k
        d2 = _norm(along + PALM_DOWN * math.tan(math.radians(bend)) + side_ax * math.tan(math.radians(side)))
        swing = q_from_mat(_arc(d, d2) @ mat_from_q(swing))
    ang = 2 * math.atan2(float(np.dot(tw[:3], along)), float(tw[3]))
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    lim = math.radians(WRIST_LIMITS["twist"])
    if abs(ang) > lim:
        ang = math.copysign(lim, ang)
        tw = np.append(along * math.sin(ang / 2), math.cos(ang / 2))
    return q_mul(swing, tw)


def _axis_angle(k, a):
    return _rot(k, math.cos(a), math.sin(a))


FINGERS4 = ("Index", "Middle", "Ring", "Little")
SEGS = ("Proximal", "Intermediate", "Distal")
FLEX_FULL = {"Proximal": 85.0, "Intermediate": 100.0, "Distal": 70.0}  # degrees of a tight fist
# what the tracker reads (degrees) for an open hand and a fist, per finger and joint (medians over a recorded
# 38 s take; the H calibration replaces them with yours). The tracker under-reads fists badly (knuckles ~45 deg
# instead of ~90, curled fingers hide each other), so the avatar's fist stayed half open. Readings are stretched
# from this range onto 0..FLEX_FULL, with the gain capped (MAX_GAIN) so tracking noise isn't blown up.
HAND_RANGE_DEFAULT = {
    "Index": {"Proximal": (29.0, 41.0), "Intermediate": (28.6, 55.6), "Distal": (6.0, 38.1)},
    "Middle": {"Proximal": (33.0, 41.4), "Intermediate": (26.2, 69.3), "Distal": (11.1, 35.6)},
    "Ring": {"Proximal": (40.4, 45.7), "Intermediate": (19.3, 61.9), "Distal": (13.4, 30.8)},
    "Little": {"Proximal": (38.6, 48.4), "Intermediate": (14.2, 41.6), "Distal": (12.0, 47.6)}}
MAX_GAIN = {"Proximal": 2.2, "Intermediate": 2.0, "Distal": 1.8}  # higher let noise through as glitches
# the same for WiLoR's finger points (wilor_client.py), which read open hands straighter and fists more curled
HAND_RANGE_WILOR = {
    "Index": {"Proximal": (15.7, 24.8), "Intermediate": (20.2, 66.2), "Distal": (0.9, 30.5)},
    "Middle": {"Proximal": (27.8, 34.3), "Intermediate": (15.9, 69.0), "Distal": (0.9, 29.0)},
    "Ring": {"Proximal": (25.0, 28.1), "Intermediate": (21.1, 72.8), "Distal": (1.1, 30.3)},
    "Little": {"Proximal": (24.8, 34.8), "Intermediate": (16.8, 61.3), "Distal": (-1.9, 31.8)}}
THUMB_TUCKED = (45.0, 40.0, 35.0, 25.0)  # swing, tilt, two hinges of a thumb folded over curled fingers
# thumb (degrees): base joint swing in the palm plane (+ = toward the fingers) and tilt toward the palm,
# then two hinges
THUMB_LIMITS = {"swing": (-30.0, 55.0), "tilt": (-20.0, 65.0), "Intermediate": (-10.0, 60.0), "Distal": (-15.0, 80.0)}


def _flat(x, fallback):
    """x with its palm-normal component removed (direction in the palm plane), or fallback if x is along it."""
    f = x - PALM_DOWN * float(x @ PALM_DOWN)
    return _norm(f) if np.linalg.norm(f) > 1e-3 else fallback


def _hinge_angle(u, w, h):
    """Signed bend from direction u to w about the hinge axis h (toward the palm = positive)."""
    return math.degrees(math.atan2(float(_cross(u, w) @ h), float(u @ w)))


def finger_angles(v, along, side):
    """Raw angles (degrees) of one finger from its 3 segment directions in the hand's rest frame:
    (knuckle bend, knuckle spread toward the thumb side, middle-joint bend, tip-joint bend).
    The hinge axis is fixed per finger (from the knuckle's sideways direction), so it doesn't flip when the
    finger curls past 90 degrees into a fist."""
    palm = PALM_DOWN
    a, p, s = float(v[0] @ along), float(v[0] @ palm), float(v[0] @ side)
    sp = math.atan2(s, a) if math.hypot(a, s) > 1e-3 else 0.0
    h = _norm(_cross(_flat(v[0], along), palm))
    return (math.degrees(math.atan2(p, math.hypot(a, s))), math.degrees(sp),
            _hinge_angle(v[0], v[1], h), _hinge_angle(v[1], v[2], h))


def _finger_bones(S, finger, f0, sp, f1, f2, along, side):
    """Local rotations of a finger from its angles (degrees): knuckle bend + spread, two pure hinges, no twist."""
    palm = PALM_DOWN
    f0, sp, f1, f2 = (math.radians(x) for x in (f0, sp, f1, f2))
    spread_dir = along * math.cos(sp) + side * math.sin(sp)
    h = _norm(_cross(spread_dir, palm))
    g0 = _axis_angle(h, f0) @ _arc(along, spread_dir)
    g1 = _axis_angle(h, f1) @ g0
    g2 = _axis_angle(h, f1 + f2) @ g0
    return {f"{S}{finger}Proximal": q_from_mat(g0), f"{S}{finger}Intermediate": q_from_mat(g0.T @ g1),
            f"{S}{finger}Distal": q_from_mat(g1.T @ g2)}


def _swing_axis(rest_dir, along):
    """Axis (up or down) about which a positive swing turns the thumb's rest direction toward the fingers."""
    up = np.array([0, 1.0, 0])
    return up if float(_cross(rest_dir, along) @ up) > 0 else -up


# The thumb is turned about its own axis compared with the fingers (its nail faces sideways), so its two outer
# joints bend across the palm, not straight down into it. Measured on WiLoR's hands of a recorded take: the real
# bend axis is turned 62 deg (middle half 54-69 deg, 1109 bent thumbs) from "bend toward the palm".
THUMB_PRONATION = 62.0


def _thumb_hinge(d0, S):
    """Hinge axis of the thumb's two outer joints for the thumb base direction d0 (hand rest frame)."""
    hv = _norm(_cross(d0, PALM_DOWN))
    a = math.radians(-THUMB_PRONATION if S == "Right" else THUMB_PRONATION)
    return _norm(hv * math.cos(a) + _cross(hv, d0) * math.sin(a))


def thumb_angles(v, rest_dir, along, S="Left"):
    """Raw thumb angles (degrees): base swing in the palm plane toward the fingers, base tilt toward the palm,
    then the two hinge bends. v: 3 segment directions in the hand's rest frame."""
    palm = PALM_DOWN
    r = _flat(rest_dir, rest_dir)
    d = _flat(v[0], r)
    k = _swing_axis(rest_dir, along)
    swing = math.degrees(math.atan2(float(_cross(r, d) @ k), float(r @ d)))
    # tilt toward the palm relative to the thumb's rest direction (an avatar's rest thumb may already point down)
    tilt = math.degrees(math.asin(float(np.clip(v[0] @ palm, -1, 1))) -
                        math.asin(float(np.clip(_norm(rest_dir) @ palm, -1, 1))))
    h = _thumb_hinge(v[0], S)
    return swing, tilt, _hinge_angle(v[0], v[1], h), _hinge_angle(v[1], v[2], h)


def _thumb_bones(S, swing, tilt, f1, f2, rest_dir):
    palm = PALM_DOWN
    g_swing = _axis_angle(_swing_axis(rest_dir, ARM_DIR[S]), math.radians(swing))
    d = g_swing @ _flat(rest_dir, rest_dir)
    g0 = _axis_angle(_norm(_cross(d, palm)), math.radians(tilt)) @ g_swing
    h = _thumb_hinge(g0 @ rest_dir, S)
    g1 = _axis_angle(h, math.radians(f1)) @ g0
    g2 = _axis_angle(h, math.radians(f1 + f2)) @ g0
    return {f"{S}ThumbProximal": q_from_mat(g0), f"{S}ThumbIntermediate": q_from_mat(g0.T @ g1),
            f"{S}ThumbDistal": q_from_mat(g1.T @ g2)}


def _hand_frame(S, hw):
    """The hand's rotation from its 21 points (unity axes): wrist -> middle knuckle and the palm normal."""
    n = _cross(hw[5] - hw[0], hw[17] - hw[0]) if S == "Left" else _cross(hw[17] - hw[0], hw[5] - hw[0])
    return basis(hw[9] - hw[0], n) @ basis(ARM_DIR[S], PALM_DOWN).T


def raw_finger_flex(hand):
    """{finger: (knuckle, middle, tip) bend in degrees} as the tracker reads it (for the K hand calibration);
    from WiLoR's points when the hand has them."""
    S = hand["side"].capitalize()
    hw = to_unity(hand.get("_world_fingers", hand["_world"]))
    hg = _hand_frame(S, hw)
    out = {}
    for f in FINGERS4:
        ids = FINGER_IDS[f]
        f0, _, f1, f2 = finger_angles([_norm(hg.T @ (hw[ids[k + 1]] - hw[ids[k]])) for k in range(3)],
                                      ARM_DIR[S], np.array([0, 0, 1.0]))
        out[f] = (f0, f1, f2)
    return out


def hand_range_from(open_samples, fist_samples, defaults=None):
    """Calibration result from raw_finger_flex samples of open hands and fists: medians per finger and joint.
    A joint whose fist reading isn't clearly above its open reading keeps the default."""
    defaults = defaults or HAND_RANGE_DEFAULT
    rng = {}
    for f in FINGERS4:
        rng[f] = {}
        for j, seg in enumerate(SEGS):
            o = float(np.median([s[f][j] for s in open_samples]))
            c = float(np.median([s[f][j] for s in fist_samples]))
            rng[f][seg] = (round(o, 1), round(c, 1)) if c - o > 10 else defaults[f][seg]
    return rng


def _finger_set(S, seg_dirs, along, side, hand_range=None, thumb_out=None, rest=None):
    """All five digits as anatomical joints. Solving each joint as a free 3D rotation turned tracking noise into
    twist flips and joint-limit jumps; here fingers are knuckle bend + spread and two hinges, the thumb a
    swinging/tilting base and two hinges, with:
    - your bend range stretched toward the full anatomical range (fists close, open hands open)
    - knuckles following the middle joints when a finger curls (real fingers flex together; the tracker
      loses the knuckle bend in a fist)
    - the tip joint following 2/3 of the middle joint (as in real fingers; halves its noise)
    - neighbouring fingers not allowed to cross sideways through each other
    - the thumb folded over the fingers when the ring/little fingers are curled and its tip is close to the
      index knuckle (fist, peace...): the tracker draws a hidden thumb straight. thumb_out = distance of the
      thumb tip from the index knuckle / wrist-to-index-knuckle distance (large = thumbs up)."""
    rng = hand_range or HAND_RANGE_DEFAULT
    ang = {}
    for f in FINGERS4:
        f0, sp, f1, f2 = finger_angles(seg_dirs[f], along, side)
        flex = []
        for x, seg in zip((f0, f1, f2), SEGS):
            lo, hi = rng.get(f, HAND_RANGE_DEFAULT[f]).get(seg, HAND_RANGE_DEFAULT[f][seg])
            gain = min(FLEX_FULL[seg] / max(hi - lo, 1e-3), MAX_GAIN[seg])
            flex.append((x - lo) * gain)
        f0, f1, f2 = flex
        # curling: the knuckle bends with the finger (fades in smoothly from a 30 to 70 deg middle-joint bend)
        f0 = max(f0, 0.75 * f1 * _clamp((f1 - 30) / 40, 0, 1))
        f0, f1, f2 = (_clamp(x, FINGER_LIMITS[seg][0], FINGER_LIMITS[seg][1]) for x, seg in zip((f0, f1, f2), SEGS))
        f2 = 0.5 * f2 + 0.5 * (FLEX_FULL["Distal"] / FLEX_FULL["Intermediate"]) * f1
        ang[f] = [f0, _clamp(sp, -FINGER_LIMITS["Proximal"][2], FINGER_LIMITS["Proximal"][2]), f1, f2]
    for a, b in zip(FINGERS4, FINGERS4[1:]):  # index spread >= middle >= ring >= little (toward the thumb side)
        if ang[a][1] < ang[b][1] - 2:
            m = (ang[a][1] + ang[b][1]) / 2
            ang[a][1], ang[b][1] = m + 1, m - 1
    out = {}
    for f in FINGERS4:
        out.update(_finger_bones(S, f, *ang[f], along, side))
    rest_dir = _norm(np.array((rest or REST)[f"{S}ThumbIntermediate"][1], dtype=float))  # the avatar's rest thumb
    t = list(thumb_angles(seg_dirs["Thumb"], rest_dir, along, S))
    if thumb_out is not None:
        curled = (ang["Ring"][2] + ang["Little"][2]) / 2
        tuck = _clamp((curled - 50) / 40, 0, 1) * _clamp((0.75 - thumb_out) / 0.25, 0, 1)
        t = [a + (b - a) * tuck for a, b in zip(t, THUMB_TUCKED)]
    lim = THUMB_LIMITS
    out.update(_thumb_bones(S, *(_clamp(x, *lim[k]) for x, k in zip(t, ("swing", "tilt", "Intermediate",
                                                                                  "Distal"))), rest_dir))
    return out


def limb_rotations(rest_dir, rest_bend, a, b, c, fallback_bend):
    """Global rotations of a 2-bone limb (a->b->c) from its T-pose direction and bend-plane normal."""
    u, f = _norm(b - a), _norm(c - b)
    n = _cross(u, f)
    if np.linalg.norm(n) < 0.05:  # nearly straight: the bend plane is undefined, keep a sensible twist
        n = fallback_bend
    rest = basis(rest_dir, rest_bend)
    return basis(u, n) @ rest.T, basis(f, n) @ rest.T


def mirror_name(name):
    for a, b in (("Left", "Right"), ("_L", "_R")):
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return name


# ---------------------------------------------------------------- face shapes
VOWELS = ("A", "I", "U", "E", "O")
DEFAULT_MOUTH = {"open": 0.02, "width": 0.5}  # typical relaxed closed mouth, as fractions of eye-corner distance
# smoothed lightly: speech and blinks are fast
FAST_SHAPES = ("jaw", "mouth", "cheekPuff", "A", "I", "U", "E", "O", "eyeBlink", "Blink")


def mouth_features(pts, bs):
    """Mouth shape measured on the 3D face mesh, scaled by the distance between the outer eye corners
    (so it doesn't change with distance or head turn): lip opening, mouth width, lip rounding."""
    iod = np.linalg.norm(pts[33] - pts[263]) + 1e-6
    return {"open": float(np.linalg.norm(pts[13] - pts[14]) / iod),     # inner lips
            "width": float(np.linalg.norm(pts[61] - pts[291]) / iod),   # mouth corners
            "pucker": float(bs.get("mouthPucker", 0.0)), "funnel": float(bs.get("mouthFunnel", 0.0))}


class MouthModel:
    """Mouth vowels (VRM A/I/U/E/O) from mouth geometry.
    With a personal vowel calibration (you say each vowel once) it matches your mouth to your own five vowel
    shapes plus your closed mouth; without it, it uses general rules for opening / width / rounding."""

    def __init__(self, neutral=None, vowels=None):
        self.neutral = dict(neutral or DEFAULT_MOUTH)
        self.set_vowels(vowels)

    def set_vowels(self, vowels):
        """Each vowel becomes a path from your closed mouth (0) to your recorded vowel shape (1)."""
        self.protos = None
        if vowels and all(v in vowels for v in VOWELS):
            # your relaxed mouth, including its resting lip rounding: MediaPipe reports 0.3-0.45 pucker on some
            # relaxed mouths, and assuming 0 made a closed mouth read as a third of the way to "U"
            rest = self._vec({"pucker": 0.0, "funnel": 0.0} | self.neutral)
            self.protos = {v: self._vec(vowels[v]) - rest for v in VOWELS}
            self.rest = rest
            gaps = [self._seg(self.protos[a], self.protos[b])[1] ** 2 for a in VOWELS for b in VOWELS if a != b]
            self.sigma2 = max(min(gaps) * 0.3, 0.02)  # softness: below the closest pair of vowel shapes

    def adapt_rest(self, feat, dt, tau=4.0):
        """Slowly re-learn your resting lip rounding while you're silent with your lips together.
        MediaPipe's reading of a relaxed mouth drifts between sessions (0.33, 0.44, 0.60 pucker for the same
        person), and a stale value makes a closed mouth look like "U"."""
        a = 1 - math.exp(-dt / tau)
        for k in ("pucker", "funnel"):
            self.neutral[k] = self.neutral.get(k, 0.0) + (feat[k] - self.neutral.get(k, 0.0)) * a
        if self.protos:
            self.rest = self._vec({"pucker": 0.0, "funnel": 0.0} | self.neutral)

    def lips_apart(self, feat):
        """0 = lips together (at your resting gap), 1 = clearly apart."""
        return float(np.clip((feat["open"] - self.neutral["open"] - 0.01) / 0.04, 0, 1))

    @staticmethod
    def _seg(x, p):
        """How far along the path to p the point x is (0..1), and its distance from that path."""
        t = float(np.clip(np.dot(x, p) / (np.dot(p, p) + 1e-9), 0, 1))
        return t, float(np.linalg.norm(x - t * p))

    def _vec(self, m):
        """Feature vector, each axis scaled to a similar range."""
        return np.array([(m["open"] - self.neutral["open"]) * 4,
                         (m["width"] / self.neutral["width"] - 1) * 4,
                         m.get("pucker", 0.0), m.get("funnel", 0.0)])

    def __call__(self, feat):
        clip = lambda v: float(np.clip(v, 0, 1)) + 0.0
        if self.protos:
            x = self._vec(feat) - self.rest
            seg = {v: self._seg(x, p) for v, p in self.protos.items()}
            w = np.exp(-np.array([seg[v][1] ** 2 for v in VOWELS]) / self.sigma2)
            w = w / (w.sum() + 1e-9)
            return {v: float(wi * seg[v][0]) for v, wi in zip(VOWELS, w)}
        feat = {**feat, "pucker": max(feat["pucker"] - self.neutral.get("pucker", 0.0), 0.0),  # beyond your resting
                "funnel": max(feat["funnel"] - self.neutral.get("funnel", 0.0), 0.0)}  # lip rounding
        # how open: 0 closed .. 1 wide open. 0.30 of the eye-corner distance counts as a full "A": normal speech
        # rarely opens wider (a live session with 0.45 never got past A = 0.4)
        o = clip((feat["open"] - self.neutral["open"]) / 0.30)
        o_small = clip((feat["open"] - self.neutral["open"]) / 0.04)      # lips just apart
        wd = feat["width"] / self.neutral["width"] - 1                     # >0 wider (I/E), <0 narrower (O/U)
        rnd = clip(max(feat["pucker"], feat["funnel"]) * 1.5 + clip(-wd * 3))
        wide = clip(wd * 4)
        return {"A": clip(o * (1 - rnd) * (1 - wide * 0.5) * 1.3), "I": clip(wide * (1 - o * 2) * o_small),
                "U": clip(rnd * (1 - o * 2) * o_small), "E": clip(wide * min(o * 3, 1) * (1 - o)),
                "O": clip(o * rnd * 1.5)}


class AutoRange:
    """Rescales each expression so that your own strongest version of it reaches 1.0.
    Starts with no boost (remembered peak = 1.0); the peak then fades (half-life ~1 min) toward the strongest
    you actually make, so the gain grows only as it learns your range, and is capped at 1/floor (2x).
    Blinks (own calibrated mapping) and gaze directions are left alone."""

    SKIP = ("eyeBlink", "eyeLook")

    def __init__(self, floor=0.5, half_life=60.0):
        self.floor, self.half_life, self.peak, self.t = floor, half_life, {}, None

    def __call__(self, shapes, t):
        decay = 0.5 ** ((t - self.t) / self.half_life) if self.t is not None else 1.0
        self.t = t
        out = {}
        for k, v in shapes.items():
            if k.startswith(self.SKIP):
                out[k] = v
                continue
            self.peak[k] = max(self.peak.get(k, 1.0) * decay, v)
            out[k] = min(v / max(self.peak[k], self.floor), 1.0)
        return out


class BlinkRange:
    """Per-eye blink mapping that learns your own open-eye level and blink strength.
    MediaPipe rarely reports a closed eye as 1.0 and the level differs per person, glasses and lighting:
    in one recorded session typical blinks peaked at only 0.36, so fixed thresholds missed most of them.
    Output: 0 below 20% of the way from your open level to your recent blink peak, 1 above 75%."""

    def __init__(self, half_life=20.0):
        self.half_life, self.base, self.peak, self.t = half_life, None, 0.6, None

    def __call__(self, v, t):
        if self.base is None:
            self.base, self.t = min(v, 0.3), t
        dt = max(t - self.t, 0.0)
        self.t = t
        span = self.peak - self.base
        if v < self.base + 0.3 * span:  # eye open: follow its resting level slowly (~3 s)
            self.base += (v - self.base) * (1 - math.exp(-dt / 3.0))
        self.peak = max(self.peak * 0.5 ** (dt / self.half_life), v, self.base + 0.2)
        span = self.peak - self.base
        return float(np.clip((v - self.base - 0.2 * span) / (0.55 * span), 0, 1))


BODY_FOLLOW = 0.2  # share of the head's turn / tilt the upper body follows when the shoulders aren't visible


def link_blinks(left, right, yaw=0.0, closed=0.6, open_=0.25):
    """Both eyes blink together (their average) unless you clearly wink: one eye at least `closed` while the other
    is at most `open_`. The face tracker reads the two eyes separately, so a normal blink often came out
    lopsided (one eye half shut) - VSeeFace calls this "link eye blinking". With the head turned (yaw, degrees,
    > 0 = to your right) the eye farther from the camera is seen edge-on and reads as closing: on a recorded take,
    turned > 20 deg right, the far eye was the "closed" one in 83% of wink-like frames. So from 10 to 25 deg of
    turn the far eye follows the near one more and more."""
    w = _clamp((abs(yaw) - 10.0) / 15.0, 0, 1)
    if w > 0:
        if yaw > 0:
            left += (right - left) * w
        else:
            right += (left - right) * w
    if max(left, right) >= closed and min(left, right) <= open_:
        return left, right
    m = (left + right) / 2
    return m, m


def vrm_shapes(bs, visemes):
    """VRM 0.x preset clips: blinks, look up/down and mouth vowels (expressions come from hotkeys / gestures).
    bs must already hold the adapted eyeBlink values (see AvatarRig)."""
    g = lambda k: bs.get(k, 0.0)
    out = {"Blink_L": g("eyeBlinkLeft"), "Blink_R": g("eyeBlinkRight"), **visemes}
    out["Blink"] = min(out["Blink_L"], out["Blink_R"])
    out["LookUp"] = _clamp((g("eyeLookUpLeft") + g("eyeLookUpRight")) / 2 * 1.5, 0, 1)
    out["LookDown"] = _clamp((g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2 * 1.5, 0, 1)
    return out


# ---------------------------------------------------------------- the rig
class AvatarRig:
    def __init__(self, mirror=False, root_motion=True, smoothing=True, legs=False,
                 expressiveness=1.0, auto_range=True, idle=True, predict_ms=0.0, mic_weight=0.5, smooth_scale=1.0):
        """idle: breathing, automatic blinks and a gentle head drift when tracking is lost.
        predict_ms: extrapolate bones this far ahead to hide latency. mic_weight: how much the voice decides
        the vowel while you speak. smooth_scale: < 1 smoother, > 1 snappier."""
        self.mirror, self.root_motion, self.smoothing = mirror, root_motion, smoothing
        self.legs, self.expressiveness, self.idle, self.mic_weight = legs, expressiveness, idle, mic_weight
        self.predict_s, self.smooth_scale = predict_ms / 1000, smooth_scale
        self._last_pose, self._last_shapes, self._next_blink, self._t_prev = {}, None, None, None
        self.auto_range = AutoRange() if auto_range else None
        self.blink = {"eyeBlinkLeft": BlinkRange(), "eyeBlinkRight": BlinkRange()}
        self.face_refiner = None  # params from face_refiner.py, set by the studio
        self.mouth = MouthModel()
        self._last_hand = {}  # side -> (time, {bone: quaternion}) to bridge short tracking dropouts
        self._calib = None
        self.f = FilterBank(min_cutoff=1.5, beta=0.8)
        self.smooth_params = load_smoothing()
        self._hand_in, self._hand_out = {}, {}
        self._last_hand_px, self._good_arkit, self._good_visemes = {}, {}, None
        self._shrug_base, self._shrug_out = {}, {}
        self.no_clip, self.body_scale = True, 1.0  # keep_arm_out_of_body
        self.contacts = True  # pinch_contact / two_hand_contact
        self.face_contacts = True  # face_contact: a hand on your face rests on the avatar's face (_face_touch)
        self._touch_face, self._touch, self._touch_s, self._touch_t = None, {}, {}, None
        self.rest = REST      # bone layout: the avatar's own proportions once the 3D viewer sends them
        self._prev_q = {}
        self.bad_frames = 0   # frames where a bone came out as NaN / inf and its last good value was used
        self.link_blinks = True        # both eyes blink together unless you clearly wink (link_blinks())
        self.body_follows_head = True  # shoulders out of view: the upper body turns a little with the head
        self.predict_body_s = 0.0      # look-ahead for torso, head and arms only (fingers and face stay as tracked)
        self._head_offset = None        # body-model head frame -> face frame, as of the last face frame (_head)
        self._head_src, self._head_switch, self._head_last = None, None, None

    def set_proportions(self, offsets):
        """The avatar's own rest skeleton: {bone: offset from its parent in metres, rig axes}. Bones the avatar
        doesn't have keep the default. Used for contacts, no-clipping and the thumb's rest direction."""
        rest = {}
        for name, (parent, off) in REST.items():
            o = offsets.get(name)
            rest[name] = (parent, tuple(float(x) for x in o) if o is not None and np.linalg.norm(o) > 1e-4 else off)
        self.rest = rest

    # -------- calibration: neutral head / torso / face, standing spot, personal vowels
    @property
    def calib(self):
        return self._calib

    @calib.setter
    def calib(self, cal):
        self._calib = cal
        cal = cal or {}
        neutral = dict(cal["mouth"]) if "mouth" in cal else None
        if neutral is not None:  # older calibration files: take resting lip rounding from the face baseline
            for k, b in (("pucker", "mouthPucker"), ("funnel", "mouthFunnel")):
                neutral.setdefault(k, cal.get("blend", {}).get(b, 0.0))
        self.mouth = MouthModel(neutral, cal.get("vowels"))

    def mouth_now(self, res):
        """Current mouth measurements (for the vowel calibration)."""
        face = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
        return mouth_features(face["_pts"], face.get("blendshapes", {})) if face else None

    def set_vowels(self, samples, voice=None):
        """samples: {vowel: [mouth_features, ...]} recorded while saying each vowel.
        voice: {vowel: [(F1, F2), ...]} formants heard from the microphone at the same time."""
        vowels = {v: {k: float(np.median([s[k] for s in ss])) for k in ss[0]} for v, ss in samples.items() if ss}
        cal = {**(self.calib or {}), "vowels": vowels}
        if voice and all(len(voice.get(v, [])) >= 3 for v in VOWELS):
            cal["voice"] = {v: np.median(np.array(voice[v]), axis=0).tolist() for v in VOWELS}
        self.calib = cal
        return vowels

    @staticmethod
    def _shoulder_py(body):
        """Image height (px) of the shoulder line in the main camera's picture, or None."""
        if body is None or "_cam" in body or body["_vis"][11] < 0.5 or body["_vis"][12] < 0.5:
            return None
        return float((body["_pts"][11, 1] + body["_pts"][12, 1]) / 2)

    def neutral_from(self, res, f_px):
        """Neutral pose values from one frame (the caller averages a few frames)."""
        out = {}
        sy = self._shoulder_py(res["body"])
        if sy is not None:
            out["shoulder_py"] = sy
        face = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
        if face:
            out["head"] = self._head_matrix(face)
            out["blend"] = dict(face.get("blendshapes", {}))
            out["head_px"] = face["box"][0] + face["box"][2] / 2
            out["mouth"] = mouth_features(face["_pts"], face.get("blendshapes", {}))
            out["head_angles"] = [face["head_pose"][k] for k in ("yaw", "pitch", "roll")]
        chest = self._chest_matrix(res["body"]) if res["body"] else None
        if chest is not None:
            out["chest"] = chest
        if res["depth"]:
            out["distance_cm"] = res["depth"]["distance_cm"]
        return out

    def set_calibration(self, samples):
        """Average several neutral_from() results into the calibration."""
        cal = {}
        for key in ("head", "chest"):
            ms = [s[key] for s in samples if key in s]
            if ms:
                u, _, vt = np.linalg.svd(np.mean(ms, 0))  # closest rotation to the mean
                cal[key] = (u @ vt).tolist()
        for key in ("distance_cm", "head_px", "shoulder_py"):
            vs = [s[key] for s in samples if key in s]
            if vs:
                cal[key] = float(np.median(vs))
        bl = [s["blend"] for s in samples if "blend" in s]
        if bl:
            cal["blend"] = {k: float(np.median([b.get(k, 0) for b in bl])) for k in bl[0]}
        ha = [s["head_angles"] for s in samples if "head_angles" in s]
        if ha:
            cal["head_angles"] = np.median(ha, axis=0).tolist()
        mo = [s["mouth"] for s in samples if "mouth" in s]
        if mo:
            cal["mouth"] = {k: float(np.median([m[k] for m in mo])) for k in ("open", "width", "pucker", "funnel")}
        # keep the other calibrations across neutral re-calibrations (pressing C used to drop the K hand range)
        for keep in ("vowels", "voice", "gaze", "hand_range", "hand_range_wilor", "cameras"):
            if self.calib and keep in self.calib:
                cal[keep] = self.calib[keep]
        self.calib = cal
        return cal

    # -------- pieces
    @staticmethod
    def _head_matrix(face):
        pts = face["_pts"]
        if face.get("_R") is not None:  # a side camera's face (multicam.py): turned into the main camera's axes
            pts = pts @ np.asarray(face["_R"]).T
        p = to_unity(pts)
        right = p[33] - p[263]   # person's right eye corner minus left: avatar +X
        up = p[10] - p[152]      # forehead minus chin
        return basis(right, up)

    @staticmethod
    def _body_head_matrix(body):
        """Head frame from the body model's 3D points: right ear minus left, nose in front of the ears. The body model
        still sees a head turned 90 deg or looking down, where the face mesh is lost. On take.mp4 it was ~9 deg off
        the face mesh's frame (after a fixed offset) and shook more (0.9 vs 0.6 deg per frame), so it only steps in
        while the face is gone. None if the body model doesn't see the head."""
        if body is None or body["_vis"][[0, 7, 8]].max() < 0.5:
            return None
        w = to_unity(body["_world"])
        right = w[8] - w[7]
        return basis(right, _cross(w[0] - (w[7] + w[8]) / 2, right))

    def _head(self, face, body, face_hidden, t):
        """The head's frame (before calibration), or None while it can't be known (then held, then eased to rest).
        - face seen: the face mesh;
        - face lost but not behind a hand (turned far away, looking down): the last face frame, turned on by however
          much the body model's head has turned since - nothing jumps, and a turn past 90 deg is followed (it used
          to hold 0.3 s and then ease back to facing front);
        - face behind a hand: None (held, as before - on take.mp4 holding ended up as close to the returning face as
          following the body model, whose head points are under the hand too).
        Switching between the face mesh and the body model crossfades over HEAD_SWITCH_S."""
        mb = self._body_head_matrix(body)
        if face is not None:
            m, src = self._head_matrix(face), "face"
            if mb is not None:  # (lightly smoothed: the body model's frame shakes more than the face mesh's)
                off = q_from_mat(mb.T @ m)
                self._head_offset = off if self._head_offset is None else q_slerp(self._head_offset, off, 0.3)
        elif mb is not None and not face_hidden:
            # (a full turn reads 10-20 deg short: the face mesh already under-reads just before it's lost. Making the
            # body model's turn since then bigger didn't help: x1.6 moved the far ends 6 deg, and the head was
            # further off when the face came back)
            m, src = mb @ mat_from_q(self._head_offset if self._head_offset is not None else BODY_HEAD_OFFSET), "body"
        else:
            self._head_src = self._head_switch = None
            return None
        q = q_from_mat(m)
        if self._head_src is not None and src != self._head_src:
            self._head_switch = (t, self._head_last)
        self._head_src = src
        if self._head_switch is not None:
            t0, q0 = self._head_switch
            f = (t - t0) / HEAD_SWITCH_S
            if f >= 1 or f < 0:
                self._head_switch = None
            else:
                q = q_slerp(q0, q, f)
        self._head_last = q
        return mat_from_q(q)

    @staticmethod
    def _chest_matrix(body, lean_deg=0.0):
        """lean_deg: forward lean (toward the camera) to use when the hips are hidden (seated)."""
        w, vis = to_unity(body["_world"]), body["_vis"]
        if vis[11] < 0.5 or vis[12] < 0.5:
            return None
        right = w[12] - w[11]  # person's right shoulder minus left
        if vis[23] > 0.5 and vis[24] > 0.5:
            up = (w[11] + w[12]) / 2 - (w[23] + w[24]) / 2
        else:  # hips hidden (sitting at a desk): upright, turned / tilted by the shoulders, leaned by lean_deg
            a = math.radians(lean_deg)
            up = np.array([0, math.cos(a), math.sin(a)])
        return basis(right, up)

    def _seated_lean(self, body, res):
        """Seated (hips hidden): moving toward / away from the camera is a lean from the hips, not a walk.
        Degrees toward the camera, from the distance change since calibration over the torso length."""
        if body is None or (body["_vis"][23] > 0.5 and body["_vis"][24] > 0.5):
            return 0.0, False
        if not (self.calib and res["depth"] and "distance_cm" in self.calib):
            return 0.0, True
        closer = (self.calib["distance_cm"] - res["depth"]["distance_cm"]) / 100
        return math.degrees(math.asin(float(np.clip(closer / TORSO_LEN, -0.6, 0.8)))), True

    def _relative(self, key, m):
        if self.calib and key in self.calib:
            return m @ np.array(self.calib[key]).T
        return m

    def _smooth_shapes(self, group, shapes, t):
        """Mouth shapes get light smoothing so fast speech isn't blurred; the rest a bit more."""
        out = {}
        for fast in (True, False):
            names = [k for k in shapes if k.startswith(FAST_SHAPES) == fast]
            if names:
                key = f"{group}:{fast}:" + ",".join(names)
                vals = self.f(key, [shapes[k] for k in names], t,
                              min_cutoff=8.0 if fast else 4.0, beta=4.0 if fast else 2.0)
                out.update({k: _clamp(v, 0, 1) for k, v in zip(names, vals)})
        return out

    def _auto_blink(self, t):
        """1.0 during a 150 ms blink every 2.5-6 s, else 0."""
        if self._next_blink is None or t > self._next_blink + 0.15:
            self._next_blink = t + float(np.random.uniform(2.5, 6.0))
        return 1.0 if self._next_blink <= t <= self._next_blink + 0.15 else 0.0

    @staticmethod
    def _hand_bones(S, hand, la, hand_range=None, rest=None):
        """Hand orientation from the palm, then the fingers and thumb as anatomical joints (_finger_set).
        hand_range: your open-hand / fist bend angles from the H calibration (else HAND_RANGE_DEFAULT)."""
        out = {}
        hw = to_unity(hand["_world"])
        d = hw[9] - hw[0]
        n = _cross(hw[5] - hw[0], hw[17] - hw[0]) if S == "Left" else _cross(hw[17] - hw[0], hw[5] - hw[0])
        hg = basis(d, n) @ basis(ARM_DIR[S], PALM_DOWN).T
        out[f"{S}Hand"] = q_from_mat(la.T @ hg)
        along, side = ARM_DIR[S], np.array([0, 0, 1.0])
        wilor = "_world_fingers" in hand
        if wilor:  # finger shapes from WiLoR, in its own hand frame (wilor_client.py)
            hw = to_unity(hand["_world_fingers"])
            hg = _hand_frame(S, hw)
            hand_range = hand_range or HAND_RANGE_WILOR
        seg_dirs = {f: [_norm(hg.T @ (hw[ids[k + 1]] - hw[ids[k]])) for k in range(3)] for f, ids in FINGER_IDS.items()}
        if HINGE_FINGERS:
            # the tucked-thumb fix makes up for MediaPipe drawing a hidden thumb straight; WiLoR doesn't need it
            thumb_out = None if wilor else float(np.linalg.norm(hw[4] - hw[5]) / (np.linalg.norm(hw[0] - hw[5]) or 1.0))
            out.update(_finger_set(S, seg_dirs, along, side, hand_range, thumb_out, rest))
            return out
        for finger, ids in FINGER_IDS.items():
            rest_dir = _norm(np.array(REST[f"{S}{finger}Intermediate"][1], dtype=float))
            rest_up, obs_up = ((PALM_DOWN, hg @ PALM_DOWN) if finger == "Thumb"
                               else (np.array([0, 0, 1.0]), hg @ np.array([0, 0, 1.0])))
            rest = basis(rest_dir, rest_up)
            parent = hg
            for k, seg in enumerate(("Proximal", "Intermediate", "Distal")):
                gk = basis(hw[ids[k + 1]] - hw[ids[k]], obs_up) @ rest.T
                lk = parent.T @ gk
                if finger != "Thumb":
                    lk = limit_finger_joint(lk, ARM_DIR[S], seg)
                out[f"{S}{finger}{seg}"] = q_from_mat(lk)
                parent = parent @ lk
        return out

    # -------- occlusion: what the camera can't see is kept instead of guessed
    def _occlusion(self, res, face, hands):
        """Which parts are hidden by a hand right now.
        -> (covered face regions {"mouth", "eyes"}, face hidden by a hand, {side: hand/arm out of sight near the
        head}). A hand over the mouth makes MediaPipe read nonsense mouth shapes; a face fully behind a hand, or a
        hand behind the head, would otherwise fade to neutral / drop to rest after LOSS_HOLD_S."""
        covered = set()
        # (picture positions only compare within the main camera: items from a side camera carry "_cam")
        if face is not None and "_cam" not in face:
            self._last_face_box = face["box"]
            fp = face["_pts"][:, :2]
            for region, ids in (("mouth", (61, 291, 0, 17)), ("eyes", (33, 133, 159, 145, 362, 263, 386, 374,
                                                                             70, 300))):
                lo, hi = fp[list(ids)].min(0), fp[list(ids)].max(0)
                pad = (hi - lo) * 0.1 + 2
                for h in hands.values():  # at least 3 hand points over that part of the face (not just nearby)
                    if "_cam" in h:
                        continue
                    p = h["_pts"][:, :2]
                    if np.sum(np.all((p > lo - pad) & (p < hi + pad), axis=1)) >= 3:
                        covered.add(region)
        box = getattr(self, "_last_face_box", None)
        body = res.get("body")
        near = {}
        for S in ("Left", "Right"):
            h = hands.get(S)
            if h is not None and "_cam" not in h:
                self._last_hand_px[S] = h["_pts"][:, :2].mean(0)
            elif h is None and body is not None and "_cam" not in body:  # the body model keeps estimating a
                # hidden wrist: a dropped hand moves away
                self._last_hand_px[S] = body["_pts"][POSE_IDS[S]["w"], :2]
            p = self._last_hand_px.get(S)
            near[S] = bool(box is not None and p is not None and
                           np.linalg.norm(p - [box[0] + box[2] / 2, box[1] + box[3] / 2]) < 1.3 * max(box[2], box[3]))
        face_hidden = face is None and any(near.values())
        behind = {S: near[S] and hands.get(S) is None for S in ("Left", "Right")}
        return covered, face_hidden, behind

    def _freeze_covered(self, arkit, visemes, covered):
        """Shapes of a face region covered by a hand keep their last uncovered value."""
        region_of = lambda k: ("mouth" if k.startswith(("mouth", "jaw", "cheek", "tongue")) else
                               "eyes" if k.startswith(("eye", "brow")) else None)
        for k, v in list(arkit.items()):
            if region_of(k) in covered and k in self._good_arkit:
                arkit[k] = self._good_arkit[k]
            elif region_of(k) not in covered:
                self._good_arkit[k] = v
        if "mouth" in covered and self._good_visemes is not None:
            visemes = dict(self._good_visemes)
        elif "mouth" not in covered:
            self._good_visemes = dict(visemes)
        return arkit, visemes

    def _face_touch(self, local, face, hands, face_hidden, behind, t):
        """Hands on the face (face_contact / solve_face_contact), eased in time: a touch's strength rises over
        ~0.1 s and falls over ~0.15 s (the touched spots are kept for that), and a hand that drops out while on the
        face (common: it covers itself) keeps its touch - HAND_HOLD_S, or as long as it's out of sight near the head
        (behind, see _occlusion). Switching per frame made the avatar's hand jump 15-20 cm on and off the face."""
        if face is not None:
            self._touch_face = face
        ref = face if face is not None else self._touch_face if face_hidden else None  # (hidden: the face seen last)
        cal_head = np.array(self.calib["head"]) if self.calib and "head" in self.calib else np.eye(3)
        dt = max(t - self._touch_t, 0.0) if self._touch_t is not None else None
        self._touch_t = t
        for S in ("Left", "Right"):
            hand = hands.get(S)
            pairs, c = face_contact(ref, hand) if ref is not None else ([], 0.0)
            s, last = self._touch_s.get(S, 0.0), self._touch.get(S)
            if pairs:
                self._touch[S] = last = (pairs, ref, t)
            elif hand is None and last is not None and (t - last[2] < HAND_HOLD_S or
                                                        behind[S] and t - last[2] < OCCLUDED_HOLD_S):
                c = s
            if dt is not None:
                s += (c - s) * (1 - math.exp(-dt / (0.1 if c > s else 0.15)))
            else:
                s = c
            self._touch_s[S] = s
            if s > 0.01 and last is not None:
                pairs, ref_, _ = last
                solve_face_contact(local, S, pairs, s, ref_["_pts"], self._head_matrix(ref_), cal_head, self.rest)

    def _shrug(self, S, body, chest):
        """Shoulder shrug in degrees from how far the shoulder rises toward its ear (ears barely move when you
        nod, unlike the mouth), against your relaxed level - which adapts quickly when the gap grows and only
        slowly when it shrinks, so a held shrug isn't learned as normal. 1 cm dead zone, up to 30 deg."""
        if body is None:
            return self._shrug_out.get(S, 0.0)
        ear, sh = (7, 11) if S == "Left" else (8, 12)
        if body["_vis"][ear] < 0.3 or body["_vis"][sh] < 0.5:
            return self._shrug_out.get(S, 0.0)
        w = to_unity(body["_world"])
        d = float((w[ear] - w[sh]) @ chest[:, 1])
        base = self._shrug_base.get(S, d)
        base += (d - base) * (0.2 if d > base else 0.002)
        self._shrug_base[S] = base
        rise = max(base - d - 0.01, 0.0)
        out = math.degrees(math.asin(min(rise / 0.15, 0.5)))  # ~15 cm collarbone
        self._shrug_out[S] = out
        return out

    # -------- main
    def _hold_or_fade(self, group, names, local, tracked, t, hold_s=LOSS_HOLD_S):
        """While a body part is tracked, remember its bones. When tracking is lost, keep the last pose for
        hold_s (LOSS_HOLD_S, or OCCLUDED_HOLD_S when a hand hides it), then ease over LOSS_FADE_S into the
        default pose already in `local` (no snapping).
        Returns how far the fade has gone (0 = tracked / holding, 1 = fully at rest)."""
        if tracked:
            self._last_pose[group] = (t, {n: local[n] for n in names if n in local})
            return 0.0
        if group not in self._last_pose:
            return 1.0
        t_seen, last = self._last_pose[group]
        f = float(np.clip((t - t_seen - hold_s) / LOSS_FADE_S, 0, 1))
        for n, q in last.items():
            local[n] = q_slerp(q, local.get(n, IDENTITY), f)
        return f

    def __call__(self, res, f_px, t, gesture_shapes=None, mic=None, extra_shapes=None, arkit_extra=None):
        """gesture_shapes / extra_shapes: expression clips to force on (gestures, hotkeys).
        mic: audio.MicLipSync.read() result, blended into the mouth vowels.
        arkit_extra: ARKit shapes from the mouth model (tongueOut, cheekPuff, mouthFunnel)."""
        g = {}       # global rotation matrices
        local = {}   # local quaternions
        body, faces = res["body"], res["faces"]
        face = max(faces, key=lambda fc: fc["box"][2]) if faces else None

        # torso
        lean, seated = self._seated_lean(body, res)
        chest = self._chest_matrix(body, lean) if body else None
        torso_tracked = chest is not None
        chest = self._relative("chest", chest) if chest is not None else np.eye(3)
        if not torso_tracked and face and self.body_follows_head:
            # shoulders out of view (close framing): the upper body turns / tilts a little with the head, as a real
            # one does, instead of standing stiff; the head itself stays exactly where it was tracked
            chest = mat_from_q(q_slerp_identity(q_from_mat(self._relative("head", self._head_matrix(face))),
                                                BODY_FOLLOW))
        hips_right = chest[:, 0]
        if body is not None and body["_vis"][23] > 0.5 and body["_vis"][24] > 0.5:
            w = to_unity(body["_world"])
            hips_right = w[24] - w[23]
        yaw = math.atan2(-hips_right[2], hips_right[0])
        g["Hips"] = rot_y(yaw)
        g["Chest"] = chest
        local["Hips"] = q_from_mat(g["Hips"])
        # the upper body bends over two joints (spine + chest), like a real back, instead of one hinge
        half = q_slerp_identity(q_from_mat(g["Hips"].T @ chest), 0.5)
        local["Spine"], local["Chest"] = half, half.copy()

        hands = {h["side"].capitalize(): h for h in res["hands"]}
        covered, face_hidden, behind = self._occlusion(res, face, hands)

        # head + neck (neck takes 30% of the head's rotation relative to the chest)
        head_m = self._head(face, body, face_hidden, t)
        head = self._relative("head", head_m) if head_m is not None else chest
        head_local = q_from_mat(chest.T @ head)
        neck = q_slerp_identity(head_local, 0.3)
        local["Neck"] = neck
        local["Head"] = q_mul(np.array([-neck[0], -neck[1], -neck[2], neck[3]]), head_local)

        # eyes from gaze. With the G calibration, gaze is in screen units (-1 = left/top edge, +1 = right/bottom
        # edge as you see the screen) and screen edges map to 15 deg / 10 deg of eye turn; without it the raw iris
        # offset is used with a generic scale.
        gaze = None
        if face and face.get("gaze"):
            gaze = self.screen_gaze(face["gaze"])
            if self.calib and "gaze" in self.calib:
                eye = rot_y(math.radians(-gaze[0] * 15)) @ rot_x(math.radians(gaze[1] * 10))
            else:
                eye = rot_y(math.radians(-gaze[0] * 25)) @ rot_x(math.radians(gaze[1] * 15))
            local["LeftEye"] = local["RightEye"] = q_from_mat(eye)
        elif head_m is not None:  # head followed by the body model: eyes look straight ahead
            local["LeftEye"], local["RightEye"] = IDENTITY.copy(), IDENTITY.copy()

        # arms, legs
        arm_tracked = {}
        for S in ("Left", "Right"):
            j = POSE_IDS[S]
            vis = body["_vis"] if body else np.zeros(33)
            fallback_bend = chest @ ARM_BEND[S]
            arm_tracked[S] = bool(vis[j["s"]] > 0.5 and vis[j["e"]] > 0.5 and vis[j["w"]] > 0.5)
            if arm_tracked[S]:
                w = to_unity(body["_world"])
                ua, la = limb_rotations(ARM_DIR[S], ARM_BEND[S], w[j["s"]], w[j["e"]], w[j["w"]], fallback_bend)
            else:  # not visible: arms relaxed down by the sides
                ua = chest @ rot_z(ARMS_DOWN if S == "Left" else -ARMS_DOWN)
                la = ua
            g[f"{S}UpperArm"], g[f"{S}LowerArm"] = ua, la
            # shoulder: lifts with the arm once it rises above 20 deg from horizontal (35% of the extra rise),
            # like a real shoulder blade, or by your measured shrug; the upper arm is then expressed relative to
            # the lifted shoulder
            elevation = math.degrees(math.asin(float(np.clip(np.dot(ua @ ARM_DIR[S], chest[:, 1]), -1, 1))))
            # measured shrug (a raised arm lifts the shoulder too, so it's the larger of the two, not the sum)
            lift = math.radians(max(max(elevation - 20, 0) * 0.35, self._shrug(S, body, chest)))
            sh = rot_z(-lift if S == "Left" else lift)
            local[f"{S}Shoulder"] = q_from_mat(sh)
            local[f"{S}UpperArm"] = q_from_mat((chest @ sh).T @ ua)
            local[f"{S}LowerArm"] = q_from_mat(ua.T @ la)

            # hand + fingers. If the hand drops out for a moment (common when hands overlap), keep its last
            # pose for HAND_HOLD_S instead of snapping to a relaxed hand.
            hand = hands.get(S)
            relaxed = {f"{S}Hand": IDENTITY.copy()}  # follow the forearm, fingers slightly curled
            for finger in FINGER_IDS:
                bend = math.radians(15 if finger != "Thumb" else 5)
                for seg in ("Proximal", "Intermediate", "Distal"):
                    relaxed[f"{S}{finger}{seg}"] = q_from_mat(rot_z(bend if S == "Left" else -bend))
            if hand is not None:
                cal_key = "hand_range_wilor" if "_world_fingers" in hand else "hand_range"
                hand_local = self._hand_bones(S, hand, la, (self.calib or {}).get(cal_key), self.rest)
                seen_before = S in self._last_hand and t - self._last_hand[S][0] < 0.15
                if not seen_before:  # (re)appeared: ease in from whatever the avatar showed, don't snap
                    self._hand_in[S] = (t, dict(self._hand_out.get(S, relaxed)))
                self._last_hand[S] = (t, hand_local)
            elif S in self._last_hand and t - self._last_hand[S][0] < (OCCLUDED_HOLD_S if behind[S] else HAND_HOLD_S):
                hand_local = self._last_hand[S][1]
            else:  # lost: ease from the last pose to the relaxed hand over LOSS_FADE_S
                t_seen, last = self._last_hand.get(S, (t - HAND_HOLD_S - LOSS_FADE_S, relaxed))
                f = float(np.clip((t - t_seen - (OCCLUDED_HOLD_S if behind[S] else HAND_HOLD_S)) / LOSS_FADE_S, 0, 1))
                hand_local = {k: q_slerp(last[k], relaxed[k], f) for k in relaxed}
            if S in self._hand_in:
                t_in, start = self._hand_in[S]
                f = float(np.clip((t - t_in) / HAND_EASE_S, 0, 1))
                hand_local = {k: q_slerp(start.get(k, q), q, f) for k, q in hand_local.items()}
                if f >= 1:
                    del self._hand_in[S]
            if f"{S}Hand" in hand_local:
                hand_local[f"{S}Hand"] = limit_wrist(hand_local[f"{S}Hand"], S)
            self._hand_out[S] = hand_local
            local.update(hand_local)
            # forearm twist: VRM has no twist bone, so a turned wrist twists only at the wrist and the skin there
            # pinches ("candy wrapper"). Hand half of the twist around the forearm axis to the forearm instead;
            # the hand ends up in exactly the same place.
            tw = twist_about(local[f"{S}Hand"], ARM_DIR[S])
            half = q_slerp_identity(tw, 0.5)
            local[f"{S}LowerArm"] = q_mul(local[f"{S}LowerArm"], half)
            local[f"{S}Hand"] = q_mul(np.array([-half[0], -half[1], -half[2], half[3]]), local[f"{S}Hand"])

            if self.legs and vis[j["h"]] > 0.5 and vis[j["k"]] > 0.5 and vis[j["a"]] > 0.5:
                w = to_unity(body["_world"])
                ul, ll = limb_rotations(LEG_DIR, LEG_BEND, w[j["h"]], w[j["k"]], w[j["a"]], g["Hips"] @ LEG_BEND)
                local[f"{S}UpperLeg"] = q_from_mat(g["Hips"].T @ ul)
                local[f"{S}LowerLeg"] = q_from_mat(ul.T @ ll)

        # lost tracking: hold, then ease into rest instead of snapping
        head_fade = self._hold_or_fade("head", ("Neck", "Head", "LeftEye", "RightEye"), local, head_m is not None, t,
                                       OCCLUDED_HOLD_S if face_hidden else LOSS_HOLD_S)
        self._hold_or_fade("torso", ("Hips", "Spine"), local, torso_tracked, t)
        for S in ("Left", "Right"):
            self._hold_or_fade(f"arm{S}", (f"{S}Shoulder", f"{S}UpperArm", f"{S}LowerArm"), local, arm_tracked[S], t,
                               OCCLUDED_HOLD_S if behind[S] else LOSS_HOLD_S)

        # idle motion so the avatar never freezes: breathing, plus a slow head drift while the face is lost
        if self.idle:
            breath = math.sin(2 * math.pi * 0.22 * t)
            local["Spine"] = q_mul(local["Spine"], q_from_mat(rot_x(math.radians(1.2 * breath))))
            if head_fade > 0:
                sway = rot_y(math.radians(3 * math.sin(0.31 * t) * head_fade)) @ \
                    rot_x(math.radians(1.5 * math.sin(0.23 * t + 1) * head_fade))
                local["Head"] = q_mul(local["Head"], q_from_mat(sway))

        if self.no_clip:  # hands / arms never pass through the torso or head
            for S in ("Left", "Right"):
                keep_arm_out_of_body(local, S, self.body_scale, self.rest)
        if self.contacts:  # fingertips you touch together touch on the avatar too
            for S, hand in hands.items():
                finger, c = pinch_contact(hand)
                if finger and c > 0:
                    solve_pinch(local, S, finger, c, self.rest)
            pairs, c = two_hand_contact(hands)
            if pairs and c > 0:
                solve_two_hand(local, pairs, c, self.rest)
        if self.face_contacts:  # a hand on your face rests on the avatar's face at the same spot
            self._face_touch(local, face, hands, face_hidden, behind, t)
        # wrist limits once more on the final pose, in the frame the skin bends in (hand vs. the forearm bone, after
        # the twist split): a held forearm with a live hand could still end up 3-4 deg past them
        for S in ("Left", "Right"):
            if f"{S}Hand" in local:
                local[f"{S}Hand"] = limit_wrist(local[f"{S}Hand"], S)

        # root: walk the avatar with your distance and sideways position
        root = np.zeros(3)
        if self.root_motion and self.calib and res["depth"] and "distance_cm" in self.calib:
            dist = res["depth"]["distance_cm"]
            if not seated:  # standing: closer to the camera = walk toward it (+Z); seated it's a lean (above)
                root[2] = -(dist - self.calib["distance_cm"]) / 100
            if face and "head_px" in self.calib and "_cam" not in face:
                px = face["box"][0] + face["box"][2] / 2
                root[0] = -(px - self.calib["head_px"]) / f_px * dist / 100
            # up / down (sitting up, slouching, crouching): the shoulder line's height vs. calibration, minus the
            # drop that leaning already shows (so a lean isn't counted twice)
            sy = self._shoulder_py(body)
            if sy is not None and "shoulder_py" in self.calib:
                drop = (sy - self.calib["shoulder_py"]) / f_px * dist / 100
                lean = math.acos(float(np.clip(chest[1, 1], -1, 1)))  # chest tilt from upright
                drop -= TORSO_LEN * (1 - math.cos(lean))
                root[1] = -float(np.clip(drop if abs(drop) > 0.015 else 0.0, -0.4, 0.3))

        # face shapes: remove your resting face, stretch each shape to your own range, then the expressiveness gain
        arkit = dict(face.get("blendshapes", {})) if face else {}
        if arkit and self.face_refiner:  # learned from your own guided recording (face_refiner.py): it replaces
            # the generic resting-face subtraction, auto-range and blink adaptation below
            from avatarstudio.personal.face_refiner import apply as refine
            arkit = refine(self.face_refiner, arkit)
        elif self.calib and "blend" in self.calib:
            for k, b in self.calib["blend"].items():
                if k in arkit and b > 0.02 and not k.startswith("eyeBlink"):
                    arkit[k] = float(np.clip((arkit[k] - b) / max(1 - b, 1e-3), 0, 1))
        visemes = {}
        if face:
            feat = mouth_features(face["_pts"], face.get("blendshapes", {}))
            visemes = self.mouth(feat)
            if mic is not None and mic["level"] < 0.05:  # silent: vowels only if the lips are really apart
                apart = self.mouth.lips_apart(feat)
                visemes = {v: x * apart for v, x in visemes.items()}
                if apart < 0.1:
                    self.mouth.adapt_rest(feat, max(t - (self._t_prev or t), 0.0))
        self._t_prev = t
        if self.auto_range and arkit and not self.face_refiner:
            arkit = self.auto_range(arkit, t)
        for eye in ("eyeBlinkLeft", "eyeBlinkRight"):  # learned per-eye blink range (VRM and Perfect Sync)
            if eye in arkit and not self.face_refiner:
                arkit[eye] = self.blink[eye](arkit[eye], t)
        if self.link_blinks and "eyeBlinkLeft" in arkit and "eyeBlinkRight" in arkit:
            arkit["eyeBlinkLeft"], arkit["eyeBlinkRight"] = link_blinks(
                arkit["eyeBlinkLeft"], arkit["eyeBlinkRight"], face["head_pose"]["yaw"] if face else 0.0)
        if mic and (face or mic["level"] > 0):  # voice lip sync, also while the face isn't visible
            visemes = fuse(visemes, mic, self.mic_weight)
        if face and arkit_extra:  # the mouth model sees what MediaPipe can't (tongue) or underreads
            for k, v in arkit_extra.items():
                arkit[k] = max(arkit.get(k, 0.0), v)
        if self.expressiveness != 1.0:
            arkit = {k: min(v * self.expressiveness, 1.0) for k, v in arkit.items()}
            visemes = {k: min(v * self.expressiveness, 1.0) for k, v in visemes.items()}
        if face:
            arkit, visemes = self._freeze_covered(arkit, visemes, covered)
            vrm = vrm_shapes(arkit, visemes)
            if arkit_extra and "tongueOut" in arkit_extra:  # VRM has no tongue preset: a custom clip, if the model has one
                vrm["TongueOut"] = arkit.get("tongueOut", 0.0)
            self._last_shapes = (t, arkit, vrm)
        else:  # face lost: fade the last face to neutral (receivers keep the last value otherwise),
            # keep voice lip sync, and blink by itself
            t_seen, last_arkit, last_vrm = self._last_shapes if self._last_shapes else (t, {}, {})
            keep = 1 - float(np.clip((t - t_seen - (OCCLUDED_HOLD_S if face_hidden else LOSS_HOLD_S)) / LOSS_FADE_S, 0, 1))
            arkit = {k: v * keep for k, v in last_arkit.items()}
            vrm = {k: v * keep for k, v in last_vrm.items()}
            vrm.update({v: max(vrm.get(v, 0.0) * keep, visemes.get(v, 0.0)) for v in VOWELS})
            if self.idle:
                blink = self._auto_blink(t) * (1 - keep)
                for k in ("Blink", "Blink_L", "Blink_R"):
                    vrm[k] = max(vrm.get(k, 0.0), blink)
                for k in ("eyeBlinkLeft", "eyeBlinkRight"):
                    arkit[k] = max(arkit.get(k, 0.0), blink)
        for shapes in (gesture_shapes, extra_shapes):
            for k, v in (shapes or {}).items():
                vrm[k] = max(vrm.get(k, 0.0), v)

        # never pass a broken number on: a NaN from a degenerate measurement (two tracked points on top of each
        # other, ...) would stick in the smoothing filters - and in the VTuber app - for the rest of the session
        if not np.isfinite(np.array(list(local.values()))).all():
            self.bad_frames += 1
            for name, q in local.items():
                if not np.isfinite(q).all() or float(q @ q) < 1e-6:
                    local[name] = self._prev_q.get(name, IDENTITY)
        if not np.isfinite(root).all():
            root = np.zeros(3)
        arkit = {k: v for k, v in arkit.items() if math.isfinite(v)}
        vrm = {k: v for k, v in vrm.items() if math.isfinite(v)}

        # smoothing (quaternions are kept on one hemisphere so they don't flip sign)
        if self.smoothing:
            for name, q in local.items():
                prev = self._prev_q.get(name)
                if prev is not None and np.dot(prev, q) < 0:
                    q = -q
                group = smooth_group(name)
                mc, beta = self.smooth_params[group]
                lead = max(self.predict_s, self.predict_body_s) if group == "body" else self.predict_s
                q = self.f("q:" + name, q, t, min_cutoff=mc * self.smooth_scale, beta=beta, lead=lead)
                local[name] = q = q / np.linalg.norm(q)
                self._prev_q[name] = q
            root = self.f("root", root, t, min_cutoff=1.0 * self.smooth_scale, beta=0.5, lead=self.predict_s)
            arkit = self._smooth_shapes("arkit", arkit, t)
            vrm = self._smooth_shapes("vrm", vrm, t)

        if self.mirror:  # reflect across the avatar's YZ plane and swap left/right
            local = {mirror_name(k): np.array([q[0], -q[1], -q[2], q[3]]) for k, q in local.items()}
            root = root * np.array([-1, 1, 1])
            arkit = {mirror_name(k): v for k, v in arkit.items()}
            vrm = {mirror_name(k): v for k, v in vrm.items()}
        return {"bones": local, "root": root, "arkit": arkit, "vrm": vrm, "gaze": gaze}

    def screen_gaze(self, g):
        """Raw iris offset -> calibrated (x, y); identity when there is no gaze calibration."""
        c = (self.calib or {}).get("gaze")
        if not c:
            return float(g["x"]), float(g["y"])
        return (float(np.clip((g["x"] - c["x0"]) / c["xs"], -1.5, 1.5)),
                float(np.clip((g["y"] - c["y0"]) / c["ys"], -1.5, 1.5)))

    def set_gaze(self, samples):
        """samples: [(target_x, target_y, raw_x, raw_y)] with targets in screen units (-1..1).
        Fits raw = offset + scale * target per axis (least squares)."""
        s = np.array(samples, float)
        cal = {}
        for axis, (ti, ri) in (("x", (0, 2)), ("y", (1, 3))):
            a = np.column_stack([np.ones(len(s)), s[:, ti]])
            (off, scale), *_ = np.linalg.lstsq(a, s[:, ri], rcond=None)
            if abs(scale) < 0.02:
                raise ValueError(f"eyes barely moved along {axis}; keep your head still and follow the dot with your eyes")
            cal[f"{axis}0"], cal[f"{axis}s"] = float(off), float(scale)
        self.calib = {**(self.calib or {}), "gaze": cal}
        return cal


# ---------------------------------------------------------------- forward kinematics
# simple body volume (in the rest skeleton's proportions, metres): a torso capsule along the spine, pushed a
# little forward (the chest is deeper in front of the spine than behind it), and a sphere for the head
TORSO_RADIUS, TORSO_FORWARD, HEAD_RADIUS, CLEARANCE = 0.11, 0.02, 0.11, 0.015


def _capsule_push(p, a, b, r):
    """(outward unit vector, depth) if point p is inside the capsule a-b of radius r, else None."""
    ab = b - a
    s = float(np.clip((p - a) @ ab / (ab @ ab), 0, 1)) if ab @ ab > 1e-12 else 0.0  # (a == b: a sphere)
    c = a + ab * s
    d = p - c
    dist = float(np.linalg.norm(d))
    if dist >= r:
        return None
    n = d / dist if dist > 1e-6 else np.array([0, 0, 1.0])
    return n, r - dist


def _body_push(p, pos, rot, scale=1.0):
    """Largest push needed to get point p out of the torso capsule / head sphere: (vector) or None.
    scale: body thickness (settings page) for bulkier or slimmer avatars."""
    fwd = rot["Chest"] @ np.array([0, 0, TORSO_FORWARD * scale])
    head = pos["Head"] + rot["Head"] @ np.array([0, 0.09, 0.02])
    pushes = [_capsule_push(p, pos["Spine"] + fwd, pos["Neck"] + fwd, TORSO_RADIUS * scale + CLEARANCE),
              _capsule_push(p, head, head, HEAD_RADIUS * scale + CLEARANCE)]
    pushes = [x for x in pushes if x]
    if not pushes:
        return None
    n, depth = max(pushes, key=lambda x: x[1])
    return n * depth


def reach(local, S, target, pos, rot):
    """Re-solve shoulder -> elbow -> wrist (2-bone IK, elbow kept on its side) so the wrist goes to `target`.
    pos / rot: forward kinematics of `local`. Edits `local`."""
    up, lo, hand = f"{S}UpperArm", f"{S}LowerArm", f"{S}Hand"
    sh, el, wr = pos[up], pos[lo], pos[hand]
    g_sh, g_up, g_lo = rot[f"{S}Shoulder"], rot[up], rot[lo]
    l1, l2 = float(np.linalg.norm(el - sh)), float(np.linalg.norm(wr - el))
    to = target - sh
    d = float(np.clip(np.linalg.norm(to), abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
    axis = _norm(to)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole = (el - sh) - axis * float((el - sh) @ axis)
    pole = _norm(pole) if np.linalg.norm(pole) > 1e-6 else _norm(_cross(axis, np.array([0, 1.0, 0])))
    new_el = sh + axis * a + pole * h
    new_wr = sh + axis * d
    g_up2 = _arc(_norm(el - sh), _norm(new_el - sh)) @ g_up
    g_lo_tmp = g_up2 @ (g_up.T @ g_lo)
    cur_dir = _norm(g_lo_tmp @ (g_lo.T @ _norm(wr - el)))  # the forearm's direction after the upper arm moved
    g_lo2 = _arc(cur_dir, _norm(new_wr - new_el)) @ g_lo_tmp
    local[up] = q_from_mat(g_sh.T @ g_up2)
    local[lo] = q_from_mat(g_up2.T @ g_lo2)


def keep_arm_out_of_body(local, S, scale=1.0, rest=None):
    """If the elbow, wrist or middle of the hand would be inside the torso or head, bend the arm (2-bone IK
    from the shoulder, keeping the elbow on its side) so they rest on the surface instead. Edits `local`."""
    up, lo, hand = f"{S}UpperArm", f"{S}LowerArm", f"{S}Hand"
    for _ in range(2):
        pos, rot = forward_kinematics(local, rest=rest, only=("Head", hand))
        sh, el, wr = pos[up], pos[lo], pos[hand]
        # elbow inside: swing the whole arm out at the shoulder
        push = _body_push(el, pos, rot, scale)
        if push is not None:
            r1 = _arc(_norm(el - sh), _norm(el + push - sh))
            local[up] = q_from_mat(rot[f"{S}Shoulder"].T @ (r1 @ rot[up]))
            continue
        # wrist / hand inside: move the wrist out by the deepest push
        mid = wr + rot[hand] @ (ARM_DIR[S] * 0.08)
        pushes = [x for x in (_body_push(wr, pos, rot, scale), _body_push(mid, pos, rot, scale)) if x is not None]
        if not pushes:
            return
        reach(local, S, wr + max(pushes, key=lambda v: float(np.linalg.norm(v))), pos, rot)


TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Little": 20}  # hand landmark of each fingertip


def _tip(pos, rot, rest, S, finger):
    """Avatar fingertip (the tip segment taken as long as the last bone)."""
    k = f"{S}{finger}Distal"
    return pos[k] + rot[k] @ np.asarray(rest[k][1], float)


def pinch_contact(hand):
    """(finger, strength 0..1) if the thumb tip touches a fingertip of the same hand, else (None, 0).
    Touching = tips within 0.3 palm lengths (~3 cm), full strength from 0.18."""
    w = np.asarray(hand.get("_world_fingers", hand["_world"]), float)
    palm = float(np.linalg.norm(w[0] - w[9])) or 1.0
    d = {f: float(np.linalg.norm(w[4] - w[i])) / palm for f, i in TIPS.items() if f != "Thumb"}
    f = min(d, key=d.get)
    return (f, float(np.clip((0.3 - d[f]) / 0.12, 0, 1))) if d[f] < 0.3 else (None, 0.0)


def _chain(local_or_bones, names):
    """Global (hand-frame) rotations along a finger chain."""
    g, out = np.eye(3), []
    for n in names:
        g = g @ mat_from_q(local_or_bones[n])
        out.append(g)
    return out


def solve_pinch(local, S, finger, strength, rest):
    """Make the avatar's thumb tip and `finger`'s tip meet, in the avatar's own proportions (the tracked tips
    touch, but different finger lengths left a gap): thumb (4 angles) and that finger's knuckle + middle joint
    (tip joint coupled) are solved together, all within human limits, staying as close as possible to the
    tracked pose. Real pinches bend both; the thumb alone often couldn't reach a straighter finger.
    Solved as least squares (gap in cm, plus 0.01 x each angle's change in degrees) with Levenberg-Marquardt:
    ~1 ms. (A general-purpose search took ~150 ms per pinching hand and stalled the avatar on every pinch.)"""
    along, side = ARM_DIR[S], np.array([0, 0, 1.0])
    t_names = [f"{S}Thumb{s}" for s in SEGS]
    f_names = [f"{S}{finger}{s}" for s in SEGS]
    t_base, f_base = (np.asarray(rest[n[0]][1], float) for n in (t_names, f_names))
    t_seg0, t_seg1 = (np.asarray(rest[n][1], float) for n in t_names[1:])
    f_seg0, f_seg1 = (np.asarray(rest[n][1], float) for n in f_names[1:])
    rest_dir = _norm(t_seg0)
    lim, fl = THUMB_LIMITS, FINGER_LIMITS
    lo = np.array([lim["swing"][0], lim["tilt"][0], lim["Intermediate"][0], lim["Distal"][0],
                   fl["Proximal"][0], fl["Intermediate"][0]])
    hi = np.array([lim["swing"][1], lim["tilt"][1], lim["Intermediate"][1], lim["Distal"][1],
                   fl["Proximal"][1], fl["Intermediate"][1]])
    g = _chain(local, t_names)
    t0 = thumb_angles([m @ rest_dir for m in g], rest_dir, along, S)
    g = _chain(local, f_names)
    a0 = finger_angles([m @ along for m in g], along, side)
    x0 = np.clip(np.array([*t0, a0[0], a0[2]]), lo, hi)
    # what doesn't change while solving: the thumb's swing axis and flat rest direction, the finger's spread
    k_sw, r_flat = _swing_axis(rest_dir, along), _flat(rest_dir, rest_dir)
    sp = math.radians(a0[1])
    spread_dir = along * math.cos(sp) + side * math.sin(sp)
    h_f, f_arc = _norm(_cross(spread_dir, PALM_DOWN)), _arc(along, spread_dir)
    rad = math.pi / 180

    def gap(x):
        """Thumb tip minus finger tip (hand frame, metres) for angles x (degrees) - _thumb_bones / _finger_bones
        without building quaternions."""
        sw, tl, t1, t2, f0, f1 = x * rad
        g_sw = _axis_angle(k_sw, sw)
        g0 = _axis_angle(_norm(_cross(g_sw @ r_flat, PALM_DOWN)), tl) @ g_sw
        h = _thumb_hinge(g0 @ rest_dir, S)
        tt = t_base + g0 @ t_seg0 + _axis_angle(h, t1) @ (g0 @ t_seg1) + _axis_angle(h, t1 + t2) @ (g0 @ t_seg1)
        fg0 = _axis_angle(h_f, f0) @ f_arc
        ft = f_base + fg0 @ f_seg0 + _axis_angle(h_f, f1) @ (fg0 @ f_seg1) + _axis_angle(h_f, 1.7 * f1) @ (fg0 @ f_seg1)
        return tt - ft

    def residual(x, d):
        return np.concatenate([d * 100.0, (x - x0) * 0.01])

    x, d = x0, gap(x0)
    r = residual(x, d)
    cost, lam, eps = float(r @ r), 1e-3, 0.25
    for _ in range(30):
        if float(d @ d) < 2.5e-7:  # tips within 0.5 mm
            break
        jac = np.zeros((9, 6))
        for j in range(6):
            xj = x.copy()
            xj[j] += eps if xj[j] + eps <= hi[j] else -eps
            jac[:3, j] = (gap(xj) - d) * 100.0 / (xj[j] - x[j])
        jac[3:, :] = np.eye(6) * 0.01
        a, b = jac.T @ jac, jac.T @ r
        while lam < 1e6:
            x_new = np.clip(x - np.linalg.solve(a + lam * np.diag(np.diag(a)), b), lo, hi)
            d_new = gap(x_new)
            r_new = residual(x_new, d_new)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                x, d, r, lam = x_new, d_new, r_new, max(lam / 3, 1e-7)
                break
            lam *= 4
        else:
            break
        if cost - c_new < 1e-9 * max(cost, 1e-12):
            cost = c_new
            break
        cost = c_new
    tb = _thumb_bones(S, *x[:4], rest_dir)
    fb = _finger_bones(S, finger, x[4], a0[1], x[5], 0.7 * x[5], along, side)
    for k in t_names:
        local[k] = q_slerp(local[k], tb[k], strength)
    for k in f_names:
        local[k] = q_slerp(local[k], fb[k], strength)


def two_hand_contact(res_hands):
    """[(left finger, right finger)] whose tips touch each other, and a strength 0..1 (2D: tips within 0.25 hand
    sizes on screen, both hands at a similar distance)."""
    L, R = res_hands.get("Left"), res_hands.get("Right")
    if L is None or R is None or "_cam" in L or "_cam" in R:  # (picture positions of one camera only)
        return [], 0.0
    if L.get("distance_cm") and R.get("distance_cm") and abs(L["distance_cm"] - R["distance_cm"]) > 10:
        return [], 0.0
    size = (np.linalg.norm(L["_pts"][0, :2] - L["_pts"][9, :2]) + np.linalg.norm(R["_pts"][0, :2] - R["_pts"][9, :2])) / 2
    pairs, best = [], 1e9
    for f, i in TIPS.items():
        for g, j in TIPS.items():
            d = float(np.linalg.norm(L["_pts"][i, :2] - R["_pts"][j, :2])) / max(size, 1.0)
            if d < 0.25:
                pairs.append((f, g))
                best = min(best, d)
    return pairs, float(np.clip((0.25 - best) / 0.1, 0, 1)) if pairs else 0.0


def solve_two_hand(local, pairs, strength, rest):
    """Move both wrists (arm IK) so the touching fingertips meet halfway, in the avatar's proportions. Moving a
    wrist also turns the hand a little, so it repeats until the gap is under 5 mm (at most 6 passes)."""
    only = tuple(f"Left{f}Distal" for f, _ in pairs) + tuple(f"Right{g}Distal" for _, g in pairs)
    for _ in range(6):
        pos, rot = forward_kinematics(local, rest=rest, only=only)
        dl = np.mean([_tip(pos, rot, rest, "Left", f) for f, _ in pairs], 0)
        dr = np.mean([_tip(pos, rot, rest, "Right", g) for _, g in pairs], 0)
        if np.linalg.norm(dl - dr) * strength < 0.005:
            return
        mid = (dl + dr) / 2
        for S, tip in (("Left", dl), ("Right", dr)):  # (the arms are separate chains: one FK serves both)
            reach(local, S, pos[f"{S}Hand"] + (mid - tip) * strength, pos, rot)


FACE_TOUCH = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Little": 20, "palm": 9}  # hand points (landmarks)
FACE_TOUCH_NEAR = 0.06   # a hand point touches when a face mesh point is this close (share of the face's width)...
FACE_TOUCH_REACH = 0.3   # ...and from this close it's eased toward the face (0.45 / 0.6 measured the same)
FACE_TOUCH_SIZE = (1.30, 1.36)  # hand looks this much bigger than the pupil gap: on the face .. in front of it
USER_IPD, FINGER_PAD = 0.063, 0.008  # your pupil gap (as all_in_one assumes); a fingertip's thickness (m)


def face_contact(face, hand):
    """[(hand part, nearest face mesh point, picture offset from it in px)] for the hand points at or near the
    face, and a strength 0..1.
    - Near: 1 once a point touches (a face mesh point within FACE_TOUCH_NEAR of it - the mesh covers the face
      densely and nothing else), easing to 0 at FACE_TOUCH_REACH.
    - At the face, not in front of it: the hand's largest palm segment over the pupil gap, both in the picture
      (a closer hand looks bigger). On take.mp4 hands on the face read 1.11 (10th-90th 0.91-1.24), arms crossed in
      front of it 1.41 (1.23-1.46). Nearly a yes / no (a half-strength touch leaves the avatar's hand hanging
      between the body model's depth and the face); _face_touch eases it in time. The distances all_in_one
      estimates couldn't tell them apart (hands on the face: -52 .. +59 cm from the face)."""
    if face is None or hand is None or "_cam" in face or "_cam" in hand:
        return [], 0.0
    fp, hp = face["_pts"][:, :2], hand["_pts"][:, :2]
    width = float(np.ptp(fp[:, 0])) or 1.0
    pairs, best = [], 1e9
    for part, k in FACE_TOUCH.items():
        d = np.linalg.norm(fp - hp[k], axis=1)
        j = int(np.argmin(d))
        if d[j] < FACE_TOUCH_REACH * width:
            pairs.append((part, j, hp[k] - fp[j]))
            best = min(best, float(d[j]))
    if not pairs:
        return [], 0.0
    near = _clamp((FACE_TOUCH_REACH * width - best) / ((FACE_TOUCH_REACH - FACE_TOUCH_NEAR) * width), 0, 1)
    size = max(float(np.linalg.norm(hp[a] - hp[b])) for a, b in ((0, 5), (0, 17), (5, 17), (0, 9))) / \
        max(float(np.linalg.norm(fp[468] - fp[473])), 1.0)
    lo, hi = FACE_TOUCH_SIZE
    return pairs, near * _clamp((hi - size) / (hi - lo), 0, 1)


def _hand_point(pos, rot, rest, S, part):
    return pos[f"{S}MiddleProximal"] if part == "palm" else _tip(pos, rot, rest, S, part)


def solve_face_contact(local, S, pairs, strength, face_pts, face_m, calib_head, rest):
    """Move the wrist (arm IK) so the hand points on your face land on the same spots of the avatar's face (and
    points just beside it, beside it at its depth). A spot: your face mesh point plus the hand point's offset from
    it in the picture, around your eyes' midpoint (metres from the 6.3 cm pupil gap, in your head's frame), put on
    the avatar from its eyes in its head's frame, scaled by its eye spacing, pushed out by a fingertip's thickness.
    On take.mp4 the avatar's hand on its face was ~30 cm in front of it (median): the body model's depth for a hand
    at the face is far off, and the no-clip head sphere reaches further forward than a face."""
    p = to_unity(face_pts)
    eye = (p[468] + p[473]) / 2
    m_per_px = USER_IPD / max(float(np.linalg.norm(p[468] - p[473])), 1e-6)
    hand = f"{S}Hand"
    pos, rot = forward_kinematics(local, rest=rest, only=(hand, f"{S}MiddleProximal", "LeftEye", "RightEye") +
                                  tuple(f"{S}{part}Distal" for part, *_ in pairs if part != "palm"))
    eyes = (pos["LeftEye"] + pos["RightEye"]) / 2
    scale = float(np.linalg.norm(pos["LeftEye"] - pos["RightEye"])) / USER_IPD
    to_avatar = rot["Head"] @ calib_head
    centre = pos["Head"] + rot["Head"] @ np.array([0, 0.09, 0.02])
    target = np.zeros(3)
    for _, j, off in pairs:
        t = eyes + to_avatar @ (face_m.T @ (p[j] + to_unity((*off, 0.0)) - eye) * m_per_px) * scale
        target += t + _norm(t - centre) * FINGER_PAD
    target /= len(pairs)
    # the hand's own pose doesn't change below (only the arm moves): its points are fixed in the hand's frame
    off = rot[hand].T @ (np.mean([_hand_point(pos, rot, rest, S, part) for part, *_ in pairs], 0) - pos[hand])
    point = pos[hand] + rot[hand] @ off
    goal = point + (target - point) * strength  # (strength once: per pass it would compound)
    # At the face the arm is folded (wrist ~0.27 arm lengths from the shoulder): moving the wrist turns the
    # forearm and hand a lot and the touch point lags behind. Moving the wrist by the point's error each pass left
    # 5 cm after 3 passes (8 to get under 4 mm); estimating how the point follows the wrist (Broyden) takes ~5.
    up, lo = f"{S}UpperArm", f"{S}LowerArm"
    J = np.eye(3)
    best = (float(np.linalg.norm(goal - point)), local[up], local[lo])
    for _ in range(8):
        err = goal - point
        e = float(np.linalg.norm(err))
        if e < best[0]:
            best = (e, local[up], local[lo])
        if e < 0.004:
            return
        move = np.linalg.solve(J, err)
        if float(np.linalg.norm(move)) > 2 * e:  # (a bad estimate: start over from "the point follows the wrist")
            J, move = np.eye(3), err
        wrist = pos[hand]
        reach(local, S, wrist + move, pos, rot)
        pos, rot = forward_kinematics(local, rest=rest, only=(hand,))
        moved, new = pos[hand] - wrist, pos[hand] + rot[hand] @ off
        if float(moved @ moved) > 1e-10:
            J += np.outer(new - point - J @ moved, moved) / float(moved @ moved)
        point = new
    if float(np.linalg.norm(goal - point)) > best[0]:
        local[up], local[lo] = best[1], best[2]


_FK_ORDER, _FK_OFFSETS = {}, {}


def _fk_order(only):
    """The bones needed to place `only` (them and their ancestors), parents first."""
    order = _FK_ORDER.get(only)
    if order is None:
        need = set()
        for n in only:
            while n is not None and n not in need:
                need.add(n)
                n = REST[n][0]
        order = _FK_ORDER[only] = [n for n in REST if n in need]
    return order


def forward_kinematics(bones, root=np.zeros(3), rest=None, only=None):
    """Global positions and rotations of every rest-skeleton bone given local quaternions. rest: the avatar's
    own proportions (AvatarRig.rest, sent by the 3D viewer), default REST. only: a tuple of bone names - just
    those and their parents are computed (the solvers need an arm and the torso, not all 51 bones)."""
    rest = rest or REST
    offs = _FK_OFFSETS.get(id(rest))
    if offs is None or offs[0] is not rest:  # offsets as arrays, once per skeleton
        offs = _FK_OFFSETS[id(rest)] = (rest, {k: np.array(o, float) for k, (_, o) in rest.items()})
    offs = offs[1]
    pos, rot = {}, {}
    for name in (_fk_order(only) if only else rest):  # ordered parents-first
        parent = rest[name][0]
        q = bones.get(name, IDENTITY)
        if parent is None:
            rot[name] = mat_from_q(q)
            pos[name] = offs[name] + root
        else:
            pos[name] = pos[parent] + rot[parent] @ offs[name]
            rot[name] = rot[parent] @ mat_from_q(q)
    return pos, rot
