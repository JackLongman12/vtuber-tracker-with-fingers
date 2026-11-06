"""Known avatar poses / faces, sent instead of live tracking, to check a VTuber app + model end to end.

Run:  .venv\\Scripts\\python.exe app.py studio --test-pattern
Each step holds for a few seconds; compare your avatar with the description. If a step looks wrong (an arm goes
backwards, the head turns the other way, a vowel shows the wrong mouth), that tells you exactly which mapping is off.
"Left/right" always means the AVATAR's own left/right (as if you were the avatar), unless it says "your".
"""
import math

import numpy as np

from avatarstudio.avatar.rig import IDENTITY, REST, q_from_mat, rot_x, rot_y, rot_z

D = math.radians
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
SEGS = ("Proximal", "Intermediate", "Distal")


def _q(m):
    return q_from_mat(m)


def _fist(side, except_index=False):
    bones = {}
    for f in FINGERS:
        if except_index and f == "Index":
            continue
        for s in SEGS:
            if f == "Thumb":
                m = rot_y(D(-35) if side == "Left" else D(35))
            else:
                m = rot_z(D(75) if side == "Left" else D(-75))  # curl toward the palm (palm faces down in T-pose)
            bones[f"{side}{f}{s}"] = _q(m)
    return bones


ARMS_DOWN = {"LeftUpperArm": _q(rot_z(D(70))), "RightUpperArm": _q(rot_z(D(-70)))}

# (description, bones, shapes, root)
STEPS = [
    ("T-pose: both arms straight out to the sides, palms down", {}, {}, None),
    ("Both arms relaxed down by the sides", ARMS_DOWN, {}, None),
    ("LEFT arm points straight forward, toward the camera / you", {"LeftUpperArm": _q(rot_y(D(90)))}, {}, None),
    ("RIGHT arm points straight up", {"RightUpperArm": _q(rot_z(D(90)))}, {}, None),
    ("LEFT arm out to the side, elbow bent 90 deg, forearm pointing forward",
     {"LeftLowerArm": _q(rot_y(D(90)))}, {}, None),
    ("LEFT shoulder shrugs up (arm out to the side)", {"LeftShoulder": _q(rot_z(D(-15)))}, {}, None),
    ("LEFT wrist turned: palm faces UP (arm out to the side)", {"LeftHand": _q(rot_x(D(180)))}, {}, None),
    ("LEFT hand makes a fist", _fist("Left"), {}, None),
    ("RIGHT hand points with the index finger, other fingers curled", _fist("Right", except_index=True), {}, None),
    ("Head turns toward the avatar's LEFT (to YOUR right as you face it)", {**ARMS_DOWN, "Head": _q(rot_y(D(-30)))},
     {}, None),
    ("Head looks UP", {**ARMS_DOWN, "Head": _q(rot_x(D(-25)))}, {}, None),
    ("Head tilts toward the avatar's RIGHT shoulder", {**ARMS_DOWN, "Head": _q(rot_z(D(-20)))}, {}, None),
    ("Upper body leans FORWARD, toward the camera", {**ARMS_DOWN, "Spine": _q(rot_x(D(15)))}, {}, None),
    ("Eyes look to the avatar's LEFT", {**ARMS_DOWN, "LeftEye": _q(rot_y(D(-15))), "RightEye": _q(rot_y(D(-15)))},
     {}, None),
    ("Avatar steps 30 cm toward the camera", ARMS_DOWN, {}, (0.0, 0.0, 0.3)),
    ("LEFT eye closed (wink); right eye open", ARMS_DOWN, {"vrm": {"Blink_L": 1.0}, "arkit": {"eyeBlinkLeft": 1.0}},
     None),
    ("Mouth: A (aah, wide open)", ARMS_DOWN, {"vrm": {"A": 1.0}, "arkit": {"jawOpen": 0.8}}, None),
    ("Mouth: I (eee, wide and flat)", ARMS_DOWN, {"vrm": {"I": 1.0}, "arkit": {"mouthStretchLeft": 0.6,
                                                                            "mouthStretchRight": 0.6}}, None),
    ("Mouth: U (ooo, small and round)", ARMS_DOWN, {"vrm": {"U": 1.0}, "arkit": {"mouthPucker": 0.9}}, None),
    ("Mouth: E (eh)", ARMS_DOWN, {"vrm": {"E": 1.0}, "arkit": {"jawOpen": 0.35, "mouthStretchLeft": 0.4,
                                                               "mouthStretchRight": 0.4}}, None),
    ("Mouth: O (oh, open and round)", ARMS_DOWN, {"vrm": {"O": 1.0}, "arkit": {"jawOpen": 0.5, "mouthFunnel": 0.8}},
     None),
    ("Expression: Joy / happy", ARMS_DOWN, {"vrm": {"Joy": 1.0}, "arkit": {"mouthSmileLeft": 0.9,
                                                                      "mouthSmileRight": 0.9}}, None),
    ("Perfect Sync only: LEFT side of the mouth smiles, right side doesn't", ARMS_DOWN,
     {"arkit": {"mouthSmileLeft": 1.0}}, None),
    ("Perfect Sync only: cheeks puffed", ARMS_DOWN, {"arkit": {"cheekPuff": 1.0}}, None),
]
STEP_S = 3.0
ALL_SHAPES = {"Blink", "Blink_L", "Blink_R", "A", "I", "U", "E", "O", "Joy", "Angry", "Sorrow", "Surprised", "Fun"}


def frame_at(t, mirror=False):
    """(step index, description, avatar dict) for time t (seconds since start); loops forever."""
    i = int(t // STEP_S) % len(STEPS)
    desc, bones, shapes, root = STEPS[i]
    b = dict(bones)
    vrm = {k: 0.0 for k in ALL_SHAPES}
    vrm.update(shapes.get("vrm", {}))
    arkit = {k: 0.0 for k in ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
                              "mouthPucker", "mouthFunnel", "mouthStretchLeft", "mouthStretchRight", "cheekPuff")}
    arkit.update(shapes.get("arkit", {}))
    # every bone is sent every step (receivers keep a bone's last value, so a raised arm would otherwise stay up)
    out = {"bones": {**{n: IDENTITY.copy() for n in REST}, **b},
           "root": np.array(root or (0.0, 0.0, 0.0)), "vrm": vrm, "arkit": arkit, "gaze": (0.0, 0.0)}
    if mirror:
        from avatarstudio.avatar.rig import mirror_name
        out["bones"] = {mirror_name(k): np.array([q[0], -q[1], -q[2], q[3]]) for k, q in out["bones"].items()}
        out["root"] = out["root"] * np.array([-1, 1, 1])
        out["vrm"] = {mirror_name(k): v for k, v in out["vrm"].items()}
        out["arkit"] = {mirror_name(k): v for k, v in out["arkit"].items()}
    return i, desc, out
