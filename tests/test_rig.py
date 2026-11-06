"""Self-test for rig.py: rebuild the skeleton from the computed bone rotations (forward kinematics) and
check every limb points where the tracker saw it, then check mirroring swaps sides correctly.

Usage: .venv\\Scripts\\python.exe tests/test_rig.py [images...]
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # (the project folder, where the avatarstudio package is)
os.chdir(ROOT)  # (models/ is found from there)
from avatarstudio.tracking.all_in_one import AllInOneTracker, focal_px  # noqa: E402
from avatarstudio.avatar.rig import (ARM_DIR, AvatarRig, FINGER_IDS, LEG_DIR, POSE_IDS, REST,  # noqa: E402
                                     forward_kinematics, mirror_name, to_unity)


def angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))


def check(res, f, name):
    rig = AvatarRig(smoothing=False, root_motion=False, idle=False)
    for t in (0.0, 0.1, 0.2):  # hands ease in when they first appear: check after that
        out = rig(res, f, t)
    pos, rot = forward_kinematics(out["bones"])
    errors = {}
    body = res["body"]
    if body is not None:
        w, vis = to_unity(body["_world"]), body["_vis"]
        for S in ("Left", "Right"):
            j = POSE_IDS[S]
            for bone, a, b, rest in ((f"{S}UpperArm", "s", "e", ARM_DIR[S]), (f"{S}LowerArm", "e", "w", ARM_DIR[S]),
                                     (f"{S}UpperLeg", "h", "k", LEG_DIR), (f"{S}LowerLeg", "k", "a", LEG_DIR)):
                if vis[j[a]] > 0.5 and vis[j[b]] > 0.5 and bone in out["bones"]:
                    if "Arm" in bone and vis[j["w"]] < 0.5:
                        continue
                    errors[bone] = angle_deg(rot[bone] @ rest, w[j[b]] - w[j[a]])
    for hand in res["hands"]:
        S = hand["side"].capitalize()
        hw = to_unity(hand["_world"])
        errors[f"{S}Hand"] = angle_deg(rot[f"{S}Hand"] @ ARM_DIR[S], hw[9] - hw[0])
        # fingertip direction of the middle finger's last bone
        errors[f"{S}MiddleDistal"] = angle_deg(rot[f"{S}MiddleDistal"] @ ARM_DIR[S], hw[12] - hw[11])
    if res["faces"]:
        p = to_unity(max(res["faces"], key=lambda fc: fc["box"][2])["_pts"])
        errors["Head(up)"] = angle_deg(rot["Head"] @ np.array([0, 1.0, 0]), p[10] - p[152])
    print(f"{name}: " + ", ".join(f"{k} {v:.1f}" for k, v in errors.items()))

    # mirror: every bone on one side must equal the reflected bone of the other side
    mr = AvatarRig(smoothing=False, root_motion=False, mirror=True, idle=False)
    for t in (0.0, 0.1, 0.2):
        m = mr(res, f, t)
    worst = 0.0
    for bname, q in out["bones"].items():
        mq = m["bones"][mirror_name(bname)]
        expect = np.array([q[0], -q[1], -q[2], q[3]])
        worst = max(worst, 1 - abs(float(np.dot(mq, expect))))
    print(f"   mirror check: worst quaternion mismatch {worst:.2e}")
    return errors


def main():
    images = sys.argv[1:] or ["tests/images/astronaut-waving.jpg", "tests/images/portrait.jpg",
                              "tests/images/astronaut-phone.jpg"]  # (public domain: tests/images/SOURCES.md)
    tracker = AllInOneTracker(video=False)
    all_err = []
    for path in images:
        frame = cv2.imread(path)
        res = tracker(frame)
        all_err += list(check(res, focal_px(frame.shape[1], 65), path).values())
    print(f"\nlimb direction error: median {np.median(all_err):.1f} deg, max {np.max(all_err):.1f} deg "
          f"over {len(all_err)} bones")
    print(f"rest skeleton has {len(REST)} bones; fingers per hand: {len(FINGER_IDS)}")


if __name__ == "__main__":
    main()
