"""All-in-one tracker: face mesh, head pose, gaze, eyes/mouth, face shapes, full body and arms,
both hands and every finger, plus depth (distance from the camera, moving closer/away, body turn and lean),
from a single call per frame.

Inside, one tracker runs three networks on each frame, side by side:
  - MediaPipe FaceLandmarker: 478 face landmarks incl. irises + blendshapes (up to 4 faces)
  - MediaPipe HandLandmarker: 21 landmarks per hand with its own palm detector (up to 2 hands)
  - MediaPipe PoseLandmarker: 33 body landmarks in 2D and 3D (metres): arms, legs, torso turn and lean
(+ optionally the GPU whole-body model, wholebody.py, for left/right of crossing hands)
Depth: distance is estimated from the gap between the pupils (about 6.3 cm on adults), or from the body's
3D size when no face is visible, and each hand gets its own distance from its 3D size. Absolute distance
assumes a typical webcam field of view (--fov) and pupil gap (--ipd), so treat it as ~10-15% accurate;
changes in distance (moving a few cm closer or further) are much more precise.

Usage:
  .venv\\Scripts\\python.exe -m avatarstudio.tracking.all_in_one --webcam              # q quit, m mesh, b body, r reset depth
  .venv\\Scripts\\python.exe -m avatarstudio.tracking.all_in_one photo.jpg             # -> photo_tracked.jpg + printed summary
  .venv\\Scripts\\python.exe -m avatarstudio.tracking.all_in_one --webcam --log run.jsonl   # also save per-frame data
"""
import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def _import_mediapipe():
    """Import MediaPipe without the two parts of its package start-up this app never uses: its audio
    classifier (which loads sounddevice) and its drawing helpers (which load matplotlib) - 0.6 s of the
    studio's start. They come back as empty modules through the normal import system, so MediaPipe's own
    `import ...drawing_utils` lines still work."""
    import importlib.abc
    import importlib.machinery
    import sys

    class Skip(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        names = {"mediapipe.tasks.python.audio", "mediapipe.tasks.python.vision.drawing_utils",
                 "mediapipe.tasks.python.vision.drawing_styles"}

        def find_spec(self, name, path, target=None):
            return importlib.machinery.ModuleSpec(name, self) if name in self.names else None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            pass

    skip = Skip()
    sys.meta_path.insert(0, skip)
    try:
        import mediapipe
        from mediapipe.tasks.python import BaseOptions, vision
    finally:
        sys.meta_path.remove(skip)
    return mediapipe, BaseOptions, vision


mp, BaseOptions, vision = _import_mediapipe()

FACE_MODEL = "models/face_landmarker.task"
HAND_MODEL = "models/hand_landmarker.task"
POSE_MODELS = {"lite": "models/pose_landmarker_lite.task", "full": "models/pose_landmarker_full.task"}
VIS = 0.5                  # minimum landmark visibility to use / draw a body point
EYE_MAX_YAW = 30.0         # beyond this head turn, distance comes from the body instead of the eyes
DEPTH_BLEND_S = 0.75       # crossfade time when switching between eye- and body-based distance
WB_FRESH_MS = 100          # a GPU whole-body result older than this is not used for left/right
WB_KEEP_MS = 500           # ...and it keeps running this long after left/right stopped being in doubt
WB_GPU_MS = 90             # with GPU tracking it shares the GPU thread (~17 ms a run): one run per 90 ms at most

FINGERS = {  # landmark chains from base to tip (MediaPipe hand model)
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
GESTURES = {  # extended fingers (thumb, index, middle, ring, pinky) -> name
    (0, 0, 0, 0, 0): "fist",
    (1, 1, 1, 1, 1): "open palm",
    (0, 1, 0, 0, 0): "pointing",
    (0, 1, 1, 0, 0): "peace",
    (0, 1, 1, 1, 0): "three",
    (0, 1, 1, 1, 1): "four",
    (1, 1, 0, 0, 1): "I love you",
    (0, 1, 0, 0, 1): "rock on",
    (1, 0, 0, 0, 1): "call me",
    (1, 1, 0, 0, 0): "L shape",
}
# body landmark ids (person's own side)
LIMBS = {"left": dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
         "right": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28)}


def _np(landmarks, w, h):
    """MediaPipe normalized landmarks -> (N, 3) pixel array (z in the same scale as x).
    (gpu_tracking.py hands over pixel arrays already.)"""
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return np.array([(p.x * w, p.y * h, p.z * w) for p in landmarks], dtype=np.float32)


def _xyz(landmarks):
    """MediaPipe world landmarks -> (N, 3) array (metres)."""
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)


def _bend(a, b, c):
    """Angle in degrees the chain a->b->c turns at b (0 = straight)."""
    u, v = b - a, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def _angle(a, b, c):
    """Inner angle at b between b->a and b->c in degrees (180 = straight line)."""
    return 180.0 - _bend(a, b, c)


def focal_px(width, fov_deg):
    return (width / 2) / math.tan(math.radians(fov_deg) / 2)


def _distance_from_size(world_pts, px_pts, segments, f):
    """Distance (cm) from how big known-size 3D segments (metres) appear in the image (pixels)."""
    real = sum(np.linalg.norm(world_pts[a, :2] - world_pts[b, :2]) for a, b in segments)
    px = sum(np.linalg.norm(px_pts[a, :2] - px_pts[b, :2]) for a, b in segments)
    return float(f * real / px * 100) if px > 1 and real > 0 else None


# ---------------------------------------------------------------- face / head
def head_pose(pts):
    """Yaw / pitch / roll in degrees from 3D face landmarks.
    yaw > 0: turned to the person's right; pitch > 0: looking up; roll > 0: tilted clockwise in the image."""
    x_axis = pts[263] - pts[33]           # across the eyes (image left -> right)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = pts[152] - pts[10]           # forehead -> chin
    y_axis -= x_axis * np.dot(y_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)     # points away from the camera when facing it
    yaw = math.degrees(math.atan2(z_axis[0], z_axis[2]))
    pitch = math.degrees(math.atan2(z_axis[1], math.hypot(z_axis[0], z_axis[2])))
    roll = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    return yaw, pitch, roll


def gaze(pts):
    """Horizontal / vertical iris position inside each eye, averaged: -1 .. 1 (0 = centre)."""
    hs, vs = [], []
    for iris, outer, inner, top, bottom in ((468, 33, 133, 159, 145), (473, 263, 362, 386, 374)):
        left, right = sorted((pts[outer][0], pts[inner][0]))
        hs.append((pts[iris][0] - (left + right) / 2) / max((right - left) / 2, 1e-6))
        vs.append((pts[iris][1] - (pts[top][1] + pts[bottom][1]) / 2) /
                  max(abs(pts[bottom][1] - pts[top][1]) / 2, 1e-6))
    h, v = float(np.mean(hs)), float(np.mean(vs))
    label = "centre"
    if abs(h) > 0.35 or abs(v) > 0.6:
        label = ("right" if h > 0 else "left") if abs(h) / 0.35 >= abs(v) / 0.6 else ("down" if v > 0 else "up")
    return {"x": round(h, 2), "y": round(v, 2), "direction": label + " (image)"}


def analyze_face(landmarks, blendshapes, w, h, f, ipd_cm):
    pts = _np(landmarks, w, h)
    x0, y0 = pts[:, :2].min(0)
    x1, y1 = pts[:, :2].max(0)
    yaw, pitch, roll = head_pose(pts)
    # pupil-to-pupil gap measured in 3D (z included) so turning the head doesn't shrink it
    ipd_px = float(np.linalg.norm(pts[468] - pts[473]))
    face = {
        "box": [round(float(v), 1) for v in (x0, y0, x1 - x0, y1 - y0)],
        "head_pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)},
        "gaze": gaze(pts),
        "distance_cm": round(f * ipd_cm / ipd_px, 1) if ipd_px > 1 else None,
        "_pts": pts,
    }
    if blendshapes:
        bs = {c.category_name: c.score for c in blendshapes if c.category_name != "_neutral"}
        face["blendshapes"] = {k: round(float(v), 3) for k, v in bs.items()}  # the 52 ARKit shapes
        face["eyes_open"] = {"left": bs.get("eyeBlinkLeft", 0) < 0.5, "right": bs.get("eyeBlinkRight", 0) < 0.5}
        face["mouth_open"] = round(bs.get("jawOpen", 0), 2)
        face["smile"] = round((bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2, 2)
    return face


# ---------------------------------------------------------------- body / arms
def analyze_body(pts, wp, vis, w, h, f, legs=False):
    """pts: (33, 3) image points, wp: (33, 3) metres with the origin between the hips, vis: (33,) visibility."""
    ok = lambda *ids: all(vis[i] > VIS for i in ids)
    body = {"visible_points": int((vis > VIS).sum()), "_pts": pts, "_vis": vis, "_world": wp}

    # turn: direction of the shoulder line (and hip line) seen from above. 0 = facing the camera,
    # > 0 = turned to the person's right. z grows away from the camera.
    lines = [wp[11] - wp[12]] if ok(11, 12) else []
    if ok(23, 24):
        lines.append(wp[23] - wp[24])
    if lines:
        v = np.mean(lines, 0)
        body["turn"] = round(math.degrees(math.atan2(-v[2], v[0])), 1)
    # lean: torso tilt toward (+) or away from (-) the camera. MediaPipe's depth for the hips is biased,
    # so upright people read about +15-20; compare changes rather than trusting the absolute value.
    if ok(11, 12, 23, 24):
        t = (wp[11] + wp[12]) / 2 - (wp[23] + wp[24]) / 2
        body["lean"] = round(math.degrees(math.atan2(-t[2], -t[1])), 1)
        body["side_lean"] = round(math.degrees(math.atan2(t[0], -t[1])), 1)

    for side, j in LIMBS.items():
        arm = {}
        if ok(j["shoulder"], j["elbow"]):
            if ok(j["hip"]):  # 0 = arm down by the side, 90 = out level, 180 = straight up
                arm["shoulder_angle"] = round(_angle(wp[j["hip"]], wp[j["shoulder"]], wp[j["elbow"]]), 0)
            if ok(j["wrist"]):
                arm["elbow_angle"] = round(_angle(wp[j["shoulder"]], wp[j["elbow"]], wp[j["wrist"]]), 0)
                arm["raised"] = bool(pts[j["wrist"]][1] < pts[j["shoulder"]][1])
                # how far the wrist is in front of (+) or behind (-) the shoulder, toward the camera
                arm["reach_cm"] = round(float(wp[j["shoulder"]][2] - wp[j["wrist"]][2]) * 100, 1)
        if arm:
            body[f"{side}_arm"] = arm
        if legs and ok(j["hip"], j["knee"], j["ankle"]):
            body[f"{side}_leg"] = {"knee_angle": round(_angle(wp[j["hip"]], wp[j["knee"]], wp[j["ankle"]]), 0)}

    segs = [(a, b) for a, b in ((11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (12, 14)) if ok(a, b)]
    d = _distance_from_size(wp, pts, segs, f) if segs else None
    body["distance_cm"] = round(d, 1) if d else None
    return body


# ---------------------------------------------------------------- hands / fingers
def analyze_hand(pts, wp, side, score, w, h, f):
    """pts: (21, 3) image points, wp: (21, 3) metres, hand-centred."""
    fingers = {}
    for name, chain in FINGERS.items():
        # flexion inside the finger only (middle + tip joints; for the thumb: knuckle + tip joint),
        # so fingers spread sideways at the knuckles still count as straight
        joints = [wp[i] for i in chain]
        bends = [_bend(joints[i], joints[i + 1], joints[i + 2]) for i in range(len(joints) - 2)]
        curl = min(sum(bends) / (110 if name == "thumb" else 180), 1.0)
        if name == "thumb":  # a straight thumb tucked across the palm is still "closed"
            extended = curl < 0.4 and np.linalg.norm(wp[4] - wp[5]) > 0.6 * np.linalg.norm(wp[0] - wp[5])
        else:
            extended = curl < 0.4
        fingers[name] = {"extended": bool(extended), "curl": round(float(curl), 2),
                         "tip": [round(float(v), 1) for v in pts[chain[-1], :2]]}
    pattern = tuple(int(f_["extended"]) for f_ in fingers.values())
    gesture = GESTURES.get(pattern, "")
    if pattern == (1, 0, 0, 0, 0):
        gesture = "thumbs up" if pts[4][1] < pts[2][1] else "thumbs down"
    pinch_cm = float(np.linalg.norm(wp[4] - wp[8]) * 100)
    if pinch_cm < 3 and fingers["middle"]["extended"]:
        gesture = "OK"
    palm = [(0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (0, 9)]  # rigid palm bones
    d = _distance_from_size(wp, pts, palm, f)
    return {
        "side": side,
        "score": round(float(score), 2),
        **palm_facing(wp, side),
        "fingers": fingers,
        "fingers_up": int(sum(pattern)),
        "gesture": gesture,
        "pinch_cm": round(pinch_cm, 1),
        "distance_cm": round(d, 1) if d else None,
        "_pts": pts,
        "_world": wp,
    }


PHANTOM_REACH = 3.5  # body sizes (see phantom_hand)


def phantom_hand(hand, body):
    """A hand the hand model reports where no arm can be: its wrist further than PHANTOM_REACH body sizes (shoulder
    width, or twice the ear gap when side-on) from both of the body model's wrists. Turned away from the camera, the
    hand model found a "hand" in the hair at the back of the head; the rig then froze the head as if a hand covered
    the face and pulled the avatar's arm up to it. Real hands on the test takes: at most 2.7 (2.1-2.7 when the body
    model lost a hand near the face side-on); the phantom: 4.7-6.5."""
    if body is None:
        return False
    p = body["_pts"]
    size = max(float(np.linalg.norm(p[11, :2] - p[12, :2])), 2 * float(np.linalg.norm(p[7, :2] - p[8, :2])), 1.0)
    reach = min(float(np.linalg.norm(hand["_pts"][0, :2] - p[i, :2])) for i in (15, 16))
    return reach > PHANTOM_REACH * size


def palm_facing(wp, side):
    """Which side of the hand faces the camera, from the palm's surface normal.
    palm_angle: 0 = palm squarely toward the camera, 90 = edge-on, 180 = back of the hand toward the camera."""
    a, b = wp[5] - wp[0], wp[17] - wp[0]           # wrist -> index knuckle, wrist -> pinky knuckle
    n = np.cross(a, b) if side == "right" else np.cross(b, a)  # out of the palm (mirror-image for the left hand)
    n /= np.linalg.norm(n) + 1e-9
    angle = math.degrees(math.acos(float(np.clip(-n[2], -1, 1))))  # camera is toward -z
    facing = "palm" if angle < 70 else "back" if angle > 110 else "side"
    return {"facing": facing, "palm_angle": round(angle, 0)}


# ---------------------------------------------------------------- depth over time
class DepthTracker:
    """Smooths the distance estimate and turns it into speed / 'moving closer or away'."""

    def __init__(self, smooth_s=0.25):
        self.smooth_s, self.dist, self.vel, self.t, self.ref = smooth_s, None, 0.0, None, None

    def reset(self):
        self.ref = self.dist

    def __call__(self, dist, t, source):
        if dist is None:
            return None
        if self.dist is None or self.t is None:
            self.dist, self.t = dist, t
        else:
            dt = max(t - self.t, 1e-3)
            a = math.exp(-dt / self.smooth_s)
            new = a * self.dist + (1 - a) * dist
            self.vel = a * self.vel + (1 - a) * (new - self.dist) / dt
            self.dist, self.t = new, t
        if self.ref is None:
            self.ref = self.dist
        movement = "still"
        if self.vel < -8:
            movement = "moving closer"
        elif self.vel > 8:
            movement = "moving away"
        return {"distance_cm": round(self.dist, 1), "change_cm": round(self.dist - self.ref, 1),
                "speed_cm_s": round(self.vel, 1), "movement": movement, "source": source}


# ---------------------------------------------------------------- tracker
class AllInOneTracker:
    def __init__(self, video=True, max_faces=4, max_hands=2, pose_model="full", fov=65.0, ipd_cm=6.3,
                 upper_body=True, wholebody=False, gpu_tracking=False):
        """upper_body: skip legs (torso, arms, head only).
        wholebody: also run the GPU whole-body model (wholebody.py) to keep left/right right.
        gpu_tracking: run MediaPipe's body and face models on the GPU (gpu_tracking.py; video with one face, the
        full body model). Without a DirectML GPU they run on the CPU there; without its models, MediaPipe as usual."""
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        self.video, self.fov, self.ipd_cm = video, fov, ipd_cm
        self.upper_body = upper_body
        # the three MediaPipe models release the GIL, so running them side by side halves tracking time;
        # the GPU whole-body model runs alongside them on a fourth thread
        self.pool = ThreadPoolExecutor(5)  # face, hands, body, whole-body (GPU), spare
        self.wb, self._wb_box, self._wb_job, self._wb_last = None, None, None, None
        self._lm_args = (max_faces, max_hands, pose_model)
        self.gpu = None
        if gpu_tracking and video and max_faces == 1:
            try:
                from avatarstudio.tracking.gpu_tracking import GpuTracker
                self.gpu = GpuTracker(lambda: self._landmarkers(vision.RunningMode.IMAGE, 1, max_hands, "full"),
                                      max_hands)
            except Exception as e:  # no DirectML GPU, models missing, ...
                print(f"GPU tracking unavailable ({str(e).splitlines()[0][:120]}); tracking on the CPU", flush=True)
        if wholebody:
            self.set_wholebody(True)
        if self.gpu is None:
            self.face_lm, self.hand_lm, self.pose_lm = self._landmarkers(mode, *self._lm_args)
        self.depth = DepthTracker()
        self.body_scale = None  # learned eyes/body distance ratio
        self.eye_weight, self._depth_t = 0.0, None
        self._last_ts = -1

    @staticmethod
    def _landmarkers(mode, max_faces, max_hands, pose_model):
        face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL), running_mode=mode,
            num_faces=max_faces, output_face_blendshapes=True,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4))
        hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL), running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
        pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODELS[pose_model]), running_mode=mode,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4))
        return face_lm, hand_lm, pose_lm

    def _gpu_track(self, rgb, image, ts, face):
        """gpu_tracking.py's frame (it handles GPU errors itself, on the CPU). Anything else going wrong: the frame
        comes out empty, and after 3 in a row tracking moves to MediaPipe's own for good."""
        try:
            out = self.gpu(rgb, image, ts, face)
            self._gpu_fails = 0
            return out
        except Exception as e:
            self._gpu_fails = getattr(self, "_gpu_fails", 0) + 1
            if self._gpu_fails >= 3:
                print(f"GPU tracking failed ({str(e).splitlines()[0][:120]}); MediaPipe's tracking from now on",
                      flush=True)
                self.face_lm, self.hand_lm, self.pose_lm = self._landmarkers(vision.RunningMode.VIDEO, *self._lm_args)
                self.gpu = None
            from avatarstudio.tracking.gpu_tracking import Result
            return None, Result(pose_landmarks=[]), Result(hand_landmarks=[], hand_world_landmarks=[], handedness=[])

    def _submit_gpu(self, fn, *args):
        """Whole-body model work: with GPU tracking on, through its GPU thread (after the frame's own GPU work) -
        two DirectML sessions running at the same time on different threads hung the GPU."""
        return self.gpu.worker.submit(fn, *args, background=True) if self.gpu is not None else self.pool.submit(fn, *args)

    def set_optics(self, fov, ipd_cm):
        """A new field of view / eye spacing (the settings page) while tracking: distances so far (smoothed, the
        reference, the body estimate's ratio to the eyes) are rescaled to match, so nothing jumps. -> the factor
        distances changed by (the calibration's distance needs the same)."""
        k_f = math.tan(math.radians(self.fov) / 2) / math.tan(math.radians(fov) / 2)
        k = k_f * ipd_cm / self.ipd_cm
        self.fov, self.ipd_cm = fov, ipd_cm
        d = self.depth
        if d.dist is not None:
            d.dist, d.vel = d.dist * k, d.vel * k
        if d.ref is not None:
            d.ref *= k
        if self.body_scale is not None:  # (the body's size doesn't depend on the eye spacing)
            self.body_scale *= k / k_f
        return k

    def depth_source(self, face_d, body_d, yaw, t):
        """Combine the eye- and body-based distances without jumps when one of them comes or goes:
        - the eyes are ignored when the head is turned more than EYE_MAX_YAW (they read wrong there)
        - the body estimate is rescaled to agree with the eyes, re-learning the ratio while both are seen
        - switching between them crossfades over DEPTH_BLEND_S seconds instead of snapping.
        Tuned by replaying a recorded webcam session: the median jump at a switch fell from 10.6% to 1.8%."""
        if face_d and abs(yaw) > EYE_MAX_YAW:
            face_d = None
        if face_d and body_d:
            ratio = float(np.clip(face_d / body_d, 0.5, 2.0))
            self.body_scale = ratio if self.body_scale is None else 0.7 * self.body_scale + 0.3 * ratio
        body_s = body_d * self.body_scale if (body_d and self.body_scale) else body_d

        target = 1.0 if face_d else 0.0
        if not self.video or self._depth_t is None:
            self.eye_weight = target
        else:
            step = (t - self._depth_t) / DEPTH_BLEND_S
            self.eye_weight += float(np.clip(target - self.eye_weight, -step, step))
        self._depth_t = t

        if face_d and body_s:
            d = self.eye_weight * face_d + (1 - self.eye_weight) * body_s
        else:
            d, self.eye_weight = face_d or body_s, target
        if d is None:
            return None, None
        src = "eyes" if self.eye_weight > 0.99 else "body" if self.eye_weight < 0.01 else "eyes+body"
        return round(d, 1), src

    def set_wholebody(self, on, background=False):
        """Switch the GPU whole-body model on/off while running. Returns whether it is (or is being turned) on.
        background: load it on a helper thread (~0.6 s for the GPU session) - tracking starts without it and it
        joins in when ready; it's only needed when hands cross, so nothing waits for it."""
        if on and self.wb is None and not getattr(self, "_wb_loading", False):
            def load():
                try:
                    from avatarstudio.tracking.wholebody import WholeBody
                    wb = self._submit_gpu(WholeBody).result() if self.gpu is not None else WholeBody()
                    if self._wb_wanted:
                        self.wb = wb
                except Exception as e:  # model missing, no GPU runtime, ...: keep tracking without it
                    print(f"whole-body model unavailable ({e}); using MediaPipe only", flush=True)
                    self._wb_failed = True
                finally:
                    self._wb_loading = False
            self._wb_wanted, self._wb_loading, self._wb_failed = True, True, False
            if background:
                threading.Thread(target=load, daemon=True).start()
            else:
                load()
        elif not on:
            self._wb_wanted = False
            self.wb, self._wb_job, self._wb_last = None, None, None
        return on and not getattr(self, "_wb_failed", False)

    def _assign_sides(self, hands, body, rt=None):
        """Decide which hand is left / right so it stays right when hands cross or overlap.
        The hand model guesses left/right from the hand's look alone and can swap or duplicate the label
        when hands cross; the body models know which wrist belongs to which arm. Priority: whole-body model
        wrists (if running), MediaPipe body wrists, continuity with the previous frame, the hand model's label."""
        if not hands:
            self._prev_wrists = {}
            return
        wrist = lambda h: h["_pts"][0, :2]
        cands = {}
        source = "body"
        if rt is not None:
            for side, i in (("left", 9), ("right", 10)):
                if rt[1][i] > 0.5:
                    cands[side] = rt[0][i]
            source = "whole-body model"
        if len(cands) < len(hands) and body is not None:
            for side, i in (("left", 15), ("right", 16)):
                if body["_vis"][i] > 0.5 and side not in cands:
                    cands[side] = body["_pts"][i, :2]
                    source = "body"
        if len(cands) < len(hands) and getattr(self, "_prev_wrists", None):
            cands, source = {**self._prev_wrists, **cands}, "tracking"
        size = max(np.linalg.norm(h["_pts"][0, :2] - h["_pts"][9, :2]) for h in hands)
        if len(cands) >= len(hands):
            sides = list(cands)
            if len(hands) == 1:
                best = min(sides, key=lambda s: np.linalg.norm(wrist(hands[0]) - cands[s]))
                if np.linalg.norm(wrist(hands[0]) - cands[best]) < 2.5 * size:
                    hands[0]["side"], hands[0]["side_from"] = best, source
            else:
                a, b = hands[0], hands[1]
                keep = (np.linalg.norm(wrist(a) - cands["left"]) + np.linalg.norm(wrist(b) - cands["right"])
                        if {"left", "right"} <= set(cands) else None)
                swap = (np.linalg.norm(wrist(a) - cands["right"]) + np.linalg.norm(wrist(b) - cands["left"])
                        if keep is not None else None)
                if keep is not None and min(keep, swap) < 5 * size:
                    a["side"], b["side"] = ("left", "right") if keep <= swap else ("right", "left")
                    a["side_from"] = b["side_from"] = source
        if len(hands) == 2 and hands[0]["side"] == hands[1]["side"]:  # still a duplicate label: split by position
            hands.sort(key=lambda h: wrist(h)[0])
            hands[0]["side"], hands[1]["side"] = "right", "left"  # person's right hand is on the image left
        for h in hands:
            h.setdefault("side_from", "hand model")
            h.update(palm_facing(h["_world"], h["side"]))  # palm direction depends on which hand it is
        self._prev_wrists = {h["side"]: wrist(h) for h in hands}

    def _start_wholebody(self, frame_bgr, w, h, ts):
        """Run the GPU whole-body model for left/right. On a single photo, returns the job (the result is waited
        for). In video a result one frame old works just as well: returns the newest finished result (if at
        most WB_FRESH_MS old) and never makes the frame
        wait for the GPU. It then only runs while left/right is in doubt (_wholebody_needed): it shares the GPU
        with WiLoR, and running it all the time made WiLoR's fingers ~20 ms later."""
        wb = self.wb
        if wb is None:
            self._wb_job = self._wb_last = None
            return None
        box = self._wb_box or wb.box_from(None, w, h)
        if not self.video:  # (single photos have no previous frame)
            return self.pool.submit(self._safe_wholebody, wb, frame_bgr, box)
        job = self._wb_job
        if job is not None and job[0].done():
            self._wb_last = (job[0].result(), job[1])
            job = self._wb_job = None
        if self._wholebody_needed():
            self._wb_until = ts + WB_KEEP_MS
        if job is None and ts <= getattr(self, "_wb_until", -1) and (
                self.gpu is None or ts - getattr(self, "_wb_start", -WB_GPU_MS) >= WB_GPU_MS):
            self._wb_job, self._wb_start = (self._submit_gpu(self._safe_wholebody, wb, frame_bgr, box), ts), ts
        last = self._wb_last
        return last[0] if last is not None and ts - last[1] <= WB_FRESH_MS else None

    def _wholebody_needed(self):
        """Left/right is in doubt when two hands are close enough to cross, or the body model can't vouch for a
        hand: no visible body wrist of its side next to it, or both body wrists next to it (hands together -
        the body model confuses them there). From the previous frame."""
        hands, body = getattr(self, "_prev_hands", []), getattr(self, "_prev_body", None)
        if not hands:
            return False
        size = max(np.linalg.norm(h["_pts"][0, :2] - h["_pts"][9, :2]) for h in hands)
        if len(hands) == 2 and np.linalg.norm(hands[0]["_pts"][0, :2] - hands[1]["_pts"][0, :2]) < 3 * size:
            return True
        if body is None:
            return True
        for h in hands:
            near = {side for side, i in (("left", 15), ("right", 16)) if body["_vis"][i] >= 0.5 and
                    np.linalg.norm(body["_pts"][i, :2] - h["_pts"][0, :2]) <= 2.5 * size}
            if near != {h["side"]}:
                return True
        return False

    def _safe_wholebody(self, wb, frame_bgr, box):
        """The GPU can drop out (laptop graphics switching, driver reset): restart the model once, and if it
        keeps failing turn it off and carry on with MediaPipe alone instead of crashing."""
        try:
            out = wb(frame_bgr, box)
            self._wb_fails = 0
            return out
        except Exception as e:
            self._wb_fails = getattr(self, "_wb_fails", 0) + 1
            if self._wb_fails == 3:
                print(f"GPU whole-body model failed ({str(e).splitlines()[0][:120]}); restarting it", flush=True)
                try:
                    from avatarstudio.tracking.wholebody import WholeBody
                    self.wb = WholeBody()
                except Exception:
                    self.wb = None
            elif self._wb_fails >= 6:
                print("GPU whole-body model keeps failing: turned off (MediaPipe only)", flush=True)
                self.wb = None
            return None

    def _run(self, landmarker, image, ts):
        return landmarker.detect_for_video(image, ts) if self.video else landmarker.detect(image)

    def __call__(self, frame_bgr, timestamp_ms=None, face=True):
        """face=False skips the face model this frame (an extra camera only needs it while the main one has lost
        your face, see multicam.py)."""
        h, w = frame_bgr.shape[:2]
        f = focal_px(w, self.fov)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(timestamp_ms if timestamp_ms is not None else time.monotonic() * 1000)
        ts = self._last_ts = max(ts, self._last_ts + 1)
        if self.gpu is not None:
            fr, pr, hr = self._gpu_track(rgb, image, ts, face)
            rt = self._start_wholebody(frame_bgr, w, h, ts)  # (its GPU run queues behind this frame's)
        else:
            rt = self._start_wholebody(frame_bgr, w, h, ts)
            face_job = self.pool.submit(self._run, self.face_lm, image, ts) if face else None
            pose_job = self.pool.submit(self._run, self.pose_lm, image, ts)
            hr = self._run(self.hand_lm, image, ts)  # (on this thread, while the face and body run in the pool)
            fr = face_job.result() if face_job is not None else None
            pr = pose_job.result()
        if rt is not None and not isinstance(rt, tuple):  # a single photo waits for its result
            rt = rt.result()

        out = {"faces": [], "head": None, "body": None, "hands": [], "depth": None}

        # faces
        for i, lms in enumerate(fr.face_landmarks if fr is not None else []):
            bs = fr.face_blendshapes[i] if fr.face_blendshapes else None
            out["faces"].append(analyze_face(lms, bs, w, h, f, self.ipd_cm))

        # full body + arms; it also keeps the head position when no face is visible
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            pts, wp = _np(lms, w, h), _xyz(pr.pose_world_landmarks[0])
            vis = pr.visibility if isinstance(lms, np.ndarray) else \
                np.array([p.visibility or 0 for p in lms], dtype=np.float32)
            body = analyze_body(pts, wp, vis, w, h, f, legs=not self.upper_body)
            out["body"] = body
            seen = [i for i in (0, 2, 5, 7, 8) if body["_vis"][i] > VIS]  # nose, eyes, ears
            if seen and not out["faces"]:
                c = body["_pts"][seen, :2].mean(0)
                out["head"] = {"center": [round(float(c[0]), 1), round(float(c[1]), 1)], "source": "body pose"}
        main_face = max(out["faces"], key=lambda fc: fc["box"][2]) if out["faces"] else None
        if main_face:
            x, y, bw, bh = main_face["box"]
            out["head"] = {"center": [round(x + bw / 2, 1), round(y + bh / 2, 1)], "source": "face mesh"}

        # hands + fingers ("left"/"right" is the person's own hand, for an un-mirrored camera frame)
        hands = []
        for lms, world, hd in zip(hr.hand_landmarks, hr.hand_world_landmarks, hr.handedness):
            pts, wp = _np(lms, w, h), _xyz(world)
            hands.append(analyze_hand(pts, wp, hd[0].category_name.lower(), hd[0].score, w, h, f))
        for hand in sorted(hands, key=lambda h_: -h_["score"]):  # drop a duplicate of the same hand
            size = np.linalg.norm(hand["_pts"][0, :2] - hand["_pts"][9, :2])
            if all(np.linalg.norm(hand["_pts"][0, :2] - o["_pts"][0, :2]) > 0.5 * size for o in out["hands"]):
                out["hands"].append(hand)
        out["hands"] = [hand for hand in out["hands"] if not phantom_hand(hand, out["body"])]
        self._assign_sides(out["hands"], out["body"], rt)
        self._prev_hands, self._prev_body = out["hands"], out["body"]

        # whole-body model: next frame's person box from this frame (it runs ahead in parallel)
        if self.wb is not None:
            keep = rt[1] > 0.3 if rt is not None else None
            if rt is not None and keep[:23].sum() >= 3:
                self._wb_box = self.wb.box_from(rt[0][keep], w, h)
            elif out["body"] is not None:
                self._wb_box = self.wb.box_from(out["body"]["_pts"][out["body"]["_vis"] > 0.3, :2], w, h)
            else:
                self._wb_box = None
            out["wholebody"] = {"_pts": rt[0], "_scores": rt[1]} if rt is not None else None

        # depth: pupils are the most precise ruler; fall back to the body's size
        face_d = main_face["distance_cm"] if main_face else None
        body_d = out["body"]["distance_cm"] if out["body"] else None
        yaw = main_face["head_pose"]["yaw"] if main_face else 0.0
        raw, src = self.depth_source(face_d, body_d, yaw, ts / 1000)
        if self.video:
            out["depth"] = self.depth(raw, ts / 1000, src)
        elif raw:
            out["depth"] = {"distance_cm": raw, "source": src}
        return out


# ---------------------------------------------------------------- drawing
C = vision.FaceLandmarksConnections
FACE_PARTS = [C.FACE_LANDMARKS_FACE_OVAL, C.FACE_LANDMARKS_LIPS, C.FACE_LANDMARKS_LEFT_EYE,
              C.FACE_LANDMARKS_RIGHT_EYE, C.FACE_LANDMARKS_LEFT_EYEBROW, C.FACE_LANDMARKS_RIGHT_EYEBROW,
              C.FACE_LANDMARKS_NOSE]
BODY_LINKS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
              if c.start > 10 and c.end > 10]  # skip the face points; the face mesh covers those
UPPER_BODY_LINKS = [(a, b) for a, b in BODY_LINKS if a <= 24 and b <= 24]  # torso, arms, hands
BODY_LEFT, BODY_RIGHT = (255, 160, 60), (60, 160, 255)  # person's left / right side (BGR)
FINGER_COLORS = {"thumb": (0, 140, 255), "index": (0, 220, 255), "middle": (0, 255, 120),
                 "ring": (255, 200, 0), "pinky": (255, 80, 200)}


def _lines(img, pts, conns, color, t=1):
    for a, b in conns:
        cv2.line(img, tuple(map(int, pts[a][:2])), tuple(map(int, pts[b][:2])), color, t, cv2.LINE_AA)


def _draw_body(frame, body):
    pts, vis = body["_pts"], body["_vis"]
    legs = any(k.endswith("_leg") for k in body)
    for a, b in BODY_LINKS if legs else UPPER_BODY_LINKS:
        if vis[a] > VIS and vis[b] > VIS:
            col = BODY_LEFT if a % 2 and b % 2 else BODY_RIGHT if not a % 2 and not b % 2 else (220, 220, 220)
            _lines(frame, pts, [(a, b)], col, 3)
    for i in range(11, 33 if legs else 25):
        if vis[i] > VIS:
            cv2.circle(frame, tuple(map(int, pts[i][:2])), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_hud(frame, res, mirror):
    """Depth / body panel with a top-down view, in the top-right corner."""
    h, w = frame.shape[:2]
    pw, ph = 300, 320
    x0, y0 = w - pw - 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def line(s, i, col=(235, 235, 235)):
        cv2.putText(frame, s, (x0 + 10, y0 + 22 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)

    d, body = res["depth"], res["body"] or {}
    i = 0
    if d:
        line(f"Distance  {d['distance_cm']:.0f} cm  ({d['source']})", i, (120, 255, 255)); i += 1
        if "change_cm" in d:
            col = (120, 255, 120) if d["movement"] == "moving closer" else \
                  (120, 160, 255) if d["movement"] == "moving away" else (235, 235, 235)
            line(f"{d['movement']}  {d['change_cm']:+.0f} cm since start", i, col); i += 1
    else:
        line("Distance  --", i); i += 1
    if "turn" in body:
        line(f"Body turn {body['turn']:+.0f} deg", i); i += 1
    for side in ("left", "right"):
        arm = body.get(f"{side}_arm")
        if arm and "elbow_angle" in arm:
            up = " UP" if arm["raised"] else ""
            line(f"{side[0].upper()} arm: elbow {arm['elbow_angle']:.0f}  reach {arm['reach_cm']:+.0f}cm{up}", i); i += 1
    for hand in res["hands"]:
        if hand["distance_cm"]:
            line(f"{hand['side']} hand {hand['distance_cm']:.0f} cm", i); i += 1

    # top-down map: camera at the bottom, you further up the further away you are
    mx0, my0, mw, mh = x0 + 10, y0 + 22 + 19 * i, pw - 20, ph - (22 + 19 * i) - 10
    if mh < 60:
        return
    cv2.rectangle(frame, (mx0, my0), (mx0 + mw, my0 + mh), (90, 90, 90), 1)
    cam = (mx0 + mw // 2, my0 + mh - 6)
    cv2.drawMarker(frame, cam, (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 12, 2)
    cv2.putText(frame, "camera", (cam[0] + 10, cam[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
    near, far = 25.0, 250.0
    for cm in (50, 100, 150, 200):
        yy = int(cam[1] - (cm - near) / (far - near) * (mh - 20))
        cv2.line(frame, (mx0 + 2, yy), (mx0 + 8, yy), (110, 110, 110), 1)
        cv2.putText(frame, f"{cm}", (mx0 + 11, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1)
    if not d:
        return
    py = int(cam[1] - (np.clip(d["distance_cm"], near, far) - near) / (far - near) * (mh - 20))
    p = np.array([cam[0], py], dtype=np.float32)
    sx = 1 if mirror else -1  # in the mirrored view, turning to your right points to screen-right
    if "turn" in body:
        t = math.radians(body["turn"])
        half = 22 * np.array([math.cos(t), -sx * math.sin(t)], dtype=np.float32)
        cv2.line(frame, tuple((p - half).astype(int)), tuple((p + half).astype(int)), (255, 160, 60), 3, cv2.LINE_AA)
    cv2.circle(frame, tuple(p.astype(int)), 8, (230, 230, 230), -1, cv2.LINE_AA)
    if res["faces"]:  # nose direction from head yaw; facing the camera points down the map
        yaw = math.radians(max(res["faces"], key=lambda fc: fc["box"][2])["head_pose"]["yaw"])
        tip = p + 26 * np.array([sx * math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        cv2.arrowedLine(frame, tuple(p.astype(int)), tuple(tip.astype(int)), (255, 120, 0), 2, cv2.LINE_AA, tipLength=0.35)


def draw(frame, res, mirror=False, mesh=True, body=True, hud=True):
    """Draw shapes on the original frame, optionally mirror it, then put text on top."""
    h, w = frame.shape[:2]
    if body and res["body"]:
        _draw_body(frame, res["body"])
    for face in res["faces"]:
        pts = face["_pts"]
        if mesh:
            for part in FACE_PARTS:
                _lines(frame, pts, [(c.start, c.end) for c in part], (80, 220, 80))
            for i in (468, 473):
                cv2.circle(frame, tuple(map(int, pts[i][:2])), 3, (0, 255, 255), -1, cv2.LINE_AA)
        # head direction arrow from the nose tip
        yaw, pitch = face["head_pose"]["yaw"], face["head_pose"]["pitch"]
        nose = pts[1][:2]
        length = face["box"][2] * 0.6
        tip = nose + length * np.array([-math.sin(math.radians(yaw)), -math.sin(math.radians(pitch))])
        cv2.arrowedLine(frame, tuple(map(int, nose)), tuple(map(int, tip)), (255, 120, 0), 3, cv2.LINE_AA, tipLength=0.2)
    for hand in res["hands"]:
        pts = hand["_pts"]
        _lines(frame, pts, [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)], (230, 230, 230), 2)
        for name, chain in FINGERS.items():
            col = FINGER_COLORS[name] if hand["fingers"][name]["extended"] else (110, 110, 110)
            _lines(frame, pts, list(zip(chain[:-1], chain[1:])), col, 3)
            cv2.circle(frame, tuple(map(int, pts[chain[-1]][:2])), 5, col, -1, cv2.LINE_AA)

    if mirror:
        frame = cv2.flip(frame, 1)
    fx = (lambda x: w - x) if mirror else (lambda x: x)

    def text(s, x, y, col=(255, 255, 255), scale=0.6):
        x = int(min(max(x, 4), w - 10 * len(s) * scale))
        y = int(min(max(y, 16), h - 6))
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)

    for face in res["faces"]:
        x, y, bw, bh = face["box"]
        left = min(fx(x), fx(x + bw))
        hp = face["head_pose"]
        lines = [f"yaw {hp['yaw']:+.0f}  pitch {hp['pitch']:+.0f}  roll {hp['roll']:+.0f}"]
        if face.get("eyes_open"):
            eo = face["eyes_open"].values()
            eyes = "open" if all(eo) else ("closed" if not any(eo) else "wink")
            lines.append(f"eyes {eyes}  mouth {face['mouth_open']:.0%}  smile {face['smile']:.0%}")
        for i, s in enumerate(lines):
            text(s, left, y + bh + 22 + 20 * i, scale=0.5)
    if not res["faces"] and res["head"]:
        cx, cy = res["head"]["center"]
        text("head (face not visible)", fx(cx) - 80, cy, (255, 180, 0))
    for hand in res["hands"]:
        wx, wy = hand["_pts"][0][:2]
        label = f"{hand['side']} hand ({hand['facing']}): {hand['fingers_up']} up"
        if hand["gesture"]:
            label += f", {hand['gesture']}"
        text(label, fx(wx) - 60, wy + 25, (0, 220, 255), 0.55)
    if hud:
        _draw_hud(frame, res, mirror)
    return frame


def public(res):
    """Result without the raw landmark arrays (for printing / JSON logging)."""
    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")} if d else d
    return {"faces": [clean(f) for f in res["faces"]], "head": res["head"], "body": clean(res["body"]),
            "wholebody": res.get("wholebody") is not None,
            "hands": [clean(h) for h in res["hands"]], "depth": res["depth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pose", choices=list(POSE_MODELS), default="full", help="body model: full (accurate) or lite (faster)")
    ap.add_argument("--fov", type=float, default=65.0, help="camera horizontal field of view in degrees (for distance)")
    ap.add_argument("--ipd", type=float, default=6.3, help="your pupil-to-pupil distance in cm (for distance)")
    ap.add_argument("--no-mirror", action="store_true", help="webcam: don't show a selfie-style mirrored view")
    ap.add_argument("--log", help="webcam: write per-frame tracking data to this .jsonl file")
    ap.add_argument("--full-body", action="store_true", help="also track legs (default: torso, arms and head)")
    args = ap.parse_args()
    opts = dict(pose_model=args.pose, fov=args.fov, ipd_cm=args.ipd,
                upper_body=not args.full_body)

    if args.webcam:
        tracker = AllInOneTracker(video=True, **opts)
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        log = open(args.log, "w") if args.log else None
        fps, t0, show = 0.0, time.time(), {"mesh": True, "body": True}
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from the camera (try --camera 1)")
                break
            res = tracker(frame)
            if log:
                log.write(json.dumps({"t": round(time.time(), 3), **public(res)}) + "\n")
            frame = draw(frame, res, mirror=not args.no_mirror, **show)
            t1 = time.time()
            fps, t0 = 0.9 * fps + 0.1 / max(t1 - t0, 1e-6), t1
            cv2.putText(frame, f"{fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("All-in-one tracker (q quit, m mesh, b body, r reset depth)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show["mesh"] = not show["mesh"]
            if key == ord("b"):
                show["body"] = not show["body"]
            if key == ord("r"):
                tracker.depth.reset()
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        res = AllInOneTracker(video=False, **opts)(frame)
        out = os.path.splitext(args.image)[0] + "_tracked.jpg"
        cv2.imwrite(out, draw(frame, res))
        print(json.dumps(public(res), indent=1))
        print(f"-> {out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
