"""Experimental: a 2nd and 3rd webcam (e.g. left and right of the main, centre one).

The main camera stays in charge of everything that is measured in its picture (face position, distance, the
studio window, WiLoR's finger crops). The extra cameras each run their own tracker alongside it, and what they see
is merged into the main camera's 3D view:
  - body: each of the 33 body points is averaged over the cameras that see it, weighted by how clearly they do
    (so an arm hidden behind the body from the front, or a hand turned away, comes from a side camera)
  - hands: a hand seen by several cameras is averaged (cameras looking at the palm or the back of the hand count
    more than one seeing it edge-on); a hand only a side camera sees is added
  - face: if the main camera loses your face (turned far away from it), a side camera's face is used for the head
Lining the cameras up needs no checkerboard: MediaPipe's 3D body points are centred on the hips in every camera,
so the cameras differ only by a rotation, found (Kabsch) from the body points both see clearly during the normal
neutral calibration (C). Until then the extra cameras are shown but not used.

Frames are not synchronised (each camera runs on its own clock; the newest frame of each is used), so fast
motion can disagree by up to one frame between cameras. studio.py --record saves each extra camera's video in step
with the main one, frame for frame (take_camera2.mp4, ...): mocap.py merges them offline, lined up from the take.
"""
import copy
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from avatarstudio.tracking.all_in_one import AllInOneTracker
from avatarstudio.tracking.camera import CameraStream

ALIGN_POINTS = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24)  # nose, ears, shoulders, elbows, wrists, hips
ALIGN_VIS = 0.7      # both cameras must see a point at least this clearly to use it for lining up
FUSE_VIS = 0.5       # a camera's body point counts in the merge from this visibility on
MAX_ALIGN_CM = 8.0   # mean leftover distance after lining up above which a camera is not used
SWAP = {7: 8, 8: 7, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23}  # left <-> right body points
MIRROR_BLOCK = 15    # frames per stretch the mirror test compares (0.5 s at 30 fps) ...
MIRROR_BLOCKS = 6    # ... it needs at least this many (the C calibration's 20 frames are too few: not tested)
MIRROR_T = -2.0      # mirrored when the mirror image fits better by this many standard errors


def kabsch(a, b):
    """Rotation r with r @ a_i ~ b_i (a, b: N x 3, each centred on its own mean) and whether a mirror image fits
    better (a camera whose picture is mirrored can't be lined up by a rotation)."""
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, d < 0


def _leftover(a, b, det):
    """Per point: distance (m) left after the best rotation (det 1) or mirror image (det -1) takes a onto b."""
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ np.diag([1.0, 1.0, det * float(np.sign(np.linalg.det(vt.T @ u.T)))]) @ u.T
    return np.linalg.norm(a @ r.T - b, axis=1)


def mirror_test(frames):
    """Is this camera's picture mirrored? On a mirrored picture the body model also swaps which side it calls left,
    so a mirrored camera's body is the main camera's mirror image with left and right swapped - which a rotation
    fits nearly as well while you hold a symmetric pose: only one-sided moves tell them apart. frames: per frame,
    (main camera's points, this camera's, this camera's with left and right swapped). Both explanations are fitted
    over all frames and compared stretch by stretch (MIRROR_BLOCK frames). Returns (rotation fit cm, mirror fit cm,
    t: mirror minus rotation fit per stretch in standard errors, negative = the mirror image fits better), or None
    with too few frames."""
    nb = len(frames) // MIRROR_BLOCK
    if nb < MIRROR_BLOCKS:
        return None
    b, a, s = (np.concatenate([f[k] for f in frames]).astype(float) for k in range(3))
    starts = np.cumsum([0] + [len(f[0]) for f in frames[:-1]])
    count = np.array([len(f[0]) for f in frames])
    rot, mir = (np.add.reduceat(e, starts) / count for e in (_leftover(a, b, 1.0), _leftover(s, b, -1.0)))
    diff = (mir - rot)[:nb * MIRROR_BLOCK].reshape(nb, MIRROR_BLOCK).mean(1)
    t = diff.mean() / max(diff.std(ddof=1) / math.sqrt(nb), 1e-9)
    return float(rot.mean() * 100), float(mir.mean() * 100), float(t)


def body_weight(vis):
    """How much each of a camera's body points counts in the merge (0 where it doesn't see the point clearly)."""
    return np.where(vis > FUSE_VIS, vis * vis, 0.0)


def _hand_weight(hand):
    """How much a camera's view of a hand counts: its confidence, more when the camera sees palm or back than
    edge-on (fingers overlap edge-on)."""
    return float(hand.get("score", 0.5)) * max(0.2, abs(math.cos(math.radians(hand.get("palm_angle", 90.0)))))


class ExtraCamera:
    """One extra webcam (or a video file standing in for one) with its own tracker."""

    def __init__(self, name, index=None, size=(1280, 720), fps=30, video=None):
        self.name, self.R, self.error_cm = name, None, None
        self.cam = CameraStream(index, *size, fps=fps) if video is None else None
        self.video = cv2.VideoCapture(video) if video is not None else None
        if (self.cam is not None and not self.cam.ok) or (self.video is not None and not self.video.isOpened()):
            print(f"{name}: {'camera ' + str(index) if video is None else video} not found "
                  f"(it keeps trying if it's a camera)", flush=True)
        self.tracker = AllInOneTracker(video=True, max_faces=1)
        self.frame, self.res, self._frame_id = None, None, 0

    def step(self, ts_ms, face=True):
        """Track this camera's newest frame (or keep the last result if no new frame came). face=False skips its
        face model (only needed while the main camera has lost your face)."""
        if self.video is not None:
            ok, frame = self.video.read()
        else:
            ok, frame, fid = self.cam.read(self._frame_id, timeout=0.0)
            if ok:
                self._frame_id = fid
        if not ok:
            return self.res
        self.frame = frame
        self.res = self.tracker(frame, timestamp_ms=ts_ms, face=face)
        return self.res

    def release(self):
        if self.cam is not None:
            self.cam.release()
        if self.video is not None:
            self.video.release()


class MultiCam:
    def __init__(self, extras):
        self.extras = extras
        self.pool = ThreadPoolExecutor(len(extras))
        self._calib, self.calib_report = None, {}
        self.main_had_face = False  # (last frame) - while it has, the extra cameras skip their face model

    @property
    def aligned(self):
        return [e for e in self.extras if e.R is not None]

    def submit(self, ts_ms):
        """Start every extra camera's tracking (runs alongside the main camera's). Their face model only runs
        while the main camera has lost your face (it's only used then): ~7 ms of CPU less per camera and frame."""
        face = not self.main_had_face or self._calib is not None
        return [self.pool.submit(e.step, ts_ms, face) for e in self.extras]

    @staticmethod
    def collect(jobs):
        out = []
        for j in jobs:
            try:
                out.append(j.result())
            except Exception as e:  # one camera's hiccup must not stop the others
                print(f"extra camera: {e}", flush=True)
                out.append(None)
        return out

    def set_alignment(self, cams):
        """cams: calibration["cameras"] = {name: {"R": 3x3, "error_cm": ...}}."""
        for e in self.extras:
            c = (cams or {}).get(e.name)
            e.R = np.array(c["R"], float) if c and c.get("R") is not None else None
            e.error_cm = c.get("error_cm") if c else None

    # ---------------------------------------------------------------- lining up (during the C calibration)
    def calib_start(self):
        self._calib = {e.name: ([], [], []) for e in self.extras}

    def calib_add(self, main_res, extra_res):
        if self._calib is None or main_res["body"] is None:
            return
        mb = main_res["body"]
        for e, r in zip(self.extras, extra_res):
            if r is None or r["body"] is None:
                continue
            eb = r["body"]
            ids = [i for i in ALIGN_POINTS if mb["_vis"][i] > ALIGN_VIS and eb["_vis"][i] > ALIGN_VIS]
            if len(ids) >= 4:
                self._calib[e.name][0].append(eb["_world"][ids])
                self._calib[e.name][1].append(mb["_world"][ids])
            # for the mirror test: points this camera sees on both sides, as they are and left / right swapped
            ids = [i for i in ids if eb["_vis"][SWAP.get(i, i)] > ALIGN_VIS]
            if len(ids) >= 4:
                self._calib[e.name][2].append((mb["_world"][ids], eb["_world"][ids],
                                               eb["_world"][[SWAP.get(i, i) for i in ids]]))

    def calib_finish(self):
        """{name: {"R", "error_cm", "angle_deg"}} for each camera that could be lined up; prints what happened
        (also kept in self.calib_report: {name: {"frames", "error_cm", "angle_deg", "mirror_fit_cm", "mirror_t",
        "mirrored", "note"}}). Mirrored: a mirror image fits better than a rotation, as the points are or (long
        enough, e.g. a whole take) with left and right swapped, see mirror_test."""
        out, self.calib_report = {}, {}
        for e in self.extras:
            src, dst, both = self._calib.get(e.name, ([],) * 3) if self._calib else ([],) * 3
            report = self.calib_report[e.name] = {"frames": len(src)}
            if len(src) < 5:
                report["note"] = (f"{e.name}: couldn't see your body together with the main camera - not lined up "
                                  f"(both cameras need to see your shoulders, arms and head during C)")
                print(report["note"], flush=True)
                continue
            a, b = np.concatenate(src).astype(float), np.concatenate(dst).astype(float)
            r, mirrored = kabsch(a, b)
            err = float(np.linalg.norm((a - a.mean(0)) @ r.T - (b - b.mean(0)), axis=1).mean() * 100)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))
            test = mirror_test(both)
            if test is not None:
                report.update(mirror_fit_cm=round(test[1], 1), mirror_t=round(test[2], 1))
                mirrored = mirrored or test[2] < MIRROR_T
            report.update(error_cm=round(err, 1), angle_deg=round(angle, 1), mirrored=bool(mirrored))
            if mirrored:
                report["note"] = (f"{e.name}: its picture looks mirrored (turn mirroring off in the camera's own "
                                  f"settings) - not used")
            elif err > MAX_ALIGN_CM:
                report["note"] = (f"{e.name}: lined up only roughly ({err:.1f} cm off) - not used; it needs a clear "
                                  f"view of your upper body")
            else:
                out[e.name] = {"R": r.tolist(), "error_cm": round(err, 1), "angle_deg": round(angle, 1)}
                report["note"] = f"{e.name}: lined up, {angle:.0f} deg from the main camera ({err:.1f} cm average fit)"
            print(report["note"], flush=True)
        self._calib = None
        return out

    # ---------------------------------------------------------------- merging
    def fuse(self, res, extra_res):
        """The main camera's result with the aligned extra cameras' views merged in (see the module docstring).
        Items that come only from an extra camera carry "_cam" (their picture positions belong to that camera)."""
        self.main_had_face = bool(res["faces"])
        views = [(e, r) for e, r in zip(self.extras, extra_res) if e.R is not None and r is not None]
        if not views:
            return res
        res = dict(res)
        # body
        bodies = [(e, r["body"]) for e, r in views if r["body"] is not None]
        if bodies:
            if res["body"] is not None:
                main = res["body"]
                worlds = [main["_world"]] + [b["_world"] @ e.R.T for e, b in bodies]
                vis = [main["_vis"]] + [b["_vis"] for _, b in bodies]
                w = np.array([body_weight(v) for v in vis])  # cameras x 33
                total = w.sum(0)
                fused = np.einsum("cp,cpk->pk", w, np.array(worlds)) / np.maximum(total, 1e-9)[:, None]
                body = dict(main)
                body["_world"] = np.where(total[:, None] > 0, fused, main["_world"]).astype(np.float32)
                body["_vis"] = np.max(vis, axis=0).astype(np.float32)
                res["body"] = body
            else:  # the main camera lost you: the side camera that sees most of you
                e, b = max(bodies, key=lambda eb: float((eb[1]["_vis"] > FUSE_VIS).sum()))
                body = dict(b)
                body["_world"] = (b["_world"] @ e.R.T).astype(np.float32)
                body["_cam"] = e.name
                res["body"] = body
        # hands
        hands = [dict(h) for h in res["hands"]]
        by_side = {h["side"]: h for h in hands}
        extra_hands = {}
        for e, r in views:
            for h in r["hands"]:
                extra_hands.setdefault(h["side"], []).append((e, h))
        for side, seen in extra_hands.items():
            if side in by_side:
                main = by_side[side]
                ws = [_hand_weight(main)] + [_hand_weight(h) for _, h in seen]
                pts = [main["_world"]] + [h["_world"] @ e.R.T for e, h in seen]
                main["_world"] = (np.tensordot(np.array(ws), np.array(pts), 1) / max(sum(ws), 1e-9)).astype(np.float32)
            else:
                e, h = max(seen, key=lambda eh: _hand_weight(eh[1]))
                h = dict(h)
                h["_world"] = (h["_world"] @ e.R.T).astype(np.float32)
                h["_cam"] = e.name
                hands.append(h)
        res["hands"] = hands
        # face: only when the main camera has none
        if not res["faces"]:
            faces = [(e, f) for e, r in views for f in r["faces"]]
            if faces:
                e, f = max(faces, key=lambda ef: ef[1]["box"][2])
                f = copy.copy(f)
                f["_R"], f["_cam"] = e.R, e.name
                res["faces"] = [f]
        return res

    def release(self):
        for e in self.extras:
            try:
                e.release()
            except Exception:
                pass
        self.pool.shutdown(wait=False)
