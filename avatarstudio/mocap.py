"""Offline motion capture from a video: what FreeMoCap does after a recording, for one camera (or two or three).

  .venv\\Scripts\\python.exe app.py mocap take.mp4               -> take_mocap\\ next to the video
  AvatarStudio.exe mocap take.mp4 [--out DIR] [--cutoff 6] [--full-body] [--no-wilor] [--no-overlay]
                                  [--video2 V] [--video3 V] [--main-only]

Live tracking only knows the past: it smooths with a delay and guesses ahead. Here the whole take is known:
- every frame is tracked, none dropped, and WiLoR's finger shapes are waited for on every frame (live, it skips)
- the avatar pose is solved without live smoothing or look-ahead, then filtered forward and backward in time
  (4th-order Butterworth at 6 Hz, as FreeMoCap does; the face at 12 Hz so blinks stay): no delay, no overshoot
- body points lost for up to 0.5 s are filled in (straight line between the frames either side), longer gaps
  stay empty
Writes into the output folder:
  motion.bvh          the avatar's motion for Blender / MotionBuilder / Unity (bone lengths stay the same)
  body_3d.csv, .npy   the 33 body points per frame, metres, origin between the hips: X your right, Y forward
                      (toward the camera), Z up; root_* is where you moved from your calibrated spot
  joint_angles.csv    elbows, shoulders, hips, knees (from the body points), trunk lean, head turn / look up /
                      tilt (against your calibrated neutral pose, C in the studio), wrists and fingers (from the
                      avatar pose), in degrees; empty where that part wasn't tracked
  face_arkit.csv      the 52 ARKit face shapes per frame (0..1): facial animation in Blender, Unreal, Maya
  summary.json        what was tracked, gaps filled, jitter before / after, every angle's range of motion
  overlay.mp4         the video with the cleaned-up skeleton drawn on it (orange: filled-in points)
One webcam judges depth poorly: angles across its view are good, toward / away from it much less so.
FreeMoCap's accuracy figures (joint paths within 3 cm, angles within 5 degrees) come from 3-6 calibrated cameras.
More cameras: studio.py --record take.mp4 --camera2 1 [--camera3 2] also writes take_camera2.mp4 / take_camera3.mp4,
in step frame for frame; mocap.py picks them up from next to the video. Every frame is tracked in every camera, the
cameras are lined up from the take itself (multicam.py: a rotation fitted to the body points both see clearly), then
each frame's body, hands and face are merged into the main camera's view before the avatar pose is solved. WiLoR's
fingers, the overlay and all axes stay the main camera's. summary.json says how well each camera lined up and how
much of the body came from the side cameras. This is merging, not FreeMoCap's triangulation of calibrated cameras:
it fills in what the main camera couldn't see clearly, but each camera's depth is still the body model's guess."""
import argparse
import csv
import json
import math
import os
import time

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from avatarstudio.tracking.all_in_one import AllInOneTracker, focal_px
from avatarstudio.personal.face_refiner import load as load_face_refiner
from avatarstudio.core.paths import load_json, user_file
from avatarstudio.avatar.rig import IDENTITY, AvatarRig, forward_kinematics
from avatarstudio.core.settings import Settings

BODY_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
              "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
              "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
              "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee",
              "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index",
              "right_foot_index"]
LEGS = range(25, 33)
BODY_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
              (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32), (15, 19), (16, 20), (0, 7), (0, 8)]
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
JOINTS = {"left": dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
          "right": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28)}
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
VIS = 0.5


# ---------------------------------------------------------------- clean-up
def fill_gaps(x, max_gap):
    """Rows of NaN (per column) up to max_gap long, with data on both sides, become a straight line between them.
    Returns the filled array and a mask of what was filled."""
    x = np.array(x, float)
    flat = x.reshape(len(x), -1)
    filled = np.zeros(flat.shape, bool)
    for c in range(flat.shape[1]):
        bad = ~np.isfinite(flat[:, c])
        if not bad.any() or bad.all():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
        for s, e in zip(edges[::2], edges[1::2]):  # gap = rows s..e-1
            if s > 0 and e < len(flat) and e - s <= max_gap:
                flat[s:e, c] = np.interp(np.arange(s, e), [s - 1, e], [flat[s - 1, c], flat[e, c]])
                filled[s:e, c] = True
    return flat.reshape(x.shape), filled.reshape(x.shape)


def zero_lag(x, fps, cutoff):
    """4th-order Butterworth low-pass run forward and backward (no delay), on each stretch without gaps."""
    x = np.array(x, float)
    if not cutoff or cutoff >= 0.45 * fps:
        return x
    b, a = butter(4, cutoff / (fps / 2))
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        ok = np.isfinite(flat[:, c])
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s > 15:  # (the filter needs some frames to settle at each end)
                flat[s:e, c] = filtfilt(b, a, flat[s:e, c])
    return flat.reshape(x.shape)


def filter_rotations(q, fps, cutoff):
    """q: (frames, bones, 4) quaternions. Each kept on one side (q and -q are the same turn), filtered, renormalised."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        flip = np.sum(q[i] * q[i - 1], axis=1) < 0
        q[i, flip] *= -1
    q = zero_lag(q, fps, cutoff)
    return q / np.linalg.norm(q, axis=2, keepdims=True)


def jitter_mm(p, fps):
    """How much points shake: median size of the frame-to-frame change in velocity (mm per frame), body points."""
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1) * 1000
    acc = acc[np.isfinite(acc)]
    return round(float(np.median(acc)), 2) if acc.size else None


# ---------------------------------------------------------------- angles
def _angle(a, b, c):
    """Angle at b (degrees); NaN where a point is missing."""
    u, v = a - b, c - b
    cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _turn(q):
    """Size of a local rotation (degrees); q is x, y, z, w."""
    return math.degrees(2 * math.acos(min(1.0, abs(float(q[3])))))


def joint_angles(body, bones, names, seen):
    """{column: (frames,) degrees}. body: (frames, 33, 3) cleaned body points (X right, Y forward, Z up);
    bones: (frames, bones, 4) filtered avatar rotations (local, as the rig sends them); seen: {"left" / "right" /
    "body" / "head": (frames,) bool} - where the avatar only holds or relaxes a part it lost, its angles are left
    empty."""
    out = {}
    for side, j in JOINTS.items():
        sh, el, wr, hip, kn, an = (body[:, j[k]] for k in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
        # shoulder against the trunk (toward the hip); straight down where the hips aren't seen (sitting at a desk)
        down = np.where(np.isfinite(hip).all(1, keepdims=True), hip, sh - np.array([0, 0, 1.0]))
        out[f"{side}_elbow_flexion"] = 180 - _angle(sh, el, wr)          # 0 straight
        out[f"{side}_shoulder_elevation"] = _angle(down, sh, el)         # 0 arm down, 90 level, 180 up
        out[f"{side}_hip_flexion"] = 180 - _angle(sh, hip, kn)          # 0 standing straight
        out[f"{side}_knee_flexion"] = 180 - _angle(hip, kn, an)          # 0 straight
    idx = {n: i for i, n in enumerate(names)}
    rows = {k: [] for k in ("trunk_lean_forward", "trunk_lean_right", "head_turn_right", "head_look_up",
                            "head_tilt_right")}
    side_rows = {f"{s}_{p}": [] for s in ("left", "right") for p in ["wrist_bend"] + [f"{f.lower()}_flex" for f in FINGERS]}
    for fr in bones:
        local = {n: fr[i] for n, i in idx.items()}
        _, rot = forward_kinematics(local, only=("Head",))
        up = rot["Chest"] @ np.array([0, 1.0, 0])  # avatar axes: x your right, y up, z forward
        rows["trunk_lean_forward"].append(math.degrees(math.atan2(up[2], up[1])))
        rows["trunk_lean_right"].append(math.degrees(math.atan2(up[0], up[1])))
        rel = rot["Chest"].T @ rot["Head"]  # the head against the chest: what the neck does
        fwd, right = rel @ np.array([0, 0, 1.0]), rel @ np.array([1.0, 0, 0])
        rows["head_turn_right"].append(math.degrees(math.atan2(fwd[0], fwd[2])))
        rows["head_look_up"].append(math.degrees(math.asin(np.clip(fwd[1], -1, 1))))
        rows["head_tilt_right"].append(math.degrees(math.asin(np.clip(-right[1], -1, 1))))
        for s, S in (("left", "Left"), ("right", "Right")):
            side_rows[f"{s}_wrist_bend"].append(_turn(local.get(f"{S}Hand", IDENTITY)))
            for f in FINGERS:  # the three joints together: 0 straight
                side_rows[f"{s}_{f.lower()}_flex"].append(sum(_turn(local.get(f"{S}{f}{seg}", IDENTITY))
                                                              for seg in ("Proximal", "Intermediate", "Distal")))
    for k, v in {**rows, **side_rows}.items():
        part = k.split("_")[0]
        out[k] = np.where(seen[part], np.array(v), np.nan)
    return out


def repetitions(v, fps, min_swing=30.0, min_s=0.4):
    """Exercise repetitions of one angle: peaks that rise at least min_swing degrees above the dips either side,
    at least min_s apart (gaps split the count)."""
    from scipy.signal import find_peaks
    count = 0
    ok = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        count += len(find_peaks(v[s:e], prominence=min_swing, distance=max(1, int(min_s * fps)))[0])
    return count


# ---------------------------------------------------------------- writing
def write_csv(path, header, cols, fps, decimals=4):
    """Columns frame, time_s, then header's: cols is a list of (frames,) arrays; NaN -> empty cell."""
    data = np.column_stack(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + list(header))
        for k, row in enumerate(data):
            w.writerow([k, f"{k / fps:.3f}"] + ["" if not np.isfinite(v) else f"{v:.{decimals}f}" for v in row])


def draw_overlay(video, out_path, fps, body2d, body_filled, hands2d, hands_filled):
    cap = cv2.VideoCapture(video)
    writer = None
    k = 0
    while k < len(body2d):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        for pts, filled, lines, width in [(body2d[k], body_filled[k], BODY_LINES, 3)] + \
                [(hands2d[s][k], hands_filled[s][k], HAND_LINES, 2) for s in hands2d]:
            for a, b in lines:
                if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
                    color = (0, 160, 255) if filled[a] or filled[b] else (80, 220, 80)
                    cv2.line(frame, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, width,
                             cv2.LINE_AA)
        cv2.putText(frame, f"{k / fps:6.2f} s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        k += 1
    cap.release()
    if writer is not None:
        writer.release()


# ---------------------------------------------------------------- more cameras
def extra_videos(args):
    """[(name, path)] of the take's other cameras: --video2 / --video3, else <video>_camera2.mp4 / _camera3.mp4
    next to the video (studio.py --record writes those with --camera2 / --camera3)."""
    if args.main_only:
        return []
    root, ext = os.path.splitext(args.video)
    out = []
    for name, given in (("camera2", args.video2), ("camera3", args.video3)):
        if given and not os.path.exists(given):
            raise SystemExit(f"No video at {given}")
        path = given or f"{root}_{name}{ext}"
        if os.path.exists(path):
            out.append((name, path))
    return out


def _next_frame(e, ts_ms):
    """Extra camera e's next video frame, tracked; None once its video has ended (ExtraCamera.step would hand back
    its last result again)."""
    last = e.frame
    res = e.step(ts_ms)
    return None if e.frame is last else res


def _slim(res, main):
    """An extra camera's result without what merging never uses, so the whole take fits in memory: its face while
    the main camera has yours (a side camera's face only stands in when the main camera lost it)."""
    if res is not None and main["faces"] and res["faces"]:
        res = dict(res, faces=[])
    return res


def line_up(multicam, saved, videos):
    """Each extra camera's rotation into the main camera's view, fitted (multicam.py, Kabsch) to the body points
    both cameras saw clearly over the whole take. A camera the take can't line up because they never saw you
    together falls back to the alignment your last C calibration saved, if any; a mirrored or roughly fitting one
    isn't used. Returns {name: what summary.json says about it}."""
    found = multicam.calib_finish()
    out = {}
    for e in multicam.extras:
        rep = multicam.calib_report.get(e.name, {})
        info = out[e.name] = {"video": os.path.abspath(videos[e.name]), "alignment_frames": rep.get("frames", 0)}
        old = (saved or {}).get(e.name) or {}
        if e.name in found:
            info.update(lined_up=True, alignment="from the take", error_cm=found[e.name]["error_cm"],
                        angle_deg=found[e.name]["angle_deg"], mirror_t=rep.get("mirror_t"))
        elif rep.get("frames", 0) < 5 and old.get("R") is not None:
            found[e.name] = old
            info.update(lined_up=True, alignment="saved by the last C calibration", error_cm=old.get("error_cm"),
                        angle_deg=old.get("angle_deg"))
            print(f"{e.name}: lined up as your last C calibration saved it instead", flush=True)
        else:
            info.update(lined_up=False, why=rep.get("note", "").split(": ", 1)[-1],
                        **{k: rep[k] for k in ("error_cm", "angle_deg", "mirror_fit_cm", "mirror_t", "mirrored")
                           if k in rep})
    multicam.set_alignment(found)
    return out


# ---------------------------------------------------------------- the run
def track(video, args, settings, extras=()):
    """Every frame through the tracker (+ WiLoR, waited for) and the rig without live smoothing. extras: [(name,
    path)] of other cameras' videos of the same take, in step frame for frame: each frame is tracked in every
    camera, the cameras are lined up from the whole take, then each frame is merged (multicam.py) before the rig.
    WiLoR's fingers come from the main camera. Returns per-frame records, the video's frame rate, the seconds it
    took and {camera name: alignment and how much it added} (None with one camera)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    multicam, kept = None, []  # (more cameras: every frame's results, merged once the cameras are lined up)
    if extras:
        from avatarstudio.tracking.multicam import ExtraCamera, MultiCam
        multicam = MultiCam([ExtraCamera(n, video=p) for n, p in extras])
        for e, (n, p) in zip(multicam.extras, extras):
            if not e.video.isOpened():
                raise SystemExit(f"Could not open {p}")
            count = int(e.video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total and count != total:
                print(f"{n}: {count} frames, the main video {total} - not recorded together? Frames are paired in "
                      f"order", flush=True)
        multicam.calib_start()
        print(f"cameras: main + {', '.join(n for n, _ in extras)}", flush=True)
    tracker = AllInOneTracker(video=True, fov=args.fov, ipd_cm=args.ipd, max_faces=1, upper_body=not args.full_body,
                              gpu_tracking=args.gpu_tracking or settings.gpu_tracking)
    if settings.wholebody:
        tracker.set_wholebody(True)
    rig = AvatarRig(smoothing=False, legs=args.full_body, idle=False)  # (your right is the avatar's right)
    rig.calib = load_json(user_file("calibration.json"))
    rig.face_refiner = load_face_refiner() if settings.face_refiner else None
    wilor = None
    if not args.no_wilor and settings.wilor_fingers:
        from avatarstudio.tracking.wilor_client import WilorFingers, find_wilor
        if find_wilor()[0]:
            try:
                wilor = WilorFingers()
                print("waiting for the WiLoR finger helper...", flush=True)
                t0 = time.time()
                while wilor.state == "starting" and time.time() - t0 < 120:
                    time.sleep(0.2)
                if wilor.state != "running":
                    print("WiLoR didn't start; fingers from MediaPipe", flush=True)
                    wilor.close()
                    wilor = None
            except Exception as e:
                print(f"WiLoR unavailable ({e}); fingers from MediaPipe", flush=True)
                wilor = None
    frames, pending, k, t_start, last_print = [], None, 0, time.time(), time.time()
    nan2 = lambda m: np.full((m, 2), np.nan)

    def record(res, t, main_body=None):
        """The rig for a (merged) frame, and what the exports need from it. Picture points belong to the main
        camera: none for what only a side camera saw (main_body: the main camera's own body result)."""
        avatar = rig(res, focal_px(res["_w"], args.fov), t)
        body = res["body"]
        if body is not None:
            p2 = nan2(33) if "_cam" in body else np.asarray(body["_pts"], float)[:, :2]
            if main_body is not None:
                p2 = np.where((np.asarray(main_body["_vis"]) >= VIS)[:, None], p2, np.nan)
        frames.append({
            "bones": {b: np.asarray(q, float) for b, q in avatar["bones"].items()},
            "root": np.asarray(avatar["root"], float), "arkit": dict(avatar["arkit"]),
            "body": None if body is None else (np.asarray(body["_world"], float), np.asarray(body["_vis"], float), p2),
            "hands": {h["side"]: nan2(21) if "_cam" in h else np.asarray(h["_pts"], float)[:, :2]
                      for h in res["hands"]},
            "wilor": sum("_world_fingers" in h for h in res["hands"]), "face": bool(res["faces"])})

    def finish(job):
        """A tracked frame, once WiLoR's fingers for it are in: through the rig, or (more cameras) kept until the
        cameras are lined up."""
        res, t, n, extra = job
        if n is not None:
            wilor.wait(n)
            wilor.apply(res)
        if multicam is None:
            record(res, t)
        else:
            multicam.calib_add(res, extra)
            kept.append((res, t, extra))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            t = k / fps
            jobs = [multicam.pool.submit(_next_frame, e, t * 1000) for e in multicam.extras] if multicam else None
            res = tracker(frame, timestamp_ms=t * 1000)  # (WiLoR works on the previous frame meanwhile)
            res["_w"] = frame.shape[1]
            extra = [_slim(r, res) for r in multicam.collect(jobs)] if jobs else None
            if pending is not None:
                finish(pending)
            hands = [h for h in res["hands"] if "_cam" not in h]
            n = None
            if wilor is not None and wilor.state == "running" and hands:
                n = wilor.done + 1
                wilor.submit(frame, hands)
            pending = (res, t, n, extra)
            if time.time() - last_print > 2:
                last_print = time.time()
                rate = k / (time.time() - t_start)
                left = f", about {(total - k) / rate:.0f} s left" if total > k else ""
                print(f"tracked {k}{f'/{total}' if total else ''} frames ({rate:.1f} fps{left})", flush=True)
        if pending is not None:
            finish(pending)
    finally:
        cap.release()
        if wilor is not None:
            wilor.close()
        if multicam is not None:
            multicam.release()
    cameras = None
    if multicam is not None:
        print(f"lining the cameras up from the take ({len(kept)} frames)...", flush=True)
        cams = {"main": {"video": os.path.abspath(video)},
                **line_up(multicam, (rig.calib or {}).get("cameras"), dict(extras))}
        stats = merge(kept, multicam, record, args)
        for name, s in stats.pop("per_camera").items():
            cams[name].update(s)
        cameras = {"cameras": cams, **stats}
    return frames, fps, time.time() - t_start, cameras


def merge(kept, multicam, record, args):
    """Every kept frame merged over the lined-up cameras (multicam.fuse) and through the rig (record). Returns how
    much of the body the side cameras gave: the share of the merge weight (multicam.body_weight) per camera, the
    body points only a side camera saw, and the frames where body / face / a hand came from a side camera alone."""
    from avatarstudio.tracking.multicam import FUSE_VIS, body_weight
    pts = [i for i in range(33) if args.full_body or i not in LEGS]
    names = ["main"] + [e.name for e in multicam.extras]
    weight = dict.fromkeys(names, 0.0)
    tracked = dict.fromkeys(names[1:], 0)
    alone = dict.fromkeys(("body", "face", "left_hand", "right_hand"), 0)
    seen = only_side = 0
    for i, (res, t, extra) in enumerate(kept):
        kept[i] = None  # (memory goes back as the take is merged)
        fused = multicam.fuse(res, extra)
        record(fused, t, res["body"])
        for e, r in zip(multicam.extras, extra):
            tracked[e.name] += r is not None and r["body"] is not None
        body = fused["body"]
        if body is not None:
            if "_cam" in body:  # the main camera lost you: one side camera's body
                alone["body"] += 1
                views = [(body["_cam"], body["_vis"])]
            else:
                views = [("main", res["body"]["_vis"])] + [
                    (e.name, r["body"]["_vis"]) for e, r in zip(multicam.extras, extra)
                    if e.R is not None and r is not None and r["body"] is not None]
            for name, vis in views:
                weight[name] += float(body_weight(np.asarray(vis, float)[pts]).sum())
            any_cam = np.asarray(body["_vis"])[pts] > FUSE_VIS
            main = np.asarray(res["body"]["_vis"])[pts] > FUSE_VIS if "_cam" not in body else np.zeros(len(pts), bool)
            seen += int(any_cam.sum())
            only_side += int((any_cam & ~main).sum())
        alone["face"] += bool(fused["faces"]) and "_cam" in fused["faces"][0]
        for h in fused["hands"]:
            alone[f"{h['side']}_hand"] += "_cam" in h
    n, total = max(len(kept), 1), sum(weight.values()) or 1.0
    out = {name: {"body_weight_share": round(weight[name] / total, 3)} for name in names}
    for name in names[1:]:
        out[name]["body_tracked_share"] = round(tracked[name] / n, 3)
    return {"per_camera": out,
            "body_points_from_side_cameras": {"weight_share": round(1 - weight["main"] / total, 3),
                                              "seen_only_by_side_cameras": round(only_side / max(seen, 1), 4)},
            "frames_only_side_cameras_saw": {k: round(v / n, 4) for k, v in alone.items()}}


def main():
    ap = argparse.ArgumentParser(description="Offline motion capture from a video (BVH, CSV, joint angles, overlay)")
    ap.add_argument("video", nargs="?", default=user_file("take.mp4"))
    ap.add_argument("--out", help="output folder (default: <video name>_mocap next to the video)")
    ap.add_argument("--cutoff", type=float, default=6.0, help="body / hands filter in Hz (0 = no filtering)")
    ap.add_argument("--face-cutoff", type=float, default=12.0, help="face shapes filter in Hz")
    ap.add_argument("--max-gap", type=float, default=0.5, help="fill body points lost for up to this long (s)")
    ap.add_argument("--full-body", action="store_true", help="also legs (default: torso, arms and head)")
    ap.add_argument("--no-wilor", action="store_true", help="fingers from MediaPipe only")
    ap.add_argument("--gpu-tracking", action="store_true", help="body and face models on the GPU")
    ap.add_argument("--no-overlay", action="store_true", help="skip overlay.mp4")
    ap.add_argument("--fov", type=float, help="camera field of view in degrees (default: the setting)")
    ap.add_argument("--ipd", type=float, help="your pupil distance in cm (default: the setting)")
    ap.add_argument("--video2", help="the 2nd camera's video of the same take, in step frame for frame (default: "
                                     "<video>_camera2.mp4 next to the video, if it's there)")
    ap.add_argument("--video3", help="the 3rd camera's (default: <video>_camera3.mp4)")
    ap.add_argument("--main-only", action="store_true", help="only the main video, even if other cameras' are there")
    args = ap.parse_args()
    if not os.path.exists(args.video):
        raise SystemExit(f"No video at {args.video} (record a take first, or pass a video file)")
    out = args.out or os.path.splitext(args.video)[0] + "_mocap"
    os.makedirs(out, exist_ok=True)
    settings = Settings.load()
    args.fov = args.fov or settings.camera_fov  # (the same as live tracking, from Settings > Motion)
    args.ipd = args.ipd or settings.eye_spacing_cm
    frames, fps, track_s, cameras = track(args.video, args, settings, extra_videos(args))
    process(frames, fps, track_s, args, out, cameras)


def process(frames, fps, track_s, args, out, cameras=None):
    """Clean-up and every export, from track()'s records (cameras: its report on the other cameras, if any)."""
    n = len(frames)
    if n == 0:
        raise SystemExit("No frames could be read from the video")
    print(f"tracked {n} frames in {track_s:.0f} s ({n / track_s:.1f} fps); cleaning up...", flush=True)
    max_gap = int(round(args.max_gap * fps))

    # avatar motion: filtered rotations and root
    names = list(frames[0]["bones"])
    q = np.array([[fr["bones"].get(b, IDENTITY) for b in names] for fr in frames])
    q_f = filter_rotations(q, fps, args.cutoff)
    root = zero_lag(np.array([fr["root"] for fr in frames]), fps, args.cutoff)
    from avatarstudio.avatar.bvh import write_bvh
    write_bvh(os.path.join(out, "motion.bvh"),
              [{"bones": dict(zip(names, qf)), "root": r} for qf, r in zip(q_f, root)], fps)

    # body points: lost / unseen ones empty, short gaps filled, then filtered. MediaPipe's axes (x image right, y
    # down, z away) -> X your right, Y forward, Z up
    body = np.full((n, 33, 3), np.nan)
    body2d = np.full((n, 33, 2), np.nan)
    for i, fr in enumerate(frames):
        if fr["body"] is not None:
            w, vis, p2 = fr["body"]
            seen = vis >= VIS
            if not args.full_body:
                seen[list(LEGS)] = False
            body[i, seen] = np.stack([-w[seen, 0], -w[seen, 2], -w[seen, 1]], 1)
            body2d[i, seen] = p2[seen]
    raw_jitter = jitter_mm(body, fps)
    body, filled = fill_gaps(body, max_gap)
    body = zero_lag(body, fps, args.cutoff)
    body2d, filled2d = fill_gaps(body2d, max_gap)
    body2d = zero_lag(body2d, fps, args.cutoff)
    np.save(os.path.join(out, "body_3d.npy"), body.astype(np.float32))
    header = ["root_x", "root_y", "root_z"] + [f"{b}_{a}" for b in BODY_NAMES for a in "xyz"]
    root_b = np.stack([root[:, 0], root[:, 2], root[:, 1]], 1)  # (avatar axes x right, y up, z forward)
    write_csv(os.path.join(out, "body_3d.csv"), header, [*root_b.T, *body.reshape(n, -1).T], fps)

    seen = {s: np.array([s in fr["hands"] for fr in frames]) for s in ("left", "right")}
    seen["trunk"] = np.array([fr["body"] is not None for fr in frames])
    seen["head"] = seen["trunk"] | np.array([fr["face"] for fr in frames])
    angles = joint_angles(body, q_f, names, seen)
    write_csv(os.path.join(out, "joint_angles.csv"), list(angles), list(angles.values()), fps, decimals=1)

    shapes = sorted({k for fr in frames for k in fr["arkit"]})
    face = np.array([[fr["arkit"].get(s, 0.0) for s in shapes] for fr in frames], float)
    face = np.clip(zero_lag(face, fps, args.face_cutoff), 0, 1)
    write_csv(os.path.join(out, "face_arkit.csv"), shapes, list(face.T), fps, decimals=3)

    if not args.no_overlay:
        print("drawing overlay.mp4...", flush=True)
        hands2d, hands_filled = {}, {}
        for side in ("left", "right"):
            h = np.full((n, 21, 2), np.nan)
            for i, fr in enumerate(frames):
                if side in fr["hands"]:
                    h[i] = fr["hands"][side]
            h, hf = fill_gaps(h, max_gap)
            hands2d[side], hands_filled[side] = zero_lag(h, fps, args.cutoff), hf[..., 0]
        draw_overlay(args.video, os.path.join(out, "overlay.mp4"), fps, body2d, filled2d[..., 0], hands2d,
                     hands_filled)

    share = lambda key: round(float(np.mean([bool(fr[key]) for fr in frames])), 3)
    hand_frames = sum(len(fr["hands"]) for fr in frames)
    rom = {}
    for k, v in angles.items():
        ok = v[np.isfinite(v)]
        if ok.size:
            rom[k] = {"min": round(float(ok.min()), 1), "p5": round(float(np.percentile(ok, 5)), 1),
                      "median": round(float(np.median(ok)), 1), "p95": round(float(np.percentile(ok, 95)), 1),
                      "max": round(float(ok.max()), 1), "frames": int(ok.size), "repetitions": repetitions(v, fps)}
    upper = [i for i in range(33) if args.full_body or i not in LEGS]
    summary = {
        "video": os.path.abspath(args.video), "frames": n, "fps": round(fps, 3), "seconds": round(n / fps, 2),
        "processing": {"seconds": round(track_s, 1), "fps": round(n / track_s, 1)},
        "filter": {"type": "Butterworth, 4th order, forward and backward (no delay)", "body_hz": args.cutoff,
                   "face_hz": args.face_cutoff},
        "tracked_share": {"face": share("face"), "body": share("body"),
                          "left_hand": round(float(np.mean(["left" in fr["hands"] for fr in frames])), 3),
                          "right_hand": round(float(np.mean(["right" in fr["hands"] for fr in frames])), 3),
                          "hands_with_wilor_fingers": round(sum(fr["wilor"] for fr in frames) / max(hand_frames, 1), 3)},
        "body_points": {"filled_share": round(float(filled[:, upper, 0].mean()), 4),
                        "empty_share": round(float(np.isnan(body[:, upper, 0]).mean()), 4),
                        "max_gap_filled_s": args.max_gap},
        "jitter_mm_per_frame": {"before": raw_jitter, "after": jitter_mm(body, fps)},
        "range_of_motion_deg": rom,
        "axes": "metres; X your right, Y forward (toward the camera), Z up; origin between the hips",
    }
    if cameras:  # (axes are the main camera's: the others are turned into its view)
        summary.update(cameras)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done -> {out}")
    print(f"  {n} frames, {n / fps:.1f} s; tracked: face {summary['tracked_share']['face']:.0%}, body "
          f"{summary['tracked_share']['body']:.0%}; body points filled in {summary['body_points']['filled_share']:.1%}, "
          f"left empty {summary['body_points']['empty_share']:.1%}")
    print(f"  jitter {raw_jitter} -> {summary['jitter_mm_per_frame']['after']} mm per frame")
    if cameras:
        side = cameras["body_points_from_side_cameras"]
        print("  cameras: " + "; ".join(
            f"{name}: " + (f"lined up {c['angle_deg']} deg off ({c['error_cm']} cm fit), {c['body_weight_share']:.0%} "
                           f"of the body" if c["lined_up"] else c["why"])
            for name, c in cameras["cameras"].items() if name != "main"))
        print(f"  side cameras: {side['weight_share']:.0%} of the body-point weight, "
              f"{side['seen_only_by_side_cameras']:.1%} of the points only they saw")
    print("  motion.bvh, body_3d.csv/.npy, joint_angles.csv, face_arkit.csv, summary.json"
          + ("" if args.no_overlay else ", overlay.mp4"))


if __name__ == "__main__":
    main()
