"""Avatar studio: webcam -> all-in-one tracking -> avatar rig -> VTuber apps / 3D tools.

  .venv\\Scripts\\python.exe app.py studio                         # webcam, VMC to 127.0.0.1:39539, 3D viewer stream on :8765
  .venv\\Scripts\\python.exe app.py studio --mirror-avatar         # avatar moves like your mirror image
  .venv\\Scripts\\python.exe app.py studio --record take1.mp4 --bvh take1.bvh   # save video + motion for Blender
  .venv\\Scripts\\python.exe app.py studio --input take1.mp4       # replay a recording through the tracker

Keys: c calibrate (sit/stand neutral, look at the screen, hold still for 1 s),
      v vowel calibration (after c: say A, I, U, E, O when prompted),
      g eye-gaze calibration (follow 6 dots with your eyes, head still), k hands (open, then fists),
      p pause (the avatar holds its pose), f face monitor, q quit, m face mesh, b body skeleton, h panel.
      From any window: Ctrl+Alt+C recalibrate, Ctrl+Alt+P pause, Ctrl+Alt+1-9 expressions.
Outputs:
  VMC protocol (VSeeFace, VNyan, Warudo, VMagicMirror, ...): bones, root motion, VRM + Perfect Sync blendshapes
  WebSocket JSON on ws://127.0.0.1:8765, and a live 3D viewer page at http://127.0.0.1:8000
"""
import argparse
import json
import math
import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np
import psutil

from avatarstudio.tracking.all_in_one import AllInOneTracker, draw, focal_px, public
from avatarstudio.audio.mic import MicLipSync
from avatarstudio.tracking.camera import CameraStream
from avatarstudio.personal.face_refiner import load as load_face_refiner
from avatarstudio.personal.gestures import GestureRecognizer
from avatarstudio.engine.latency import ClapLatency
from avatarstudio.personal.mouth_model import MODEL_PATH as MOUTH_MODEL, MouthModel
from avatarstudio.tracking.wilor_client import WilorFingers, find_wilor
from avatarstudio.core.perf import boost_process
from avatarstudio.avatar.rig import HAND_RANGE_DEFAULT, HAND_RANGE_WILOR, AutoRange, AvatarRig, hand_range_from, raw_finger_flex, to_unity
from avatarstudio.core.settings import GlobalHotkeys, Settings, list_profiles, load_hotkeys, load_profile, save_profile
from avatarstudio.engine.streaming import VMCSender, WebSocketStream, serve_viewer
from avatarstudio.avatar.testpattern import STEPS as TEST_STEPS, frame_at
from avatarstudio.engine.vts import VTubeStudio, vts_parameters
from avatarstudio.core.paths import load_json, save_json, user_file

CALIB_PATH = user_file("calibration.json")
CALIB_FRAMES = 20
HANDCAL_STEP_S = 3.0  # K hand calibration: open hands, then fists, this long each
VOWEL_PROMPTS = [("A", "aah  (as in 'father')"), ("I", "eee  (as in 'see')"), ("U", "ooo  (as in 'blue')"),
                 ("E", "eh  (as in 'bed')"), ("O", "oh  (as in 'go')")]
VOWEL_READY_S, VOWEL_HOLD_S = 1.2, 1.5
MONITOR_SHAPES = ["A", "I", "U", "E", "O", "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
                  "mouthFunnel", "mouthPressLeft", "cheekPuff", "browInnerUp", "browDownLeft", "browOuterUpLeft",
                  "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeSquintLeft"]


def draw_face_monitor(view, frame, res, avatar):
    """Bottom-left panel: zoomed mouth + live bars of what is being sent to the avatar."""
    h, w = view.shape[:2]
    face = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
    x0, y1 = 10, h - 34
    pw, bar_h = 330, 13
    ph = 20 + len(MONITOR_SHAPES) * bar_h + 120
    y0 = y1 - ph
    overlay = view.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y1), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
    if face is not None:  # mouth close-up, mirrored like the main view
        p = face["_pts"]
        mw = np.linalg.norm(p[61, :2] - p[291, :2])
        cx, cy = (p[61, :2] + p[291, :2]) / 2
        half = max(int(mw * 0.9), 10)
        x_a, y_a = int(max(cx - half, 0)), int(max(cy - half * 0.6, 0))
        crop = frame[y_a:int(cy + half * 0.6), x_a:int(cx + half)]
        if crop.size:
            crop = cv2.flip(cv2.resize(crop, (200, 110)), 1)
            view[y0 + 5:y0 + 115, x0 + 65:x0 + 265] = crop
    shapes = {**avatar["arkit"], **avatar["vrm"]}
    for i, name in enumerate(MONITOR_SHAPES):
        v = shapes.get(name, 0.0)
        yy = y0 + 128 + i * bar_h
        col = (255, 200, 80) if len(name) == 1 else (120, 220, 120)
        cv2.putText(view, name, (x0 + 8, yy + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(view, (x0 + 130, yy + 2), (x0 + pw - 10, yy + 10), (60, 60, 60), -1)
        cv2.rectangle(view, (x0 + 130, yy + 2), (x0 + 130 + int((pw - 140) * v), yy + 10), col, -1)


EMPTY_RESULT = {"faces": [], "hands": [], "body": None, "head": None, "depth": None}
GAZE_TARGETS = [(0, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]  # screen units: -1 = left / top
GAZE_SETTLE_S, GAZE_HOLD_S = 1.0, 1.2
GAZE_WINDOW = "Gaze calibration"


def screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def show_gaze_target(tx, ty, i, dt):
    """Full-screen black window with one dot; follow it with your eyes, keep your head still."""
    w, h = screen_size()
    img = np.zeros((h, w, 3), np.uint8)
    x, y = int(w / 2 + tx * w * 0.42), int(h / 2 + ty * h * 0.40)
    pulse = 14 + int(6 * abs(math.sin(dt * 5)))
    cv2.circle(img, (x, y), pulse, (60, 220, 255) if dt >= GAZE_SETTLE_S else (120, 120, 120), -1, cv2.LINE_AA)
    msg = f"Look at the dot with your EYES only - keep your head still   ({i + 1}/{len(GAZE_TARGETS)})"
    size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(img, msg, ((w - size[0]) // 2, int(h * 0.5) + (60 if ty <= 0 else -60) if tx == 0 else h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.namedWindow(GAZE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(GAZE_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(GAZE_WINDOW, img)


def run_test_pattern(vmc, ws, mirror, args):
    """Send known poses / faces in a loop (no camera) and show what the avatar should be doing."""
    print("Test pattern: compare your avatar with the description in the window. Q quits.")
    t0, last = time.time(), -1
    while True:
        t = time.time() - t0
        i, desc, avatar = frame_at(t, mirror)
        if i != last:
            print(f"step {i + 1}/{len(TEST_STEPS)}: {desc}", flush=True)
            last = i
        if vmc:
            vmc.send(avatar)
        if ws:
            ws.send({"type": "frame", "t": round(t, 3), "test_step": f"{i + 1}/{len(TEST_STEPS)}: {desc}",
                     "bones": {k: np.round(q, 5).tolist() for k, q in avatar["bones"].items()},
                     "root": avatar["root"].tolist(), "vrm": avatar["vrm"], "arkit": avatar["arkit"],
                     "gestures": [], "distance_cm": None, "body_world": None})
        if not args.headless:
            img = np.zeros((300, 1100, 3), np.uint8)
            cv2.putText(img, f"Step {i + 1}/{len(TEST_STEPS)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 230, 255), 2)
            words, line, y = desc.split(), "", 110
            for wd in words:  # wrap
                if len(line) + len(wd) > 55:
                    cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2)
                    line, y = "", y + 45
                line += wd + " "
            cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2)
            cv2.putText(img, "Q quits", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
            cv2.imshow("Avatar test pattern", img)
            if cv2.waitKey(33) & 0xFF in (ord("q"), ord("Q")):
                break
        else:
            time.sleep(1 / 30)
        if args.max_seconds and t > args.max_seconds:
            break
    cv2.destroyAllWindows()


def draw_banner(view, lines):
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, h // 2 - 70), (w, h // 2 + 20 + 40 * (len(lines) - 1)), (20, 20, 20), -1)
    for i, (s, scale) in enumerate(lines):
        size = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(view, s, ((w - size[0]) // 2, h // 2 - 20 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (80, 230, 255), 2, cv2.LINE_AA)


class Recorder:
    """--record: MP4 encoding on its own thread (~3-4 ms a frame on the tracking loop otherwise). Frames are
    written in order; release() finishes the file."""

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.q = queue.Queue(maxsize=120)  # (4 s at 30 fps; if the disk can't keep up, the loop waits)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, frame):
        self.q.put(frame)

    def _run(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            self.writer.write(f)

    def release(self):
        self.q.put(None)
        self.thread.join(timeout=30)
        self.writer.release()


class ExtraRecorders:
    """--record with --camera2 / --camera3: each extra camera's video next to the main one (take_camera2.mp4, ...)
    for mocap.py. Every time the main recording writes a frame, each extra camera's newest frame is written too
    (black until its first frame arrives), so frame k of every file is the same moment and the files stay
    frame-for-frame in step."""

    def __init__(self, path, fps, extras):
        root, ext = os.path.splitext(path)
        self.paths = {e.name: f"{root}_{e.name}{ext or '.mp4'}" for e in extras}
        self.fps, self.writers, self.owed = fps, {}, {e.name: 0 for e in extras}

    def write(self, extras, n):
        for e in extras:
            frame = e.frame
            if frame is None:
                self.owed[e.name] += n
                continue
            if e.name not in self.writers:
                size = (frame.shape[1], frame.shape[0])
                self.writers[e.name] = (Recorder(self.paths[e.name], self.fps, size), size)
                black = np.zeros_like(frame)
                for _ in range(self.owed[e.name]):
                    self.writers[e.name][0].write(black)
            rec, size = self.writers[e.name]
            if (frame.shape[1], frame.shape[0]) != size:  # (a frame of another size would be dropped: out of step)
                frame = cv2.resize(frame, size)
            for _ in range(n):
                rec.write(frame)

    def release(self):
        for name, (rec, _) in self.writers.items():
            rec.release()
            print(f"video saved -> {self.paths[name]}")


MAIN_WINDOW = "Avatar studio (c calibrate, q quit)"


class StudioWindow:
    """The preview window, drawn and shown on its own thread. Drawing it (face mesh, skeleton, panel, face monitor)
    takes ~4 ms and showing it 1-2 ms more; on the tracking loop that made each new camera frame wait up to ~6 ms
    before tracking started. The loop now hands over its newest frame and carries on; a frame the window hasn't
    drawn yet is simply replaced by the next one. OpenCV windows belong to the thread that made them, so the gaze
    calibration dots are shown from here too, and key presses come back through keys()."""

    def __init__(self, settings):
        self.settings, self.job, self.gaze, self._gaze_open = settings, None, None, False
        self.key_queue, self.running, self.errors = queue.Queue(), True, 0
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def show(self, frame, res, avatar, banners, text, warn, views=()):
        """views: [(label, frame, result)] of extra cameras, shown small in the top-left corner."""
        with self.cond:
            self.job = (frame, res, avatar, banners, text, warn, list(views))
            self.cond.notify()

    def keys(self):
        out = []
        while not self.key_queue.empty():
            out.append(self.key_queue.get_nowait())
        return out

    def _render(self, *job):
        cv2.imshow(MAIN_WINDOW, self._compose(*job))

    def _compose(self, frame, res, avatar, banners, text, warn, views):
        s = self.settings
        # (on a copy: the frame is shared - WiLoR's worker cuts hand patches from it)
        view = draw(frame.copy(), res, mirror=True, mesh=s.show_mesh, body=s.show_body, hud=s.show_panel)
        if s.show_face_monitor:
            draw_face_monitor(view, frame, res, avatar)
        y = 10
        for label, f, r in views:  # extra cameras, small, with what they track
            if f is None:
                continue
            small = draw(f.copy(), r, mirror=True, mesh=False, body=True, hud=False) if r else cv2.flip(f, 1)
            tw = 280
            small = cv2.resize(small, (tw, int(small.shape[0] * tw / small.shape[1])))
            th = small.shape[0]
            if y + th > view.shape[0] or tw > view.shape[1]:
                break
            view[y:y + th, 10:10 + tw] = small
            cv2.rectangle(view, (10, y), (10 + tw, y + th), (200, 200, 200), 1)
            cv2.putText(view, label, (16, y + th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, label, (16, y + th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 255), 1, cv2.LINE_AA)
            y += th + 8
        for lines in banners:
            draw_banner(view, lines)
        h = view.shape[0]
        cv2.putText(view, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255) if warn else (230, 230, 230), 1, cv2.LINE_AA)
        return view

    def _loop(self):
        while self.running:
            with self.cond:
                self.cond.wait_for(lambda: self.job is not None or not self.running, timeout=0.03)
                job, self.job = self.job, None
            if getattr(self, "closed", False):
                job = None
            try:
                if job is not None:
                    self._render(*job)
                gaze = self.gaze
                if gaze is not None:
                    show_gaze_target(*gaze)
                    self._gaze_open = True
                elif self._gaze_open:
                    cv2.destroyWindow(GAZE_WINDOW)
                    self._gaze_open = False
                key = cv2.waitKey(1) & 0xFF
                if job is not None:
                    self._shown = True
                if getattr(self, "_shown", False) and cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    key, self.closed = ord("q"), True  # closed with its X: quit, like Q (instead of reopening it)
            except Exception:  # a drawing problem must not take the window (or the keys) away
                self.errors += 1
                if self.errors == 1:
                    traceback.print_exc()
                continue
            if key != 255:
                self.key_queue.put(key + 32 if 65 <= key <= 90 else key)  # Caps Lock / Shift: C and c alike
        cv2.destroyAllWindows()

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify()
        self.thread.join(timeout=2.0)


class StreamView(StudioWindow):
    """--ui: the same picture for the Avatar Studio app's window instead of one of its own. Drawn on this thread as
    before, scaled to 960 wide and sent as JPEG (preview.py, camera.mjpg) - only while the app shows it. The app
    sends the keys (they arrive as messages) and draws the eye-gaze dots itself (the frame data has the target)."""
    WIDTH = 960

    def __init__(self, settings, hub):
        self.hub = hub
        super().__init__(settings)

    def _render(self, *job):
        view = self._compose(*job)
        h, w = view.shape[:2]
        if w > self.WIDTH:
            view = cv2.resize(view, (self.WIDTH, round(h * self.WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.hub.publish(jpeg.tobytes())

    def _loop(self):
        while self.running:
            with self.cond:
                self.cond.wait_for(lambda: self.job is not None or not self.running, timeout=0.1)
                job, self.job = self.job, None
            if job is None or not self.hub.watched():
                continue
            try:
                self._render(*job)
            except Exception:  # a drawing problem must not stop the picture for good
                self.errors += 1
                if self.errors == 1:
                    traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input", help="video file (or a still image, looped) instead of the webcam")
    ap.add_argument("--vmc", default="127.0.0.1:39539", help="VMC receiver host:port")
    ap.add_argument("--no-vmc", action="store_true")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--no-ws", action="store_true", help="no WebSocket stream / 3D viewer")
    ap.add_argument("--viewer-port", type=int, default=8000, help="where the 3D viewer page is served")
    ap.add_argument("--mirror-avatar", action="store_true", help="raise your right hand -> avatar raises its left")
    ap.add_argument("--no-root-motion", action="store_true", help="don't move the avatar when you move")
    ap.add_argument("--pose", choices=["full", "lite"], default="full",
                    help="body model: full, or lite (faster, less accurate arms and torso)")
    ap.add_argument("--full-body", action="store_true", help="also track legs (default: torso, arms and head)")
    ap.add_argument("--resolution", default="1280x720", help="webcam resolution, e.g. 640x480 for slower PCs")
    ap.add_argument("--fps", type=int, default=30, choices=[30, 60],
                    help="webcam frame rate to ask for: 60 halves the camera's share of the delay, if it can")
    ap.add_argument("--camera2", type=int, help="experimental: a 2nd webcam (e.g. left of you), see multicam.py")
    ap.add_argument("--camera3", type=int, help="experimental: a 3rd webcam (e.g. right of you)")
    ap.add_argument("--input2", help="a video file standing in for the 2nd camera (replays / tests)")
    ap.add_argument("--input3", help="a video file standing in for the 3rd camera")
    ap.add_argument("--expressiveness", type=float, default=1.0,
                    help="multiply all face shapes (e.g. 1.3 = more exaggerated, 0.8 = subtler)")
    ap.add_argument("--no-auto-range", action="store_true",
                    help="don't stretch each face shape to your own maximum")
    ap.add_argument("--fov", type=float, default=65.0)
    ap.add_argument("--ipd", type=float, default=6.3)
    ap.add_argument("--record", help="save the raw camera video here (mp4) for replay; extra cameras go next to it, "
                                     "in step frame for frame (take_camera2.mp4, take_camera3.mp4) for mocap.py")
    ap.add_argument("--bvh", help="save the avatar motion here as BVH when you quit")
    ap.add_argument("--log", help="save per-frame tracking + avatar data (.jsonl)")
    ap.add_argument("--headless", action="store_true", help="no window (for testing)")
    ap.add_argument("--ui", action="store_true", help="no window of its own: the Avatar Studio app shows the picture "
                                                      "(http://127.0.0.1:8000/camera.mjpg) and sends the keys")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0 = until q)")
    ap.add_argument("--no-mic", action="store_true", help="no microphone lip sync")
    ap.add_argument("--mic-device", help="microphone name or number (default: Windows default input)")
    ap.add_argument("--vts", action="store_true", help="also send face tracking to VTube Studio (Live2D)")
    ap.add_argument("--no-global-hotkeys", action="store_true", help="only window keys 1-9, no Ctrl+Alt+1-9")
    ap.add_argument("--test-pattern", action="store_true",
                    help="send a fixed sequence of poses/faces (no camera) to check your VTuber app + model")
    ap.add_argument("--latency-test", action="store_true",
                    help="clap slowly in front of the camera; measures real motion-to-avatar delay")
    ap.add_argument("--stats-every", type=float, default=0, help="print fps/memory every N seconds (long runs)")
    ap.add_argument("--no-wilor", action="store_true", help="fingers from MediaPipe only (no WiLoR helper)")
    ap.add_argument("--gpu-tracking", action="store_true",
                    help="run the body and face models on the GPU (gpu_tracking.py; also a setting)")
    ap.add_argument("--no-boost", action="store_true",
                    help="let Windows throttle the studio like any background program")
    args = ap.parse_args()
    if not args.no_boost:
        print(f"performance: {boost_process()}")

    # settings: settings.json (changed live from the settings page), overridden by command-line flags for this run
    # only: `saved` mirrors the file, so a flag like --no-mic is never written into it
    saved = Settings.load()
    settings = Settings.load()
    settings.update({k: v for k, v in {
        "mirror_avatar": True if args.mirror_avatar else None, "root_motion": False if args.no_root_motion else None,
        "auto_range": False if args.no_auto_range else None,
        "expressiveness": args.expressiveness if args.expressiveness != 1.0 else None,
        "mic_lipsync": False if args.no_mic else None, "vts": True if args.vts else None,
        "wilor_fingers": False if args.no_wilor else None}.items()
        if v is not None})

    cam_job = None  # the webcam opens (DirectShow can take a second) while the models load
    if not args.input and not args.test_pattern:
        from concurrent.futures import ThreadPoolExecutor
        cam_job = ThreadPoolExecutor(1).submit(CameraStream, args.camera, *map(int, args.resolution.split("x")),
                                               fps=args.fps)
    tracker = AllInOneTracker(video=True, pose_model=args.pose, fov=args.fov, ipd_cm=args.ipd, max_faces=1,
                              upper_body=not args.full_body, gpu_tracking=args.gpu_tracking or settings.gpu_tracking)
    rig = AvatarRig(legs=args.full_body)
    rig.calib = load_json(CALIB_PATH)
    gestures = GestureRecognizer()
    hotkeys = load_hotkeys()
    active_expr = set()  # expressions switched on by hotkeys / the settings page

    def toggle_expression(name):
        active_expr.symmetric_difference_update({name})

    hotkey_actions = queue.Queue()  # studio actions from the global hotkeys (they arrive on another thread)
    global_keys = None if args.no_global_hotkeys or args.headless else GlobalHotkeys(
        hotkeys, toggle_expression, actions={"c": lambda: hotkey_actions.put("calibrate"),
                                             "p": lambda: hotkey_actions.put("pause")})
    mic, vts, mouth_net, wilor = None, None, None, None
    mic_device = int(args.mic_device) if args.mic_device and args.mic_device.isdigit() else args.mic_device
    mic_starting = threading.Event()  # set while the microphone opens on its helper thread

    def start_mic():
        """Open the microphone on a helper thread (sounddevice + speech detection take ~0.5 s): tracking starts
        right away and lip sync joins when it's ready."""
        def run():
            nonlocal mic
            try:
                m = MicLipSync(mic_device, (rig.calib or {}).get("voice"), vad=settings.speech_detection,
                               denoise=settings.noise_suppression)
                if settings.mic_lipsync and mic is None:
                    mic = m
                else:  # (switched off meanwhile)
                    m.close()
            except Exception as e:
                print(f"microphone unavailable ({e}); lip sync uses the camera only", flush=True)
                settings.mic_lipsync = False
            finally:
                mic_starting.clear()
        mic_starting.set()
        threading.Thread(target=run, daemon=True).start()

    def apply_settings():
        """Push the current settings into the rig and start/stop the microphone and VTube Studio link."""
        nonlocal mic, vts, mouth_net, wilor
        rig.mirror, rig.root_motion = settings.mirror_avatar, settings.root_motion
        rig.expressiveness, rig.idle, rig.mic_weight = settings.expressiveness, settings.idle, settings.mic_weight
        rig.predict_s, rig.smooth_scale = settings.predict_ms / 1000, settings.smoothing
        rig.predict_body_s = settings.predict_body_ms / 1000
        rig.link_blinks, rig.body_follows_head = settings.link_blinks, settings.body_follows_head
        rig.no_clip, rig.body_scale = settings.no_clip, settings.body_size
        if settings.auto_range and rig.auto_range is None:
            rig.auto_range = AutoRange()
        elif not settings.auto_range:
            rig.auto_range = None
        if mic is not None and (mic.use_vad, mic.use_denoise) != (settings.speech_detection,
                                                                  settings.noise_suppression):
            mic.close()  # reopen with the new voice processing
            mic = None
        if settings.mic_lipsync and mic is None and not mic_starting.is_set():
            start_mic()
        elif not settings.mic_lipsync and mic is not None:
            mic.close()
            mic = None
        if tracker.set_wholebody(settings.wholebody, background=True) != settings.wholebody:
            settings.wholebody = False  # model or GPU runtime unavailable
        rig.face_refiner = load_face_refiner() if settings.face_refiner else None
        if settings.wilor_fingers and wilor is None and find_wilor()[0]:
            try:
                wilor = WilorFingers()
                print("WiLoR finger helper starting (fingers from MediaPipe until it's ready)")
            except Exception as e:
                print(f"WiLoR unavailable ({e})")
        elif not settings.wilor_fingers and wilor is not None:
            wilor.close()
            wilor = None
        if settings.mouth_model and mouth_net is None and os.path.exists(MOUTH_MODEL):
            try:
                mouth_net = MouthModel()
                print(f"mouth model: {', '.join(mouth_net.enabled) or 'nothing reliable enough'}")
            except Exception as e:
                print(f"mouth model unavailable ({e})")
        elif not settings.mouth_model:
            mouth_net = None
        if settings.vts and vts is None:
            vts = VTubeStudio()
        elif not settings.vts and vts is not None:
            vts.stop()
            vts = None

    apply_settings()
    vmc_host, vmc_port = args.vmc.rsplit(":", 1)
    vmc = None if args.no_vmc else VMCSender(vmc_host, int(vmc_port))
    ws = None if args.no_ws else WebSocketStream(port=args.ws_port)
    viewer_url = None
    preview = None  # --ui: the picture for the app's window (preview.py)
    if args.ui and not args.headless:
        from avatarstudio.engine.preview import PreviewHub
        preview = PreviewHub()
    if ws:
        try:
            routes = {"/camera.mjpg": preview.serve} if preview else None
            viewer_url = serve_viewer(args.viewer_port, routes=routes) + (f"/?ws=ws://127.0.0.1:{args.ws_port}"
                                                                          if args.ws_port != 8765 else "")
            print(f"3D viewer: {viewer_url}")
        except OSError:
            print(f"port {args.viewer_port} is busy; open viewer/index.html another way")

    if args.test_pattern:
        run_test_pattern(vmc, ws, settings.mirror_avatar, args)
        return

    still, cap, cam = None, None, None
    if args.input and os.path.splitext(args.input)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        still = cv2.imread(args.input)
        if still is None:
            raise SystemExit(f"Could not read {args.input}")
    elif args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cam = cam_job.result()
    file_fps = cap.get(cv2.CAP_PROP_FPS) if cap else 0
    cam_id = 0
    multicam = None  # experimental 2nd / 3rd camera (multicam.py)
    extra = [(n, i, v) for n, i, v in (("camera2", args.camera2, args.input2), ("camera3", args.camera3, args.input3))
             if i is not None or v]
    for n, i, v in list(extra):  # the same webcam can't be opened twice (it fails or steals the frames)
        if v is None and (i == (None if args.input else args.camera) or [e[1] for e in extra].count(i) > 1):
            print(f"{n}: camera {i} is already used - ignored")
            extra.remove((n, i, v))
    if extra:
        from avatarstudio.tracking.multicam import ExtraCamera, MultiCam
        size = tuple(map(int, args.resolution.split("x")))
        try:
            multicam = MultiCam([ExtraCamera(n, i, size, args.fps, v) for n, i, v in extra])
            multicam.set_alignment((rig.calib or {}).get("cameras"))
            ready = [e.name for e in multicam.aligned]
            print(f"extra cameras (experimental): {', '.join(n for n, _, _ in extra)}; "
                  + (f"lined up: {', '.join(ready)}" if ready else "press C (neutral pose) to line them up"))
        except Exception as e:
            print(f"extra cameras unavailable ({e}); carrying on with the main camera")
            multicam = None
    timing = {k: 0.0 for k in ("camera wait", "tracking", "rig", "send", "draw")}

    writer, log, motion = None, open(args.log, "w") if args.log else None, []
    extra_rec = None  # (--record with extra cameras: their videos, in step with the main one)
    no_camera, cam_lost_warned, pipeline_ms, last_stats = False, False, [], time.time()
    clap = ClapLatency() if args.latency_test else None
    window = None if args.headless else StreamView(settings, preview) if preview else StudioWindow(settings)
    if clap is not None and not settings.mic_lipsync:
        print("the latency test needs the microphone (don't use --no-mic)")
    gaze_cal = None  # gaze calibration state: {"i": target index, "t0": start, "samples": [...]}
    hand_cal = None  # K hand calibration state: {"t0": start, "open": [...], "fist": [...]}
    vowel = None  # vowel calibration state: {"i": prompt index, "t0": start time, "samples": {...}, "voice": {...}}
    calib_samples, frame_i, t_start = None, 0, time.time()
    fps, t_prev, last_status = 0.0, time.time(), 0.0
    paused = False  # P / Ctrl+Alt+P: the avatar holds its pose, tracking stops (and stops using the CPU)
    fps_checked = False  # (asked for 60 fps: say once if the camera can't)

    def set_paused(on):
        nonlocal paused, calib_samples, hand_cal, vowel, gaze_cal
        if on == paused:
            return
        paused = on
        if on:  # a calibration can't record while paused
            calib_samples = hand_cal = vowel = gaze_cal = None
            if window is not None:
                window.gaze = None
        print("paused - the avatar holds its pose (P or Ctrl+Alt+P resumes)" if on else "tracking resumed", flush=True)

    def start_vowels():
        nonlocal vowel
        set_paused(False)
        if not rig.calib:
            print("calibrate the neutral pose (C) before the vowels (V)")
            return
        vowel = {"i": 0, "t0": time.time(), "samples": {v: [] for v, _ in VOWEL_PROMPTS},
                 "voice": {v: [] for v, _ in VOWEL_PROMPTS}}

    def nonlocal_gaze_start():
        nonlocal gaze_cal
        set_paused(False)
        gaze_cal = {"i": 0, "t0": time.time(), "samples": []}

    def nonlocal_hand_cal_start():
        nonlocal hand_cal
        set_paused(False)
        hand_cal = {"t0": time.time(), "open": [], "fist": [], "wilor": False}

    def start_calibration():
        nonlocal calib_samples
        set_paused(False)
        calib_samples = []
        if multicam:
            multicam.calib_start()

    errors = {}  # part -> how many frames it failed on
    last_avatar = {"bones": {}, "root": np.zeros(3), "arkit": {}, "vrm": {}, "gaze": None}

    def guarded(part, fn, *a, default=None, **kw):
        """fn(*a, **kw), or `default` if it raises: one bad frame (a driver hiccup, an odd tracking result)
        skips that part for that frame instead of ending the session. The first failure is printed in full,
        then only how often it happens."""
        try:
            return fn(*a, **kw)
        except Exception:
            n = errors[part] = errors.get(part, 0) + 1
            if n == 1:
                traceback.print_exc()
                print(f"{part} failed on a frame - carrying on (further failures are only counted)", flush=True)
            elif n in (10, 100, 1000, 10000):
                print(f"{part} has failed on {n} frames", flush=True)
            return default

    def change_settings(values):
        settings.update(values)
        saved.update(values)
        saved.save()
        apply_settings()

    def handle_message(m):
        """Messages from the settings page."""
        kind = m.get("type")
        if kind == "set":
            change_settings(m.get("values", {}))
        elif kind == "action":
            name = m.get("name")
            if name == "calibrate":
                start_calibration()
            elif name == "pause":
                set_paused(not paused)
            elif name == "reset_settings":  # every setting back to its default (calibration stays)
                change_settings(Settings().as_dict())
                print("settings reset to defaults", flush=True)
            elif name == "vowels":
                start_vowels()
            elif name == "gaze":
                nonlocal_gaze_start()
            elif name == "hands":
                nonlocal_hand_cal_start()
            elif name == "reset_depth":
                tracker.depth.reset()
            elif name == "toggle_expression" and m.get("arg") in set(hotkeys.values()):  # (the page's buttons)
                toggle_expression(m.get("arg"))
        elif kind == "key" and window is not None:  # the app's buttons and keyboard: like keys in the studio window
            key = str(m.get("key", ""))[:1].lower()
            if key:
                window.key_queue.put(ord(key))
        elif kind == "avatar_rest":  # the 3D viewer's avatar proportions (contacts, no-clipping, thumb direction)
            rig.set_proportions(m.get("bones", {}))
            print(f"avatar proportions received ({len(m.get('bones', {}))} bones)", flush=True)
        elif kind == "save_profile":
            print(f"profile saved: {save_profile(m.get('name', ''), settings, rig.calib)}")
        elif kind == "load_profile":
            try:
                p = load_profile(m.get("name", ""))
            except OSError:
                return
            change_settings(p.get("settings", {}))
            if p.get("calibration"):
                rig.calib = p["calibration"]
                save_json(CALIB_PATH, rig.calib)
                if mic:
                    mic.set_voice(rig.calib.get("voice"))
            apply_settings()
            print(f"profile loaded: {m.get('name')}")

    try:
        while True:
            t_a = time.perf_counter()
            if ws:
                for m in ws.messages():
                    try:  # a malformed message from a page must not stop tracking
                        handle_message(m) if isinstance(m, dict) else None
                    except Exception as e:
                        print(f"ignored a message from the page ({type(e).__name__}: {e})", flush=True)
            while not hotkey_actions.empty():  # Ctrl+Alt+C / Ctrl+Alt+P from any window
                action = hotkey_actions.get_nowait()
                if action == "calibrate":
                    start_calibration()
                elif action == "pause":
                    set_paused(not paused)
            if still is not None:
                frame = still.copy()
            elif cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                ok, frame, cam_id = cam.read(cam_id)
                if not ok:  # camera lost: keep the avatar alive (it eases to rest) while the camera reconnects
                    if not cam_lost_warned:
                        print("camera lost - trying to reconnect (try --camera 1 if it never comes back)")
                        cam_lost_warned = True
                    frame = np.zeros((720, 1280, 3), np.uint8)
                    cv2.putText(frame, "camera lost - reconnecting...", (380, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0, 0, 255), 2)
                    no_camera = True
                else:
                    frame, no_camera, cam_lost_warned = frame.copy(), False, False
            frame_i += 1
            arrived = cam.frame_time if (cam and not no_camera) else t_a  # when this frame reached the studio
            t_b = time.perf_counter()
            timing["camera wait"] += t_b - t_a
            # video files run on their own clock so replays behave like the original take
            t = frame_i / file_fps if file_fps else time.time() - t_start
            if args.record and writer is None:
                h, w = frame.shape[:2]
                rec_fps = args.fps if cam else (file_fps or 30)
                writer = Recorder(args.record, round(rec_fps), (w, h))
                extra_rec = ExtraRecorders(args.record, round(rec_fps), multicam.extras) if multicam else None
                rec_t0, rec_n = time.perf_counter(), 0
            rec_add = 0  # (frames the recording writes this loop: the extra cameras' files write as many)
            if writer:
                # live: on a real-time schedule (a frame repeated when tracking was slower than the camera), so the
                # take always replays at the speed it was recorded - one frame per loop played a 60 fps camera's
                # take too fast whenever tracking couldn't keep up with it
                due = int((time.perf_counter() - rec_t0) * rec_fps) + 1 if cam else rec_n + 1
                rec_add = max(due - rec_n, 0)
                while rec_n < due:
                    writer.write(frame)
                    rec_n += 1

            extra_jobs = multicam.submit(t * 1000) if multicam and not paused else None  # alongside the main one
            res = EMPTY_RESULT if no_camera or paused else guarded("tracking", tracker, frame, timestamp_ms=t * 1000,
                                                                   default=EMPTY_RESULT)
            extra_res = multicam.collect(extra_jobs) if extra_jobs else None
            if extra_rec is not None and rec_add:  # (after collect: each extra camera's frame of this loop)
                guarded("extra camera recording", extra_rec.write, multicam.extras, rec_add)
            if extra_res is not None:
                if calib_samples is not None:  # the C calibration also lines the extra cameras up
                    multicam.calib_add(res, extra_res)
                res = guarded("extra cameras", multicam.fuse, res, extra_res, default=res)
            t_c = time.perf_counter()
            timing["tracking"] += t_c - t_b
            f = focal_px(frame.shape[1], args.fov)
            if wilor is not None and not no_camera and not paused:  # finger shapes from WiLoR (newest result)
                guarded("WiLoR", wilor.submit, frame, [h for h in res["hands"] if "_cam" not in h])
                guarded("WiLoR", wilor.apply, res)
            labels, gesture_shapes = guarded("gestures", gestures, res["hands"], default=({}, {}))
            for hand in res["hands"]:  # trained gestures replace the built-in rules when a model exists
                if gestures.ok:
                    hand["gesture"] = labels.get(hand["side"], "")

            if hand_cal is not None:  # K: open hands, then fists -> your finger bend range
                dt = time.time() - hand_cal["t0"]
                phase = "open" if dt < HANDCAL_STEP_S else "fist" if dt < 2 * HANDCAL_STEP_S else None
                if phase and dt % HANDCAL_STEP_S > 1.0:  # 1 s to get into position, then record
                    hand_cal[phase] += [raw_finger_flex(h) for h in res["hands"]]
                    hand_cal["wilor"] |= any("_world_fingers" in h for h in res["hands"])
                if phase is None:
                    if len(hand_cal["open"]) >= 15 and len(hand_cal["fist"]) >= 15:
                        # WiLoR and MediaPipe read fingers differently: each gets its own range
                        key, defaults = (("hand_range_wilor", HAND_RANGE_WILOR) if hand_cal["wilor"]
                                         else ("hand_range", HAND_RANGE_DEFAULT))
                        rig.calib = {**(rig.calib or {}), key: hand_range_from(hand_cal["open"], hand_cal["fist"], defaults)}
                        save_json(CALIB_PATH, rig.calib)
                        print(f"hand calibration saved -> {CALIB_PATH}")
                    else:
                        print("hand calibration: hands weren't visible enough - try again (K)")
                    hand_cal = None
            if calib_samples is not None:
                calib_samples.append(rig.neutral_from(res, f))
                if len(calib_samples) >= CALIB_FRAMES:
                    cal = rig.set_calibration(calib_samples)
                    if multicam:
                        cams = multicam.calib_finish()
                        cal = rig.calib = {**cal, "cameras": {**cal.get("cameras", {}), **cams}}
                        multicam.set_alignment(cal["cameras"])
                    save_json(CALIB_PATH, cal)
                    tracker.depth.reset()
                    calib_samples = None
                    print(f"calibrated ({', '.join(cal)}) -> {CALIB_PATH}")

            banner = None
            if hand_cal is not None:
                dt = time.time() - hand_cal["t0"]
                step = ("Open BOTH hands flat, fingers straight, palms to the camera" if dt < HANDCAL_STEP_S
                        else "Now make tight FISTS with both hands")
                banner = [(f"Hand calibration: {step}", 0.8),
                          (f"{len(hand_cal['open'])} open / {len(hand_cal['fist'])} fist samples", 0.7)]
            if vowel is not None:  # personal vowel calibration: say each vowel and hold it
                name, how = VOWEL_PROMPTS[vowel["i"]]
                dt = time.time() - vowel["t0"]
                if dt < VOWEL_READY_S:
                    banner = [(f"Next: say  {how}", 1.0), ("and hold it", 0.8)]
                elif dt < VOWEL_READY_S + VOWEL_HOLD_S:
                    m = rig.mouth_now(res)
                    if m:
                        vowel["samples"][name].append(m)
                    fm = mic.formants_now() if mic and mic.read()["level"] > 0.15 else None
                    if fm:
                        vowel["voice"][name].append(fm)
                    banner = [(f"Say  {how}  now", 1.1), (f"{'#' * int((dt - VOWEL_READY_S) / VOWEL_HOLD_S * 20)}", 0.8)]
                else:
                    vowel["i"], vowel["t0"] = vowel["i"] + 1, time.time()
                    if vowel["i"] == len(VOWEL_PROMPTS):
                        missing = [v for v, s in vowel["samples"].items() if not s]
                        if missing:
                            print(f"vowel calibration failed: no face seen for {missing}")
                        else:
                            rig.set_vowels(vowel["samples"], vowel["voice"])
                            save_json(CALIB_PATH, rig.calib)
                            voice = "and voice " if "voice" in rig.calib else "(voice not heard clearly - mic uses defaults) "
                            if mic:
                                mic.set_voice(rig.calib.get("voice"))
                            print(f"vowels {voice}calibrated -> {CALIB_PATH}")
                        vowel = None

            if gaze_cal is not None:  # eye-gaze calibration: follow the dot with your eyes, head still
                tx, ty = GAZE_TARGETS[gaze_cal["i"]]
                dt = time.time() - gaze_cal["t0"]
                face0 = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
                if GAZE_SETTLE_S <= dt < GAZE_SETTLE_S + GAZE_HOLD_S and face0 and face0.get("gaze"):
                    gaze_cal["samples"].append((tx, ty, face0["gaze"]["x"], face0["gaze"]["y"]))
                if window is not None:
                    window.gaze = (tx, ty, gaze_cal["i"], dt)
                if dt >= GAZE_SETTLE_S + GAZE_HOLD_S:
                    gaze_cal["i"], gaze_cal["t0"] = gaze_cal["i"] + 1, time.time()
                    if gaze_cal["i"] == len(GAZE_TARGETS):
                        if window is not None:
                            window.gaze = None  # (the window thread closes the dots)
                        try:
                            g = rig.set_gaze(gaze_cal["samples"])
                            save_json(CALIB_PATH, rig.calib)
                            print(f"eye gaze calibrated (screen edges = iris offset {abs(g['xs']):.2f} across, "
                                  f"{abs(g['ys']):.2f} up/down) -> {CALIB_PATH}")
                        except ValueError as e:
                            print(f"gaze calibration failed: {e}")
                        gaze_cal = None

            mic_state = mic.read() if mic else None
            main_face = max(res["faces"], key=lambda fc: fc["box"][2]) if res["faces"] else None
            mouth_shapes = guarded("mouth model", mouth_net, frame, main_face) \
                if mouth_net and main_face and not no_camera else None
            avatar = None if paused else guarded("rig", rig, res, f, t, gesture_shapes, mic=mic_state,
                                                 extra_shapes={n: 1.0 for n in active_expr}, arkit_extra=mouth_shapes)
            avatar = last_avatar = avatar or last_avatar  # (paused or a failed frame: the last pose again)
            if paused:
                banner = [("PAUSED - the avatar holds its pose", 1.0), ("P or Ctrl+Alt+P resumes", 0.7)]
            t_d = time.perf_counter()
            timing["rig"] += t_d - t_c
            if vmc:
                guarded("VMC output", vmc.send, avatar)
            if vts:
                params = guarded("VTube Studio output", vts_parameters, res, avatar, rig.calib)
                if params:
                    guarded("VTube Studio output", vts.send, params)
            if ws or log:
                payload = {"type": "frame", "t": round(t, 3),
                           "bones": {k: np.round(q, 5).tolist() for k, q in avatar["bones"].items()},
                           "root": np.round(avatar["root"], 4).tolist(), "vrm": avatar["vrm"], "arkit": avatar["arkit"],
                           "gestures": [g for g in labels.values() if g] if gestures.ok
                           else [h["gesture"] for h in res["hands"] if h["gesture"]],
                           "distance_cm": res["depth"]["distance_cm"] if res["depth"] else None,
                           "body_world": to_unity(res["body"]["_world"]).round(4).tolist() if res["body"] else None,
                           "wilor_hands": sum("_world_fingers" in h for h in res["hands"])}
                if banner or calib_samples is not None:  # what the studio window shows as a prompt (for the app)
                    payload["prompt"] = [line for line, _ in banner] if banner else \
                        ["Calibrating: sit as usual, look at the screen and hold still"]
                if gaze_cal is not None:  # the app draws the eye-gaze dot: x, y (-1..1), which one, of how many, s
                    payload["gaze_target"] = [*GAZE_TARGETS[gaze_cal["i"]], gaze_cal["i"], len(GAZE_TARGETS),
                                              round(time.time() - gaze_cal["t0"], 2)]
                if time.time() - last_status > 0.25:  # status for the settings page, 4x a second
                    last_status = time.time()
                    payload["status"] = {
                        "settings": settings.as_dict(), "fps": fps,
                        "cameras": [{"name": e.name, "aligned": e.R is not None, "seen": e.res is not None}
                                    for e in multicam.extras] if multicam else [],
                        "face": bool(res["faces"]), "hands": len(res["hands"]),
                        "calibrated": bool(rig.calib), "vowels": bool(rig.calib and "vowels" in rig.calib),
                        "face_refiner": rig.face_refiner is not None, "wholebody": tracker.wb is not None,
                        "mouth_shapes": mouth_net.enabled if mouth_net else [],
                        "speech": round(mic_state.get("speech", 0), 2) if mic_state else None,
                        "mic": round(mic_state["level"], 2) if mic_state else None,
                        "vts": vts.state if vts else "off", "hotkeys": hotkeys, "active": sorted(active_expr),
                        "paused": paused, "wilor": wilor.state if wilor else "off",
                        "profiles": list_profiles()}
                if ws:
                    ws.send(payload)
                if log:
                    log.write(json.dumps({**payload, "tracking": public(res), "mouth": rig.mouth_now(res)}) + "\n")
            if args.bvh:
                motion.append({"bones": avatar["bones"], "root": avatar["root"]})
            t_e = time.perf_counter()
            timing["send"] += t_e - t_d
            if not no_camera and not paused:
                if not pipeline_ms:  # the first frame out: how long the start took
                    print(f"ready: avatar data flowing {time.time() - psutil.Process().create_time():.1f} s after "
                          f"start", flush=True)
                pipeline_ms.append((t_e - arrived) * 1000)  # frame arrival -> avatar data sent
                del pipeline_ms[:-3000]
            if clap is not None and mic is not None and not no_camera:
                clap.update(arrived, res, list(mic.onsets))
            if cam and args.fps > 30 and not fps_checked and time.time() - t_start > 4:
                fps_checked = True
                if cam.camera_fps < args.fps * 0.75:
                    print(f"the camera delivers {cam.camera_fps:.0f} fps, not {args.fps}: it can't do {args.fps} fps "
                          f"at {args.resolution} (a lower --resolution may)", flush=True)
            if args.stats_every and time.time() - last_stats > args.stats_every:
                last_stats = time.time()
                mem = psutil.Process().memory_info().rss / 2 ** 20
                recent = pipeline_ms[-300:]
                print(f"[{time.time() - t_start:6.0f} s] {frame_i / (time.time() - t_start):.1f} fps avg, "
                      f"memory {mem:.0f} MB, pipeline {np.median(recent) if recent else 0:.0f} ms, "
                      f"camera reconnects {cam.reconnects if cam else 0}, mic restarts {mic.restarts if mic else 0}",
                      flush=True)

            t1 = time.time()
            fps, t_prev = 0.9 * fps + 0.1 / max(t1 - t_prev, 1e-6), t1
            if window is not None:
                banners = [banner] if banner and not preview else []  # (--ui: the app shows the prompt itself)
                if clap is not None:
                    s = clap.summary()
                    lines = [("Latency test: clap slowly, hands in view", 0.9),
                             (f"{len(clap.results)}/{clap.needed} claps measured" + (
                                 f"   camera delay {s['camera_ms']:.0f} ms" if s else ""), 0.8),
                             (f"ignored: {(mic.sounds_rejected if mic else 0) + clap.ignored} other sounds (speech, typing, "
                              f"claps not seen on camera)", 0.6)]
                    if clap.done and pipeline_ms:
                        total = s["camera_ms"] + float(np.median(pipeline_ms[-300:]))
                        lines.append((f"motion -> avatar data sent: about {total:.0f} ms  (press Q)", 0.8))
                    banners.append(lines)
                status = ["PAUSED" if paused else f"{fps:.0f} fps",
                          f"VMC -> {args.vmc}" if vmc else "VMC off",
                          f"3D viewer {viewer_url}" if viewer_url else "viewer off",
                          "CALIBRATING - hold still" if calib_samples is not None else
                          ("press C to calibrate" if not rig.calib else
                           "calibrated + vowels" if "vowels" in rig.calib else "calibrated - press V for vowels"),
                          "gestures: trained" if gestures.ok else "gestures: built-in",
                          f"mic {mic_state['level']:.0%}" if mic_state else "mic off",
                          f"VTS: {vts.state.split(' -')[0]}" if vts else "",
                          ("expr: " + ",".join(sorted(active_expr))) if active_expr else "",
                          "mirror avatar" if settings.mirror_avatar else "",
                          f"settings {viewer_url.split('/?')[0]}/settings.html" if viewer_url else ""]
                if args.record:
                    status.append("REC")
                text = "" if preview else "  |  ".join(s for s in status if s)  # (--ui: the app has a status bar)
                views = [(e.name + ("" if e.R is not None else " - press C to line up"), e.frame, e.res)
                         for e in multicam.extras] if multicam else []
                window.show(frame, res, avatar, banners, text, warn=not rig.calib, views=views)
                keys = window.keys()
                if ord("q") in keys:
                    break
                for key in keys:
                    if key == ord("c"):
                        start_calibration()
                    if key == ord("v"):
                        start_vowels()
                    if key == ord("g"):
                        nonlocal_gaze_start()
                    if key == ord("k"):
                        nonlocal_hand_cal_start()
                    if key == ord("p"):
                        set_paused(not paused)
                    if chr(key) in hotkeys:
                        toggle_expression(hotkeys[chr(key)])
                    toggles = {"f": "show_face_monitor", "m": "show_mesh", "b": "show_body", "h": "show_panel"}
                    if chr(key) in toggles:
                        field = toggles[chr(key)]
                        change_settings({field: not getattr(settings, field)})
                timing["draw"] += time.perf_counter() - t_e
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
            cap_fps = 30 if paused else settings.max_fps  # (paused: no tracking, so don't spin on a file / photo)
            if cap_fps > 0:  # frame-rate cap: leave CPU for games / OBS
                spare = 1 / cap_fps - (time.perf_counter() - t_a)
                if spare > 0:
                    time.sleep(spare)

    except KeyboardInterrupt:  # Ctrl+C in the console
        pass
    finally:  # whatever ended the loop: report, close the camera, microphone, helpers and files
        if frame_i:
            total = time.time() - t_start
            print(f"{frame_i} frames in {total:.1f} s = {frame_i / total:.1f} fps"
                  + (f" (camera delivers {cam.camera_fps:.0f} fps)" if cam else ""))
            print("average per frame: " + ", ".join(f"{k} {v / frame_i * 1000:.1f} ms" for k, v in timing.items()))
            if pipeline_ms:
                print(f"frame arrival -> avatar data sent: median {np.median(pipeline_ms):.0f} ms, "
                      f"95th percentile {np.percentile(pipeline_ms, 95):.0f} ms")
            if clap is not None:
                s = clap.summary()
                if s:
                    print(f"clap test: camera delay {s['camera_ms']:.0f} ms (spread {s['spread_ms']:.0f} ms, "
                          f"{s['claps']} claps); total motion -> avatar data sent about "
                          f"{s['camera_ms'] + np.median(pipeline_ms):.0f} ms")
                else:
                    print("clap test: no clap was matched (clap slowly with both hands in view)")
        if errors or rig.bad_frames:
            print("frames with a failure (skipped): " + ", ".join(
                [f"{k} {v}" for k, v in errors.items()] + ([f"NaN pose {rig.bad_frames}"] if rig.bad_frames else [])))
        if wilor:
            print(wilor.summary())
        # each one on its own: one that fails to close must not leave the camera or the GPU helper running
        for part, close in (("video file", cap and cap.release), ("camera", cam and cam.release),
                            ("microphone", mic and mic.close), ("WiLoR helper", wilor and wilor.close),
                            ("VTube Studio link", vts and vts.stop), ("hotkeys", global_keys and global_keys.stop),
                            ("recording", writer and writer.release), ("log", log and log.close),
                            ("extra cameras", multicam and multicam.release)):
            if close:
                guarded(f"closing the {part}", close)
        if writer:
            print(f"video saved -> {args.record}")
        if extra_rec is not None:
            guarded("closing the extra camera recordings", extra_rec.release)
        if args.bvh and motion:
            from avatarstudio.avatar.bvh import write_bvh
            rate = file_fps or len(motion) / max(time.time() - t_start, 1e-3)
            guarded("BVH export", write_bvh, args.bvh, motion, rate)
            print(f"motion saved -> {args.bvh} ({len(motion)} frames at {rate:.1f} fps)")
        if window is not None:
            window.close()


if __name__ == "__main__":
    main()
