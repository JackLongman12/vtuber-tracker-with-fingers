"""Launcher window: pick camera + microphone, see what's set up, start everything with buttons.

  AvatarStudio.exe            (packaged)  or   .venv\\Scripts\\python.exe app.py   (development)

Each tool runs in its own console window (so its prompts and messages are visible); the launcher stays open.
Minimizing it puts it in the tray (Start / Stop tracking, viewer, settings from the tray menu). It can start
tracking by itself when it opens, and add Start menu / desktop shortcuts to the one-file app.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from avatarstudio.ui import webwindow
from avatarstudio.core.os_support import open_folder
from avatarstudio.core.paths import app_dir, load_json, save_json, user_file

PREFS_PATH = user_file("launcher.json")
FPS_CHOICES = ["30 fps", "60 fps (if the camera can)"]


def command(*args):
    """How to run app.py's subcommands: the exe itself when packaged, python app.py in development."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    python = sys.executable.replace("pythonw.exe", "python.exe")  # tools need a console for their messages
    return [python, os.path.join(app_dir(), "app.py"), *args]


def find_cameras(max_index=4):
    import cv2
    found = []
    for i in range(max_index):  # (DirectShow on Windows, like camera.py; elsewhere OpenCV's usual one)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


def find_microphones():
    try:
        import sounddevice as sd
        default = sd.default.device[0]
        # each sound API lists the same microphones again: Windows' WASAPI, elsewhere the OS's default one
        # (Core Audio on macOS, ALSA / PulseAudio on Linux)
        api = "Windows WASAPI" if sys.platform == "win32" else sd.query_hostapis(sd.default.hostapi)["name"]
        mics = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and sd.query_hostapis(d["hostapi"])["name"] == api]
        return mics, default
    except Exception:
        return [], None


def setup_steps():
    """[(what, done, how)] in the order a new user should do them; the first not done is "next"."""
    calib = load_json(user_file("calibration.json"), {}) or {}
    if not isinstance(calib, dict):
        calib = {}
    return [
        ("Start tracking", None, "the button above; then turn on VMC receiving (port 39539) in your VTuber app, or "
                                 "open the 3D viewer"),
        ("Neutral pose", bool(calib), "sit as usual, look at the screen, press C in the studio window (Ctrl+Alt+C "
                                      "from anywhere)"),
        ("Vowels + voice", "vowels" in calib, "press V and say A, I, U, E, O when asked"),
        ("Hands: open, then fists", "hand_range" in calib or "hand_range_wilor" in calib,
         "press K, hold both hands open, then make fists (3 s each)"),
        ("Eye gaze (optional)", "gaze" in calib, "press G and follow the dots with your eyes, head still"),
        ("Your face (optional, ~2 min)", os.path.exists(user_file("face_refiner.json")),
         "Personalize > Train face: makes smiles, winks and brows read like yours"),
    ]


def info_lines():
    """[(text, ok)] extras worth knowing about."""
    out = []
    try:
        from avatarstudio.tracking.wilor_client import find_wilor
        wilor = find_wilor()[0] is not None
    except Exception:
        wilor = False
    out.append(("Finger model (WiLoR, GPU): installed" if wilor else "Finger model (WiLoR): not installed - "
                "fingers from MediaPipe", wilor))
    out.append(("Smoothing tuned to you" if os.path.exists(user_file("smoothing.json"))
                else "Smoothing: defaults (Personalize > Record a take, then Tune)",
                os.path.exists(user_file("smoothing.json"))))
    return out


class Tip:
    """A short explanation shown while the mouse is over a widget."""

    def __init__(self, widget, text):
        self.widget, self.text, self.win = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _):
        x, y = self.widget.winfo_rootx() + 10, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        tk.Label(self.win, text=self.text, justify="left", background="#ffffe8", relief="solid", borderwidth=1,
                 wraplength=320, padx=6, pady=3).pack()

    def hide(self, _):
        if self.win is not None:
            self.win.destroy()
            self.win = None


def app_exe():
    """The one-file app (set by its launcher), else None (development)."""
    exe = os.environ.get("AVATARSTUDIO_EXE")
    return exe if exe and os.path.exists(exe) else None


def make_shortcuts(exe):
    """Start menu and desktop shortcuts to the one-file app. Returns where they were made."""
    import ctypes.wintypes
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf)  # CSIDL_DESKTOPDIRECTORY
    places = [os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs"), buf.value]
    made = []
    for folder in places:
        if not folder or not os.path.isdir(folder):
            continue
        lnk = os.path.join(folder, "Avatar Studio.lnk")
        env = {**os.environ, "AVS_LNK": lnk, "AVS_EXE": exe, "AVS_DIR": os.path.dirname(exe)}
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:AVS_LNK); $s.TargetPath = $env:AVS_EXE; "
              "$s.WorkingDirectory = $env:AVS_DIR; $s.IconLocation = $env:AVS_EXE + ',0'; "
              "$s.Description = 'Webcam motion capture for VTuber avatars'; $s.Save()")
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=env,
                           capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and os.path.exists(lnk):
            made.append(lnk)
    return made


def tray_image():
    """The app icon (packaging/icon.ico in development, icon.ico next to the packaged app), else a drawn one."""
    from PIL import Image, ImageDraw
    here = app_dir()
    for p in (os.path.join(here, "icon.ico"), os.path.join(here, "packaging", "icon.ico")):
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGBA").resize((64, 64))
            except OSError:
                pass
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, 62, 62], radius=14, fill=(60, 120, 230, 255))
    d.ellipse([16, 12, 48, 44], fill=(255, 255, 255, 255))
    return img


class Launcher:
    def __init__(self, root):
        self.root = root
        build = os.path.basename(os.getcwd()) if getattr(sys, "frozen", False) else ""
        root.title("Avatar Studio" + (f"  ({build.replace('AvatarStudio_', 'version ')})" if build else ""))
        root.resizable(False, False)
        self.prefs = self._load_prefs()
        self.studio, self.tray, self.calls = None, None, queue.Queue()
        pad = dict(padx=10, pady=4)

        top = ttk.Frame(root, padding=12)
        top.grid(sticky="nsew")
        ttk.Label(top, text="Avatar Studio", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                  sticky="w")
        ttk.Label(top, text="Webcam tracking for VTuber avatars (VMC port 39539)").grid(row=1, column=0, columnspan=2,
                                                                                         sticky="w")

        dev = ttk.LabelFrame(top, text="Devices", padding=8)
        dev.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Label(dev, text="Camera").grid(row=0, column=0, sticky="w")
        self.camera = ttk.Combobox(dev, state="readonly", width=24, values=["searching..."])
        self.camera.current(0)
        self.camera.grid(row=0, column=1, **pad)
        self.fps = ttk.Combobox(dev, state="readonly", width=22, values=FPS_CHOICES)
        self.fps.current(FPS_CHOICES.index(self.prefs.get("fps")) if self.prefs.get("fps") in FPS_CHOICES else 0)
        self.fps.grid(row=0, column=2, **pad)
        ttk.Label(dev, text="Microphone").grid(row=1, column=0, sticky="w")
        self.mics, default_mic = find_microphones()
        names = ["Windows default"] + [n for _, n in self.mics] + ["No microphone"]
        self.mic = ttk.Combobox(dev, state="readonly", width=52, values=names)
        saved = self.prefs.get("mic")
        self.mic.current(names.index(saved) if saved in names else 0)
        self.mic.grid(row=1, column=1, columnspan=2, sticky="w", **pad)
        ttk.Label(dev, text="Extra cameras\n(experimental)").grid(row=2, column=0, sticky="w")
        self.extra_cams = []
        for k, key in enumerate(("camera2", "camera3")):
            box = ttk.Combobox(dev, state="readonly", width=24 if k == 0 else 22, values=["none"])
            box.current(0)
            box.grid(row=2, column=1 + k, **pad)
            self.extra_cams.append((key, box))
        threading.Thread(target=self._fill_cameras, daemon=True).start()

        run = ttk.Frame(top, padding=(0, 8))
        run.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.state_lbl = tk.Label(run, text="", font=("Segoe UI", 11, "bold"))
        self.state_lbl.grid(row=0, column=0, sticky="w")
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 8))
        self.toggle_btn = ttk.Button(run, text="Start tracking", style="Big.TButton", command=self.toggle_studio)
        self.toggle_btn.grid(row=0, column=1, padx=16)
        Tip(self.toggle_btn, "Starts the studio with the camera and microphone above: your avatar moves in your "
                             "VTuber app (VMC, port 39539) and in the 3D viewer. Its own window shows the camera; "
                             "press Q there (or Stop here) to end.")
        self.autostart = tk.BooleanVar(value=bool(self.prefs.get("autostart", False)))
        ttk.Checkbutton(run, text="Start tracking when Avatar Studio opens", variable=self.autostart,
                        command=self._save_prefs).grid(row=0, column=2, sticky="w")

        groups = ttk.Frame(top)
        groups.grid(row=4, column=0, columnspan=3, sticky="ew")
        buttons = {
            "Open": [("3D viewer", lambda: webwindow.open_window("viewer"),
                      "Your avatar in its own window (drop your .vrm file onto it). Load it there once per "
                      "session: fingertip contacts then use your avatar's own proportions. For OBS: browser "
                      "source http://127.0.0.1:8000"),
                     ("Settings page", lambda: webwindow.open_window("settings"),
                      "Everything adjustable while tracking runs: smoothing, look-ahead, lip sync, expressions, "
                      "pause, calibration buttons, profiles.")],
            "Check": [("Test avatar setup", lambda: self.run("studio", "--test-pattern"),
                       "Sends 24 known poses (no camera needed) so you can check your VTuber app and model "
                       "move the right way."),
                      ("Measure latency", lambda: self.run("studio", "--latency-test", camera=True, mic=True),
                       "Clap slowly 6 times with both hands in view: measures the real delay from your motion to "
                       "the avatar.")],
            "Personalize": [("Train face (~2 min)", lambda: self.run("train-face"),
                             "Guided expressions that teach it your face (smiles, winks, brows, tongue, cheeks)."),
                            ("Record a take (1 min)", self.record_take,
                             "Records the camera for a minute; used to tune smoothing to your movements."),
                            ("Tune smoothing", self.tune, "Learns how much to smooth from your recorded take."),
                            ("Custom gestures", self.gesture_help, "How to train your own hand gestures.")],
        }
        if app_exe() and sys.platform == "win32":  # (Start menu + desktop .lnk shortcuts: Windows only)
            buttons["Personalize"].append(("Shortcut on desktop + Start", self.shortcuts,
                                           "Adds Avatar Studio to the Start menu and the desktop."))
            buttons["Check"].append(("Uninstall...", self.uninstall,
                                     "Removes Avatar Studio from this PC (your data is kept unless you choose)."))
        for col, (title, items) in enumerate(buttons.items()):
            box = ttk.LabelFrame(groups, text=title, padding=6)
            box.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            for r, (text, cmd, tip) in enumerate(items):
                b = ttk.Button(box, text=text, command=cmd, width=24)
                b.grid(row=r, column=0, padx=4, pady=3)
                Tip(b, tip)
        box = ttk.LabelFrame(groups, text="Motion capture (from a recording, for animation and movement analysis)",
                             padding=6)
        box.grid(row=1, column=0, columnspan=len(buttons), sticky="ew", pady=(8, 0))
        for c, (text, cmd, tip) in enumerate([
                ("Record for mocap", self.record_mocap, "Records the camera until you press Q in the studio window."),
                ("Mocap a video", self.mocap,
                 "Turns a recording into clean motion: BVH for Blender / Unity, 3D body points and joint angles "
                 "(CSV), face animation curves, a video with the skeleton drawn on. The whole take is filtered at "
                 "once, so there's no delay and much less jitter than live."),
                ("Open mocap results", self.open_mocap, "The folder with the last mocap results.")]):
            b = ttk.Button(box, text=text, command=cmd, width=24)
            b.grid(row=0, column=c, padx=4, pady=3)
            Tip(b, tip)

        self.status = ttk.LabelFrame(top, text="Getting started", padding=8)
        self.status.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.status_labels = []
        opts = ttk.Frame(top)
        opts.grid(row=6, column=0, columnspan=3, sticky="w")
        self.to_tray = tk.BooleanVar(value=bool(self.prefs.get("tray", True)))
        try:
            # Windows only: on macOS the tray icon would need this window's own event loop (and minimizing to the
            # Dock is usual there); Linux trays need a GTK loop. There the window just minimizes.
            if sys.platform != "win32":
                raise ImportError
            import pystray  # noqa: F401
            ttk.Checkbutton(opts, text="Minimize to the tray", variable=self.to_tray,
                            command=self._save_prefs).grid(row=0, column=0, sticky="w")
        except ImportError:
            self.to_tray.set(False)
        ttk.Label(top, text="Studio window keys: C neutral pose · V vowels · K hands · G eye gaze · P pause · "
                            "1-9 expressions · Q quit.   From any window: Ctrl+Alt+C recalibrate, Ctrl+Alt+P pause.",
                  foreground="#666").grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._show_state()
        self.refresh()
        root.bind("<Unmap>", self._on_minimize)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(200, self._pump)
        if self.autostart.get():
            root.after(1500, self.start_studio)

    # ---------------------------------------------------------------- devices and prefs
    def _load_prefs(self):
        p = load_json(PREFS_PATH, {})
        return p if isinstance(p, dict) else {}

    def _save_prefs(self):
        self.prefs.update({"camera": self.camera.get(), "mic": self.mic.get(), "fps": self.fps.get(),
                           "autostart": self.autostart.get(), "tray": self.to_tray.get(),
                           **{key: box.get() for key, box in self.extra_cams}})
        try:
            save_json(PREFS_PATH, self.prefs)
        except OSError:
            pass

    def _fill_cameras(self):
        cams = find_cameras()
        names = [f"Camera {i}" for i in cams] or ["Camera 0"]

        def apply():
            self.camera.configure(values=names)
            saved = self.prefs.get("camera")
            self.camera.current(names.index(saved) if saved in names else 0)
            for key, box in self.extra_cams:  # 2nd / 3rd camera: "none" unless picked before and still there
                box.configure(values=["none"] + names)
                saved = self.prefs.get(key)
                box.current((["none"] + names).index(saved) if saved in names else 0)
        self.calls.put(apply)

    def device_args(self, camera=True, mic=True):
        args = []
        if camera:
            cam = self.camera.get()
            if cam.startswith("Camera "):
                args += ["--camera", cam.split()[1]]
            if self.fps.get().startswith("60"):
                args += ["--fps", "60"]
            used = {cam}
            for key, box in self.extra_cams:
                extra = box.get()
                if extra.startswith("Camera ") and extra not in used:
                    args += [f"--{key}", extra.split()[1]]
                    used.add(extra)
        if mic:
            m = self.mic.get()
            if m == "No microphone":
                args.append("--no-mic")
            elif m != "Windows default":
                idx = next((i for i, n in self.mics if n == m), None)
                if idx is not None:
                    args += ["--mic-device", str(idx)]
        return args

    # ---------------------------------------------------------------- running things
    def run(self, *args, camera=None, mic=None):
        """Start a tool in its own console window (Windows). On macOS / Linux there's no such window: its messages
        go to logs/<tool>.log in your data folder."""
        self._save_prefs()
        extra = []
        if args and args[0] == "studio":
            extra = self.device_args(camera is not False, mic is not False)
        elif args and args[0] == "train-face":
            extra = self.device_args(camera=True, mic=False)
        log = None
        if sys.platform == "win32":
            how = dict(creationflags=subprocess.CREATE_NEW_CONSOLE, env={**os.environ, "AVATAR_PAUSE": "1"})
        else:
            try:
                os.makedirs(user_file("logs"), exist_ok=True)
                log = open(user_file(os.path.join("logs", f"{args[0]}.log")), "w")
            except OSError:
                pass
            how = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT if log else None)
        try:
            proc = subprocess.Popen(command(*args, *extra), **how)
        except OSError as e:
            messagebox.showerror("Avatar Studio", f"Could not start it:\n{e}")
            return None
        finally:
            if log:
                log.close()  # (the tool has its own copy)
        threading.Thread(target=self._wait_and_refresh, args=(proc,), daemon=True).start()
        return proc

    def _wait_and_refresh(self, proc):
        proc.wait()
        self.calls.put(self.refresh)

    def _pump(self):
        """Run what other threads (camera search, finished tools, the tray menu) asked of the window: Tk itself
        must only be used from this thread."""
        while not self.calls.empty():
            try:
                self.calls.get_nowait()()
            except Exception as e:
                print(f"launcher: {e}")
        self.root.after(200, self._pump)

    def toggle_studio(self):
        if self.studio and self.studio.poll() is None:
            self.stop_studio()
        else:
            self.start_studio()

    def start_studio(self):
        if self.studio and self.studio.poll() is None:
            return
        self.studio = self.run("studio")
        self._show_state()
        if self.studio is not None:
            self._watch_studio()

    def _show_state(self):
        running = bool(self.studio and self.studio.poll() is None)
        self.state_lbl.configure(text="\u25cf  Tracking - running" if running else "\u25cb  Not tracking",
                                 foreground="#1a7f37" if running else "#888")
        self.toggle_btn.configure(text="Stop tracking" if running else "Start tracking")

    def _watch_studio(self):
        if self.studio and self.studio.poll() is None:
            self.root.after(500, self._watch_studio)
        else:
            self._show_state()
            self.refresh()

    def stop_studio(self):
        if self.studio and self.studio.poll() is None:
            self.studio.terminate()

    def record_take(self):
        messagebox.showinfo("Record a take", "The studio opens and records the camera to take.mp4.\n\n"
                            "For about a minute: talk, turn your head, wave, cross your hands, make fists, "
                            "then sit still for 10 seconds. Press Q in the studio window to stop.")
        self.run("studio", "--record", user_file("take.mp4"), "--no-vmc")

    def tune(self):
        if not os.path.exists(user_file("take.mp4")):
            messagebox.showinfo("Tune smoothing", "Record a take first.")
            return
        self.run("tune-smoothing", user_file("take.mp4"))

    def record_mocap(self):
        messagebox.showinfo("Record for mocap", "The studio opens and records the camera. Do your moves (whole "
                            "body in view if you want the legs), then press Q in the studio window to stop.\n\n"
                            "Then press 'Mocap a video' to turn the recording into motion.")
        path = user_file(time.strftime("mocap_%Y%m%d-%H%M%S.mp4"))
        self.prefs["mocap_video"] = path
        self.run("studio", "--record", path, "--no-vmc")

    def mocap(self):
        from tkinter import filedialog
        last = self.prefs.get("mocap_video") or user_file("take.mp4")
        path = filedialog.askopenfilename(
            title="Video to motion-capture", initialdir=os.path.dirname(last),
            initialfile=os.path.basename(last) if os.path.exists(last) else "",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self.prefs["mocap_video"] = path
            self.run("mocap", path)

    def open_mocap(self):
        last = self.prefs.get("mocap_video")
        folder = os.path.splitext(last)[0] + "_mocap" if last else None
        if folder and os.path.isdir(folder):
            open_folder(folder)
        else:
            messagebox.showinfo("Mocap results", "No results yet: press 'Mocap a video' first.")

    def gesture_help(self):
        messagebox.showinfo("Custom gestures", "Custom hand gestures are recorded from a console:\n\n"
                            "AvatarStudio.exe gestures record NAME\nAvatarStudio.exe gestures train\n\n"
                            "Then map the gesture to an expression in hotkeys.json.")

    def shortcuts(self):
        try:
            made = make_shortcuts(app_exe())
        except Exception as e:
            made, err = [], str(e)
        else:
            err = ""
        if made:
            messagebox.showinfo("Shortcuts", "Made:\n" + "\n".join(made))
        else:
            messagebox.showwarning("Shortcuts", f"Could not make the shortcuts. {err}")

    def uninstall(self):
        from avatarstudio.core import uninstall
        p = uninstall.plan()
        total = sum(mb for _, mb in p["app"])
        if not messagebox.askokcancel("Uninstall Avatar Studio", f"Remove Avatar Studio from this PC (about {total} MB)?"
                                      "\n\nThe app, its unpacked files, caches and shortcuts are deleted."):
            return
        data = (f"\n\n{p['data'][0]} ({p['data'][1]} MB)" if p["data"] else "")
        answer = messagebox.askyesnocancel("Your data", "Also delete your data: calibration, settings, recordings, "
                                           "face and gesture training, profiles? It goes to the Recycle Bin."
                                           f"{data}\n\nYes: delete it too.  No: keep it.")
        if answer is None:
            return
        self.stop_studio()
        r = uninstall.start(answer)
        if not r["ok"]:
            messagebox.showwarning("Uninstall", r["error"])
            return
        if self.tray is not None:  # (closing without saving the window's prefs: they're being removed)
            self.tray.stop()
        self.root.destroy()

    def refresh(self):
        """The checklist: done steps ticked, the first open one marked as next, then the extras."""
        for w in self.status_labels:
            w.destroy()
        self.status_labels = []
        running = bool(self.studio and self.studio.poll() is None)
        steps = setup_steps()
        steps[0] = (steps[0][0], running or any(d for _, d, _ in steps[1:]), steps[0][2])
        nxt = next((i for i, (_, done, _) in enumerate(steps) if not done), None)
        for i, (what, done, how) in enumerate(steps):
            mark = "\u2714" if done else ("\u25b6" if i == nxt else "\u25cb")
            color = "#1a7f37" if done else ("#0b5cad" if i == nxt else "#888")
            text = f"{mark}  {i + 1}. {what}" + (f"  - next: {how}" if i == nxt else "")
            lab = ttk.Label(self.status, text=text, foreground=color,
                            font=("Segoe UI", 9, "bold") if i == nxt else None, wraplength=640)
            lab.grid(row=i, column=0, sticky="w", pady=1)
            self.status_labels.append(lab)
        for k, (text, ok) in enumerate(info_lines()):
            lab = ttk.Label(self.status, text=("\u2714  " if ok else "\u25cb  ") + text,
                            foreground="#1a7f37" if ok else "#888")
            lab.grid(row=len(steps) + k, column=0, sticky="w", pady=(6 if k == 0 else 1, 1))
            self.status_labels.append(lab)

    # ---------------------------------------------------------------- tray
    def _on_minimize(self, event):
        if event.widget is self.root and self.root.state() == "iconic" and self.to_tray.get():
            self.root.after(0, self._to_tray)

    def _to_tray(self):
        try:
            import pystray
        except ImportError:
            return
        if self.tray is not None:
            return
        do = lambda fn: (lambda icon, item: self.calls.put(fn))  # menu clicks arrive on the tray's thread
        running = lambda item: bool(self.studio and self.studio.poll() is None)
        menu = pystray.Menu(
            pystray.MenuItem("Show Avatar Studio", do(self._from_tray), default=True),
            pystray.MenuItem("Start tracking", do(self.start_studio), enabled=lambda item: not running(item)),
            pystray.MenuItem("Stop tracking", do(self.stop_studio), enabled=running),
            pystray.MenuItem("3D viewer", do(lambda: webwindow.open_window("viewer"))),
            pystray.MenuItem("Settings page", do(lambda: webwindow.open_window("settings"))),
            pystray.MenuItem("Quit launcher", do(self.quit)))
        self.tray = pystray.Icon("AvatarStudio", tray_image(), "Avatar Studio", menu)
        self.tray.run_detached()
        self.root.withdraw()

    def _from_tray(self):
        if self.tray is not None:
            self.tray.stop()
            self.tray = None
        self.root.deiconify()
        self.root.lift()

    def quit(self):
        self._save_prefs()
        if self.tray is not None:
            self.tray.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
