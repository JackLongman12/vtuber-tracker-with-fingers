"""Avatar Studio in one window: the camera picture, the 3D avatar, the settings and the tools (viewer/app.html).

  AvatarStudio.exe            (development: .venv\\Scripts\\python.exe app.py)

The window is a web page in the OS's web view, like the viewer windows (webwindow.py). Tracking runs as before in
its own process, the engine (studio.py --ui): no window of its own; its camera picture (camera.mjpg on :8000) and
the avatar data (ws://127.0.0.1:8765) come into this window, and the page sends it settings, keys and button
presses the same way. A separate process keeps the window responsive whatever tracking does, and VMC / OBS work as
before. This module starts and stops the engine and the tools, lists cameras and microphones, keeps the device
choices (launcher.json, shared with the old launcher), changes settings while tracking is off, and on Windows puts
the app in the tray when minimized. Where no web view is available the old launcher window opens instead.
"""
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

from avatarstudio.ui import launcher
from avatarstudio.core.paths import load_json, save_json, user_file
from avatarstudio.version import GITHUB_REPO, VERSION

FPS_CHOICES = launcher.FPS_CHOICES
CALIB_PATH = user_file("calibration.json")
NO_WINDOW = dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else {}
# MediaPipe's / TensorFlow Lite's own log lines (kept in the log files, not shown in the app)
NOISE = re.compile(r"^([WIEF]\d{4} \d|WARNING: All log messages before absl|WARNING: Logging before InitGoogle|"
                   r"INFO: Created TensorFlow Lite)")


class Process:
    """A tool or the engine running in the background: its messages are kept (the page shows the last ones) and
    written to logs/<name>.log in your data folder."""

    def __init__(self, name, args, mode=None):
        self.name, self.mode, self.started = name, mode or name, time.time()
        self.lines, self.count = collections.deque(maxlen=400), 0  # (count: lines so far, for "what's new")
        os.makedirs(user_file("logs"), exist_ok=True)
        self.log = open(user_file(os.path.join("logs", f"{name}.log")), "w", encoding="utf-8")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        env.pop("AVATAR_PAUSE", None)
        self.proc = subprocess.Popen(launcher.command(*args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                     env=env, **NO_WINDOW)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                self.log.write(line + "\n")
                self.log.flush()
                if not NOISE.match(line):
                    self.lines.append(line)
                    self.count += 1
        self.log.close()

    def running(self):
        return self.proc.poll() is None

    def state(self, n=12):
        return {"name": self.name, "mode": self.mode, "id": int(self.started * 1000), "running": self.running(),
                "code": self.proc.poll(), "since": round(time.time() - self.started, 1), "count": self.count,
                "lines": list(self.lines)[-n:]}

    def stop(self, gently=None, wait=8.0):
        """gently(): asks it to finish on its own (the engine saves recordings and motion when it quits like
        that); after `wait` seconds it's terminated."""
        if not self.running():
            return
        if gently:
            try:
                gently()
                self.proc.wait(wait)
                return
            except Exception:
                pass
        self.proc.terminate()
        try:
            self.proc.wait(3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def tell_engine(message):
    """One message to the engine over its WebSocket (like the page sends them)."""
    from websockets.sync.client import connect
    with connect("ws://127.0.0.1:8765", open_timeout=2) as ws:
        ws.send(json.dumps(message))


class AppApi:
    """What app.html can ask of this process (window.pywebview.api.*; every call runs on its own thread).
    Names starting with _ stay hidden from the page."""

    def __init__(self):
        self._window, self._engine, self._jobs, self._tray, self._fullscreen = None, None, {}, None, False
        self._update = {"state": "idle"}  # (updates: see the section below)
        self._prefs = load_json(launcher.PREFS_PATH, {}) or {}
        self._prefs = self._prefs if isinstance(self._prefs, dict) else {}
        self._mics, self._cameras = [], None

    def attach(self, window):  # (webwindow.main, before the window shows; not when one was open already)
        self._window = window
        if sys.platform == "win32":
            window.events.minimized += self._to_tray
        self._mics = launcher.find_microphones()[0]
        # (cameras are looked for in the background: opening each one takes a moment)
        threading.Thread(target=self._find_cameras, daemon=True).start()
        if self._prefs.get("check_updates", True):  # (one request to GitHub per start, in the background)
            threading.Thread(target=self._check_updates, daemon=True).start()

    def _find_cameras(self):
        """In a process of its own (app.py cameras): opening the cameras loads their drivers, and one still busy
        when the window closes kept this process from ever ending (a laptop camera's vendor plug-in; the process
        then also held the app's single-instance name). The search process just finishes on its own."""
        try:
            out = subprocess.run(launcher.command("cameras"), capture_output=True, text=True, timeout=30,
                                 stdin=subprocess.DEVNULL, **NO_WINDOW).stdout
            self._cameras = json.loads(out.strip().splitlines()[-1])
        except Exception:
            self._cameras = []

    # ------------------------------------------------------------ state for the page
    def get_state(self):
        e = self._engine
        return {"engine": e.state() if e else None,
                "jobs": {n: j.state() for n, j in self._jobs.items()},
                "devices": {"cameras": [f"Camera {i}" for i in self._cameras] if self._cameras is not None else None,
                            "mics": [n for _, n in self._mics]},
                "prefs": self._prefs, "fps_choices": FPS_CHOICES, "platform": sys.platform,
                "tray": sys.platform == "win32", "version": VERSION, "packaged": self._app_file() is not None,
                "update": self._update, "updates_from": GITHUB_REPO}

    def set_prefs(self, values):
        self._prefs.update({k: v for k, v in values.items() if k in (
            "camera", "mic", "fps", "camera2", "camera3", "autostart", "tray", "layout", "check_updates")})
        self._save_prefs()
        return self._prefs

    def _save_prefs(self):
        try:
            save_json(launcher.PREFS_PATH, self._prefs)
        except OSError:
            pass

    def checklist(self):
        steps = [{"what": w, "done": bool(d) if d is not None else None, "how": h} for w, d, h in launcher.setup_steps()]
        steps[0]["done"] = self._engine_running() or any(s["done"] for s in steps[1:])
        return {"steps": steps, "info": [{"text": t, "ok": ok} for t, ok in launcher.info_lines()]}

    # ------------------------------------------------------------ the engine
    def _engine_running(self):
        return self._engine is not None and self._engine.running()

    def _device_args(self, camera=True, mic=True):
        p, args = self._prefs, []
        if camera:
            cam = p.get("camera") or "Camera 0"
            if cam.startswith("Camera "):
                args += ["--camera", cam.split()[1]]
            if str(p.get("fps", "")).startswith("60"):
                args += ["--fps", "60"]
            used = {cam}
            for key in ("camera2", "camera3"):
                extra = p.get(key) or "none"
                if extra.startswith("Camera ") and extra not in used:
                    args += [f"--{key}", extra.split()[1]]
                    used.add(extra)
        if mic:
            m = p.get("mic") or "Windows default"
            if m == "No microphone":
                args.append("--no-mic")
            else:
                idx = next((i for i, n in self._mics if n == m), None)
                if idx is not None:
                    args += ["--mic-device", str(idx)]
        return args

    def start(self, mode="live", arg=None):
        """Start the engine: live tracking, or one of its other modes (recording a take, the avatar test, the
        latency test), or a tool that needs the camera and shows on the Live page the same way (face training,
        recording a gesture; arg = its name). Only one of them at a time: they share the camera."""
        if self._engine_running():
            if self._engine.mode == mode:
                return {"ok": True}
            self.stop()
        if mode == "gesture" and not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", str(arg or "")):
            return {"ok": False, "error": "Gesture names use letters, digits, - and _ (no spaces)"}
        cam = self._device_args(mic=False)[:2]  # (--camera N)
        studio = {"live": self._device_args(),
                  "record_take": self._device_args(mic=False) + ["--record", user_file("take.mp4"), "--no-vmc"],
                  "record_mocap": self._device_args(mic=False) + ["--record", self._new_mocap_path(), "--no-vmc"]
                  if mode == "record_mocap" else [],
                  "test_pattern": ["--test-pattern", "--headless"],
                  "latency": self._device_args() + ["--latency-test"]}
        if mode in studio:
            name, args = "studio", ["studio", "--ui", *studio[mode], *ENGINE_EXTRA]
        elif mode == "train_face":
            name, args = "train-face", ["train-face", "--ui", *cam, *TOOL_EXTRA]
        elif mode == "gesture":
            name, args = "gesture-record", ["gestures", "record", arg, "--ui", *cam, *TOOL_EXTRA]
        else:
            return {"ok": False, "error": f"unknown mode {mode}"}
        try:
            self._engine = Process(name, args, mode)
        except OSError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def stop(self):
        """Ends the engine like Q in its old window (a recording is finished and saved), else after 8 s for sure.
        (The avatar test doesn't listen, so it's just ended.)"""
        e = self._engine
        if e is not None:
            e.stop(gently=None if e.mode == "test_pattern" else lambda: tell_engine({"type": "key", "key": "q"}))
        return {"ok": True}

    def _new_mocap_path(self):
        self._prefs["mocap_video"] = user_file(time.strftime("mocap_%Y%m%d-%H%M%S.mp4"))
        self._save_prefs()
        return self._prefs["mocap_video"]

    # ------------------------------------------------------------ tools
    def run_tool(self, tool):
        """Background tools (no camera): smoothing from your take, motion capture of a video (asks which), training
        the gestures. Their messages and progress show in the app."""
        if tool == "train-gestures":
            args = ["gestures", "train"]
        elif tool == "tune-smoothing":
            if not os.path.exists(user_file("take.mp4")):
                return {"ok": False, "error": "Record a take first"}
            args = ["tune-smoothing", user_file("take.mp4")]
        elif tool == "mocap":
            path = self._pick_video()
            if not path:
                return {"ok": False, "cancelled": True}
            self._prefs["mocap_video"] = path
            self._save_prefs()
            args = ["mocap", path]
        else:
            return {"ok": False, "error": f"unknown tool {tool}"}
        old = self._jobs.get(tool)
        if old and old.running():
            return {"ok": False, "error": "It's already running"}
        try:
            self._jobs[tool] = Process(tool, args)
        except OSError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def stop_tool(self, tool):
        j = self._jobs.get(tool)
        if j:
            j.stop()
        return {"ok": True}

    # ------------------------------------------------------------ your gestures (gestures.py's files)
    def gestures_state(self):
        import numpy as np
        from avatarstudio.personal import gestures  # (OpenCV, scikit-learn: only loaded when the page asks)
        names = []
        if os.path.isdir(gestures.DATA):
            for f in sorted(os.listdir(gestures.DATA)):
                if f.endswith(".npy"):
                    try:
                        n = int(np.load(os.path.join(gestures.DATA, f), mmap_mode="r").shape[0])
                    except (OSError, ValueError):
                        n = 0
                    names.append({"name": f[:-4], "samples": n})
        newest = max((os.path.getmtime(os.path.join(gestures.DATA, f["name"] + ".npy")) for f in names), default=0)
        trained = os.path.exists(gestures.MODEL)
        actions = load_json(gestures.ACTIONS, None) if os.path.exists(gestures.ACTIONS) else dict(gestures.DEFAULT_ACTIONS)
        from avatarstudio.core.settings import load_hotkeys
        return {"gestures": names, "trained": trained,
                "up_to_date": trained and os.path.getmtime(gestures.MODEL) >= newest,
                "actions": {k: v for k, v in (actions or {}).items() if not k.startswith("_")},
                "expressions": sorted(set(load_hotkeys().values()) | {"Joy", "Angry", "Sorrow", "Fun", "Surprised"})}

    def set_gesture_action(self, name, expression):
        """Which expression a gesture shows (actions.json; "" = none)."""
        from avatarstudio.personal import gestures
        actions = load_json(gestures.ACTIONS, None) if os.path.exists(gestures.ACTIONS) else dict(gestures.DEFAULT_ACTIONS)
        actions = actions if isinstance(actions, dict) else dict(gestures.DEFAULT_ACTIONS)
        if expression:
            actions[name] = expression
        else:
            actions.pop(name, None)
        os.makedirs(gestures.DIR, exist_ok=True)
        save_json(gestures.ACTIONS, actions, indent=2)
        return {"ok": True}

    def delete_gesture(self, name):
        from avatarstudio.personal import gestures
        path = os.path.join(gestures.DATA, f"{name}.npy")
        if os.path.dirname(os.path.abspath(path)) != os.path.abspath(gestures.DATA) or not os.path.exists(path):
            return {"ok": False, "error": "No such gesture"}
        os.remove(path)
        return {"ok": True}

    def _pick_video(self):
        import webview
        last = self._prefs.get("mocap_video") or user_file("take.mp4")
        picked = self._window.create_file_dialog(
            webview.FileDialog.OPEN, directory=os.path.dirname(last),
            file_types=("Videos (*.mp4;*.avi;*.mov;*.mkv;*.webm)", "All files (*.*)"))
        return picked[0] if picked else None

    def open_mocap_results(self):
        from avatarstudio.core.os_support import open_folder
        last = self._prefs.get("mocap_video")
        folder = os.path.splitext(last)[0] + "_mocap" if last else None
        if folder and os.path.isdir(folder):
            open_folder(folder)
            return {"ok": True}
        return {"ok": False, "error": "No results yet: motion-capture a video first"}

    def open_data_folder(self, sub=""):
        from avatarstudio.core.os_support import open_folder
        path = user_file(sub) if sub else os.path.dirname(user_file("x"))
        os.makedirs(path, exist_ok=True)
        open_folder(path)
        return {"ok": True}

    # ------------------------------------------------------------ settings while tracking is off
    # (while it runs, the page changes them through the engine, which also applies them at once)
    def get_settings(self):
        from avatarstudio.core.settings import Settings, list_profiles, load_hotkeys
        return {"settings": Settings.load().as_dict(), "profiles": list_profiles(), "hotkeys": load_hotkeys()}

    def set_settings(self, values):
        from avatarstudio.core.settings import Settings
        s = Settings.load()
        s.update(values)
        s.save()
        return s.as_dict()

    def reset_settings(self):
        from avatarstudio.core.settings import Settings
        Settings().save()
        return Settings().as_dict()

    def save_profile(self, name):
        from avatarstudio.core.settings import Settings, save_profile
        return save_profile(name, Settings.load(), load_json(CALIB_PATH))

    def load_profile(self, name):
        from avatarstudio.core.settings import Settings, load_profile
        try:
            p = load_profile(name)
        except OSError:
            return {"ok": False, "error": "No such profile"}
        s = Settings.load()
        s.update(p.get("settings", {}))
        s.save()
        if p.get("calibration"):
            save_json(CALIB_PATH, p["calibration"])
        return {"ok": True}

    # ------------------------------------------------------------ updates (core/updater.py, from GitHub releases)
    def _app_file(self):
        """The one-file AvatarStudio.exe this app was started from (its launcher says), else None (development)."""
        exe = os.environ.get("AVATARSTUDIO_EXE")
        return exe if getattr(sys, "frozen", False) and exe and os.path.exists(exe) else None

    def check_updates(self):
        if self._update["state"] not in ("checking", "downloading", "installing"):
            threading.Thread(target=self._check_updates, daemon=True).start()
        return self._update

    def _check_updates(self):
        from avatarstudio.core import updater
        if not GITHUB_REPO:
            self._update = {"state": "off"}
            return
        self._update = {"state": "checking"}
        try:
            info = updater.check()
            self._update = {"state": "available", "info": info} if info else {"state": "latest"}
            self._prefs["update_checked"] = time.time()
            self._save_prefs()
        except Exception as e:  # (offline, GitHub unreachable, ...): try again next time
            self._update = {"state": "error", "error": f"Couldn't check for updates ({str(e)[:120]})"}

    def install_update(self):
        """Download the update (with progress), then swap it in and restart: tracking stops first."""
        info, app = self._update.get("info"), self._app_file()
        if self._update["state"] != "available" or not info:
            return {"ok": False, "error": "No update to install"}
        if not app:
            return {"ok": False, "error": "Updates install into the packaged app; in development use git pull"}
        threading.Thread(target=self._install_update, args=(info, app), daemon=True).start()
        return {"ok": True}

    def _install_update(self, info, app):
        from avatarstudio.core import updater

        def progress(done, total):
            self._update = {"state": "downloading", "info": info, "progress": done / total if total else 0,
                            "done_mb": round(done / 2 ** 20), "total_mb": round(total / 2 ** 20)}
        try:
            progress(0, info.get("size", 0))
            path = updater.download(info, os.path.dirname(app), progress)
            self._update = {"state": "installing", "info": info}
            self.stop()
            for j in self._jobs.values():
                j.stop()
            updater.install(path, app)
        except Exception as e:
            self._update = {"state": "available", "info": info, "error": f"The update didn't install: {e}"}
            return
        time.sleep(1.0)  # (the page shows "Restarting..." for a moment)
        self._window.destroy()  # (main() then ends this process; the new version is starting)

    # ------------------------------------------------------------ uninstall (core/uninstall.py)
    def uninstall_plan(self):
        from avatarstudio.core import uninstall
        return uninstall.plan()

    def uninstall(self, delete_data=False):
        """Tracking and tools stop, the clean-up script starts, the window closes (the script waits for that)."""
        from avatarstudio.core import uninstall
        if not uninstall.app_file():
            return uninstall.start(bool(delete_data))  # (the error: development run / not Windows)
        self.stop()
        for j in self._jobs.values():
            j.stop()
        r = uninstall.start(bool(delete_data))
        if r["ok"]:
            threading.Thread(target=lambda: (time.sleep(1.5), self._window.destroy()), daemon=True).start()
        return r

    # ------------------------------------------------------------ window
    def fullscreen(self, on):
        """Full screen for the eye-gaze calibration dots (and back)."""
        if self._window is not None and bool(on) != self._fullscreen:
            self._fullscreen = bool(on)
            self._window.toggle_fullscreen()
        return self._fullscreen

    def _to_tray(self):
        """Windows: minimized with "Minimize to the tray" on -> just an icon by the clock (its menu starts / stops
        tracking)."""
        if not self._prefs.get("tray", True) or self._tray is not None:
            return
        try:
            import pystray
        except ImportError:
            return

        def show(icon=None, item=None):
            self._tray, tray = None, self._tray
            if tray is not None:
                tray.stop()
            self._window.show()
            self._window.restore()

        def quit_app(icon, item):
            show()
            self._window.destroy()
        menu = pystray.Menu(
            pystray.MenuItem("Show Avatar Studio", show, default=True),
            pystray.MenuItem("Start tracking", lambda i, it: self.start(), enabled=lambda it: not self._engine_running()),
            pystray.MenuItem("Stop tracking", lambda i, it: self.stop(), enabled=lambda it: self._engine_running()),
            pystray.MenuItem("Quit Avatar Studio", quit_app))
        self._tray = pystray.Icon("AvatarStudio", launcher.tray_image(), "Avatar Studio", menu)
        self._tray.run_detached()
        self._window.hide()

    def _shutdown(self):
        """The window closed: tracking and the tools end with it."""
        if self._tray is not None:
            self._tray.stop()
        self.stop()
        for j in self._jobs.values():
            j.stop()


ENGINE_EXTRA = []  # (the checks add e.g. --input take.mp4 here)
TOOL_EXTRA = []  # (and to face training / gesture recording)


def main():
    from avatarstudio.ui import webwindow
    api = AppApi()
    try:
        webwindow.main("app", api=api, fallback=launcher.main)
    finally:
        api._shutdown()
        # leave right away: a request from the page still on its way when the window closed would otherwise reach
        # .NET while Python shuts down, which crashes the process (exit code 82 instead of 0)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
