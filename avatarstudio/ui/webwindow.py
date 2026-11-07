"""The 3D viewer and the settings page as Avatar Studio's own windows (Edge WebView2 through pywebview) instead of
browser tabs.

  AvatarStudio.exe viewer          (development: .venv\\Scripts\\python.exe app.py viewer)
  AvatarStudio.exe settings

The pages are the same files a browser gets (viewer/). Each window serves them itself on a free local port, so it
opens even before tracking starts (the page shows "reconnecting..." until the studio runs) and talks to the studio
over its WebSocket (ws://127.0.0.1:8765), exactly like a browser tab. http://127.0.0.1:8000 (served by the studio)
keeps working for browsers and OBS browser sources. One window of each kind: opening it again brings the open one
to the front. Size, position and maximized state are remembered (windows.json in your data folder). Without
pywebview or the WebView2 runtime, the page opens in the browser as before.

macOS / Linux: the OS's own web view (WKWebView through pyobjc; WebKitGTK or Qt WebEngine), which pywebview picks.
There the open window is found through a lock file in the cache folder and asked over a local socket to come to
the front, and its size and position are followed through pywebview's events. That way was tried on Windows only
(AVATARSTUDIO_PORTABLE_WINDOWS=1 uses it there, for the checks).
"""
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from avatarstudio.core.paths import app_dir, cache_dir, load_json, save_json, user_file, web_dir

PAGES = {"viewer": ("Avatar Studio - 3D viewer", "index.html", (1000, 700)),
         "settings": ("Avatar Studio - Settings", "settings.html", (1040, 800)),
         "app": ("Avatar Studio", "app.html", (1320, 840))}  # (the whole app in one window: mainwindow.py)
MIN_SIZE = {"app": (960, 620)}
STUDIO_URL = "http://127.0.0.1:8000"
PREFS_PATH = user_file("windows.json")
WEBVIEW2_KEY = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WINDOWS = sys.platform == "win32" and not os.environ.get("AVATARSTUDIO_PORTABLE_WINDOWS")
if WINDOWS:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def webview2_version():
    """The installed WebView2 runtime's version (per machine or per user), else None."""
    import winreg
    for root, path in ((winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\{WEBVIEW2_KEY}"),
                       (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\{WEBVIEW2_KEY}"),
                       (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\{WEBVIEW2_KEY}")):
        try:
            with winreg.OpenKey(root, path) as key:
                version = winreg.QueryValueEx(key, "pv")[0]
        except OSError:
            continue
        if version and version != "0.0.0.0":  # (an uninstalled runtime leaves 0.0.0.0 behind)
            return version
    return None


def available():
    """pywebview and what it needs here are there (checked without importing them): pythonnet and the WebView2
    runtime on Windows, pyobjc on macOS, GTK (gi) or Qt (qtpy) on Linux."""
    import importlib.util

    def has(m):  # (an imported clr has no __spec__, which find_spec trips over: the windows' own links)
        return m in sys.modules or importlib.util.find_spec(m) is not None
    if sys.platform == "win32":
        return has("webview") and has("clr") and webview2_version() is not None
    if sys.platform == "darwin":
        return has("webview") and has("objc")
    return has("webview") and (has("gi") or has("qtpy"))


def find(name):
    """The open window of this kind (its handle), else 0."""
    return user32.FindWindowW(None, PAGES[name][0])


def bring_to_front(hwnd):
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def open_in_browser(name):
    webbrowser.open(STUDIO_URL + ("/settings.html" if name == "settings" else ""))


def open_window(name):
    """Show the viewer / settings window: the open one to the front, else a new one (its own process, no
    console), or the browser where app windows can't work."""
    if WINDOWS:
        hwnd = find(name)
        if hwnd:
            bring_to_front(hwnd)
            return
    elif ask_to_show(name):
        return
    if not available():
        open_in_browser(name)
        return
    args = [name] if getattr(sys, "frozen", False) else [os.path.join(app_dir(), "app.py"), name]
    how = dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else dict(start_new_session=True)
    try:
        subprocess.Popen([sys.executable, *args], **how)
    except OSError:
        open_in_browser(name)


# ---------------------------------------------------------------- macOS / Linux: one window of each kind
def _lock_file(name, ext):
    return os.path.join(cache_dir(), f"window-{name}.{ext}")


def hold_lock(name):
    """This window kind's lock, held until the process ends (the OS frees it however it ends); None if another
    process holds it."""
    os.makedirs(cache_dir(), exist_ok=True)
    f = open(_lock_file(name, "lock"), "a")
    try:
        if os.name == "nt":  # (the checks on Windows)
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def ask_to_show(name):
    """Ask the open window of this kind to come to the front, over its local socket. -> whether it answered."""
    try:
        with open(_lock_file(name, "port")) as f:
            port = int(f.read())
        with socket.create_connection(("127.0.0.1", port), timeout=1) as s:
            s.sendall(b"show")
            return s.recv(2) == b"ok"
    except (OSError, ValueError):
        return False


def answer_show_requests(name, window):
    """Listen for ask_to_show() on a free local port (written next to the lock)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    save_port = _lock_file(name, "port")
    with open(save_port + ".tmp", "w") as f:
        f.write(str(srv.getsockname()[1]))
    os.replace(save_port + ".tmp", save_port)

    def come_forward():
        try:
            window.restore()
            window.show()
        except Exception:
            pass

    def run():
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    if conn.recv(8) == b"show":
                        # answer first: restore() waits until the window is on screen (up to 20 s while it's
                        # still starting), longer than the asking side waits for the answer
                        conn.sendall(b"ok")
                        threading.Thread(target=come_forward, daemon=True).start()
                except OSError:
                    pass
    threading.Thread(target=run, daemon=True).start()


class Bounds:
    """The window's normal size and position (and whether it's maximized), followed through pywebview's events:
    asking the window itself while it closes isn't safe on every backend. Moves and resizes that came with
    maximizing or minimizing (they can arrive just before that event) are undone."""

    def __init__(self, bounds, maximized):
        self.bounds, self.maximized, self.minimized = dict(bounds), maximized, False
        self.recent, self.lock = [], threading.Lock()

    def moved(self, x, y):
        if x > -10000:  # (Windows parks minimized windows at -32000)
            self._update(x=x, y=y)

    def resized(self, width, height):
        self._update(width=width, height=height)

    def _update(self, **change):
        with self.lock:
            if not (self.maximized or self.minimized):
                self.recent.append((time.monotonic(), dict(self.bounds)))
                self.recent = self.recent[-20:]
                self.bounds.update(change)

    def state(self, kind):  # "maximized", "minimized" or "restored"
        with self.lock:
            if kind == "restored":
                self.maximized = self.minimized = False
                return
            while self.recent and self.recent[-1][0] > time.monotonic() - 0.3:
                self.bounds = self.recent.pop()[1]
            setattr(self, kind, True)


def on_a_screen(b):
    """Whether a remembered spot is still on one of the screens (pywebview's own list)."""
    try:
        import webview
        return any(s.x <= b["x"] + b["width"] // 2 < s.x + s.width and s.y <= b["y"] + 10 < s.y + s.height
                   for s in webview.screens)
    except Exception:
        return True


# ---------------------------------------------------------------- Windows: placing and remembering through WinForms
def place(window, first):
    """Before the window shows: the first time, centred on the screen with the mouse (pywebview's own centring
    comes too late, leaving Windows' cascade spot); a remembered spot on a monitor that's gone now: centred on the
    main screen. Never bigger than its screen."""
    from System.Drawing import Point, Rectangle
    from System.Windows.Forms import Cursor, FormStartPosition, Screen
    form = window.native
    b = form.Bounds
    kept = not first and any(s.WorkingArea.Contains(Point(b.X + b.Width // 2, b.Y + 10)) for s in Screen.AllScreens)
    screen = Screen.FromRectangle(b) if kept else Screen.FromPoint(Cursor.Position) if first else Screen.PrimaryScreen
    a = screen.WorkingArea
    w, h = min(b.Width, a.Width), min(b.Height, a.Height)
    form.StartPosition = FormStartPosition.Manual
    form.Bounds = Rectangle(b.X, b.Y, w, h) if kept else Rectangle(a.X + (a.Width - w) // 2, a.Y + (a.Height - h) // 2,
                                                                   w, h)


def window_bounds(window):
    """On closing: the window's normal size and position (also while maximized or minimized), in logical pixels
    like pywebview's own sizes, and whether it's maximized."""
    form = window.native
    state = str(form.WindowState)
    b = form.Bounds if state == "Normal" else form.RestoreBounds
    scale = form._scale
    return {"x": round(b.X / scale), "y": round(b.Y / scale), "width": round(b.Width / scale),
            "height": round(b.Height / scale)}, state == "Maximized"


def remember(name, bounds, maximized):
    prefs = load_json(PREFS_PATH, {})
    prefs = prefs if isinstance(prefs, dict) else {}
    prefs[name] = {**bounds, "maximized": maximized}
    try:
        save_json(PREFS_PATH, prefs, indent=1)
    except OSError:
        pass


# ---------------------------------------------------------------- the window
def serve(directory, port=0, routes=None):
    """Serve `directory` at http://127.0.0.1:port (0 = any free port) from a background thread; returns the URL.
    Pages are always revalidated (a new build's files are used at once; unchanged ones come back as 304).
    routes: {path: function(handler)} answered by those functions instead (the studio's camera.mjpg)."""
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from avatarstudio.core.localweb import SECURITY_HEADERS, allowed

    class Handler(SimpleHTTPRequestHandler):
        # Python's guess comes from the Windows registry, where some programs set .js to text/plain; browsers
        # refuse to run modules served like that
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}

        def _refuse(self):
            """Requests from other websites (core/localweb.py) get nothing."""
            if allowed(self.headers):
                return False
            self.send_error(403, "Only Avatar Studio's own pages can use this address")
            return True

        def do_GET(self):
            if self._refuse():
                return None
            route = (routes or {}).get(self.path.split("?")[0])
            if route:
                return route(self)
            return super().do_GET()

        def do_HEAD(self):
            if self._refuse():
                return None
            return super().do_HEAD()

        def list_directory(self, path):  # (no folder listings)
            self.send_error(404)
            return None

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            super().end_headers()

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=directory))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


class Api:
    """What the pages can ask of their window (window.pywebview.api)."""

    def open(self, name):  # the viewer's "Settings" link and the settings page's "open 3D viewer"
        if name in PAGES:
            open_window(name)


_lock = None  # (macOS / Linux: this window's lock, kept while it runs)


def main(name, func=None, api=None, fallback=None):
    """Run the window (`func(window)`, if given, runs in a thread once it's up: used by the checks).
    api: what its page can call (default Api; api.attach(window), if it has one, gets the window before it shows).
    fallback: what to do where no window can be made (default: the page in the browser)."""
    global _lock
    fallback = fallback or (lambda: open_in_browser(name))
    title, page, (width, height) = PAGES[name]
    if WINDOWS:
        kernel32.CreateMutexW(None, False, f"AvatarStudio-window-{name}")  # (held until this process ends)
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS: one is open or opening: show that one instead
            for _ in range(50):
                hwnd = find(name)
                if hwnd:
                    bring_to_front(hwnd)
                    return
                time.sleep(0.1)
            # no window in 5 s: a copy stuck while closing still holds the name (a camera driver kept one from
            # ending): open anyway rather than nothing at all
    else:
        _lock = hold_lock(name)
        if _lock is None:  # one is open or opening: show that one instead
            for _ in range(50):
                if ask_to_show(name):
                    return
                time.sleep(0.1)
            # (no answer in 5 s: stuck while closing, as above - open anyway)
    try:
        import webview
        if WINDOWS:
            from webview.platforms import winforms  # loads .NET and WebView2's assemblies (fails without them)
            if not winforms.is_chromium:  # pywebview would fall back to Internet Explorer: can't show the pages
                raise RuntimeError("the WebView2 runtime isn't installed")
    except Exception as e:
        print(f"app window unavailable ({e}); falling back")
        fallback()
        return
    import logging

    class InitFailed(logging.Handler):  # WebView2 there but broken: pywebview only logs it, leaving a blank window
        def emit(self, record):
            if "WebView2 initialization failed" in record.getMessage():
                fallback()
                # (this runs on the window's own thread, where destroy() would wait for itself)
                threading.Thread(target=window.destroy, daemon=True).start()

    logging.getLogger("pywebview").addHandler(InitFailed(logging.ERROR))
    saved = (load_json(PREFS_PATH, {}) or {}).get(name)
    saved = saved if isinstance(saved, dict) else {}
    bounds = {"width": width, "height": height}
    for keys in (("width", "height"), ("x", "y")):  # (no position saved yet: centred)
        try:
            bounds.update({k: int(saved[k]) for k in keys})
        except (KeyError, TypeError, ValueError):
            pass
    if not WINDOWS and "x" in bounds and not on_a_screen(bounds):  # (a monitor that's gone: pywebview centres it)
        del bounds["x"], bounds["y"]
    origin = serve(web_dir())
    url = origin + "/" + page
    webview.settings["ALLOW_FILE_URLS"] = False  # (a dropped file must not open in the window)
    window = webview.create_window(title, url, js_api=api or Api(), min_size=MIN_SIZE.get(name, (480, 360)),
                                   background_color="#111418", maximized=bool(saved.get("maximized")), **bounds)

    def stay_home(*_):
        """The window shows only its own pages: what its page can call (window.pywebview.api: uninstall, tools,
        updates) would be there for any page it showed, e.g. a link dropped onto it."""
        try:
            now = window.get_current_url() or ""
        except Exception:
            return
        if now != origin and not now.startswith(origin + "/"):
            print(f"blocked a page that isn't Avatar Studio's own in the {name} window", flush=True)
            window.load_url(url)
    window.events.loaded += stay_home
    if hasattr(api, "attach"):
        api.attach(window)
    if WINDOWS:
        window.events.before_show += lambda window: place(window, first="x" not in bounds)
        window.events.closing += lambda window: remember(name, *window_bounds(window))
    else:
        follow = Bounds(bounds, bool(saved.get("maximized")))
        window.events.moved += follow.moved
        window.events.resized += follow.resized
        window.events.maximized += lambda: follow.state("maximized")
        window.events.minimized += lambda: follow.state("minimized")
        window.events.restored += lambda: follow.state("restored")
        window.events.closing += lambda: remember(name, follow.bounds, follow.maximized)
        answer_show_requests(name, window)
    icon = next((p for p in (os.path.join(app_dir(), "icon.ico"), os.path.join(app_dir(), "packaging", "icon.ico"))
                 if os.path.exists(p)), None)
    # private_mode off + a fixed folder: WebView2 keeps its profile, so later windows start quicker
    try:
        webview.start(func, (window,) if func else None, private_mode=False, icon=icon,
                      storage_path=os.path.join(cache_dir(), "WebView2" if sys.platform == "win32" else "webview"),
                      debug=bool(os.environ.get("AVATARSTUDIO_WEBVIEW_DEBUG")))
    except Exception as e:  # (Linux without GTK's or Qt's web view, ...)
        print(f"app window unavailable ({e}); falling back")
        fallback()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PAGES else "viewer")
