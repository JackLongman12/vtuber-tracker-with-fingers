"""Where your own data lives: calibration, settings, face / mouth training, smoothing, profiles.

Packaged builds keep it in %APPDATA%\\AvatarStudio, so every build (and every new build) sees the same data;
nothing has to be copied between builds (macOS: ~/Library/Application Support/AvatarStudio, Linux:
~/.local/share/AvatarStudio). Development runs keep it in the project's data/ folder (not in git).
AVATARSTUDIO_DATA overrides both (tests use it so they never touch your real data).

Also where the app's own files are: app_dir() (models/; the project folder in development, next to the program when
packaged) and web_dir() (the app window's pages, avatarstudio/ui/web).

The first time a build starts with an empty data folder it copies the newest copy of each item it can find in
older builds next to it (and Distributions/_saved_user_data) - copies, it never deletes anything.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # (the project folder)
ITEMS =["calibration.json", "settings.json", "hotkeys.json", "face_refiner.json", "mouth_model.npz",
         "mouth_data.npz", "profiles", "gestures", "smoothing.json",
         "launcher.json", "vts_token.json", "take.mp4"]


def _os_folder(kind):
    """This OS's folder for the user's own data ("data") or for machine-local caches ("cache"), + AvatarStudio."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("APPDATA" if kind == "data" else "LOCALAPPDATA") or home
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support" if kind == "data" else "Caches")
    else:  # (Linux and the like: the XDG folders)
        default = os.path.join(home, ".local", "share") if kind == "data" else os.path.join(home, ".cache")
        base = os.environ.get("XDG_DATA_HOME" if kind == "data" else "XDG_CACHE_HOME") or default
    return os.path.join(base, "AvatarStudio")


def data_dir():
    d = os.environ.get("AVATARSTUDIO_DATA")
    if not d:
        if getattr(sys, "frozen", False):
            d = _os_folder("data")
        else:
            d = os.path.join(ROOT, "data")
    os.makedirs(d, exist_ok=True)
    return d


def cache_dir():
    """Machine-local things that aren't your data (the viewer windows' browser cache): %LOCALAPPDATA%\\AvatarStudio,
    ~/Library/Caches/AvatarStudio, ~/.cache/AvatarStudio."""
    return _os_folder("cache")


def app_dir():
    """Where the app's own files are (models/, app.py): next to the packaged app, or the project folder. In a
    macOS app bundle they may sit in Contents/Resources instead of next to the program in Contents/MacOS."""
    if not getattr(sys, "frozen", False):
        return ROOT
    here = os.path.dirname(sys.executable)
    resources = os.path.join(here, "..", "Resources")
    if sys.platform == "darwin" and not os.path.isdir(os.path.join(here, "models")) and os.path.isdir(resources):
        return os.path.abspath(resources)
    return here


def web_dir():
    """The pages of the app window, the 3D viewer and the settings page (avatarstudio/ui/web; web/ when packaged)."""
    if getattr(sys, "frozen", False):
        return os.path.join(app_dir(), "web")
    return os.path.join(ROOT, "avatarstudio", "ui", "web")


def user_file(name):
    """Full path of one of your data files / folders (see ITEMS)."""
    return os.path.join(data_dir(), name)


def save_json(path, data, **kw):
    """Write JSON so a crash or power loss mid-write can't leave a half-written file: write a temporary file
    next to it, then swap it in (atomic on Windows and Linux)."""
    import json
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, **kw)
    os.replace(tmp, path)


def load_json(path, default=None):
    """JSON from path, or `default` if it's missing or unreadable. An unreadable file is kept as <name>.broken
    (and reported) instead of stopping the app."""
    import json
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        try:
            os.replace(path, f"{path}.broken")
        except OSError:
            pass
        print(f"{os.path.basename(path)} was unreadable ({e}); using defaults (old file kept as .broken)")
        return default


def _mtime(path):
    if os.path.isdir(path):
        times = [os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs]
        return max(times) if times else 0.0
    return os.path.getmtime(path)


def migrate_from_builds(app_dir):
    """Packaged build: fill an empty data folder from older builds (newest copy of each item wins)."""
    target = data_dir()
    if os.path.abspath(target) == os.path.abspath(app_dir) or any(os.path.exists(os.path.join(target, i))
                                                                   for i in ITEMS):
        return []
    parent = os.path.dirname(os.path.abspath(app_dir))
    sources = [app_dir]
    if os.path.isdir(parent):
        sources += [os.path.join(parent, d) for d in os.listdir(parent) if d.startswith("AvatarStudio_")]
        saved = os.path.join(parent, "_saved_user_data")
        if os.path.isdir(saved):
            sources += [os.path.join(saved, d) for d in os.listdir(saved)]
    copied = []
    for item in ITEMS:
        found = [os.path.join(s, item) for s in sources if os.path.exists(os.path.join(s, item))]
        found = [p for p in found if not (os.path.isdir(p) and not os.listdir(p))]
        if not found:
            continue
        newest = max(found, key=_mtime)
        dst = os.path.join(target, item)
        if os.path.isdir(newest):
            shutil.copytree(newest, dst)
        else:
            shutil.copy2(newest, dst)
        copied.append(f"{item} (from {os.path.basename(os.path.dirname(newest))})")
    return copied
