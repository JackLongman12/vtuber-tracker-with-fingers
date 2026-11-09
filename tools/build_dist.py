"""Build Avatar Studio as ONE file: Distributions/AvatarStudio.exe (it replaces the previous build, so shortcuts to
it keep working; the version is shown in the launcher's title).

  .venv\\Scripts\\python.exe tools/build_dist.py

The app (PyInstaller folder build: AvatarStudio.exe + its runtime, the models, the 3D viewer with its three.js /
three-vrm, the WiLoR helper script, the licences of all of it) is zipped onto the end of a small launcher (packaging/single_exe.cs, compiled with the C# compiler that
comes with Windows). On the first start of a version it unpacks itself once into
%LOCALAPPDATA%\\AvatarStudio\\app\\<version> and runs from there; your data stays in %APPDATA%\\AvatarStudio.
Double-click it for the launcher window (camera, microphone, every tool as a button); arguments work as before,
e.g. AvatarStudio.exe studio --test-pattern. PyTorch is left out (the live app uses ONNX Runtime).
"""
import glob
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # (the paths below are from the project folder)
DIST_DIR = "Distributions"
KEEP = 1  # one app in Distributions: a new build replaces the previous one
WEB = os.path.join("avatarstudio", "ui", "web")  # -> web/ in the app (avatarstudio.core.paths.web_dir)
MODELS = ["face_landmarker.task", "hand_landmarker.task", "pose_landmarker_full.task", "pose_landmarker_lite.task",
          "rtmw-x-wholebody-384x288.onnx", "silero_vad.onnx", "rnnoise.dll", "backbone.onnx",
          "mp_pose_face_palm.onnx", "mp_hand_landmarks.onnx", "mp_face_blendshapes.onnx"]  # (last 3: gpu_tracking.py)
# what the 3D viewer's import map loads (bundled: the viewer works without internet)
VIEWER_LIBS = ["three-0.160.0/build/three.module.js", "three-0.160.0/examples/jsm/controls/OrbitControls.js",
               "three-0.160.0/examples/jsm/loaders/GLTFLoader.js",
               "three-0.160.0/examples/jsm/utils/BufferGeometryUtils.js", "three-vrm-3.4.0/three-vrm.module.js"]
# the viewer / settings windows (webwindow.py): pywebview's WebView2 assemblies and pythonnet's runtime
WINDOW_DLLS = ["/webview/lib/Microsoft.Web.WebView2.Core.dll", "/webview/lib/Microsoft.Web.WebView2.WinForms.dll",
               "/pythonnet/runtime/Python.Runtime.dll"]
ICON = os.path.join("packaging", "icon.ico")
STUB_SRC = os.path.join("packaging", "single_exe.cs")
CSC = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Microsoft.NET", "Framework64", "v4.0.30319", "csc.exe")
STORED = (".onnx", ".task", ".tflite", ".zip", ".png", ".jpg", ".glb", ".vrm")  # already compressed / dense
MODEL_LICENSES = os.path.join("packaging", "licenses")  # the models' licence texts (the packages' come with them)
LICENSE_FILES = ("LICEN", "COPYING", "NOTICE", "AUTHORS", "COPYRIGHT")  # what a package's licence files start with


def build_app(work):
    """PyInstaller folder build plus models, viewer and the WiLoR helper script. Returns its folder."""
    cmd = [sys.executable, "-m", "PyInstaller", "app.py", "--name", "AvatarStudio", "--noconfirm", "--onedir",
           "--icon", os.path.abspath(ICON),
           "--distpath", os.path.join(work, "dist"), "--workpath", os.path.join(work, "work"),
           "--specpath", work,
           "--collect-all", "mediapipe", "--collect-all", "onnxruntime", "--collect-all", "_sounddevice_data",
           "--collect-submodules", "sklearn", "--hidden-import", "sklearn.neural_network",
           "--hidden-import", "websockets.asyncio.server", "--hidden-import", "websockets.sync.client",
           "--hidden-import", "pynput.keyboard._win32", "--hidden-import", "pynput.mouse._win32",
           "--hidden-import", "sklearn.linear_model",
           "--hidden-import", "pystray._win32",  # (pystray picks its backend by name at run time)
           # every module of the app: main.py loads the tools by name, which PyInstaller can't follow
           "--collect-submodules", "avatarstudio"]
    # matplotlib: only MediaPipe's drawing helpers import it, and all_in_one skips those (tkinter stays: launcher)
    for mod in ("torch", "torchvision", "pandas", "IPython", "pythonosc", "pyarrow", "huggingface_hub",
                "matplotlib"):
        cmd += ["--exclude-module", mod]
    # pywebview: only its WinForms + WebView2 backend. Without Internet Explorer's (mshtml) a PC lacking WebView2
    # gets the browser fallback instead of a window that can't show the pages
    for backend in ("qt", "gtk", "cocoa", "android", "cef", "mshtml"):
        cmd += ["--exclude-module", f"webview.platforms.{backend}"]
    subprocess.run(cmd, check=True)
    app = os.path.join(work, "dist", "AvatarStudio")
    os.makedirs(os.path.join(app, "models"), exist_ok=True)
    for m in MODELS:
        shutil.copy2(os.path.join("models", m), os.path.join(app, "models", m))
    shutil.copytree(WEB, os.path.join(app, "web"), ignore=shutil.ignore_patterns("_test_models", "*.vrm"))
    # the WiLoR finger helper runs in the project's .venv_wilor (PyTorch), not inside the exe: ship its script
    for script in (os.path.join("avatarstudio", "tracking", "wilor_server.py"),
                   os.path.join("avatarstudio", "core", "perf.py")):
        shutil.copy2(script, os.path.join(app, os.path.basename(script)))
    shutil.copy2(ICON, os.path.join(app, "icon.ico"))  # the tray icon
    collect_licenses(work, app)
    return app


def collect_licenses(work, app):
    """LICENSE.txt and THIRD_PARTY_NOTICES.md next to the exe, and licenses/: the licence files of every installed
    package PyInstaller put into the app (found by the site-packages paths in its build lists), Python's own, and
    packaging/licenses (the models'). licenses/INDEX.txt lists the packages with their versions and licences."""
    shutil.copy2("LICENSE", os.path.join(app, "LICENSE.txt"))
    shutil.copy2("THIRD_PARTY_NOTICES.md", os.path.join(app, "THIRD_PARTY_NOTICES.md"))
    out = os.path.join(app, "licenses")
    shutil.copytree(MODEL_LICENSES, os.path.join(out, "models"))
    os.makedirs(os.path.join(out, "Python"))
    shutil.copy2(os.path.join(sys.base_prefix, "LICENSE.txt"), os.path.join(out, "Python", "LICENSE.txt"))
    tops = set()
    for toc in glob.glob(os.path.join(work, "work", "AvatarStudio", "*.toc")):
        text = open(toc, encoding="utf-8", errors="replace").read()
        tops.update(re.findall(r"site-packages[\\/]+([\w.\-]+)", text))
    owners = importlib.metadata.packages_distributions()
    names, unknown = set(), []
    for top in tops:
        if top.endswith(".dist-info"):  # (--collect-all copies some packages' metadata too)
            found = [top[:-len(".dist-info")].rsplit("-", 1)[0]]
        else:
            found = owners.get(top.split(".")[0]) or unknown.append(top) or []
        names.update(re.sub(r"[-_.]+", "-", n).lower() for n in found)  # (one spelling per package)
    index = [f"Python {sys.version.split()[0]:24} PSF-2.0  (Python/LICENSE.txt)"]
    for name in sorted(names, key=str.lower):
        dist = importlib.metadata.distribution(name)
        folder = f"{dist.metadata['Name']}-{dist.version}"
        copied = []
        for f in dist.files or []:
            if not f.name.upper().startswith(LICENSE_FILES) or f.suffix in (".py", ".pyc", ".pyi"):
                continue
            parts = f.parts[1:] if f.parts[0].endswith(".dist-info") else f.parts
            if parts and parts[0] == "licenses":
                parts = parts[1:]
            target = os.path.join(out, folder, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(dist.locate_file(f), target)
            copied.append("/".join(parts))
        copied = list(dict.fromkeys(copied))
        meta = dist.metadata
        lic = meta.get("License-Expression") or (meta.get("License") or "").strip().split("\n")[0][:120] or \
            ", ".join(c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License"))
        files = ", ".join(copied[:3]) + (f" and {len(copied) - 3} more" if len(copied) > 3 else "")
        index.append(f"{folder:31} {lic or '-'}  ({files or 'no licence file in the package'})")
    index.append("Models: models/ (see THIRD_PARTY_NOTICES.md). 3D viewer: web/lib/*/LICENSE.")
    with open(os.path.join(out, "INDEX.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(index) + "\n")
    print(f"licences: {len(names)} packages" + (f" (no package found for {sorted(unknown)})" if unknown else ""))


def build_stub(work):
    if not os.path.exists(CSC):
        raise SystemExit(f"C# compiler not found at {CSC} (it comes with Windows' .NET Framework 4)")
    out = os.path.join(work, "stub.exe")
    subprocess.run([CSC, "/nologo", "/optimize", "/target:winexe", f"/out:{out}", f"/win32icon:{ICON}",
                    "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll",
                    "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", STUB_SRC], check=True)
    return out


def pack(stub, app, target, stamp):
    """stub + the app zipped onto its end (zip offsets count from the start of the file, like a self-extracting
    archive, so the launcher reads its own exe as the zip)."""
    tmp = target + ".partial"
    shutil.copy2(stub, tmp)
    with zipfile.ZipFile(tmp, "a", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr("BUILD", stamp)
        z.comment = f"AVSBUILD={stamp}".encode()  # the very end of the file: the launcher reads only that
        for root, _, files in os.walk(app):
            for name in files:
                path = os.path.join(root, name)
                arc = "app/" + os.path.relpath(path, app).replace(os.sep, "/")
                z.write(path, arc, compress_type=zipfile.ZIP_STORED if name.lower().endswith(STORED) else None)
    with zipfile.ZipFile(tmp) as z:  # quick self-check before it replaces anything
        names = set(z.namelist())
        assert "BUILD" in names and "app/AvatarStudio.exe" in names and "app/models/face_landmarker.task" in names
        assert "app/models/mp_pose_face_palm.onnx" in names
        missing = [f for f in VIEWER_LIBS if f"app/web/lib/{f}" not in names] + \
                  [f for f in WINDOW_DLLS if not any(n.endswith(f) for n in names)] + \
                  [f for f in ("app.html", "app.js", "app.css", "index.html") if f"app/web/{f}" not in names] + \
                  [f for f in ("wilor_server.py", "perf.py") if f"app/{f}" not in names] + \
                  [f for f in ("LICENSE.txt", "THIRD_PARTY_NOTICES.md", "licenses/INDEX.txt",
                               "licenses/Python/LICENSE.txt", "licenses/models/Apache-2.0.txt")
                   if f"app/{f}" not in names] + \
                  [f"licenses of {p}" for p in ("mediapipe-", "pynput-", "pystray-", "pywebview-", "pyinstaller-")
                   if not any(n.startswith(f"app/licenses/{p}") for n in names)]
        assert not missing, f"not in the build: {missing}"
    try:
        os.replace(tmp, target)
    except OSError:
        os.remove(tmp)  # (no half-finished 500 MB file left next to the app)
        raise


def main():
    if sys.platform != "win32":  # (the one-file launcher is a Windows program; a macOS .app / Linux build: not yet)
        raise SystemExit("build_dist.py makes the Windows app. On macOS / Linux run from source: docs/TECHNICAL.md "
                         "\"macOS and Linux\".")
    os.chdir(ROOT)
    missing = [f for f in VIEWER_LIBS if not os.path.exists(os.path.join(WEB, "lib", f))]
    if missing:
        raise SystemExit(f"{WEB}/lib is missing {missing} (three.js 0.160.0 / three-vrm 3.4.0, see docs/TECHNICAL.md)")
    sys.path.insert(0, ROOT)
    from avatarstudio.version import VERSION
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"AvatarStudio_{VERSION}_{stamp}"  # (the build's name: its unpacked folder, the launcher's title)
    os.makedirs(DIST_DIR, exist_ok=True)
    work = "build_tmp"
    shutil.rmtree(work, ignore_errors=True)
    print("building (a few minutes)...", flush=True)
    t0 = time.time()
    app = build_app(work)
    stub = build_stub(work)
    target = os.path.join(DIST_DIR, "AvatarStudio.exe")
    try:
        pack(stub, app, target, name)
    except PermissionError:  # the previous build is running right now: keep it, put this one next to it
        target = os.path.join(DIST_DIR, f"{name}.exe")
        pack(stub, app, target, name)
        print("AvatarStudio.exe is open, so this build was saved under its version name; close the app and "
              "build again to replace it")
    shutil.rmtree(work, ignore_errors=True)
    print(f"built {target} ({os.path.getsize(target) / 2 ** 20:.0f} MB, one file) in {time.time() - t0:.0f} s")
    for old in prune(DIST_DIR, KEEP):
        print(f"removed {old}")


USER_FILES = (".json", ".npz", ".mp4", ".jsonl", ".bvh")
USER_DIRS = ("profiles", "gestures")


def _backup_dir():
    return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "AvatarStudio", "old_builds_backup")


def prune(folder, keep):
    """Keep the newest `keep` one-file builds (names sort by their date-time stamp). Folder builds from before
    the one-file format are removed; anything the user made in one (calibration, training, recordings) and the
    old _saved_user_data folder are first moved to %APPDATA%\\AvatarStudio\\old_builds_backup. Returns what was
    removed."""
    removed = []
    # dated builds: older naming, or saved next to an AvatarStudio.exe that was open during a build
    exes = sorted(f for f in os.listdir(folder) if f.startswith("AvatarStudio_") and f.endswith(".exe"))
    keep_dated = 0 if os.path.exists(os.path.join(folder, "AvatarStudio.exe")) and \
        not (exes and os.path.getmtime(os.path.join(folder, exes[-1])) >
             os.path.getmtime(os.path.join(folder, "AvatarStudio.exe"))) else keep
    for old in exes[:-keep_dated] if keep_dated else exes:
        try:
            os.remove(os.path.join(folder, old))
            removed.append(old)
        except OSError:  # running right now: next time
            pass
    backup = _backup_dir()
    for d in sorted(os.listdir(folder)):
        src = os.path.join(folder, d)
        if not os.path.isdir(src):
            continue
        if d == "_saved_user_data":
            os.makedirs(backup, exist_ok=True)
            for sub in os.listdir(src):
                shutil.copytree(os.path.join(src, sub), os.path.join(backup, sub), dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
            removed.append(f"{d} (moved to {backup})")
        elif d.startswith("AvatarStudio_"):  # an old folder build
            saved = os.path.join(backup, d)
            for item in os.listdir(src):
                path = os.path.join(src, item)
                if os.path.isfile(path) and item.endswith(USER_FILES) and item != "hotkeys.json":
                    os.makedirs(saved, exist_ok=True)
                    shutil.copy2(path, saved)
                elif os.path.isdir(path) and item in USER_DIRS:
                    shutil.copytree(path, os.path.join(saved, item), dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
            removed.append(f"{d} (old folder build{'; your files from it: ' + saved if os.path.isdir(saved) else ''})")
    return removed


if __name__ == "__main__":
    main()
