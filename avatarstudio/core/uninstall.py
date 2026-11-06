"""Remove Avatar Studio from this PC: Settings > Uninstall in the app, or `AvatarStudio.exe uninstall`.

The app has no installer. What it leaves on a PC (Windows):
  %LOCALAPPDATA%\\AvatarStudio   the unpacked app (this version and the previous one, ~800 MB each), window caches
  %APPDATA%\\AvatarStudio        your data: calibration, settings, recordings, face / gesture training, profiles
  Start menu and desktop        "Avatar Studio.lnk", if you made the shortcuts
  %TEMP%\\AvatarStudio-wilor     the WiLoR helper's scripts, if WiLoR was used
  AvatarStudio.exe              the downloaded app itself (plus .old.exe / .download left by an update)
Nothing in the registry, no services, no autostart outside the app's own setting.

The running app can't delete its own files, so a small hidden PowerShell script does it after the app has closed:
it retries for up to two minutes (until every Avatar Studio process has ended), then deletes itself. Your data is
kept unless you ask for it to go, and then it goes to the Recycle Bin, so it can still be restored.
"""
import os
import subprocess
import sys
import tempfile

from avatarstudio.core.paths import _os_folder, data_dir

SHORTCUT = "Avatar Studio.lnk"
WAIT_S = 120


def _desktop():
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf)  # CSIDL_DESKTOPDIRECTORY
        return buf.value or None
    except Exception:
        return None


def app_file():
    """The one-file AvatarStudio.exe this app was started from (its launcher says), else None (development)."""
    exe = os.environ.get("AVATARSTUDIO_EXE")
    return exe if getattr(sys, "frozen", False) and exe and os.path.exists(exe) else None


def targets(delete_data):
    """-> (paths to delete, paths to send to the Recycle Bin): only what exists."""
    remove = [_os_folder("cache"), os.path.join(tempfile.gettempdir(), "AvatarStudio-wilor")]
    start_menu = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs")
    remove += [os.path.join(d, SHORTCUT) for d in (start_menu, _desktop()) if d]
    exe = app_file()
    if exe:
        remove += [exe, os.path.splitext(exe)[0] + ".old.exe", exe + ".download"]
    recycle = [data_dir()] if delete_data else []
    exists = lambda ps: [p for p in ps if p and os.path.lexists(p)]
    return exists(remove), exists(recycle)


def _size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 2 ** 20
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
               if os.path.exists(os.path.join(r, f))) / 2 ** 20


def plan():
    """What an uninstall would remove, for the confirmation: {"app": [(path, MB)], "data": (path, MB), "packaged"}."""
    remove, _ = targets(False)
    data = data_dir()
    return {"app": [(p, round(_size_mb(p))) for p in remove],
            "data": (data, round(_size_mb(data))) if os.path.isdir(data) else None,
            "packaged": bool(app_file()), "supported": sys.platform == "win32"}


def _ps_list(paths):
    return "@(" + ",".join("'" + p.replace("'", "''") + "'" for p in paths) + ")"


def launch(remove, recycle, wait_s=WAIT_S):
    """Start the hidden clean-up script (it outlives this process). -> the script's path."""
    script = os.path.join(tempfile.gettempdir(), f"AvatarStudio-uninstall-{os.getpid()}.ps1")
    text = f"""$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName Microsoft.VisualBasic
$remove = {_ps_list(remove)}
$recycle = {_ps_list(recycle)}
$deadline = (Get-Date).AddSeconds({int(wait_s)})
do {{
    foreach ($p in $recycle) {{
        if (Test-Path -LiteralPath $p) {{
            try {{ [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($p, 'OnlyErrorDialogs', 'SendToRecycleBin') }} catch {{}}
        }}
    }}
    foreach ($p in $remove) {{ Remove-Item -LiteralPath $p -Recurse -Force }}
    $left = @($remove + $recycle | Where-Object {{ Test-Path -LiteralPath $_ }})
    if ($left.Count -eq 0) {{ break }}
    Start-Sleep -Milliseconds 700
}} while ((Get-Date) -lt $deadline)
Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force
"""
    with open(script, "w", encoding="utf-8-sig") as f:
        f.write(text)
    # (no console window; its own process group, so it lives on after the app. DETACHED_PROCESS made PowerShell
    # quit at once without doing anything)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script],
                     creationflags=flags, close_fds=True, cwd=tempfile.gettempdir(),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return script


def start(delete_data):
    """Start removing the packaged app (and your data if asked). The caller then quits. -> {"ok", ...}."""
    if sys.platform != "win32":
        return {"ok": False, "error": "Uninstalling from the app is Windows only: delete the app and its folders"}
    if not app_file():
        return {"ok": False, "error": "This is a development run: there's nothing installed (delete the project "
                                      "folder to remove it)"}
    remove, recycle = targets(delete_data)
    launch(remove, recycle)
    return {"ok": True, "removed": remove, "recycled": recycle}


def main():
    """`AvatarStudio.exe uninstall [--delete-data] [--yes]` in a console."""
    import argparse
    ap = argparse.ArgumentParser(description="Remove Avatar Studio from this PC")
    ap.add_argument("--delete-data", action="store_true",
                    help="also your data (calibration, recordings, training, profiles), to the Recycle Bin")
    ap.add_argument("--yes", action="store_true", help="don't ask")
    args = ap.parse_args()
    p = plan()
    if not p["packaged"]:
        raise SystemExit("This is a development run: there's nothing installed (delete the project folder).")
    print("Avatar Studio will be removed:")
    for path, mb in p["app"]:
        print(f"  {path}  ({mb} MB)")
    if p["data"]:
        print(("Your data goes to the Recycle Bin:" if args.delete_data else "Your data is KEPT (--delete-data removes it):")
              + f"\n  {p['data'][0]}  ({p['data'][1]} MB)")
    if not args.yes and input("\nUninstall now? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing was removed.")
        return
    r = start(args.delete_data)
    print("Removing it now; it's gone a few seconds after this window closes." if r["ok"] else r["error"])
