"""The commands of AvatarStudio.exe (and of `python app.py` in development).

  AvatarStudio.exe                           the app: camera, avatar, settings and tools in one window (ui/mainwindow.py)
  AvatarStudio.exe launcher                  the classic launcher window (also where no web view is available)
  AvatarStudio.exe studio [studio options]   the tracking engine (engine/studio.py; --help lists its options)
  AvatarStudio.exe viewer / settings         the 3D viewer / settings page in their own window (ui/webwindow.py)
  AvatarStudio.exe train-face                personal face refiner + tongue / cheek model recording
  AvatarStudio.exe tune-smoothing take.mp4   tune motion smoothing from a recording
  AvatarStudio.exe gestures record NAME      custom gestures (record / train / list / delete)
  AvatarStudio.exe mocap take.mp4            offline motion capture from a video (BVH, CSV, joint angles)
  AvatarStudio.exe --test-pattern ...        any other options go to the studio (older launchers)
"""
import importlib
import os
import sys

TOOLS = {"train-face": "avatarstudio.personal.face_refiner", "tune-smoothing": "avatarstudio.personal.tune_smoothing",
         "gestures": "avatarstudio.personal.gestures", "studio": "avatarstudio.engine.studio",
         "mocap": "avatarstudio.mocap"}


def main():
    # run from the app's own folder, where models/ is
    from avatarstudio.core import paths
    here = paths.app_dir()
    os.chdir(here)
    if getattr(sys, "frozen", False):  # your data lives in %APPDATA%\AvatarStudio, shared by every build
        copied = paths.migrate_from_builds(here)
        if copied:
            print(f"copied your data from older builds into {paths.data_dir()}: {', '.join(copied)}")
        from avatarstudio.core import updater
        updater.cleanup(os.environ.get("AVATARSTUDIO_EXE"))  # (the previous version's file, after an update)
    args = sys.argv[1:]
    if not args or args[0] == "launcher":
        from avatarstudio.ui import webwindow
        if args or not webwindow.available():
            from avatarstudio.ui import launcher
            launcher.main()
        else:
            from avatarstudio.ui import mainwindow
            mainwindow.main()
        return
    if args[0] == "cameras":  # (the app window's camera search, in a process of its own: mainwindow.py)
        import json
        from avatarstudio.ui import launcher
        print(json.dumps(launcher.find_cameras()), flush=True)
        return
    if args[0] in ("viewer", "settings"):
        from avatarstudio.ui import webwindow
        webwindow.main(args[0])
        return
    name, rest = (TOOLS[args[0]], args[1:]) if args[0] in TOOLS else (TOOLS["studio"], args)
    sys.argv = [sys.argv[0]] + rest
    try:
        importlib.import_module(name).main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if os.environ.get("AVATAR_PAUSE"):  # started from the launcher: keep the window open to read the results
            try:
                input("\nPress Enter to close this window")
            except EOFError:
                pass


if __name__ == "__main__":
    main()
